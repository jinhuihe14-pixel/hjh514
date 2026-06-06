import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from data_storage import DataStorage
from models import WeatherType, HolidayType


class SalesPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models: Dict[Tuple[str, str], xgb.XGBRegressor] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_importance: Dict[Tuple[str, str], Dict[str, float]] = {}
        
        self._init_label_encoders()
    
    def _init_label_encoders(self):
        self.label_encoders["weather"] = LabelEncoder()
        self.label_encoders["weather"].fit([w.value for w in WeatherType])
        
        self.label_encoders["holiday_type"] = LabelEncoder()
        self.label_encoders["holiday_type"].fit([h.value for h in HolidayType])
        
        self.label_encoders["business_activity"] = LabelEncoder()
        self.label_encoders["business_activity"].fit(["", "促销", "新品上市", "商圈活动", "节日活动"])
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        features["day_of_week"] = features["sale_date"].dt.dayofweek
        features["day_of_month"] = features["sale_date"].dt.day
        features["month"] = features["sale_date"].dt.month
        features["week_of_year"] = features["sale_date"].dt.isocalendar().week
        
        features["weather_encoded"] = self.label_encoders["weather"].transform(features["weather"])
        features["holiday_encoded"] = self.label_encoders["holiday_type"].transform(features["holiday_type"])
        
        activities = features["business_activity"].fillna("").apply(
            lambda x: x if x in self.label_encoders["business_activity"].classes_ else ""
        )
        features["activity_encoded"] = self.label_encoders["business_activity"].transform(activities)
        
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
        features["is_holiday"] = (features["holiday_type"] == HolidayType.HOLIDAY.value).astype(int)
        
        features["temperature_squared"] = features["temperature"] ** 2
        
        return features
    
    def train(self, data_storage: DataStorage, retrain: bool = False) -> Dict[str, float]:
        sales_df = data_storage.get_sales_dataframe()
        sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
        
        if sales_df.empty:
            return {"status": "error", "message": "无销售数据"}
        
        store_drink_pairs = sales_df[["store_id", "drink_id"]].drop_duplicates().values
        
        metrics = {}
        total_mae = 0
        total_rmse = 0
        count = 0
        
        for store_id, drink_id in store_drink_pairs:
            model_key = (store_id, drink_id)
            
            if not retrain and model_key in self.models:
                continue
            
            pair_data = sales_df[
                (sales_df["store_id"] == store_id) & 
                (sales_df["drink_id"] == drink_id)
            ].sort_values("sale_date").copy()
            
            if len(pair_data) < 30:
                continue
            
            pair_data["lag_7"] = pair_data["quantity"].shift(7)
            pair_data["lag_14"] = pair_data["quantity"].shift(14)
            pair_data["rolling_mean_7"] = pair_data["quantity"].rolling(7).mean()
            pair_data["rolling_mean_14"] = pair_data["quantity"].rolling(14).mean()
            pair_data["rolling_std_7"] = pair_data["quantity"].rolling(7).std()
            
            pair_data = pair_data.dropna()
            
            if len(pair_data) < 15:
                continue
            
            features = self._prepare_features(pair_data)
            
            feature_cols = [
                "temperature", "temperature_squared",
                "weather_encoded", "holiday_encoded", "activity_encoded",
                "day_of_week", "day_of_month", "month", "week_of_year",
                "is_weekend", "is_holiday",
                "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
                "has_promotion", "has_new_product"
            ]
            
            X = features[feature_cols]
            y = features["quantity"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models[model_key] = model
            
            importance = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance[model_key] = dict(sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            ))
            
            metrics[f"{store_id}_{drink_id}"] = {
                "mae": mae,
                "rmse": rmse,
                "samples": len(pair_data)
            }
            
            total_mae += mae
            total_rmse += rmse
            count += 1
        
        if count > 0:
            metrics["overall"] = {
                "avg_mae": total_mae / count,
                "avg_rmse": total_rmse / count,
                "models_trained": count
            }
        
        self._save_models()
        
        return metrics
    
    def predict(self, store_id: str, drink_id: str,
                future_dates: List[date],
                weather_forecast: List[WeatherType],
                temperature_forecast: List[float],
                holiday_types: List[HolidayType],
                has_promotion: Optional[List[bool]] = None,
                has_new_product: Optional[List[bool]] = None,
                business_activities: Optional[List[str]] = None,
                sales_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        
        model_key = (store_id, drink_id)
        
        if model_key not in self.models:
            return []
        
        if has_promotion is None:
            has_promotion = [False] * len(future_dates)
        if has_new_product is None:
            has_new_product = [False] * len(future_dates)
        if business_activities is None:
            business_activities = [""] * len(future_dates)
        
        future_data = pd.DataFrame({
            "sale_date": pd.to_datetime(future_dates),
            "weather": [w.value for w in weather_forecast],
            "temperature": temperature_forecast,
            "holiday_type": [h.value for h in holiday_types],
            "has_promotion": has_promotion,
            "has_new_product": has_new_product,
            "business_activity": business_activities
        })
        
        if sales_df is not None and not sales_df.empty:
            sales_df = sales_df.copy()
            sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
            pair_sales = sales_df[
                (sales_df["store_id"] == store_id) & 
                (sales_df["drink_id"] == drink_id)
            ].sort_values("sale_date")
            
            if not pair_sales.empty:
                last_14_days = pair_sales.tail(14)["quantity"].values
                if len(last_14_days) >= 7:
                    future_data["lag_7"] = last_14_days[-7] if len(last_14_days) >= 7 else np.mean(last_14_days)
                    future_data["lag_14"] = last_14_days[0] if len(last_14_days) >= 14 else np.mean(last_14_days)
                    future_data["rolling_mean_7"] = np.mean(last_14_days[-7:])
                    future_data["rolling_mean_14"] = np.mean(last_14_days)
                    future_data["rolling_std_7"] = np.std(last_14_days[-7:]) if len(last_14_days) >= 7 else 0
                else:
                    for col in ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7"]:
                        future_data[col] = np.mean(last_14_days) if len(last_14_days) > 0 else 0
            else:
                for col in ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7"]:
                    future_data[col] = 0
        else:
            for col in ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7"]:
                future_data[col] = 0
        
        features = self._prepare_features(future_data)
        
        feature_cols = [
            "temperature", "temperature_squared",
            "weather_encoded", "holiday_encoded", "activity_encoded",
            "day_of_week", "day_of_month", "month", "week_of_year",
            "is_weekend", "is_holiday",
            "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
            "has_promotion", "has_new_product"
        ]
        
        X = features[feature_cols]
        predictions = self.models[model_key].predict(X)
        
        results = []
        for i, pred_date in enumerate(future_dates):
            results.append({
                "date": pred_date,
                "predicted_quantity": max(0, round(predictions[i], 2)),
                "weather": weather_forecast[i].value,
                "temperature": temperature_forecast[i],
                "holiday_type": holiday_types[i].value
            })
        
        return results
    
    def _save_models(self):
        model_data = {
            "models": {f"{k[0]}_{k[1]}": v for k, v in self.models.items()},
            "label_encoders": self.label_encoders,
            "feature_importance": {f"{k[0]}_{k[1]}": v for k, v in self.feature_importance.items()}
        }
        
        with open(os.path.join(self.model_dir, "sales_predictor.pkl"), "wb") as f:
            pickle.dump(model_data, f)
    
    def load_models(self):
        model_path = os.path.join(self.model_dir, "sales_predictor.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                
                self.models = {}
                for k, v in model_data["models"].items():
                    store_id, drink_id = k.split("_", 1)
                    self.models[(store_id, drink_id)] = v
                
                self.label_encoders = model_data["label_encoders"]
                
                self.feature_importance = {}
                for k, v in model_data["feature_importance"].items():
                    store_id, drink_id = k.split("_", 1)
                    self.feature_importance[(store_id, drink_id)] = v
    
    def get_feature_importance(self, store_id: str, drink_id: str) -> Dict[str, float]:
        return self.feature_importance.get((store_id, drink_id), {})
