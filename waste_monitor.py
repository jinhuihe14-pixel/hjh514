import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data_storage import DataStorage
from models import WasteAlert, Ingredient, Store


class WasteMonitor:
    def __init__(self, data_storage: DataStorage):
        self.data_storage = data_storage
        
        self.default_threshold = 0.15
        self.short_life_threshold = 0.25
        self.consecutive_days_threshold = 3
        
        self.waste_rate_history: Dict[Tuple[str, str], List[Tuple[date, float]]] = {}
    
    def calculate_waste_rate(self, store_id: str, ingredient_id: str,
                             start_date: date, end_date: date) -> pd.DataFrame:
        inventory_df = self.data_storage.get_inventory_dataframe()
        
        if inventory_df.empty:
            return pd.DataFrame()
        
        inventory_df["record_date"] = pd.to_datetime(inventory_df["record_date"])
        
        mask = (
            (inventory_df["store_id"] == store_id) &
            (inventory_df["ingredient_id"] == ingredient_id) &
            (inventory_df["record_date"].dt.date >= start_date) &
            (inventory_df["record_date"].dt.date <= end_date)
        )
        
        data = inventory_df[mask].copy()
        
        if data.empty:
            return pd.DataFrame()
        
        data = data.sort_values("record_date")
        
        data["total_available"] = data["beginning_inventory"] + data["purchased_quantity"]
        data["waste_rate"] = data.apply(
            lambda row: row["wasted_quantity"] / row["total_available"] 
            if row["total_available"] > 0 else 0,
            axis=1
        )
        
        return data
    
    def get_consecutive_high_waste_days(self, waste_rates: List[float],
                                        threshold: float) -> int:
        max_consecutive = 0
        current_consecutive = 0
        
        for rate in waste_rates:
            if rate > threshold:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def check_waste_alerts(self, check_date: Optional[date] = None,
                           lookback_days: int = 7) -> List[WasteAlert]:
        if check_date is None:
            check_date = date.today()
        
        start_date = check_date - timedelta(days=lookback_days)
        
        alerts = []
        
        for store_id in self.data_storage.stores:
            for ingredient_id, ingredient in self.data_storage.ingredients.items():
                threshold = (self.short_life_threshold 
                            if ingredient.is_short_life 
                            else self.default_threshold)
                
                waste_data = self.calculate_waste_rate(
                    store_id, ingredient_id, start_date, check_date
                )
                
                if waste_data.empty or len(waste_data) < 3:
                    continue
                
                recent_rates = waste_data.tail(lookback_days)["waste_rate"].tolist()
                avg_waste_rate = np.mean(recent_rates)
                
                consecutive_days = self.get_consecutive_high_waste_days(
                    recent_rates, threshold
                )
                
                if (avg_waste_rate > threshold and 
                    consecutive_days >= self.consecutive_days_threshold):
                    
                    suggestion = self.generate_waste_suggestion(
                        store_id, ingredient_id, avg_waste_rate, consecutive_days
                    )
                    
                    alert = WasteAlert(
                        store_id=store_id,
                        ingredient_id=ingredient_id,
                        alert_date=check_date,
                        avg_waste_rate=round(avg_waste_rate, 4),
                        threshold=threshold,
                        consecutive_days=consecutive_days,
                        suggestion=suggestion
                    )
                    
                    alerts.append(alert)
        
        return alerts
    
    def generate_waste_suggestion(self, store_id: str, ingredient_id: str,
                                   avg_waste_rate: float, consecutive_days: int) -> str:
        ingredient = self.data_storage.ingredients.get(ingredient_id)
        store = self.data_storage.stores.get(store_id)
        
        if not ingredient or not store:
            return "建议核查原料库存管理情况"
        
        suggestions = []
        
        if ingredient.is_short_life:
            suggestions.append(f"【{ingredient.name}】为短保原料(保质期{ingredient.shelf_life_days}天)")
        
        suggestions.append(f"连续{consecutive_days}天损耗率偏高，平均{avg_waste_rate:.1%}")
        
        reduction_pct = min(30, int(avg_waste_rate * 100 * 0.5))
        suggestions.append(f"建议将该原料备货量缩减{reduction_pct}%")
        
        if consecutive_days >= 5:
            suggestions.append("建议检查供应链配送频次，考虑增加配送频率减少单次采购量")
        
        if avg_waste_rate > 0.3:
            suggestions.append("建议核查是否存在存储不当或操作浪费问题")
        
        return "；".join(suggestions)
    
    def generate_waste_report(self, start_date: date, end_date: date) -> pd.DataFrame:
        inventory_df = self.data_storage.get_inventory_dataframe()
        
        if inventory_df.empty:
            return pd.DataFrame()
        
        inventory_df["record_date"] = pd.to_datetime(inventory_df["record_date"])
        
        mask = (
            (inventory_df["record_date"].dt.date >= start_date) &
            (inventory_df["record_date"].dt.date <= end_date)
        )
        
        data = inventory_df[mask].copy()
        
        if data.empty:
            return pd.DataFrame()
        
        data["total_available"] = data["beginning_inventory"] + data["purchased_quantity"]
        
        summary = data.groupby(["store_id", "ingredient_id"]).agg({
            "total_available": "sum",
            "used_quantity": "sum",
            "wasted_quantity": "sum"
        }).reset_index()
        
        summary["waste_rate"] = summary.apply(
            lambda row: row["wasted_quantity"] / row["total_available"] 
            if row["total_available"] > 0 else 0,
            axis=1
        )
        
        summary["store_name"] = summary["store_id"].apply(
            lambda x: self.data_storage.stores.get(x, Store("", "", "", "", "")).name
        )
        summary["ingredient_name"] = summary["ingredient_id"].apply(
            lambda x: self.data_storage.ingredients.get(x, Ingredient("", "", "", 0, "")).name
        )
        summary["is_short_life"] = summary["ingredient_id"].apply(
            lambda x: "是" if self.data_storage.ingredients.get(x, Ingredient("", "", "", 0, "")).is_short_life else "否"
        )
        
        result = summary[[
            "store_id", "store_name", "ingredient_id", "ingredient_name",
            "is_short_life", "total_available", "used_quantity", 
            "wasted_quantity", "waste_rate"
        ]].copy()
        
        result = result.sort_values("waste_rate", ascending=False)
        
        return result
    
    def get_waste_trends(self, store_id: str, ingredient_id: str,
                         days: int = 30) -> Dict[str, List]:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        waste_data = self.calculate_waste_rate(store_id, ingredient_id, start_date, end_date)
        
        if waste_data.empty:
            return {"dates": [], "waste_rates": [], "avg_rate": 0}
        
        return {
            "dates": waste_data["record_date"].dt.date.tolist(),
            "waste_rates": waste_data["waste_rate"].tolist(),
            "avg_rate": round(waste_data["waste_rate"].mean(), 4)
        }
    
    def get_store_waste_ranking(self, start_date: date, end_date: date,
                                 top_n: int = 10) -> pd.DataFrame:
        report = self.generate_waste_report(start_date, end_date)
        
        if report.empty:
            return pd.DataFrame()
        
        store_summary = report.groupby(["store_id", "store_name"]).agg({
            "total_available": "sum",
            "wasted_quantity": "sum",
            "waste_rate": "mean"
        }).reset_index()
        
        store_summary = store_summary.sort_values("waste_rate", ascending=False)
        
        return store_summary.head(top_n)
    
    def get_ingredient_waste_ranking(self, start_date: date, end_date: date,
                                      top_n: int = 10) -> pd.DataFrame:
        report = self.generate_waste_report(start_date, end_date)
        
        if report.empty:
            return pd.DataFrame()
        
        ing_summary = report.groupby(["ingredient_id", "ingredient_name", "is_short_life"]).agg({
            "total_available": "sum",
            "wasted_quantity": "sum",
            "waste_rate": "mean"
        }).reset_index()
        
        ing_summary = ing_summary.sort_values("waste_rate", ascending=False)
        
        return ing_summary.head(top_n)
