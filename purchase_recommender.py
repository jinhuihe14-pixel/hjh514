import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data_storage import DataStorage
from models import (
    PurchaseSuggestion, Store, Ingredient, Drink,
    WeatherType, HolidayType
)


class PurchaseRecommender:
    def __init__(self, data_storage: DataStorage, sales_predictor):
        self.data_storage = data_storage
        self.sales_predictor = sales_predictor
        
        self.safety_stock_factor = {
            "short_life": 0.3,
            "normal": 0.5
        }
        
        self.shelf_life_adjustment = {
            1: 0.6,
            2: 0.7,
            3: 0.8,
            7: 0.9
        }
    
    def calculate_ingredient_demand(self, store_id: str,
                                    future_dates: List[date],
                                    weather_forecast: List[WeatherType],
                                    temperature_forecast: List[float],
                                    holiday_types: List[HolidayType]) -> Dict[str, List[float]]:
        ingredient_demand: Dict[str, List[float]] = {}
        
        for ingredient_id in self.data_storage.ingredients:
            ingredient_demand[ingredient_id] = [0.0] * len(future_dates)
        
        sales_df = self.data_storage.get_sales_dataframe()
        
        for drink_id, drink in self.data_storage.drinks.items():
            predictions = self.sales_predictor.predict(
                store_id=store_id,
                drink_id=drink_id,
                future_dates=future_dates,
                weather_forecast=weather_forecast,
                temperature_forecast=temperature_forecast,
                holiday_types=holiday_types,
                sales_df=sales_df
            )
            
            if not predictions:
                continue
            
            for ingredient_id, amount in drink.recipe.items():
                if ingredient_id in ingredient_demand:
                    for i, pred in enumerate(predictions):
                        ingredient_demand[ingredient_id][i] += pred["predicted_quantity"] * amount
        
        return ingredient_demand
    
    def get_current_inventory(self, store_id: str) -> Dict[str, float]:
        inventory_df = self.data_storage.get_inventory_dataframe()
        
        if inventory_df.empty:
            return {}
        
        store_inventory = inventory_df[inventory_df["store_id"] == store_id].copy()
        store_inventory["record_date"] = pd.to_datetime(store_inventory["record_date"])
        
        latest_inventory = {}
        
        for ingredient_id in self.data_storage.ingredients:
            ing_data = store_inventory[store_inventory["ingredient_id"] == ingredient_id]
            if not ing_data.empty:
                latest = ing_data.sort_values("record_date").iloc[-1]
                latest_inventory[ingredient_id] = latest["ending_inventory"]
        
        return latest_inventory
    
    def get_forecast_accuracy(self, store_id: str, drink_id: str) -> float:
        importance = self.sales_predictor.get_feature_importance(store_id, drink_id)
        if importance:
            return 0.85
        return 0.7
    
    def generate_purchase_suggestions(self,
                                      store_id: str,
                                      future_dates: List[date],
                                      weather_forecast: List[WeatherType],
                                      temperature_forecast: List[float],
                                      holiday_types: List[HolidayType],
                                      purchase_day: int = 0) -> List[PurchaseSuggestion]:
        if store_id not in self.data_storage.stores:
            return []
        
        ingredient_demand = self.calculate_ingredient_demand(
            store_id, future_dates, weather_forecast, temperature_forecast, holiday_types
        )
        
        current_inventory = self.get_current_inventory(store_id)
        
        suggestions = []
        
        for ingredient_id, ingredient in self.data_storage.ingredients.items():
            if ingredient_id not in ingredient_demand:
                continue
            
            demand_list = ingredient_demand[ingredient_id]
            shelf_life = ingredient.shelf_life_days
            
            cover_days = min(shelf_life, len(future_dates) - purchase_day)
            cover_days = max(1, cover_days)
            
            total_demand = sum(demand_list[purchase_day:purchase_day + cover_days])
            
            adjustment = 1.0
            for days, adj in sorted(self.shelf_life_adjustment.items()):
                if shelf_life <= days:
                    adjustment = adj
                    break
            
            if ingredient.is_short_life:
                adjustment *= 0.7
            
            avg_demand = total_demand / cover_days if cover_days > 0 else 0
            
            safety_stock_factor = self.safety_stock_factor["short_life"] if ingredient.is_short_life else self.safety_stock_factor["normal"]
            safety_stock = avg_demand * safety_stock_factor
            
            current_stock = current_inventory.get(ingredient_id, 0)
            
            suggested_quantity = (total_demand * adjustment + safety_stock - current_stock)
            suggested_quantity = max(0, round(suggested_quantity, 2))
            
            reason_parts = []
            if ingredient.is_short_life:
                reason_parts.append(f"短保原料(保质期{shelf_life}天)，已降低备货")
            else:
                reason_parts.append(f"保质期{shelf_life}天")
            
            reason_parts.append(f"预测{cover_days}天需求: {total_demand:.1f}{ingredient.unit}")
            reason_parts.append(f"当前库存: {current_stock:.1f}{ingredient.unit}")
            
            if adjustment < 1.0:
                reason_parts.append(f"短保调整系数: {adjustment:.1%}")
            
            reason = "；".join(reason_parts)
            
            avg_accuracy = 0.0
            count = 0
            for drink_id, drink in self.data_storage.drinks.items():
                if ingredient_id in drink.recipe:
                    avg_accuracy += self.get_forecast_accuracy(store_id, drink_id)
                    count += 1
            confidence = avg_accuracy / count if count > 0 else 0.7
            
            suggestion = PurchaseSuggestion(
                store_id=store_id,
                ingredient_id=ingredient_id,
                suggestion_date=future_dates[purchase_day],
                suggested_quantity=suggested_quantity,
                reason=reason,
                confidence_level=round(confidence, 2)
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def generate_centralized_purchase_list(self,
                                           future_dates: List[date],
                                           weather_forecast: Dict[str, List[WeatherType]],
                                           temperature_forecast: Dict[str, List[float]],
                                           holiday_types: List[HolidayType]) -> pd.DataFrame:
        all_suggestions = []
        
        for store_id in self.data_storage.stores:
            store_weather = weather_forecast.get(store_id, [WeatherType.SUNNY] * len(future_dates))
            store_temp = temperature_forecast.get(store_id, [25.0] * len(future_dates))
            
            suggestions = self.generate_purchase_suggestions(
                store_id=store_id,
                future_dates=future_dates,
                weather_forecast=store_weather,
                temperature_forecast=store_temp,
                holiday_types=holiday_types
            )
            
            for sugg in suggestions:
                ingredient = self.data_storage.ingredients.get(sugg.ingredient_id)
                store = self.data_storage.stores.get(sugg.store_id)
                
                all_suggestions.append({
                    "门店ID": sugg.store_id,
                    "门店名称": store.name if store else "",
                    "原料ID": sugg.ingredient_id,
                    "原料名称": ingredient.name if ingredient else "",
                    "原料分类": ingredient.category if ingredient else "",
                    "建议采购量": sugg.suggested_quantity,
                    "单位": ingredient.unit if ingredient else "",
                    "是否短保": "是" if ingredient and ingredient.is_short_life else "否",
                    "置信度": sugg.confidence_level,
                    "建议日期": sugg.suggestion_date,
                    "说明": sugg.reason
                })
        
        df = pd.DataFrame(all_suggestions)
        
        summary = df.groupby(["原料ID", "原料名称", "单位", "原料分类", "是否短保"]).agg({
            "建议采购量": "sum",
            "门店ID": "count"
        }).reset_index()
        summary = summary.rename(columns={"门店ID": "涉及门店数"})
        
        return df, summary
    
    def update_model_with_new_data(self, new_sales_data: List[Dict],
                                   new_inventory_data: List[Dict]) -> Dict:
        from models import DailySalesRecord, InventoryRecord
        
        for data in new_sales_data:
            record = DailySalesRecord(
                store_id=data["store_id"],
                drink_id=data["drink_id"],
                sale_date=data["sale_date"],
                quantity=data["quantity"],
                temperature=data["temperature"],
                weather=data["weather"],
                holiday_type=data["holiday_type"],
                has_promotion=data.get("has_promotion", False),
                has_new_product=data.get("has_new_product", False),
                business_activity=data.get("business_activity", "")
            )
            self.data_storage.add_sales_record(record)
        
        for data in new_inventory_data:
            record = InventoryRecord(
                store_id=data["store_id"],
                ingredient_id=data["ingredient_id"],
                record_date=data["record_date"],
                beginning_inventory=data["beginning_inventory"],
                purchased_quantity=data["purchased_quantity"],
                used_quantity=data["used_quantity"],
                wasted_quantity=data["wasted_quantity"],
                ending_inventory=data["ending_inventory"]
            )
            self.data_storage.add_inventory_record(record)
        
        metrics = self.sales_predictor.train(self.data_storage, retrain=True)
        
        self.data_storage.save_to_files()
        
        return metrics
