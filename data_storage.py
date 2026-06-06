import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
import json
import os

from models import (
    Store, Ingredient, Drink, DailySalesRecord,
    InventoryRecord, WeatherType, HolidayType
)


class DataStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.stores: Dict[str, Store] = {}
        self.ingredients: Dict[str, Ingredient] = {}
        self.drinks: Dict[str, Drink] = {}
        self.sales_records: List[DailySalesRecord] = []
        self.inventory_records: List[InventoryRecord] = []
    
    def add_store(self, store: Store):
        self.stores[store.store_id] = store
    
    def add_ingredient(self, ingredient: Ingredient):
        self.ingredients[ingredient.ingredient_id] = ingredient
    
    def add_drink(self, drink: Drink):
        self.drinks[drink.drink_id] = drink
    
    def add_sales_record(self, record: DailySalesRecord):
        self.sales_records.append(record)
    
    def add_inventory_record(self, record: InventoryRecord):
        self.inventory_records.append(record)
    
    def get_sales_dataframe(self) -> pd.DataFrame:
        data = []
        for record in self.sales_records:
            data.append({
                "store_id": record.store_id,
                "drink_id": record.drink_id,
                "sale_date": record.sale_date,
                "quantity": record.quantity,
                "temperature": record.temperature,
                "weather": record.weather.value,
                "holiday_type": record.holiday_type.value,
                "has_promotion": record.has_promotion,
                "has_new_product": record.has_new_product,
                "business_activity": record.business_activity
            })
        return pd.DataFrame(data)
    
    def get_inventory_dataframe(self) -> pd.DataFrame:
        data = []
        for record in self.inventory_records:
            data.append({
                "store_id": record.store_id,
                "ingredient_id": record.ingredient_id,
                "record_date": record.record_date,
                "beginning_inventory": record.beginning_inventory,
                "purchased_quantity": record.purchased_quantity,
                "used_quantity": record.used_quantity,
                "wasted_quantity": record.wasted_quantity,
                "ending_inventory": record.ending_inventory
            })
        return pd.DataFrame(data)
    
    def save_to_files(self):
        stores_data = {
            sid: {
                "store_id": s.store_id,
                "name": s.name,
                "address": s.address,
                "district": s.district,
                "business_area": s.business_area
            }
            for sid, s in self.stores.items()
        }
        with open(os.path.join(self.data_dir, "stores.json"), "w", encoding="utf-8") as f:
            json.dump(stores_data, f, ensure_ascii=False, indent=2)
        
        ingredients_data = {
            iid: {
                "ingredient_id": i.ingredient_id,
                "name": i.name,
                "category": i.category,
                "shelf_life_days": i.shelf_life_days,
                "unit": i.unit,
                "is_short_life": i.is_short_life
            }
            for iid, i in self.ingredients.items()
        }
        with open(os.path.join(self.data_dir, "ingredients.json"), "w", encoding="utf-8") as f:
            json.dump(ingredients_data, f, ensure_ascii=False, indent=2)
        
        drinks_data = {
            did: {
                "drink_id": d.drink_id,
                "name": d.name,
                "category": d.category,
                "price": d.price,
                "recipe": d.recipe
            }
            for did, d in self.drinks.items()
        }
        with open(os.path.join(self.data_dir, "drinks.json"), "w", encoding="utf-8") as f:
            json.dump(drinks_data, f, ensure_ascii=False, indent=2)
        
        self.get_sales_dataframe().to_csv(
            os.path.join(self.data_dir, "sales_records.csv"),
            index=False, encoding="utf-8"
        )
        
        self.get_inventory_dataframe().to_csv(
            os.path.join(self.data_dir, "inventory_records.csv"),
            index=False, encoding="utf-8"
        )
    
    def load_from_files(self):
        if os.path.exists(os.path.join(self.data_dir, "stores.json")):
            with open(os.path.join(self.data_dir, "stores.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                for sid, s_data in data.items():
                    self.stores[sid] = Store(**s_data)
        
        if os.path.exists(os.path.join(self.data_dir, "ingredients.json")):
            with open(os.path.join(self.data_dir, "ingredients.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                for iid, i_data in data.items():
                    self.ingredients[iid] = Ingredient(**i_data)
        
        if os.path.exists(os.path.join(self.data_dir, "drinks.json")):
            with open(os.path.join(self.data_dir, "drinks.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                for did, d_data in data.items():
                    self.drinks[did] = Drink(**d_data)
        
        sales_path = os.path.join(self.data_dir, "sales_records.csv")
        if os.path.exists(sales_path):
            df = pd.read_csv(sales_path)
            for _, row in df.iterrows():
                self.sales_records.append(DailySalesRecord(
                    store_id=row["store_id"],
                    drink_id=row["drink_id"],
                    sale_date=datetime.strptime(row["sale_date"], "%Y-%m-%d").date(),
                    quantity=int(row["quantity"]),
                    temperature=float(row["temperature"]),
                    weather=WeatherType(row["weather"]),
                    holiday_type=HolidayType(row["holiday_type"]),
                    has_promotion=bool(row["has_promotion"]),
                    has_new_product=bool(row["has_new_product"]),
                    business_activity=str(row["business_activity"]) if pd.notna(row["business_activity"]) else ""
                ))
        
        inventory_path = os.path.join(self.data_dir, "inventory_records.csv")
        if os.path.exists(inventory_path):
            df = pd.read_csv(inventory_path)
            for _, row in df.iterrows():
                self.inventory_records.append(InventoryRecord(
                    store_id=row["store_id"],
                    ingredient_id=row["ingredient_id"],
                    record_date=datetime.strptime(row["record_date"], "%Y-%m-%d").date(),
                    beginning_inventory=float(row["beginning_inventory"]),
                    purchased_quantity=float(row["purchased_quantity"]),
                    used_quantity=float(row["used_quantity"]),
                    wasted_quantity=float(row["wasted_quantity"]),
                    ending_inventory=float(row["ending_inventory"])
                ))
