import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict
import random

from models import (
    Store, Ingredient, Drink, DailySalesRecord, InventoryRecord,
    WeatherType, HolidayType
)
from data_storage import DataStorage


class SampleDataGenerator:
    def __init__(self, data_storage: DataStorage):
        self.data_storage = data_storage
        np.random.seed(42)
        random.seed(42)
    
    def generate_stores(self):
        stores_data = [
            Store("S001", "朝阳路店", "北京市朝阳区朝阳路1号", "朝阳区", "CBD商圈"),
            Store("S002", "海淀大街店", "北京市海淀区海淀大街2号", "海淀区", "中关村商圈"),
            Store("S003", "西直门旗舰店", "北京市西城区西直门外大街3号", "西城区", "西直门商圈"),
            Store("S004", "东单店", "北京市东城区东单北大街4号", "东城区", "王府井商圈"),
            Store("S005", "三里屯店", "北京市朝阳区三里屯路5号", "朝阳区", "三里屯商圈"),
            Store("S006", "国贸店", "北京市朝阳区建国门外大街6号", "朝阳区", "国贸商圈"),
            Store("S007", "五道口店", "北京市海淀区成府路7号", "海淀区", "五道口商圈"),
            Store("S008", "望京店", "北京市朝阳区望京街8号", "朝阳区", "望京商圈"),
            Store("S009", "西单店", "北京市西城区西单北大街9号", "西城区", "西单商圈"),
        ]
        
        for store in stores_data:
            self.data_storage.add_store(store)
        
        return stores_data
    
    def generate_ingredients(self):
        ingredients_data = [
            Ingredient("I001", "红茶", "茶叶", 180, "g", False),
            Ingredient("I002", "绿茶", "茶叶", 180, "g", False),
            Ingredient("I003", "乌龙茶", "茶叶", 180, "g", False),
            Ingredient("I004", "鲜奶", "乳制品", 3, "ml", True),
            Ingredient("I005", "芝士", "乳制品", 7, "g", True),
            Ingredient("I006", "珍珠", "配料", 90, "g", False),
            Ingredient("I007", "椰果", "配料", 90, "g", False),
            Ingredient("I008", "芋圆", "配料", 90, "g", False),
            Ingredient("I009", "草莓", "水果", 2, "g", True),
            Ingredient("I010", "芒果", "水果", 3, "g", True),
            Ingredient("I011", "柠檬", "水果", 7, "g", True),
            Ingredient("I012", "水蜜桃", "水果", 3, "g", True),
            Ingredient("I013", "西瓜", "水果", 1, "g", True),
            Ingredient("I014", "糖浆", "配料", 365, "ml", False),
            Ingredient("I015", "冰块", "配料", 1, "g", False),
        ]
        
        for ingredient in ingredients_data:
            self.data_storage.add_ingredient(ingredient)
        
        return ingredients_data
    
    def generate_drinks(self):
        drinks_data = [
            Drink("D001", "珍珠奶茶", "奶茶", 18, {
                "I001": 5.0, "I004": 200.0, "I006": 50.0, "I014": 20.0
            }),
            Drink("D002", "芝士奶盖绿茶", "奶茶", 22, {
                "I002": 5.0, "I004": 150.0, "I005": 30.0, "I014": 20.0
            }),
            Drink("D003", "杨枝甘露", "水果茶", 25, {
                "I004": 100.0, "I010": 80.0, "I007": 30.0, "I014": 15.0
            }),
            Drink("D004", "草莓多多", "水果茶", 23, {
                "I009": 100.0, "I004": 150.0, "I014": 20.0
            }),
            Drink("D005", "蜜桃乌龙", "纯茶", 15, {
                "I003": 6.0, "I012": 50.0, "I014": 15.0
            }),
            Drink("D006", "柠檬绿茶", "水果茶", 16, {
                "I002": 5.0, "I011": 30.0, "I014": 25.0
            }),
            Drink("D007", "芋圆奶茶", "奶茶", 19, {
                "I001": 5.0, "I004": 200.0, "I008": 40.0, "I014": 20.0
            }),
            Drink("D008", "西瓜啵啵", "水果茶", 20, {
                "I013": 150.0, "I006": 30.0, "I014": 15.0
            }),
        ]
        
        for drink in drinks_data:
            self.data_storage.add_drink(drink)
        
        return drinks_data
    
    def generate_weather_for_date(self, sale_date: date) -> tuple:
        month = sale_date.month
        
        if 6 <= month <= 8:
            avg_temp = 28 + np.random.normal(0, 3)
            weather_options = [WeatherType.SUNNY, WeatherType.SUNNY, WeatherType.CLOUDY, WeatherType.RAINY]
        elif month in [12, 1, 2]:
            avg_temp = 0 + np.random.normal(0, 4)
            weather_options = [WeatherType.SUNNY, WeatherType.CLOUDY, WeatherType.SNOWY]
        elif 3 <= month <= 5:
            avg_temp = 18 + np.random.normal(0, 4)
            weather_options = [WeatherType.SUNNY, WeatherType.CLOUDY, WeatherType.RAINY]
        else:
            avg_temp = 15 + np.random.normal(0, 4)
            weather_options = [WeatherType.SUNNY, WeatherType.CLOUDY, WeatherType.RAINY]
        
        temperature = round(max(-10, min(40, avg_temp)), 1)
        weather = random.choice(weather_options)
        
        return temperature, weather
    
    def get_holiday_type(self, sale_date: date) -> HolidayType:
        day_of_week = sale_date.weekday()
        
        if day_of_week >= 5:
            return HolidayType.WEEKEND
        
        holidays = [
            (1, 1), (1, 2), (1, 3),
            (5, 1), (5, 2), (5, 3),
            (10, 1), (10, 2), (10, 3),
        ]
        
        if (sale_date.month, sale_date.day) in holidays:
            return HolidayType.HOLIDAY
        
        return HolidayType.WORKDAY
    
    def generate_sales_records(self, start_date: date, end_date: date):
        delta = end_date - start_date
        
        for store_id in self.data_storage.stores:
            store_factor = {
                "S001": 1.0, "S002": 1.2, "S003": 1.5, "S004": 1.3,
                "S005": 1.4, "S006": 1.6, "S007": 0.9, "S008": 1.1, "S009": 1.4
            }.get(store_id, 1.0)
            
            for i in range(delta.days + 1):
                sale_date = start_date + timedelta(days=i)
                
                temperature, weather = self.generate_weather_for_date(sale_date)
                holiday_type = self.get_holiday_type(sale_date)
                
                temp_factor = 1.0
                if temperature > 30:
                    temp_factor = 1.3
                elif temperature < 10:
                    temp_factor = 0.7
                
                weekend_factor = 1.2 if holiday_type != HolidayType.WORKDAY else 1.0
                
                weather_factor = 1.0
                if weather in [WeatherType.RAINY, WeatherType.SNOWY]:
                    weather_factor = 0.8
                
                has_promotion = random.random() < 0.1
                promotion_factor = 1.3 if has_promotion else 1.0
                
                has_new_product = random.random() < 0.05
                new_product_factor = 1.2 if has_new_product else 1.0
                
                business_activity = ""
                if has_promotion:
                    business_activity = "促销"
                elif has_new_product:
                    business_activity = "新品上市"
                
                for drink_id in self.data_storage.drinks:
                    drink = self.data_storage.drinks[drink_id]
                    
                    base_sales = {
                        "D001": 80, "D002": 60, "D003": 50, "D004": 45,
                        "D005": 35, "D006": 40, "D007": 55, "D008": 40
                    }.get(drink_id, 50)
                    
                    seasonal_factor = 1.0
                    if "水果" in drink.category:
                        if 6 <= sale_date.month <= 8:
                            seasonal_factor = 1.4
                        elif 12 <= sale_date.month <= 2:
                            seasonal_factor = 0.6
                    
                    expected = (base_sales * store_factor * temp_factor * weekend_factor * 
                              weather_factor * promotion_factor * new_product_factor * seasonal_factor)
                    
                    quantity = max(5, int(np.random.normal(expected, expected * 0.15)))
                    
                    record = DailySalesRecord(
                        store_id=store_id,
                        drink_id=drink_id,
                        sale_date=sale_date,
                        quantity=quantity,
                        temperature=temperature,
                        weather=weather,
                        holiday_type=holiday_type,
                        has_promotion=has_promotion,
                        has_new_product=has_new_product,
                        business_activity=business_activity
                    )
                    
                    self.data_storage.add_sales_record(record)
    
    def generate_inventory_records(self, start_date: date, end_date: date):
        delta = end_date - start_date
        
        for store_id in self.data_storage.stores:
            for ingredient_id in self.data_storage.ingredients:
                ingredient = self.data_storage.ingredients[ingredient_id]
                
                ending_inventory = 1000.0 if ingredient.is_short_life else 5000.0
                
                for i in range(delta.days + 1):
                    record_date = start_date + timedelta(days=i)
                    
                    base_usage = 0
                    for drink_id, drink in self.data_storage.drinks.items():
                        if ingredient_id in drink.recipe:
                            sales_records = [
                                r for r in self.data_storage.sales_records
                                if r.store_id == store_id and 
                                r.drink_id == drink_id and 
                                r.sale_date == record_date
                            ]
                            for sr in sales_records:
                                base_usage += sr.quantity * drink.recipe[ingredient_id]
                    
                    usage_variance = np.random.normal(1.0, 0.05)
                    used_quantity = base_usage * usage_variance
                    
                    waste_rate = 0.05
                    if ingredient.is_short_life:
                        waste_rate = 0.15 if random.random() < 0.3 else 0.08
                    
                    wasted_quantity = ending_inventory * waste_rate
                    
                    safety_factor = 1.2 if not ingredient.is_short_life else 0.8
                    purchased_quantity = max(0, used_quantity * safety_factor - ending_inventory + used_quantity)
                    purchased_quantity = round(purchased_quantity, 2)
                    
                    record = InventoryRecord(
                        store_id=store_id,
                        ingredient_id=ingredient_id,
                        record_date=record_date,
                        beginning_inventory=round(ending_inventory, 2),
                        purchased_quantity=purchased_quantity,
                        used_quantity=round(used_quantity, 2),
                        wasted_quantity=round(wasted_quantity, 2),
                        ending_inventory=round(ending_inventory - used_quantity - wasted_quantity + purchased_quantity, 2)
                    )
                    
                    self.data_storage.add_inventory_record(record)
                    
                    ending_inventory = record.ending_inventory
    
    def generate_all_data(self, years: int = 2):
        print("正在生成门店数据...")
        self.generate_stores()
        print(f"生成 {len(self.data_storage.stores)} 家门店")
        
        print("正在生成原料数据...")
        self.generate_ingredients()
        print(f"生成 {len(self.data_storage.ingredients)} 种原料")
        
        print("正在生成饮品数据...")
        self.generate_drinks()
        print(f"生成 {len(self.data_storage.drinks)} 款饮品")
        
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=365 * years)
        
        print(f"正在生成 {years} 年销售数据...")
        self.generate_sales_records(start_date, end_date)
        print(f"生成 {len(self.data_storage.sales_records)} 条销售记录")
        
        print("正在生成库存和损耗数据...")
        self.generate_inventory_records(start_date, end_date)
        print(f"生成 {len(self.data_storage.inventory_records)} 条库存记录")
        
        print("保存数据到文件...")
        self.data_storage.save_to_files()
        
        print("示例数据生成完成！")
