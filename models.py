from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date, datetime
from enum import Enum


class WeatherType(Enum):
    SUNNY = "晴"
    CLOUDY = "多云"
    OVERCAST = "阴"
    RAINY = "雨"
    SNOWY = "雪"


class HolidayType(Enum):
    WORKDAY = "工作日"
    WEEKEND = "周末"
    HOLIDAY = "节假日"


@dataclass
class Store:
    store_id: str
    name: str
    address: str
    district: str
    business_area: str


@dataclass
class Ingredient:
    ingredient_id: str
    name: str
    category: str
    shelf_life_days: int
    unit: str
    is_short_life: bool = False


@dataclass
class Drink:
    drink_id: str
    name: str
    category: str
    price: float
    recipe: Dict[str, float] = field(default_factory=dict)


@dataclass
class DailySalesRecord:
    store_id: str
    drink_id: str
    sale_date: date
    quantity: int
    temperature: float
    weather: WeatherType
    holiday_type: HolidayType
    has_promotion: bool = False
    has_new_product: bool = False
    business_activity: str = ""


@dataclass
class InventoryRecord:
    store_id: str
    ingredient_id: str
    record_date: date
    beginning_inventory: float
    purchased_quantity: float
    used_quantity: float
    wasted_quantity: float
    ending_inventory: float


@dataclass
class PurchaseSuggestion:
    store_id: str
    ingredient_id: str
    suggestion_date: date
    suggested_quantity: float
    reason: str
    confidence_level: float


@dataclass
class WasteAlert:
    store_id: str
    ingredient_id: str
    alert_date: date
    avg_waste_rate: float
    threshold: float
    consecutive_days: int
    suggestion: str
