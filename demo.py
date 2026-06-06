import os
from datetime import date, datetime, timedelta
import pandas as pd

from data_storage import DataStorage
from sample_data_generator import SampleDataGenerator
from sales_predictor import SalesPredictor
from purchase_recommender import PurchaseRecommender
from waste_monitor import WasteMonitor
from models import WeatherType, HolidayType


def main():
    print("=" * 60)
    print("新式茶饮连锁 - 机器学习预测系统")
    print("=" * 60)
    print()
    
    data_storage = DataStorage()
    
    data_file = os.path.join("data", "stores.json")
    if not os.path.exists(data_file):
        print("未找到数据文件，正在生成示例数据...")
        print()
        generator = SampleDataGenerator(data_storage)
        generator.generate_all_data(years=2)
    else:
        print("加载已有数据...")
        data_storage.load_from_files()
        print(f"已加载 {len(data_storage.stores)} 家门店")
        print(f"已加载 {len(data_storage.ingredients)} 种原料")
        print(f"已加载 {len(data_storage.drinks)} 款饮品")
        print(f"已加载 {len(data_storage.sales_records)} 条销售记录")
        print(f"已加载 {len(data_storage.inventory_records)} 条库存记录")
    print()
    
    print("=" * 60)
    print("步骤1: 训练销量预测模型")
    print("=" * 60)
    print()
    
    predictor = SalesPredictor()
    model_file = os.path.join("models", "sales_predictor.pkl")
    if os.path.exists(model_file):
        print("加载已有模型...")
        predictor.load_models()
        print(f"已加载 {len(predictor.models)} 个预测模型")
    else:
        print("训练销量预测模型中...")
        metrics = predictor.train(data_storage)
        if "overall" in metrics:
            print(f"模型训练完成！共训练 {metrics['overall']['models_trained']} 个模型")
            print(f"平均 MAE: {metrics['overall']['avg_mae']:.2f}")
            print(f"平均 RMSE: {metrics['overall']['avg_rmse']:.2f}")
    print()
    
    print("=" * 60)
    print("步骤2: 单店饮品销量预测演示")
    print("=" * 60)
    print()
    
    store_id = "S001"
    drink_id = "D001"
    store = data_storage.stores[store_id]
    drink = data_storage.drinks[drink_id]
    
    future_dates = [date.today() + timedelta(days=i) for i in range(3)]
    weather_forecast = [WeatherType.SUNNY, WeatherType.CLOUDY, WeatherType.RAINY]
    temperature_forecast = [32.5, 28.0, 25.0]
    holiday_types = [HolidayType.WORKDAY, HolidayType.WORKDAY, HolidayType.WEEKEND]
    
    print(f"门店: {store.name} ({store_id})")
    print(f"饮品: {drink.name} ({drink_id})")
    print(f"预测未来3天销量:")
    print()
    
    sales_df = data_storage.get_sales_dataframe()
    predictions = predictor.predict(
        store_id=store_id,
        drink_id=drink_id,
        future_dates=future_dates,
        weather_forecast=weather_forecast,
        temperature_forecast=temperature_forecast,
        holiday_types=holiday_types,
        sales_df=sales_df
    )
    
    for pred in predictions:
        print(f"  {pred['date']}: {pred['predicted_quantity']} 杯 "
              f"(天气: {pred['weather']}, 气温: {pred['temperature']}℃, {pred['holiday_type']})")
    print()
    
    importance = predictor.get_feature_importance(store_id, drink_id)
    if importance:
        print("特征重要性 Top 5:")
        for i, (feature, score) in enumerate(list(importance.items())[:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    print()
    
    print("=" * 60)
    print("步骤3: 生成原料采购建议")
    print("=" * 60)
    print()
    
    recommender = PurchaseRecommender(data_storage, predictor)
    
    print(f"生成 {store.name} 未来3天的原料采购建议:")
    print()
    
    suggestions = recommender.generate_purchase_suggestions(
        store_id=store_id,
        future_dates=future_dates,
        weather_forecast=weather_forecast,
        temperature_forecast=temperature_forecast,
        holiday_types=holiday_types
    )
    
    print(f"{'原料名称':<10} {'分类':<8} {'是否短保':<8} {'建议采购量':<12} {'单位':<6} {'置信度':<8}")
    print("-" * 70)
    
    short_life_count = 0
    for sugg in suggestions:
        ingredient = data_storage.ingredients[sugg.ingredient_id]
        short_life_mark = "*" if ingredient.is_short_life else " "
        if ingredient.is_short_life:
            short_life_count += 1
        
        print(f"{ingredient.name:<10} {ingredient.category:<8} "
              f"{'是' if ingredient.is_short_life else '否':<8} "
              f"{sugg.suggested_quantity:>10.1f} {ingredient.unit:<6} "
              f"{sugg.confidence_level:>6.0%}")
    
    print()
    print(f"共 {len(suggestions)} 种原料，其中 {short_life_count} 种为短保原料"
          f"（已自动降低备货量）")
    print()
    
    print("=" * 60)
    print("步骤4: 生成全门店集中采购清单")
    print("=" * 60)
    print()
    
    weather_forecast_all = {}
    temperature_forecast_all = {}
    for sid in data_storage.stores:
        weather_forecast_all[sid] = weather_forecast
        temperature_forecast_all[sid] = temperature_forecast
    
    detail_df, summary_df = recommender.generate_centralized_purchase_list(
        future_dates=future_dates,
        weather_forecast=weather_forecast_all,
        temperature_forecast=temperature_forecast_all,
        holiday_types=holiday_types
    )
    
    print("采购汇总清单 (按原料汇总):")
    print()
    print(f"{'原料名称':<10} {'分类':<8} {'是否短保':<8} "
          f"{'总采购量':<12} {'单位':<6} {'涉及门店数':<8}")
    print("-" * 70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['原料名称']:<10} {row['原料分类']:<8} {row['是否短保']:<8} "
              f"{row['建议采购量']:>10.1f} {row['单位']:<6} {row['涉及门店数']:>8}")
    print()
    
    print("=" * 60)
    print("步骤5: 损耗监控与预警")
    print("=" * 60)
    print()
    
    monitor = WasteMonitor(data_storage)
    check_date = date.today() - timedelta(days=1)
    
    print("检查原料损耗预警...")
    print()
    
    alerts = monitor.check_waste_alerts(check_date=check_date, lookback_days=7)
    
    if alerts:
        print(f"发现 {len(alerts)} 个损耗预警:")
        print()
        for alert in alerts:
            store = data_storage.stores[alert.store_id]
            ingredient = data_storage.ingredients[alert.ingredient_id]
            print(f"【预警】门店: {store.name}, 原料: {ingredient.name}")
            print(f"       连续 {alert.consecutive_days} 天损耗率偏高, "
                  f"平均 {alert.avg_waste_rate:.1%}")
            print(f"       建议: {alert.suggestion}")
            print()
    else:
        print("未发现高损耗预警，各门店原料损耗水平正常")
    print()
    
    print("生成月度损耗报告...")
    start_date = check_date - timedelta(days=30)
    waste_report = monitor.generate_waste_report(start_date, check_date)
    
    if not waste_report.empty:
        print()
        print("损耗率最高的前5种原料 (按门店):")
        print()
        top_waste = waste_report.head(5)
        for _, row in top_waste.iterrows():
            print(f"  {row['store_name']} - {row['ingredient_name']}: "
                  f"{row['waste_rate']:.1%} "
                  f"(报废 {row['wasted_quantity']:.0f}{data_storage.ingredients[row['ingredient_id']].unit})")
    print()
    
    print("=" * 60)
    print("步骤6: 模型迭代更新演示")
    print("=" * 60)
    print()
    
    print("系统每周自动使用新数据迭代模型")
    print("当前可通过调用 update_model_with_new_data() 方法")
    print("传入新的销售和库存数据进行模型更新")
    print()
    
    print("=" * 60)
    print("系统演示完成！")
    print("=" * 60)
    print()
    print("核心功能总结:")
    print("  ✓ 9家门店销售数据采集与存储")
    print("  ✓ XGBoost时序预测模型 (气温、天气、节假日等特征)")
    print("  ✓ 未来3天分门店分饮品销量预测")
    print("  ✓ 原料采购建议生成 (短保原料自动降低备货)")
    print("  ✓ 全门店集中采购清单汇总")
    print("  ✓ 原料损耗实时监控与预警")
    print("  ✓ 每周模型迭代更新机制")
    print()


if __name__ == "__main__":
    main()
