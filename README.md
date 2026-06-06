# 新式茶饮连锁 - 机器学习预测系统

基于近2年门店销售数据搭建的机器学习预测系统，用于优化原料采购和降低损耗。

## 系统架构

```
├── models.py              # 数据模型定义
├── data_storage.py        # 数据存储模块
├── sales_predictor.py     # 销量预测模块 (XGBoost)
├── purchase_recommender.py # 采购建议模块
├── waste_monitor.py       # 损耗监控模块
├── sample_data_generator.py # 示例数据生成器
├── demo.py                # 系统演示脚本
├── requirements.txt       # 依赖包
├── data/                  # 数据文件目录
└── models/                # 模型文件目录
```

## 核心功能

### 1. 数据模型
- **门店信息**: 9家街边门店的基本信息
- **原料信息**: 茶叶、鲜奶、鲜果等15种原料，包含保质期信息
- **饮品信息**: 8款饮品及其配方
- **销售记录**: 每日销量、气温、天气、节假日等特征
- **库存记录**: 原料出入库和报废记录

### 2. 销量预测模型 (XGBoost)
**训练特征**:
- 时序特征: 历史销量滞后值(7天、14天)、移动平均值、标准差
- 天气特征: 当日气温、气温平方、天气类型
- 日期特征: 星期几、日、月、周、是否周末/节假日
- 活动特征: 促销活动、新品上市、商圈活动

### 3. 采购建议生成
- 基于饮品销量预测计算原料需求
- 考虑原料保质期，短保原料自动降低备货量(60%-70%)
- 结合当前库存水平计算建议采购量
- 支持分门店采购和集中采购两种模式

### 4. 损耗监控与预警
- 实时计算原料损耗率
- 识别连续3天以上高损耗原料
- 自动发出预警并给出备货调整建议
- 生成门店和原料损耗排名报告

### 5. 模型迭代机制
- 每周用新销售和库存数据重新训练模型
- 持续提升预测精准度

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行系统演示
```bash
python demo.py
```

演示脚本会自动:
- 生成9家门店2年的示例销售和库存数据
- 训练XGBoost销量预测模型
- 演示未来3天销量预测
- 生成原料采购建议
- 展示损耗监控和预警功能

## 文件说明

### 数据模型 ([models.py](file:///e:/solo/hjh514/models.py))
定义了所有核心数据类：
- `Store`: 门店信息
- `Ingredient`: 原料信息 (含保质期标记)
- `Drink`: 饮品信息 (含配方)
- `DailySalesRecord`: 每日销售记录
- `InventoryRecord`: 库存记录
- `PurchaseSuggestion`: 采购建议
- `WasteAlert`: 损耗预警

### 数据存储 ([data_storage.py](file:///e:/solo/hjh514/data_storage.py))
- 内存数据管理
- JSON/CSV 文件持久化
- DataFrame 格式转换

### 销量预测 ([sales_predictor.py](file:///e:/solo/hjh514/sales_predictor.py))
- 针对每家门店每款饮品单独训练模型
- 支持模型保存和加载
- 特征重要性分析

### 采购建议 ([purchase_recommender.py](file:///e:/solo/hjh514/purchase_recommender.py))
- 原料需求换算 (饮品销量 × 配方用量)
- 保质期适配调整
- 安全库存计算
- 集中采购清单汇总

### 损耗监控 ([waste_monitor.py](file:///e:/solo/hjh514/waste_monitor.py))
- 损耗率计算 (报废量 / 总可用量)
- 连续高损耗检测
- 损耗报告生成
- 门店/原料损耗排名

## 业务价值

1. **降低缺货率**: 高温天气热销饮品提前备货
2. **减少损耗**: 阴雨天短保原料自动缩减备货建议
3. **集中采购**: 采购部门统一配货，降低采购成本
4. **数据驱动**: 从经验备货转向模型预测
5. **持续优化**: 每周模型迭代，预测精度不断提升

## 预测特征说明

| 特征类别 | 具体特征 |
|---------|---------|
| 时序特征 | lag_7, lag_14, rolling_mean_7/14, rolling_std_7 |
| 天气特征 | temperature, temperature_squared, weather |
| 日期特征 | day_of_week, day_of_month, month, week_of_year, is_weekend, is_holiday |
| 活动特征 | has_promotion, has_new_product, business_activity |

## 损耗预警阈值

| 原料类型 | 损耗率阈值 |
|---------|-----------|
| 普通原料 | 15% |
| 短保原料 | 25% |
