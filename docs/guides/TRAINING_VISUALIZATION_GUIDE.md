# 训练可视化指南

## 📊 概述

新训练框架已集成可视化功能，可以自动生成样本质量检查和因子重要性分析的可视化图表，方便快速判断数据质量和模型特征重要性。

## 🚀 功能特性

### 1. 样本质量可视化

在训练过程中和样本质量检查时，会自动生成以下可视化图表：

- **涨幅分布图**：展示总涨幅和最高涨幅的分布情况
- **时间分布图**：展示样本在不同年份和月份的分布
- **涨幅箱线图**：展示涨幅的统计分布（中位数、四分位数等）
- **异常值检测图**：使用IQR方法检测异常值和极端值
- **质量报告**：生成JSON格式的综合质量报告

### 2. 因子重要性可视化

训练完成后，会自动生成以下因子重要性分析图表：

- **Top 20 特征重要性条形图**：最重要的20个特征及其重要性得分
- **重要性分布直方图**：所有特征重要性的分布情况
- **类别汇总热力图**：按特征类别（价格、涨跌幅、成交量等）汇总的重要性
- **累积重要性图**：显示需要多少特征达到80%/90%的累积重要性

## 📁 输出位置

### 新训练框架（ModelTrainer）

图表保存在模型版本目录下：
```
data/models/{model_name}/versions/{version}/charts/
├── index.html                          # 索引页面
├── positive_sample_quality_*.html      # 正样本质量图表
├── negative_sample_quality_*.html     # 负样本质量图表
└── {model_name}_{version}_feature_*.html  # 因子重要性图表
```

### 旧训练脚本（train_xgboost_timeseries.py）

图表保存在：
```
data/training/charts/
├── index.html
├── positive_sample_quality_*.html
├── negative_sample_quality_*.html
└── xgboost_timeseries_*_feature_*.html
```

### 样本质量检查工具

图表保存在：
```
data/training/charts/
├── index.html
└── sample_quality_check_*.html
```

## 🔧 使用方法

### 方法1：自动生成（推荐）

在新训练框架中，所有可视化会自动生成：

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('breakout_launch_scorer')
model, metrics = trainer.train_version()
# 所有可视化图表会自动生成到 data/models/breakout_launch_scorer/versions/{version}/charts/
# 包括：
# - 样本质量可视化
# - 特征质量评估可视化
# - 因子重要性可视化
# - 模型训练过程可视化
# - 模型结果评测可视化
```

### 方法2：样本质量检查

运行样本质量检查工具，会自动生成可视化：

```bash
python scripts/check_sample_quality.py
# 图表会生成到 data/training/charts/
```

### 方法3：手动调用

如果需要手动生成可视化：

```python
from src.visualization.training_visualizer import TrainingVisualizer
import pandas as pd

# 创建可视化器
visualizer = TrainingVisualizer(output_dir="data/training/charts")

# 样本质量可视化
df_samples = pd.read_csv('data/training/samples/positive_samples.csv')
visualizer.visualize_sample_quality(df_samples, save_prefix="my_samples")

# 因子重要性可视化
feature_importance = pd.DataFrame({
    'feature': ['feature1', 'feature2', ...],
    'importance': [0.1, 0.2, ...]
})
visualizer.visualize_feature_importance(
    feature_importance, 
    model_name="my_model",
    top_n=20
)

# 生成索引页面
visualizer.generate_index_page(model_name="my_model")
```

## 📈 图表说明

### 样本质量图表

1. **涨幅分布图**
   - 横轴：涨幅百分比
   - 纵轴：样本数量
   - 用途：检查涨幅分布是否合理，是否有异常峰值

2. **时间分布图**
   - 上部分：按年份分布
   - 下部分：按月份分布
   - 用途：检查样本时间分布是否均匀，是否存在时间偏差

3. **涨幅箱线图**
   - 显示中位数、四分位数、异常值
   - 用途：快速了解涨幅的统计特征

4. **异常值检测图**
   - 绿色：正常值
   - 橙色：IQR方法检测的异常值
   - 红色：极端值（>200%）
   - 用途：识别需要特别关注的异常样本

### 因子重要性图表

1. **Top 20 特征重要性条形图**
   - 横轴：重要性得分
   - 纵轴：特征名称
   - 用途：快速识别最重要的特征

2. **重要性分布直方图**
   - 显示所有特征重要性的分布
   - 包含平均值和中位数线
   - 用途：了解特征重要性的整体分布

3. **类别汇总热力图**
   - 按特征类别汇总重要性
   - 颜色越深表示该类别越重要
   - 用途：了解哪些类型的特征更重要

4. **累积重要性图**
   - 显示累积重要性百分比
   - 标注80%和90%阈值
   - 用途：确定需要保留多少特征

## 💡 使用建议

1. **训练前检查样本质量**
   - 运行 `python scripts/check_sample_quality.py`
   - 查看生成的图表，确保样本质量良好

2. **训练后分析因子重要性**
   - 训练完成后，自动生成的图表会保存在模型版本目录下
   - 打开 `index.html` 查看所有图表
   - 重点关注Top 20特征，考虑是否需要调整特征工程

3. **对比不同版本**
   - 不同版本的图表保存在各自的版本目录下
   - 可以对比不同版本的因子重要性，了解模型变化

## 🔍 故障排除

### 问题1：图表未生成

**可能原因**：
- 缺少依赖包（plotly）
- 输出目录权限问题
- 数据文件不存在

**解决方法**：
```bash
# 安装依赖
pip install plotly

# 检查目录权限
ls -la data/training/charts/

# 检查数据文件
ls -la data/training/samples/
```

### 问题2：图表显示异常

**可能原因**：
- 数据格式不正确
- 缺少必要的列

**解决方法**：
- 检查数据文件是否包含必需的列（如 `total_return`, `t1_date` 等）
- 查看日志中的错误信息

### 问题3：中文显示乱码

**解决方法**：
- 图表使用HTML格式，浏览器会自动处理中文
- 如果仍有问题，检查浏览器编码设置

## 📝 注意事项

1. **图表格式**：所有图表使用Plotly生成，保存为HTML格式，可以在浏览器中直接打开
2. **文件大小**：HTML文件可能较大（几MB），但包含完整的交互功能
3. **浏览器兼容性**：建议使用Chrome、Firefox或Edge等现代浏览器
4. **自动生成**：在新训练框架中，可视化会自动生成，无需手动调用

## 🎯 下一步

- 查看生成的图表，分析样本质量和因子重要性
- 根据分析结果调整特征工程或模型参数
- 对比不同版本的图表，跟踪模型改进情况

