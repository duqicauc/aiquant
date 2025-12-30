# 模型生命周期和集成测试用例

**日期**: 2025-12-30

---

## 📋 新增测试文件

### 1. 模型训练器测试 (`test_model_trainer.py`)

**测试类**: `TestModelTrainer`

**测试用例** (10+个)：

| 测试用例 | 说明 |
|---------|------|
| `test_init` | 测试训练器初始化 |
| `test_increment_version` | 测试版本号递增逻辑 |
| `test_load_and_prepare_data` | 测试数据加载和准备 |
| `test_extract_features` | 测试特征提取 |
| `test_timeseries_split` | 测试时间序列划分 |
| `test_train_model` | 测试模型训练 |
| `test_save_model` | 测试模型保存 |

**测试类**: `TestModelTrainerIntegration`

| 测试用例 | 说明 |
|---------|------|
| `test_full_training_workflow` | 完整训练流程（标记为slow） |
| `test_train_version_creates_new_version` | 测试训练创建新版本 |

---

### 2. 模型预测器测试 (`test_model_predictor.py`)

**测试类**: `TestModelPredictor`

**测试用例** (10+个)：

| 测试用例 | 说明 |
|---------|------|
| `test_init` | 测试预测器初始化 |
| `test_load_model` | 测试模型加载 |
| `test_get_valid_stocks` | 测试获取有效股票列表 |
| `test_extract_stock_features` | 测试提取股票特征 |
| `test_extract_stock_features_insufficient_data` | 测试数据不足情况 |
| `test_predict_with_mock_model` | 测试预测（使用Mock模型） |
| `test_save_predictions` | 测试保存预测结果 |

**测试类**: `TestModelPredictorIntegration`

| 测试用例 | 说明 |
|---------|------|
| `test_full_prediction_workflow` | 完整预测流程（标记为slow） |
| `test_predict_with_latest_version` | 测试使用最新版本预测 |

---

### 3. 训练流程集成测试 (`test_model_training_pipeline.py`)

**测试类**: `TestModelTrainingPipeline`

**测试用例** (4个)：

| 测试用例 | 说明 |
|---------|------|
| `test_training_pipeline_creates_version` | 测试训练流程创建版本 |
| `test_training_pipeline_version_metadata` | 测试训练流程更新版本元数据 |
| `test_training_pipeline_with_real_data` | 使用真实数据的训练流程（标记为slow） |
| `test_training_pipeline_increments_version` | 测试训练流程自动递增版本号 |

**测试类**: `TestModelTrainingAndVersionManagement`

| 测试用例 | 说明 |
|---------|------|
| `test_training_and_version_promotion` | 测试训练后版本提升 |

---

### 4. 预测流程集成测试 (`test_model_prediction_pipeline.py`)

**测试类**: `TestModelPredictionPipeline`

**测试用例** (5个)：

| 测试用例 | 说明 |
|---------|------|
| `test_prediction_pipeline_loads_model` | 测试预测流程加载模型 |
| `test_prediction_pipeline_extracts_features` | 测试预测流程特征提取 |
| `test_prediction_pipeline_full_workflow` | 测试完整预测流程 |
| `test_prediction_pipeline_with_latest_version` | 测试使用最新版本预测 |
| `test_prediction_pipeline_saves_metadata` | 测试预测流程保存元数据 |

**测试类**: `TestTrainingAndPredictionIntegration`

| 测试用例 | 说明 |
|---------|------|
| `test_train_then_predict_workflow` | 测试训练后立即预测的完整流程 |

---

## 📊 测试统计

| 类别 | 测试文件数 | 测试用例数 | 覆盖模块 |
|------|-----------|-----------|---------|
| 模型训练器 | 1 | 10+ | ModelTrainer |
| 模型预测器 | 1 | 10+ | ModelPredictor |
| 训练流程集成 | 1 | 5 | 训练流程 |
| 预测流程集成 | 1 | 6 | 预测流程 |
| **总计** | **4** | **30+** | **4个核心模块** |

---

## 🚀 运行测试

### 运行所有新增测试

```bash
# 运行模型生命周期测试
pytest tests/models/test_model_trainer.py tests/models/test_model_predictor.py -v

# 运行集成测试
pytest tests/integration/test_model_training_pipeline.py tests/integration/test_model_prediction_pipeline.py -v

# 运行所有新增测试（排除slow标记）
pytest tests/models/test_model_trainer.py \
       tests/models/test_model_predictor.py \
       tests/integration/test_model_training_pipeline.py \
       tests/integration/test_model_prediction_pipeline.py \
       -v -m "not slow"
```

### 运行特定测试

```bash
# 只运行单元测试
pytest tests/models/test_model_trainer.py::TestModelTrainer -v

# 只运行集成测试
pytest tests/integration/ -v -m integration

# 运行包含slow标记的测试
pytest tests/models/test_model_trainer.py -v -m slow
```

### 查看覆盖率

```bash
# 查看模型生命周期模块覆盖率
pytest tests/models/test_model_trainer.py \
       tests/models/test_model_predictor.py \
       --cov=src/models/lifecycle \
       --cov-report=term-missing \
       --cov-report=html:htmlcov

# 查看完整覆盖率报告
open htmlcov/index.html
```

---

## 📈 覆盖率提升

### 模型生命周期模块

| 模块 | 之前覆盖率 | 预计覆盖率 | 提升 |
|------|-----------|-----------|------|
| `ModelTrainer` | ~9% | ~60% | +51% |
| `ModelPredictor` | ~10% | ~65% | +55% |
| **总体** | **~30%** | **~60%** | **+30%** |

### 集成测试

| 测试类型 | 之前 | 新增 | 状态 |
|---------|------|------|------|
| 训练流程 | 0个 | 5个 | ✅ |
| 预测流程 | 0个 | 6个 | ✅ |
| 训练+预测 | 0个 | 1个 | ✅ |

---

## 🎯 测试覆盖的功能点

### ModelTrainer

- ✅ 初始化
- ✅ 版本创建和管理
- ✅ 数据加载和准备
- ✅ 特征提取
- ✅ 时间序列划分
- ✅ 模型训练
- ✅ 模型保存
- ✅ 版本元数据更新
- ✅ 版本号自动递增

### ModelPredictor

- ✅ 初始化
- ✅ 模型加载
- ✅ 股票列表获取和筛选
- ✅ 特征提取
- ✅ 批量预测
- ✅ 结果保存
- ✅ 元数据保存
- ✅ 使用最新版本预测

### 集成流程

- ✅ 完整训练流程
- ✅ 完整预测流程
- ✅ 训练后预测流程
- ✅ 版本管理和提升

---

## ⚠️ 注意事项

1. **Mock数据**: 大部分测试使用Mock数据，确保快速执行
2. **Slow标记**: 需要真实数据的测试标记为`@pytest.mark.slow`，默认跳过
3. **临时目录**: 所有测试使用临时目录，不会影响实际数据
4. **依赖Mock**: 测试依赖Mock的DataManager，避免真实API调用

---

## 🔄 后续改进建议

1. **真实数据测试**: 添加使用真实数据的集成测试（标记为slow）
2. **错误处理测试**: 增加异常情况的测试用例
3. **性能测试**: 添加大数据量的性能测试
4. **边界条件**: 增加边界条件和极端情况的测试

---

**最后更新**: 2025-12-30

