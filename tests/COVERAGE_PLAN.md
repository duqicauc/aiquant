# 测试覆盖率提升计划

## 目标：85%覆盖率

当前覆盖率：~10%，需要大幅提升。

## 测试策略

### 1. 核心模块优先（已完成 ✅）

- [x] `src/utils/` - 工具模块（100%覆盖率）
- [x] `src/models/model_registry.py` - 模型注册表（97%覆盖率）
- [x] `src/data/data_manager.py` - 数据管理器（需要提升）
- [x] `src/data/storage/cache_manager.py` - 缓存管理器（需要提升）
- [x] `src/strategy/screening/financial_filter.py` - 财务筛选器（55%覆盖率）

### 2. 数据获取模块（进行中）

- [x] `src/data/fetcher/tushare_fetcher.py` - Tushare获取器（真实API测试）
- [ ] `src/data/fetcher/base_fetcher.py` - 基础获取器

### 3. 策略模块（进行中）

- [x] `src/strategy/screening/financial_filter.py` - 财务筛选器
- [x] `src/strategy/screening/positive_sample_screener.py` - 正样本筛选器
- [ ] `src/strategy/screening/negative_sample_screener_v2.py` - 负样本筛选器V2
- [ ] `src/strategy/screening/negative_sample_screener.py` - 负样本筛选器

### 4. 模型模块（待完成）

- [x] `src/models/model_registry.py` - 模型注册表
- [ ] `src/models/stock_selection/left_breakout/` - 左侧起爆点模型（需要mock数据）

### 5. 分析模块（待完成）

- [ ] `src/analysis/market_analyzer.py` - 市场分析器（需要真实数据）
- [ ] `src/analysis/stock_health_checker.py` - 股票健康检查（需要真实数据）

## 测试类型

### 单元测试（Mock数据）
- 使用mock避免外部依赖
- 快速执行
- 测试核心逻辑

### 集成测试（真实数据）
- 使用真实API和数据
- 标记为 `@pytest.mark.api` 和 `@pytest.mark.slow`
- 验证端到端功能

## 运行测试

```bash
# 运行所有测试（包括真实API）
pytest

# 只运行单元测试（跳过真实API）
pytest -m "not api"

# 只运行快速测试
pytest -m "not slow"

# 查看覆盖率
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 覆盖率目标分解

| 模块 | 当前 | 目标 | 状态 |
|------|------|------|------|
| `src/utils/` | 100% | 100% | ✅ |
| `src/models/model_registry.py` | 97% | 100% | ✅ |
| `src/data/data_manager.py` | 46% | 85% | 🔄 |
| `src/data/fetcher/tushare_fetcher.py` | 12% | 70% | 🔄 |
| `src/data/storage/cache_manager.py` | 15% | 85% | 🔄 |
| `src/strategy/screening/financial_filter.py` | 55% | 85% | 🔄 |
| `src/strategy/screening/positive_sample_screener.py` | 9% | 70% | 🔄 |
| `src/models/stock_selection/left_breakout/` | 0% | 60% | ⏳ |
| `src/analysis/` | 0% | 50% | ⏳ |

## 下一步行动

1. ✅ 完成核心工具模块测试
2. 🔄 完成数据管理模块测试（进行中）
3. ⏳ 完成策略模块测试
4. ⏳ 完成模型模块测试
5. ⏳ 完成分析模块测试

## 注意事项

- 真实API测试需要Tushare配置
- 某些测试需要较长时间，标记为 `@pytest.mark.slow`
- 保持测试独立，不依赖执行顺序
- 使用fixtures共享测试数据
