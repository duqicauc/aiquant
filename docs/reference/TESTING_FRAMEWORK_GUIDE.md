# 测试框架使用指南

## 📋 概述

本项目已建立完整的测试框架，确保每次代码修改都有配套的测试用例。本文档介绍如何使用测试框架、编写测试用例以及维护测试覆盖率。

## 🎯 测试框架目标

1. **确保代码质量**: 每次代码修改都有对应的测试用例
2. **提高覆盖率**: 目标覆盖率 ≥ 85%
3. **快速反馈**: 测试运行快速，提供即时反馈
4. **易于维护**: 测试代码清晰、可维护

## 🏗️ 测试框架结构

```
tests/
├── conftest.py              # pytest配置和fixtures
├── utils/                   # 工具模块测试
├── data/                    # 数据模块测试
├── strategy/                # 策略模块测试
├── models/                  # 模型模块测试
├── analysis/                # 分析模块测试
├── backtest/                # 回测模块测试
├── visualization/          # 可视化模块测试
└── integration/             # 集成测试
```

## 🚀 快速开始

### 1. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块的测试
pytest tests/utils/

# 运行特定测试文件
pytest tests/utils/test_rate_limiter.py

# 运行特定测试函数
pytest tests/utils/test_rate_limiter.py::TestRateLimiter::test_init

# 运行并显示覆盖率
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### 2. 使用测试脚本

```bash
# 使用测试运行脚本
bash tests/run_tests.sh

# 运行并生成覆盖率报告
bash tests/run_tests.sh --coverage

# 只运行单元测试
bash tests/run_tests.sh --unit

# 只运行集成测试
bash tests/run_tests.sh --integration
```

## 📝 编写测试用例

### 1. 为新代码生成测试模板

```bash
# 为单个文件生成测试模板
python scripts/generate_test_template.py src/data/data_manager.py

# 为整个目录生成测试模板
python scripts/generate_test_template.py src/utils/ --recursive
```

### 2. 测试用例结构

```python
"""测试模块: data.data_manager"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data.data_manager import DataManager

class TestDataManager:
    """DataManager测试类"""

    @pytest.mark.unit
    def test_init(self, mock_data_manager):
        """测试初始化"""
        dm = DataManager(source='tushare')
        assert dm.source == 'tushare'

    @pytest.mark.unit
    def test_get_stock_list(self, mock_data_manager):
        """测试获取股票列表"""
        result = mock_data_manager.get_stock_list()
        assert len(result) > 0
        assert 'ts_code' in result.columns
```

### 3. 使用测试标记

```python
@pytest.mark.unit          # 单元测试（快速，使用mock）
@pytest.mark.integration   # 集成测试（测试完整流程）
@pytest.mark.slow          # 慢速测试（需要网络或数据库）
@pytest.mark.api           # 需要API调用的测试
@pytest.mark.mock          # 使用mock的测试
@pytest.mark.data          # 需要数据文件的测试
@pytest.mark.model         # 需要模型文件的测试
@pytest.mark.real          # 真实数据测试
@pytest.mark.smoke         # 冒烟测试
@pytest.mark.regression    # 回归测试
```

### 4. 使用Fixtures

```python
# 使用预定义的fixtures
def test_something(mock_data_manager, sample_stock_data):
    result = mock_data_manager.process(sample_stock_data)
    assert result is not None

# 可用的fixtures（在conftest.py中定义）
# - project_path: 项目根目录路径
# - test_data_dir: 测试数据目录
# - temp_dir: 临时文件目录
# - mock_data_manager: 模拟的DataManager
# - sample_stock_data: 示例股票数据
# - sample_stocks_df: 示例股票列表DataFrame
# - mock_tushare_fetcher: 模拟的TushareFetcher
# - mock_config: 模拟配置对象
# - sample_model_data: 示例模型数据
# - mock_model: 模拟模型对象
# - sample_prediction_result: 示例预测结果
# - clean_temp_dir: 清理临时目录
# - mock_xgboost_model: 模拟XGBoost模型
# - sample_time_series_data: 示例时间序列数据
# - mock_cache_db: 模拟缓存数据库路径
# - sample_technical_indicators: 示例技术指标数据
```

## 🔍 检查测试覆盖率

### 1. 检查整体覆盖率

```bash
# 运行覆盖率检查
python scripts/check_test_coverage.py

# 生成详细报告
python scripts/check_test_coverage.py --report

# 检查特定文件
python scripts/check_test_coverage.py --file src/data/data_manager.py
```

### 2. 检查修改的文件

```bash
# 检查修改的文件是否有测试
python scripts/check_test_coverage.py --modified
```

### 3. 查看覆盖率报告

```bash
# 生成HTML报告
pytest --cov=src --cov-report=html

# 打开报告
open htmlcov/index.html
```

## ✅ 确保测试配套

### 1. 提交前检查

```bash
# 检查修改的文件是否有测试
python scripts/ensure_tests.py

# 严格模式（要求覆盖率>=80%）
python scripts/ensure_tests.py --strict

# 自动生成缺失的测试模板
python scripts/ensure_tests.py --generate
```

### 2. 工作流程

1. **修改代码** → 在 `src/` 目录下修改或添加代码
2. **生成测试模板** → 运行 `python scripts/ensure_tests.py --generate`
3. **编写测试** → 补充完整的测试用例
4. **运行测试** → 运行 `pytest` 确保测试通过
5. **检查覆盖率** → 运行 `python scripts/check_test_coverage.py --modified`
6. **提交代码** → 确保所有测试通过且覆盖率达标

## 📊 测试覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| `src/utils/` | ≥ 80% | - |
| `src/data/` | ≥ 70% | - |
| `src/strategy/` | ≥ 70% | - |
| `src/models/` | ≥ 60% | - |
| `src/analysis/` | ≥ 60% | - |
| **总体** | **≥ 85%** | **9.90%** |

## 🎨 测试最佳实践

### 1. 测试命名

- 测试文件: `test_*.py`
- 测试类: `Test*`
- 测试函数: `test_*`

### 2. 测试结构

```python
class TestSomeClass:
    """SomeClass测试类"""

    def test_init(self):
        """测试初始化"""
        # Arrange: 准备测试数据
        # Act: 执行测试
        # Assert: 验证结果

    def test_method_success(self):
        """测试方法成功情况"""
        pass

    def test_method_failure(self):
        """测试方法失败情况"""
        pass

    def test_edge_cases(self):
        """测试边界情况"""
        pass
```

### 3. 使用Mock

```python
from unittest.mock import Mock, patch, MagicMock

def test_with_mock(mock_data_manager):
    # 使用fixture中的mock
    result = mock_data_manager.get_stock_list()
    assert result is not None

def test_with_patch():
    # 使用patch装饰器
    with patch('src.module.external_function') as mock_func:
        mock_func.return_value = 'mocked_value'
        result = some_function()
        assert result == 'mocked_value'
```

### 4. 测试数据

- 使用fixtures提供测试数据
- 避免依赖真实API或数据库
- 使用mock模拟外部依赖

### 5. 测试独立性

- 每个测试应该独立运行
- 不依赖其他测试的执行顺序
- 使用fixtures清理测试环境

## 🛠️ 工具和脚本

### 1. 测试模板生成器

```bash
python scripts/generate_test_template.py <source_file>
```

功能：
- 自动分析源代码结构
- 生成测试用例模板
- 包含类和函数的测试框架

### 2. 覆盖率检查工具

```bash
python scripts/check_test_coverage.py [options]
```

功能：
- 检查整体覆盖率
- 检查特定文件覆盖率
- 检查修改文件的覆盖率
- 生成详细报告

### 3. 测试确保工具

```bash
python scripts/ensure_tests.py [options]
```

功能：
- 检查修改的文件是否有测试
- 自动生成缺失的测试模板
- 严格模式检查覆盖率

## 🔧 配置说明

### pytest.ini

主要配置：
- 测试发现规则
- 输出选项
- 测试标记
- 日志配置

### conftest.py

提供：
- 全局fixtures
- 测试环境设置
- Mock配置

## 📚 相关文档

- [测试README](../tests/README.md)
- [覆盖率状态](../tests/COVERAGE_STATUS.md)
- [测试指南](../docs/TESTING_GUIDE.md)

## ❓ 常见问题

### 1. 测试需要真实API调用怎么办？

使用 `@pytest.mark.api` 标记，并在CI中配置API密钥，或使用mock。

### 2. 测试需要数据库怎么办？

使用 `@pytest.mark.slow` 标记，或使用内存数据库（如SQLite）。

### 3. 如何跳过某些测试？

```python
@pytest.mark.skip(reason="功能未实现")
def test_unimplemented():
    pass

@pytest.mark.skipif(condition, reason="需要特定条件")
def test_conditional():
    pass
```

### 4. 测试运行太慢怎么办？

```bash
# 只运行单元测试（快速）
pytest -m unit

# 跳过慢速测试
pytest -m "not slow"
```

## 🎯 下一步

- [ ] 提高测试覆盖率到85%
- [ ] 为所有核心模块添加测试
- [ ] 建立CI/CD测试流程
- [ ] 添加性能测试
- [ ] 添加端到端测试
