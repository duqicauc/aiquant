# 测试指南

## 概述

本项目已建立完整的测试框架，使用 `pytest` 作为测试运行器，`pytest-cov` 进行代码覆盖率统计。

## 测试框架结构

```
tests/
├── conftest.py                    # pytest配置和fixtures
├── utils/                         # 工具模块测试
│   ├── test_rate_limiter.py      # 限流器测试
│   └── test_date_utils.py        # 日期工具测试
├── data/                          # 数据模块测试
│   └── test_data_manager.py      # 数据管理器测试
├── strategy/                      # 策略模块测试
├── models/                         # 模型模块测试
│   └── test_model_registry.py   # 模型注册表测试
├── integration/                   # 集成测试
│   └── test_data_flow.py         # 数据流测试
├── README.md                      # 测试文档
└── run_tests.sh                   # 测试运行脚本
```

## 快速开始

### 安装测试依赖

```bash
pip install pytest pytest-cov
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行并生成覆盖率报告
pytest --cov=src --cov-report=html

# 使用测试脚本
bash tests/run_tests.sh

# 运行特定模块
bash tests/run_tests.sh --path tests/utils/

# 运行并显示覆盖率
bash tests/run_tests.sh --coverage
```

## 已实现的测试

### 1. 工具模块测试 (`tests/utils/`)

#### `test_rate_limiter.py`
- ✅ `RateLimiter` 基本功能测试
- ✅ `TushareRateLimiter` 积分限流测试
- ✅ 全局限流器测试
- ✅ 重试装饰器测试
- ✅ 安全API调用装饰器测试

#### `test_date_utils.py`
- ✅ 日期格式化测试
- ✅ 交易日判断测试
- ✅ 交易日列表获取测试
- ✅ 最近日期获取测试

### 2. 数据模块测试 (`tests/data/`)

#### `test_data_manager.py`
- ✅ DataManager初始化测试
- ✅ 数据源选择测试
- ✅ 股票列表获取测试
- ✅ 日线数据获取测试

### 3. 策略模块测试 (`tests/strategy/`)
- ✅ 不同列名处理测试

### 4. 模型模块测试 (`tests/models/`)

#### `test_model_registry.py`
- ✅ ModelConfig配置测试
- ✅ 模型注册测试
- ✅ 模型路径获取测试
- ✅ 元数据保存和加载测试
- ✅ 目录自动创建测试

### 5. 集成测试 (`tests/integration/`)

#### `test_data_flow.py`
- ✅ 数据流完整流程测试
- ✅ 从数据获取到预测的流程测试

## 测试覆盖率

当前测试覆盖率目标：

- **总体覆盖率**: ≥ 60%
- **核心模块覆盖率**:
  - `src/utils/`: ≥ 80% ✅
  - `src/data/`: ≥ 70% ✅
  - `src/strategy/`: ≥ 70% ✅
  - `src/models/`: ≥ 60% ✅

查看覆盖率报告：

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 测试标记

使用pytest标记来分类测试：

```python
@pytest.mark.unit          # 单元测试
@pytest.mark.integration   # 集成测试
@pytest.mark.slow          # 慢速测试
@pytest.mark.api           # 需要API的测试
@pytest.mark.mock          # 使用mock的测试
```

运行特定类型的测试：

```bash
# 只运行单元测试
pytest -m unit

# 跳过慢速测试
pytest -m "not slow"

# 只运行集成测试
pytest -m integration
```

## Fixtures

在 `conftest.py` 中定义的常用fixtures：

- `project_path` - 项目根目录
- `test_data_dir` - 测试数据目录
- `temp_dir` - 临时文件目录
- `mock_data_manager` - 模拟DataManager
- `sample_stock_data` - 示例股票数据
- `sample_stocks_df` - 示例股票DataFrame
- `mock_tushare_fetcher` - 模拟TushareFetcher

使用示例：

```python
def test_something(mock_data_manager, sample_stocks_df):
    result = mock_data_manager.get_stock_list()
    assert len(result) > 0
```

## 编写新测试

### 测试文件命名

- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试函数：`test_*`

### 测试结构示例

```python
import pytest
from src.module import SomeClass

class TestSomeClass:
    """SomeClass测试类"""
    
    def test_init(self):
        """测试初始化"""
        obj = SomeClass()
        assert obj is not None
    
    def test_method(self):
        """测试方法"""
        obj = SomeClass()
        result = obj.method()
        assert result == expected_value
```

### 使用Mock

```python
from unittest.mock import Mock, patch

def test_with_mock(mock_data_manager):
    with patch('src.module.external_function') as mock_func:
        mock_func.return_value = 'mocked_value'
        result = some_function()
        assert result == 'mocked_value'
```

## CI/CD集成

项目已配置GitHub Actions工作流（`.github/workflows/tests.yml`），在以下情况自动运行测试：

- Push到main或develop分支
- 创建Pull Request

## 最佳实践

1. **测试隔离**: 每个测试应该独立，不依赖其他测试
2. **使用Mock**: 避免真实API调用和数据库操作
3. **测试命名**: 使用描述性的测试名称
4. **测试文档**: 为每个测试添加docstring说明
5. **覆盖率**: 保持核心模块的高覆盖率
6. **快速反馈**: 单元测试应该快速运行

## 下一步计划

- [ ] 增加更多单元测试覆盖边界情况
- [ ] 添加性能测试
- [ ] 添加端到端测试
- [ ] 提高测试覆盖率到80%+
- [ ] 添加测试数据管理工具
- [ ] 添加测试报告生成

## 相关文档

- [测试README](tests/README.md) - 详细的测试文档
- [pytest文档](https://docs.pytest.org/) - pytest官方文档
- [pytest-cov文档](https://pytest-cov.readthedocs.io/) - 覆盖率工具文档

