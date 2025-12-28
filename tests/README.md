# 测试文档

## 测试框架

本项目使用 `pytest` 作为测试框架，配合 `pytest-cov` 进行代码覆盖率统计。

## 目录结构

```
tests/
├── conftest.py              # pytest配置文件，提供fixtures
├── utils/                   # 工具模块测试
│   ├── test_rate_limiter.py
│   └── test_date_utils.py
├── data/                    # 数据模块测试
│   └── test_data_manager.py
├── strategy/                # 策略模块测试
│   └── test_financial_filter.py
├── models/                  # 模型模块测试
│   └── test_model_registry.py
└── README.md               # 本文件
```

## 运行测试

### 运行所有测试

```bash
# 运行所有测试
pytest

# 运行并显示覆盖率
pytest --cov=src --cov-report=html

# 运行特定模块的测试
pytest tests/utils/

# 运行特定测试文件
pytest tests/utils/test_rate_limiter.py

# 运行特定测试函数
pytest tests/utils/test_rate_limiter.py::TestRateLimiter::test_init
```

### 运行特定类型的测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 跳过慢速测试
pytest -m "not slow"

# 跳过需要API的测试
pytest -m "not api"
```

### 测试覆盖率

```bash
# 生成HTML覆盖率报告
pytest --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html

# 要求最低覆盖率（当前设置为60%）
pytest --cov=src --cov-fail-under=60
```

## 测试标记

测试可以使用以下标记：

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.slow` - 慢速测试（需要网络或数据库）
- `@pytest.mark.api` - 需要API调用的测试
- `@pytest.mark.mock` - 使用mock的测试

使用示例：

```python
@pytest.mark.unit
def test_simple_function():
    assert 1 + 1 == 2

@pytest.mark.slow
@pytest.mark.api
def test_api_call():
    # 需要实际API调用的测试
    pass
```

## Fixtures

在 `conftest.py` 中定义了以下常用fixtures：

- `project_path` - 项目根目录路径
- `test_data_dir` - 测试数据目录
- `temp_dir` - 临时文件目录
- `mock_data_manager` - 模拟的DataManager
- `sample_stock_data` - 示例股票数据
- `sample_stocks_df` - 示例股票列表DataFrame
- `mock_tushare_fetcher` - 模拟的TushareFetcher

使用示例：

```python
def test_something(mock_data_manager, sample_stocks_df):
    result = mock_data_manager.get_stock_list()
    assert len(result) > 0
```

## 编写测试

### 测试命名规范

- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试函数：`test_*`

### 测试结构

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

## 测试覆盖率目标

当前项目的测试覆盖率目标：

- **总体覆盖率**: ≥ 60%
- **核心模块覆盖率**: ≥ 80%
  - `src/utils/`: ≥ 80%
  - `src/data/`: ≥ 70%
  - `src/strategy/`: ≥ 70%
  - `src/models/`: ≥ 60%

## 持续集成

测试应该在以下情况运行：

1. **提交代码前**: 运行 `pytest` 确保所有测试通过
2. **Pull Request**: CI自动运行测试
3. **代码合并前**: 确保覆盖率不低于目标值

## 常见问题

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

## 下一步

- [ ] 增加更多单元测试
- [ ] 添加集成测试
- [ ] 提高测试覆盖率
- [ ] 添加性能测试
- [ ] 添加端到端测试

