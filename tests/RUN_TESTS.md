# 测试运行指南

## 快速开始

### 运行所有测试

```bash
# 运行所有测试（包括真实API测试）
pytest

# 只运行单元测试（跳过真实API，更快）
pytest -m "not api"

# 只运行快速测试（跳过慢速测试）
pytest -m "not slow"

# 运行特定模块
pytest tests/utils/
pytest tests/data/
pytest tests/strategy/
pytest tests/models/
```

### 查看覆盖率

```bash
# 生成HTML覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term-missing

# 打开覆盖率报告
open htmlcov/index.html

# 只查看覆盖率，不运行测试
pytest --cov=src --cov-report=term-missing --collect-only
```

## 测试分类

### 单元测试（快速，使用Mock）

```bash
# 运行所有单元测试
pytest -m unit

# 运行特定模块的单元测试
pytest -m unit tests/utils/
pytest -m unit tests/data/test_data_manager.py
```

### 集成测试（需要真实数据）

```bash
# 运行集成测试（需要Tushare配置）
pytest -m integration

# 运行API测试（需要网络和Tushare配置）
pytest -m api

# 运行慢速测试
pytest -m slow
```

## 测试标记说明

- `@pytest.mark.unit` - 单元测试，使用Mock，快速执行
- `@pytest.mark.integration` - 集成测试，测试完整流程
- `@pytest.mark.api` - 需要真实API调用（Tushare）
- `@pytest.mark.slow` - 慢速测试（需要网络或数据库操作）
- `@pytest.mark.mock` - 使用Mock的测试

## 真实API测试配置

如果Tushare配置无效，相关测试会自动跳过：

```python
try:
    from config.data_source import data_source_config
    data_source_config.validate_tushare()
except Exception as e:
    pytest.skip(f"Tushare配置无效: {e}")
```

## 测试输出

### 详细输出

```bash
# 显示详细输出
pytest -v

# 显示最详细输出
pytest -vv

# 显示测试进度
pytest --tb=short -v
```

### 只显示失败

```bash
# 只显示失败的测试
pytest --tb=short -x

# 在第一个失败时停止
pytest -x

# 显示失败测试的局部变量
pytest -l
```

## 覆盖率目标

当前覆盖率要求：**90%**

查看各模块覆盖率：

```bash
pytest --cov=src --cov-report=term-missing
```

## 常见问题

### 1. 测试失败：Tushare配置无效

**解决**：配置Tushare Token，或跳过API测试：
```bash
pytest -m "not api"
```

### 2. 测试很慢

**解决**：只运行单元测试：
```bash
pytest -m "not slow"
```

### 3. 覆盖率不达标

**解决**：查看覆盖率报告，找出未覆盖的代码：
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### 4. 数据库锁定错误

**解决**：确保测试使用临时数据库，每个测试独立：
```python
@pytest.fixture
def cache_manager(self, temp_dir):
    db_path = temp_dir / 'test_cache.db'
    if db_path.exists():
        db_path.unlink()
    return CacheManager(db_path=str(db_path))
```

## CI/CD集成

测试已配置GitHub Actions（`.github/workflows/tests.yml`），在以下情况自动运行：

- Push到main或develop分支
- 创建Pull Request

CI配置会：
- 运行所有测试
- 检查覆盖率（要求≥90%）
- 生成覆盖率报告

## 最佳实践

1. **运行测试前提交代码**
   ```bash
   git add .
   git commit -m "Add tests"
   pytest  # 确保测试通过
   ```

2. **本地开发时只运行快速测试**
   ```bash
   pytest -m "not api and not slow"
   ```

3. **提交前运行完整测试**
   ```bash
   pytest  # 包括所有测试
   ```

4. **定期检查覆盖率**
   ```bash
   pytest --cov=src --cov-report=html
   ```

## 测试文件组织

```
tests/
├── conftest.py              # 公共fixtures
├── utils/                   # 工具模块测试
├── data/                    # 数据模块测试
├── strategy/                # 策略模块测试
├── models/                  # 模型模块测试
├── integration/             # 集成测试
└── README.md               # 测试文档
```

## 相关文档

- [测试指南](../docs/TESTING_GUIDE.md)
- [覆盖率计划](COVERAGE_PLAN.md)
- [测试总结](SUMMARY.md)

