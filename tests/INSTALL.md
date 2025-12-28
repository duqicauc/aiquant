# 测试环境安装指南

## 问题：pytest 命令未找到

如果遇到 `pytest: command not found` 错误，请按照以下步骤安装测试依赖。

## 安装方法

### 方法1：使用 pip（推荐）

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 或者从 requirements.txt 安装（已包含 pytest）
pip install -r requirements.txt
```

### 方法2：使用 conda（如果使用 conda 环境）

```bash
conda install pytest pytest-cov -y
```

### 方法3：使用 python -m pytest

如果 pytest 已安装但不在 PATH 中，可以使用：

```bash
# 使用 python -m pytest 代替 pytest
python -m pytest tests/utils/test_rate_limiter.py -v
```

## 验证安装

安装完成后，验证 pytest 是否可用：

```bash
# 方法1：直接运行 pytest
pytest --version

# 方法2：使用 python -m
python -m pytest --version

# 方法3：检查是否在 PATH 中
which pytest
```

## 运行测试

安装成功后，可以运行测试：

```bash
# 运行所有测试
pytest

# 或使用 python -m
python -m pytest

# 运行特定测试文件
pytest tests/utils/test_rate_limiter.py -v

# 使用测试脚本
bash tests/run_tests.sh
```

## 常见问题

### 1. PermissionError

如果遇到权限错误，尝试：

```bash
# 使用用户安装
pip install --user pytest pytest-cov

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
pip install pytest pytest-cov
```

### 2. 网络问题

如果无法从 PyPI 下载，可以使用国内镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-cov
```

### 3. conda 环境问题

如果使用 conda，确保激活了正确的环境：

```bash
conda activate your_env_name
pip install pytest pytest-cov
```

## 检查 requirements.txt

项目的 `requirements.txt` 已经包含了测试依赖：

```
pytest>=7.4.0
pytest-cov>=4.1.0
```

如果已经安装了 requirements.txt，pytest 应该已经可用。

