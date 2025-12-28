# SSL权限问题修复指南

## 问题描述

在macOS上运行left_breakout相关脚本时，可能遇到以下错误：

```
PermissionError: [Errno 1] Operation not permitted
```

这是因为macOS的安全限制阻止了Python访问SSL证书文件。

## 解决方案

### 方案1：授予终端完全磁盘访问权限（推荐）

1. 打开"系统设置" > "隐私与安全性" > "完全磁盘访问权限"
2. 点击"+"添加终端应用（Terminal.app 或 iTerm.app）
3. 重启终端后重新运行脚本

### 方案2：使用系统Python（如果可用）

```bash
# 检查系统Python
/usr/bin/python3 --version

# 使用系统Python运行
/usr/bin/python3 scripts/prepare_left_breakout_samples.py
```

### 方案3：重新安装certifi和requests

```bash
pip install --upgrade --force-reinstall certifi requests
```

### 方案4：临时禁用SSL验证（仅开发环境，不推荐）

```bash
export PYTHONHTTPSVERIFY=0
python scripts/prepare_left_breakout_samples.py
```

## 验证修复

运行以下命令验证SSL是否正常工作：

```bash
python -c "import requests; print('✓ SSL正常')"
```

## 缓存优先

即使遇到SSL问题，脚本也会优先使用缓存数据：
- 缓存位置：`data/cache/quant_data.db`
- 当前缓存：5110只股票，1999-2025年数据
- 如果缓存充足，可能不需要访问网络

