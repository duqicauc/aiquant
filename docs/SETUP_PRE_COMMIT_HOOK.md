# Git Pre-commit Hook 设置指南

## 📋 概述

Git pre-commit hook 可以在提交代码前自动检查测试用例，确保所有修改的代码都有对应的测试。

## 🚀 快速设置

### 方法1: 手动设置（推荐）

```bash
# 1. 确保pre-commit文件存在
ls -la .git/hooks/pre-commit

# 2. 设置执行权限
chmod +x .git/hooks/pre-commit

# 3. 测试是否工作
git add .
git commit -m "test commit"
```

### 方法2: 使用脚本设置

```bash
# 创建设置脚本
cat > setup_pre_commit.sh << 'EOF'
#!/bin/bash
if [ -f .git/hooks/pre-commit ]; then
    chmod +x .git/hooks/pre-commit
    echo "✓ Pre-commit hook 已设置"
else
    echo "✗ Pre-commit hook 文件不存在"
    exit 1
fi
EOF

# 运行设置脚本
chmod +x setup_pre_commit.sh
./setup_pre_commit.sh
```

## 🔍 功能说明

Pre-commit hook 会：

1. **检查修改的文件**: 自动检测staged的Python源代码文件
2. **检查测试用例**: 确保每个修改的文件都有对应的测试文件
3. **检查覆盖率**: 在严格模式下检查测试覆盖率是否>=80%
4. **阻止提交**: 如果检查失败，会阻止代码提交

## ⚙️ 配置选项

### 跳过pre-commit检查

如果确实需要跳过检查（不推荐），可以使用：

```bash
git commit --no-verify -m "your message"
```

### 修改检查规则

编辑 `.git/hooks/pre-commit` 文件，可以：

- 修改覆盖率要求（默认80%）
- 添加其他检查规则
- 自定义错误消息

## 🐛 常见问题

### 1. Permission denied

**错误**: `chmod: Unable to change file mode`

**解决**: 
```bash
# 使用sudo（如果需要）
sudo chmod +x .git/hooks/pre-commit

# 或者检查文件权限
ls -la .git/hooks/pre-commit
```

### 2. Hook不执行

**检查**:
```bash
# 确认文件存在且有执行权限
ls -la .git/hooks/pre-commit

# 确认Git配置
git config core.hooksPath
```

### 3. 测试工具找不到

**错误**: `python scripts/ensure_tests.py: command not found`

**解决**: 确保在项目根目录运行，或使用完整路径：
```bash
# 在pre-commit中，使用绝对路径或确保在项目根目录
cd /path/to/project && python scripts/ensure_tests.py
```

## 📝 自定义Hook

如果需要自定义pre-commit hook，可以编辑 `.git/hooks/pre-commit`：

```bash
#!/bin/bash
# 添加自定义检查
echo "运行自定义检查..."

# 运行测试确保工具
python scripts/ensure_tests.py --strict

# 运行其他检查
# python scripts/check_code_quality.py
# python scripts/check_documentation.py

exit 0
```

## ✅ 验证设置

设置完成后，可以通过以下方式验证：

```bash
# 1. 修改一个Python文件
echo "# test" >> src/utils/logger.py

# 2. 暂存文件
git add src/utils/logger.py

# 3. 尝试提交（应该触发hook）
git commit -m "test commit"

# 如果hook正常工作，会看到检查输出
```

## 🎯 最佳实践

1. **不要跳过检查**: 除非特殊情况，不要使用 `--no-verify`
2. **保持hook更新**: 如果hook逻辑更新，记得同步到所有开发环境
3. **团队协作**: 确保团队成员都设置了pre-commit hook
4. **CI/CD补充**: pre-commit hook是本地检查，CI/CD中也要有相同的检查

## 📚 相关文档

- [测试框架使用指南](TESTING_FRAMEWORK_GUIDE.md)
- [测试确保工具说明](../scripts/ensure_tests.py)

