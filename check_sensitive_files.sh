#!/bin/bash
# 检查敏感文件脚本
# 用于在提交到 Git 之前检查是否包含敏感信息

echo "=========================================="
echo "检查敏感文件和大型数据文件"
echo "=========================================="

# 检查 .env 文件
echo ""
echo "1. 检查 .env 文件..."
if [ -f ".env" ]; then
    echo "   ⚠️  警告: 发现 .env 文件，请确保已在 .gitignore 中忽略"
else
    echo "   ✓ 未发现 .env 文件"
fi

# 检查包含 token 的文件
echo ""
echo "2. 检查可能包含 token 的文件..."
grep -r "TUSHARE_TOKEN=" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.txt" --include="*.env" . 2>/dev/null | grep -v "YOUR_TUSHARE_TOKEN" | grep -v ".git" | while read line; do
    echo "   ⚠️  警告: 发现可能的 token: $line"
done

# 检查大型文件（>10MB）
echo ""
echo "3. 检查大型文件（>10MB）..."
find . -type f -size +10M ! -path "./.git/*" ! -path "./venv/*" ! -path "./env/*" 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "   ⚠️  警告: 发现大型文件: $file ($size)"
done

# 检查数据库文件
echo ""
echo "4. 检查数据库文件..."
find . -type f \( -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" \) ! -path "./.git/*" 2>/dev/null | while read file; do
    echo "   ⚠️  警告: 发现数据库文件: $file"
done

# 检查模型文件
echo ""
echo "5. 检查模型文件..."
find . -type f \( -name "*.joblib" -o -name "*.pkl" -o -name "*.h5" \) ! -path "./.git/*" 2>/dev/null | while read file; do
    echo "   ⚠️  警告: 发现模型文件: $file"
done

echo ""
echo "=========================================="
echo "检查完成！"
echo "=========================================="
echo ""
echo "提示："
echo "1. 确保所有敏感信息都在 .env 文件中，且 .env 已在 .gitignore 中"
echo "2. 大型数据文件和模型文件不应提交到 Git"
echo "3. 使用 'git status' 查看将要提交的文件"
echo "4. 使用 'git add .' 前请仔细检查"

