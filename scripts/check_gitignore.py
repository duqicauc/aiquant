#!/usr/bin/env python3
"""
检查 .gitignore 是否意外忽略了代码文件
用于确保所有重要的源代码文件都被 Git 跟踪
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def get_git_root() -> Path:
    """获取 Git 仓库根目录"""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True
    )
    return Path(result.stdout.strip())


def check_ignored_files(patterns: List[str]) -> List[Tuple[str, str]]:
    """
    检查指定模式的文件是否被忽略
    
    Returns:
        List of (file_path, ignore_rule) tuples for ignored files
    """
    git_root = get_git_root()
    ignored_files = []
    
    for pattern in patterns:
        # 查找所有匹配的文件
        files = list(git_root.glob(pattern))
        
        for file_path in files:
            if not file_path.is_file():
                continue
                
            # 检查文件是否被忽略
            result = subprocess.run(
                ["git", "check-ignore", "-v", str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                ignore_rule = result.stdout.strip()
                ignored_files.append((str(file_path.relative_to(git_root)), ignore_rule))
    
    return ignored_files


def check_untracked_code_files() -> List[str]:
    """检查未跟踪的代码文件"""
    git_root = get_git_root()
    untracked = []
    
    # 获取未跟踪的文件
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        cwd=git_root
    )
    
    code_extensions = {'.py', '.yaml', '.yml', '.sh', '.ini', '.cfg', '.txt', '.md'}
    
    for line in result.stdout.splitlines():
        if line.startswith('??'):
            file_path = line[3:].strip()
            path = Path(file_path)
            
            # 检查是否是代码文件
            if path.suffix in code_extensions:
                # 排除一些应该被忽略的文件
                if not any([
                    file_path.startswith('data/'),
                    file_path.startswith('logs/'),
                    file_path.endswith('.log'),
                    file_path.endswith('.pid'),
                    'cache' in file_path.lower(),
                    'temp' in file_path.lower(),
                    '__pycache__' in file_path,
                ]):
                    untracked.append(file_path)
    
    return untracked


def main():
    """主函数"""
    print("=" * 60)
    print("检查 .gitignore 配置 - 确保代码文件不被意外忽略")
    print("=" * 60)
    print()
    
    # 检查重要的代码文件模式
    important_patterns = [
        "src/**/*.py",
        "scripts/**/*.py",
        "config/**/*.py",
        "config/**/*.yaml",
        "config/**/*.yml",
        "tests/**/*.py",
        "*.py",
        "*.sh",
        "*.yaml",
        "*.yml",
        "requirements*.txt",
        "pytest.ini",
        "README.md",
        "docs/**/*.md",
    ]
    
    print("1. 检查重要代码文件是否被忽略...")
    ignored_files = check_ignored_files(important_patterns)
    
    if ignored_files:
        print("   ⚠️  警告：发现被忽略的代码文件：")
        for file_path, ignore_rule in ignored_files:
            print(f"      - {file_path}")
            print(f"        被规则忽略: {ignore_rule}")
        print()
    else:
        print("   ✓ 没有发现被忽略的重要代码文件")
        print()
    
    # 检查未跟踪的代码文件
    print("2. 检查未跟踪的代码文件...")
    untracked = check_untracked_code_files()
    
    if untracked:
        print("   ℹ️  发现未跟踪的代码文件（可能需要添加到 Git）：")
        for file_path in sorted(untracked):
            print(f"      - {file_path}")
        print()
        print("   提示：使用 'git add <file>' 添加这些文件")
        print()
    else:
        print("   ✓ 没有发现未跟踪的代码文件")
        print()
    
    # 总结
    print("=" * 60)
    if ignored_files:
        print("⚠️  发现潜在问题：有代码文件被忽略")
        print("   请检查 .gitignore 规则，确保重要代码文件被跟踪")
        return 1
    else:
        print("✓ 检查通过：没有发现代码文件被意外忽略")
        return 0


if __name__ == "__main__":
    sys.exit(main())

