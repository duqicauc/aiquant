#!/usr/bin/env python3
"""
测试用例生成模板工具

为新的源代码文件自动生成测试用例模板，确保每个代码修改都有配套测试。

使用方法:
    python scripts/generate_test_template.py src/path/to/module.py
    python scripts/generate_test_template.py src/data/data_manager.py --output tests/data/test_data_manager.py
"""

import sys
import os
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTemplateGenerator:
    """测试用例模板生成器"""
    
    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.source_code = source_file.read_text(encoding='utf-8')
        self.tree = ast.parse(self.source_code)
        self.module_name = self._extract_module_name()
        self.classes = []
        self.functions = []
        self._analyze_code()
    
    def _extract_module_name(self) -> str:
        """提取模块名称"""
        # 从文件路径提取模块名
        # src/data/data_manager.py -> data.data_manager
        parts = self.source_file.parts
        if 'src' in parts:
            idx = parts.index('src')
            module_parts = parts[idx+1:-1] + [self.source_file.stem]
            return '.'.join(module_parts)
        return self.source_file.stem
    
    def _analyze_code(self):
        """分析代码结构"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                self.classes.append({
                    'name': node.name,
                    'methods': methods,
                    'docstring': ast.get_docstring(node)
                })
            elif isinstance(node, ast.FunctionDef) and not any(
                isinstance(parent, ast.ClassDef) for parent in ast.walk(self.tree)
                if hasattr(parent, 'body') and node in getattr(parent, 'body', [])
            ):
                # 顶级函数（不在类中）
                self.functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node)
                })
    
    def generate_test_file(self, output_path: Path = None) -> str:
        """生成测试文件内容"""
        if output_path is None:
            # 自动生成测试文件路径
            # src/data/data_manager.py -> tests/data/test_data_manager.py
            parts = self.source_file.parts
            if 'src' in parts:
                idx = parts.index('src')
                test_parts = ['tests'] + list(parts[idx+1:-1]) + [f'test_{parts[-1]}']
                output_path = project_root / Path(*test_parts)
            else:
                output_path = project_root / 'tests' / f'test_{self.source_file.name}'
        
        # 生成测试文件内容
        lines = []
        lines.append('"""')
        lines.append(f'测试模块: {self.module_name}')
        lines.append('')
        lines.append('自动生成的测试模板，请补充完整的测试用例')
        lines.append('"""')
        lines.append('')
        lines.append('import pytest')
        lines.append('import pandas as pd')
        lines.append('import numpy as np')
        lines.append('from unittest.mock import Mock, patch, MagicMock')
        lines.append('')
        lines.append(f'from {self.module_name} import (')
        
        # 导入类和函数
        imports = []
        for cls in self.classes:
            imports.append(f'    {cls["name"]},')
        for func in self.functions:
            imports.append(f'    {func["name"]},')
        
        if imports:
            lines.extend(imports)
            lines.append(')')
        else:
            lines.append('    # 请添加需要测试的类和函数')
            lines.append(')')
        
        lines.append('')
        lines.append('')
        
        # 为每个类生成测试类
        for cls in self.classes:
            lines.append(f'class Test{cls["name"]}:')
            lines.append(f'    """{cls["name"]}测试类"""')
            lines.append('')
            
            # 初始化测试
            lines.append('    def test_init(self):')
            lines.append(f'        """测试{cls["name"]}初始化"""')
            lines.append('        # TODO: 实现初始化测试')
            lines.append('        pass')
            lines.append('')
            
            # 为每个方法生成测试
            for method in cls['methods']:
                if not method.startswith('_') or method == '__init__':
                    lines.append(f'    def test_{method}(self):')
                    lines.append(f'        """测试{method}方法"""')
                    lines.append('        # TODO: 实现测试')
                    lines.append('        pass')
                    lines.append('')
        
        # 为每个顶级函数生成测试
        for func in self.functions:
            if not func['name'].startswith('_'):
                lines.append(f'@pytest.mark.unit')
                lines.append(f'def test_{func["name"]}():')
                lines.append(f'    """测试{func["name"]}函数"""')
                lines.append('    # TODO: 实现测试')
                lines.append('    pass')
                lines.append('')
        
        return '\n'.join(lines)
    
    def save_test_file(self, output_path: Path = None):
        """保存测试文件"""
        if output_path is None:
            # 自动生成路径
            parts = self.source_file.parts
            if 'src' in parts:
                idx = parts.index('src')
                test_parts = ['tests'] + list(parts[idx+1:-1]) + [f'test_{parts[-1]}']
                output_path = project_root / Path(*test_parts)
            else:
                output_path = project_root / 'tests' / f'test_{self.source_file.name}'
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果文件已存在，询问是否覆盖
        if output_path.exists():
            response = input(f'文件 {output_path} 已存在，是否覆盖？(y/N): ')
            if response.lower() != 'y':
                print(f'已取消，文件未修改: {output_path}')
                return
        
        # 生成并保存
        content = self.generate_test_file(output_path)
        output_path.write_text(content, encoding='utf-8')
        print(f'✓ 测试模板已生成: {output_path}')
        print(f'  请补充完整的测试用例')


def main():
    parser = argparse.ArgumentParser(
        description='为源代码文件生成测试用例模板',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为data_manager.py生成测试模板
  python scripts/generate_test_template.py src/data/data_manager.py
  
  # 指定输出路径
  python scripts/generate_test_template.py src/data/data_manager.py --output tests/data/test_data_manager.py
  
  # 为整个目录生成测试模板
  python scripts/generate_test_template.py src/utils/ --recursive
        """
    )
    parser.add_argument(
        'source',
        type=Path,
        help='源代码文件路径（相对于项目根目录）'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='测试文件输出路径（可选，默认自动生成）'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归处理目录中的所有Python文件'
    )
    
    args = parser.parse_args()
    
    source_path = project_root / args.source if not args.source.is_absolute() else args.source
    
    if not source_path.exists():
        print(f'错误: 文件不存在: {source_path}')
        sys.exit(1)
    
    if source_path.is_file():
        # 处理单个文件
        if not source_path.suffix == '.py':
            print(f'错误: 不是Python文件: {source_path}')
            sys.exit(1)
        
        generator = TestTemplateGenerator(source_path)
        generator.save_test_file(args.output)
    
    elif source_path.is_dir() and args.recursive:
        # 递归处理目录
        py_files = list(source_path.rglob('*.py'))
        # 排除__init__.py和测试文件
        py_files = [f for f in py_files if not f.name.startswith('__') and 'test_' not in f.name]
        
        print(f'找到 {len(py_files)} 个Python文件')
        for py_file in py_files:
            try:
                generator = TestTemplateGenerator(py_file)
                generator.save_test_file()
            except Exception as e:
                print(f'警告: 处理 {py_file} 时出错: {e}')
    
    else:
        print(f'错误: 请使用 --recursive 选项处理目录')
        sys.exit(1)


if __name__ == '__main__':
    main()

