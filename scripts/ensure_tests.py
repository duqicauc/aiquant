#!/usr/bin/env python3
"""
确保代码修改配套测试用例的检查工具

在提交代码前运行此脚本，检查修改的代码是否有对应的测试用例。

使用方法:
    python scripts/ensure_tests.py              # 检查所有修改的文件
    python scripts/ensure_tests.py --strict     # 严格模式（要求测试覆盖率>=80%）
    python scripts/ensure_tests.py --generate   # 自动生成缺失的测试模板
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEnsurer:
    """测试用例确保工具"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / 'src'
        self.tests_dir = project_root / 'tests'
    
    def get_modified_files(self) -> List[Path]:
        """获取修改的源代码文件"""
        try:
            # 获取staged和unstaged的文件
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--cached'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            staged_files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            unstaged_files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            all_files = set(staged_files + unstaged_files)
            
            # 只保留src目录下的Python文件
            modified = []
            for file_str in all_files:
                if file_str.startswith('src/') and file_str.endswith('.py'):
                    file_path = self.project_root / file_str
                    if file_path.exists():
                        modified.append(file_path)
            
            return modified
        except Exception as e:
            print(f'警告: 无法获取修改的文件: {e}')
            return []
    
    def find_test_file(self, source_file: Path) -> Path:
        """查找对应的测试文件"""
        # src/data/data_manager.py -> tests/data/test_data_manager.py
        parts = source_file.parts
        if 'src' in parts:
            idx = parts.index('src')
            test_parts = ['tests'] + list(parts[idx+1:-1]) + [f'test_{parts[-1]}']
            test_file = self.project_root / Path(*test_parts)
            return test_file
        return None
    
    def check_test_exists(self, source_file: Path) -> Tuple[bool, Path]:
        """检查测试文件是否存在"""
        test_file = self.find_test_file(source_file)
        if test_file and test_file.exists():
            return True, test_file
        return False, test_file
    
    def check_all_modified(self, strict: bool = False) -> Dict[str, List[Path]]:
        """检查所有修改的文件"""
        modified_files = self.get_modified_files()
        
        if not modified_files:
            print("没有找到修改的源代码文件")
            return {'missing': [], 'exists': []}
        
        print(f"找到 {len(modified_files)} 个修改的源代码文件:\n")
        
        missing_tests = []
        existing_tests = []
        
        for source_file in modified_files:
            has_test, test_file = self.check_test_exists(source_file)
            
            status = "✓" if has_test else "✗"
            print(f"{status} {source_file.relative_to(self.project_root)}")
            
            if has_test:
                existing_tests.append(source_file)
                print(f"   测试文件: {test_file.relative_to(self.project_root)}")
            else:
                missing_tests.append(source_file)
                print(f"   缺少测试文件: {test_file.relative_to(self.project_root) if test_file else 'N/A'}")
        
        return {
            'missing': missing_tests,
            'exists': existing_tests
        }
    
    def generate_missing_tests(self, files: List[Path]):
        """为缺失测试的文件生成测试模板"""
        from scripts.generate_test_template import TestTemplateGenerator
        
        for source_file in files:
            try:
                generator = TestTemplateGenerator(source_file)
                generator.save_test_file()
            except Exception as e:
                print(f"警告: 无法为 {source_file} 生成测试模板: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='确保代码修改配套测试用例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
此工具会检查修改的源代码文件是否有对应的测试用例。

在提交代码前运行此脚本，确保所有修改都有测试覆盖。

示例:
  # 检查修改的文件
  python scripts/ensure_tests.py
  
  # 严格模式（要求覆盖率>=80%）
  python scripts/ensure_tests.py --strict
  
  # 自动生成缺失的测试模板
  python scripts/ensure_tests.py --generate
        """
    )
    parser.add_argument(
        '--strict', '-s',
        action='store_true',
        help='严格模式：要求测试覆盖率>=80%'
    )
    parser.add_argument(
        '--generate', '-g',
        action='store_true',
        help='自动生成缺失的测试模板'
    )
    
    args = parser.parse_args()
    
    ensurer = TestEnsurer(project_root)
    
    print("=" * 80)
    print("检查代码修改是否配套测试用例")
    print("=" * 80)
    print()
    
    results = ensurer.check_all_modified(strict=args.strict)
    
    print()
    print("=" * 80)
    
    if results['missing']:
        print(f"\n警告: {len(results['missing'])} 个文件缺少测试用例:")
        for file in results['missing']:
            print(f"  - {file.relative_to(project_root)}")
        
        if args.generate:
            print("\n正在生成测试模板...")
            ensurer.generate_missing_tests(results['missing'])
            print("\n✓ 测试模板已生成，请补充完整的测试用例")
        else:
            print("\n提示: 使用 --generate 选项自动生成测试模板")
            print("      python scripts/ensure_tests.py --generate")
        
        if args.strict:
            print("\n✗ 严格模式：部分文件缺少测试用例，请添加测试后再提交")
            sys.exit(1)
    else:
        print("\n✓ 所有修改的文件都有对应的测试用例")
        
        if args.strict:
            # 在严格模式下，还需要检查覆盖率
            print("\n检查测试覆盖率...")
            try:
                from scripts.check_test_coverage import CoverageChecker
                checker = CoverageChecker(project_root)
                results = checker.check_modified_files()
                
                uncovered = [r for r in results if r['coverage'] < 80]
                if uncovered:
                    print(f"\n警告: {len(uncovered)} 个文件测试覆盖率低于80%")
                    for r in uncovered:
                        print(f"  - {r['file']}: {r['coverage']:.2f}%")
                    print("\n✗ 严格模式：部分文件测试覆盖率不足，请补充测试")
                    sys.exit(1)
                else:
                    print("✓ 所有文件的测试覆盖率都达到80%以上")
            except Exception as e:
                print(f"警告: 无法检查覆盖率: {e}")


if __name__ == '__main__':
    main()

