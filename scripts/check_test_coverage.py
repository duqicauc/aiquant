#!/usr/bin/env python3
"""
测试覆盖率检查和报告工具

检查代码修改是否配套了测试用例，并生成覆盖率报告。

使用方法:
    python scripts/check_test_coverage.py                    # 检查整体覆盖率
    python scripts/check_test_coverage.py --file src/module.py # 检查特定文件
    python scripts/check_test_coverage.py --modified           # 检查修改的文件
    python scripts/check_test_coverage.py --report             # 生成详细报告
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CoverageChecker:
    """测试覆盖率检查器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / 'src'
        self.tests_dir = project_root / 'tests'
    
    def run_coverage(self, files: List[Path] = None) -> Dict:
        """运行覆盖率检查"""
        cmd = ['pytest', '--cov=src', '--cov-report=json', '--cov-report=term-missing', '-q']
        
        if files:
            # 只测试特定文件相关的测试
            test_files = []
            for file in files:
                test_file = self._find_test_file(file)
                if test_file:
                    test_files.append(str(test_file))
            if test_files:
                cmd.extend(test_files)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 读取JSON报告
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except subprocess.TimeoutExpired:
            print('警告: 测试超时')
            return {}
        except Exception as e:
            print(f'错误: 运行覆盖率检查失败: {e}')
            return {}
    
    def _find_test_file(self, source_file: Path) -> Path:
        """查找对应的测试文件"""
        # src/data/data_manager.py -> tests/data/test_data_manager.py
        parts = source_file.parts
        if 'src' in parts:
            idx = parts.index('src')
            test_parts = ['tests'] + list(parts[idx+1:-1]) + [f'test_{parts[-1]}']
            test_file = self.project_root / Path(*test_parts)
            if test_file.exists():
                return test_file
        return None
    
    def check_file_coverage(self, file_path: Path) -> Dict:
        """检查单个文件的覆盖率"""
        coverage_data = self.run_coverage([file_path])
        
        if not coverage_data:
            return {
                'file': str(file_path),
                'covered': False,
                'coverage': 0.0,
                'missing_lines': []
            }
        
        # 查找文件覆盖率
        relative_path = str(file_path.relative_to(self.project_root))
        files_data = coverage_data.get('files', {})
        
        for file_key, file_data in files_data.items():
            if relative_path in file_key or file_key.endswith(relative_path):
                return {
                    'file': file_key,
                    'covered': True,
                    'coverage': file_data.get('summary', {}).get('percent_covered', 0.0),
                    'missing_lines': file_data.get('missing_lines', []),
                    'statements': file_data.get('summary', {}).get('num_statements', 0),
                    'covered_statements': file_data.get('summary', {}).get('covered_lines', 0)
                }
        
        return {
            'file': relative_path,
            'covered': False,
            'coverage': 0.0,
            'missing_lines': []
        }
    
    def check_modified_files(self) -> List[Dict]:
        """检查修改的文件是否有测试"""
        try:
            # 使用git获取修改的文件
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            modified_files = []
            for line in result.stdout.strip().split('\n'):
                if line and line.startswith('src/'):
                    file_path = self.project_root / line
                    if file_path.exists() and file_path.suffix == '.py':
                        modified_files.append(file_path)
            
            results = []
            for file_path in modified_files:
                result = self.check_file_coverage(file_path)
                results.append(result)
            
            return results
        except Exception as e:
            print(f'警告: 无法获取修改的文件: {e}')
            return []
    
    def generate_report(self, coverage_data: Dict = None) -> str:
        """生成覆盖率报告"""
        if coverage_data is None:
            coverage_data = self.run_coverage()
        
        if not coverage_data:
            return "无法生成覆盖率报告"
        
        summary = coverage_data.get('totals', {})
        files_data = coverage_data.get('files', {})
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("测试覆盖率报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 总体统计
        report_lines.append("总体统计:")
        report_lines.append(f"  总语句数: {summary.get('num_statements', 0)}")
        report_lines.append(f"  已覆盖: {summary.get('covered_lines', 0)}")
        report_lines.append(f"  未覆盖: {summary.get('missing_lines', 0)}")
        report_lines.append(f"  覆盖率: {summary.get('percent_covered', 0.0):.2f}%")
        report_lines.append("")
        
        # 按模块统计
        report_lines.append("模块覆盖率:")
        report_lines.append("-" * 80)
        
        module_stats = {}
        for file_key, file_data in files_data.items():
            module = file_key.split('/')[0] if '/' in file_key else file_key
            if module not in module_stats:
                module_stats[module] = {
                    'files': 0,
                    'total_statements': 0,
                    'covered_statements': 0
                }
            
            summary_data = file_data.get('summary', {})
            module_stats[module]['files'] += 1
            module_stats[module]['total_statements'] += summary_data.get('num_statements', 0)
            module_stats[module]['covered_statements'] += summary_data.get('covered_lines', 0)
        
        for module, stats in sorted(module_stats.items()):
            coverage = (stats['covered_statements'] / stats['total_statements'] * 100 
                       if stats['total_statements'] > 0 else 0)
            report_lines.append(
                f"  {module:20s} {coverage:6.2f}% "
                f"({stats['covered_statements']}/{stats['total_statements']} statements)"
            )
        
        report_lines.append("")
        
        # 低覆盖率文件
        report_lines.append("低覆盖率文件 (< 80%):")
        report_lines.append("-" * 80)
        
        low_coverage_files = []
        for file_key, file_data in files_data.items():
            summary_data = file_data.get('summary', {})
            coverage = summary_data.get('percent_covered', 0.0)
            if coverage < 80.0:
                low_coverage_files.append((file_key, coverage, summary_data))
        
        low_coverage_files.sort(key=lambda x: x[1])
        
        for file_key, coverage, summary_data in low_coverage_files[:20]:  # 只显示前20个
            report_lines.append(
                f"  {file_key:50s} {coverage:6.2f}% "
                f"({summary_data.get('covered_lines', 0)}/{summary_data.get('num_statements', 0)})"
            )
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description='检查测试覆盖率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查整体覆盖率
  python scripts/check_test_coverage.py
  
  # 检查特定文件
  python scripts/check_test_coverage.py --file src/data/data_manager.py
  
  # 检查修改的文件
  python scripts/check_test_coverage.py --modified
  
  # 生成详细报告
  python scripts/check_test_coverage.py --report
        """
    )
    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='检查特定文件的覆盖率'
    )
    parser.add_argument(
        '--modified', '-m',
        action='store_true',
        help='检查修改的文件是否有测试'
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='生成详细的覆盖率报告'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='报告输出文件路径'
    )
    
    args = parser.parse_args()
    
    checker = CoverageChecker(project_root)
    
    if args.modified:
        # 检查修改的文件
        print("检查修改的文件...")
        results = checker.check_modified_files()
        
        if not results:
            print("没有找到修改的Python文件")
            return
        
        print(f"\n找到 {len(results)} 个修改的文件:\n")
        
        for result in results:
            status = "✓" if result['coverage'] >= 80 else "✗"
            print(f"{status} {result['file']}")
            print(f"  覆盖率: {result['coverage']:.2f}%")
            if result['missing_lines']:
                print(f"  未覆盖行数: {len(result['missing_lines'])}")
            print()
        
        # 检查是否有未覆盖的文件
        uncovered = [r for r in results if r['coverage'] < 80]
        if uncovered:
            print(f"\n警告: {len(uncovered)} 个文件覆盖率低于80%")
            print("请为这些文件添加测试用例")
            sys.exit(1)
        else:
            print("✓ 所有修改的文件都有足够的测试覆盖")
    
    elif args.file:
        # 检查特定文件
        file_path = project_root / args.file if not args.file.is_absolute() else args.file
        result = checker.check_file_coverage(file_path)
        
        print(f"文件: {result['file']}")
        print(f"覆盖率: {result['coverage']:.2f}%")
        print(f"语句数: {result.get('statements', 0)}")
        print(f"已覆盖: {result.get('covered_statements', 0)}")
        
        if result['missing_lines']:
            print(f"\n未覆盖的行: {result['missing_lines'][:20]}")  # 只显示前20行
    
    else:
        # 生成整体报告
        print("运行测试覆盖率检查...")
        coverage_data = checker.run_coverage()
        
        if args.report:
            report = checker.generate_report(coverage_data)
            print(report)
            
            if args.output:
                output_path = project_root / args.output if not args.output.is_absolute() else args.output
                output_path.write_text(report, encoding='utf-8')
                print(f"\n报告已保存到: {output_path}")
        else:
            # 简单输出
            summary = coverage_data.get('totals', {})
            coverage = summary.get('percent_covered', 0.0)
            print(f"\n总体覆盖率: {coverage:.2f}%")
            
            if coverage < 85.0:
                print(f"警告: 覆盖率低于目标值 85%")
                print("请运行 'python scripts/check_test_coverage.py --report' 查看详细报告")
                sys.exit(1)
            else:
                print("✓ 覆盖率达标")


if __name__ == '__main__':
    main()

