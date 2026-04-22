#!/usr/bin/env python3
"""
检查测试覆盖率

分析核心模块的测试覆盖情况
"""
import subprocess
import sys
from pathlib import Path

# 核心模块列表
CORE_MODULES = {
    "数据管理": [
        "src/data/data_manager.py",
        "src/data/fetcher/tushare_fetcher.py",
        "src/data/storage/cache_manager.py",
    ],
    "模型管理": [
        "src/models/lifecycle/iterator.py",
        "src/models/lifecycle/trainer.py",
        "src/models/lifecycle/predictor.py",
        "src/models/model_registry.py",
    ],
    "策略模块": [
        "src/strategy/screening/positive_sample_screener.py",
        "src/strategy/screening/negative_sample_screener_v2.py",
    ],
    "分析模块": [
        "src/analysis/stock_health_checker.py",
        "src/analysis/market_analyzer.py",
    ],
    "配置管理": [
        "config/settings.py",
        "config/config.py",
    ],
}


def run_coverage_check():
    """运行覆盖率检查"""
    print("=" * 80)
    print("测试覆盖率分析")
    print("=" * 80)

    # 运行pytest覆盖率
    cmd = [sys.executable, "-m", "pytest", "--cov=src", "--cov=config", "--cov-report=term-missing", "-q", "tests/"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # 解析输出
        lines = result.stdout.split("\n")

        # 查找覆盖率报告部分
        in_coverage = False
        coverage_lines = []

        for line in lines:
            if "Name" in line and "Stmts" in line:
                in_coverage = True
                coverage_lines.append(line)
                continue
            if in_coverage:
                if line.strip() and not line.startswith("-"):
                    coverage_lines.append(line)
                elif "TOTAL" in line:
                    coverage_lines.append(line)
                    break

        # 打印覆盖率报告
        print("\n📊 覆盖率报告:")
        print("-" * 80)
        for line in coverage_lines:
            print(line)

        # 分析核心模块
        print("\n" + "=" * 80)
        print("核心模块覆盖情况")
        print("=" * 80)

        for category, modules in CORE_MODULES.items():
            print(f"\n📦 {category}:")
            for module in modules:
                # 在覆盖率报告中查找该模块
                module_name = module.replace("/", ".").replace(".py", "")
                found = False
                for line in coverage_lines:
                    if module_name in line or Path(module).name in line:
                        print(f"  {Path(module).name}: {line.strip()}")
                        found = True
                        break
                if not found:
                    print(f"  {Path(module).name}: ❌ 未找到覆盖率数据")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False


if __name__ == "__main__":
    success = run_coverage_check()
    sys.exit(0 if success else 1)
