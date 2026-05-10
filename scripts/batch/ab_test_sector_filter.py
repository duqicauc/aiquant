#!/usr/bin/env python3
"""
SectorFilter A/B 测试脚本
对照组 vs 实验组并行跑三季回测，自动对比结果

用法:
    python scripts/batch/ab_test_sector_filter.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PERIODS = {
    "2024q4": ("20241008", "20241231"),
    "2025q1": ("20250102", "20250331"),
    "2026q1": ("20260105", "20260331"),
}

BASE_PARAMS = {
    "per_stock": 50000,
    "capital": 500000,
    "stop_loss": 10.0,
    "trailing_stop": 2.0,
    "trailing_activation": 5.0,
    "prediction_dir": "data/prediction/v291_stk_factor",
}

VARIANTS = {
    "baseline": {"enable_sector_filter": False, "suffix": "baseline"},
    "sector_full": {"enable_sector_filter": True, "suffix": "sector_full"},
    "sector_hot_only": {"enable_sector_filter": True, "suffix": "sector_hot_only"},  # 需配合修改policy开关
    "sector_policy_only": {"enable_sector_filter": True, "suffix": "sector_policy_only"},
}


def run_backtest(period_name, start, end, variant_name, params, output_dir):
    """运行单个回测"""
    cmd = [
        "python3", "scripts/backtest_v291_realistic.py",
        "--start-date", start,
        "--end-date", end,
        "--per-stock", str(params["per_stock"]),
        "--capital", str(params["capital"]),
        "--stop-loss", str(params["stop_loss"]),
        "--trailing-stop", str(params["trailing_stop"]),
        "--trailing-activation", str(params["trailing_activation"]),
        "--prediction-dir", params["prediction_dir"],
        "--output-dir", str(output_dir),
    ]
    if params.get("enable_sector_filter"):
        cmd.append("--enable-sector-filter")

    log_file = Path(output_dir) / "run.log"
    print(f"[{period_name}/{variant_name}] 启动回测...")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    log_file.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        print(f"  ❌ 失败: {result.stderr[:200]}")
        return False
    print(f"  ✅ 完成")
    return True


def extract_metrics(output_dir):
    """从回测报告中提取关键指标"""
    report_path = Path(output_dir) / "backtest_report.md"
    if not report_path.exists():
        return None

    text = report_path.read_text()
    metrics = {}
    for line in text.split("\n"):
        if "总收益率" in line:
            metrics["total_return"] = line.split("|")[-2].strip()
        elif "最大回撤" in line:
            metrics["max_drawdown"] = line.split("|")[-2].strip()
        elif "胜率" in line:
            metrics["win_rate"] = line.split("|")[-2].strip()
        elif "盈亏比" in line:
            metrics["profit_loss_ratio"] = line.split("|")[-2].strip()
        elif "总交易费用" in line:
            metrics["total_fees"] = line.split("|")[-2].strip()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="baseline,sector_full", help="测试的变体，逗号分隔")
    parser.add_argument("--periods", default="2024q4,2025q1,2026q1", help="回测期间，逗号分隔")
    args = parser.parse_args()

    variant_names = args.variants.split(",")
    period_names = args.periods.split(",")

    results = {}

    for period_name in period_names:
        if period_name not in PERIODS:
            print(f"跳过未知期间: {period_name}")
            continue
        start, end = PERIODS[period_name]
        results[period_name] = {}

        for var_name in variant_names:
            if var_name not in VARIANTS:
                print(f"跳过未知变体: {var_name}")
                continue

            var_cfg = VARIANTS[var_name]
            output_dir = PROJECT_ROOT / "data" / "results" / f"v291_ab_{period_name}_{var_cfg['suffix']}"
            output_dir.mkdir(parents=True, exist_ok=True)

            params = {**BASE_PARAMS, **var_cfg}
            success = run_backtest(period_name, start, end, var_name, params, output_dir)

            if success:
                metrics = extract_metrics(output_dir)
                results[period_name][var_name] = metrics
                print(f"  指标: {metrics}")

    # 汇总报告
    print("\n" + "=" * 60)
    print("A/B 测试汇总")
    print("=" * 60)
    for period_name, period_results in results.items():
        print(f"\n{period_name}:")
        for var_name, metrics in period_results.items():
            if metrics:
                print(f"  {var_name:20s}: 收益={metrics.get('total_return','N/A'):8s} 回撤={metrics.get('max_drawdown','N/A'):8s} 胜率={metrics.get('win_rate','N/A'):8s} 盈亏比={metrics.get('profit_loss_ratio','N/A'):8s}")

    # 保存JSON
    summary_path = PROJECT_ROOT / "data" / "results" / "v291_ab_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
