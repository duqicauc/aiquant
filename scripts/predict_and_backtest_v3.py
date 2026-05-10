#!/usr/bin/env python3
"""
v3.0.0 端到端预测 + 回测流水线

1. V3Predictor 生成预测信号
2. 保存预测结果到 data/prediction/v3.0.0/
3. qlib 风格回测
4. vectorbt 回测
5. 结果对比

Usage:
    python scripts/predict_and_backtest_v3.py \
        --start 20260101 --end 20260430 \
        --top_k 10 --backtest
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.qlib_backtest import QlibStyleBacktest
from src.backtest.vbt_backtest import VBTBacktest
from src.models.v3_predictor import V3Predictor
from src.utils.logger import log

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v3.0.0"


def run_prediction(start_date: str, end_date: str) -> Path:
    """运行预测并保存"""
    log.info(f"{'='*60}")
    log.info("Step 1: V3.0.0 预测生成")
    log.info(f"{'='*60}")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    predictor = V3Predictor()

    results = predictor.predict_range(start_date, end_date)

    for date, df_pred in results.items():
        # 保存全部预测
        path_all = PREDICTION_DIR / f"predictions_{date}_all.csv"
        df_pred.to_csv(path_all, index=False)

        # 保存 Top100
        path_top100 = PREDICTION_DIR / f"predictions_{date}_top100.csv"
        df_pred.head(100).to_csv(path_top100, index=False)

        # 保存 Top50
        path_top50 = PREDICTION_DIR / f"predictions_{date}_top50.csv"
        df_pred.head(50).to_csv(path_top50, index=False)

    log.success(f"预测完成: {len(results)} 个交易日, 保存至 {PREDICTION_DIR}")
    return PREDICTION_DIR


def run_qlib_backtest(start_date: str, end_date: str, top_k: int, output_dir: Path):
    """qlib 风格回测"""
    log.info(f"{'='*60}")
    log.info("Step 2: qlib 风格回测")
    log.info(f"{'='*60}")

    bt = QlibStyleBacktest(prediction_dir=str(PREDICTION_DIR))
    result = bt.run(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        drop_n=5,
        hold_days=5,
    )

    if result:
        out = output_dir / "qlib"
        out.mkdir(parents=True, exist_ok=True)
        bt.save_report(result, out / "report.json")
        result["daily_returns"].to_csv(out / "daily_returns.csv", index=False)
        result["portfolio"].to_csv(out / "portfolio.csv", index=False)
        log.success(f"qlib 回测结果: {out}")
        return result["metrics"]
    return {}


def run_vbt_backtest(start_date: str, end_date: str, top_k: int, output_dir: Path):
    """vectorbt 回测"""
    log.info(f"{'='*60}")
    log.info("Step 3: vectorbt 回测")
    log.info(f"{'='*60}")

    bt = VBTBacktest(prediction_dir=str(PREDICTION_DIR))
    result = bt.run(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        drop_n=5,
        hold_days=5,
    )

    if result:
        out = output_dir / "vbt"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "report.json", "w") as f:
            json.dump(
                {
                    "metrics": {
                        k: float(v) if isinstance(v, float) else v for k, v in result.items() if k not in ["portfolio"]
                    },
                    "params": result["params"],
                },
                f,
                indent=2,
            )
        log.success(f"vectorbt 回测结果: {out}")
        return {
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
        }
    return {}


def print_comparison(qlib_metrics: dict, vbt_metrics: dict):
    """打印两框架对比"""
    log.info(f"\n{'='*60}")
    log.info("回测结果对比")
    log.info(f"{'='*60}")
    log.info(f"{'指标':<15} {'qlib':>15} {'vectorbt':>15}")
    log.info(f"{'-'*60}")

    keys = [
        ("总收益", "total_return", lambda x: f"{x*100:.2f}%"),
        ("年化收益", "annual_return", lambda x: f"{x*100:.2f}%"),
        ("夏普比率", "sharpe_ratio", lambda x: f"{x:.2f}"),
        ("最大回撤", "max_drawdown", lambda x: f"{x*100:.2f}%"),
    ]

    for name, key, fmt in keys:
        qv = qlib_metrics.get(key, None)
        vv = vbt_metrics.get(key, None)
        qs = fmt(qv) if qv is not None else "N/A"
        vs = fmt(vv) if vv is not None else "N/A"
        log.info(f"{name:<15} {qs:>15} {vs:>15}")

    log.info(f"{'='*60}")
    log.info("注意: qlib 为日频等权近似，vectorbt 为向量化信号近似")
    log.info("精确回测请使用 src.backtest.backtester_realistic.RealisticBacktester")


def main():
    parser = argparse.ArgumentParser(description="v3.0.0 预测 + 回测流水线")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--top_k", type=int, default=10, help="每日持仓数量")
    parser.add_argument("--predict", action="store_true", default=True, help="执行预测")
    parser.add_argument("--backtest", action="store_true", help="执行回测")
    parser.add_argument("--output_dir", default="data/backtest/v3_comparison", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 预测
    if args.predict:
        run_prediction(args.start, args.end)

    # Step 2 & 3: 回测
    qlib_metrics = {}
    vbt_metrics = {}

    if args.backtest:
        qlib_metrics = run_qlib_backtest(args.start, args.end, args.top_k, output_dir)
        vbt_metrics = run_vbt_backtest(args.start, args.end, args.top_k, output_dir)
        print_comparison(qlib_metrics, vbt_metrics)
    else:
        log.info("跳过回测（使用 --backtest 启用）")

    log.success("流水线完成!")


if __name__ == "__main__":
    main()
