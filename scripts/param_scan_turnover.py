#!/usr/bin/env python3
"""
RealisticBacktester 换手率参数扫描

扫描 hold_days × top_n_hold 组合，找到最优换仓灵敏度。

Usage:
    python scripts/param_scan_turnover.py --start 20260105 --end 20260508
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtester_realistic import RealisticBacktester
from src.utils.logger import log

PREDICTION_DIR = str(PROJECT_ROOT / "data" / "prediction" / "v3.0.0")
OUTPUT_DIR = PROJECT_ROOT / "data" / "backtest" / "param_scan_turnover"


def run_scan(start_date: str, end_date: str, top_n_buy: int = 10):
    log.info("=" * 70)
    log.info(f"换手率参数扫描: {start_date} ~ {end_date}, top_n_buy={top_n_buy}")
    log.info("=" * 70)

    # 参数网格
    hold_days_list = [0, 2, 3, 5, 7, 10]
    top_n_hold_list = [10, 15, 20, 30, 50]

    results = []
    total = len(hold_days_list) * len(top_n_hold_list)
    count = 0

    for hold in hold_days_list:
        for top_n in top_n_hold_list:
            count += 1
            log.info(f"[{count}/{total}] hold_days={hold}, top_n_hold={top_n}")

            bt = RealisticBacktester(
                prediction_dir=PREDICTION_DIR,
                initial_capital=10_000_000,
                per_stock_amount=300_000,
                top_n_buy=top_n_buy,
                hold_days=hold,
                top_n_hold=top_n,
            )

            try:
                result = bt.run(start_date, end_date)
            except Exception as e:
                log.error(f"  回测失败: {e}")
                continue

            if not result:
                log.warning("  回测结果为空")
                continue

            # 统计交易次数
            txs = result.get("transactions", [])
            if isinstance(txs, pd.DataFrame):
                buy_count = len(txs[txs["action"] == "BUY"]) if not txs.empty else 0
                sell_count = len(txs[txs["action"] == "SELL"]) if not txs.empty else 0
            else:
                buy_count = len([t for t in txs if isinstance(t, dict) and t.get("action") == "BUY"])
                sell_count = len([t for t in txs if isinstance(t, dict) and t.get("action") == "SELL"])

            total_ret = result.get("total_return", 0)  # 已经是百分比形式
            final_val = result.get("final_value", 0)

            # 计算日均收益和夏普（简化）
            daily_values = result.get("daily_values", [])
            if isinstance(daily_values, pd.DataFrame):
                daily_values = daily_values.to_dict('records')
            if len(daily_values) >= 2:
                rets = []
                for i in range(1, len(daily_values)):
                    prev = daily_values[i - 1]["total_value"]
                    curr = daily_values[i]["total_value"]
                    rets.append((curr - prev) / prev)
                mean_ret = pd.Series(rets).mean()
                std_ret = pd.Series(rets).std()
                sharpe = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
            else:
                sharpe = 0

            row = {
                "hold_days": hold,
                "top_n_hold": top_n,
                "total_return_pct": round(total_ret, 2),
                "final_value": int(final_val),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "sharpe": round(sharpe, 2),
            }
            results.append(row)
            log.info(
                f"  收益={row['total_return_pct']:6.2f}% | 夏普={row['sharpe']:5.2f} | "
                f"买入={buy_count:3d} | 卖出={sell_count:3d}"
            )

    if not results:
        log.error("无有效结果")
        return

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    csv_path = OUTPUT_DIR / f"scan_{start_date}_{end_date}_top{top_n_buy}.csv"
    df.to_csv(csv_path, index=False)
    log.success(f"结果已保存: {csv_path}")

    # 打印最优（按收益排序）
    print("\n" + "=" * 70)
    print("按总收益排序 Top 10")
    print("=" * 70)
    print(
        f"{'hold':>4} {'topN':>4} {'收益%':>8} {'夏普':>6} {'买入':>5} {'卖出':>5}"
    )
    print("-" * 70)
    for _, row in df.sort_values("total_return_pct", ascending=False).head(10).iterrows():
        print(
            f"{int(row['hold_days']):>4} {int(row['top_n_hold']):>4} "
            f"{row['total_return_pct']:>8.2f} {row['sharpe']:>6.2f} "
            f"{int(row['buy_count']):>5} {int(row['sell_count']):>5}"
        )

    # 打印最优（按夏普排序）
    print("\n" + "=" * 70)
    print("按夏普排序 Top 10")
    print("=" * 70)
    print(
        f"{'hold':>4} {'topN':>4} {'收益%':>8} {'夏普':>6} {'买入':>5} {'卖出':>5}"
    )
    print("-" * 70)
    for _, row in df.sort_values("sharpe", ascending=False).head(10).iterrows():
        print(
            f"{int(row['hold_days']):>4} {int(row['top_n_hold']):>4} "
            f"{row['total_return_pct']:>8.2f} {row['sharpe']:>6.2f} "
            f"{int(row['buy_count']):>5} {int(row['sell_count']):>5}"
        )

    # 保存 JSON 摘要
    summary = {
        "period": f"{start_date}~{end_date}",
        "top_n_buy": top_n_buy,
        "best_by_return": df.sort_values("total_return_pct", ascending=False).iloc[0].to_dict(),
        "best_by_sharpe": df.sort_values("sharpe", ascending=False).iloc[0].to_dict(),
        "all_results": results,
    }
    json_path = OUTPUT_DIR / f"scan_{start_date}_{end_date}_top{top_n_buy}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.success(f"摘要已保存: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="RealisticBacktester 换手率参数扫描")
    parser.add_argument("--start", default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260508", help="结束日期 YYYYMMDD")
    parser.add_argument("--top_n_buy", type=int, default=10, help="每日买入数量")
    args = parser.parse_args()

    run_scan(args.start, args.end, args.top_n_buy)


if __name__ == "__main__":
    main()
