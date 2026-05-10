#!/usr/bin/env python3
"""
Phase 2: 止损与移动止盈参数扫描

扫描 stop_loss_pct × trailing_stop_pct × trailing_stop_activation 组合，
基于 Phase 1 最优参数 (hold_days=3, top_n_hold=20)。

Usage:
    python scripts/param_scan_stoploss.py --start 20260105 --end 20260508
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "backtest" / "param_scan_stoploss"


def run_scan(start_date: str, end_date: str):
    log.info("=" * 70)
    log.info(f"Phase 2 止损/止盈参数扫描: {start_date} ~ {end_date}")
    log.info("基础参数: hold_days=3, top_n_hold=20, top_n_buy=10")
    log.info("=" * 70)

    # 参数网格
    stop_loss_list = [4.0, 6.0, 8.0, 10.0]
    trailing_stop_list = [2.0, 3.0, 5.0]
    activation_list = [3.0, 5.0]

    results = []
    total = len(stop_loss_list) * len(trailing_stop_list) * len(activation_list)
    count = 0

    for sl in stop_loss_list:
        for ts in trailing_stop_list:
            for act in activation_list:
                count += 1
                log.info(f"[{count}/{total}] stop_loss={sl}%, trailing_stop={ts}%, activation={act}%")

                bt = RealisticBacktester(
                    prediction_dir=PREDICTION_DIR,
                    initial_capital=10_000_000,
                    per_stock_amount=300_000,
                    top_n_buy=10,
                    hold_days=3,
                    top_n_hold=20,
                    stop_loss_pct=sl,
                    trailing_stop_pct=ts,
                    trailing_stop_activation=act,
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

                total_ret = result.get("total_return", 0)
                final_val = result.get("final_value", 0)

                # 计算回撤
                daily_values = result.get("daily_values", [])
                if isinstance(daily_values, pd.DataFrame):
                    daily_values = daily_values.to_dict('records')
                peak = result['initial_capital']
                max_dd = 0
                for v in daily_values:
                    tv = v['total_value']
                    if tv > peak:
                        peak = tv
                    dd = (peak - tv) / peak
                    if dd > max_dd:
                        max_dd = dd

                # 计算夏普（简化）
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
                    "stop_loss_pct": sl,
                    "trailing_stop_pct": ts,
                    "trailing_stop_activation": act,
                    "total_return_pct": round(total_ret, 2),
                    "final_value": int(final_val),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "max_drawdown_pct": round(max_dd * 100, 2),
                    "sharpe": round(sharpe, 2),
                }
                results.append(row)
                log.info(
                    f"  收益={row['total_return_pct']:6.2f}% | 回撤={row['max_drawdown_pct']:5.2f}% | "
                    f"夏普={row['sharpe']:5.2f} | 买入={buy_count:3d} | 卖出={sell_count:3d}"
                )

    if not results:
        log.error("无有效结果")
        return

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    csv_path = OUTPUT_DIR / f"scan_{start_date}_{end_date}_phase2.csv"
    df.to_csv(csv_path, index=False)
    log.success(f"结果已保存: {csv_path}")

    # 打印最优（按收益排序）
    print("\n" + "=" * 70)
    print("按总收益排序 Top 10")
    print("=" * 70)
    print(
        f"{'SL%':>5} {'TS%':>5} {'ACT%':>5} {'收益%':>8} {'回撤%':>7} {'夏普':>6} {'买入':>5} {'卖出':>5}"
    )
    print("-" * 70)
    for _, row in df.sort_values("total_return_pct", ascending=False).head(10).iterrows():
        print(
            f"{row['stop_loss_pct']:>5.0f} {row['trailing_stop_pct']:>5.0f} {row['trailing_stop_activation']:>5.0f} "
            f"{row['total_return_pct']:>8.2f} {row['max_drawdown_pct']:>7.2f} {row['sharpe']:>6.2f} "
            f"{int(row['buy_count']):>5} {int(row['sell_count']):>5}"
        )

    # 打印最优（按夏普排序）
    print("\n" + "=" * 70)
    print("按夏普排序 Top 10")
    print("=" * 70)
    print(
        f"{'SL%':>5} {'TS%':>5} {'ACT%':>5} {'收益%':>8} {'回撤%':>7} {'夏普':>6} {'买入':>5} {'卖出':>5}"
    )
    print("-" * 70)
    for _, row in df.sort_values("sharpe", ascending=False).head(10).iterrows():
        print(
            f"{row['stop_loss_pct']:>5.0f} {row['trailing_stop_pct']:>5.0f} {row['trailing_stop_activation']:>5.0f} "
            f"{row['total_return_pct']:>8.2f} {row['max_drawdown_pct']:>7.2f} {row['sharpe']:>6.2f} "
            f"{int(row['buy_count']):>5} {int(row['sell_count']):>5}"
        )

    # 保存 JSON 摘要
    summary = {
        "period": f"{start_date}~{end_date}",
        "base_params": {"hold_days": 3, "top_n_hold": 20, "top_n_buy": 10},
        "best_by_return": df.sort_values("total_return_pct", ascending=False).iloc[0].to_dict(),
        "best_by_sharpe": df.sort_values("sharpe", ascending=False).iloc[0].to_dict(),
        "all_results": results,
    }
    json_path = OUTPUT_DIR / f"scan_{start_date}_{end_date}_phase2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.success(f"摘要已保存: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 止损/止盈参数扫描")
    parser.add_argument("--start", default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260508", help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    run_scan(args.start, args.end)


if __name__ == "__main__":
    main()
