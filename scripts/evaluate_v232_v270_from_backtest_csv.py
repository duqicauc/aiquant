#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于回测操作记录与互补策略 CSV，按来源（v2.3.2 / v2.7.0）统计实际盈亏，评估两模型在策略中的表现。

不依赖 DataManager/API，仅读取：
- backtest_operations_20260105_20260303_sl_close.csv（或同区间操作记录）
- v232_v270_complementary_YYYYMMDD.csv（选股日对应日期，用于取 source）

用法：在项目根目录执行
  python scripts/evaluate_v232_v270_from_backtest_csv.py
"""

import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"

# 回测操作文件（按实际文件名调整）
OPS_PATTERN = "backtest_operations_20260105_20260303_sl_close.csv"


def parse_signal_date(reason):
    """从买入原因解析选股日，如 '进入Top10(选股日20260105)，当日开盘价买入' -> 20260105"""
    if pd.isna(reason) or not isinstance(reason, str):
        return None
    m = re.search(r"选股日(\d{8})", reason)
    return m.group(1) if m else None


def load_complementary_source(signal_date):
    """加载选股日互补结果，返回 ts_code -> source 的字典。"""
    path = RESULTS_DIR / f"v232_v270_complementary_{signal_date}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "source" not in df.columns or "ts_code" not in df.columns:
        return {}
    return df.set_index("ts_code")["source"].to_dict()


def main():
    ops_path = RESULTS_DIR / OPS_PATTERN
    if not ops_path.exists():
        # 尝试其他可能的后缀
        candidates = list(RESULTS_DIR.glob("backtest_operations_20260105_202603*.csv"))
        if not candidates:
            print(f"未找到回测操作文件，请将 {OPS_PATTERN} 放在 {RESULTS_DIR}")
            return
        ops_path = candidates[0]

    df = pd.read_csv(ops_path, encoding="utf-8-sig")
    # 建立 买入 -> 卖出 的对应：每笔卖出的 signal_date 来自该标的最近一次买入的 reason
    buy_dates = {}
    records = []  # (ts_code, sell_date, profit, profit_pct, signal_date, source)

    for _, row in df.iterrows():
        date, op, ts_code = str(row["date"]), row["operation"], row["ts_code"]
        if op == "买入":
            signal_date = parse_signal_date(row.get("reason", ""))
            if signal_date:
                buy_dates[ts_code] = signal_date
        elif op == "卖出":
            profit = row.get("profit")
            profit_pct = row.get("profit_pct")
            if pd.isna(profit):
                profit = 0
            if pd.isna(profit_pct):
                profit_pct = 0
            signal_date = buy_dates.get(ts_code)
            source = None
            if signal_date:
                src_map = load_complementary_source(signal_date)
                source = src_map.get(ts_code)
            records.append({
                "ts_code": ts_code,
                "sell_date": date,
                "profit": float(profit),
                "profit_pct": float(profit_pct),
                "signal_date": signal_date,
                "source": source or "unknown",
            })
            if ts_code in buy_dates:
                del buy_dates[ts_code]

    if not records:
        print("没有卖出记录可统计")
        return

    df_sell = pd.DataFrame(records)
    # 只保留能对应到 source 的（v2.3.2 / v2.7.0）
    df_sell = df_sell[df_sell["source"].isin(["v2.3.2", "v2.7.0"])]
    if df_sell.empty:
        print("没有能匹配到 source 的卖出记录，请确认互补策略 CSV 中含 source 列")
        return

    # 按 source 汇总
    summary = []
    for version in ["v2.3.2", "v2.7.0"]:
        sub = df_sell[df_sell["source"] == version]
        if sub.empty:
            continue
        n = len(sub)
        win = (sub["profit"] > 0).sum()
        summary.append({
            "version": version,
            "sample_count": n,
            "win_count": int(win),
            "win_rate_pct": round(win / n * 100, 2),
            "avg_profit_pct": round(sub["profit_pct"].mean(), 2),
            "median_profit_pct": round(sub["profit_pct"].median(), 2),
            "total_profit": round(sub["profit"].sum(), 0),
            "avg_profit": round(sub["profit"].mean(), 0),
            "max_profit_pct": round(sub["profit_pct"].max(), 2),
            "min_profit_pct": round(sub["profit_pct"].min(), 2),
        })

    df_summary = pd.DataFrame(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_sell.to_csv(OUTPUT_DIR / "v232_v270_eval_by_source_20260105_20260303_detail.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(OUTPUT_DIR / "v232_v270_eval_by_source_20260105_20260303_summary.csv", index=False, encoding="utf-8-sig")

    # 控制台
    print("\n" + "=" * 70)
    print("v2.3.2 与 v2.7.0 按来源统计（回测实际卖出 20260105-20260303）")
    print("=" * 70)
    for _, row in df_summary.iterrows():
        print(
            f"\n【{row['version']}】 样本数={row['sample_count']} 胜率={row['win_rate_pct']}% "
            f"平均盈亏%={row['avg_profit_pct']:+.2f}% 总盈亏={row['total_profit']:+,.0f}元 "
            f"最大={row['max_profit_pct']:+.2f}% 最小={row['min_profit_pct']:+.2f}%"
        )

    # Markdown 报告
    md_path = OUTPUT_DIR / "v232_v270_eval_by_source_20260105_20260303_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v2.3.2 与 v2.7.0 按来源评估（基于回测实际卖出）\n\n")
        f.write("**说明**：根据回测操作记录与互补策略 CSV 的 `source` 字段，统计「来自 v2.3.2 的标的」与「来自 v2.7.0 的标的」在**实际被卖出**时的盈亏。\n\n")
        f.write("**区间**：2026-01-05 至 2026-03-03（4% 止损版回测）。\n\n")
        f.write("## 汇总\n\n")
        f.write("| 模型 | 卖出笔数 | 胜率(%) | 平均盈亏% | 中位数盈亏% | 总盈亏(元) | 最大% | 最小% |\n")
        f.write("|------|----------|--------|-----------|-------------|------------|-------|-------|\n")
        for _, row in df_summary.iterrows():
            f.write(
                f"| {row['version']} | {row['sample_count']} | {row['win_rate_pct']} | "
                f"{row['avg_profit_pct']:+.2f} | {row['median_profit_pct']:+.2f} | "
                f"{row['total_profit']:+,.0f} | {row['max_profit_pct']:+.2f} | {row['min_profit_pct']:+.2f} |\n"
            )
        f.write("\n明细见 `v232_v270_eval_by_source_20260105_20260303_detail.csv`。\n")
    print(f"\n报告已保存: {md_path}")
    return df_summary


if __name__ == "__main__":
    main()
