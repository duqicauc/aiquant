#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按选股来源(v232 vs v270)拆分 v232+v270 互补策略回测的已实现盈亏。

用法: python scripts/analyze_backtest_by_source.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "prediction" / "results"
OPS_FILE = RESULTS_DIR / "backtest_operations_20260105_20260129.csv"


def parse_signal_date(reason: str):
    """从买入原因中解析选股日，如 进入Top10(选股日20260105)，当日开盘价买入 -> 20260105"""
    if pd.isna(reason) or not isinstance(reason, str):
        return None
    m = re.search(r"选股日(\d{8})", reason)
    return m.group(1) if m else None


def load_top10_sources(signal_date: str) -> dict:
    """加载选股日互补结果，返回 Top10 的 ts_code -> source"""
    path = RESULTS_DIR / f"v232_v270_complementary_{signal_date}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "sort_key" in df.columns:
        df = df.sort_values("sort_key", ascending=False)
    elif "dual_score" in df.columns:
        df = df.sort_values("dual_score", ascending=False)
    top10 = df.head(10)
    if "source" not in top10.columns:
        return {}
    return dict(zip(top10["ts_code"], top10["source"]))


def main():
    df = pd.read_csv(OPS_FILE, encoding="utf-8-sig")
    buys = df[df["operation"] == "买入"].copy()
    sells = df[df["operation"] == "卖出"].copy()

    buys["signal_date"] = buys["reason"].apply(parse_signal_date)
    # 收集所有选股日
    signal_dates = buys["signal_date"].dropna().unique().tolist()
    # 预加载每个选股日的 Top10 source 映射
    source_cache = {}
    for sd in signal_dates:
        source_cache[sd] = load_top10_sources(sd)

    # 为每笔卖出找到“开仓”买入：该 ts_code 在此次卖出之前最近一次买入（即建仓或加仓的那次选股）
    # 简化：按时间顺序，每笔卖出对应的开仓 = 该 ts_code 在此卖出之前、上一次卖出之后的第一次买入
    # 更简单：每笔卖出对应“建仓”的选股日 = 该仓位首次买入的选股日。按日期遍历，维护每个 ts_code 的当前仓位的 entry_signal_date
    entry_by_position = {}  # ts_code -> signal_date (当前仓位是由哪天的选股进的)
    sell_attributes = []  # (ts_code, sell_date, profit, source)

    for _, row in df.iterrows():
        date, op, ts_code = row["date"], row["operation"], row["ts_code"]
        if op == "买入":
            signal_date = parse_signal_date(row["reason"])
            if signal_date:
                # 该股票此次买入对应的选股日；若已有仓位则是加仓，我们仍用首次建仓的选股日作为“来源”
                if ts_code not in entry_by_position:
                    entry_by_position[ts_code] = signal_date
        elif op == "卖出":
            profit = row.get("profit")
            if pd.isna(profit):
                profit = 0
            signal_date = entry_by_position.get(ts_code)
            source = "unknown"
            if signal_date and signal_date in source_cache:
                source = source_cache[signal_date].get(ts_code, "unknown")
            sell_attributes.append((ts_code, date, profit, source))
            # 清空该仓位，下次再买算新仓位
            if ts_code in entry_by_position:
                del entry_by_position[ts_code]

    # 汇总按 source
    by_source = {}
    for ts_code, sell_date, profit, source in sell_attributes:
        by_source.setdefault(source, {"profit": 0, "count": 0, "trades": []})
        by_source[source]["profit"] += profit
        by_source[source]["count"] += 1
        by_source[source]["trades"].append((ts_code, sell_date, profit))

    # 输出
    print("=" * 60)
    print("v232+v270 互补策略回测 — 按选股来源(v232/v270)拆分的已实现盈亏")
    print("=" * 60)
    print(f"数据: {OPS_FILE.name}")
    print()

    total_profit = 0
    for src in ["v2.3.2", "v2.7.0", "unknown"]:
        if src not in by_source:
            continue
        rec = by_source[src]
        total_profit += rec["profit"]
        label = "v232 选股" if src == "v2.3.2" else ("v270 选股" if src == "v2.7.0" else "未知来源")
        print(f"【{label}】 source={src}")
        print(f"  卖出笔数: {rec['count']}")
        print(f"  已实现盈亏合计: {rec['profit']:+,.0f} 元")
        if rec["count"] > 0:
            print(f"  平均每笔: {rec['profit']/rec['count']:+,.0f} 元")
        wins = sum(1 for _, _, p in rec["trades"] if p > 0)
        losses = sum(1 for _, _, p in rec["trades"] if p <= 0)
        print(f"  盈利笔数: {wins}, 亏损笔数: {losses}")
        print()
    print("合计已实现盈亏:", f"{total_profit:+,.0f} 元")
    print()
    # 简要结论
    v232_profit = by_source.get("v2.3.2", {}).get("profit", 0)
    v270_profit = by_source.get("v2.7.0", {}).get("profit", 0)
    print("结论:")
    if v232_profit >= 0:
        print(f"  v232 选股贡献了盈利: {v232_profit:+,.0f} 元")
    else:
        print(f"  v232 选股贡献了亏损: {v232_profit:+,.0f} 元")
    if v270_profit >= 0:
        print(f"  v270 选股贡献了盈利: {v270_profit:+,.0f} 元")
    else:
        print(f"  v270 选股贡献了亏损: {v270_profit:+,.0f} 元")


if __name__ == "__main__":
    main()
