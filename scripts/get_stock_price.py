#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_source import DataSource


def main():
    ts_code = sys.argv[1] if len(sys.argv) > 1 else "600121.SH"

    ds = DataSource()
    df = ds.get_stock_data(ts_code, start_date="20260105", end_date="20260109")
    df = df.sort_values("trade_date")

    print("=" * 80)
    print(f"{ts_code} 最近行情:")
    print("=" * 80)
    for _, row in df.tail(5).iterrows():
        date = row["trade_date"]
        close = row["close"]
        pct_chg = row.get("pct_chg", 0)
        vol = row.get("vol", 0) / 10000
        high = row.get("high", close)
        low = row.get("low", close)

        print(
            f"日期: {date}  收盘: {close:.2f}元  涨跌: {pct_chg:+.2f}%  成交量: {vol:.0f}万手  最高: {high:.2f}  最低: {low:.2f}"
        )

    print("\n" + "=" * 80)
    latest = df.iloc[-1]
    print(f"最新数据（{latest['trade_date']}）:")
    print(f"  收盘价: {latest['close']:.2f}元")
    print(f"  涨跌幅: {latest.get('pct_chg', 0):+.2f}%")
    print(f"  成交量: {latest.get('vol', 0)/10000:.0f}万手")
    print(f"  换手率: {latest.get('turnover_rate', 0):.2f}%")


if __name__ == "__main__":
    main()
