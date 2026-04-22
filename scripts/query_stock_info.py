#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询单只股票的基本信息
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager

dm = DataManager()
ts_code = "001280.SZ"

# 获取股票基本信息
stock_list = dm.get_stock_list()
stock_info = stock_list[stock_list["ts_code"] == ts_code]
if stock_info.empty:
    print(f"未找到股票: {ts_code}")
else:
    name = stock_info.iloc[0]["name"]
    print(f"股票名称: {name} ({ts_code})")
    print(f'上市日期: {stock_info.iloc[0].get("list_date", "未知")}')
    print(f'行业: {stock_info.iloc[0].get("industry", "未知")}')

    # 获取最新日线数据
    end_date = "20260126"
    start_date = "20251201"
    df = dm.get_daily_data(ts_code, start_date, end_date)
    if df is not None and len(df) > 0:
        df = df.sort_values("trade_date")
        latest = df.iloc[-1]
        print(f'\n最新交易日: {latest["trade_date"]}')
        print(f'收盘价: {latest["close"]:.2f}元')
        print(f'涨跌幅: {latest["pct_chg"]:+.2f}%')
        print(f'成交量: {latest["vol"]/10000:.0f}万手')
        print(f'成交额: {latest["amount"]/10000:.2f}万元')

        # 计算近期表现
        if len(df) >= 5:
            recent_5d = df.tail(5)
            pct_5d = (recent_5d.iloc[-1]["close"] - recent_5d.iloc[0]["close"]) / recent_5d.iloc[0]["close"] * 100
            print(f"近5日涨跌幅: {pct_5d:+.2f}%")

        if len(df) >= 20:
            recent_20d = df.tail(20)
            pct_20d = (recent_20d.iloc[-1]["close"] - recent_20d.iloc[0]["close"]) / recent_20d.iloc[0]["close"] * 100
            print(f"近20日涨跌幅: {pct_20d:+.2f}%")
    else:
        print("无法获取日线数据")
