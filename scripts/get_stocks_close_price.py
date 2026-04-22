#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取指定股票的收盘价
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import tushare as ts

# 加载环境变量
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# 初始化tushare
token = os.getenv("TUSHARE_TOKEN")
if token:
    ts.set_token(token)
    pro = ts.pro_api()
else:
    print("错误: 未设置TUSHARE_TOKEN环境变量")
    sys.exit(1)

# 股票列表
stocks = [
    "002149.SZ",  # 西部材料
    "002792.SZ",  # 通宇通讯
    "603601.SH",  # 再升科技
    "603698.SH",  # 航天工程
]

print("=" * 80)
print("获取股票1月8日收盘数据")
print("=" * 80)

for ts_code in stocks:
    try:
        # 获取最近的日线数据
        df = pro.daily(ts_code=ts_code, start_date="20260105", end_date="20260109")
        df = df.sort_values("trade_date", ascending=False)

        if not df.empty:
            latest = df.iloc[0]
            print(f"\n{ts_code}:")
            print(f"  日期: {latest['trade_date']}")
            print(f"  收盘: {latest['close']:.2f}元")
            print(f"  涨跌幅: {latest['pct_chg']:+.2f}%")
            print(f"  开盘: {latest['open']:.2f}元")
            print(f"  最高: {latest['high']:.2f}元")
            print(f"  最低: {latest['low']:.2f}元")
            print(f"  成交量: {latest['vol']/10000:.0f}万手")
            print(f"  成交额: {latest['amount']/1000:.0f}千元")
        else:
            print(f"\n{ts_code}: 未获取到数据")

    except Exception as e:
        print(f"\n{ts_code}: 获取失败 - {e}")

print("\n" + "=" * 80)
