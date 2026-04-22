#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取投资组合当前状态
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_source import DataSource
import pandas as pd
from datetime import datetime

# 持仓列表
holdings = [
    {"ts_code": "600121.SH", "shares": 6600, "cost": 4.56, "name": "郑州煤电"},
    {"ts_code": "002471.SZ", "shares": 2200, "cost": 8.261, "name": "中超控股"},
    {"ts_code": "002149.SZ", "shares": 700, "cost": 47.32, "name": "西部材料"},
    {"ts_code": "002792.SZ", "shares": 600, "cost": 55.3, "name": "通宇通讯"},
    {"ts_code": "603601.SH", "shares": 2300, "cost": 12.85, "name": "再升科技"},
    {"ts_code": "603698.SH", "shares": 800, "cost": 39.94, "name": "航天工程"},
]

ds = DataSource()

print("=" * 100)
print(f"投资组合状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

total_cost = 0
total_value = 0
portfolio_data = []

for holding in holdings:
    ts_code = holding["ts_code"]
    shares = holding["shares"]
    cost = holding["cost"]
    name = holding["name"]

    # 获取最新价格
    try:
        df = ds.get_stock_data(ts_code, start_date="20260105", end_date="20260110")
        if not df.empty:
            df = df.sort_values("trade_date", ascending=False)
            latest = df.iloc[0]
            current_price = latest["close"]
            pct_chg = latest.get("pct_chg", 0)
            trade_date = latest["trade_date"]
        else:
            current_price = cost
            pct_chg = 0
            trade_date = "20260108"
    except Exception as e:
        print(f"获取{ts_code}数据失败: {e}")
        current_price = cost
        pct_chg = 0
        trade_date = "20260108"

    # 计算盈亏
    position_cost = shares * cost
    position_value = shares * current_price
    profit = position_value - position_cost
    profit_pct = (profit / position_cost * 100) if position_cost > 0 else 0

    total_cost += position_cost
    total_value += position_value

    portfolio_data.append(
        {
            "股票": f"{name}({ts_code})",
            "持仓": f"{shares}股",
            "成本": f"{cost:.2f}元",
            "现价": f"{current_price:.2f}元",
            "今涨跌": f"{pct_chg:+.2f}%",
            "持仓成本": f"{position_cost:.0f}元",
            "持仓市值": f"{position_value:.0f}元",
            "盈亏": f"{profit:+.0f}元",
            "盈亏率": f"{profit_pct:+.2f}%",
            "更新日期": trade_date,
        }
    )

# 显示详细信息
df_portfolio = pd.DataFrame(portfolio_data)
print(df_portfolio.to_string(index=False))

print("\n" + "=" * 100)
print(f"总投入: {total_cost:.2f}元")
print(f"总市值: {total_value:.2f}元")
print(f"总盈亏: {total_value - total_cost:+.2f}元 ({(total_value - total_cost) / total_cost * 100:+.2f}%)")
print("=" * 100)
