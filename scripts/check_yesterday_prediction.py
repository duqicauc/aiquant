#!/usr/bin/env python3
"""验证昨天(4/24)预测的股票今天(4/27)实际表现"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log

# 昨天(4/24)预测的Top10
codes = [
    "300637.SZ", "600119.SH", "300077.SZ", "300013.SZ", "600658.SH",
    "002535.SZ", "300831.SZ", "002642.SZ", "300135.SZ", "002079.SZ"
]
names = {
    "300637.SZ": "扬帆新材", "600119.SH": "长江投资", "300077.SZ": "国民技术",
    "300013.SZ": "新宁物流", "600658.SH": "电子城", "002535.SZ": "林州重机",
    "300831.SZ": "派瑞股份", "002642.SZ": "荣联科技", "300135.SZ": "宝利国际",
    "002079.SZ": "苏州固锝"
}

fetcher = TushareFetcher()

# 获取4/27实际数据
df = fetcher.pro.daily(trade_date='20260427')
df = df[df['ts_code'].isin(codes)].copy()
df['name'] = df['ts_code'].map(names)
df = df.sort_values('pct_chg', ascending=False)

print("=" * 80)
print("昨天(4/24)预测 → 今天(4/27)实际表现")
print("=" * 80)
print(f"{'代码':<12} {'名称':<8} {'4/27收盘':>10} {'4/27涨幅%':>10} {'4/27换手%':>10}")
print("-" * 60)

total_return = 0
win_count = 0
for _, row in df.iterrows():
    ts_code = row['ts_code']
    name = row['name']
    close = row['close']
    chg = row['pct_chg']
    turnover = row.get('vol', 0) / row.get('amount', 1) * 100 if 'vol' in row else 0
    # 用daily_basic获取换手率
    print(f"{ts_code:<12} {name:<8} {close:>10.2f} {chg:>+9.2f}%", end="")
    total_return += chg
    if chg > 0:
        win_count += 1
        print("  ✅")
    else:
        print("  ❌")

# 获取换手率
db = fetcher.pro.daily_basic(trade_date='20260427')
if db is not None and not db.empty:
    db = db[db['ts_code'].isin(codes)][['ts_code','turnover_rate']]
    print("\n换手率:")
    for _, row in db.iterrows():
        print(f"  {row['ts_code']}: {row['turnover_rate']:.2f}%")

print("\n" + "=" * 80)
print("汇总统计")
print("=" * 80)
print(f"股票数量: {len(df)}")
print(f"上涨: {win_count} 只 | 下跌: {len(df)-win_count} 只")
print(f"胜率: {win_count/len(df)*100:.1f}%")
print(f"平均涨幅: {df['pct_chg'].mean():+.2f}%")
print(f"总涨幅(等权): {df['pct_chg'].sum():+.2f}%")
print(f"最大盈利: {df['pct_chg'].max():+.2f}%")
print(f"最大亏损: {df['pct_chg'].min():+.2f}%")

# 与大盘对比
df_index = fetcher.pro.index_daily(ts_code='000001.SH', trade_date='20260427')
if df_index is not None and not df_index.empty:
    index_chg = df_index.iloc[0]['pct_chg']
    print(f"\n上证指数: {index_chg:+.2f}%")
    alpha = df['pct_chg'].mean() - index_chg
    print(f"组合超额: {alpha:+.2f}%")
