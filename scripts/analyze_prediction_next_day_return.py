#!/usr/bin/env python3
"""
分析 v291-ensemble integrated 策略的"预测次日收益"

从回测交易记录中，统计买入后第1天/第2天/第N天的收益分布
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 读取交易记录
tx_file = PROJECT_ROOT / "data" / "prediction" / "evaluation" / "v291_integrated_2024q4" / "backtest_transactions.csv"
df = pd.read_csv(tx_file)

# 分离买入和卖出
buys = df[df['action'] == 'BUY'].copy()
sells = df[df['action'] == 'SELL'].copy()

# 按股票配对买卖记录
trades = []
for _, buy in buys.iterrows():
    # 找到同一只股票的下一次卖出
    sell = sells[(sells['ts_code'] == buy['ts_code']) & (sells['date'] >= buy['date'])].head(1)
    if not sell.empty:
        sell = sell.iloc[0]
        hold_days = (datetime.strptime(str(sell['date']), "%Y%m%d") - datetime.strptime(str(buy['date']), "%Y%m%d")).days

        buy_price = buy['price']
        sell_price = sell['price']
        return_pct = (sell_price - buy_price) / buy_price * 100

        trades.append({
            'ts_code': buy['ts_code'],
            'buy_date': buy['date'],
            'sell_date': sell['date'],
            'hold_days': hold_days,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'return_pct': return_pct,
            'sell_reason': sell['reason'],
            'qty': buy['qty'],
        })

df_trades = pd.DataFrame(trades)

print("=" * 80)
print("v291-ensemble Integrated 策略 - 交易持有期分析 (2024Q4)")
print("=" * 80)

# 总体统计
total = len(df_trades)
win = len(df_trades[df_trades['return_pct'] > 0])
loss = len(df_trades[df_trades['return_pct'] <= 0])
win_rate = win / total * 100 if total > 0 else 0
avg_return = df_trades['return_pct'].mean()
avg_win = df_trades[df_trades['return_pct'] > 0]['return_pct'].mean() if win > 0 else 0
avg_loss = df_trades[df_trades['return_pct'] <= 0]['return_pct'].mean() if loss > 0 else 0

print(f"\n总体统计:")
print(f"  总交易次数: {total}")
print(f"  胜率: {win_rate:.1f}% ({win} 胜 / {loss} 负)")
print(f"  平均收益: {avg_return:.2f}%")
print(f"  平均盈利: +{avg_win:.2f}%")
print(f"  平均亏损: {avg_loss:.2f}%")

# 按持有天数分组
print(f"\n按持有天数分组:")
print(f"{'持有天数':<8} {'次数':>6} {'胜率':>8} {'平均收益':>10} {'最大盈利':>10} {'最大亏损':>10}")
print("-" * 60)

for days in sorted(df_trades['hold_days'].unique()):
    subset = df_trades[df_trades['hold_days'] == days]
    n = len(subset)
    wr = len(subset[subset['return_pct'] > 0]) / n * 100 if n > 0 else 0
    avg = subset['return_pct'].mean()
    max_win = subset['return_pct'].max()
    max_loss = subset['return_pct'].min()
    print(f"{days:<8} {n:>6} {wr:>7.1f}% {avg:>+9.2f}% {max_win:>+9.2f}% {max_loss:>+9.2f}%")

# 次日收益详细分析（持有1天）
next_day = df_trades[df_trades['hold_days'] == 1]
print(f"\n次日卖出详细分析 (持有1天, 共 {len(next_day)} 次):")
print(f"  胜率: {len(next_day[next_day['return_pct'] > 0]) / len(next_day) * 100:.1f}%")
print(f"  平均收益: {next_day['return_pct'].mean():+.2f}%")
print(f"  收益中位数: {next_day['return_pct'].median():+.2f}%")
print(f"  最大盈利: {next_day['return_pct'].max():+.2f}%")
print(f"  最大亏损: {next_day['return_pct'].min():+.2f}%")

# 卖出原因分析
print(f"\n卖出原因分析:")
reason_stats = df_trades.groupby('sell_reason').agg({
    'return_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100]
}).round(2)
reason_stats.columns = ['次数', '平均收益', '胜率']
print(reason_stats)

# 按日期聚合：每日买入的次日平均收益
print(f"\n每日 Top10 买入的次日表现:")
daily_next_day = next_day.groupby('buy_date')['return_pct'].agg(['count', 'mean', 'sum']).round(2)
daily_next_day.columns = ['买入数', '次日平均收益', '次日总收益']
daily_next_day = daily_next_day.sort_values('次日平均收益', ascending=False)
print(daily_next_day.head(10))

print(f"\n次日收益分布 (分位数):")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"  {int(q*100)}%分位: {next_day['return_pct'].quantile(q):+.2f}%")
