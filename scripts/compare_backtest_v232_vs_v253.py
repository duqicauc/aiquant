#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同区间、同规则下对比 v232 单模型 与 v253 单模型 回测结果。

依赖：
- v232：backtest_v232_only_report_{start}_{end}.md 或 daily/operations CSV
- v253：backtest_v253_only_report_{start}_{end}.md 或 daily/operations CSV

使用步骤：
1. 生成 v253 预测：python scripts/run_predict_v253_batch.py --start-date 20260105 --end-date 20260129
2. 运行 v253 回测：python scripts/backtest_v253_only.py --start-date 20260105 --end-date 20260129
3. 本脚本对比：python scripts/compare_backtest_v232_vs_v253.py --start-date 20260105 --end-date 20260129
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

RESULTS_DIR = PROJECT_ROOT / 'data' / 'prediction' / 'results'


def load_daily_metrics(prefix: str, start_date: str, end_date: str):
    """从 daily CSV 读取资金曲线并计算收益率、最大回撤。"""
    path = RESULTS_DIR / f"backtest_{prefix}_daily_{start_date}_{end_date}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    if df.empty or 'total_assets' not in df.columns:
        return None
    df['cummax'] = df['total_assets'].cummax()
    df['drawdown_pct'] = (df['total_assets'] - df['cummax']) / df['cummax'] * 100
    max_dd = df['drawdown_pct'].min()
    max_dd_date = df.loc[df['drawdown_pct'].idxmin(), 'date'] if 'date' in df.columns else None
    final_assets = float(df['total_assets'].iloc[-1])
    if 'total_return_pct' in df.columns and len(df) > 0:
        final_return_pct = float(df['total_return_pct'].iloc[-1])
    else:
        initial_cash = 10_000_000.0
        final_return_pct = (final_assets - initial_cash) / initial_cash * 100
    return {
        'final_assets': final_assets,
        'final_return_pct': final_return_pct,
        'max_drawdown_pct': max_dd,
        'max_drawdown_date': max_dd_date,
        'daily_count': len(df),
    }


def load_operations_metrics(prefix: str, start_date: str, end_date: str):
    """从 operations CSV 读取买卖次数、胜率、平均盈亏。"""
    path = RESULTS_DIR / f"backtest_{prefix}_operations_{start_date}_{end_date}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding='utf-8-sig')
    if df.empty:
        return {}
    buys = df[df['operation'] == '买入']
    sells = df[df['operation'] == '卖出']
    total_buys = len(buys)
    total_sells = len(sells)
    if total_sells == 0:
        return {'total_buys': total_buys, 'total_sells': 0, 'win_trades': 0, 'loss_trades': 0,
                'win_rate_pct': 0, 'avg_profit': 0, 'avg_profit_pct': 0}
    win_trades = len(sells[sells['profit'] > 0])
    loss_trades = len(sells[sells['profit'] <= 0])
    win_rate_pct = win_trades / total_sells * 100
    avg_profit = sells['profit'].mean()
    avg_profit_pct = sells['profit_pct'].mean() if 'profit_pct' in sells.columns else 0
    return {
        'total_buys': total_buys,
        'total_sells': total_sells,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate_pct': win_rate_pct,
        'avg_profit': avg_profit,
        'avg_profit_pct': avg_profit_pct,
    }


def main():
    parser = argparse.ArgumentParser(description='对比 v232 与 v253 同区间回测结果')
    parser.add_argument('--start-date', type=str, default='20260105', help='开始日期 YYYYMMDD')
    parser.add_argument('--end-date', type=str, default='20260129', help='结束日期 YYYYMMDD')
    parser.add_argument('--output', type=str, default=None, help='输出对比报告路径（默认 results/backtest_v232_vs_v253_{start}_{end}.md）')
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    metrics_v232_d = load_daily_metrics('v232_only', start_date, end_date)
    metrics_v232_o = load_operations_metrics('v232_only', start_date, end_date)
    metrics_v253_d = load_daily_metrics('v253_only', start_date, end_date)
    metrics_v253_o = load_operations_metrics('v253_only', start_date, end_date)

    if metrics_v232_d is None:
        log.warning("v232 每日回测数据不存在，请先运行: python scripts/backtest_v232_only.py --start-date %s --end-date %s", start_date, end_date)
    if metrics_v253_d is None:
        log.warning("v253 每日回测数据不存在。请先: 1) python scripts/run_predict_v253_batch.py --start-date %s --end-date %s  2) python scripts/backtest_v253_only.py --start-date %s --end-date %s", start_date, end_date, start_date, end_date)

    if metrics_v232_d is None and metrics_v253_d is None:
        log.error("v232 与 v253 均无回测数据，无法对比")
        sys.exit(1)

    # 构建对比表
    rows = []
    if metrics_v232_d:
        rows.append(('v232 单模型', metrics_v232_d, metrics_v232_o))
    if metrics_v253_d:
        rows.append(('v253 单模型', metrics_v253_d, metrics_v253_o))

    log.info("")
    log.info("=" * 80)
    log.info("v232 vs v253 回测对比（同区间、同规则）")
    log.info("=" * 80)
    log.info(f"回测区间: {start_date} - {end_date}")
    log.info("规则: 前一日选股 Top10 开盘买，跌出 Top50 且连续两日收盘<MA5 在 T2 收盘卖")
    log.info("")

    # 打印表格
    headers = ['指标', 'v232 单模型', 'v253 单模型']
    data = []
    if metrics_v232_d and metrics_v253_d:
        data.append(('最终资产(元)', f"{metrics_v232_d['final_assets']:,.0f}", f"{metrics_v253_d['final_assets']:,.0f}"))
        data.append(('收益率(%)', f"{metrics_v232_d['final_return_pct']:+.2f}", f"{metrics_v253_d['final_return_pct']:+.2f}"))
        data.append(('最大回撤(%)', f"{metrics_v232_d['max_drawdown_pct']:.2f}", f"{metrics_v253_d['max_drawdown_pct']:.2f}"))
        data.append(('最大回撤日期', str(metrics_v232_d.get('max_drawdown_date') or '-'), str(metrics_v253_d.get('max_drawdown_date') or '-')))
    if metrics_v232_o and metrics_v253_o:
        data.append(('买入次数', str(metrics_v232_o.get('total_buys', '-')), str(metrics_v253_o.get('total_buys', '-'))))
        data.append(('卖出次数', str(metrics_v232_o.get('total_sells', '-')), str(metrics_v253_o.get('total_sells', '-'))))
        data.append(('胜率(%)', f"{metrics_v232_o.get('win_rate_pct', 0):.2f}", f"{metrics_v253_o.get('win_rate_pct', 0):.2f}"))
        data.append(('平均每笔盈亏(元)', f"{metrics_v232_o.get('avg_profit', 0):+,.0f}", f"{metrics_v253_o.get('avg_profit', 0):+,.0f}"))
        data.append(('平均每笔盈亏(%)', f"{metrics_v232_o.get('avg_profit_pct', 0):+.2f}", f"{metrics_v253_o.get('avg_profit_pct', 0):+.2f}"))

    for row in data:
        log.info(f"  {row[0]:<20} {row[1]:<20} {row[2]:<20}")

    out_path = Path(args.output) if args.output else RESULTS_DIR / f"backtest_v232_vs_v253_{start_date}_{end_date}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# v232 单模型 vs v253 单模型 回测对比\n\n")
        f.write(f"**回测区间**: {start_date} - {end_date}\n\n")
        f.write("**规则**: 前一日选股 Top10 当日开盘买，跌出 Top50 且连续两日收盘<MA5 在 T2 收盘卖；30万/只，先买后卖。\n\n")
        f.write("| 指标 | v232 单模型 | v253 单模型 |\n")
        f.write("|------|-------------|-------------|\n")
        for row in data:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        f.write("\n- v232 选股数据: `v2.3.2_full_{date}.csv`\n")
        f.write("- v253 选股数据: `v2.5.3_full_{date}.csv`\n")

    log.success(f"\n✓ 对比报告已保存: {out_path}")


if __name__ == '__main__':
    main()
