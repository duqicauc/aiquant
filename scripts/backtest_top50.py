#!/usr/bin/env python3
"""AIQuant Top50 历史预测回测统计"""
import sys, os, json
sys.path.insert(0, '/app')
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
PRED_DIR = Path('/app/data/prediction/v3.0.0')

def main():
    # 1. 加载 Top50 预测
    records = []
    files = sorted(PRED_DIR.glob('predictions_*_top50.csv'))
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        df['date'] = f.name.replace('predictions_', '').replace('_top50.csv', '')
        records.append(df)
    preds = pd.concat(records, ignore_index=True)
    logger.info(f"加载 {len(files)} 个文件, {len(preds)} 条预测")

    # 2. 加载 ArcticDB 价格
    ap = ArcticDataProvider()
    arctic = ap.read_daily_ohlcv()
    arctic['date_str'] = arctic['trade_date'].astype(str).str[:8]
    logger.info(f"ArcticDB: {len(arctic)} 行, 日期 {arctic.date_str.min()} ~ {arctic.date_str.max()}")

    # 3. 构建价格映射
    price_map = {}
    for code, grp in arctic.groupby('ts_code'):
        price_map[code] = dict(zip(grp['date_str'], grp['close']))

    all_dates = sorted(arctic['date_str'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    logger.info(f"交易日: {len(all_dates)} 天, 股票: {len(price_map)} 只")

    # 4. 计算各股各期收益
    horizons = {'1d': 1, '5d': 5, '10d': 10, '20d': 20}
    results = []
    for _, row in preds.iterrows():
        code, pd_date = row['ts_code'], str(row['date'])
        prob = row['prob']

        if code not in price_map or pd_date not in price_map[code]:
            continue
        pred_close = price_map[code][pd_date]
        pred_idx = date_to_idx.get(pd_date)
        if pred_idx is None:
            continue

        r = {'ts_code': code, 'name': row.get('name',''), 'pred_date': pd_date,
             'prob': prob, 'pred_close': pred_close}
        for hn, hd in horizons.items():
            ti = pred_idx + hd
            if ti < len(all_dates):
                td = all_dates[ti]
                if td in price_map[code]:
                    tc = price_map[code][td]
                    ret = (tc - pred_close) / pred_close * 100
                    r[f'ret_{hn}'] = round(ret, 2)
                    r[f'win_{hn}'] = 1 if ret > 0 else 0
                else:
                    r[f'ret_{hn}'] = None
                    r[f'win_{hn}'] = None
            else:
                r[f'ret_{hn}'] = None
                r[f'win_{hn}'] = None
        results.append(r)

    df = pd.DataFrame(results)
    logger.info(f"回测记录: {len(df)} 条")

    # 5. 按日频次计算等权重组合收益
    daily_portfolio = df.groupby('pred_date').agg({
        'ret_1d': 'mean', 'ret_5d': 'mean', 'ret_10d': 'mean', 'ret_20d': 'mean',
        'win_1d': 'mean', 'win_5d': 'mean', 'win_10d': 'mean', 'win_20d': 'mean',
    }).sort_index().reset_index()

    # 6. 输出统计
    print("=" * 70)
    print(" AIQuant Top50 预测回测统计")
    print(f" 回测区间: {preds['date'].min()} ~ {preds['date'].max()}")
    print(f" 交易日数: {daily_portfolio['pred_date'].nunique()}")
    print("=" * 70)

    for hn in ['1d', '5d', '10d', '20d']:
        col_ret = f'ret_{hn}'
        col_win = f'win_{hn}'
        valid_ret = df[df[col_ret].notna()][col_ret]
        valid_win = df[df[col_win].notna()][col_win]

        daily_ret = daily_portfolio[daily_portfolio[col_ret].notna()][col_ret]
        daily_win = daily_portfolio[daily_portfolio[col_win].notna()][col_win]

        print(f"\n{'─' * 40}")
        print(f"  【{hn} 持有期】")
        print(f"{'─' * 40}")

        # 单只股票统计
        print(f"  [按股票维度]")
        print(f"    样本数: {len(valid_ret)}")
        print(f"    胜率: {valid_win.mean()*100:.1f}%")
        print(f"    平均收益: {valid_ret.mean():+.2f}%")
        print(f"    收益中位数: {valid_ret.median():+.2f}%")
        print(f"    收益标准差: {valid_ret.std():.2f}%")
        print(f"    最大单次收益: {valid_ret.max():+.2f}%")
        print(f"    最大单次亏损: {valid_ret.min():+.2f}%")

        # 组合统计（每日等权重）
        cum_ret = (1 + daily_ret/100).prod() - 1
        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() > 0 else 0
        rolling_max = (1 + daily_ret/100).cummax()
        drawdown = ((1 + daily_ret/100) / rolling_max - 1)
        max_dd = drawdown.min()
        print(f"\n  [组合维度 - 每日等权重Top50买入]")
        print(f"    交易日数: {len(daily_ret)}")
        print(f"    组合胜率: {daily_win.mean()*100:.1f}%")
        print(f"    日均收益: {daily_ret.mean():+.3f}%")
        print(f"    累计收益: {cum_ret*100:+.1f}%")
        print(f"    年化夏普: {sharpe:.2f}")
        print(f"    最大回撤: {max_dd*100:.1f}%")
        print(f"    收益/最大回撤: {(daily_ret.mean()*len(daily_ret))/(-max_dd):.2f}")

    print("\n" + "=" * 70)
    print("  【对照：同期沪深300表现】")
    # 同期沪深300
    sh300 = price_map.get('000300.SH', {})
    sh300_dates = sorted(sh300.keys())
    start_close = sh300.get(daily_portfolio['pred_date'].min())
    end_date = daily_portfolio['pred_date'].max()
    sh300_end_dates = [d for d in sh300_dates if d >= end_date]
    end_close = sh300.get(sh300_end_dates[0]) if sh300_end_dates else None
    if start_close and end_close:
        sh300_ret = (end_close - start_close) / start_close * 100
        print(f"    区间: {daily_portfolio['pred_date'].min()} ~ {end_date}")
        print(f"    沪深300起始: {start_close:.2f}")
        print(f"    沪深300结束: {end_close:.2f}")
        print(f"    涨幅: {sh300_ret:+.2f}%")
    print("=" * 70)

    return df, daily_portfolio

if __name__ == '__main__':
    df, dp = main()
