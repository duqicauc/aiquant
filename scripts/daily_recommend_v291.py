#!/usr/bin/env python3
"""v2.9.1-ensemble Integrated 策略每日推荐"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.trading.sector_filter import SectorFilter
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log

PREDICTION_DATE = "20260427"
PRED_DIR = PROJECT_ROOT / "data" / "prediction" / "v291_stk_factor"

def main():
    # 1. 读取预测结果
    pred_file = PRED_DIR / f"predictions_{PREDICTION_DATE}_all.csv"
    df_pred = pd.read_csv(pred_file)
    log.info(f"预测数据: {len(df_pred)} 只股票")

    # 2. Integrated 策略筛选
    sector_filter = SectorFilter()
    market_state = "weak_bull"  # 默认
    df = sector_filter.filter_hot_stocks(df_pred, PREDICTION_DATE, market_state)

    # 3. 获取股票名称
    fetcher = TushareFetcher()
    df_basic = fetcher.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    name_map = dict(zip(df_basic['ts_code'], df_basic['name']))
    industry_map = dict(zip(df_basic['ts_code'], df_basic['industry']))
    df['name'] = df['ts_code'].map(name_map)
    df['industry'] = df['ts_code'].map(industry_map)

    # 4. 输出 Top15
    log.info("\n" + "=" * 100)
    log.info(f"v2.9.1-ensemble Integrated | {PREDICTION_DATE} | 明天(4/28)操作建议")
    log.info("=" * 100)
    log.info(f"{'排名':<4} {'代码':<12} {'名称':<8} {'行业':<10} {'prob':>8} {'boost':>6} {'adjusted':>10} {'市值(亿)':>8} {'涨幅%':>8} {'换手%':>8}")
    log.info("-" * 100)

    for i, (_, row) in enumerate(df.head(15).iterrows(), 1):
        ts_code = row['ts_code']
        name = str(row.get('name', ''))[:6]
        industry = str(row.get('industry', ''))[:8]
        prob = row.get('prob', 0)
        boost = row.get('sector_boost', 1.0)
        adj = row.get('adjusted_score', prob)
        mv = row.get('total_mv', 0) / 10000
        chg = row.get('pct_chg', 0)
        turnover = row.get('turnover_rate', 0)
        log.info(f"{i:<4} {ts_code:<12} {name:<8} {industry:<10} {prob:>8.4f} {boost:>6.2f} {adj:>10.4f} {mv:>8.1f} {chg:>8.2f} {turnover:>8.2f}")

    # 5. 汇总分析
    top10 = df.head(10)
    log.info("\n" + "=" * 100)
    log.info("组合特征")
    log.info("=" * 100)
    log.info(f"平均市值: {top10['total_mv'].mean()/10000:.1f}亿")
    log.info(f"平均涨幅: {top10['pct_chg'].mean():.2f}%")
    log.info(f"平均换手: {top10['turnover_rate'].mean():.2f}%")
    log.info(f"概率区间: {top10['prob'].min():.4f} ~ {top10['prob'].max():.4f}")
    log.info(f"板块加成: {top10['sector_boost'].min():.2f} ~ {top10['sector_boost'].max():.2f}")

    # 涨停股提示
    high_chg = top10[top10['pct_chg'] > 9.0]
    if len(high_chg) > 0:
        log.warning(f"\n⚠️ 涨停股 {len(high_chg)} 只: {', '.join(high_chg['ts_code'].tolist())}")

    log.info("\n操作要点:")
    log.info("1. 4/28(周二)开盘买入 Top10")
    log.info("2. 每只股票固定30万（或按资金比例）")
    log.info("3. 止损: -4% | 移动止盈: 峰值回撤5%")
    log.info("4. 跌出Top50则次日开盘卖出")

    # 保存
    out_dir = PROJECT_ROOT / "data" / "prediction" / "v291_integrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ['rank','ts_code','name','industry','prob','prob_raw','sector_boost','adjusted_score',
            'close','pct_chg','turnover_rate','total_mv']
    cols = [c for c in cols if c in df.columns]
    df[cols].head(50).to_csv(out_dir / f"predictions_{PREDICTION_DATE}_integrated_top50.csv", index=False)
    log.info(f"\n已保存: {out_dir}/predictions_{PREDICTION_DATE}_integrated_top50.csv")

if __name__ == "__main__":
    main()
