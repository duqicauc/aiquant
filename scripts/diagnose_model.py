#!/usr/bin/env python3
"""AIQuant v3.0.0 模型深度诊断：分板块、分周期、特征归因"""
import sys, os, json
sys.path.insert(0, '/app')
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import arcticdb as adb
from src.data.arctic_provider import ArcticDataProvider
from collections import defaultdict

PRED_DIR = Path('/app/data/prediction/v3.0.0')

def load_prices_and_predictions():
    """加载回测数据"""
    records = []
    for f in sorted(PRED_DIR.glob('predictions_*_top50.csv')):
        df = pd.read_csv(f)
        if df.empty: continue
        df['date'] = f.name.replace('predictions_', '').replace('_top50.csv', '')
        records.append(df)
    preds = pd.concat(records, ignore_index=True)

    ap = ArcticDataProvider()
    arctic = ap.read_daily_ohlcv()
    arctic['date_str'] = arctic['trade_date'].astype(str).str[:8]
    price_map = {}
    for code, grp in arctic.groupby('ts_code'):
        price_map[code] = dict(zip(grp['date_str'], grp['close']))
    all_dates = sorted(arctic['date_str'].unique())
    date_to_idx = {d:i for i,d in enumerate(all_dates)}
    return preds, price_map, all_dates, date_to_idx

def load_industry():
    """加载行业分类"""
    conn = sqlite3.connect('/app/data/cache/quant_data.db')
    df = pd.read_sql("SELECT ts_code, name, industry, market FROM stock_basic", conn)
    conn.close()
    return df.set_index('ts_code')

def load_model_features():
    """加载模型特征重要性"""
    path = '/app/data/models/breakout_launch_scorer/versions/v3.0.0/xgb_flat_final.json'
    with open(path) as f:
        model_json = json.load(f)
    
    # 从XGBoost dump中提取特征使用频率
    feature_counts = defaultdict(int)
    feature_names = set()
    
    def extract_features(node_str):
        """递归提取特征名"""
        for line in node_str.split('\n'):
            if '[' in line and ']' in line:
                bracket = line[line.index('[')+1:line.index(']')]
                if '<' in bracket:
                    feat = bracket.split('<')[0].strip()
                    feature_names.add(feat)
                    feature_counts[feat] += 1
    
    if 'learner' in model_json:
        learner = model_json['learner']
        if 'gradient_booster' in learner:
            booster = learner['gradient_booster']
            if 'model' in booster and 'gbtree_model_param' in booster['model']:
                # XGBoost JSON format
                trees = booster['model'].get('trees', [])
                for tree in trees:
                    if 'split_conditions' in tree:
                        # Count feature splits
                        for i in range(len(tree.get('split_indices', []))):
                            feat_idx = tree['split_indices'][i]
                            feature_names.add(f'f{feat_idx}')
                            feature_counts[f'f{feat_idx}'] += 1
                    elif 'tree_param' in tree:
                        # Another format
                        pass
    
    return feature_counts, feature_names

def main():
    print("=" * 90)
    print("  AIQuant v3.0.0 模型深度诊断")
    print("=" * 90)

    # 1. 加载数据
    preds, price_map, all_dates, date_to_idx = load_prices_and_predictions()
    industry_df = load_industry()

    # 计算1d收益
    col = 'ret_1d'
    results = []
    for _, row in preds.iterrows():
        code, pd_date, prob = row['ts_code'], str(row['date']), row['prob']
        if code not in price_map or pd_date not in price_map[code]:
            continue
        buy_close = price_map[code][pd_date]
        pred_idx = date_to_idx.get(pd_date)
        if pred_idx is None or pred_idx+1 >= len(all_dates):
            continue
        sell_date = all_dates[pred_idx+1]
        if sell_date not in price_map[code]:
            continue
        sell_close = price_map[code][sell_date]
        ret = (sell_close / buy_close - 1) * 100
        
        # 行业信息
        info = industry_df.loc[code] if code in industry_df.index else None
        industry = info['industry'] if info is not None else '未知'
        market = info['market'] if info is not None else '未知'
        
        results.append({
            'ts_code': code, 'name': row.get('name',''), 'pred_date': pd_date,
            'prob': prob, 'ret_1d': ret, 'industry': industry, 'market': market,
            'month': pd_date[:6],
        })

    df = pd.DataFrame(results)
    print(f"\n总回测记录: {len(df)}")

    # ============ A. 板块分析 ============
    print("\n\n" + "=" * 90)
    print("  【A. 分板块表现 — 1日收益】")
    print("=" * 90)

    print(f"\n  {'行业':<16} {'样本':<8} {'平均收益':<10} {'中位数':<10} {'胜率':<8} {'盈亏比':<8} {'夏普':<8} {'>5%':<8} {'<-5%':<8}")
    print(f"  {'─'*16} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for ind, grp in df.groupby('industry'):
        rets = grp['ret_1d'].values
        n = len(rets)
        avg_r = np.mean(rets)
        med_r = np.median(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins)/n*100
        plr = (np.mean(wins)/abs(np.mean(losses))) if len(losses) > 0 and abs(np.mean(losses)) > 0 else float('inf')
        sharpe = avg_r/np.std(rets,ddof=1)*(252**0.5) if np.std(rets,ddof=1)>0 else 0
        pct5 = np.sum(rets > 5)/n*100
        lm5 = np.sum(rets < -5)/n*100
        print(f"  {ind:<16} {n:<8} {avg_r:<+8.2f}%  {med_r:<+8.2f}%  {wr:<7.1f}% {plr:<7.2f}  {sharpe:<+7.2f}  {pct5:<7.1f}% {lm5:<7.1f}%")

    # ============ B. 分市场周期 ============
    print("\n\n" + "=" * 90)
    print("  【B. 分市场周期 — 1日收益】")
    print("=" * 90)

    df['period'] = df['month'].map({
        '202601': '1月(震荡)',
        '202602': '2月(偏强)',
        '202603': '3月(调整)',
        '202604': '4月(反弹)',
        '202605': '5月(震荡)',
    }).fillna('其他')

    print(f"\n  {'周期':<14} {'样本':<8} {'平均收益':<10} {'中位数':<10} {'胜率':<8} {'盈亏比':<8} {'年化夏普':<10} {'合计收益':<10}")
    print(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    for period, grp in df.groupby('period'):
        rets = grp['ret_1d'].values
        n = len(rets)
        avg = np.mean(rets)
        med = np.median(rets)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins)/n*100
        plr = (np.mean(wins)/abs(np.mean(losses))) if len(losses) > 0 and abs(np.mean(losses)) > 0 else float('inf')
        sharpe = avg/np.std(rets,ddof=1)*(252**0.5) if np.std(rets,ddof=1)>0 else 0
        total = np.sum(rets)
        print(f"  {period:<14} {n:<8} {avg:<+8.2f}%  {med:<+8.2f}%  {wr:<7.1f}% {plr:<7.2f}  {sharpe:<+9.2f}  {total:<+8.1f}%")

    # ============ C. 特征归因 ============
    print("\n\n" + "=" * 90)
    print("  【C. 特征归因 — 模型驱动因素分析】")
    print("=" * 90)

    feature_counts, _ = load_model_features()
    sorted_feats = sorted(feature_counts.items(), key=lambda x: -x[1])

    print(f"\n  模型总特征: {len(feature_counts)} 个\n")
    print(f"  Top30 高频特征（按分裂次数排序）:")
    print(f"  {'#':<4} {'特征ID':<16} {'分裂次数':<10} {'占比':<10}")
    print(f"  {'─'*4} {'─'*16} {'─'*10} {'─'*10}")

    total_splits = sum(v for k,v in sorted_feats)
    for i, (feat, count) in enumerate(sorted_feats[:30]):
        pct = count/total_splits*100
        print(f"  {i+1:<4} {feat:<16} {count:<10} {pct:<+9.2f}%")

    # Top 10 vs Rest
    top10_count = sum(v for k,v in sorted_feats[:10])
    rest_count = total_splits - top10_count
    print(f"\n  Top10 特征合计使用: {top10_count}/{total_splits} ({top10_count/total_splits*100:.1f}%)")

    # 分析概率分布的集中度
    print(f"\n\n  【概率分布诊断】")
    for month in sorted(df['month'].unique()):
        sub = df[df['month'] == month]
        print(f"    {month}: 平均概率 {sub['prob'].mean():.4f}  "
              f"Top1概率 {sub['prob'].max():.4f}  "
              f"Top50门槛 {sub['prob'].nsmallest(50).max():.4f}")

    # ============ D. 股票重复率 ============
    print(f"\n\n  【D. 选股偏好 — 连续入榜统计】")
    print("=" * 90)

    # 看哪些股票频繁出现在Top50
    stock_freq = df.groupby('ts_code').agg(
        count=('pred_date', 'count'),
        avg_ret=('ret_1d', 'mean'),
        avg_prob=('prob', 'mean'),
    ).sort_values('count', ascending=False)

    print(f"\n  日均入榜 Top20:")
    print(f"  {'股票代码':<12} {'入榜天数':<10} {'平均收益':<12} {'平均概率':<12}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*12}")
    for code, row in stock_freq.head(20).iterrows():
        name = industry_df.loc[code, 'name'] if code in industry_df.index else ''
        industry = industry_df.loc[code, 'industry'] if code in industry_df.index else ''
        print(f"  {code:<8} {name:<4} {row['count']:<10} {row['avg_ret']:<+10.2f}%  {row['avg_prob']:<.4f}")

    # ============ E. 板块集中度 ============
    print(f"\n\n  【E. 行业集中度 — 每日Top50的行业分布】")
    print("=" * 90)

    # 每天各行业出现次数
    daily_industry = df.groupby(['pred_date', 'industry']).size().reset_index(name='count')
    top_industries = daily_industry.groupby('industry')['count'].mean().sort_values(ascending=False)

    print(f"\n  平均每日各行业出镜率 Top15:")
    for ind, avg_cnt in top_industries.head(15).items():
        print(f"    {ind:<16} 日均 {avg_cnt:.1f} 只 /50")

    # HHI指数（行业集中度）
    hhi_by_day = []
    for date, grp in daily_industry.groupby('pred_date'):
        shares = grp['count'].values / 50
        hhi = np.sum(shares**2)
        hhi_by_day.append(hhi)
    print(f"\n  行业HHI（赫芬达尔指数，值越大多样性越低）:")
    print(f"    平均: {np.mean(hhi_by_day):.4f}  |  最小: {np.min(hhi_by_day):.4f}  |  最大: {np.max(hhi_by_day):.4f}")
    print(f"    解读: HHI<0.1=高度分散  0.1-0.15=中度集中  >0.15=高度集中")

    # ============ 汇总 ============
    print("\n\n" + "=" * 90)
    print("  【诊断结论】")
    print("=" * 90)

    # 找出最佳和最差的板块
    best_industry = df.groupby('industry')['ret_1d'].mean().sort_values(ascending=False)
    worst_industry = df.groupby('industry')['ret_1d'].mean().sort_values()

    print(f"\n  1. 模型表现最好的板块:")
    for ind, ret in best_industry.head(5).items():
        grp = df[df['industry'] == ind]
        wr = np.mean(grp['ret_1d'].values > 0) * 100
        print(f"     {ind:<16} +{ret:.2f}% avg, 胜率{wr:.0f}%")

    print(f"\n  2. 模型表现最差的板块:")
    for ind, ret in worst_industry.head(5).items():
        grp = df[df['industry'] == ind]
        wr = np.mean(grp['ret_1d'].values > 0) * 100
        print(f"     {ind:<16} {ret:.2f}% avg, 胜率{wr:.0f}%")

    # 稳定性分析
    monthly_sharpe = []
    for month, grp in df.groupby('month'):
        rets = grp['ret_1d'].values
        sh = np.mean(rets)/np.std(rets,ddof=1)*(21**0.5) if np.std(rets,ddof=1)>0 else 0
        monthly_sharpe.append(sh)
    sharpe_std = np.std(monthly_sharpe, ddof=1)
    print(f"\n  3. 月度夏普稳定性:")
    print(f"     月度夏普波动: {sharpe_std:.2f} {'(稳定) ' if sharpe_std < 0.3 else '(不稳定)' if sharpe_std < 1 else '(极度不稳定)'}")

    # 行业集中度诊断
    if np.mean(hhi_by_day) < 0.1:
        conc_verdict = "高度分散，行业配置中性"
    elif np.mean(hhi_by_day) < 0.15:
        conc_verdict = "中度集中，有行业偏好但不显著"
    else:
        conc_verdict = "高度集中，模型有强烈行业偏好，可能只是行业因子"
    print(f"  4. 行业集中度: HHI={np.mean(hhi_by_day):.4f} → {conc_verdict}")

    # Top特征分析
    if sorted_feats:
        top1_pct = sorted_feats[0][1]/total_splits*100
        top5_pct = sum(v for k,v in sorted_feats[:5])/total_splits*100
        print(f"  5. 特征分布: Top1特征占比{top1_pct:.1f}%, Top5合计占比{top5_pct:.1f}%")
        if top5_pct > 60:
            print(f"     ⚠️ 特征过于集中，模型依赖少数几个特征，远不如声称的5882维")
        elif top5_pct < 20:
            print(f"     ✅ 特征分布均匀，5882维有实际贡献")

    print()

if __name__ == '__main__':
    main()
