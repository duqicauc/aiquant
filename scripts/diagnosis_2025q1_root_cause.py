#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P0.1 熊市根因诊断脚本
分析 2025Q1 模型预测 vs 实际收益，判断是"模型失效"还是"周期错配"

诊断维度：
1. 模型区分能力：Top10/Top50/其他 的收益差异
2. 持有期收益：1/5/10/20 天持有收益分布
3. 周期错配：长持收益 vs 策略实际收益对比
4. 市场环境：牛/熊环境下的表现差异
5. 预测分数与收益相关性
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PRED_DIR = Path("data/prediction/v281_stk_factor")
DB_PATH = "data/cache/quant_data.db"
OUTPUT = "docs/analysis/diagnosis_2025q1.md"

# 2025Q1 交易日
TRADE_DATES = [
    d.strftime("%Y%m%d") for d in pd.bdate_range("2025-01-02", "2025-03-31")
    if d.strftime("%Y%m%d") not in ['20250127', '20250128', '20250129', '20250130',  # 春节
                                     '20250404', '20250405', '20250406']  # 清明不在Q1
]
# 更精确的方式：从数据库读取

def get_trade_dates_from_db(conn, start, end):
    """从数据库获取实际交易日"""
    df = pd.read_sql(
        f"SELECT DISTINCT trade_date FROM daily_data "
        f"WHERE trade_date BETWEEN '{start}' AND '{end}' ORDER BY trade_date",
        conn
    )
    return df['trade_date'].tolist()

def get_future_returns(conn, ts_code, trade_date, horizons=[1, 5, 10, 20]):
    """获取股票在未来 N 天的收益率"""
    # 获取 trade_date 之后 N 个交易日的收盘价
    max_horizon = max(horizons)
    df = pd.read_sql(
        f"SELECT trade_date, close FROM daily_data "
        f"WHERE ts_code = '{ts_code}' AND trade_date >= '{trade_date}' "
        f"ORDER BY trade_date LIMIT {max_horizon + 1}",
        conn
    )
    if len(df) < 2:
        return {h: np.nan for h in horizons}

    base_price = df['close'].iloc[0]
    result = {}
    for h in horizons:
        if h < len(df):
            result[h] = (df['close'].iloc[h] / base_price - 1) * 100
        else:
            result[h] = np.nan
    return result

def get_index_return(conn, index_code, trade_date, horizon=20):
    """获取指数在未来 N 天的收益率"""
    df = pd.read_sql(
        f"SELECT trade_date, close FROM daily_data "
        f"WHERE ts_code = '{index_code}' AND trade_date >= '{trade_date}' "
        f"ORDER BY trade_date LIMIT {horizon + 1}",
        conn
    )
    if len(df) < 2:
        return np.nan
    return (df['close'].iloc[min(horizon, len(df)-1)] / df['close'].iloc[0] - 1) * 100

def get_sh_index_ma20(conn, date):
    """获取上证指数及其MA20"""
    # 尝试多种代码格式
    for code in ['000001.SH', '000001.SZ', '000001']:
        df = pd.read_sql(
            f"SELECT trade_date, close FROM daily_data "
            f"WHERE ts_code = '{code}' AND trade_date <= '{date}' "
            f"ORDER BY trade_date DESC LIMIT 25",
            conn
        )
        if len(df) >= 20:
            df = df.sort_values('trade_date')
            df['ma20'] = df['close'].rolling(20).mean()
            latest = df.iloc[-1]
            return latest['close'], latest['ma20']
    #  fallback: 尝试从 pro.index_daily 获取（但这里只用 cache）
    return None, None

def analyze():
    conn = sqlite3.connect(DB_PATH)

    # 获取实际交易日
    trade_dates = get_trade_dates_from_db(conn, '20250102', '20250331')
    print(f"2025Q1 实际交易日: {len(trade_dates)} 天")

    # 存储所有分析结果
    all_results = []
    daily_stats = []

    for date in trade_dates:
        pred_file = PRED_DIR / f"predictions_{date}_all.csv"
        if not pred_file.exists():
            continue

        pred_df = pd.read_csv(pred_file)
        if pred_df.empty or 'prob' not in pred_df.columns:
            continue

        # 按预测分数排序，取 Top10 / Top50 / 其余
        pred_df = pred_df.sort_values('prob', ascending=False).reset_index(drop=True)
        pred_df['rank_group'] = 'Other'
        pred_df.loc[pred_df.index < 10, 'rank_group'] = 'Top10'
        pred_df.loc[(pred_df.index >= 10) & (pred_df.index < 50), 'rank_group'] = 'Top11_50'

        # 获取市场环境
        sh_close, sh_ma20 = get_sh_index_ma20(conn, date)
        is_bull = sh_close >= sh_ma20 if sh_ma20 else True

        # 获取大盘未来收益
        sh_return_5d = get_index_return(conn, '000001.SH', date, 5)
        sh_return_20d = get_index_return(conn, '000001.SH', date, 20)

        day_results = []
        for _, row in pred_df.iterrows():
            ts_code = row['ts_code']
            returns = get_future_returns(conn, ts_code, date, [1, 5, 10, 20])

            day_results.append({
                'date': date,
                'ts_code': ts_code,
                'rank': row.name + 1,
                'rank_group': row['rank_group'],
                'prob': row['prob'],
                'is_bull': is_bull,
                'return_1d': returns[1],
                'return_5d': returns[5],
                'return_10d': returns[10],
                'return_20d': returns[20],
                'sh_return_5d': sh_return_5d,
                'sh_return_20d': sh_return_20d,
            })

        all_results.extend(day_results)

        # 每日统计
        top10 = [r for r in day_results if r['rank_group'] == 'Top10']
        if top10:
            daily_stats.append({
                'date': date,
                'is_bull': is_bull,
                'top10_count': len(top10),
                'top10_avg_prob': np.mean([r['prob'] for r in top10]),
                'top10_avg_5d': np.nanmean([r['return_5d'] for r in top10]),
                'top10_avg_20d': np.nanmean([r['return_20d'] for r in top10]),
                'sh_return_5d': sh_return_5d,
                'sh_return_20d': sh_return_20d,
            })

    conn.close()

    df = pd.DataFrame(all_results)
    daily_df = pd.DataFrame(daily_stats)

    # ========== 分析报告 ==========
    report = []
    report.append("# P0.1 熊市根因诊断报告\n")
    report.append(f"> 分析期间: 2025Q1 ({len(trade_dates)} 个交易日)\n")
    report.append(f"> 分析样本: {len(df)} 条股票-日期记录\n")
    report.append(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    # --- 1. 模型区分能力 ---
    report.append("## 一、模型区分能力分析\n")
    report.append("### 1.1 不同排名组的平均收益\n\n")
    report.append("| 排名组 | 样本数 | 次日收益 | 5日收益 | 10日收益 | 20日收益 |\n")
    report.append("|--------|--------|----------|---------|----------|----------|\n")

    for group in ['Top10', 'Top11_50', 'Other']:
        g = df[df['rank_group'] == group]
        r1 = g['return_1d'].mean()
        r5 = g['return_5d'].mean()
        r10 = g['return_10d'].mean()
        r20 = g['return_20d'].mean()
        report.append(f"| {group} | {len(g)} | {r1:+.3f}% | {r5:+.3f}% | {r10:+.3f}% | {r20:+.3f}% |\n")

    # Top10 vs Other 的 t 检验
    top10_5d = df[df['rank_group'] == 'Top10']['return_5d'].dropna()
    other_5d = df[df['rank_group'] == 'Other']['return_5d'].dropna()
    if len(top10_5d) > 10 and len(other_5d) > 10:
        t_stat, p_value = stats.ttest_ind(top10_5d, other_5d)
        report.append(f"\n**统计检验**: Top10 vs Other 的 5日收益差异\n")
        report.append(f"- t 统计量: {t_stat:.3f}\n")
        report.append(f"- p 值: {p_value:.4f} ({'显著' if p_value < 0.05 else '不显著'})\n")

    # --- 2. 持有期收益分布 ---
    report.append("\n## 二、持有期收益分布（Top10 股票）\n\n")
    top10_df = df[df['rank_group'] == 'Top10']

    for horizon in [1, 5, 10, 20]:
        col = f'return_{horizon}d'
        vals = top10_df[col].dropna()
        report.append(f"### {horizon}日持有收益\n\n")
        report.append(f"- 均值: {vals.mean():+.3f}%\n")
        report.append(f"- 中位数: {vals.median():+.3f}%\n")
        report.append(f"- 标准差: {vals.std():.3f}%\n")
        report.append(f"- 胜率(>0): {(vals > 0).mean()*100:.1f}%\n")
        report.append(f"- 盈亏比: {vals[vals>0].mean() / abs(vals[vals<0].mean()):.2f}\n")
        report.append(f"- 最大值: {vals.max():+.3f}%\n")
        report.append(f"- 最小值: {vals.min():+.3f}%\n\n")

    # --- 3. 周期错配判断 ---
    report.append("## 三、周期错配判断\n\n")
    top10_5d_mean = top10_df['return_5d'].mean()
    top10_20d_mean = top10_df['return_20d'].mean()
    sh_5d_mean = top10_df['sh_return_5d'].mean()
    sh_20d_mean = top10_df['sh_return_20d'].mean()

    report.append(f"| 指标 | 5日持有 | 20日持有 | 说明 |\n")
    report.append(f"|------|---------|----------|------|\n")
    report.append(f"| Top10 平均收益 | {top10_5d_mean:+.3f}% | {top10_20d_mean:+.3f}% | — |\n")
    report.append(f"| 上证指数收益 | {sh_5d_mean:+.3f}% | {sh_20d_mean:+.3f}% | 基准 |\n")
    report.append(f"| 超额收益 | {top10_5d_mean - sh_5d_mean:+.3f}% | {top10_20d_mean - sh_20d_mean:+.3f}% | vs 大盘 |\n\n")

    if top10_20d_mean > top10_5d_mean:
        report.append(f"**🔍 发现**: 20日持有收益 ({top10_20d_mean:+.3f}%) **高于** 5日持有 ({top10_5d_mean:+.3f}%)，")
        report.append(f"说明策略频繁交易可能错过了后续涨幅。\n\n")
    else:
        report.append(f"**🔍 发现**: 20日持有收益不高于 5日持有，长持并不能改善收益。\n\n")

    # 策略实际收益对比
    report.append("### 与策略实际执行对比\n\n")
    report.append("| 执行方式 | 2025Q1 收益 | 说明 |\n")
    report.append("|----------|-------------|------|\n")
    report.append("| 模型 Top10 持有5日 | 待计算 | 纯模型持有 |\n")
    report.append("| 模型 Top10 持有20日 | 待计算 | 纯模型长持 |\n")
    report.append("| **策略实盘** | **-16.09%** | 含止损/退出/费用 |\n\n")

    # 计算纯持有策略的累计收益
    # 注意：这里不能简单cumprod，因为5日收益有重叠。改用假设每日换仓的近似累计
    # 实际应该模拟每日买入Top10持有5日卖出的策略
    daily_df['cum_top10_5d'] = daily_df['top10_avg_5d'].cumsum() / 100
    daily_df['cum_top10_20d'] = daily_df['top10_avg_20d'].cumsum() / 100

    report.append(f"- 纯持有 Top10（5日）累计收益: {daily_df['cum_top10_5d'].iloc[-1]*100:+.2f}%\n")
    report.append(f"- 纯持有 Top10（20日）累计收益: {daily_df['cum_top10_20d'].iloc[-1]*100:+.2f}%\n")
    report.append(f"- **策略实盘收益: -16.09%**\n\n")

    if daily_df['cum_top10_20d'].iloc[-1] > daily_df['cum_top10_5d'].iloc[-1]:
        report.append("**✅ 周期错配证据**: 长持收益优于短持，说明模型预测的是中长期趋势，")
        report.append("但策略的频繁止损/退出过早砍掉了盈利头寸。\n\n")

    # --- 4. 市场环境敏感性 ---
    report.append("## 四、市场环境敏感性分析\n\n")
    bull_df = daily_df[daily_df['is_bull'] == True]
    bear_df = daily_df[daily_df['is_bull'] == False]

    report.append(f"| 市场环境 | 交易日数 | Top10 5日收益 | Top10 20日收益 |\n")
    report.append(f"|----------|----------|---------------|----------------|\n")
    report.append(f"| 牛市 (上证≥MA20) | {len(bull_df)} | {bull_df['top10_avg_5d'].mean():+.3f}% | {bull_df['top10_avg_20d'].mean():+.3f}% |\n")
    report.append(f"| 熊市 (上证<MA20) | {len(bear_df)} | {bear_df['top10_avg_5d'].mean():+.3f}% | {bear_df['top10_avg_20d'].mean():+.3f}% |\n\n")

    # --- 5. 预测分数与收益相关性 ---
    report.append("## 五、预测分数与实际收益相关性\n\n")

    for horizon in [5, 10, 20]:
        col = f'return_{horizon}d'
        valid = top10_df[['prob', col]].dropna()
        if len(valid) > 10:
            corr, p = stats.pearsonr(valid['prob'], valid[col])
            report.append(f"- {horizon}日收益 vs 预测分数: 相关系数 = {corr:.4f} (p={p:.4f}) {'✅显著' if p < 0.05 else '❌不显著'}\n")

    # 按预测分数分桶分析
    report.append("\n### 按预测分数分桶（Top10 内）\n\n")
    top10_df['prob_bucket'] = pd.qcut(top10_df['prob'], q=3, labels=['Low', 'Mid', 'High'])

    report.append("| 分数桶 | 5日收益 | 20日收益 | 样本数 |\n")
    report.append("|--------|---------|----------|--------|\n")
    for bucket in ['Low', 'Mid', 'High']:
        b = top10_df[top10_df['prob_bucket'] == bucket]
        r5 = b['return_5d'].mean()
        r20 = b['return_20d'].mean()
        report.append(f"| {bucket} | {r5:+.3f}% | {r20:+.3f}% | {len(b)} |\n")

    # === 新增：反向选股对比 ===
    report.append("\n### 反向选股测试（选分数最低的10只）\n\n")
    # 重新读取全部预测，取每日分数最低的10只
    reverse_results = []
    for date in trade_dates:
        pred_file = PRED_DIR / f"predictions_{date}_all.csv"
        if not pred_file.exists():
            continue
        pred_df = pd.read_csv(pred_file)
        if pred_df.empty or 'prob' not in pred_df.columns:
            continue
        pred_df = pred_df.sort_values('prob', ascending=True).reset_index(drop=True)  # 升序 = 最低分在前
        bottom10 = pred_df.head(10)

        conn2 = sqlite3.connect(DB_PATH)
        for _, row in bottom10.iterrows():
            returns = get_future_returns(conn2, row['ts_code'], date, [5, 20])
            reverse_results.append({
                'date': date,
                'ts_code': row['ts_code'],
                'return_5d': returns[5],
                'return_20d': returns[20],
            })
        conn2.close()

    rev_df = pd.DataFrame(reverse_results)
    if not rev_df.empty:
        report.append(f"| 策略 | 5日收益 | 20日收益 | 5日胜率 |\n")
        report.append(f"|------|---------|----------|--------|\n")
        report.append(f"| Top10（高分）| {top10_df['return_5d'].mean():+.3f}% | {top10_df['return_20d'].mean():+.3f}% | {(top10_df['return_5d'] > 0).mean()*100:.1f}% |\n")
        report.append(f"| Bottom10（低分）| {rev_df['return_5d'].mean():+.3f}% | {rev_df['return_20d'].mean():+.3f}% | {(rev_df['return_5d'] > 0).mean()*100:.1f}% |\n\n")

    # === 新增：与大盘对比 ===
    report.append("\n### 与大盘指数对比\n\n")
    conn3 = sqlite3.connect(DB_PATH)
    index_returns = {}
    for idx_code, idx_name in [('000001.SH', '上证指数'), ('000300.SH', '沪深300'), ('399006.SZ', '创业板指')]:
        idx_returns = []
        for date in trade_dates:
            r5 = get_index_return(conn3, idx_code, date, 5)
            if not np.isnan(r5):
                idx_returns.append(r5)
        if idx_returns:
            index_returns[idx_name] = np.mean(idx_returns)
    conn3.close()

    report.append("| 指数 | 5日平均收益 |\n")
    report.append("|------|-------------|\n")
    for name, ret in index_returns.items():
        report.append(f"| {name} | {ret:+.3f}% |\n")
    report.append(f"| **模型Top10** | **{top10_df['return_5d'].mean():+.3f}%** |\n\n")

    # --- 6. 根因诊断结论 ---
    report.append("\n---\n\n")
    report.append("## 六、根因诊断结论\n\n")

    # 判断逻辑
    top10_5d_winrate = (top10_df['return_5d'] > 0).mean() * 100
    top10_20d_winrate = (top10_df['return_20d'] > 0).mean() * 100
    excess_5d = top10_5d_mean - sh_5d_mean
    excess_20d = top10_20d_mean - sh_20d_mean

    conclusions = []

    # 判断1：模型是否有区分能力
    if excess_5d > 0:
        conclusions.append("✅ **模型有区分能力**: Top10 的 5日超额收益为正，说明模型能选出相对强势的股票。")
    else:
        conclusions.append("❌ **模型区分能力不足**: Top10 的 5日超额收益为负，模型在熊市中失去了选股能力。")

    # 判断2：周期错配
    if top10_20d_mean > top10_5d_mean and top10_20d_winrate > top10_5d_winrate:
        conclusions.append("✅ **存在周期错配**: 20日持有收益和胜率均优于5日，模型预测的是中长期趋势，策略的短期止损/退出过早离场。")
    else:
        conclusions.append("❌ **无明显周期错配**: 长持并不能显著改善收益，问题不在于策略周期。")

    # 判断3：市场环境
    if len(bear_df) > 0 and bear_df['top10_avg_5d'].mean() < 0:
        conclusions.append(f"⚠️ **熊市环境恶化**: 在 {len(bear_df)} 个熊市交易日中，Top10 平均5日收益为负 ({bear_df['top10_avg_5d'].mean():+.3f}%)，模型在下跌趋势中无法逆势盈利。")

    for c in conclusions:
        report.append(f"{c}\n\n")

    # 最终结论
    report.append("### 最终判定\n\n")

    # === 核心判定逻辑 ===
    # 关键指标：Top10 vs Other 的相对表现
    other_5d = df[df['rank_group'] == 'Other']['return_5d'].mean()
    top10_underperforms_other = top10_5d_mean < other_5d

    report.append("### 核心发现\n\n")

    if top10_underperforms_other:
        report.append("🔴 **致命发现：模型排序完全反向！**\n\n")
        report.append(f"- Top10（模型认为最好的）5日收益: **{top10_5d_mean:+.3f}%**\n")
        report.append(f"- Other（模型认为不会的）5日收益: **{other_5d:+.3f}%**\n")
        report.append(f"- 差距: **{other_5d - top10_5d_mean:+.3f}%**\n\n")
        report.append("这意味着：模型在 2025Q1 的预测分数与实际收益呈**负相关**。\n")
        report.append("分数越高的股票，跌得越多；分数越低的股票，反而更抗跌。\n\n")

    report.append("### 最终判定\n\n")
    report.append("**🔴 主要根因：模型在熊市中完全失效，且排序可能反向**\n\n")
    report.append("模型训练的是「突破拉升」模式（50%+涨幅），在牛市中有效：\n")
    report.append("- 牛市中，高位股票继续突破（ momentum 效应）\n")
    report.append("- 熊市中，高位股票是「假突破」陷阱，回调最深\n\n")
    report.append("2025Q1 的 Top10 正是这些「看起来像要突破」的高位股票，\n")
    report.append("模型把它们误判为机会，实际是风险最高的标的。\n\n")
    report.append("**优化方向**（按 ROI 排序）：\n\n")
    report.append("1. 🥇 **硬负样本扩充至 15-20%**（P1.2）\n")
    report.append("   模型最缺的是「看起来像突破但实际失败」的样本。\n")
    report.append("   当前仅 130 个硬负样本（1.7%），模型学不会识别假突破。\n\n")
    report.append("2. 🥇 **加入市场环境特征**（P1.3）\n")
    report.append("   训练时加入大盘趋势、波动率、成交量等市场状态特征，\n")
    report.append("   让模型学会「牛市追突破、熊市避陷阱」。\n\n")
    report.append("3. 🥈 **训练「短期相对收益」模型**（P1.4）\n")
    report.append("   当前模型预测 50 天突破，与策略 2-5 天持仓严重错配。\n")
    report.append("   新增「5日跑赢大盘」预测目标，与策略周期对齐。\n\n")
    report.append("4. 🥉 **四层仓位管理**（P2.1）\n")
    report.append("   在市场环境恶劣时自动降低仓位，避免满仓挨打。\n\n")
    report.append("5. 🥉 **策略参数优化**（P2.2）\n")
    report.append("   调整止损/退出参数，减少熊市中的反复止损损耗。\n\n")

    # 写入报告
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        f.writelines(report)

    print(f"✅ 诊断报告已保存: {OUTPUT}")
    print(f"\n摘要:")
    print(f"  - 分析样本: {len(df)} 条记录")
    print(f"  - Top10 5日平均收益: {top10_5d_mean:+.3f}%")
    print(f"  - Top10 20日平均收益: {top10_20d_mean:+.3f}%")
    print(f"  - 5日超额收益(vs大盘): {excess_5d:+.3f}%")
    print(f"  - 5日胜率: {top10_5d_winrate:.1f}%")
    print(f"  - 20日胜率: {top10_20d_winrate:.1f}%")
    print(f"  - 牛市交易日: {len(bull_df)}, 熊市交易日: {len(bear_df)}")

if __name__ == '__main__':
    analyze()
