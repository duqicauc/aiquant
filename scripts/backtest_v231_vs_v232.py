#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测对比v2.3.1和v2.3.2的效果

使用历史预测结果的full文件，用v2.3.2的评分逻辑重新排序，
然后对比两种策略选出的Top10的实际收益。
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def calculate_v232_score(row):
    """
    使用v2.3.2评分逻辑重新计算分数
    """
    cal_prob = row['calibrated_probability']
    expected_return_score = row['expected_return_score']
    pct_chg = row['pct_chg']
    rsi_6 = row.get('rsi_6', 50)
    
    # 基础评分：0.6*校准概率 + 0.4*预期收益
    expected_return_norm = float(np.clip(expected_return_score, 0, 1))
    base_score = 0.6 * cal_prob + 0.4 * expected_return_norm
    
    penalty = 1.0
    penalty_reasons = []
    
    # 1. 追高惩罚：当日涨幅>15%
    if pct_chg > 15:
        penalty *= 0.5
        penalty_reasons.append(f"追高惩罚({pct_chg:.1f}%)")
    elif pct_chg > 10:
        penalty *= 0.8
        penalty_reasons.append(f"轻度追高({pct_chg:.1f}%)")
    
    # 2. 涨停低概率惩罚
    if pct_chg >= 9.8 and cal_prob < 0.8:
        penalty *= 0.7
        penalty_reasons.append(f"涨停低概率({cal_prob:.2f})")
    
    # 3. RSI过热惩罚
    if rsi_6 > 95:
        penalty *= 0.8
        penalty_reasons.append(f"RSI过热({rsi_6:.1f})")
    elif rsi_6 > 90:
        penalty *= 0.9
        penalty_reasons.append(f"RSI偏高({rsi_6:.1f})")
    
    final_score = base_score * penalty
    
    return pd.Series({
        'v232_score': final_score,
        'v232_penalty': penalty,
        'v232_reasons': '|'.join(penalty_reasons) if penalty_reasons else ''
    })


def evaluate_returns(df_pred, dm, pred_date, eval_date):
    """评估实际收益"""
    eval_start = (datetime.strptime(eval_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
    eval_end = (datetime.strptime(eval_date, '%Y%m%d') + timedelta(days=5)).strftime('%Y%m%d')
    
    # 批量获取数据
    stock_codes = df_pred['ts_code'].tolist()
    daily_data_dict = dm.batch_get_daily_data(stock_codes, eval_start, eval_end)
    
    returns = []
    for _, row in df_pred.iterrows():
        ts_code = row['ts_code']
        try:
            df_eval = daily_data_dict.get(ts_code)
            if df_eval is None or len(df_eval) == 0:
                returns.append(np.nan)
                continue
            
            df_eval['date_diff'] = abs(pd.to_datetime(df_eval['trade_date']) - pd.to_datetime(eval_date))
            closest = df_eval.loc[df_eval['date_diff'].idxmin()]
            
            eval_price = closest['close']
            predict_price = row['close']
            return_pct = (eval_price / predict_price - 1) * 100
            returns.append(return_pct)
        except:
            returns.append(np.nan)
    
    return returns


def backtest_single_date(pred_date, eval_date, dm):
    """回测单个日期"""
    log.info(f"\n{'='*80}")
    log.info(f"回测: {pred_date} -> {eval_date}")
    log.info(f"{'='*80}")
    
    # 读取v2.3.1的full结果
    full_file = PROJECT_ROOT / 'data' / 'prediction' / 'results' / f'v2.3.1_full_{pred_date}.csv'
    if not full_file.exists():
        log.warning(f"文件不存在: {full_file}")
        return None
    
    df_full = pd.read_csv(full_file)
    log.info(f"加载数据: {len(df_full)} 只股票")
    
    # 使用v2.3.2评分逻辑重新计算
    v232_scores = df_full.apply(calculate_v232_score, axis=1)
    df_full = pd.concat([df_full, v232_scores], axis=1)
    
    # v2.3.1 Top10 (按原始final_score排序)
    df_v231_top10 = df_full.nlargest(10, 'final_score').copy()
    
    # v2.3.2 Top10 (按v232_score排序)
    df_v232_top10 = df_full.nlargest(10, 'v232_score').copy()
    
    # 计算实际收益
    log.info("计算v2.3.1 Top10实际收益...")
    df_v231_top10['return_pct'] = evaluate_returns(df_v231_top10, dm, pred_date, eval_date)
    
    log.info("计算v2.3.2 Top10实际收益...")
    df_v232_top10['return_pct'] = evaluate_returns(df_v232_top10, dm, pred_date, eval_date)
    
    # 统计
    v231_valid = df_v231_top10[df_v231_top10['return_pct'].notna()]
    v232_valid = df_v232_top10[df_v232_top10['return_pct'].notna()]
    
    v231_stats = {
        'avg_return': v231_valid['return_pct'].mean(),
        'median_return': v231_valid['return_pct'].median(),
        'win_rate': (v231_valid['return_pct'] > 0).mean() * 100,
        'max_return': v231_valid['return_pct'].max(),
        'min_return': v231_valid['return_pct'].min(),
        'chase_high_count': (df_v231_top10['pct_chg'] > 9).sum(),
        'avg_pct_chg': df_v231_top10['pct_chg'].mean(),
    }
    
    v232_stats = {
        'avg_return': v232_valid['return_pct'].mean(),
        'median_return': v232_valid['return_pct'].median(),
        'win_rate': (v232_valid['return_pct'] > 0).mean() * 100,
        'max_return': v232_valid['return_pct'].max(),
        'min_return': v232_valid['return_pct'].min(),
        'chase_high_count': (df_v232_top10['pct_chg'] > 9).sum(),
        'avg_pct_chg': df_v232_top10['pct_chg'].mean(),
    }
    
    # 显示对比
    log.info(f"\n{'指标':<20} {'v2.3.1':<15} {'v2.3.2':<15} {'差异':<15}")
    log.info("-" * 65)
    log.info(f"{'平均收益':<20} {v231_stats['avg_return']:>+10.2f}%    {v232_stats['avg_return']:>+10.2f}%    {v232_stats['avg_return'] - v231_stats['avg_return']:>+10.2f}%")
    log.info(f"{'中位数收益':<20} {v231_stats['median_return']:>+10.2f}%    {v232_stats['median_return']:>+10.2f}%    {v232_stats['median_return'] - v231_stats['median_return']:>+10.2f}%")
    log.info(f"{'胜率':<20} {v231_stats['win_rate']:>10.1f}%    {v232_stats['win_rate']:>10.1f}%    {v232_stats['win_rate'] - v231_stats['win_rate']:>+10.1f}%")
    log.info(f"{'最高收益':<20} {v231_stats['max_return']:>+10.2f}%    {v232_stats['max_return']:>+10.2f}%")
    log.info(f"{'最低收益':<20} {v231_stats['min_return']:>+10.2f}%    {v232_stats['min_return']:>+10.2f}%")
    log.info(f"{'追高数量(>9%)':<20} {v231_stats['chase_high_count']:>10}/10    {v232_stats['chase_high_count']:>10}/10    {v232_stats['chase_high_count'] - v231_stats['chase_high_count']:>+10}")
    log.info(f"{'平均当日涨幅':<20} {v231_stats['avg_pct_chg']:>+10.2f}%    {v232_stats['avg_pct_chg']:>+10.2f}%")
    
    # 显示Top10对比
    log.info(f"\n【v2.3.1 Top10】")
    log.info(f"{'排名':<4} {'代码':<12} {'名称':<10} {'当日涨幅':<10} {'实际收益':<10}")
    log.info("-" * 50)
    for i, (_, row) in enumerate(df_v231_top10.iterrows(), 1):
        ret = row['return_pct']
        ret_str = f"{ret:+.2f}%" if pd.notna(ret) else "N/A"
        log.info(f"{i:<4} {row['ts_code']:<12} {row['name']:<10} {row['pct_chg']:>+9.2f}% {ret_str:>10}")
    
    log.info(f"\n【v2.3.2 Top10】")
    log.info(f"{'排名':<4} {'代码':<12} {'名称':<10} {'当日涨幅':<10} {'实际收益':<10} {'惩罚原因':<20}")
    log.info("-" * 80)
    for i, (_, row) in enumerate(df_v232_top10.iterrows(), 1):
        ret = row['return_pct']
        ret_str = f"{ret:+.2f}%" if pd.notna(ret) else "N/A"
        reasons = row['v232_reasons'] if row['v232_reasons'] else "-"
        log.info(f"{i:<4} {row['ts_code']:<12} {row['name']:<10} {row['pct_chg']:>+9.2f}% {ret_str:>10} {reasons:<20}")
    
    return {
        'pred_date': pred_date,
        'eval_date': eval_date,
        'v231': v231_stats,
        'v232': v232_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='回测对比v2.3.1和v2.3.2')
    parser.add_argument('--eval-date', type=str, default='20260109', help='评估日期')
    args = parser.parse_args()
    
    eval_date = args.eval_date
    
    log.info("="*80)
    log.info("v2.3.1 vs v2.3.2 回测对比")
    log.info("="*80)
    
    dm = DataManager()
    
    # 测试多个日期
    pred_dates = ['20251231', '20260105', '20260106', '20260107']
    
    all_results = []
    for pred_date in pred_dates:
        result = backtest_single_date(pred_date, eval_date, dm)
        if result:
            all_results.append(result)
    
    # 汇总统计
    if all_results:
        log.info("\n" + "="*80)
        log.info("📊 汇总统计")
        log.info("="*80)
        
        v231_returns = [r['v231']['avg_return'] for r in all_results]
        v232_returns = [r['v232']['avg_return'] for r in all_results]
        v231_win_rates = [r['v231']['win_rate'] for r in all_results]
        v232_win_rates = [r['v232']['win_rate'] for r in all_results]
        v231_chase = [r['v231']['chase_high_count'] for r in all_results]
        v232_chase = [r['v232']['chase_high_count'] for r in all_results]
        
        log.info(f"\n{'指标':<25} {'v2.3.1':<15} {'v2.3.2':<15} {'改进':<15}")
        log.info("-" * 70)
        log.info(f"{'平均收益(所有日期平均)':<25} {np.mean(v231_returns):>+10.2f}%    {np.mean(v232_returns):>+10.2f}%    {np.mean(v232_returns) - np.mean(v231_returns):>+10.2f}%")
        log.info(f"{'平均胜率(所有日期平均)':<25} {np.mean(v231_win_rates):>10.1f}%    {np.mean(v232_win_rates):>10.1f}%    {np.mean(v232_win_rates) - np.mean(v231_win_rates):>+10.1f}%")
        log.info(f"{'平均追高数量':<25} {np.mean(v231_chase):>10.1f}/10    {np.mean(v232_chase):>10.1f}/10    {np.mean(v232_chase) - np.mean(v231_chase):>+10.1f}")
        
        # 判断改进效果
        return_improvement = np.mean(v232_returns) - np.mean(v231_returns)
        chase_reduction = np.mean(v231_chase) - np.mean(v232_chase)
        
        log.info("\n" + "="*80)
        log.info("📝 结论")
        log.info("="*80)
        
        if return_improvement > 0:
            log.success(f"✅ v2.3.2平均收益提升 {return_improvement:.2f}%")
        elif return_improvement > -1:
            log.info(f"➖ v2.3.2平均收益基本持平 ({return_improvement:.2f}%)")
        else:
            log.warning(f"⚠️  v2.3.2平均收益下降 {return_improvement:.2f}%")
        
        if chase_reduction > 0:
            log.success(f"✅ v2.3.2追高数量减少 {chase_reduction:.1f}只/天")
        else:
            log.info(f"➖ 追高数量未明显减少")
        
        # 建议
        log.info("\n" + "="*80)
        log.info("💡 建议")
        log.info("="*80)
        
        if return_improvement >= 0 and chase_reduction > 0:
            log.success("推荐使用v2.3.2：收益不降低的同时降低了追高风险")
        elif return_improvement > 1:
            log.success("推荐使用v2.3.2：收益有明显提升")
        elif return_improvement > -1 and chase_reduction > 1:
            log.info("可以考虑使用v2.3.2：收益基本持平，但风险控制更好")
        else:
            log.warning("建议继续使用v2.3.1或调整v2.3.2参数")


if __name__ == '__main__':
    main()
