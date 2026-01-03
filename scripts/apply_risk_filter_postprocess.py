#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险过滤后处理分析

基于已有的v2.1.0评估结果，添加风险过滤规则，对比效果

风险过滤规则：
1. 基于预测时的34日涨幅判断（需要从数据中获取）
2. 基于RSI等技术指标
3. 基于波动率

由于原始评估数据中没有这些指标，我们需要回溯获取
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_risk_metrics(dm, ts_code, predict_date):
    """获取风险指标"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, '%Y%m%d') - timedelta(days=100)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 34:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 34日涨幅
        if len(df) >= 34:
            return_34d = (df['close'].iloc[-1] / df['close'].iloc[-34] - 1) * 100
        else:
            return_34d = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # 波动率
        volatility = df['pct_chg'].std()
        vol_mean = df['pct_chg'].rolling(20).std().mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi_14 = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # 近5日连续下跌
        last_5_pct = df['pct_chg'].tail(5)
        consecutive_down = (last_5_pct < 0).sum()
        
        # 近期涨停次数
        limit_up_count = (df['pct_chg'].tail(10) >= 9.8).sum()
        
        return {
            'return_34d': return_34d,
            'volatility': volatility,
            'volatility_mean': vol_mean if not pd.isna(vol_mean) else volatility,
            'rsi_14': rsi_14,
            'consecutive_down': consecutive_down,
            'limit_up_count': limit_up_count
        }
    except Exception as e:
        return None


def calculate_risk_score(risk_metrics):
    """计算风险系数"""
    if risk_metrics is None:
        return 0.5, [], {}
    
    risk_score = 1.0
    risk_reasons = []
    
    # 规则1: 34日涨幅过大
    return_34d = risk_metrics.get('return_34d', 0)
    if return_34d > 80:
        risk_score *= 0.3
        risk_reasons.append(f'34日涨幅过大({return_34d:.1f}%)')
    elif return_34d > 60:
        risk_score *= 0.5
        risk_reasons.append(f'34日涨幅较大({return_34d:.1f}%)')
    elif return_34d > 40:
        risk_score *= 0.7
        risk_reasons.append(f'34日涨幅偏高({return_34d:.1f}%)')
    
    # 规则2: 波动率过高
    volatility = risk_metrics.get('volatility', 0)
    vol_mean = risk_metrics.get('volatility_mean', volatility)
    if vol_mean > 0 and volatility > vol_mean * 2.5:
        risk_score *= 0.5
        risk_reasons.append('波动率过高')
    elif vol_mean > 0 and volatility > vol_mean * 2:
        risk_score *= 0.7
        risk_reasons.append('波动率偏高')
    
    # 规则3: 近5日连续下跌
    consecutive_down = risk_metrics.get('consecutive_down', 0)
    if consecutive_down >= 5:
        risk_score *= 0.4
        risk_reasons.append('连续5日下跌')
    elif consecutive_down >= 4:
        risk_score *= 0.6
        risk_reasons.append('近5日多数下跌')
    
    # 规则4: RSI超买
    rsi = risk_metrics.get('rsi_14', 50)
    if rsi > 85:
        risk_score *= 0.5
        risk_reasons.append(f'RSI超买({rsi:.1f})')
    elif rsi > 75:
        risk_score *= 0.7
        risk_reasons.append(f'RSI偏高({rsi:.1f})')
    
    # 规则5: 近期多次涨停
    limit_up_count = risk_metrics.get('limit_up_count', 0)
    if limit_up_count >= 3:
        risk_score *= 0.5
        risk_reasons.append(f'近期多次涨停({limit_up_count}次)')
    elif limit_up_count >= 2:
        risk_score *= 0.7
        risk_reasons.append(f'近期涨停({limit_up_count}次)')
    
    return risk_score, risk_reasons, risk_metrics


def main():
    log.info("="*80)
    log.info("风险过滤后处理分析")
    log.info("="*80)
    
    # 初始化
    dm = DataManager()
    
    # 读取原始评估结果
    eval_file = PROJECT_ROOT / 'data' / 'prediction' / 'evaluation' / 'v2.1.0_eval_20251212_to_20251231.csv'
    df_eval = pd.read_csv(eval_file)
    
    log.info(f"原始评估数据: {len(df_eval)} 条")
    
    # 原始统计
    log.info("\n" + "="*80)
    log.info("原始v2.1.0 Top50评估结果（无风险过滤）")
    log.info("="*80)
    
    avg_return_orig = df_eval['return_pct'].mean()
    median_return_orig = df_eval['return_pct'].median()
    win_rate_orig = (df_eval['return_pct'] > 0).mean() * 100
    max_return_orig = df_eval['return_pct'].max()
    min_return_orig = df_eval['return_pct'].min()
    
    log.info(f"  平均收益率: {avg_return_orig:.2f}%")
    log.info(f"  中位数收益: {median_return_orig:.2f}%")
    log.info(f"  胜率: {win_rate_orig:.1f}%")
    log.info(f"  最高收益: {max_return_orig:.2f}%")
    log.info(f"  最低收益: {min_return_orig:.2f}%")
    
    # 为每只股票获取风险指标
    log.info("\n获取风险指标...")
    predict_date = '20251212'
    
    risk_data = []
    for idx, row in df_eval.iterrows():
        ts_code = row['ts_code']
        
        if (idx + 1) % 10 == 0:
            log.info(f"  进度: {idx+1}/{len(df_eval)}")
        
        risk_metrics = get_risk_metrics(dm, ts_code, predict_date)
        risk_score, risk_reasons, metrics = calculate_risk_score(risk_metrics)
        
        risk_data.append({
            'ts_code': ts_code,
            'name': row['name'],
            'raw_probability': row['probability'],
            'risk_score': risk_score,
            'adjusted_probability': row['probability'] * risk_score,
            'return_pct': row['return_pct'],
            'return_34d_at_predict': metrics.get('return_34d', 0) if metrics else 0,
            'rsi_14': metrics.get('rsi_14', 50) if metrics else 50,
            'limit_up_count': metrics.get('limit_up_count', 0) if metrics else 0,
            'risk_reasons': '; '.join(risk_reasons) if risk_reasons else ''
        })
    
    df_with_risk = pd.DataFrame(risk_data)
    
    # 按调整后概率重新排序
    df_with_risk_sorted = df_with_risk.sort_values('adjusted_probability', ascending=False)
    
    # 取新的Top50
    df_filtered_top50 = df_with_risk_sorted.head(50)
    
    log.info("\n" + "="*80)
    log.info("风险过滤后 Top50评估结果")
    log.info("="*80)
    
    avg_return_filt = df_filtered_top50['return_pct'].mean()
    median_return_filt = df_filtered_top50['return_pct'].median()
    win_rate_filt = (df_filtered_top50['return_pct'] > 0).mean() * 100
    max_return_filt = df_filtered_top50['return_pct'].max()
    min_return_filt = df_filtered_top50['return_pct'].min()
    
    log.info(f"  平均收益率: {avg_return_filt:.2f}%")
    log.info(f"  中位数收益: {median_return_filt:.2f}%")
    log.info(f"  胜率: {win_rate_filt:.1f}%")
    log.info(f"  最高收益: {max_return_filt:.2f}%")
    log.info(f"  最低收益: {min_return_filt:.2f}%")
    
    # 对比
    log.info("\n" + "="*80)
    log.info("风险过滤效果对比")
    log.info("="*80)
    
    log.info("\n| 指标 | 无过滤 | 带风险过滤 | 变化 |")
    log.info("|------|--------|------------|------|")
    log.info(f"| 平均收益率 | {avg_return_orig:.2f}% | {avg_return_filt:.2f}% | {avg_return_filt - avg_return_orig:+.2f}% |")
    log.info(f"| 中位数收益 | {median_return_orig:.2f}% | {median_return_filt:.2f}% | {median_return_filt - median_return_orig:+.2f}% |")
    log.info(f"| 胜率 | {win_rate_orig:.1f}% | {win_rate_filt:.1f}% | {win_rate_filt - win_rate_orig:+.1f}% |")
    log.info(f"| 最高收益 | {max_return_orig:.2f}% | {max_return_filt:.2f}% | - |")
    log.info(f"| 最低收益 | {min_return_orig:.2f}% | {min_return_filt:.2f}% | - |")
    
    # 分析被过滤掉的股票
    orig_codes = set(df_eval['ts_code'])
    filt_codes = set(df_filtered_top50['ts_code'])
    
    filtered_out = orig_codes - filt_codes
    filtered_in = filt_codes - orig_codes
    
    log.info(f"\n被过滤掉的股票: {len(filtered_out)} 只")
    if filtered_out:
        df_out = df_with_risk[df_with_risk['ts_code'].isin(filtered_out)]
        avg_out = df_out['return_pct'].mean()
        log.info(f"  这些股票的平均收益: {avg_out:.2f}%")
        for _, row in df_out.sort_values('return_pct').iterrows():
            log.info(f"    {row['ts_code']} {row['name']}: 收益{row['return_pct']:.2f}%, 34日涨{row['return_34d_at_predict']:.1f}%, 风险[{row['risk_reasons']}]")
    
    # 高风险股票分析
    log.info("\n" + "="*80)
    log.info("高风险股票分析（原Top50中risk_score < 0.7的股票）")
    log.info("="*80)
    
    df_high_risk = df_with_risk[df_with_risk['risk_score'] < 0.7]
    if len(df_high_risk) > 0:
        log.info(f"高风险股票数: {len(df_high_risk)}")
        avg_high_risk = df_high_risk['return_pct'].mean()
        log.info(f"高风险股票平均收益: {avg_high_risk:.2f}%")
        
        for _, row in df_high_risk.sort_values('risk_score').iterrows():
            log.info(f"  {row['ts_code']} {row['name']}: 风险系数{row['risk_score']:.2f}, 收益{row['return_pct']:.2f}%, 34日涨{row['return_34d_at_predict']:.1f}%")
            if row['risk_reasons']:
                log.info(f"    风险原因: {row['risk_reasons']}")
    else:
        log.info("无高风险股票")
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'evaluation'
    
    df_with_risk.to_csv(output_dir / 'v2.1.0_with_risk_analysis.csv', index=False, encoding='utf-8-sig')
    df_filtered_top50.to_csv(output_dir / 'v2.1.0_filtered_top50.csv', index=False, encoding='utf-8-sig')
    
    log.success(f"\n✓ 结果已保存到 {output_dir}")
    
    # 最终结论
    log.info("\n" + "="*80)
    log.info("结论")
    log.info("="*80)
    
    improvement = avg_return_filt - avg_return_orig
    if improvement > 0:
        log.success(f"✓ 风险过滤有效！平均收益提升 {improvement:.2f}%")
    else:
        log.warning(f"✗ 风险过滤效果不明显，平均收益变化 {improvement:.2f}%")
    
    win_improvement = win_rate_filt - win_rate_orig
    if win_improvement > 0:
        log.success(f"✓ 胜率提升 {win_improvement:.1f}%")


if __name__ == '__main__':
    main()

