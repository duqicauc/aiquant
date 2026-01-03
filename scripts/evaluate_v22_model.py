#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.2.0模型（带概率校准+风险过滤）

预测12月12日，用12月31日评价
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_model_v22():
    """加载v2.2.0模型（带校准器）"""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v2.2.0' / 'model'
    
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'model.json'))
    
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    calibrator = joblib.load(str(model_dir / 'calibrator.pkl'))
    
    return booster, feature_names, calibrator


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
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi_14 = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # 近期涨停次数
        limit_up_count = (df['pct_chg'].tail(10) >= 9.8).sum()
        
        # 近5日下跌
        consecutive_down = (df['pct_chg'].tail(5) < 0).sum()
        
        return {
            'return_34d': return_34d,
            'rsi_14': rsi_14,
            'limit_up_count': limit_up_count,
            'consecutive_down': consecutive_down
        }
    except:
        return None


def calculate_risk_score(risk_metrics):
    """计算风险系数"""
    if risk_metrics is None:
        return 0.5, []
    
    risk_score = 1.0
    risk_reasons = []
    
    # 34日涨幅
    return_34d = risk_metrics.get('return_34d', 0)
    if return_34d > 80:
        risk_score *= 0.3
        risk_reasons.append(f'34日涨{return_34d:.0f}%')
    elif return_34d > 60:
        risk_score *= 0.5
        risk_reasons.append(f'34日涨{return_34d:.0f}%')
    elif return_34d > 40:
        risk_score *= 0.7
        risk_reasons.append(f'34日涨{return_34d:.0f}%')
    
    # RSI
    rsi = risk_metrics.get('rsi_14', 50)
    if rsi > 85:
        risk_score *= 0.5
        risk_reasons.append(f'RSI{rsi:.0f}')
    elif rsi > 75:
        risk_score *= 0.7
        risk_reasons.append(f'RSI{rsi:.0f}')
    
    # 涨停
    limit_up = risk_metrics.get('limit_up_count', 0)
    if limit_up >= 3:
        risk_score *= 0.5
        risk_reasons.append(f'{limit_up}次涨停')
    elif limit_up >= 2:
        risk_score *= 0.7
        risk_reasons.append(f'{limit_up}次涨停')
    
    # 连续下跌
    down = risk_metrics.get('consecutive_down', 0)
    if down >= 4:
        risk_score *= 0.6
        risk_reasons.append('近期下跌')
    
    return risk_score, risk_reasons


def main():
    log.info("="*80)
    log.info("v2.2.0模型评估（概率校准+风险过滤）")
    log.info("="*80)
    
    predict_date = '20251212'
    eval_date = '20251231'
    
    log.info(f"预测日期: {predict_date}")
    log.info(f"评估日期: {eval_date}")
    
    # 初始化
    dm = DataManager()
    
    # 加载模型
    log.info("\n加载v2.2.0模型...")
    booster, feature_names, calibrator = load_model_v22()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征, 带校准器")
    
    # 读取之前的评估结果（使用v2.1.0的Top50）
    eval_file = PROJECT_ROOT / 'data' / 'prediction' / 'evaluation' / 'v2.1.0_eval_20251212_to_20251231.csv'
    df_v21 = pd.read_csv(eval_file)
    
    log.info(f"\n原v2.1.0 Top50结果:")
    log.info(f"  平均收益: {df_v21['return_pct'].mean():.2f}%")
    log.info(f"  胜率: {(df_v21['return_pct'] > 0).mean()*100:.1f}%")
    
    # 获取风险指标
    log.info("\n获取风险指标...")
    risk_data = []
    for idx, row in df_v21.iterrows():
        ts_code = row['ts_code']
        
        risk_metrics = get_risk_metrics(dm, ts_code, predict_date)
        risk_score, risk_reasons = calculate_risk_score(risk_metrics)
        
        risk_data.append({
            'ts_code': ts_code,
            'name': row['name'],
            'raw_prob': row['probability'],
            'risk_score': risk_score,
            'return_pct': row['return_pct'],
            'risk_reasons': '; '.join(risk_reasons) if risk_reasons else ''
        })
    
    df_risk = pd.DataFrame(risk_data)
    
    # 风险过滤分析
    log.info("\n" + "="*80)
    log.info("风险过滤分析（阈值0.7）")
    log.info("="*80)
    
    # 低风险组
    df_low_risk = df_risk[df_risk['risk_score'] >= 0.7]
    log.info(f"\n低风险组（risk_score >= 0.7）: {len(df_low_risk)} 只")
    if len(df_low_risk) > 0:
        avg_ret = df_low_risk['return_pct'].mean()
        win_rate = (df_low_risk['return_pct'] > 0).mean() * 100
        log.info(f"  平均收益: {avg_ret:.2f}%")
        log.info(f"  胜率: {win_rate:.1f}%")
        log.info(f"  最高: {df_low_risk['return_pct'].max():.2f}%")
        log.info(f"  最低: {df_low_risk['return_pct'].min():.2f}%")
        
        log.info(f"\n低风险股票列表:")
        for _, row in df_low_risk.sort_values('return_pct', ascending=False).iterrows():
            log.info(f"    {row['ts_code']} {row['name']}: 风险{row['risk_score']:.2f}, 收益{row['return_pct']:.2f}%")
    
    # 高风险组
    df_high_risk = df_risk[df_risk['risk_score'] < 0.7]
    log.info(f"\n高风险组（risk_score < 0.7）: {len(df_high_risk)} 只")
    if len(df_high_risk) > 0:
        avg_ret = df_high_risk['return_pct'].mean()
        win_rate = (df_high_risk['return_pct'] > 0).mean() * 100
        log.info(f"  平均收益: {avg_ret:.2f}%")
        log.info(f"  胜率: {win_rate:.1f}%")
        
        log.info(f"\n被过滤掉的高风险股票:")
        for _, row in df_high_risk.sort_values('return_pct').head(10).iterrows():
            log.info(f"    {row['ts_code']} {row['name']}: 风险{row['risk_score']:.2f}, 收益{row['return_pct']:.2f}%, [{row['risk_reasons']}]")
    
    # 最终对比
    log.info("\n" + "="*80)
    log.info("最终效果对比")
    log.info("="*80)
    
    orig_avg = df_v21['return_pct'].mean()
    orig_win = (df_v21['return_pct'] > 0).mean() * 100
    
    filt_avg = df_low_risk['return_pct'].mean() if len(df_low_risk) > 0 else 0
    filt_win = (df_low_risk['return_pct'] > 0).mean() * 100 if len(df_low_risk) > 0 else 0
    
    log.info(f"\n| 指标 | 原始Top50 | 风险过滤后 | 变化 |")
    log.info(f"|------|----------|-----------|------|")
    log.info(f"| 股票数 | 50 | {len(df_low_risk)} | {len(df_low_risk)-50:+d} |")
    log.info(f"| 平均收益 | {orig_avg:.2f}% | {filt_avg:.2f}% | {filt_avg-orig_avg:+.2f}% |")
    log.info(f"| 胜率 | {orig_win:.1f}% | {filt_win:.1f}% | {filt_win-orig_win:+.1f}% |")
    
    if filt_avg > orig_avg:
        log.success(f"\n✓ 风险过滤有效！收益提升 {filt_avg-orig_avg:.2f}%")
    
    if filt_win > orig_win:
        log.success(f"✓ 胜率提升 {filt_win-orig_win:.1f}%")


if __name__ == '__main__':
    main()

