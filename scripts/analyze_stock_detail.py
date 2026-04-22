#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细分析单只股票的特征和推荐原因
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager
from scripts.archive.predict_v240 import extract_features, load_model


def analyze_stock(ts_code, predict_date='20260107'):
    """详细分析单只股票"""
    log.info("="*80)
    log.info(f"详细分析: {ts_code} - {predict_date}")
    log.info("="*80)
    
    # 1. 加载模型
    booster, feature_names, calibrator = load_model()
    
    # 2. 初始化数据管理器
    dm = DataManager()
    
    # 3. 获取股票基本信息
    stock_list = dm.get_stock_list()
    stock_info = stock_list[stock_list['ts_code'] == ts_code]
    if stock_info.empty:
        log.error(f"未找到股票: {ts_code}")
        return
    
    name = stock_info.iloc[0]['name']
    log.info(f"\n股票名称: {name} ({ts_code})")
    
    # 4. 获取日线数据
    end_date = predict_date
    start_date = (datetime.strptime(predict_date, '%Y%m%d') - timedelta(days=200)).strftime('%Y%m%d')
    
    df = dm.get_daily_data(ts_code, start_date, end_date)
    if df is None or len(df) < 60:
        log.error(f"数据不足: {ts_code}")
        return
    
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 5. 提取特征
    df = extract_features(df)
    last_row = df.iloc[-1]
    
    # 6. 构建特征向量
    feature_vector = []
    for fn in feature_names:
        val = last_row.get(fn, 0)
        if pd.isna(val) or not np.isfinite(val):
            val = 0
        feature_vector.append(float(val))
    
    # 7. 预测
    dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
    raw_prob = float(booster.predict(dmatrix)[0])
    cal_prob = float(calibrator.predict([raw_prob])[0])
    
    log.info(f"\n预测结果:")
    log.info(f"  原始概率: {raw_prob:.4f}")
    log.info(f"  校准概率: {cal_prob:.4f}")
    
    # 8. 获取特征重要性
    feature_importance = booster.get_score(importance_type='gain')
    
    # 9. 分析关键特征
    log.info(f"\n{'='*80}")
    log.info("关键特征分析")
    log.info("="*80)
    
    # 反追龙头相关特征
    anti_chasing_features = [
        'return_34d', 'price_range_pct', 'close_vs_ma10_std',
        'days_near_ma10', 'volume_shrink_ratio', 'ma10_cross_count'
    ]
    
    log.info("\n【反追龙头特征】")
    for feat in anti_chasing_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 技术指标
    tech_features = [
        'rsi_6', 'rsi_12', 'rsi_24', 'kdj_k', 'kdj_d', 'kdj_j',
        'macd', 'macd_dif', 'macd_dea'
    ]
    
    log.info("\n【技术指标】")
    for feat in tech_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 价量关系
    price_volume_features = [
        'volume_ratio', 'volume_expansion_ratio', 'volume_price_corr_10d',
        'volume_price_match_sum_10d', 'breakout_volume_ratio'
    ]
    
    log.info("\n【价量关系】")
    for feat in price_volume_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 突破特征
    breakout_features = [
        'breakout_high_10d', 'breakout_high_20d', 'breakout_high_55d',
        'breakout_ma5', 'breakout_ma10', 'breakout_ma20', 'breakout_ma55',
        'breakout_strength', 'consecutive_new_high'
    ]
    
    log.info("\n【突破特征】")
    for feat in breakout_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 动量特征
    momentum_features = [
        'momentum_5d', 'momentum_10d', 'momentum_20d',
        'momentum_strength', 'momentum_acceleration'
    ]
    
    log.info("\n【动量特征】")
    for feat in momentum_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 价格位置
    position_features = [
        'price_position_34d', 'price_position_55d',
        'price_vs_ma_34d', 'price_vs_ma_55d',
        'dist_to_resistance_20d', 'dist_to_support_20d'
    ]
    
    log.info("\n【价格位置】")
    for feat in position_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 风险特征
    risk_features = [
        'max_drawdown_20d', 'atr_ratio_14', 'volatility_34d'
    ]
    
    log.info("\n【风险特征】")
    for feat in risk_features:
        if feat in last_row:
            val = last_row[feat]
            if not pd.isna(val) and np.isfinite(val):
                log.info(f"  {feat:25s}: {val:>10.4f}")
    
    # 10. 计算特征贡献度（使用SHAP值近似）
    log.info(f"\n{'='*80}")
    log.info("Top20 重要特征及其值")
    log.info("="*80)
    
    # 获取特征重要性并排序
    importance_dict = {}
    for i, feat_name in enumerate(feature_names):
        if feat_name in feature_importance:
            importance_dict[feat_name] = feature_importance[feat_name]
    
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    
    log.info(f"\n{'特征名称':<30} {'重要性':<12} {'特征值':<12} {'归一化值':<12}")
    log.info("-" * 70)
    
    for feat_name, importance in sorted_features:
        feat_idx = feature_names.index(feat_name)
        feat_value = feature_vector[feat_idx]
        
        # 归一化到0-1（简单处理）
        if abs(feat_value) > 1:
            norm_value = min(1.0, abs(feat_value) / 100)
        else:
            norm_value = abs(feat_value)
        
        log.info(f"{feat_name:<30} {importance:<12.2f} {feat_value:<12.4f} {norm_value:<12.4f}")
    
    # 11. 历史表现
    log.info(f"\n{'='*80}")
    log.info("近期表现")
    log.info("="*80)
    
    recent_days = min(10, len(df))
    log.info(f"\n最近{recent_days}天:")
    log.info(f"{'日期':<12} {'收盘':<10} {'涨跌幅':<10} {'成交量':<15} {'RSI_6':<10}")
    log.info("-" * 60)
    
    for i in range(recent_days):
        idx = len(df) - recent_days + i
        row = df.iloc[idx]
        trade_date = row.get('trade_date', '')
        if isinstance(trade_date, pd.Timestamp):
            trade_date = trade_date.strftime('%Y-%m-%d')
        close = row.get('close', 0)
        pct_chg = row.get('pct_chg', 0)
        vol = row.get('vol', 0)
        rsi_6 = row.get('rsi_6', 0)
        
        log.info(f"{str(trade_date):<12} {close:<10.2f} {pct_chg:>+8.2f}% {vol:<15.0f} {rsi_6:<10.2f}")
    
    # 12. 操作建议
    log.info(f"\n{'='*80}")
    log.info("操作建议")
    log.info("="*80)
    
    return_34d = last_row.get('return_34d', 0)
    rsi_6 = last_row.get('rsi_6', 0)
    volume_ratio = last_row.get('volume_ratio', 1)
    breakout_strength = last_row.get('breakout_strength', 0)
    momentum_strength = last_row.get('momentum_strength', 0)
    
    log.info(f"\n【推荐理由】")
    reasons = []
    
    if return_34d < 0:
        reasons.append(f"✓ 符合反追龙头策略：T1前34日涨幅{return_34d:.2f}%，处于低位")
    
    if cal_prob > 0.9:
        reasons.append(f"✓ v2.4.0模型高概率：{cal_prob:.2%}")
    
    if rsi_6 < 70:
        reasons.append(f"✓ RSI未超买：{rsi_6:.1f}")
    elif rsi_6 > 80:
        reasons.append(f"⚠ RSI超买：{rsi_6:.1f}，注意回调风险")
    
    if volume_ratio > 1.5:
        reasons.append(f"✓ 成交量放大：{volume_ratio:.2f}倍")
    
    if breakout_strength > 0.5:
        reasons.append(f"✓ 突破强度较高：{breakout_strength:.2f}")
    
    if momentum_strength > 0:
        reasons.append(f"✓ 动量转正：{momentum_strength:.2f}%")
    
    for reason in reasons:
        log.info(f"  {reason}")
    
    log.info(f"\n【风险提示】")
    risks = []
    
    if return_34d < -15:
        risks.append(f"⚠ T1前跌幅较大：{return_34d:.2f}%，可能存在基本面问题")
    
    if rsi_6 > 80:
        risks.append(f"⚠ RSI严重超买，短期回调概率高")
    
    max_dd = last_row.get('max_drawdown_20d', 0)
    if max_dd < -10:
        risks.append(f"⚠ 近期最大回撤：{max_dd:.2f}%，波动较大")
    
    volatility = last_row.get('volatility_34d', 0)
    if volatility > 5:
        risks.append(f"⚠ 波动率较高：{volatility:.2f}%，风险较大")
    
    if not risks:
        risks.append("✓ 无明显风险信号")
    
    for risk in risks:
        log.info(f"  {risk}")
    
    log.info(f"\n【操作建议】")
    if cal_prob > 0.9 and return_34d < 0 and rsi_6 < 70:
        log.info("  🎯 建议：可以考虑小仓位试探，设置止损")
        log.info("     - 买入：当前价位或回调至支撑位")
        log.info("     - 止损：-5% ~ -8%")
        log.info("     - 止盈：+15% ~ +20%")
    elif cal_prob > 0.8:
        log.info("  📊 建议：观察为主，等待更好的入场时机")
        log.info("     - 等待回调或突破确认")
    else:
        log.info("  ⏸ 建议：暂不操作，继续观察")


def main():
    parser = argparse.ArgumentParser(description='详细分析单只股票')
    parser.add_argument('--ts_code', type=str, required=True, help='股票代码')
    parser.add_argument('--date', type=str, default='20260107', help='预测日期(YYYYMMDD)')
    
    args = parser.parse_args()
    
    analyze_stock(args.ts_code, args.date)


if __name__ == '__main__':
    main()

