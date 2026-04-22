#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.5.1模型预测脚本 - 双目标排序版

基于v2.3.1的预测逻辑，使用v2.5.0模型：
1. 使用v2.5.0模型（包含233日均线特征、时间序列划分训练）
2. 双目标排序：0.5*校准概率 + 0.5*预期收益
3. 无惩罚机制（纯模型评分）
4. 适合趋势明确、想要激进追涨的场景
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_model():
    """加载v2.5.0模型"""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v2.5.0' / 'model'
    
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'model.json'))
    
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    calibrator = joblib.load(str(model_dir / 'calibrator.pkl'))
    
    return booster, feature_names, calibrator


def extract_features(df):
    """
    提取特征（v2.5.1版本，包含233日均线特征）
    """
    df = df.copy()
    
    # ========== 基础均线 ==========
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma_20d'] = df['close'].rolling(20).mean()
    
    # ========== MACD ==========
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = df['ema12'] - df['ema26']
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = 2 * (df['macd_dif'] - df['macd_dea'])
    
    # ========== RSI ==========
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['rsi_6'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    gain12 = delta.where(delta > 0, 0).rolling(12).mean()
    loss12 = (-delta.where(delta < 0, 0)).rolling(12).mean()
    df['rsi_12'] = 100 - (100 / (1 + gain12 / (loss12 + 1e-10)))
    
    gain24 = delta.where(delta > 0, 0).rolling(24).mean()
    loss24 = (-delta.where(delta < 0, 0)).rolling(24).mean()
    df['rsi_24'] = 100 - (100 / (1 + gain24 / (loss24 + 1e-10)))
    
    # ========== KDJ ==========
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / (high_9 - low_9 + 1e-10) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # ========== 量比 ==========
    df['volume_ratio'] = df['vol'] / (df['vol'].rolling(5).mean() + 1e-8)
    
    # ========== 多周期特征（包含233日） ==========
    for period in [8, 34, 55, 233]:
        if len(df) >= period:
            df[f'return_{period}d'] = df['close'].pct_change(period) * 100
            df[f'ma_{period}d'] = df['close'].rolling(period).mean()
            df[f'price_vs_ma_{period}d'] = (df['close'] - df[f'ma_{period}d']) / df[f'ma_{period}d'] * 100
            df[f'volatility_{period}d'] = df['pct_chg'].rolling(period).std()
            df[f'high_{period}d'] = df['high'].rolling(period).max()
            df[f'low_{period}d'] = df['low'].rolling(period).min()
            price_range = df[f'high_{period}d'] - df[f'low_{period}d']
            df[f'price_position_{period}d'] = (df['close'] - df[f'low_{period}d']) / (price_range + 1e-10)
    
    # ========== 动量 ==========
    df['momentum_5d'] = df['close'].pct_change(5) * 100
    df['momentum_10d'] = df['close'].pct_change(10) * 100
    df['momentum_20d'] = df['close'].pct_change(20) * 100
    df['momentum_acceleration'] = df['momentum_5d'] - df['momentum_5d'].shift(5)
    
    # ========== 价量关系 ==========
    df['price_change'] = df['close'].diff()
    df['volume_change'] = df['vol'].diff()
    df['volume_price_corr_10d'] = df['close'].rolling(10).corr(df['vol'])
    df['volume_price_corr_20d'] = df['close'].rolling(20).corr(df['vol'])
    df['volume_price_match'] = ((df['price_change'] > 0) & (df['volume_change'] > 0)).astype(int)
    df['volume_price_match_sum_10d'] = df['volume_price_match'].rolling(10).sum()
    
    # ========== 突破特征 ==========
    for period in [10, 20, 55]:
        df[f'prev_high_{period}d'] = df['high'].rolling(period).max().shift(1)
        df[f'breakout_high_{period}d'] = (df['close'] > df[f'prev_high_{period}d']).astype(int)
        df[f'resistance_{period}d'] = df['high'].rolling(period).max()
        df[f'support_{period}d'] = df['low'].rolling(period).min()
        df[f'dist_to_resistance_{period}d'] = (df[f'resistance_{period}d'] - df['close']) / df['close'] * 100
        df[f'dist_to_support_{period}d'] = (df['close'] - df[f'support_{period}d']) / df['close'] * 100
        df[f'support_strength_{period}d'] = (df['low'] - df[f'support_{period}d']).abs().rolling(period).mean()
        df[f'resistance_strength_{period}d'] = (df[f'resistance_{period}d'] - df['high']).abs().rolling(period).mean()
    
    df['channel_width_20d'] = (df['resistance_20d'] - df['support_20d']) / df['close'] * 100
    
    # ========== MA突破 ==========
    df['ma_5d'] = df['close'].rolling(5).mean()
    df['breakout_ma5'] = (df['close'] > df['ma_5d']).astype(int)
    df['ma_10d'] = df['close'].rolling(10).mean()
    df['breakout_ma10'] = (df['close'] > df['ma_10d']).astype(int)
    df['breakout_ma20'] = (df['close'] > df['ma_20d']).astype(int)
    ma_55d = df['close'].rolling(55).mean()
    df['breakout_ma55'] = (df['close'] > ma_55d).astype(int)
    
    df['breakout_volume_ratio'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-8)
    df['high_volume_breakout'] = ((df['breakout_high_20d'] == 1) & (df['breakout_volume_ratio'] > 1.5)).astype(int)
    df['consecutive_new_high'] = df['breakout_high_10d'].rolling(5).sum()
    
    # ========== 成交量趋势 ==========
    df['volume_trend_slope_10d'] = df['vol'].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
    )
    df['volume_trend_slope_20d'] = df['vol'].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False
    )
    df['volume_breakout_count_20d'] = (df['vol'] > df['vol'].rolling(20).mean() * 1.5).rolling(20).sum()
    
    # ========== 量价背离 ==========
    df['price_up_vol_down'] = ((df['price_change'] > 0) & (df['volume_change'] < 0)).astype(int)
    df['price_up_vol_down_count_10d'] = df['price_up_vol_down'].rolling(10).sum()
    df['price_down_vol_up'] = ((df['price_change'] < 0) & (df['volume_change'] > 0)).astype(int)
    df['price_down_vol_up_count_10d'] = df['price_down_vol_up'].rolling(10).sum()
    
    # ========== OBV ==========
    df['obv'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
    df['obv_calc'] = df['obv']
    df['obv_ma10'] = df['obv'].rolling(10).mean()
    df['obv_trend'] = (df['obv'] > df['obv_ma10']).astype(int)
    
    # ========== 成交量RSV ==========
    vol_low_20 = df['vol'].rolling(20).min()
    vol_high_20 = df['vol'].rolling(20).max()
    df['volume_rsv_20d'] = (df['vol'] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100
    
    # ========== 收益预测特征 ==========
    df['momentum_strength'] = (
        df['momentum_5d'] * 0.3 + 
        df['momentum_10d'] * 0.4 + 
        df['momentum_20d'] * 0.3
    )
    
    breakout_count = (
        df['breakout_high_10d'].astype(int) + 
        df['breakout_high_20d'].astype(int) + 
        df['breakout_high_55d'].astype(int) +
        df['breakout_ma5'].astype(int) +
        df['breakout_ma10'].astype(int) +
        df['breakout_ma20'].astype(int) +
        df['breakout_ma55'].astype(int)
    )
    df['breakout_strength'] = breakout_count / 7.0
    
    vol_ma20 = df['vol'].rolling(20, min_periods=1).mean()
    df['volume_expansion_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
    df['volume_expansion_ratio'] = df['volume_expansion_ratio'].clip(upper=10.0)
    
    high_20 = df['high'].rolling(20, min_periods=1).max()
    low_20 = df['low'].rolling(20, min_periods=1).min()
    price_range_20 = high_20 - low_20
    df['price_position_score'] = (df['close'] - low_20) / (price_range_20 + 1e-10)
    
    momentum_norm = (df['momentum_strength'] / 50.0).clip(0, 1)
    volume_norm = (df['volume_expansion_ratio'] / 2.0).clip(0, 1)
    price_vol_match = df['volume_price_match_sum_10d'] / 10.0
    
    df['expected_return_score'] = (
        momentum_norm * 0.3 +
        df['breakout_strength'] * 0.25 +
        volume_norm * 0.2 +
        df['price_position_score'] * 0.15 +
        price_vol_match * 0.1
    )
    
    return df


def process_single_stock(dm, ts_code, name, predict_date, feature_names, booster, calibrator):
    """处理单只股票"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, '%Y%m%d') - timedelta(days=300)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 提取特征
        df = extract_features(df)
        last_row = df.iloc[-1]
        
        # 构建特征向量
        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))
        
        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        raw_prob = float(booster.predict(dmatrix)[0])
        cal_prob = float(calibrator.predict([raw_prob])[0])
        
        # v2.5.1: 双目标排序（0.5*校准概率 + 0.5*预期收益）
        expected_return_score = last_row.get('expected_return_score', 0.5)
        if pd.isna(expected_return_score) or not np.isfinite(expected_return_score):
            expected_return_score = 0.5
        expected_return_norm = float(np.clip(expected_return_score, 0, 1))
        
        # 纯模型评分，无惩罚
        final_score = 0.5 * cal_prob + 0.5 * expected_return_norm
        
        return {
            'ts_code': ts_code,
            'name': name,
            'close': float(last_row['close']),
            'pct_chg': float(last_row.get('pct_chg', 0)),
            'amount': float(last_row.get('amount', 0)),
            'raw_probability': raw_prob,
            'calibrated_probability': cal_prob,
            'expected_return_score': expected_return_score,
            'final_score': final_score,
            'return_34d': float(last_row.get('return_34d', 0)),
            'rsi_6': float(last_row.get('rsi_6', 0)),
            'momentum_strength': float(last_row.get('momentum_strength', 0)),
            'breakout_strength': float(last_row.get('breakout_strength', 0)),
            'volume_expansion_ratio': float(last_row.get('volume_expansion_ratio', 1.0)),
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description='v2.5.1模型预测 - 双目标排序版')
    parser.add_argument('--date', type=str, default='20251231', help='预测日期 (YYYYMMDD)')
    args = parser.parse_args()
    
    predict_date = args.date
    
    log.info("="*80)
    log.info(f"v2.5.1模型预测 - 双目标排序版（无惩罚） - {predict_date}")
    log.info("="*80)
    log.info("特点: 0.5*校准概率 + 0.5*预期收益，纯模型评分，适合激进追涨")
    
    # 初始化
    dm = DataManager()
    
    # 加载模型
    log.info("\n📦 加载v2.5.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征")
    
    # 获取股票列表
    stock_list = dm.get_stock_list()
    valid = stock_list[
        ~stock_list['name'].str.contains('ST|退', na=False) &
        ~stock_list['ts_code'].str.startswith('688') &
        ~stock_list['ts_code'].str.startswith('8')
    ].copy()
    log.info(f"📊 有效股票: {len(valid)} 只")
    
    # 批量处理
    log.info(f"\n🚀 开始预测...")
    results = []
    total = len(valid)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in valid.iterrows():
            future = executor.submit(
                process_single_stock,
                dm, row['ts_code'], row['name'], predict_date,
                feature_names, booster, calibrator
            )
            futures[future] = (row['ts_code'], row['name'])
        
        completed = 0
        error_count = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                log.info(f"进度: {completed}/{total} ({completed/total*100:.1f}%) | 成功: {len(results)}, 失败: {error_count}")
            
            result = future.result()
            if result:
                results.append(result)
            else:
                error_count += 1
    
    if not results:
        log.error("没有预测结果")
        return
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 按final_score排序
    df_results = df_results.sort_values('final_score', ascending=False).reset_index(drop=True)
    
    # Top10
    df_top10 = df_results.head(10)
    
    log.success(f"\n✓ 预测完成: {len(results)} 只股票")
    
    # 显示Top10
    log.info("\n" + "="*90)
    log.info("🏆 v2.5.1 Top10 推荐（双目标排序，无惩罚）")
    log.info("="*90)
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'综合评分':<10} {'校准概率':<10} {'预期收益':<10} {'当日涨幅':<10}")
    log.info("-" * 90)
    
    for i, (_, row) in enumerate(df_top10.iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['final_score']:<10.4f} {row['calibrated_probability']:<10.4f} "
            f"{row['expected_return_score']:<10.4f} {row['pct_chg']:>+9.2f}%"
        )
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'v2.5.1_top10_{predict_date}.csv'
    df_top10.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n💾 Top10结果已保存: {output_file}")
    
    full_output_file = output_dir / f'v2.5.1_full_{predict_date}.csv'
    df_results.to_csv(full_output_file, index=False, encoding='utf-8-sig')
    log.info(f"💾 完整结果已保存: {full_output_file}")
    
    # 统计
    log.info("\n" + "="*80)
    log.info("📊 v2.5.1 统计")
    log.info("="*80)
    chase_high = df_top10[df_top10['pct_chg'] > 9]
    log.info(f"Top10中当日涨幅>9%的股票: {len(chase_high)}/10")
    log.info(f"Top10平均当日涨幅: {df_top10['pct_chg'].mean():.2f}%")
    log.info(f"Top10平均校准概率: {df_top10['calibrated_probability'].mean():.4f}")
    log.info(f"Top10平均预期收益评分: {df_top10['expected_return_score'].mean():.4f}")


if __name__ == '__main__':
    main()
