#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.1模型预测脚本 - 候选池精简版

特点：
1. 只对候选池股票进行预测（大幅节省时间）
2. 使用v2.3.0的模型和校准器
3. 输出所有候选池股票的v2.3.1概率
4. 用于策略4的信号验证

使用方法：
  python scripts/predict_v231_watchlist.py --watchlist dual_watchlist_20260105.csv --date 20260106
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
    """加载v2.3.0模型"""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v2.3.0' / 'model'
    
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'model.json'))
    
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    calibrator = joblib.load(str(model_dir / 'calibrator.pkl'))
    
    return booster, feature_names, calibrator


def extract_features(df):
    """
    提取特征（v2.3.1版本，包含收益预测特征）
    
    输入: 单只股票的日线数据DataFrame
    输出: 包含所有特征的DataFrame
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
    
    # ========== 多周期特征 ==========
    for period in [8, 34, 55]:
        df[f'return_{period}d'] = df['close'].pct_change(period) * 100
        df[f'ma_{period}d'] = df['close'].rolling(period).mean()
        df[f'price_vs_ma_{period}d'] = (df['close'] - df[f'ma_{period}d']) / df[f'ma_{period}d'] * 100
        df[f'volatility_{period}d'] = df['pct_chg'].rolling(period).std()
        df[f'high_{period}d'] = df['high'].rolling(period).max()
        df[f'low_{period}d'] = df['low'].rolling(period).min()
        price_range = df[f'high_{period}d'] - df[f'low_{period}d']
        df[f'price_position_{period}d'] = (df['close'] - df[f'low_{period}d']) / (price_range + 1e-10)
        df[f'trend_slope_{period}d'] = df['close'].rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0, raw=False
        )
    
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
    
    # ========== 乖离率 ==========
    df['bias_short'] = (df['close'] - df['ma5']) / df['ma5'] * 100
    df['bias_mid'] = (df['close'] - df['ma10']) / df['ma10'] * 100
    df['bias_long'] = (df['close'] - df['ma_20d']) / df['ma_20d'] * 100
    
    # ========== EMA ==========
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
    
    # ========== 量比 ==========
    df['vol_ma5_ratio'] = df['vol'] / (df['vol'].rolling(5).mean() + 1e-8)
    df['vol_ma20_ratio'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-8)
    
    # ========== 涨停 ==========
    df['is_limit_up'] = (df['pct_chg'] >= 9.8).astype(int)
    
    # ========== 历史位置 ==========
    df['price_vs_hist_mean'] = (df['close'] - df['close'].rolling(34).mean()) / df['close'].rolling(34).mean() * 100
    df['price_vs_hist_high'] = (df['close'] - df['close'].rolling(34).max()) / df['close'].rolling(34).max() * 100
    df['volatility_vs_hist'] = df['pct_chg'].rolling(10).std() / (df['pct_chg'].rolling(34).std() + 1e-8)
    
    # ========== 市场相关（占位） ==========
    df['market_pct_chg'] = 0
    df['market_return_34d'] = 0
    df['market_volatility_34d'] = 0
    df['market_trend'] = 0
    df['excess_return'] = df['pct_chg']
    df['excess_return_cumsum'] = df['pct_chg'].rolling(34).sum()
    
    # ========== 风险特征 ==========
    # 最大回撤
    for period in [10, 20, 55]:
        rolling_max = df['close'].rolling(period, min_periods=1).max()
        drawdown = (df['close'] - rolling_max) / rolling_max * 100
        df[f'max_drawdown_{period}d'] = drawdown.rolling(period, min_periods=1).min()
    
    # ATR
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['atr_14'] = true_range.rolling(14, min_periods=1).mean()
    df['atr_ratio_14'] = df['atr_14'] / df['close'] * 100
    atr_mean = df['atr_14'].rolling(55, min_periods=14).mean()
    df['atr_expansion'] = df['atr_14'] / (atr_mean + 1e-10)
    
    # 距高点天数
    for period in [20, 55]:
        rolling_high = df['close'].rolling(period, min_periods=1).max()
        is_at_high = (df['close'] == rolling_high)
        days_list = []
        days_since_high = 0
        for is_high in is_at_high:
            if is_high:
                days_since_high = 0
            else:
                days_since_high += 1
            days_list.append(days_since_high)
        df[f'days_from_high_{period}d'] = days_list
    
    # 恢复比例
    rolling_low_20 = df['close'].rolling(20, min_periods=1).min()
    rolling_high_20 = df['close'].rolling(20, min_periods=1).max()
    price_range = rolling_high_20 - rolling_low_20
    df['recovery_ratio_20d'] = (df['close'] - rolling_low_20) / (price_range + 1e-10)
    
    # ========== v2.3.1新增：收益预测特征 ==========
    # 1. 动量强度（多周期动量的加权平均）
    df['momentum_strength'] = (
        df['momentum_5d'] * 0.3 + 
        df['momentum_10d'] * 0.4 + 
        df['momentum_20d'] * 0.3
    )
    
    # 2. 突破强度（突破多个阻力位的程度）
    breakout_count = (
        df['breakout_high_10d'].astype(int) + 
        df['breakout_high_20d'].astype(int) + 
        df['breakout_high_55d'].astype(int) +
        df['breakout_ma5'].astype(int) +
        df['breakout_ma10'].astype(int) +
        df['breakout_ma20'].astype(int) +
        df['breakout_ma55'].astype(int)
    )
    df['breakout_strength'] = breakout_count / 7.0  # 归一化到0-1
    
    # 3. 成交量放大倍数（相对于均量的倍数）
    vol_ma20 = df['vol'].rolling(20, min_periods=1).mean()
    df['volume_expansion_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
    df['volume_expansion_ratio'] = df['volume_expansion_ratio'].clip(upper=10.0)
    
    # 4. 价格位置评分（在通道中的位置，0-1之间）
    high_20 = df['high'].rolling(20, min_periods=1).max()
    low_20 = df['low'].rolling(20, min_periods=1).min()
    price_range_20 = high_20 - low_20
    df['price_position_score'] = (df['close'] - low_20) / (price_range_20 + 1e-10)
    
    # 5. 综合收益潜力评分（结合多个因子）
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
        start_date = (datetime.strptime(predict_date, '%Y%m%d') - timedelta(days=200)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 提取特征（包含v2.3.1的收益预测特征）
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
        
        # 计算预期收益评分
        expected_return_score = last_row.get('expected_return_score', 0.5)
        if pd.isna(expected_return_score) or not np.isfinite(expected_return_score):
            expected_return_score = 0.5
        
        return {
            'ts_code': ts_code,
            'name': name,
            'raw_probability': round(raw_prob, 4),
            'calibrated_probability': round(cal_prob, 4),
            'expected_return_score': round(expected_return_score, 4),
            'return_34d': round(last_row.get('return_34d', 0), 2),
            'close': round(last_row.get('close', 0), 2)
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description='v2.3.1候选池预测')
    parser.add_argument('--watchlist', type=str, required=True, help='候选池CSV文件路径')
    parser.add_argument('--date', type=str, required=True, help='预测日期(YYYYMMDD)')
    parser.add_argument('--threshold', type=float, default=0.7, help='触发阈值(默认0.7)')
    
    args = parser.parse_args()
    
    predict_date = args.date
    watchlist_file = Path(args.watchlist)
    threshold = args.threshold
    
    log.info("="*80)
    log.info("v2.3.1 候选池预测（精简版）")
    log.info("="*80)
    log.info(f"预测日期: {predict_date}")
    log.info(f"候选池文件: {watchlist_file}")
    log.info(f"触发阈值: {threshold}")
    log.info("")
    
    # 1. 加载候选池
    if not watchlist_file.exists():
        log.error(f"候选池文件不存在: {watchlist_file}")
        return
    
    watchlist = pd.read_csv(watchlist_file)
    log.info(f"候选池股票数: {len(watchlist)} 只")
    
    # 2. 加载模型
    log.info("\n加载v2.3.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型已加载: {len(feature_names)} 个特征")
    
    # 3. 初始化数据管理器
    log.info("\n初始化数据管理器...")
    dm = DataManager()
    
    # 4. 并行预测候选池股票
    log.info("\n开始预测候选池股票...")
    results = []
    total = len(watchlist)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for _, row in watchlist.iterrows():
            ts_code = row['ts_code']
            name = row.get('name', ts_code)
            
            future = executor.submit(
                process_single_stock,
                dm, ts_code, name, predict_date,
                feature_names, booster, calibrator
            )
            futures[future] = (ts_code, name)
        
        completed = 0
        error_count = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                log.info(f"进度: {completed}/{total} ({completed/total*100:.1f}%) | 成功: {len(results)}")
            
            result = future.result()
            if result:
                results.append(result)
            else:
                error_count += 1
    
    if not results:
        log.error("没有预测结果")
        return
    
    # 5. 处理结果
    df_results = pd.DataFrame(results)
    
    # 检查触发信号
    triggered = df_results[df_results['calibrated_probability'] >= threshold]
    
    log.success(f"\n✓ 预测完成: {len(results)} 只股票")
    log.info(f"触发信号(>={threshold}): {len(triggered)} 只")
    
    # 6. 输出触发信号
    if len(triggered) > 0:
        log.info("\n" + "="*80)
        log.info(f"🎯 买入信号触发！({len(triggered)}只)")
        log.info("="*80)
        
        triggered = triggered.sort_values('calibrated_probability', ascending=False)
        
        log.info(f"\n{'代码':<12} {'名称':<10} {'v2.3.1概率':<12} {'预期收益':<10} {'T1前涨幅':<10}")
        log.info("-" * 60)
        
        for _, row in triggered.iterrows():
            log.info(
                f"{row['ts_code']:<12} {row['name']:<10} "
                f"{row['calibrated_probability']:<12.4f} {row['expected_return_score']:<10.4f} "
                f"{row['return_34d']:>+8.1f}%"
            )
    else:
        log.info("\n" + "="*80)
        log.info("📋 候选池状态：等待触发")
        log.info("="*80)
        log.info(f"所有股票v2.3.1概率均 < {threshold}，继续观察...")
        
        # 显示接近触发的股票
        near_trigger = df_results[df_results['calibrated_probability'] >= threshold - 0.1].sort_values('calibrated_probability', ascending=False)
        if len(near_trigger) > 0:
            log.info(f"\n接近触发(>={threshold-0.1})的股票:")
            log.info(f"{'代码':<12} {'名称':<10} {'v2.3.1概率':<12}")
            log.info("-" * 35)
            for _, row in near_trigger.head(10).iterrows():
                log.info(f"{row['ts_code']:<12} {row['name']:<10} {row['calibrated_probability']:<12.4f}")
    
    # 7. 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_file = output_dir / f'v231_watchlist_{predict_date}.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 结果已保存: {output_file}")
    
    # 如果有触发信号，单独保存
    if len(triggered) > 0:
        trigger_file = output_dir / f'v231_triggered_{predict_date}.csv'
        triggered.to_csv(trigger_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 触发信号已保存: {trigger_file}")


if __name__ == '__main__':
    main()

