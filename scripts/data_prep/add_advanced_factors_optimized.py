#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版高级技术因子计算 - 整合所有v2.5.0优化方案

包含：
1. 特征对齐修复（bias、EMA、KDJ、量比等）
2. 233日均线系列特征
3. 市场环境特征
4. 重构突破特征（从二值升级为连续强度）
5. 重构量价特征（经典量价形态识别）
6. Tushare高级技术因子（布林带、CCI、MACD/KDJ优化）
7. 追高风控因子
8. 重构支撑阻力特征
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_kdj_k(df, period=9):
    """计算KDJ的K值"""
    low_n = df['low'].rolling(period).min()
    high_n = df['high'].rolling(period).max()
    rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    return k


def calculate_kdj_d(k):
    """计算KDJ的D值"""
    return k.ewm(com=2, adjust=False).mean()


def calculate_all_optimized_factors(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有优化后的高级技术因子
    
    Args:
        df: 个股日线数据
        market_df: 大盘指数数据（可选），需包含 trade_date, close, pct_chg 列
    """
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    n = len(df)
    if n < 10:
        return df
    
    # 合并市场数据（如果提供）
    if market_df is not None and not market_df.empty:
        market_df = market_df.copy()
        if 'trade_date' in market_df.columns:
            market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = pd.merge(df, market_df, on='trade_date', how='left', suffixes=('', '_market'))
    
    # ==================== 0. 特征对齐修复 ====================
    # 确保正负样本都计算这些特征
    
    # 0.1 乖离率（bias）
    for period in [5, 10, 20, 60]:
        if n >= period:
            ma = df['close'].rolling(period).mean()
            df[f'bias_{period}d'] = (df['close'] - ma) / ma * 100
    
    # 0.2 EMA
    for period in [5, 10, 20, 60]:
        if n >= period:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # 0.3 KDJ
    if n >= 9:
        df['kdj_k'] = calculate_kdj_k(df)
        df['kdj_d'] = calculate_kdj_d(df['kdj_k'])
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 0.4 量比
    if 'vol' in df.columns:
        if n >= 5:
            df['vol_ma5_ratio'] = df['vol'] / (df['vol'].rolling(5).mean() + 1e-8)
        if n >= 20:
            df['vol_ma20_ratio'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-8)
    
    # 0.5 涨停标志
    if 'pct_chg' in df.columns:
        df['is_limit_up'] = (df['pct_chg'] >= 9.8).astype(int)
    
    # ==================== 1. 233日均线系列特征（关键） ====================
    if n >= 233:
        # 1.1 基础233均线
        df['ma_233d'] = df['close'].rolling(233).mean()
        df['ema_233'] = df['close'].ewm(span=233, adjust=False).mean()
        
        # 1.2 价格相对233均线位置
        df['price_vs_ma_233d'] = (df['close'] - df['ma_233d']) / df['ma_233d'] * 100
        df['bias_233d'] = df['price_vs_ma_233d']  # 233日乖离率
        
        # 1.3 突破233均线
        df['breakout_ma233'] = (df['close'] > df['ma_233d']).astype(int)
        df['breakout_ma233_first'] = (
            (df['close'] > df['ma_233d']) & 
            (df['close'].shift(1) <= df['ma_233d'].shift(1))
        ).astype(int)  # 首次突破
        
        # 1.4 5日线与233日线关系（关键信号）
        if 'ma5' not in df.columns and n >= 5:
            df['ma5'] = df['close'].rolling(5).mean()
        if 'ma5' in df.columns:
            df['ma5_above_ma233'] = (df['ma5'] > df['ma_233d']).astype(int)
            df['ma5_cross_ma233'] = (
                (df['ma5'] > df['ma_233d']) & 
                (df['ma5'].shift(1) <= df['ma_233d'].shift(1))
            ).astype(int)  # 5日线上穿233日线
        
        # 1.5 233日均线斜率（长期趋势方向）
        df['ma233_slope'] = df['ma_233d'].diff(20) / (df['ma_233d'].shift(20) + 1e-8) * 100
        
        # 1.6 距离233日新高/新低
        high_233d = df['close'].rolling(233).max()
        low_233d = df['close'].rolling(233).min()
        df['dist_to_233d_high'] = (high_233d - df['close']) / df['close'] * 100
        df['dist_to_233d_low'] = (df['close'] - low_233d) / df['close'] * 100
        
        # 1.7 233日通道位置
        df['position_in_233d_channel'] = (df['close'] - low_233d) / (high_233d - low_233d + 1e-8)
    
    # ==================== 2. 重构突破特征（从二值升级为连续强度） ====================
    for period in [10, 20, 55, 233]:
        if n >= period:
            prev_high = df['close'].rolling(period).max().shift(1)
            # 突破幅度：高于前高多少百分比
            df[f'breakout_pct_{period}d'] = np.where(
                df['close'] > prev_high,
                (df['close'] - prev_high) / prev_high * 100,
                0
            )
    
    # 均线突破强度（价格距均线的相对位置，用ATR标准化）
    for ma_period in [20, 55, 233]:
        if n >= ma_period:
            ma = df['close'].rolling(ma_period).mean()
            # 使用14日ATR或标准差作为标准化
            atr = df['close'].rolling(14).std()
            df[f'ma{ma_period}_breakout_strength'] = (df['close'] - ma) / (atr + 1e-8)
    
    # 多均线排列得分
    if n >= 233:
        ma20 = df['close'].rolling(20).mean()
        ma55 = df['close'].rolling(55).mean()
        ma233 = df['close'].rolling(233).mean()
        df['ma_alignment_score'] = (
            (df['close'] > ma20).astype(int) * 1 +
            (ma20 > ma55).astype(int) * 1 +
            (ma55 > ma233).astype(int) * 1 +
            (df['close'] > ma233).astype(int) * 1
        ) / 4.0  # 归一化到0-1
    
    # 突破有效性（突破后N天内不回落）
    for period in [20, 55]:
        if n >= period + 3:
            breakout = (df['close'] > df['close'].rolling(period).max().shift(1))
            breakout_price = df['close'].where(breakout)
            df[f'breakout_valid_{period}d'] = (
                df['low'].rolling(3).min() > breakout_price.ffill()
            ).astype(int)
    
    # ==================== 3. 重构量价关系特征（经典量价形态识别） ====================
    if 'vol' in df.columns and n >= 20:
        vol_ma20 = df['vol'].rolling(20).mean()
        vol_ratio = df['vol'] / (vol_ma20 + 1e-8)  # 量比
        price_pct = df['pct_chg'] if 'pct_chg' in df.columns else df['close'].pct_change() * 100
        
        # 3.1 经典量价形态识别
        # 放量上涨（健康上涨信号）
        df['vol_up_rise'] = np.where(
            (price_pct > 2) & (vol_ratio > 1.5), 
            vol_ratio,  # 放量程度
            0
        )
        df['vol_up_rise_count_10d'] = (df['vol_up_rise'] > 0).rolling(10).sum()
        
        # 放量不涨（警示信号）
        df['vol_up_flat'] = np.where(
            (price_pct >= -2) & (price_pct <= 2) & (vol_ratio > 1.5),
            vol_ratio,
            0
        )
        df['vol_up_flat_count_10d'] = (df['vol_up_flat'] > 0).rolling(10).sum()
        
        # 缩量下跌（洗盘信号）
        df['vol_down_drop'] = np.where(
            (price_pct < -2) & (vol_ratio < 0.7),
            1,
            0
        )
        df['vol_down_drop_count_10d'] = df['vol_down_drop'].rolling(10).sum()
        
        # 缩量不涨（蓄势信号）
        df['vol_down_flat'] = np.where(
            (price_pct >= -2) & (price_pct <= 2) & (vol_ratio < 0.7),
            1,
            0
        )
        df['vol_down_flat_count_10d'] = df['vol_down_flat'].rolling(10).sum()
        
        # 放量下跌（危险信号）
        df['vol_up_drop'] = np.where(
            (price_pct < -2) & (vol_ratio > 1.5),
            vol_ratio,
            0
        )
        df['vol_up_drop_count_10d'] = (df['vol_up_drop'] > 0).rolling(10).sum()
        
        # 3.2 底部盘整识别
        price_range_10d = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close'] * 100
        df['price_range_10d'] = price_range_10d
        
        low_20d = df['low'].rolling(20).min()
        high_20d = df['high'].rolling(20).max()
        price_position = (df['close'] - low_20d) / (high_20d - low_20d + 1e-8)
        
        df['bottom_consolidation'] = np.where(
            (price_range_10d < 10) & (price_position < 0.3),
            1,
            0
        )
        df['bottom_consolidation_days'] = df['bottom_consolidation'].rolling(20).sum()
        
        # 缩量盘整
        vol_shrink = vol_ratio < 0.8
        df['vol_shrink_consolidation'] = np.where(
            (price_range_10d < 10) & vol_shrink,
            1,
            0
        )
        df['vol_shrink_consolidation_days'] = df['vol_shrink_consolidation'].rolling(20).sum()
        
        # 3.3 健康量价得分
        df['healthy_vol_price_score'] = (
            df['vol_up_rise_count_10d'] * 2 +      # 放量上涨（权重2）
            df['vol_down_drop_count_10d'] * 1 -    # 缩量下跌（权重1）
            df['vol_up_drop_count_10d'] * 2 -      # 放量下跌（权重-2）
            df['vol_up_flat_count_10d'] * 1        # 放量不涨（权重-1）
        ) / 10.0  # 归一化
        
        # 3.4 量价趋势一致性
        price_up = price_pct > 0
        vol_up = df['vol'] > df['vol'].shift(1)
        vol_price_same_dir = (price_up == vol_up).astype(int)
        df['vol_price_consistency_10d'] = vol_price_same_dir.rolling(10).mean()
        
        # 3.5 OBV和资金流向增强
        obv = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        obv_ma20 = obv.rolling(20).mean()
        obv_std = obv.rolling(20).std()
        df['obv_zscore'] = (obv - obv_ma20) / (obv_std + 1e-8)
        
        if 'amount' in df.columns:
            up_amount = np.where(price_pct > 0, df['amount'], 0)
            df['money_flow_ratio_10d'] = (
                pd.Series(up_amount).rolling(10).sum() / 
                (df['amount'].rolling(10).sum() + 1e-8)
            )
        
        # 3.6 放量突破确认
        breakout_20d = df['close'] > df['close'].rolling(20).max().shift(1)
        df['vol_breakout_confirm'] = np.where(
            breakout_20d & (vol_ratio > 1.5),
            vol_ratio,
            0
        )
    
    # ==================== 4. Tushare高级技术因子 ====================
    # 4.1 布林带 (BOLL)
    if n >= 20:
        boll_period = 20
        df['boll_mid'] = df['close'].rolling(boll_period).mean()
        boll_std = df['close'].rolling(boll_period).std()
        df['boll_upper'] = df['boll_mid'] + 2 * boll_std
        df['boll_lower'] = df['boll_mid'] - 2 * boll_std
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid'] * 100
        df['boll_pctb'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-8)
        df['boll_breakout_upper'] = (df['close'] > df['boll_upper']).astype(int)
        df['boll_breakout_lower'] = (df['close'] < df['boll_lower']).astype(int)
    
    # 4.2 CCI (顺势指标)
    if n >= 14:
        cci_period = 14
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(cci_period).mean()
        tp_md = (tp - tp_ma).abs().rolling(cci_period).mean()
        df['cci'] = (tp - tp_ma) / (0.015 * tp_md + 1e-8)
        df['cci_overbought'] = (df['cci'] > 100).astype(int)
        df['cci_oversold'] = (df['cci'] < -100).astype(int)
    
    # 4.3 MACD优化
    if n >= 26:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = 2 * (df['macd_dif'] - df['macd_dea'])
        df['macd_golden_cross'] = ((df['macd_dif'] > df['macd_dea']) & 
                                    (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))).astype(int)
        df['macd_death_cross'] = ((df['macd_dif'] < df['macd_dea']) & 
                                   (df['macd_dif'].shift(1) >= df['macd_dea'].shift(1))).astype(int)
        df['macd_histogram_trend'] = np.sign(df['macd'].diff())
        df['macd_histogram_accel'] = df['macd'].diff().diff()
    
    # 4.4 KDJ优化（如果还没计算）
    if 'kdj_k' not in df.columns and n >= 9:
        df['kdj_k'] = calculate_kdj_k(df)
        df['kdj_d'] = calculate_kdj_d(df['kdj_k'])
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    if 'kdj_k' in df.columns:
        df['kdj_golden_cross'] = ((df['kdj_k'] > df['kdj_d']) & 
                                   (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))).astype(int)
        df['kdj_death_cross'] = ((df['kdj_k'] < df['kdj_d']) & 
                                  (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))).astype(int)
    
    # 4.5 RSI多周期和背离检测
    for period in [6, 12, 24]:
        if n >= period:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            df[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    if 'rsi_6' in df.columns and n >= 20:
        price_new_high = df['close'] == df['close'].rolling(20).max()
        rsi_not_high = df['rsi_6'] < df['rsi_6'].rolling(20).max()
        df['rsi_bearish_divergence'] = (price_new_high & rsi_not_high).astype(int)
        
        price_new_low = df['close'] == df['close'].rolling(20).min()
        rsi_not_low = df['rsi_6'] > df['rsi_6'].rolling(20).min()
        df['rsi_bullish_divergence'] = (price_new_low & rsi_not_low).astype(int)
    
    # ==================== 5. 追高风控因子 ====================
    if 'pct_chg' in df.columns:
        # 5.1 短期涨幅风险
        up_streak = (df['pct_chg'] > 0).astype(int)
        # 计算连续上涨天数
        consecutive = []
        count = 0
        for val in up_streak:
            if val == 1:
                count += 1
            else:
                count = 0
            consecutive.append(count)
        df['consecutive_up_days'] = pd.Series(consecutive) * up_streak  # 下跌时为0
        
        df['surge_3d'] = (df['close'] / df['close'].shift(3) - 1) * 100
        df['surge_5d'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['surge_10d'] = (df['close'] / df['close'].shift(10) - 1) * 100
        df['high_surge_risk'] = ((df['surge_3d'] > 15) | (df['surge_5d'] > 25)).astype(int)
        
        # 5.2 均线乖离风险（如果还没计算）
        for period in [5, 10, 20, 60]:
            if f'bias_{period}d' not in df.columns and n >= period:
                ma = df['close'].rolling(period).mean()
                df[f'bias_{period}d'] = (df['close'] - ma) / ma * 100
        
        if all(f'bias_{p}d' in df.columns for p in [5, 10, 20]):
            df['high_bias_risk'] = (
                (df['bias_5d'] > 10) | 
                (df['bias_10d'] > 15) | 
                (df['bias_20d'] > 20)
            ).astype(int)
        
        # 5.3 技术指标超买风险
        rsi_overbought = (df['rsi_6'] > 80).astype(int) if 'rsi_6' in df.columns else pd.Series([0] * n)
        kdj_overbought = ((df['kdj_k'] > 80) & (df['kdj_d'] > 80)).astype(int) if all(c in df.columns for c in ['kdj_k', 'kdj_d']) else pd.Series([0] * n)
        cci_overbought = df['cci_overbought'] if 'cci_overbought' in df.columns else pd.Series([0] * n)
        
        df['tech_overbought_risk'] = (
            rsi_overbought + kdj_overbought + cci_overbought
        ) / 3.0
        
        # 5.4 相对历史位置风险
        for period in [20, 55, 120]:
            if n >= period:
                hist_high = df['close'].rolling(period).max()
                hist_low = df['close'].rolling(period).min()
                df[f'hist_position_{period}d'] = (df['close'] - hist_low) / (hist_high - hist_low + 1e-8)
        
        if 'hist_position_120d' in df.columns:
            df['near_hist_high_risk'] = (df['hist_position_120d'] > 0.8).astype(int)
        
        # 5.5 综合追高风险评分
        df['chasing_risk_score'] = (
            df['high_surge_risk'] * 0.3 +
            df.get('high_bias_risk', pd.Series([0] * n)) * 0.2 +
            df['tech_overbought_risk'] * 0.2 +
            df.get('near_hist_high_risk', pd.Series([0] * n)) * 0.15 +
            (df['consecutive_up_days'] > 5).astype(int) * 0.15
        )
        
        # 追高风险等级
        df['chasing_risk_level'] = pd.cut(
            df['chasing_risk_score'], 
            bins=[-0.01, 0.3, 0.6, 1.01], 
            labels=[0, 1, 2]
        ).astype(int)
    
    # ==================== 6. 重构支撑阻力特征 ====================
    for period in [20, 55]:
        if n >= period:
            support = df['low'].rolling(period).min()
            resistance = df['high'].rolling(period).max()
            
            # 用ATR标准化距离
            atr = df['close'].rolling(14).std()
            df[f'support_dist_atr_{period}d'] = (df['close'] - support) / (atr + 1e-8)
            df[f'resistance_dist_atr_{period}d'] = (resistance - df['close']) / (atr + 1e-8)
            
            # 通道位置
            channel = resistance - support
            df[f'channel_position_{period}d'] = (df['close'] - support) / (channel + 1e-8)
    
    # 近期新高/新低频率
    for period in [10, 20]:
        if n >= period:
            new_high = df['close'] == df['close'].rolling(period).max()
            new_low = df['close'] == df['close'].rolling(period).min()
            df[f'new_high_freq_{period}d'] = new_high.rolling(period).sum() / period
            df[f'new_low_freq_{period}d'] = new_low.rolling(period).sum() / period
    
    # 价格趋势加速度
    if n >= 15:
        ma10 = df['close'].rolling(10).mean()
        ma10_slope = ma10.diff(5) / (ma10.shift(5) + 1e-8) * 100
        df['trend_acceleration'] = ma10_slope.diff(5)
    
    # ==================== 7. 市场环境特征 ====================
    if market_df is not None and not market_df.empty:
        # 7.1 大盘涨跌幅（如果已合并）
        if 'pct_chg_market' in df.columns or 'market_pct_chg' in df.columns:
            market_pct_col = 'pct_chg_market' if 'pct_chg_market' in df.columns else 'market_pct_chg'
            df['market_pct_chg'] = df[market_pct_col]
        elif 'close_market' in df.columns or 'market_close' in df.columns:
            market_close_col = 'close_market' if 'close_market' in df.columns else 'market_close'
            df['market_pct_chg'] = df[market_close_col].pct_change() * 100
        
        # 7.2 大盘多周期收益率
        if 'market_close' in df.columns or 'close_market' in df.columns:
            market_close_col = 'close_market' if 'close_market' in df.columns else 'market_close'
            for period in [5, 20, 34]:
                if n >= period:
                    df[f'market_return_{period}d'] = df[market_close_col].pct_change(period) * 100
        
        # 7.3 大盘波动率
        if 'market_pct_chg' in df.columns:
            if n >= 20:
                df['market_volatility_20d'] = df['market_pct_chg'].rolling(20).std()
        
        # 7.4 个股相对大盘强度（超额收益）
        if 'pct_chg' in df.columns and 'market_pct_chg' in df.columns:
            df['excess_return_1d'] = df['pct_chg'] - df['market_pct_chg']
        
        if 'close' in df.columns and 'market_close' in df.columns:
            market_close_col = 'market_close' if 'market_close' in df.columns else 'close_market'
            for period in [5, 20, 34]:
                if n >= period:
                    stock_return = df['close'].pct_change(period) * 100
                    market_return = df[market_close_col].pct_change(period) * 100
                    df[f'excess_return_{period}d'] = stock_return - market_return
        
        # 7.5 相对强度指标（RS）
        if 'close' in df.columns and 'market_close' in df.columns:
            market_close_col = 'market_close' if 'market_close' in df.columns else 'close_market'
            if n >= 20:
                stock_return_20d = df['close'].pct_change(20) * 100
                market_return_20d = df[market_close_col].pct_change(20) * 100
                df['relative_strength_20d'] = np.where(
                    market_return_20d != 0,
                    stock_return_20d / (market_return_20d + 1e-8),
                    1
                )
        
        # 7.6 市场情绪代理（使用波动率变化）
        if 'market_volatility_20d' in df.columns and n >= 60:
            df['market_fear_indicator'] = df['market_volatility_20d'] / df['market_volatility_20d'].rolling(60).mean()
    
    # ==================== 8. 保留原有的多周期特征 ====================
    for tf in [8, 34, 55]:
        if n >= tf:
            if f'return_{tf}d' not in df.columns:
                df[f'return_{tf}d'] = (df['close'] - df['close'].shift(tf)) / df['close'].shift(tf) * 100
            if f'ma_{tf}d' not in df.columns:
                df[f'ma_{tf}d'] = df['close'].rolling(tf).mean()
            if f'price_vs_ma_{tf}d' not in df.columns:
                df[f'price_vs_ma_{tf}d'] = (df['close'] - df[f'ma_{tf}d']) / df[f'ma_{tf}d'] * 100
            if f'volatility_{tf}d' not in df.columns:
                df[f'volatility_{tf}d'] = df['pct_chg'].rolling(tf).std() if 'pct_chg' in df.columns else df['close'].pct_change().rolling(tf).std()
    
    # ==================== 9. 动量因子 ====================
    for period in [5, 10, 20]:
        if n >= period:
            if f'momentum_{period}d' not in df.columns:
                df[f'momentum_{period}d'] = df['close'].pct_change(period) * 100
    
    if 'momentum_10d' in df.columns and n >= 15:
        df['momentum_acceleration'] = df['momentum_10d'].diff(5)
    
    return df


def process_sample_batch(
    dm: DataManager,
    features_df: pd.DataFrame,
    sample_ids: list,
    max_lookback: int = 233,  # 增加到233以支持233日均线
    market_index_code: str = '000001.SH'  # 上证指数
) -> list:
    """
    批量处理样本（使用优化后的特征计算，包含市场环境特征）
    
    Args:
        dm: DataManager实例
        features_df: 特征数据DataFrame
        sample_ids: 样本ID列表
        max_lookback: 最大回看天数
        market_index_code: 市场指数代码（默认上证指数）
    """
    results = []
    
    # 获取市场数据（一次性获取，避免重复调用）
    market_data_cache = {}
    
    for sample_id in sample_ids:
        sample_data = features_df[features_df['sample_id'] == sample_id].copy()
        
        if sample_data.empty:
            continue
        
        ts_code = sample_data['ts_code'].iloc[0]
        
        try:
            # 获取日线数据
            min_date = pd.to_datetime(sample_data['trade_date']).min()
            max_date = pd.to_datetime(sample_data['trade_date']).max()
            
            extended_start = (min_date - timedelta(days=max_lookback + 30)).strftime('%Y%m%d')
            end_date = max_date.strftime('%Y%m%d')
            
            df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
            
            if df_daily is None or df_daily.empty:
                results.append(sample_data)
                continue
            
            if 'trade_date' not in df_daily.columns:
                results.append(sample_data)
                continue
            
            # 获取市场数据（使用缓存，避免重复API调用）
            # 注意：DataManager内部已有缓存机制，这里只是进程内缓存
            date_key = f"{extended_start}_{end_date}"
            if date_key not in market_data_cache:
                try:
                    market_df = dm.get_index_daily(market_index_code, extended_start, end_date)
                    if market_df is not None and not market_df.empty:
                        market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
                        # 重命名列以便合并
                        market_df = market_df.rename(columns={
                            'close': 'market_close',
                            'pct_chg': 'market_pct_chg'
                        })
                        market_data_cache[date_key] = market_df
                    else:
                        market_data_cache[date_key] = None
                except Exception as e:
                    log.warning(f"获取市场数据失败: {e}")
                    market_data_cache[date_key] = None
            
            market_df = market_data_cache.get(date_key)
            
            # 计算优化后的高级因子（包含市场环境特征）
            df_with_factors = calculate_all_optimized_factors(df_daily, market_df)
            df_with_factors['trade_date'] = pd.to_datetime(df_with_factors['trade_date'])
            
            # 筛选新增的列
            new_cols = [c for c in df_with_factors.columns if c not in sample_data.columns and c != 'trade_date']
            
            if not new_cols:
                results.append(sample_data)
                continue
            
            # 合并
            sample_data['trade_date'] = pd.to_datetime(sample_data['trade_date'])
            merged = pd.merge(
                sample_data,
                df_with_factors[['trade_date'] + new_cols],
                on='trade_date',
                how='left'
            )
            results.append(merged)
            
        except Exception as e:
            log.warning(f"处理样本 {sample_id} 时出错: {e}")
            results.append(sample_data)
    
    return results


def add_optimized_factors_with_checkpoint(
    input_file: str,
    output_file: str,
    checkpoint_file: str,
    dm: DataManager,
    batch_size: int = 100
):
    """
    带断点续传的优化因子添加
    """
    log.info("="*80)
    log.info(f"处理文件: {input_file}")
    log.info("="*80)
    
    # 加载数据
    df = pd.read_csv(input_file)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    all_sample_ids = df['sample_id'].unique().tolist()
    total_samples = len(all_sample_ids)
    
    log.info(f"总样本数: {total_samples}")
    
    # 检查是否有断点
    processed_ids = set()
    processed_results = []
    
    if os.path.exists(checkpoint_file):
        log.info(f"发现断点文件，加载已处理的数据...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df['sample_id'].unique())
        processed_results.append(checkpoint_df)
        log.info(f"已处理: {len(processed_ids)} 个样本")
    
    # 筛选未处理的样本
    remaining_ids = [sid for sid in all_sample_ids if sid not in processed_ids]
    log.info(f"待处理: {len(remaining_ids)} 个样本")
    
    if not remaining_ids:
        log.success("所有样本已处理完成！")
        if processed_results:
            final_df = pd.concat(processed_results, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            log.success(f"✓ 结果已保存: {output_file}")
        return
    
    # 批量处理
    batch_results = processed_results.copy()
    
    for i in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[i:i+batch_size]
        current_batch = i // batch_size + 1
        total_batches = (len(remaining_ids) + batch_size - 1) // batch_size
        
        log.info(f"\n处理批次 {current_batch}/{total_batches} ({len(batch_ids)} 个样本)")
        
        batch_df = df[df['sample_id'].isin(batch_ids)]
        batch_result = process_sample_batch(dm, batch_df, batch_ids)
        
        if batch_result:
            batch_df_result = pd.concat(batch_result, ignore_index=True)
            batch_results.append(batch_df_result)
            
            # 保存断点
            checkpoint_df = pd.concat(batch_results, ignore_index=True)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            log.info(f"✓ 断点已保存 (累计: {checkpoint_df['sample_id'].nunique()} 个样本)")
        
        # 进度
        progress = (len(processed_ids) + i + len(batch_ids)) / total_samples * 100
        log.info(f"总进度: {progress:.1f}%")
        
        # 短暂休息避免API限制
        time.sleep(0.5)
    
    # 保存最终结果
    if batch_results:
        final_df = pd.concat(batch_results, ignore_index=True)
        final_df = final_df.ffill().bfill()
        final_df.to_csv(output_file, index=False)
        
        new_cols = len(final_df.columns) - len(df.columns)
        log.success(f"✓ 处理完成！新增 {new_cols} 个因子")
        log.success(f"✓ 结果已保存: {output_file}")
        
        # 清理断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            log.info("✓ 断点文件已清理")


def main():
    log.info("="*80)
    log.info("为特征数据添加优化后的高级技术因子（断点续传版）")
    log.info("="*80)
    log.info("\n优化内容：")
    log.info("  1. 特征对齐修复（bias、EMA、KDJ、量比等）")
    log.info("  2. 233日均线系列特征")
    log.info("  3. 重构突破特征（连续强度）")
    log.info("  4. 重构量价特征（经典形态识别）")
    log.info("  5. Tushare高级技术因子（布林带、CCI、MACD/KDJ优化）")
    log.info("  6. 追高风控因子")
    log.info("  7. 重构支撑阻力特征")
    log.info("="*80)
    
    # 文件路径（使用v5版本）
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv'
    
    # 如果v5不存在，fallback到v4
    if not pos_input.exists():
        pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v4.csv'
    if not neg_input.exists():
        neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v4.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_optimized.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_optimized.csv'
    
    pos_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'processed' / '.checkpoint_pos_optimized.csv'
    neg_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'features' / '.checkpoint_neg_optimized.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本
    if os.path.exists(pos_output):
        log.success(f"\n[步骤2] 正样本优化特征已完成，跳过")
        log.info(f"   输出文件: {pos_output}")
        if os.path.exists(pos_checkpoint):
            os.remove(pos_checkpoint)
            log.info("   ✓ 已清理正样本checkpoint")
    else:
        log.info("\n[步骤2] 处理正样本优化特征...")
        add_optimized_factors_with_checkpoint(
            str(pos_input), str(pos_output), str(pos_checkpoint), dm
        )
    
    # 处理负样本
    if os.path.exists(neg_output):
        log.success(f"\n[步骤3] 负样本优化特征已完成，跳过")
        log.info(f"   输出文件: {neg_output}")
        if os.path.exists(neg_checkpoint):
            os.remove(neg_checkpoint)
            log.info("   ✓ 已清理负样本checkpoint")
    else:
        log.info("\n[步骤3] 处理负样本优化特征...")
        add_optimized_factors_with_checkpoint(
            str(neg_input), str(neg_output), str(neg_checkpoint), dm
        )
    
    log.info("\n" + "="*80)
    log.success("✅ 优化后的高级技术因子添加完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()
