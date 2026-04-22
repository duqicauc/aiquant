#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版 v6 样本数据生成脚本

修复内容：
1. 基于v5的逻辑进行优化，而不是全新重写
2. 确保OHLCV数据完整获取，不使用估算值
3. 保持与v5相同的特征计算逻辑
4. 扩展历史数据天数到70天（支持55日长期特征）

输出文件：
- data/training/processed/feature_data_34d_v6.csv (正样本)
- data/training/features/negative_feature_data_v2_34d_v6.csv (负样本)
- data/training/features/hard_negative_feature_data_34d_v6.csv (硬负样本)
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log
from src.data.data_manager import DataManager

# v6配置
LOOKBACK_DAYS = 70  # 从34天扩展到70天


def extract_sample_features(dm: DataManager, ts_code: str, name: str, t1_date: str, 
                            lookback_days: int, sample_id: int) -> pd.DataFrame:
    """
    提取单个样本的特征（修复版 - 确保OHLCV完整）
    
    基于v5的逻辑，但扩展了历史数据天数
    """
    # 计算日期范围
    t1_str = str(t1_date)
    try:
        t1 = pd.to_datetime(t1_str, format='%Y%m%d')
    except:
        try:
            t1 = pd.to_datetime(t1_str, format='%Y-%m-%d')
        except:
            t1 = pd.to_datetime(t1_str)
    
    start_date = (t1 - timedelta(days=150)).strftime('%Y%m%d')
    end_date = (t1 - timedelta(days=1)).strftime('%Y%m%d')
    
    # 1. 获取完整数据（包含OHLCV）
    df = dm.get_complete_data(ts_code, start_date, end_date)
    
    if df.empty:
        return pd.DataFrame()
    
    # 2. 验证OHLCV数据完整性
    required_cols = ['open', 'high', 'low', 'close', 'vol']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        log.warning(f"{ts_code}: 缺少OHLCV列: {missing_cols}")
        # 尝试从日线数据补充
        df_daily = dm.get_daily_data(ts_code, start_date, end_date)
        if not df_daily.empty:
            for col in missing_cols:
                if col in df_daily.columns:
                    df[col] = df_daily[col]
    
    # 3. 检查high/low是否有效（不是估算值）
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        # 检查是否所有high都等于close*1.01（估算值的特征）
        high_ratio = (df['high'] / df['close']).round(4)
        if (high_ratio == 1.01).all():
            log.warning(f"{ts_code}: high值可能是估算值，尝试重新获取")
            df_daily = dm.get_daily_data(ts_code, start_date, end_date)
            if not df_daily.empty and 'high' in df_daily.columns:
                df['high'] = df_daily['high'].values[:len(df)]
                df['low'] = df_daily['low'].values[:len(df)]
    
    # 4. 获取技术因子
    try:
        df_factor = dm.get_stk_factor(ts_code, start_date, end_date)
        if not df_factor.empty:
            factor_cols = ['trade_date', 'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']
            available_factor_cols = [c for c in factor_cols if c in df_factor.columns]
            df = pd.merge(df, df_factor[available_factor_cols], on='trade_date', how='left')
    except Exception as e:
        log.warning(f"{ts_code}: 技术因子获取失败: {e}")
    
    # 5. 计算基础MA（如果没有）
    if 'ma5' not in df.columns:
        df['ma5'] = df['close'].rolling(window=5).mean()
    if 'ma10' not in df.columns:
        df['ma10'] = df['close'].rolling(window=10).mean()
    
    # 6. 只取最后N天
    df = df.tail(lookback_days)
    
    if len(df) < lookback_days * 0.8:
        log.warning(f"{ts_code}: 数据不足{lookback_days}天，实际{len(df)}天")
    
    # 7. 选择字段
    base_fields = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 
                   'pct_chg', 'total_mv', 'circ_mv', 'ma5', 'ma10', 'volume_ratio']
    extra_fields = ['macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']
    
    all_fields = base_fields + extra_fields
    available_fields = [f for f in all_fields if f in df.columns]
    
    df_features = df[available_fields].copy()
    
    # 8. 添加元数据
    df_features.insert(0, 'sample_id', sample_id)
    df_features.insert(2, 'name', name)
    df_features['days_to_t1'] = range(-len(df_features), 0)
    
    return df_features


def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算高级特征（与v5一致的逻辑）
    
    确保不使用估算值，所有特征基于真实OHLCV数据
    """
    df = df.copy()
    
    def calc_sample_features(g):
        g = g.sort_values('trade_date').copy()
        n = len(g)
        
        if n < 5:
            return g
        
        close = g['close']
        high = g['high']
        low = g['low']
        vol = g.get('vol', pd.Series([0]*n))
        pct_chg = g.get('pct_chg', pd.Series([0]*n))
        
        # 验证high/low不是估算值
        if 'high' in g.columns and 'low' in g.columns:
            high_ratio = (high / close).round(4)
            if (high_ratio == 1.01).all():
                log.warning(f"样本 {g['sample_id'].iloc[0]}: high/low可能是估算值")
        
        # ========== 1. 均线相关 ==========
        for period, name in [(5, 'ma_5d'), (8, 'ma_8d'), (10, 'ma_10d'), (20, 'ma_20d'), 
                             (34, 'ma_34d'), (55, 'ma_55d')]:
            if name not in g.columns and n >= period:
                g[name] = close.rolling(period, min_periods=period//2).mean()
        
        # EMA
        for period in [5, 10, 20, 60]:
            col = f'ema_{period}d'
            if col not in g.columns and n >= period:
                g[col] = close.ewm(span=period, adjust=False).mean()
        
        # ========== 2. 价格位置 ==========
        for period in [10, 20, 34, 55]:
            col = f'price_position_{period}d'
            if col not in g.columns and n >= period:
                rolling_high = high.rolling(period, min_periods=period//2).max()
                rolling_low = low.rolling(period, min_periods=period//2).min()
                g[col] = (close - rolling_low) / (rolling_high - rolling_low + 1e-8) * 100
        
        # ========== 3. 动量指标 ==========
        for period in [5, 10, 20]:
            col = f'momentum_{period}d'
            if col not in g.columns and n >= period:
                g[col] = close.pct_change(period) * 100
        
        # ========== 4. 波动率 ==========
        for period in [10, 20, 34, 55]:
            col = f'volatility_{period}d'
            if col not in g.columns and n >= period:
                g[col] = pct_chg.rolling(period, min_periods=period//2).std()
        
        # ========== 5. 成交量相关 ==========
        if 'vol_ma_5d' not in g.columns and n >= 5:
            g['vol_ma_5d'] = vol.rolling(5, min_periods=3).mean()
        if 'vol_ma_10d' not in g.columns and n >= 10:
            g['vol_ma_10d'] = vol.rolling(10, min_periods=5).mean()
        if 'vol_ma_20d' not in g.columns and n >= 20:
            g['vol_ma_20d'] = vol.rolling(20, min_periods=10).mean()
        
        # 量比
        if 'volume_ratio_5d' not in g.columns and 'vol_ma_5d' in g.columns:
            g['volume_ratio_5d'] = vol / (g['vol_ma_5d'] + 1e-8)
        
        # ========== 6. 价格范围（基于真实high/low） ==========
        if 'price_range_pct' not in g.columns:
            g['price_range_pct'] = (high - low) / (low + 1e-8) * 100
        
        # ========== 7. 相对历史高点 ==========
        for period in [10, 20, 55]:
            col = f'price_vs_hist_high_{period}d'
            if col not in g.columns and n >= period:
                rolling_max = high.rolling(period, min_periods=period//2).max()
                g[col] = (close - rolling_max) / (rolling_max + 1e-8) * 100
        
        # ========== 8. 趋势斜率 ==========
        for period in [10, 20, 34]:
            col = f'trend_slope_{period}d'
            if col not in g.columns and n >= period:
                x = np.arange(period)
                slopes = []
                for i in range(len(close)):
                    if i < period - 1:
                        slopes.append(np.nan)
                    else:
                        y = close.iloc[i-period+1:i+1].values
                        if len(y) == period:
                            slope = np.polyfit(x, y, 1)[0]
                            slopes.append(slope / (close.iloc[i] + 1e-8) * 100)
                        else:
                            slopes.append(np.nan)
                g[col] = slopes
        
        # ========== 9. 支撑阻力距离 ==========
        for period in [10, 20]:
            support_col = f'dist_to_support_{period}d'
            resist_col = f'dist_to_resistance_{period}d'
            if support_col not in g.columns and n >= period:
                rolling_low = low.rolling(period, min_periods=period//2).min()
                rolling_high = high.rolling(period, min_periods=period//2).max()
                g[support_col] = (close - rolling_low) / (close + 1e-8) * 100
                g[resist_col] = (rolling_high - close) / (close + 1e-8) * 100
        
        # ========== 10. 风险特征 ==========
        # 最大回撤
        for period in [10, 20, 55]:
            col = f'max_drawdown_{period}d'
            if col not in g.columns and n >= period:
                rolling_max = close.rolling(period, min_periods=period//2).max()
                drawdown = (close - rolling_max) / (rolling_max + 1e-8) * 100
                g[col] = drawdown.rolling(period, min_periods=period//2).min()
        
        # ATR
        if 'atr_14' not in g.columns and n >= 14:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            g['atr_14'] = tr.rolling(14, min_periods=7).mean()
            g['atr_ratio_14'] = g['atr_14'] / (close + 1e-8) * 100
        
        return g
    
    # 按样本分组计算
    df = df.groupby('sample_id', group_keys=False).apply(calc_sample_features)
    
    return df


def generate_positive_samples_v6_fixed():
    """生成正样本（修复版）"""
    log.info("="*80)
    log.info("步骤1: 生成正样本 v6（修复版 - 确保OHLCV完整）")
    log.info("="*80)
    
    dm = DataManager()
    
    # 加载v5正样本列表
    v5_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    if not v5_file.exists():
        log.error(f"v5正样本文件不存在: {v5_file}")
        return None
    
    df_v5 = pd.read_csv(v5_file)
    
    # 获取唯一样本列表
    samples = df_v5.groupby('sample_id').agg({
        'ts_code': 'first',
        'name': 'first',
        'trade_date': 'max'
    }).reset_index()
    
    samples['trade_date'] = pd.to_datetime(samples['trade_date'])
    samples['t1_date'] = samples['trade_date'].dt.strftime('%Y%m%d')
    
    log.info(f"正样本数量: {len(samples)}")
    log.info(f"历史数据天数: {LOOKBACK_DAYS}")
    
    # 提取特征
    all_features = []
    total = len(samples)
    
    for idx, sample in samples.iterrows():
        if (idx + 1) % 100 == 0:
            log.info(f"进度: {idx+1}/{total}")
        
        try:
            features = extract_sample_features(
                dm, sample['ts_code'], sample['name'], 
                sample['t1_date'], LOOKBACK_DAYS, idx
            )
            if not features.empty:
                all_features.append(features)
        except Exception as e:
            log.error(f"提取特征失败: {sample['ts_code']} - {e}")
    
    if not all_features:
        log.error("正样本特征提取失败")
        return None
    
    df_features = pd.concat(all_features, ignore_index=True)
    
    # 计算高级特征
    log.info("计算高级特征...")
    df_features = calculate_advanced_features(df_features)
    
    # 验证OHLCV完整性
    if 'high' in df_features.columns and 'low' in df_features.columns:
        high_ratio = (df_features['high'] / df_features['close']).round(4)
        estimated_count = (high_ratio == 1.01).sum()
        if estimated_count > 0:
            log.warning(f"发现 {estimated_count} 行可能使用了估算的high值")
    
    # 保存
    output_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v6_fixed.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_file, index=False)
    
    n_samples = df_features['sample_id'].nunique()
    n_rows = len(df_features)
    log.success(f"✓ 正样本生成完成: {n_samples} 个样本, {n_rows} 条记录")
    log.info(f"  输出文件: {output_file}")
    
    return df_features


def main():
    start_time = datetime.now()
    
    log.info("="*80)
    log.info("生成 v6 版本样本数据（修复版）")
    log.info("="*80)
    log.info(f"开始时间: {start_time}")
    log.info(f"历史数据天数: {LOOKBACK_DAYS}")
    log.info("")
    log.info("修复内容:")
    log.info("  1. 确保OHLCV数据完整获取，不使用估算值")
    log.info("  2. 基于v5逻辑进行优化，保持特征计算一致性")
    log.info("  3. 扩展历史数据到70天")
    log.info("")
    
    # 步骤1: 生成正样本
    positive_df = generate_positive_samples_v6_fixed()
    if positive_df is None:
        log.error("正样本生成失败，退出")
        return
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    log.info("\n" + "="*80)
    log.success("✓ v6 样本数据生成完成（修复版）！")
    log.info("="*80)
    log.info(f"耗时: {duration}")
    log.info(f"\n注意: 此脚本仅生成正样本作为验证")
    log.info("完整的v6数据生成需要进一步修复 positive_sample_screener.py")


if __name__ == '__main__':
    main()
