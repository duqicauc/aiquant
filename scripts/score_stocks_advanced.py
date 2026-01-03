#!/usr/bin/env python3
"""
股票评分脚本 - 高级版（支持市场因子和高级技术因子）

特点：
1. 与 train_xgboost_timeseries.py 特征提取方式完全一致
2. 支持市场因子（market_pct_chg, excess_return 等）
3. 支持高级技术因子（动量、量价配合、突破形态、支撑阻力等）
4. 支持历史回测（指定日期）

使用方法：
    # 对最新收盘数据评分
    python scripts/score_stocks_advanced.py
    
    # 对20251225收盘后评分
    python scripts/score_stocks_advanced.py --date 20251225
    
    # 限制评分数量（测试用）
    python scripts/score_stocks_advanced.py --max-stocks 100
"""
import sys
import os
import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import pandas as pd
import numpy as np
import xgboost as xgb

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_model_and_features():
    """
    加载最新训练的模型和特征名称
    
    Returns:
        booster: XGBoost Booster模型
        feature_names: 特征名称列表
        model_info: 模型信息字典
    """
    log.info("="*80)
    log.info("加载模型")
    log.info("="*80)
    
    # 方案1：从 v1.4.0 版本目录加载
    version_model_path = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v1.4.0' / 'model' / 'model.json'
    
    # 方案2：从训练模型目录加载最新模型
    training_model_dir = PROJECT_ROOT / 'data' / 'training' / 'models'
    metrics_file = PROJECT_ROOT / 'data' / 'training' / 'metrics' / 'xgboost_timeseries_v2_metrics.json'
    
    # 加载最新的训练模型
    model_files = list(training_model_dir.glob('xgboost_timeseries_v2_*.json'))
    if model_files:
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        log.info(f"加载最新训练模型: {model_path.name}")
    elif version_model_path.exists():
        model_path = version_model_path
        log.info(f"加载版本模型: v1.4.0")
    else:
        raise FileNotFoundError("未找到任何可用的模型文件")
    
    # 加载 Booster
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    
    # 【关键】从模型内部获取特征名称，确保顺序与训练时一致
    feature_names = booster.feature_names
    if feature_names is None:
        # 如果模型内部没有特征名称，尝试其他方式
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            if 'feature_importance' in metrics:
                feature_names = [item['feature'] for item in metrics['feature_importance']]
                log.warning(f"从 metrics 文件加载特征名称（可能顺序不一致）: {len(feature_names)} 个特征")
        else:
            raise ValueError("无法获取特征名称")
    else:
        log.info(f"从模型内部获取特征名称: {len(feature_names)} 个特征")
    
    model_info = {
        'model_path': str(model_path),
        'model_name': 'breakout_launch_scorer',
        'version': 'v1.4.0',
        'feature_count': len(feature_names)
    }
    
    log.success(f"✓ 模型加载成功，特征数: {len(feature_names)}")
    
    return booster, feature_names, model_info


def _vectorized_rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    """
    向量化计算滚动窗口线性回归斜率（比循环快50倍）
    
    使用公式: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    其中 x = [0, 1, 2, ..., window-1]
    """
    n = len(y)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    # 预计算 x 相关常量 (x = [0, 1, ..., window-1])
    x = np.arange(window)
    sum_x = x.sum()  # = window*(window-1)/2
    sum_x2 = (x ** 2).sum()  # = window*(window-1)*(2*window-1)/6
    denom = window * sum_x2 - sum_x ** 2
    
    if denom == 0:
        return result
    
    # 使用 cumsum 技巧计算滚动 sum(y) 和 sum(xy)
    y_cumsum = np.cumsum(y)
    xy_cumsum = np.cumsum(np.arange(n) * y)
    
    # 对每个窗口位置计算斜率
    for i in range(window - 1, n):
        start = i - window + 1
        if start == 0:
            sum_y = y_cumsum[i]
            # sum(xy) 需要调整 x 的偏移
            sum_xy = np.sum(x * y[start:i+1])
        else:
            sum_y = y_cumsum[i] - y_cumsum[start - 1]
            sum_xy = np.sum(x * y[start:i+1])
        
        result[i] = (window * sum_xy - sum_x * sum_y) / denom
    
    return result


def _vectorized_rolling_slope_fast(y: np.ndarray, window: int) -> np.ndarray:
    """
    更快的向量化滚动斜率计算（使用 pandas rolling）
    """
    import pandas as pd
    
    n = len(y)
    if n < window:
        return np.full(n, np.nan)
    
    # 创建 Series
    s = pd.Series(y)
    
    # x 常量
    x = np.arange(window)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denom = window * sum_x2 - sum_x ** 2
    
    if denom == 0:
        return np.full(n, np.nan)
    
    # 滚动计算 sum(y)
    sum_y = s.rolling(window).sum()
    
    # 滚动计算 sum(i*y_i) 然后调整
    # 技巧: 对于窗口 [y_{t-w+1}, ..., y_t]，sum(x*y) = sum((j - (t-w+1)) * y_j)
    idx = np.arange(n)
    weighted = pd.Series(idx * y)
    sum_idx_y = weighted.rolling(window).sum()
    
    # 调整为 sum(x*y) where x = [0, 1, ..., w-1]
    # sum_xy = sum((j - (t-w+1)) * y_j) = sum_idx_y - (t-w+1) * sum_y
    t_minus_w_plus_1 = pd.Series(idx - window + 1)
    sum_xy = sum_idx_y - t_minus_w_plus_1 * sum_y
    
    # 计算斜率
    slope = (window * sum_xy - sum_x * sum_y) / denom
    
    return slope.values


def calculate_advanced_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算高级技术因子（与 add_advanced_factors_v2.py 中的逻辑一致）
    """
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    n = len(df)
    if n < 10:
        return df
    
    # ==================== 1. 动量因子 ====================
    for period in [5, 10, 20]:
        if n >= period:
            df[f'momentum_{period}d'] = df['close'].pct_change(period) * 100
    
    if 'momentum_10d' in df.columns and n >= 15:
        df['momentum_acceleration'] = df['momentum_10d'].diff(5)
    
    # ==================== 2. 量价配合 ====================
    if 'vol' in df.columns:
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()
        
        if n >= 10:
            df['volume_price_corr_10d'] = df['price_change'].rolling(10).corr(df['volume_change'])
        if n >= 20:
            df['volume_price_corr_20d'] = df['price_change'].rolling(20).corr(df['volume_change'])
        
        df['volume_price_match'] = np.where(
            (df['price_change'] > 0) & (df['volume_change'] > 0), 1,
            np.where((df['price_change'] < 0) & (df['volume_change'] < 0), 1, -1)
        )
        if n >= 10:
            df['volume_price_match_sum_10d'] = df['volume_price_match'].rolling(10).sum()
    
    # ==================== 3. 多时间框架特征 ====================
    for tf in [8, 34, 55]:
        if n >= tf:
            df[f'return_{tf}d'] = (df['close'] - df['close'].shift(tf)) / df['close'].shift(tf) * 100
            df[f'ma_{tf}d'] = df['close'].rolling(tf).mean()
            df[f'price_vs_ma_{tf}d'] = (df['close'] - df[f'ma_{tf}d']) / df[f'ma_{tf}d'] * 100
            df[f'volatility_{tf}d'] = df['pct_chg'].rolling(tf).std()
            df[f'high_{tf}d'] = df['close'].rolling(tf).max()
            df[f'low_{tf}d'] = df['close'].rolling(tf).min()
            df[f'price_position_{tf}d'] = (df['close'] - df[f'low_{tf}d']) / (df[f'high_{tf}d'] - df[f'low_{tf}d'] + 1e-8) * 100
            
            # 趋势斜率（纯向量化计算，更快）
            df[f'trend_slope_{tf}d'] = _vectorized_rolling_slope_fast(df['close'].values, tf)
    
    # ==================== 4. 突破形态 ====================
    for period in [10, 20, 55]:
        if n >= period:
            df[f'prev_high_{period}d'] = df['close'].shift(1).rolling(period).max()
            df[f'breakout_high_{period}d'] = (df['close'] > df[f'prev_high_{period}d']).astype(int)
    
    for ma_period in [5, 10, 20, 55]:
        ma_col = f'ma_{ma_period}d'
        if ma_col not in df.columns and n >= ma_period:
            df[ma_col] = df['close'].rolling(ma_period).mean()
        
        if ma_col in df.columns:
            df[f'breakout_ma{ma_period}'] = (
                (df['close'] > df[ma_col]) & 
                (df['close'].shift(1) <= df[ma_col].shift(1))
            ).astype(int)
    
    if 'vol' in df.columns and n >= 20:
        vol_ma20 = df['vol'].rolling(20).mean()
        df['breakout_volume_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
        df['high_volume_breakout'] = (df['breakout_volume_ratio'] > 1.5).astype(int)
    
    if 'breakout_high_20d' in df.columns:
        consecutive = []
        count = 0
        for val in df['breakout_high_20d']:
            if val == 1:
                count += 1
            else:
                count = 0
            consecutive.append(count)
        df['consecutive_new_high'] = consecutive
    
    # ==================== 5. 支撑/阻力位 ====================
    for period in [10, 20, 55]:
        if n >= period:
            df[f'resistance_{period}d'] = df['close'].shift(1).rolling(period).max()
            df[f'support_{period}d'] = df['close'].shift(1).rolling(period).min()
            df[f'dist_to_resistance_{period}d'] = (df[f'resistance_{period}d'] - df['close']) / df['close'] * 100
            df[f'dist_to_support_{period}d'] = (df['close'] - df[f'support_{period}d']) / df['close'] * 100
            
            near_support = (abs(df['close'] - df[f'support_{period}d']) / df['close'] < 0.02).astype(int)
            df[f'support_strength_{period}d'] = near_support.rolling(period).sum()
            
            near_resistance = (abs(df['close'] - df[f'resistance_{period}d']) / df['close'] < 0.02).astype(int)
            df[f'resistance_strength_{period}d'] = near_resistance.rolling(period).sum()
    
    if 'resistance_20d' in df.columns and 'support_20d' in df.columns:
        df['channel_width_20d'] = (df['resistance_20d'] - df['support_20d']) / df['close'] * 100
    
    # ==================== 6. 成交量特征增强 ====================
    if 'vol' in df.columns:
        for period in [10, 20]:
            if n >= period:
                # 纯向量化计算成交量趋势斜率
                vol_slope = _vectorized_rolling_slope_fast(df['vol'].values, period)
                vol_ma = df['vol'].rolling(period).mean().values
                df[f'volume_trend_slope_{period}d'] = vol_slope / (vol_ma + 1e-8) * 100
        
        if n >= 20:
            vol_ma20 = df['vol'].rolling(20).mean()
            vol_breakout = (df['vol'] > vol_ma20 * 2).astype(int)
            df['volume_breakout_count_20d'] = vol_breakout.rolling(20).sum()
        
        df['price_up_vol_down'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['vol'] < df['vol'].shift(1))
        ).astype(int)
        if n >= 10:
            df['price_up_vol_down_count_10d'] = df['price_up_vol_down'].rolling(10).sum()
        
        df['price_down_vol_up'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['vol'] > df['vol'].shift(1))
        ).astype(int)
        if n >= 10:
            df['price_down_vol_up_count_10d'] = df['price_down_vol_up'].rolling(10).sum()
        
        if n >= 20:
            vol_high_20 = df['vol'].rolling(20).max()
            vol_low_20 = df['vol'].rolling(20).min()
            df['volume_rsv_20d'] = (df['vol'] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-8) * 100
        
        df['obv_calc'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        if n >= 10:
            df['obv_ma10'] = df['obv_calc'].rolling(10).mean()
            df['obv_trend'] = (df['obv_calc'] > df['obv_ma10']).astype(int)
    
    return df


def get_cached_market_data(dm: DataManager, target_date: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    获取并缓存市场数据（只调用一次API）
    
    Args:
        dm: DataManager实例
        target_date: 目标日期
        lookback_days: 回看天数
        
    Returns:
        df_market: 包含市场因子的DataFrame
    """
    try:
        start_date = (datetime.strptime(target_date, '%Y%m%d') - timedelta(days=lookback_days)).strftime('%Y%m%d')
        
        # 获取沪深300指数
        df_index = dm.get_index_daily('000300.SH', start_date, target_date)
        
        if df_index is None or df_index.empty:
            log.warning("获取沪深300指数数据失败")
            return None
        
        df_index['trade_date'] = pd.to_datetime(df_index['trade_date'])
        df_index = df_index.rename(columns={'pct_chg': 'market_pct_chg', 'close': 'market_close'})
        
        # 计算市场34日收益率和波动率
        df_index = df_index.sort_values('trade_date')
        df_index['market_return_34d'] = df_index['market_close'].pct_change(34) * 100
        df_index['market_volatility_34d'] = df_index['market_pct_chg'].rolling(34).std()
        df_index['market_trend'] = (df_index['market_return_34d'] > 0).astype(int)
        
        log.success(f"✓ 市场数据已缓存: {len(df_index)} 条记录")
        return df_index[['trade_date', 'market_pct_chg', 'market_return_34d', 'market_volatility_34d', 'market_trend']]
        
    except Exception as e:
        log.warning(f"获取市场数据失败: {e}")
        return None


def calculate_market_factors(df: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场因子（使用缓存的市场数据，避免重复API调用）
    
    Args:
        df: 股票日线数据
        df_market: 缓存的市场数据（由get_cached_market_data返回）
    """
    df = df.copy()
    
    if df_market is None:
        return df
    
    try:
        # 合并市场数据
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = pd.merge(
            df,
            df_market,
            on='trade_date',
            how='left'
        )
        
        # 计算超额收益
        if 'pct_chg' in df.columns and 'market_pct_chg' in df.columns:
            df['excess_return'] = df['pct_chg'] - df['market_pct_chg']
            df['excess_return_cumsum'] = df['excess_return'].cumsum()
        
        # 历史价格统计
        df['price_vs_hist_mean'] = (df['close'] - df['close'].rolling(34).mean()) / df['close'].rolling(34).mean() * 100
    except Exception as e:
        pass  # 静默处理，不影响评分
    
    return df


def extract_features_from_sample(sample_data: pd.DataFrame, feature_names: list) -> dict:
    """
    从34天时序数据提取特征（与 train_xgboost_timeseries.py 中的 extract_features_with_time 一致）
    
    Args:
        sample_data: 34天的日线数据
        feature_names: 特征名称列表
        
    Returns:
        feature_dict: 特征字典
    """
    if len(sample_data) < 20:
        return None
    
    feature_dict = {}
    
    # 价格特征
    feature_dict['close_mean'] = sample_data['close'].mean()
    feature_dict['close_std'] = sample_data['close'].std()
    feature_dict['close_max'] = sample_data['close'].max()
    feature_dict['close_min'] = sample_data['close'].min()
    feature_dict['close_trend'] = (
        (sample_data['close'].iloc[-1] - sample_data['close'].iloc[0]) / 
        sample_data['close'].iloc[0] * 100
    )
    
    # 涨跌幅特征
    feature_dict['pct_chg_mean'] = sample_data['pct_chg'].mean()
    feature_dict['pct_chg_std'] = sample_data['pct_chg'].std()
    feature_dict['pct_chg_sum'] = sample_data['pct_chg'].sum()
    feature_dict['positive_days'] = (sample_data['pct_chg'] > 0).sum()
    feature_dict['negative_days'] = (sample_data['pct_chg'] < 0).sum()
    feature_dict['max_gain'] = sample_data['pct_chg'].max()
    feature_dict['max_loss'] = sample_data['pct_chg'].min()
    
    # 量比特征
    if 'volume_ratio' in sample_data.columns:
        feature_dict['volume_ratio_mean'] = sample_data['volume_ratio'].mean()
        feature_dict['volume_ratio_max'] = sample_data['volume_ratio'].max()
        feature_dict['volume_ratio_gt_2'] = (sample_data['volume_ratio'] > 2).sum()
        feature_dict['volume_ratio_gt_4'] = (sample_data['volume_ratio'] > 4).sum()
    
    # MACD特征
    if 'macd' in sample_data.columns:
        macd_data = sample_data['macd'].dropna()
        if len(macd_data) > 0:
            feature_dict['macd_mean'] = macd_data.mean()
            feature_dict['macd_positive_days'] = (macd_data > 0).sum()
            feature_dict['macd_max'] = macd_data.max()
    
    # MA特征
    if 'ma5' in sample_data.columns:
        feature_dict['ma5_mean'] = sample_data['ma5'].mean()
        feature_dict['price_above_ma5'] = (sample_data['close'] > sample_data['ma5']).sum()
    
    if 'ma10' in sample_data.columns:
        feature_dict['ma10_mean'] = sample_data['ma10'].mean()
        feature_dict['price_above_ma10'] = (sample_data['close'] > sample_data['ma10']).sum()
    
    # 市值特征
    if 'total_mv' in sample_data.columns:
        mv_data = sample_data['total_mv'].dropna()
        if len(mv_data) > 0:
            feature_dict['total_mv_mean'] = mv_data.mean()
    
    if 'circ_mv' in sample_data.columns:
        circ_mv_data = sample_data['circ_mv'].dropna()
        if len(circ_mv_data) > 0:
            feature_dict['circ_mv_mean'] = circ_mv_data.mean()
    
    # RSI特征
    for rsi_period in [6, 12, 24]:
        col = f'rsi_{rsi_period}'
        if col in sample_data.columns:
            rsi_data = sample_data[col].dropna()
            if len(rsi_data) > 0:
                feature_dict[f'rsi_{rsi_period}_mean'] = rsi_data.mean()
                feature_dict[f'rsi_{rsi_period}_std'] = rsi_data.std()
                feature_dict[f'rsi_{rsi_period}_last'] = rsi_data.iloc[-1]
                feature_dict[f'rsi_{rsi_period}_max'] = rsi_data.max()
                feature_dict[f'rsi_{rsi_period}_min'] = rsi_data.min()
                feature_dict[f'rsi_{rsi_period}_gt_70'] = (rsi_data > 70).sum()
                feature_dict[f'rsi_{rsi_period}_lt_30'] = (rsi_data < 30).sum()
    
    # 动量特征
    days = len(sample_data)
    if days >= 7:
        feature_dict['return_1w'] = (
            (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-7]) /
            sample_data['close'].iloc[-7] * 100
        )
    if days >= 14:
        feature_dict['return_2w'] = (
            (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-14]) /
            sample_data['close'].iloc[-14] * 100
        )
    
    # ===== 市场因子特征 =====
    if 'market_pct_chg' in sample_data.columns:
        market_data = sample_data['market_pct_chg'].dropna()
        if len(market_data) > 0:
            feature_dict['market_pct_chg_mean'] = market_data.mean()
    
    if 'market_return_34d' in sample_data.columns:
        market_return_data = sample_data['market_return_34d'].dropna()
        if len(market_return_data) > 0:
            feature_dict['market_return_34d_last'] = market_return_data.iloc[-1]
    
    if 'market_volatility_34d' in sample_data.columns:
        market_vol_data = sample_data['market_volatility_34d'].dropna()
        if len(market_vol_data) > 0:
            feature_dict['market_volatility_34d_last'] = market_vol_data.iloc[-1]
    
    if 'market_trend' in sample_data.columns:
        market_trend_data = sample_data['market_trend'].dropna()
        if len(market_trend_data) > 0:
            feature_dict['market_trend_last'] = market_trend_data.iloc[-1]
    
    if 'excess_return' in sample_data.columns:
        excess_data = sample_data['excess_return'].dropna()
        if len(excess_data) > 0:
            feature_dict['excess_return_mean'] = excess_data.mean()
            feature_dict['excess_return_sum'] = excess_data.sum()
            feature_dict['excess_return_positive_days'] = (excess_data > 0).sum()
    
    if 'excess_return_cumsum' in sample_data.columns:
        excess_cumsum_data = sample_data['excess_return_cumsum'].dropna()
        if len(excess_cumsum_data) > 0:
            feature_dict['excess_return_cumsum_last'] = excess_cumsum_data.iloc[-1]
    
    if 'price_vs_hist_mean' in sample_data.columns:
        hist_mean_data = sample_data['price_vs_hist_mean'].dropna()
        if len(hist_mean_data) > 0:
            feature_dict['price_vs_hist_mean_last'] = hist_mean_data.iloc[-1]
    
    # ===== 新技术因子特征（full）=====
    # 换手率
    if 'turnover_rate_f' in sample_data.columns:
        turnover_data = sample_data['turnover_rate_f'].dropna()
        if len(turnover_data) > 0:
            feature_dict['turnover_rate_f_mean'] = turnover_data.mean()
            feature_dict['turnover_rate_f_max'] = turnover_data.max()
            feature_dict['turnover_rate_f_std'] = turnover_data.std()
    
    # 乖离率BIAS
    for bias_type in ['short', 'mid', 'long']:
        col = f'bias_{bias_type}'
        if col in sample_data.columns:
            bias_data = sample_data[col].dropna()
            if len(bias_data) > 0:
                feature_dict[f'{col}_last'] = bias_data.iloc[-1]
                if bias_type == 'short':
                    feature_dict[f'{col}_mean'] = bias_data.mean()
    
    # EMA
    if 'ema_5' in sample_data.columns and 'ema_20' in sample_data.columns:
        ema5 = sample_data['ema_5'].dropna()
        ema20 = sample_data['ema_20'].dropna()
        if len(ema5) > 0 and len(ema20) > 0:
            feature_dict['ema_ratio_5_20'] = ema5.iloc[-1] / ema20.iloc[-1] if ema20.iloc[-1] != 0 else 1
            if len(sample_data['close'].dropna()) > 0:
                close_last = sample_data['close'].dropna().iloc[-1]
                feature_dict['price_vs_ema5'] = (close_last - ema5.iloc[-1]) / ema5.iloc[-1] * 100 if ema5.iloc[-1] != 0 else 0
                feature_dict['price_vs_ema20'] = (close_last - ema20.iloc[-1]) / ema20.iloc[-1] * 100 if ema20.iloc[-1] != 0 else 0
    if 'ema_60' in sample_data.columns:
        ema60 = sample_data['ema_60'].dropna()
        if len(ema60) > 0 and len(sample_data['close'].dropna()) > 0:
            close_last = sample_data['close'].dropna().iloc[-1]
            feature_dict['price_vs_ema60'] = (close_last - ema60.iloc[-1]) / ema60.iloc[-1] * 100 if ema60.iloc[-1] != 0 else 0
    
    # KDJ
    for kdj_type in ['k', 'd', 'j']:
        col = f'kdj_{kdj_type}'
        if col in sample_data.columns:
            kdj_data = sample_data[col].dropna()
            if len(kdj_data) > 0:
                feature_dict[f'{col}_last'] = kdj_data.iloc[-1]
                if kdj_type == 'k':
                    feature_dict[f'{col}_mean'] = kdj_data.mean()
                if kdj_type == 'j':
                    feature_dict['kdj_j_overbought'] = (kdj_data > 80).sum()
                    feature_dict['kdj_j_oversold'] = (kdj_data < 20).sum()
    
    # 涨停统计
    if 'is_limit_up' in sample_data.columns:
        is_limit = sample_data['is_limit_up'].dropna()
        if len(is_limit) > 0:
            feature_dict['limit_up_count'] = is_limit.sum()
    
    # OBV
    if 'obv' in sample_data.columns:
        obv = sample_data['obv'].dropna()
        if len(obv) > 0:
            feature_dict['obv_change'] = (obv.iloc[-1] - obv.iloc[0]) / abs(obv.iloc[0]) * 100 if obv.iloc[0] != 0 else 0
            feature_dict['obv_trend'] = 1 if obv.iloc[-1] > obv.mean() else 0
    
    # 成交量与均量比
    for vol_period in [5, 20]:
        col = f'vol_ma{vol_period}_ratio'
        if col in sample_data.columns:
            vol_r = sample_data[col].dropna()
            if len(vol_r) > 0:
                feature_dict[f'{col}_mean'] = vol_r.mean()
                feature_dict[f'{col}_max'] = vol_r.max()
    
    # ===== 高级技术因子（advanced）=====
    # 动量因子
    for period in [5, 10, 20]:
        col = f'momentum_{period}d'
        if col in sample_data.columns:
            data = sample_data[col].dropna()
            if len(data) > 0:
                feature_dict[f'{col}_last'] = data.iloc[-1]
                feature_dict[f'{col}_mean'] = data.mean()
    
    if 'momentum_acceleration' in sample_data.columns:
        data = sample_data['momentum_acceleration'].dropna()
        if len(data) > 0:
            feature_dict['momentum_acceleration_last'] = data.iloc[-1]
    
    # 量价配合度
    if 'volume_price_corr_10d' in sample_data.columns:
        data = sample_data['volume_price_corr_10d'].dropna()
        if len(data) > 0:
            feature_dict['volume_price_corr_last'] = data.iloc[-1]
    if 'volume_price_match_sum_10d' in sample_data.columns:
        data = sample_data['volume_price_match_sum_10d'].dropna()
        if len(data) > 0:
            feature_dict['volume_price_match_sum'] = data.iloc[-1]
    
    # 多时间框架特征
    for tf in [8, 55]:
        for metric in ['return', 'price_vs_ma', 'volatility', 'price_position', 'trend_slope']:
            col = f'{metric}_{tf}d'
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_last'] = data.iloc[-1]
    
    # 突破形态
    for period in [10, 20, 55]:
        col = f'breakout_high_{period}d'
        if col in sample_data.columns:
            data = sample_data[col].dropna()
            if len(data) > 0:
                feature_dict[f'{col}_sum'] = data.sum()
    
    for ma in [5, 10, 20, 55]:
        col = f'breakout_ma{ma}'
        if col in sample_data.columns:
            data = sample_data[col].dropna()
            if len(data) > 0:
                feature_dict[f'{col}_sum'] = data.sum()
    
    if 'high_volume_breakout' in sample_data.columns:
        data = sample_data['high_volume_breakout'].dropna()
        if len(data) > 0:
            feature_dict['high_volume_breakout_sum'] = data.sum()
    
    if 'consecutive_new_high' in sample_data.columns:
        data = sample_data['consecutive_new_high'].dropna()
        if len(data) > 0:
            feature_dict['consecutive_new_high_max'] = data.max()
    
    # 支撑阻力
    for period in [10, 20]:
        for metric in ['dist_to_support', 'dist_to_resistance']:
            col = f'{metric}_{period}d'
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_last'] = data.iloc[-1]
        
        for metric in ['support_strength', 'resistance_strength']:
            col = f'{metric}_{period}d'
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f'{col}_last'] = data.iloc[-1]
    
    if 'channel_width_20d' in sample_data.columns:
        data = sample_data['channel_width_20d'].dropna()
        if len(data) > 0:
            feature_dict['channel_width_last'] = data.iloc[-1]
    
    # 高级成交量
    for col in ['volume_trend_slope_10d', 'volume_trend_slope_20d']:
        if col in sample_data.columns:
            data = sample_data[col].dropna()
            if len(data) > 0:
                feature_dict[f'{col}_last'] = data.iloc[-1]
    
    if 'volume_breakout_count_20d' in sample_data.columns:
        data = sample_data['volume_breakout_count_20d'].dropna()
        if len(data) > 0:
            feature_dict['volume_breakout_count'] = data.iloc[-1]
    
    if 'price_up_vol_down_count_10d' in sample_data.columns:
        data = sample_data['price_up_vol_down_count_10d'].dropna()
        if len(data) > 0:
            feature_dict['price_up_vol_down_count'] = data.iloc[-1]
    
    if 'price_down_vol_up_count_10d' in sample_data.columns:
        data = sample_data['price_down_vol_up_count_10d'].dropna()
        if len(data) > 0:
            feature_dict['price_down_vol_up_count'] = data.iloc[-1]
    
    if 'volume_rsv_20d' in sample_data.columns:
        data = sample_data['volume_rsv_20d'].dropna()
        if len(data) > 0:
            feature_dict['volume_rsv_last'] = data.iloc[-1]
    
    if 'obv_trend' in sample_data.columns:
        data = sample_data['obv_trend'].dropna()
        if len(data) > 0:
            feature_dict['obv_trend_sum'] = data.sum()
    
    return feature_dict


def get_valid_stocks(dm: DataManager, target_date: datetime) -> pd.DataFrame:
    """获取有效股票列表"""
    log.info("="*80)
    log.info("获取股票列表")
    log.info("="*80)
    
    stock_list = dm.get_stock_list()
    log.info(f"✓ 获取到 {len(stock_list)} 只股票")
    
    excluded = {'st': 0, 'new': 0, 'delisted': 0, 'bj': 0}
    valid_stocks = []
    
    for _, stock in stock_list.iterrows():
        ts_code = stock['ts_code']
        name = stock['name']
        
        # 排除ST
        if 'ST' in name or 'st' in name.lower() or '*' in name:
            excluded['st'] += 1
            continue
        
        # 排除退市
        if '退' in name:
            excluded['delisted'] += 1
            continue
        
        # 排除北交所
        if ts_code.endswith('.BJ'):
            excluded['bj'] += 1
            continue
        
        # 检查上市天数
        list_date = stock.get('list_date', '')
        if list_date:
            try:
                days = (target_date - pd.to_datetime(list_date)).days
                if days < 120:
                    excluded['new'] += 1
                    continue
            except:
                pass
        
        valid_stocks.append(stock)
    
    log.info(f"\n剔除统计: ST={excluded['st']}, 次新={excluded['new']}, "
            f"退市={excluded['delisted']}, 北交所={excluded['bj']}")
    log.info(f"✓ 符合条件: {len(valid_stocks)} 只")
    
    return pd.DataFrame(valid_stocks)


def score_single_stock(dm: DataManager, ts_code: str, name: str, 
                       target_date: datetime, feature_names: list,
                       df_market: pd.DataFrame = None,
                       lookback_days: int = 34, max_lookback: int = 90) -> dict:
    """
    对单只股票进行特征提取和评分
    
    Args:
        df_market: 缓存的市场数据（避免重复API调用）
    """
    try:
        # 获取日线数据（获取更长时间以计算高级因子）
        end_date = target_date.strftime('%Y%m%d')
        start_date = (target_date - timedelta(days=max_lookback)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date)
        
        if df is None or len(df) < 20:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 确保数值列
        for col in ['close', 'pct_chg', 'vol', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算基础技术指标
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(10).mean()
        
        # 量比
        if 'volume_ratio' not in df.columns:
            vol_ma5 = df['vol'].rolling(5).mean()
            df['volume_ratio'] = df['vol'] / vol_ma5
        
        # MACD
        if 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd_dif'] = ema12 - ema26
            df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
            df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
        
        # RSI
        for period in [6, 12, 24]:
            col = f'rsi_{period}'
            if col not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / (loss + 1e-8)
                df[col] = 100 - (100 / (1 + rs))
        
        # KDJ
        if 'kdj_k' not in df.columns:
            n, m1, m2 = 9, 3, 3
            low_n = df['low'].rolling(n).min()
            high_n = df['high'].rolling(n).max()
            rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
            df['kdj_k'] = rsv.ewm(com=m1-1, adjust=False).mean()
            df['kdj_d'] = df['kdj_k'].ewm(com=m2-1, adjust=False).mean()
            df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # EMA
        for ema_period in [5, 10, 20, 60]:
            col = f'ema_{ema_period}'
            if col not in df.columns:
                df[col] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        # BIAS (乖离率)
        if 'bias_short' not in df.columns:
            df['bias_short'] = (df['close'] - df['close'].rolling(6).mean()) / df['close'].rolling(6).mean() * 100
        if 'bias_mid' not in df.columns:
            df['bias_mid'] = (df['close'] - df['close'].rolling(12).mean()) / df['close'].rolling(12).mean() * 100
        if 'bias_long' not in df.columns:
            df['bias_long'] = (df['close'] - df['close'].rolling(24).mean()) / df['close'].rolling(24).mean() * 100
        
        # OBV
        if 'obv' not in df.columns:
            df['obv'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        
        # 成交量/均量比
        if 'vol_ma5_ratio' not in df.columns:
            vol_ma5 = df['vol'].rolling(5).mean()
            df['vol_ma5_ratio'] = df['vol'] / (vol_ma5 + 1e-8)
        if 'vol_ma20_ratio' not in df.columns:
            vol_ma20 = df['vol'].rolling(20).mean()
            df['vol_ma20_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
        
        # 涨停判断
        if 'is_limit_up' not in df.columns:
            df['is_limit_up'] = (df['pct_chg'] >= 9.5).astype(int)
        
        # 计算市场因子（使用缓存的市场数据）
        df = calculate_market_factors(df, df_market)
        
        # 计算高级技术因子
        df = calculate_advanced_factors(df)
        
        # 取最近34天数据
        df_sample = df.tail(lookback_days).copy()
        
        if len(df_sample) < 20:
            return None
        
        # 提取特征
        features = extract_features_from_sample(df_sample, feature_names)
        
        if features is None:
            return None
        
        # 添加元数据
        features['ts_code'] = ts_code
        features['name'] = name
        features['latest_date'] = df_sample['trade_date'].iloc[-1]
        features['latest_close'] = df_sample['close'].iloc[-1]
        
        return features
        
    except Exception as e:
        return None


def score_all_stocks(dm: DataManager, booster: xgb.Booster, feature_names: list,
                    valid_stocks: pd.DataFrame, target_date: datetime,
                    max_stocks: int = None) -> pd.DataFrame:
    """对所有股票进行评分"""
    log.info("="*80)
    log.info("开始评分")
    log.info("="*80)
    
    if max_stocks:
        valid_stocks = valid_stocks.head(max_stocks)
        log.info(f"⚠️ 测试模式：仅评分前 {max_stocks} 只")
    
    total = len(valid_stocks)
    features_list = []
    stock_info_list = []
    stats = {'success': 0, 'no_data': 0, 'error': 0}
    
    # 【优化】预先获取并缓存市场数据（只调用一次API）
    target_date_str = target_date.strftime('%Y%m%d')
    df_market = get_cached_market_data(dm, target_date_str, lookback_days=120)
    
    # 提取特征
    for i, (_, stock) in enumerate(valid_stocks.iterrows()):
        if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total:
            log.info(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        
        ts_code = stock['ts_code']
        name = stock['name']
        
        features = score_single_stock(dm, ts_code, name, target_date, feature_names, df_market=df_market)
        
        if features is None:
            stats['no_data'] += 1
            continue
        
        features_list.append(features)
        stock_info_list.append({
            'ts_code': ts_code,
            'name': name,
            'features': features
        })
        stats['success'] += 1
    
    log.info(f"\n特征提取: 成功={stats['success']}, 无数据={stats['no_data']}")
    
    if not features_list:
        log.error("没有成功提取特征的股票")
        return pd.DataFrame()
    
    # 批量预测
    log.info("批量预测...")
    feature_vectors = []
    for features in features_list:
        vector = []
        for name in feature_names:
            value = features.get(name, 0)
            if pd.isna(value):
                value = 0
            vector.append(value)
        feature_vectors.append(vector)
    
    dmatrix = xgb.DMatrix(feature_vectors, feature_names=feature_names)
    probabilities = booster.predict(dmatrix)
    
    # 构建结果
    results = []
    for i, info in enumerate(stock_info_list):
        features = info['features']
        results.append({
            '股票代码': info['ts_code'],
            '股票名称': info['name'],
            '牛股概率': float(probabilities[i]),
            '数据日期': features.get('latest_date', ''),
            '最新价格': features.get('latest_close', 0),
            '34日涨幅%': round(features.get('close_trend', 0), 2),
            '累计涨跌%': round(features.get('pct_chg_sum', 0), 2),
            '1周涨幅%': round(features.get('return_1w', 0), 2),
            '2周涨幅%': round(features.get('return_2w', 0), 2),
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('牛股概率', ascending=False).reset_index(drop=True)
    
    log.success(f"✓ 评分完成: {len(df_results)} 只股票")
    
    return df_results


def apply_risk_filter(df_scores: pd.DataFrame, 
                     max_34d_return: float = 50.0,
                     filter_mode: str = '降权') -> pd.DataFrame:
    """
    对评分结果应用风险过滤
    
    Args:
        df_scores: 评分结果DataFrame
        max_34d_return: 34日涨幅阈值（超过此值的股票会被处理）
        filter_mode: 处理模式
            - '降权': 降低牛股概率（推荐，保留模型识别能力）
            - '标记': 仅添加风险标记，不改变概率
            - '过滤': 直接移除高风险股票
    
    Returns:
        处理后的DataFrame
    """
    df_filtered = df_scores.copy()
    
    # 识别高风险股票
    high_risk_mask = df_filtered['34日涨幅%'] > max_34d_return
    high_risk_count = high_risk_mask.sum()
    
    if high_risk_count == 0:
        log.info(f"✓ 无高风险股票（34日涨幅>{max_34d_return}%）")
        return df_filtered
    
    log.warning(f"⚠️  发现 {high_risk_count} 只高风险股票（34日涨幅>{max_34d_return}%）")
    
    if filter_mode == '降权':
        # 降权策略：根据涨幅超阈值程度降低概率
        # 例如：涨幅60% → 降权20%，涨幅80% → 降权40%
        def calculate_penalty(row):
            excess = row['34日涨幅%'] - max_34d_return
            # 每超过10%降权5%，最大降权50%
            penalty_rate = min(0.5, excess / 10 * 0.05)
            return row['牛股概率'] * (1 - penalty_rate)
        
        df_filtered.loc[high_risk_mask, '原始概率'] = df_filtered.loc[high_risk_mask, '牛股概率']
        df_filtered.loc[high_risk_mask, '牛股概率'] = df_filtered.loc[high_risk_mask].apply(calculate_penalty, axis=1)
        df_filtered.loc[high_risk_mask, '风险标记'] = '高风险-已降权'
        
        # 重新排序
        df_filtered = df_filtered.sort_values('牛股概率', ascending=False).reset_index(drop=True)
        
        log.info(f"✓ 已对 {high_risk_count} 只股票进行降权处理")
        
        # 显示降权详情
        high_risk_stocks = df_filtered[df_filtered['风险标记'] == '高风险-已降权']
        if len(high_risk_stocks) > 0:
            log.info("\n降权股票详情:")
            log.info(f"{'代码':<12} {'名称':<10} {'原始概率':<10} {'降权后':<10} {'34日%':<8}")
            log.info("-" * 60)
            for _, row in high_risk_stocks.head(10).iterrows():
                original = row.get('原始概率', row['牛股概率'])
                log.info(f"{row['股票代码']:<12} {row['股票名称']:<10} "
                        f"{original:<10.4f} {row['牛股概率']:<10.4f} {row['34日涨幅%']:<8.2f}")
    
    elif filter_mode == '标记':
        # 仅添加标记，不改变概率
        df_filtered.loc[high_risk_mask, '风险标记'] = '高风险-追高'
        log.info(f"✓ 已标记 {high_risk_count} 只高风险股票")
    
    elif filter_mode == '过滤':
        # 直接移除
        df_filtered = df_filtered[~high_risk_mask].reset_index(drop=True)
        log.info(f"✓ 已过滤 {high_risk_count} 只高风险股票，剩余 {len(df_filtered)} 只")
    
    return df_filtered


def save_results(df_scores: pd.DataFrame, df_top: pd.DataFrame, 
                target_date: datetime, model_info: dict, top_n: int = 50):
    """保存结果"""
    date_str = target_date.strftime('%Y%m%d')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 输出目录
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 完整评分
    scores_file = output_dir / f"stock_scores_advanced_{date_str}.csv"
    df_scores.to_csv(scores_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ 完整评分: {scores_file}")
    
    # Top N
    top_file = output_dir / f"top_{top_n}_advanced_{date_str}.csv"
    df_top.to_csv(top_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ Top {top_n}: {top_file}")
    
    # 元数据
    metadata = {
        'prediction_date': date_str,
        'model': model_info,
        'total_scored': len(df_scores),
        'top_n': top_n,
        'created_at': datetime.now().isoformat(),
        'top_stocks': [
            {'rank': i+1, 'code': row['股票代码'], 'name': row['股票名称'],
             'probability': float(row['牛股概率'])}
            for i, row in df_top.iterrows()
        ]
    }
    
    metadata_dir = PROJECT_ROOT / 'data' / 'prediction' / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"prediction_metadata_advanced_{date_str}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return scores_file, top_file


def main():
    parser = argparse.ArgumentParser(description='股票评分（高级版，支持全部特征）')
    parser.add_argument('--date', '-d', default=None, help='目标日期（YYYYMMDD格式，如20251225）')
    parser.add_argument('--max-stocks', type=int, default=None, help='最大评分数量（测试用）')
    parser.add_argument('--top-n', type=int, default=50, help='Top N推荐数量')
    parser.add_argument('--risk-threshold', type=float, default=50.0, 
                       help='34日涨幅风险阈值（超过此值会被处理，默认50.0）')
    parser.add_argument('--risk-mode', choices=['降权', '标记', '过滤'], default='降权',
                       help='风险处理模式：降权（推荐）、标记、过滤')
    parser.add_argument('--disable-risk-filter', action='store_true',
                       help='禁用风险过滤（保留所有股票）')
    
    args = parser.parse_args()
    
    # 解析日期
    if args.date:
        target_date = datetime.strptime(args.date, '%Y%m%d')
        log.info(f"📅 目标日期: {target_date.strftime('%Y年%m月%d日')} 收盘后")
    else:
        target_date = datetime.now()
        log.info(f"📅 使用当前日期: {target_date.strftime('%Y年%m月%d日')}")
    
    log.info("="*80)
    log.info("股票评分系统（高级版 - 支持市场因子+高级技术因子）")
    log.info("="*80)
    
    try:
        # 1. 加载模型
        booster, feature_names, model_info = load_model_and_features()
        
        # 2. 初始化数据管理器
        log.info("\n初始化数据管理器...")
        dm = DataManager()
        log.success("✓ 数据管理器初始化完成")
        
        # 3. 获取股票列表
        valid_stocks = get_valid_stocks(dm, target_date)
        
        # 4. 评分
        df_scores = score_all_stocks(
            dm, booster, feature_names, valid_stocks, 
            target_date, args.max_stocks
        )
        
        if df_scores.empty:
            log.error("评分失败，没有结果")
            return
        
        # 5. 应用风险过滤（可选）
        if not args.disable_risk_filter:
            log.info("\n" + "="*80)
            log.info("风险过滤")
            log.info("="*80)
            log.info(f"风险阈值: 34日涨幅 > {args.risk_threshold}%")
            log.info(f"处理模式: {args.risk_mode}")
            df_scores = apply_risk_filter(
                df_scores, 
                max_34d_return=args.risk_threshold,
                filter_mode=args.risk_mode
            )
        else:
            log.info("\n⚠️  风险过滤已禁用")
        
        # 6. Top N
        df_top = df_scores.head(args.top_n)
        
        # 7. 显示结果
        log.info("\n" + "="*80)
        log.info(f"Top {args.top_n} 推荐")
        log.info("="*80)
        
        # 检查是否有风险标记列
        has_risk_marker = '风险标记' in df_top.columns
        
        if has_risk_marker:
            print(f"\n{'序号':<4} {'代码':<12} {'名称':<10} {'概率':<8} {'最新价':<8} {'34日%':<8} {'风险':<10}")
            print("-" * 70)
            for i, row in df_top.iterrows():
                risk_marker = row.get('风险标记', '')
                print(f"{i+1:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
                      f"{row['牛股概率']:.4f} {row['最新价格']:<8.2f} {row['34日涨幅%']:<8.2f} {risk_marker:<10}")
        else:
            print(f"\n{'序号':<4} {'代码':<12} {'名称':<10} {'概率':<8} {'最新价':<8} {'34日%':<8}")
            print("-" * 60)
            for i, row in df_top.iterrows():
                print(f"{i+1:<4} {row['股票代码']:<12} {row['股票名称']:<10} "
                      f"{row['牛股概率']:.4f} {row['最新价格']:<8.2f} {row['34日涨幅%']:<8.2f}")
        
        # 8. 保存结果
        save_results(df_scores, df_top, target_date, model_info, args.top_n)
        
        log.success("\n✅ 评分完成！")
        
    except Exception as e:
        log.error(f"评分失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

