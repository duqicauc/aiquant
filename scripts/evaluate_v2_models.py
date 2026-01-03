"""
V2模型预测评估脚本

用v2.0.0和v2.1.0模型预测指定日期的股票，并评估预测效果

使用方法：
    python scripts/evaluate_v2_models.py --predict-date 20251212 --eval-date 20251231
"""
import sys
import os
import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def safe_to_datetime(date_value):
    """安全地将日期值转换为datetime类型"""
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(date_value, errors='coerce')


def load_model(version: str):
    """加载指定版本的模型"""
    model_path = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version / 'model' / 'model.json'
    feature_path = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version / 'model' / 'feature_names.json'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    
    with open(feature_path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    
    return booster, feature_names


def get_valid_stock_list(dm):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list(list_status='L')
    
    # 过滤ST
    st_mask = stock_list['name'].str.contains('ST', na=False, case=False)
    stock_list = stock_list[~st_mask]
    
    # 过滤北交所
    bj_mask = stock_list['ts_code'].str.endswith('.BJ')
    stock_list = stock_list[~bj_mask]
    
    # 过滤退市整理期
    delisting_mask = stock_list['name'].str.contains('退', na=False)
    stock_list = stock_list[~delisting_mask]
    
    return stock_list


def _vectorized_rolling_slope_fast(y: np.ndarray, window: int) -> np.ndarray:
    """向量化滚动斜率计算"""
    n = len(y)
    if n < window:
        return np.full(n, np.nan)
    
    s = pd.Series(y)
    x = np.arange(window)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denom = window * sum_x2 - sum_x ** 2
    
    if denom == 0:
        return np.full(n, np.nan)
    
    sum_y = s.rolling(window).sum()
    idx = np.arange(n)
    weighted = pd.Series(idx * y)
    sum_idx_y = weighted.rolling(window).sum()
    t_minus_w_plus_1 = pd.Series(idx - window + 1)
    sum_xy = sum_idx_y - t_minus_w_plus_1 * sum_y
    slope = (window * sum_xy - sum_x * sum_y) / denom
    
    return slope.values


def calculate_advanced_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算高级技术因子"""
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    n = len(df)
    if n < 10:
        return df
    
    # 1. 动量因子
    for period in [5, 10, 20]:
        if n >= period:
            df[f'momentum_{period}d'] = df['close'].pct_change(period) * 100
    
    if 'momentum_10d' in df.columns and n >= 15:
        df['momentum_acceleration'] = df['momentum_10d'].diff(5)
    
    # 2. 量价配合
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
    
    # 3. 多时间框架特征
    for tf in [8, 34, 55]:
        if n >= tf:
            df[f'return_{tf}d'] = (df['close'] - df['close'].shift(tf)) / df['close'].shift(tf) * 100
            df[f'ma_{tf}d'] = df['close'].rolling(tf).mean()
            df[f'price_vs_ma_{tf}d'] = (df['close'] - df[f'ma_{tf}d']) / df[f'ma_{tf}d'] * 100
            df[f'volatility_{tf}d'] = df['pct_chg'].rolling(tf).std()
            df[f'high_{tf}d'] = df['close'].rolling(tf).max()
            df[f'low_{tf}d'] = df['close'].rolling(tf).min()
            df[f'price_position_{tf}d'] = (df['close'] - df[f'low_{tf}d']) / (df[f'high_{tf}d'] - df[f'low_{tf}d'] + 1e-8) * 100
            df[f'trend_slope_{tf}d'] = _vectorized_rolling_slope_fast(df['close'].values, tf)
    
    # 4. 突破形态
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
    
    # 5. 支撑/阻力位
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
    
    # 6. 成交量特征增强
    if 'vol' in df.columns:
        for period in [10, 20]:
            if n >= period:
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


def get_market_data(dm, predict_date: str, lookback_days: int = 120) -> pd.DataFrame:
    """获取市场数据"""
    try:
        start_date = (datetime.strptime(predict_date, '%Y%m%d') - timedelta(days=lookback_days)).strftime('%Y%m%d')
        df_index = dm.get_index_daily('000300.SH', start_date, predict_date)
        
        if df_index is None or df_index.empty:
            return None
        
        df_index['trade_date'] = pd.to_datetime(df_index['trade_date'])
        df_index = df_index.rename(columns={'pct_chg': 'market_pct_chg', 'close': 'market_close'})
        df_index = df_index.sort_values('trade_date')
        df_index['market_return_34d'] = df_index['market_close'].pct_change(34) * 100
        df_index['market_volatility_34d'] = df_index['market_pct_chg'].rolling(34).std()
        df_index['market_trend'] = (df_index['market_return_34d'] > 0).astype(int)
        
        return df_index[['trade_date', 'market_pct_chg', 'market_return_34d', 'market_volatility_34d', 'market_trend']]
    except Exception as e:
        log.warning(f"获取市场数据失败: {e}")
        return None


def extract_features_v20(df: pd.DataFrame) -> dict:
    """提取v2.0.0模型所需的27个基础特征"""
    if len(df) < 20:
        return None
    
    features = {}
    
    # 价格特征
    features['close_mean'] = df['close'].mean()
    features['close_std'] = df['close'].std()
    features['close_max'] = df['close'].max()
    features['close_min'] = df['close'].min()
    features['close_trend'] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    
    # 涨跌幅特征
    features['pct_chg_mean'] = df['pct_chg'].mean()
    features['pct_chg_std'] = df['pct_chg'].std()
    features['pct_chg_sum'] = df['pct_chg'].sum()
    features['positive_days'] = (df['pct_chg'] > 0).sum()
    features['negative_days'] = (df['pct_chg'] < 0).sum()
    features['max_gain'] = df['pct_chg'].max()
    features['max_loss'] = df['pct_chg'].min()
    
    # 量比特征
    if 'volume_ratio' in df.columns:
        vr_data = df['volume_ratio'].dropna()
        if len(vr_data) > 0:
            features['volume_ratio_mean'] = vr_data.mean()
            features['volume_ratio_max'] = vr_data.max()
            features['volume_ratio_gt_2'] = (vr_data > 2).sum()
            features['volume_ratio_gt_4'] = (vr_data > 4).sum()
    
    # MACD特征
    if 'macd' in df.columns:
        macd_data = df['macd'].dropna()
        if len(macd_data) > 0:
            features['macd_mean'] = macd_data.mean()
            features['macd_positive_days'] = (macd_data > 0).sum()
            features['macd_max'] = macd_data.max()
    
    # MA特征
    if 'ma5' in df.columns:
        features['ma5_mean'] = df['ma5'].mean()
        features['price_above_ma5'] = (df['close'] > df['ma5']).sum()
    
    if 'ma10' in df.columns:
        features['ma10_mean'] = df['ma10'].mean()
        features['price_above_ma10'] = (df['close'] > df['ma10']).sum()
    
    # 市值特征
    if 'total_mv' in df.columns:
        mv_data = df['total_mv'].dropna()
        if len(mv_data) > 0:
            features['total_mv_mean'] = mv_data.mean()
    
    if 'circ_mv' in df.columns:
        circ_mv_data = df['circ_mv'].dropna()
        if len(circ_mv_data) > 0:
            features['circ_mv_mean'] = circ_mv_data.mean()
    
    # 动量特征
    days = len(df)
    if days >= 7:
        features['return_1w'] = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
    if days >= 14:
        features['return_2w'] = (df['close'].iloc[-1] - df['close'].iloc[-14]) / df['close'].iloc[-14] * 100
    
    return features


def extract_features_v21(df: pd.DataFrame, feature_names: list) -> dict:
    """提取v2.1.0模型所需的125个高级特征"""
    if len(df) < 20:
        return None
    
    features = {}
    
    # 对每个特征名，从数据中取值（使用均值或最后值）
    for fn in feature_names:
        if fn in df.columns:
            col_data = pd.to_numeric(df[fn], errors='coerce').dropna()
            if len(col_data) > 0:
                # 根据特征名后缀决定聚合方式
                if fn.endswith('_last') or fn.endswith('_sum') or fn.endswith('_count'):
                    features[fn] = col_data.iloc[-1]
                elif fn.endswith('_mean'):
                    features[fn] = col_data.mean()
                elif fn.endswith('_std'):
                    features[fn] = col_data.std()
                elif fn.endswith('_max'):
                    features[fn] = col_data.max()
                elif fn.endswith('_min'):
                    features[fn] = col_data.min()
                else:
                    # 默认使用均值
                    features[fn] = col_data.mean()
    
    return features


def get_stock_data(dm, ts_code: str, predict_date: str, df_market: pd.DataFrame = None, lookback_days: int = 90):
    """获取股票数据并计算所有必要的因子"""
    end_dt = pd.to_datetime(predict_date)
    start_date = (end_dt - timedelta(days=lookback_days)).strftime('%Y%m%d')
    
    try:
        # 获取日线数据
        df = dm.get_complete_data(ts_code, start_date, predict_date)
        if df.empty or len(df) < 34:
            return None
        
        # 获取技术因子
        try:
            df_factor = dm.get_stk_factor(ts_code, start_date, predict_date)
            if not df_factor.empty:
                factor_cols = ['trade_date', 'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24',
                              'turnover_rate_f', 'kdj_k', 'kdj_d', 'kdj_j', 'obv']
                available_cols = [c for c in factor_cols if c in df_factor.columns]
                df = pd.merge(df, df_factor[available_cols], on='trade_date', how='left')
        except Exception:
            pass
        
        # 计算MA
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        
        # 合并市场数据
        if df_market is not None:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = pd.merge(df, df_market, on='trade_date', how='left')
            
            # 计算超额收益
            if 'pct_chg' in df.columns and 'market_pct_chg' in df.columns:
                df['excess_return'] = df['pct_chg'] - df['market_pct_chg']
                df['excess_return_cumsum'] = df['excess_return'].cumsum()
        
        # 计算高级技术因子
        df = calculate_advanced_factors(df)
        
        # 计算其他派生因子
        if 'ma5' in df.columns:
            df['bias_short'] = (df['close'] - df['ma5']) / df['ma5'] * 100
        if 'ma10' in df.columns:
            df['bias_mid'] = (df['close'] - df['ma10']) / df['ma10'] * 100
        if 'ma_20d' in df.columns:
            df['bias_long'] = (df['close'] - df['ma_20d']) / df['ma_20d'] * 100
        
        # EMA
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
        
        # vol_ma比率
        if 'vol' in df.columns:
            df['vol_ma5_ratio'] = df['vol'] / (df['vol'].rolling(5).mean() + 1e-8)
            df['vol_ma20_ratio'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-8)
        
        # 涨停标记
        df['is_limit_up'] = (df['pct_chg'] >= 9.8).astype(int)
        
        # 历史价格位置
        if len(df) >= 34:
            df['price_vs_hist_mean'] = (df['close'] - df['close'].rolling(34).mean()) / df['close'].rolling(34).mean() * 100
            df['price_vs_hist_high'] = (df['close'] - df['close'].rolling(34).max()) / df['close'].rolling(34).max() * 100
            df['volatility_vs_hist'] = df['pct_chg'].rolling(10).std() / (df['pct_chg'].rolling(34).std() + 1e-8)
        
        # 取最后34天
        df = df.tail(34)
        
        return df
    
    except Exception as e:
        return None


def score_stocks(dm, stock_list, booster, feature_names, predict_date, version, df_market):
    """对股票进行评分"""
    log.info(f"\n开始用 {version} 模型评分...")
    
    results = []
    total = len(stock_list)
    
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        if (idx + 1) % 200 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")
        
        # 获取股票数据
        df = get_stock_data(dm, ts_code, predict_date, df_market)
        if df is None or len(df) < 20:
            continue
        
        # 根据版本提取特征
        if version == 'v2.0.0':
            features = extract_features_v20(df)
        else:
            features = extract_features_v21(df, feature_names)
        
        if features is None:
            continue
        
        # 构建特征向量
        feature_vector = []
        for fn in feature_names:
            if fn in features:
                val = features[fn]
                feature_vector.append(0 if pd.isna(val) else val)
            else:
                feature_vector.append(0)
        
        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        prob = booster.predict(dmatrix)[0]
        
        results.append({
            'ts_code': ts_code,
            'name': name,
            'probability': prob,
            'predict_price': df['close'].iloc[-1],
            'predict_date': str(df['trade_date'].iloc[-1])[:10]
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('probability', ascending=False)
    
    log.success(f"✓ {version} 评分完成: {len(df_results)} 只股票")
    
    return df_results


def evaluate_predictions(dm, df_predictions, eval_date, version):
    """评估预测结果"""
    log.info(f"\n评估 {version} 预测结果...")
    
    results = []
    
    for idx, row in df_predictions.iterrows():
        ts_code = row['ts_code']
        predict_price = row['predict_price']
        
        try:
            eval_dt = pd.to_datetime(eval_date)
            start_date = (eval_dt - timedelta(days=5)).strftime('%Y%m%d')
            end_date = eval_date
            
            df = dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
            
            if df.empty:
                results.append({**row.to_dict(), 'eval_price': None, 'return_pct': None, 'status': '无数据'})
                continue
            
            eval_price = df['close'].iloc[-1]
            eval_date_actual = df['trade_date'].iloc[-1]
            
            if predict_price > 0:
                return_pct = (eval_price - predict_price) / predict_price * 100
            else:
                return_pct = None
            
            results.append({
                **row.to_dict(),
                'eval_price': eval_price,
                'eval_date_actual': eval_date_actual,
                'return_pct': return_pct,
                'status': '正常'
            })
            
        except Exception as e:
            results.append({**row.to_dict(), 'eval_price': None, 'return_pct': None, 'status': f'错误:{e}'})
    
    df_eval = pd.DataFrame(results)
    
    return df_eval


def print_evaluation_summary(df_eval, version):
    """打印评估摘要"""
    log.info(f"\n{'='*80}")
    log.info(f"{version} 模型评估结果")
    log.info(f"{'='*80}")
    
    df_valid = df_eval[df_eval['return_pct'].notna()]
    
    if len(df_valid) == 0:
        log.warning("无有效评估数据")
        return {}
    
    avg_return = df_valid['return_pct'].mean()
    median_return = df_valid['return_pct'].median()
    win_rate = (df_valid['return_pct'] > 0).sum() / len(df_valid) * 100
    max_return = df_valid['return_pct'].max()
    min_return = df_valid['return_pct'].min()
    
    log.info(f"\n📊 整体统计（Top 50）:")
    log.info(f"  有效股票数: {len(df_valid)}")
    log.info(f"  平均收益率: {avg_return:.2f}%")
    log.info(f"  中位数收益: {median_return:.2f}%")
    log.info(f"  胜率: {win_rate:.1f}%")
    log.info(f"  最高收益: {max_return:.2f}%")
    log.info(f"  最低收益: {min_return:.2f}%")
    
    # 按概率区间分析
    log.info(f"\n📈 按概率区间分析:")
    bins = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6)]
    for low, high in bins:
        subset = df_valid[(df_valid['probability'] >= low) & (df_valid['probability'] < high)]
        if len(subset) > 0:
            sub_return = subset['return_pct'].mean()
            sub_win = (subset['return_pct'] > 0).sum() / len(subset) * 100
            log.info(f"  概率 {low:.0%}-{high:.0%}: {len(subset)}只, 平均收益{sub_return:.2f}%, 胜率{sub_win:.1f}%")
    
    # 显示收益最好的5只
    log.info(f"\n🏆 收益最高的5只:")
    top5 = df_valid.nlargest(5, 'return_pct')
    for _, row in top5.iterrows():
        log.info(f"  {row['ts_code']} {row['name']}: 概率{row['probability']:.2%}, 收益{row['return_pct']:.2f}%")
    
    # 显示收益最差的5只
    log.info(f"\n❌ 收益最低的5只:")
    bottom5 = df_valid.nsmallest(5, 'return_pct')
    for _, row in bottom5.iterrows():
        log.info(f"  {row['ts_code']} {row['name']}: 概率{row['probability']:.2%}, 收益{row['return_pct']:.2f}%")
    
    return {
        'avg_return': avg_return,
        'median_return': median_return,
        'win_rate': win_rate,
        'max_return': max_return,
        'min_return': min_return,
        'valid_count': len(df_valid)
    }


def main():
    parser = argparse.ArgumentParser(description='V2模型预测评估')
    parser.add_argument('--predict-date', type=str, default='20251212', help='预测日期')
    parser.add_argument('--eval-date', type=str, default='20251231', help='评估日期')
    parser.add_argument('--top-n', type=int, default=50, help='Top N股票数量')
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("V2模型预测评估")
    log.info("="*80)
    log.info(f"预测日期: {args.predict_date}")
    log.info(f"评估日期: {args.eval_date}")
    log.info(f"Top N: {args.top_n}")
    log.info("")
    
    # 初始化数据管理器
    dm = DataManager()
    
    # 获取股票列表
    stock_list = get_valid_stock_list(dm)
    log.info(f"有效股票数: {len(stock_list)}")
    
    # 获取市场数据
    log.info("\n获取市场数据...")
    df_market = get_market_data(dm, args.predict_date)
    if df_market is not None:
        log.success(f"✓ 市场数据已获取: {len(df_market)} 条记录")
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载两个模型
    log.info("\n加载模型...")
    booster_v20, features_v20 = load_model('v2.0.0')
    log.success(f"✓ v2.0.0 模型加载成功: {len(features_v20)} 个特征")
    
    booster_v21, features_v21 = load_model('v2.1.0')
    log.success(f"✓ v2.1.0 模型加载成功: {len(features_v21)} 个特征")
    
    # v2.0.0 评分
    log.info("\n" + "="*80)
    log.info("v2.0.0 模型评分（27个基础特征）")
    log.info("="*80)
    df_scores_v20 = score_stocks(dm, stock_list, booster_v20, features_v20, args.predict_date, 'v2.0.0', df_market)
    df_top50_v20 = df_scores_v20.head(args.top_n)
    
    # 保存Top50
    df_top50_v20.to_csv(output_dir / f'v2.0.0_top{args.top_n}_{args.predict_date}.csv', index=False)
    log.info(f"\nv2.0.0 Top {args.top_n}股票:")
    for i, row in df_top50_v20.head(10).iterrows():
        log.info(f"  {row['ts_code']} {row['name']}: {row['probability']:.4f}")
    
    # v2.0.0 评估
    df_eval_v20 = evaluate_predictions(dm, df_top50_v20, args.eval_date, 'v2.0.0')
    stats_v20 = print_evaluation_summary(df_eval_v20, 'v2.0.0')
    
    # v2.1.0 评分
    log.info("\n" + "="*80)
    log.info("v2.1.0 模型评分（125个高级特征）")
    log.info("="*80)
    df_scores_v21 = score_stocks(dm, stock_list, booster_v21, features_v21, args.predict_date, 'v2.1.0', df_market)
    df_top50_v21 = df_scores_v21.head(args.top_n)
    
    # 保存Top50
    df_top50_v21.to_csv(output_dir / f'v2.1.0_top{args.top_n}_{args.predict_date}.csv', index=False)
    log.info(f"\nv2.1.0 Top {args.top_n}股票:")
    for i, row in df_top50_v21.head(10).iterrows():
        log.info(f"  {row['ts_code']} {row['name']}: {row['probability']:.4f}")
    
    # v2.1.0 评估
    df_eval_v21 = evaluate_predictions(dm, df_top50_v21, args.eval_date, 'v2.1.0')
    stats_v21 = print_evaluation_summary(df_eval_v21, 'v2.1.0')
    
    # 保存评估结果
    df_eval_v20.to_csv(output_dir / f'v2.0.0_eval_{args.predict_date}_to_{args.eval_date}.csv', index=False)
    df_eval_v21.to_csv(output_dir / f'v2.1.0_eval_{args.predict_date}_to_{args.eval_date}.csv', index=False)
    
    log.success(f"\n✓ 评估结果已保存到 {output_dir}")
    
    # 对比两个模型
    log.info("\n" + "="*80)
    log.info("两模型对比")
    log.info("="*80)
    
    if stats_v20 and stats_v21:
        log.info(f"\n| 指标 | v2.0.0 | v2.1.0 |")
        log.info(f"|------|--------|--------|")
        log.info(f"| 平均收益率 | {stats_v20['avg_return']:.2f}% | {stats_v21['avg_return']:.2f}% |")
        log.info(f"| 中位数收益 | {stats_v20['median_return']:.2f}% | {stats_v21['median_return']:.2f}% |")
        log.info(f"| 胜率 | {stats_v20['win_rate']:.1f}% | {stats_v21['win_rate']:.1f}% |")
        log.info(f"| 最高收益 | {stats_v20['max_return']:.2f}% | {stats_v21['max_return']:.2f}% |")
        log.info(f"| 最低收益 | {stats_v20['min_return']:.2f}% | {stats_v21['min_return']:.2f}% |")
    
    # 找出两个模型都推荐的股票
    common_stocks = set(df_top50_v20['ts_code']) & set(df_top50_v21['ts_code'])
    log.info(f"\n两个模型共同推荐的股票: {len(common_stocks)} 只")
    
    if common_stocks:
        df_common = df_eval_v20[df_eval_v20['ts_code'].isin(common_stocks)]
        df_common_valid = df_common[df_common['return_pct'].notna()]
        if len(df_common_valid) > 0:
            log.info(f"共同推荐股票的平均收益: {df_common_valid['return_pct'].mean():.2f}%")
            log.info(f"共同推荐股票的胜率: {(df_common_valid['return_pct'] > 0).mean()*100:.1f}%")
            
            log.info(f"\n共同推荐的股票列表:")
            for _, row in df_common_valid.iterrows():
                log.info(f"  {row['ts_code']} {row['name']}: 收益{row['return_pct']:.2f}%")


if __name__ == '__main__':
    main()
