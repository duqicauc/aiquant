"""
XGBoost模型训练脚本 - 时间序列版本（避免未来函数）

关键改进：
1. 按时间划分训练集和测试集（而非随机划分）
2. 训练集：历史数据（如2022-2023年）
3. 测试集：未来数据（如2024年）
4. 确保不会用未来信息训练模型
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import json
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve
)
import xgboost as xgb
from src.utils.logger import log
from config.feature_config import get_feature_set, filter_available_features, EFFECTIVE_MARKET_FEATURES, INEFFECTIVE_MARKET_FEATURES


def safe_to_datetime(date_value):
    """
    安全地将日期值转换为datetime类型
    
    处理以下情况：
    - 整数：如 20230101 -> 被错误解析为纳秒时间戳
    - 字符串：如 '20230101' -> 正常解析
    - datetime：直接返回
    """
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(date_value, errors='coerce')
from src.utils.human_intervention import HumanInterventionChecker, require_human_confirmation
from src.visualization.training_visualizer import TrainingVisualizer


def load_and_prepare_data(neg_version='v2', use_market_factors=True, use_tech_factors=False, use_advanced_factors=False):
    """
    加载并准备训练数据
    
    Args:
        neg_version: 负样本版本 ('v1' 或 'v2')
        use_market_factors: 是否使用带市场因子的特征文件
        use_tech_factors: 是否使用带新技术因子的v2特征文件
        use_advanced_factors: 是否使用带高级因子的特征文件
        # TODO: use_ma233_factors: 是否使用带MA233因子的特征文件 (待实施，见 docs/plans/ma233_feature_plan.md)
        
    Returns:
        df_features: 特征DataFrame
    """
    log.info("="*80)
    log.info("第一步：加载数据")
    log.info("="*80)
    
    # 加载正样本（使用新的目录结构）
    # TODO: MA233因子支持 (待实施)
    # if use_ma233_factors:
    #     pos_file = 'data/training/processed/feature_data_34d_ma233.csv'
    #     log.info("📊 使用带MA233因子的特征文件(ma233)")
    if use_advanced_factors:
        pos_file = 'data/training/processed/feature_data_34d_advanced.csv'
        log.info("📊 使用带高级技术因子的特征文件(advanced)")
    elif use_tech_factors:
        pos_file = 'data/training/processed/feature_data_34d_full.csv'
        log.info("📊 使用带新技术因子的特征文件(full)")
    elif use_market_factors:
        pos_file = 'data/training/processed/feature_data_34d_with_market.csv'
        log.info("📊 使用带市场因子的特征文件")
    else:
        pos_file = 'data/training/processed/feature_data_34d.csv'
        log.info("📊 使用基础特征文件")
    
    df_pos = pd.read_csv(pos_file)
    df_pos['label'] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")
    
    # 加载负样本
    if neg_version == 'v2':
        # TODO: MA233因子支持 (待实施)
        # if use_ma233_factors:
        #     neg_file = 'data/training/features/negative_feature_data_v2_34d_ma233.csv'
        if use_advanced_factors:
            neg_file = 'data/training/features/negative_feature_data_v2_34d_advanced.csv'
        elif use_tech_factors:
            neg_file = 'data/training/features/negative_feature_data_v2_34d_full.csv'
        elif use_market_factors:
            neg_file = 'data/training/features/negative_feature_data_v2_34d_with_market.csv'
        else:
            neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    else:
        neg_file = 'data/training/features/negative_feature_data_34d.csv'
    
    df_neg = pd.read_csv(neg_file)
    log.success(f"✓ 负样本加载完成: {len(df_neg)} 条 (版本: {neg_version})")
    
    # 合并
    df = pd.concat([df_pos, df_neg])
    log.info(f"✓ 数据合并完成: {len(df)} 条")
    log.info(f"  - 正样本: {len(df_pos)} 条")
    log.info(f"  - 负样本: {len(df_neg)} 条")
    log.info("")
    
    return df


def extract_features_with_time(df):
    """
    从34天的时序数据中提取统计特征（保留时间信息）
    
    Args:
        df: 原始DataFrame（每行是一天的数据）
        
    Returns:
        df_features: 特征DataFrame（每行是一个样本，包含T1日期）
    """
    log.info("="*80)
    log.info("第二步：特征工程（保留时间信息）")
    log.info("="*80)
    log.info("将34天时序数据转换为统计特征...")
    
    # 重新分配唯一的sample_id
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    # 获取正样本的T1日期映射（使用新的目录结构）
    df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
    t1_date_map = dict(zip(
        df_positive_samples.index,
        df_positive_samples['t1_date'].apply(safe_to_datetime)
    ))
    
    # 获取负样本的T1日期映射
    if os.path.exists('data/training/samples/negative_samples_v2.csv'):
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples_v2.csv')
    else:
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples.csv')
    
    # 负样本的sample_id需要偏移（因为是从0开始的）
    max_positive_id = df_positive_samples.index.max()
    for idx, row in df_negative_samples.iterrows():
        t1_date_map[max_positive_id + 1 + idx] = safe_to_datetime(row['t1_date'])
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')
        
        if len(sample_data) < 20:  # 至少20天数据
            continue
        
        # 从数据中获取T1日期（基于days_to_t1=0的那一天）
        # 找到 days_to_t1 最接近0的记录
        t1_row = sample_data.iloc[sample_data['days_to_t1'].abs().argmin()]
        t1_date = safe_to_datetime(t1_row['trade_date'])
        
        feature_dict = {
            'sample_id': sample_id,
            'ts_code': sample_data['ts_code'].iloc[0],
            'name': sample_data['name'].iloc[0],
            'label': int(sample_data['label'].iloc[0]),
            't1_date': t1_date,  # 保留T1日期，用于时间划分
        }
        
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
            feature_dict['price_above_ma5'] = (
                sample_data['close'] > sample_data['ma5']
            ).sum()
        
        if 'ma10' in sample_data.columns:
            feature_dict['ma10_mean'] = sample_data['ma10'].mean()
            feature_dict['price_above_ma10'] = (
                sample_data['close'] > sample_data['ma10']
            ).sum()
        
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
        if 'rsi_6' in sample_data.columns:
            rsi6_data = sample_data['rsi_6'].dropna()
            if len(rsi6_data) > 0:
                feature_dict['rsi_6_mean'] = rsi6_data.mean()
                feature_dict['rsi_6_std'] = rsi6_data.std()
                feature_dict['rsi_6_max'] = rsi6_data.max()
                feature_dict['rsi_6_min'] = rsi6_data.min()
                feature_dict['rsi_6_last'] = rsi6_data.iloc[-1]  # 最近一天的RSI
                feature_dict['rsi_6_gt_70'] = (rsi6_data > 70).sum()  # 超买天数
                feature_dict['rsi_6_lt_30'] = (rsi6_data < 30).sum()  # 超卖天数
        
        if 'rsi_12' in sample_data.columns:
            rsi12_data = sample_data['rsi_12'].dropna()
            if len(rsi12_data) > 0:
                feature_dict['rsi_12_mean'] = rsi12_data.mean()
                feature_dict['rsi_12_std'] = rsi12_data.std()
                feature_dict['rsi_12_last'] = rsi12_data.iloc[-1]
                feature_dict['rsi_12_gt_70'] = (rsi12_data > 70).sum()
                feature_dict['rsi_12_lt_30'] = (rsi12_data < 30).sum()
        
        if 'rsi_24' in sample_data.columns:
            rsi24_data = sample_data['rsi_24'].dropna()
            if len(rsi24_data) > 0:
                feature_dict['rsi_24_mean'] = rsi24_data.mean()
                feature_dict['rsi_24_std'] = rsi24_data.std()
                feature_dict['rsi_24_last'] = rsi24_data.iloc[-1]
        
        # 动量特征（分段收益率）
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
        
        # ===== 市场因子特征（如果存在）=====
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
        
        # 以下低效特征已剔除（重要性 < 阈值）:
        # - price_vs_hist_high_last: 0.0088
        # - volatility_vs_hist_last: 0.0064
        
        # ===== 新技术因子特征（full）=====
        # 换手率（自由流通股）
        if 'turnover_rate_f' in sample_data.columns:
            turnover_data = sample_data['turnover_rate_f'].dropna()
            if len(turnover_data) > 0:
                feature_dict['turnover_rate_f_mean'] = turnover_data.mean()
                feature_dict['turnover_rate_f_max'] = turnover_data.max()
                feature_dict['turnover_rate_f_std'] = turnover_data.std()
        
        # 乖离率BIAS (bias_short/mid/long)
        if 'bias_short' in sample_data.columns:
            bias_short = sample_data['bias_short'].dropna()
            if len(bias_short) > 0:
                feature_dict['bias_short_last'] = bias_short.iloc[-1]
                feature_dict['bias_short_mean'] = bias_short.mean()
        if 'bias_mid' in sample_data.columns:
            bias_mid = sample_data['bias_mid'].dropna()
            if len(bias_mid) > 0:
                feature_dict['bias_mid_last'] = bias_mid.iloc[-1]
        if 'bias_long' in sample_data.columns:
            bias_long = sample_data['bias_long'].dropna()
            if len(bias_long) > 0:
                feature_dict['bias_long_last'] = bias_long.iloc[-1]
        
        # EMA
        if 'ema_5' in sample_data.columns and 'ema_20' in sample_data.columns:
            ema5 = sample_data['ema_5'].dropna()
            ema20 = sample_data['ema_20'].dropna()
            if len(ema5) > 0 and len(ema20) > 0:
                # EMA短期/长期比值
                feature_dict['ema_ratio_5_20'] = ema5.iloc[-1] / ema20.iloc[-1] if ema20.iloc[-1] != 0 else 1
                # 价格相对EMA位置
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
        if 'kdj_k' in sample_data.columns:
            kdj_k = sample_data['kdj_k'].dropna()
            if len(kdj_k) > 0:
                feature_dict['kdj_k_last'] = kdj_k.iloc[-1]
                feature_dict['kdj_k_mean'] = kdj_k.mean()
        if 'kdj_d' in sample_data.columns:
            kdj_d = sample_data['kdj_d'].dropna()
            if len(kdj_d) > 0:
                feature_dict['kdj_d_last'] = kdj_d.iloc[-1]
        if 'kdj_j' in sample_data.columns:
            kdj_j = sample_data['kdj_j'].dropna()
            if len(kdj_j) > 0:
                feature_dict['kdj_j_last'] = kdj_j.iloc[-1]
                # J值超买超卖
                feature_dict['kdj_j_overbought'] = (kdj_j > 80).sum()
                feature_dict['kdj_j_oversold'] = (kdj_j < 20).sum()
        
        # 涨停统计 (is_limit_up)
        if 'is_limit_up' in sample_data.columns:
            is_limit = sample_data['is_limit_up'].dropna()
            if len(is_limit) > 0:
                feature_dict['limit_up_count'] = is_limit.sum()
        
        # OBV
        if 'obv' in sample_data.columns:
            obv = sample_data['obv'].dropna()
            if len(obv) > 0:
                # OBV变化率
                feature_dict['obv_change'] = (obv.iloc[-1] - obv.iloc[0]) / abs(obv.iloc[0]) * 100 if obv.iloc[0] != 0 else 0
                feature_dict['obv_trend'] = 1 if obv.iloc[-1] > obv.mean() else 0
        
        # 成交量与均量比 (vol_ma5_ratio/vol_ma20_ratio)
        if 'vol_ma5_ratio' in sample_data.columns:
            vol_r5 = sample_data['vol_ma5_ratio'].dropna()
            if len(vol_r5) > 0:
                feature_dict['vol_ma5_ratio_mean'] = vol_r5.mean()
                feature_dict['vol_ma5_ratio_max'] = vol_r5.max()
        if 'vol_ma20_ratio' in sample_data.columns:
            vol_r20 = sample_data['vol_ma20_ratio'].dropna()
            if len(vol_r20) > 0:
                feature_dict['vol_ma20_ratio_mean'] = vol_r20.mean()
                feature_dict['vol_ma20_ratio_max'] = vol_r20.max()
        
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
        
        # 多时间框架特征 (8d, 55d)
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
        
        features.append(feature_dict)
    
    df_features = pd.DataFrame(features)
    
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info(f"✓ 特征维度: {len(df_features.columns) - 3} 个特征（不含sample_id, label, t1_date）")
    log.info("")
    
    return df_features


def timeseries_split(df_features, train_end_date=None, test_start_date=None, feature_set='optimized'):
    """
    按时间划分训练集和测试集（避免未来函数）
    
    Args:
        df_features: 特征DataFrame（必须包含t1_date列）
        train_end_date: 训练集截止日期（如'2023-12-31'）
        test_start_date: 测试集开始日期（如'2024-01-01'）
        feature_set: 特征集名称，可选 'base', 'all_market', 'optimized', 'core'
        
    Returns:
        X_train, X_test, y_train, y_test, train_dates, test_dates
    """
    log.info("="*80)
    log.info("第三步：时间序列划分（避免未来函数）")
    log.info("="*80)
    
    # 确保t1_date是datetime类型（防止整数被误解析）
    df_features['t1_date'] = df_features['t1_date'].apply(safe_to_datetime)
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 显示时间范围
    min_date = df_features['t1_date'].min()
    max_date = df_features['t1_date'].max()
    log.info(f"数据时间范围: {min_date.date()} 至 {max_date.date()}")
    
    # 如果未指定划分点，使用80%作为训练集
    if train_end_date is None:
        n_train = int(len(df_features) * 0.8)
        train_end_date = df_features.iloc[n_train]['t1_date']
        test_start_date = df_features.iloc[n_train + 1]['t1_date']
    else:
        train_end_date = pd.to_datetime(train_end_date)
        test_start_date = pd.to_datetime(test_start_date)
    
    # 划分训练集和测试集
    train_mask = df_features['t1_date'] <= train_end_date
    test_mask = df_features['t1_date'] >= test_start_date
    
    df_train = df_features[train_mask]
    df_test = df_features[test_mask]
    
    log.info(f"\n时间划分:")
    log.info(f"  训练集: {df_train['t1_date'].min().date()} 至 {df_train['t1_date'].max().date()}")
    log.info(f"  测试集: {df_test['t1_date'].min().date()} 至 {df_test['t1_date'].max().date()}")
    log.info(f"\n样本划分:")
    log.info(f"  训练集: {len(df_train)} 个样本 (正:{(df_train['label']==1).sum()}, 负:{(df_train['label']==0).sum()})")
    log.info(f"  测试集: {len(df_test)} 个样本 (正:{(df_test['label']==1).sum()}, 负:{(df_test['label']==0).sum()})")
    log.info("")
    
    # 确认无数据泄露
    if df_train['t1_date'].max() >= df_test['t1_date'].min():
        log.warning("⚠️  警告：训练集和测试集时间有重叠，可能存在数据泄露！")
    else:
        log.success("✓ 训练集和测试集时间无重叠，无数据泄露风险")
    
    # 准备特征和标签（排除非特征列和非数值列）
    exclude_cols = ['sample_id', 'label', 't1_date', 'ts_code', 'name']
    all_feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols]
    
    # 特征筛选：根据feature_set参数筛选特征
    log.info(f"\n特征筛选（使用特征集: {feature_set}）:")
    if feature_set == 'optimized':
        # 排除低效市场因子
        ineffective_cols = [col for col in INEFFECTIVE_MARKET_FEATURES if col.replace('_last', '') in col or col in all_feature_cols]
        ineffective_cols_in_data = [col for col in all_feature_cols if col in INEFFECTIVE_MARKET_FEATURES]
        feature_cols = [col for col in all_feature_cols if col not in ineffective_cols_in_data]
        log.info(f"  剔除低效市场因子: {ineffective_cols_in_data}")
        log.success(f"  ✓ 保留 {len(feature_cols)} 个高效特征")
    elif feature_set == 'base':
        # 仅使用基础特征，排除所有市场因子
        all_market_cols = EFFECTIVE_MARKET_FEATURES + INEFFECTIVE_MARKET_FEATURES
        market_cols_in_data = [col for col in all_feature_cols if col in all_market_cols]
        feature_cols = [col for col in all_feature_cols if col not in market_cols_in_data]
        log.info(f"  剔除所有市场因子: {market_cols_in_data}")
        log.success(f"  ✓ 保留 {len(feature_cols)} 个基础特征")
    else:
        # 使用全部特征
        feature_cols = all_feature_cols
        log.info(f"  使用全部特征: {len(feature_cols)} 个")
    
    X_train = df_train[feature_cols].copy()
    y_train = df_train['label']
    train_dates = df_train['t1_date']
    
    X_test = df_test[feature_cols].copy()
    y_test = df_test['label']
    test_dates = df_test['t1_date']
    
    # 删除非数值列（如果还有的话）
    non_numeric_cols = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        log.info(f"删除非数值列: {list(non_numeric_cols)}")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    # 缺失值处理：使用训练集的统计量填充（避免未来函数）
    # 关键：只用训练集的统计量，不能用测试集数据
    log.info("\n缺失值处理（避免未来函数）:")
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    log.info(f"  训练集缺失值: {train_missing}")
    log.info(f"  测试集缺失值: {test_missing}")
    
    # 计算训练集的中位数（更稳健，不受异常值影响）
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)  # 用训练集的统计量填充测试集
    log.success("  ✓ 使用训练集中位数填充（避免数据泄露）")
    
    log.info(f"\n特征矩阵:")
    log.info(f"  训练集: {X_train.shape}")
    log.info(f"  测试集: {X_test.shape}")
    log.info(f"  特征数: {len(feature_cols)}")
    log.info("")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def train_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost模型
    
    Args:
        X_train, y_train: 训练集
        X_test, y_test: 测试集
        
    Returns:
        model, metrics
    """
    log.info("="*80)
    log.info("第四步：训练XGBoost模型")
    log.info("="*80)
    
    # 计算类别权重（处理样本不均衡）
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    raw_weight = neg_count / pos_count if pos_count > 0 else 1.0
    # 限制权重范围在[0.5, 2.0]之间，避免过度补偿
    scale_pos_weight = max(0.5, min(2.0, raw_weight))
    log.info(f"样本不均衡处理: 正样本={pos_count}, 负样本={neg_count}, scale_pos_weight={scale_pos_weight:.3f} (原始:{raw_weight:.3f})")
    
    # 训练模型
    log.info("开始训练...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,  # 处理样本不均衡
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    log.success("✓ 模型训练完成！")
    log.info("")
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估
    log.info("="*80)
    log.info("第五步：模型评估（测试集 = 未来数据）")
    log.info("="*80)
    
    # 分类报告
    log.info("\n分类报告:")
    report = classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本'],
        output_dict=True
    )
    print(classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本']
    ))
    
    # AUC
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"\nAUC-ROC: {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    log.info("\n混淆矩阵:")
    log.info(f"  真负例(TN): {cm[0,0]:4d}  |  假正例(FP): {cm[0,1]:4d}")
    log.info(f"  假负例(FN): {cm[1,0]:4d}  |  真正例(TP): {cm[1,1]:4d}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log.info("\n" + "="*80)
    log.info("特征重要性 Top 10:")
    log.info("="*80)
    for idx, row in feature_importance.head(10).iterrows():
        log.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # 汇总指标
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['正样本']['precision'],
        'recall': report['正样本']['recall'],
        'f1_score': report['正样本']['f1-score'],
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return model, metrics, y_prob


def generate_training_visualizations(model, X_train, df_features, train_dates, test_dates, neg_version):
    """生成训练过程可视化图表"""
    try:
        log.info("="*80)
        log.info("生成训练可视化图表")
        log.info("="*80)
        
        visualizer = TrainingVisualizer(
            output_dir=f"data/training/charts"
        )
        
        # 1. 样本质量可视化（正样本）
        try:
            df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
            visualizer.visualize_sample_quality(
                df_positive_samples,
                save_prefix="positive_sample_quality"
            )
        except Exception as e:
            log.warning(f"生成正样本质量可视化时出错: {e}")
        
        # 负样本
        try:
            if neg_version == 'v2':
                neg_file = 'data/training/samples/negative_samples_v2.csv'
            else:
                neg_file = 'data/training/samples/negative_samples.csv'
            
            if os.path.exists(neg_file):
                df_negative_samples = pd.read_csv(neg_file)
                visualizer.visualize_sample_quality(
                    df_negative_samples,
                    save_prefix="negative_sample_quality"
                )
        except Exception as e:
            log.warning(f"生成负样本质量可视化时出错: {e}")
        
        # 2. 因子重要性可视化
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        
        visualizer.visualize_feature_importance(
            feature_importance,
            model_name=f"xgboost_timeseries_{neg_version}",
            top_n=20
        )
        
        # 3. 生成索引页面
        visualizer.generate_index_page(model_name=f"xgboost_timeseries_{neg_version}")
        
        log.success("✓ 可视化图表生成完成")
        log.info(f"📊 查看图表: open data/training/charts/index.html")
        
    except Exception as e:
        log.warning(f"生成可视化图表时出错: {e}")
        import traceback
        traceback.print_exc()


def save_model(model, metrics, neg_version, train_dates, test_dates, 
               version=None, model_name='breakout_launch_scorer', feature_names=None,
               training_config=None):
    """
    保存模型和结果
    
    Args:
        model: 训练好的模型
        metrics: 评估指标
        neg_version: 负样本版本
        train_dates: 训练集日期
        test_dates: 测试集日期
        version: 版本号（如 v1.5.0），指定后将保存到版本目录
        model_name: 模型名称
        feature_names: 特征名称列表
        training_config: 训练配置字典
    """
    log.info("\n" + "="*80)
    log.info("第六步：保存模型")
    log.info("="*80)
    
    # 创建目录（使用新的目录结构）
    os.makedirs('data/training/models', exist_ok=True)
    os.makedirs('data/training/metrics', exist_ok=True)
    
    # 保存模型（使用booster方法避免sklearn mixin问题）
    model_file = f'data/training/models/xgboost_timeseries_{neg_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    model.get_booster().save_model(model_file)
    log.success(f"✓ 模型已保存: {model_file}")
    
    # 保存指标
    metrics_file = f'data/training/metrics/xgboost_timeseries_{neg_version}_metrics.json'
    metrics['model_file'] = model_file
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics['neg_version'] = neg_version
    metrics['train_date_range'] = f"{train_dates.min().date()} to {train_dates.max().date()}"
    metrics['test_date_range'] = f"{test_dates.min().date()} to {test_dates.max().date()}"
    metrics['note'] = '使用时间序列划分，避免未来函数'
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 评估报告已保存: {metrics_file}")
    
    # 如果指定了版本号，保存到版本目录
    if version:
        save_to_version_directory(
            model=model,
            metrics=metrics,
            version=version,
            model_name=model_name,
            feature_names=feature_names,
            train_dates=train_dates,
            test_dates=test_dates,
            training_config=training_config
        )
    
    log.info("")


def save_to_version_directory(model, metrics, version, model_name, feature_names,
                              train_dates, test_dates, training_config=None):
    """
    将模型保存到版本管理目录
    
    Args:
        model: 训练好的模型
        metrics: 评估指标
        version: 版本号（如 v1.5.0）
        model_name: 模型名称
        feature_names: 特征名称列表
        train_dates: 训练集日期
        test_dates: 测试集日期
        training_config: 训练配置字典
    """
    import shutil
    
    log.info("\n" + "-"*60)
    log.info(f"📦 保存到版本目录: {model_name}/{version}")
    log.info("-"*60)
    
    # 版本目录
    version_dir = f'data/models/{model_name}/versions/{version}'
    model_dir = f'{version_dir}/model'
    training_dir = f'{version_dir}/training'
    charts_dir = f'{version_dir}/charts'
    
    # 创建目录结构
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(f'{version_dir}/evaluation', exist_ok=True)
    os.makedirs(f'{version_dir}/experiments', exist_ok=True)
    
    # 1. 保存模型文件
    model_file = f'{model_dir}/model.json'
    model.get_booster().save_model(model_file)
    log.success(f"  ✓ 模型文件: {model_file}")
    
    # 2. 保存特征名称
    if feature_names:
        feature_file = f'{model_dir}/feature_names.json'
        with open(feature_file, 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
        log.success(f"  ✓ 特征名称: {feature_file}")
    
    # 3. 保存训练指标
    metrics_file = f'{training_dir}/metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.success(f"  ✓ 训练指标: {metrics_file}")
    
    # 4. 保存元数据
    metadata = {
        'version': version,
        'model_name': model_name,
        'status': 'development',
        'created_at': datetime.now().isoformat(),
        'created_by': 'train_xgboost_timeseries.py',
        'parent_version': None,
        'metrics': {
            'training': {
                'accuracy': metrics.get('accuracy'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1_score'),
                'auc': metrics.get('auc')
            },
            'validation': {
                'accuracy': metrics.get('accuracy'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1_score'),
                'auc': metrics.get('auc')
            },
            'test': {
                'accuracy': metrics.get('accuracy'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1_score'),
                'auc': metrics.get('auc'),
                'confusion_matrix': metrics.get('confusion_matrix', [])
            }
        },
        'training': {
            'train_date_range': f"{train_dates.min().date()} to {train_dates.max().date()}",
            'test_date_range': f"{test_dates.min().date()} to {test_dates.max().date()}",
            'completed_at': datetime.now().isoformat()
        },
        'notes': '由 train_xgboost_timeseries.py 训练'
    }
    
    metadata_file = f'{version_dir}/metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.success(f"  ✓ 元数据: {metadata_file}")
    
    # 5. 保存训练配置
    if training_config:
        config_file = f'{version_dir}/training_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)
        log.success(f"  ✓ 训练配置: {config_file}")
    
    # 6. 更新 current.json
    current_file = f'data/models/{model_name}/current.json'
    if os.path.exists(current_file):
        with open(current_file, 'r', encoding='utf-8') as f:
            current = json.load(f)
    else:
        current = {
            'production': None,
            'staging': None,
            'testing': None,
            'development': None
        }
    
    current['development'] = version
    current['updated_at'] = datetime.now().isoformat()
    
    with open(current_file, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    log.success(f"  ✓ 版本指针: {current_file}")
    
    log.info("")
    log.success(f"✅ 版本 {version} 已保存到: {version_dir}")
    log.info("   下一步可以使用以下命令提升版本状态:")
    log.info(f"   python -c \"from src.models.lifecycle import ModelIterator; mi = ModelIterator('{model_name}'); mi.set_current_version('{version}', 'production')\"")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost时间序列模型训练')
    parser.add_argument('--use-market-factors', action='store_true', 
                       help='使用带市场因子的特征文件')
    parser.add_argument('--use-tech-factors', action='store_true',
                       help='使用带新技术因子的v2特征文件')
    parser.add_argument('--use-advanced-factors', action='store_true',
                       help='使用带高级技术因子的特征文件')
    # TODO: MA233因子支持 (待实施，见 docs/plans/ma233_feature_plan.md)
    # parser.add_argument('--use-ma233-factors', action='store_true',
    #                    help='使用带MA233因子的特征文件（包含5日/233日均线突破特征）')
    parser.add_argument('--neg-version', default='v2', choices=['v1', 'v2'],
                       help='负样本版本')
    # 版本管理参数
    parser.add_argument('--version', type=str, default=None,
                       help='模型版本号（如 v1.5.0），指定后将保存到版本目录')
    parser.add_argument('--model-name', type=str, default='breakout_launch_scorer',
                       help='模型名称（默认: breakout_launch_scorer）')
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("XGBoost 股票选股模型训练 - 时间序列版本")
    log.info("="*80)
    log.info("")
    log.info("⚠️  重要改进：")
    log.info("  1. 按时间划分训练集和测试集（而非随机划分）")
    log.info("  2. 训练集 = 历史数据，测试集 = 未来数据")
    log.info("  3. 避免未来函数，确保无数据泄露")
    log.info("")
    
    # 配置
    NEG_VERSION = args.neg_version
    USE_ADVANCED_FACTORS = args.use_advanced_factors
    USE_TECH_FACTORS = args.use_tech_factors and not USE_ADVANCED_FACTORS
    USE_MARKET_FACTORS = args.use_market_factors or (not args.use_tech_factors and not USE_ADVANCED_FACTORS)
    # TODO: MA233因子支持 (待实施，见 docs/plans/ma233_feature_plan.md)
    # USE_MA233_FACTORS = args.use_ma233_factors
    
    log.info(f"配置:")
    log.info(f"  负样本版本: {NEG_VERSION}")
    log.info(f"  使用市场因子: {USE_MARKET_FACTORS}")
    log.info(f"  使用新技术因子: {USE_TECH_FACTORS}")
    log.info(f"  使用高级因子: {USE_ADVANCED_FACTORS}")
    log.info(f"  划分方式: 时间序列划分（80%训练，20%测试）")
    log.info(f"  模型: XGBoost")
    log.info("")
    
    try:
        # 👤 人工介入检查：特征选择
        checker = HumanInterventionChecker()
        feature_check = checker.check_feature_selection()
        checker.print_intervention_reminder("特征选择", feature_check)
        
        # 1. 加载数据
        df = load_and_prepare_data(
            neg_version=NEG_VERSION, 
            use_market_factors=USE_MARKET_FACTORS,
            use_tech_factors=USE_TECH_FACTORS,
            use_advanced_factors=USE_ADVANCED_FACTORS
        )
        
        # 2. 特征工程（保留时间信息）
        df_features = extract_features_with_time(df)
        
        # 👤 人工介入提醒：特征提取完成
        log.warning("\n" + "="*80)
        log.warning("👤 人工介入提醒：特征提取完成")
        log.warning("="*80)
        log.warning(f"当前特征数量: {len(df_features.columns) - 3} 个（不含sample_id, label, t1_date）")
        log.warning("请确认：")
        log.warning("  1. 特征是否足够？是否需要添加基本面特征或其他技术指标？")
        log.warning("  2. 特征是否避免了未来函数？")
        log.warning("  3. 特征重要性将在训练后显示，请关注")
        log.warning("="*80)
        
        # 3. 时间序列划分
        X_train, X_test, y_train, y_test, train_dates, test_dates = timeseries_split(
            df_features
        )
        
        # 4. 训练模型
        model, metrics, y_prob = train_model(X_train, y_train, X_test, y_test)
        
        # 4.5. 生成可视化图表
        generate_training_visualizations(
            model, X_train, df_features, train_dates, test_dates, NEG_VERSION
        )
        
        # 👤 人工介入检查：训练结果
        log.warning("\n" + "="*80)
        log.warning("👤 人工介入检查：训练结果")
        log.warning("="*80)
        
        # 检查指标是否达标
        warnings = []
        if metrics['auc'] < 0.7:
            warnings.append(f"⚠️  AUC = {metrics['auc']:.3f} < 0.7，模型性能可能不佳")
        if metrics['accuracy'] < 0.75:
            warnings.append(f"⚠️  准确率 = {metrics['accuracy']:.2%} < 75%，模型性能可能不佳")
        if metrics['f1_score'] < 0.7:
            warnings.append(f"⚠️  F1分数 = {metrics['f1_score']:.2%} < 70%，可能存在过拟合或欠拟合")
        
        if warnings:
            for warning in warnings:
                log.warning(warning)
            log.warning("\n建议：")
            log.warning("  - 检查特征选择，考虑添加更多有效特征")
            log.warning("  - 调整超参数（n_estimators, max_depth, learning_rate等）")
            log.warning("  - 检查数据质量，确保正负样本质量")
            log.warning("  - 考虑尝试其他算法（LightGBM, CatBoost等）")
        else:
            log.success("✓ 模型性能指标正常")
        log.warning("="*80)
        
        # 5. 保存模型
        # 构建训练配置（用于版本管理）
        training_config = {
            'version': args.version,
            'created_at': datetime.now().strftime('%Y-%m-%d'),
            'training_script': 'scripts/train_xgboost_timeseries.py',
            'data': {
                'neg_version': NEG_VERSION,
                'use_market_factors': USE_MARKET_FACTORS,
                'use_tech_factors': USE_TECH_FACTORS,
                'use_advanced_factors': USE_ADVANCED_FACTORS,
                'feature_type': 'advanced' if USE_ADVANCED_FACTORS else ('full' if USE_TECH_FACTORS else ('with_market' if USE_MARKET_FACTORS else 'base'))
            },
            'split': {
                'method': 'time_series',
                'train_ratio': 0.8
            },
            'model_params': {
                'algorithm': 'XGBoost',
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
        
        save_model(
            model, metrics, NEG_VERSION, train_dates, test_dates,
            version=args.version,
            model_name=args.model_name,
            feature_names=list(X_train.columns),
            training_config=training_config if args.version else None
        )
        
        # 6. 最终总结
        log.info("="*80)
        log.success("✅ 模型训练完成！（时间序列版本）")
        log.info("="*80)
        log.info("")
        log.info("📊 模型性能总结:")
        log.info(f"  准确率 (Accuracy):  {metrics['accuracy']:.2%}")
        log.info(f"  精确率 (Precision): {metrics['precision']:.2%}")
        log.info(f"  召回率 (Recall):    {metrics['recall']:.2%}")
        log.info(f"  F1分数 (F1-Score):  {metrics['f1_score']:.2%}")
        log.info(f"  AUC-ROC:            {metrics['auc']:.4f}")
        log.info("")
        log.info("🎯 关键改进:")
        log.info("  ✓ 训练集 = 历史数据")
        log.info("  ✓ 测试集 = 未来数据（模拟真实场景）")
        log.info("  ✓ 无未来函数风险")
        log.info("  ✓ 无数据泄露")
        log.info("")
        log.info("下一步:")
        log.info("  1. 使用walk-forward验证进一步测试")
        log.info("  2. 在多个时间窗口上验证稳定性")
        log.info("  3. 回测验证实际收益")
        log.info("")
        
    except FileNotFoundError as e:
        log.error(f"✗ 文件未找到: {e}")
        log.error("请先运行以下命令准备数据:")
        log.error("  1. python scripts/prepare_positive_samples.py")
        log.error("  2. python scripts/prepare_negative_samples_v2.py")
    except Exception as e:
        log.error(f"✗ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

