# -*- coding: utf-8 -*-
"""
特征配置
定义模型训练使用的特征集合，支持特征筛选和组合
"""

# 基础技术特征（原始特征）
BASE_FEATURES = [
    'close_mean', 'close_std', 'close_min', 'close_max',
    'pct_chg_mean', 'pct_chg_std', 'pct_chg_min', 'pct_chg_max',
    'vol_mean', 'vol_std', 'vol_max',
    'positive_days', 'negative_days',
    'volume_ratio_mean', 'volume_ratio_std', 'volume_ratio_max',
    'turnover_rate_mean', 'turnover_rate_std', 'turnover_rate_max',
    'macd_mean', 'macd_std', 'macd_positive_days',
    'rsi_6_mean', 'rsi_6_std',
    'rsi_12_mean', 'rsi_12_std',
]

# 高效市场因子（经过特征重要性筛选）
# 筛选标准：特征重要性 > 平均重要性的50% (阈值: 0.0263)
EFFECTIVE_MARKET_FEATURES = [
    'excess_return_mean',        # 超额收益均值 (重要性: 0.1037, 排名第2)
    'price_vs_hist_mean_last',   # 价格相对历史均值 (重要性: 0.0480)
    'price_vs_hist_high_last',   # 价格相对历史高点 (重要性: 0.0448)
    'excess_return_sum',         # 累计超额收益 (重要性: 0.0293)
]

# 低效市场因子（已剔除）
# 筛选标准：特征重要性 < 平均重要性的50% (阈值: 0.0263)
INEFFECTIVE_MARKET_FEATURES = [
    'market_trend_last',             # 市场趋势 (0.0256)
    'market_volatility_34d_last',    # 市场波动率 (0.0245)
    'excess_return_positive_days',   # 超额收益正天数 (0.0245)
    'market_return_34d_last',        # 市场34日收益 (0.0237)
    'volatility_vs_hist_last',       # 波动率相对历史 (0.0232)
    'market_pct_chg_mean',           # 市场涨跌幅均值 (0.0228)
    'excess_return_cumsum_last',     # 累计超额收益最后值 (0.0203)
]

# 预定义特征组合
FEATURE_SETS = {
    # 基础特征（无市场因子）- AUC: 87.75%
    'base': BASE_FEATURES,
    
    # 基础 + 全部市场因子 - AUC: 82.36% (不推荐)
    'all_market': BASE_FEATURES + EFFECTIVE_MARKET_FEATURES + INEFFECTIVE_MARKET_FEATURES,
    
    # 基础 + 高效市场因子 - AUC: 88.59% (推荐)
    'optimized': BASE_FEATURES + EFFECTIVE_MARKET_FEATURES,
    
    # 核心特征（简化版）
    'core': [
        'pct_chg_mean', 'pct_chg_std', 'positive_days', 'negative_days',
        'volume_ratio_mean', 'macd_mean', 'rsi_6_mean',
        'excess_return_mean', 'price_vs_hist_mean_last', 'price_vs_hist_high_last',
    ]
}

# 默认特征集
DEFAULT_FEATURE_SET = 'optimized'


def get_feature_set(name: str = None) -> list:
    """
    获取指定的特征集合
    
    Args:
        name: 特征集名称，可选 'base', 'all_market', 'optimized', 'core'
              默认使用 'optimized'
    
    Returns:
        特征名称列表
    """
    if name is None:
        name = DEFAULT_FEATURE_SET
    
    if name not in FEATURE_SETS:
        raise ValueError(f"未知的特征集: {name}. 可选: {list(FEATURE_SETS.keys())}")
    
    return FEATURE_SETS[name]


def filter_available_features(feature_list: list, available_columns: list) -> list:
    """
    过滤出可用的特征
    
    Args:
        feature_list: 期望的特征列表
        available_columns: 实际可用的列名列表
    
    Returns:
        实际可用的特征列表
    """
    return [f for f in feature_list if f in available_columns]

