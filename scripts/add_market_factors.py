"""
为现有特征数据添加市场因子

新增因子：
1. 市场整体趋势
   - market_return_34d: 大盘指数34日涨跌幅
   - market_volatility_34d: 市场34日波动率
   - market_trend: 市场趋势（价格相对均线位置）
   
2. 相对强度特征
   - excess_return: 单日超额收益（个股涨跌幅 - 市场涨跌幅）
   - excess_return_34d: 34日累计超额收益
   - relative_strength: 相对强度指数
   - beta: 个股对市场的敏感度
   
3. 历史强度特征
   - price_vs_hist_mean: 价格相对历史均值位置
   - price_vs_hist_high: 价格相对历史高点位置
   - volatility_vs_hist: 波动率相对历史均值
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.data.market_factors import MarketFactors
from src.utils.logger import log


def add_market_factors_to_features(
    features_df: pd.DataFrame,
    fetcher,
    window: int = 34
) -> pd.DataFrame:
    """
    为特征数据添加市场因子
    
    Args:
        features_df: 特征数据（需包含 trade_date, pct_chg, close）
        fetcher: 数据获取器
        window: 计算窗口
        
    Returns:
        添加了市场因子的DataFrame
    """
    log.info("="*80)
    log.info("添加市场因子")
    log.info("="*80)
    
    mf = MarketFactors(fetcher)
    
    # 确保日期格式
    features_df['trade_date'] = pd.to_datetime(features_df['trade_date'])
    
    # 获取日期范围
    start_date = features_df['trade_date'].min()
    end_date = features_df['trade_date'].max()
    
    # 往前多取数据用于计算
    extended_start = start_date - timedelta(days=window * 3 + 50)
    
    log.info(f"数据日期范围: {start_date.date()} 至 {end_date.date()}")
    log.info(f"获取市场数据: {extended_start.date()} 至 {end_date.date()}")
    
    # 获取市场指数数据
    market_data = mf.get_index_data(
        index_code='000001.SH',
        start_date=extended_start.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    
    if market_data.empty:
        log.error("无法获取市场数据！")
        return features_df
    
    log.success(f"✓ 获取市场数据: {len(market_data)} 条")
    
    # 计算市场趋势指标
    market_data = mf.calculate_market_return(market_data, window)
    
    # 准备合并的市场数据
    market_cols_to_merge = [
        'trade_date',
        'pct_chg',  # 市场日涨跌幅
        f'market_return_{window}d',
        f'market_volatility_{window}d',
        'market_trend'
    ]
    
    # 只保留存在的列
    market_cols_to_merge = [c for c in market_cols_to_merge if c in market_data.columns]
    market_subset = market_data[market_cols_to_merge].copy()
    
    # 重命名市场涨跌幅列
    market_subset = market_subset.rename(columns={'pct_chg': 'market_pct_chg'})
    
    log.info(f"市场因子列: {market_subset.columns.tolist()}")
    
    # 合并市场数据到特征数据
    result = pd.merge(features_df, market_subset, on='trade_date', how='left')
    
    # 计算超额收益
    if 'pct_chg' in result.columns and 'market_pct_chg' in result.columns:
        result['excess_return'] = result['pct_chg'] - result['market_pct_chg']
        log.info("✓ 计算超额收益 (excess_return)")
    
    # 按样本分组计算更多指标
    log.info("按样本计算相对强度指标...")
    
    # 计算每个样本的累计超额收益
    if 'sample_id' in result.columns:
        # 按样本分组计算累计超额收益
        result['excess_return_cumsum'] = result.groupby('sample_id')['excess_return'].cumsum()
    
    # 计算历史强度特征（使用滚动窗口）
    log.info("计算历史强度特征...")
    
    # 价格相对历史均值（使用较长的回看窗口）
    lookback = 60  # 使用60天历史数据
    
    # 这些指标需要按股票分组计算
    if 'ts_code' in result.columns:
        result = result.sort_values(['ts_code', 'trade_date'])
        
        # 价格相对历史均值
        result['price_vs_hist_mean'] = result.groupby('ts_code').apply(
            lambda x: (x['close'] / x['close'].rolling(window=lookback, min_periods=20).mean() - 1) * 100
        ).reset_index(level=0, drop=True)
        
        # 价格相对历史高点
        result['price_vs_hist_high'] = result.groupby('ts_code').apply(
            lambda x: (x['close'] / x['close'].rolling(window=lookback, min_periods=20).max() - 1) * 100
        ).reset_index(level=0, drop=True)
        
        # 波动率相对历史（当前波动率/历史平均波动率）
        result['current_volatility'] = result.groupby('ts_code')['pct_chg'].transform(
            lambda x: x.rolling(window=window, min_periods=10).std()
        )
        result['hist_volatility'] = result.groupby('ts_code')['pct_chg'].transform(
            lambda x: x.rolling(window=lookback, min_periods=20).std()
        )
        result['volatility_vs_hist'] = result['current_volatility'] / result['hist_volatility']
        
        # 清理临时列
        result = result.drop(columns=['current_volatility', 'hist_volatility'], errors='ignore')
    
    # 统计新增的列
    new_cols = [c for c in result.columns if c not in features_df.columns]
    log.success(f"✓ 新增 {len(new_cols)} 个市场因子: {new_cols}")
    
    return result


def main():
    """主函数"""
    log.info("="*80)
    log.info("为特征数据添加市场因子")
    log.info("="*80)
    
    # 文件路径
    POSITIVE_FEATURES_FILE = 'data/training/processed/feature_data_34d.csv'
    NEGATIVE_FEATURES_FILE = 'data/training/features/negative_feature_data_v2_34d.csv'
    
    OUTPUT_POSITIVE = 'data/training/processed/feature_data_34d_with_market.csv'
    OUTPUT_NEGATIVE = 'data/training/features/negative_feature_data_v2_34d_with_market.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager()
    log.success("✓ 初始化完成")
    
    # 处理正样本特征
    log.info("\n[步骤2] 处理正样本特征...")
    if os.path.exists(POSITIVE_FEATURES_FILE):
        df_pos = pd.read_csv(POSITIVE_FEATURES_FILE)
        log.info(f"加载正样本特征: {len(df_pos)} 条")
        
        df_pos_with_market = add_market_factors_to_features(
            df_pos, dm.fetcher, window=34
        )
        
        df_pos_with_market.to_csv(OUTPUT_POSITIVE, index=False)
        log.success(f"✓ 正样本特征已保存: {OUTPUT_POSITIVE}")
        log.info(f"  原始特征数: {len(df_pos.columns)}")
        log.info(f"  新特征数: {len(df_pos_with_market.columns)}")
    else:
        log.warning(f"正样本特征文件不存在: {POSITIVE_FEATURES_FILE}")
    
    # 处理负样本特征
    log.info("\n[步骤3] 处理负样本特征...")
    if os.path.exists(NEGATIVE_FEATURES_FILE):
        df_neg = pd.read_csv(NEGATIVE_FEATURES_FILE)
        log.info(f"加载负样本特征: {len(df_neg)} 条")
        
        df_neg_with_market = add_market_factors_to_features(
            df_neg, dm.fetcher, window=34
        )
        
        df_neg_with_market.to_csv(OUTPUT_NEGATIVE, index=False)
        log.success(f"✓ 负样本特征已保存: {OUTPUT_NEGATIVE}")
        log.info(f"  原始特征数: {len(df_neg.columns)}")
        log.info(f"  新特征数: {len(df_neg_with_market.columns)}")
    else:
        log.warning(f"负样本特征文件不存在: {NEGATIVE_FEATURES_FILE}")
    
    log.info("\n" + "="*80)
    log.success("✅ 市场因子添加完成！")
    log.info("="*80)
    
    log.info("\n新增的市场因子说明：")
    log.info("  1. market_pct_chg: 市场（上证指数）日涨跌幅")
    log.info("  2. market_return_34d: 市场34日累计涨跌幅")
    log.info("  3. market_volatility_34d: 市场34日波动率")
    log.info("  4. market_trend: 市场趋势（价格相对均线位置）")
    log.info("  5. excess_return: 单日超额收益（个股 - 市场）")
    log.info("  6. excess_return_cumsum: 累计超额收益")
    log.info("  7. price_vs_hist_mean: 价格相对历史均值位置")
    log.info("  8. price_vs_hist_high: 价格相对历史高点位置")
    log.info("  9. volatility_vs_hist: 波动率相对历史均值")
    
    log.info("\n下一步：")
    log.info("  1. 使用新特征重新训练模型")
    log.info("  2. 比较新旧模型的表现")


if __name__ == '__main__':
    main()

