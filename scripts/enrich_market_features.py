#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为训练数据补充市场环境特征

对以下三个文件补充真实的市场环境特征：
1. data/training/processed/feature_data_34d_v5.csv（正样本）
2. data/training/features/negative_feature_data_v2_34d_v5.csv（负样本）
3. data/training/features/hard_negative_feature_data_34d_v5.csv（硬负样本）

市场环境特征包括：
- market_pct_chg: 大盘当日涨跌幅
- market_return_34d: 大盘34日收益率
- market_volatility_34d: 大盘34日波动率
- market_trend: 大盘趋势（相对34日均线位置）
- excess_return: 个股超额收益（个股涨跌幅 - 大盘涨跌幅）
- excess_return_cumsum: 累计超额收益（34日滚动累计）
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

from src.data.data_manager import DataManager
from src.data.market_factors import MarketFactors
from src.utils.logger import log


def enrich_with_market_features(df: pd.DataFrame, dm: DataManager, window: int = 34) -> pd.DataFrame:
    """
    为样本数据补充市场环境特征
    
    Args:
        df: 样本数据（需包含 trade_date, pct_chg）
        dm: DataManager 实例
        window: 计算窗口（默认34天）
    
    Returns:
        补充了市场环境特征的DataFrame
    """
    log.info(f"开始补充市场环境特征，数据量: {len(df)} 条")
    
    # 确保 trade_date 是日期类型
    if 'trade_date' not in df.columns:
        log.error("数据中缺少 trade_date 列")
        return df
    
    df = df.copy()
    
    # 转换日期格式
    if df['trade_date'].dtype != 'datetime64[ns]':
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d', errors='coerce')
        except:
            try:
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
            except:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    
    # 删除日期为空的记录
    df = df.dropna(subset=['trade_date']).copy()
    
    if len(df) == 0:
        log.error("日期解析后数据为空")
        return df
    
    # 获取日期范围
    start_date = df['trade_date'].min()
    end_date = df['trade_date'].max()
    
    # 往前多取一些数据用于计算滚动指标
    extended_start = start_date - timedelta(days=window * 2 + 50)
    
    log.info(f"获取市场指数数据: {extended_start.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")
    
    # 获取上证指数数据
    index_code = '000001.SH'
    market_data = dm.get_index_daily(
        index_code,
        extended_start.strftime('%Y%m%d'),
        end_date.strftime('%Y%m%d')
    )
    
    if market_data.empty:
        log.warning("无法获取市场数据，跳过市场环境特征补充")
        return df
    
    # 确保 market_data 的 trade_date 是日期类型
    if market_data['trade_date'].dtype != 'datetime64[ns]':
        market_data['trade_date'] = pd.to_datetime(market_data['trade_date'], errors='coerce')
    
    market_data = market_data.sort_values('trade_date').reset_index(drop=True)
    
    # 计算市场特征
    log.info("计算市场环境特征...")
    
    # 1. 大盘当日涨跌幅
    market_data['market_pct_chg'] = market_data['pct_chg']
    
    # 2. 大盘34日收益率
    market_data['market_return_34d'] = market_data['close'].pct_change(window) * 100
    
    # 3. 大盘34日波动率
    market_data['market_volatility_34d'] = market_data['pct_chg'].rolling(window).std()
    
    # 4. 大盘趋势（相对34日均线位置，百分比）
    market_ma34 = market_data['close'].rolling(window).mean()
    market_data['market_trend'] = (market_data['close'] / market_ma34 - 1) * 100
    
    # ========== v2.5.5 新增市场环境特征 ==========
    
    # 5. 市场短期动量（5日、10日、20日）
    market_data['market_momentum_5d'] = market_data['close'].pct_change(5) * 100
    market_data['market_momentum_10d'] = market_data['close'].pct_change(10) * 100
    market_data['market_momentum_20d'] = market_data['close'].pct_change(20) * 100
    
    # 6. 市场状态（牛市/熊市/震荡市）
    # 基于均线位置：close > ma20 > ma55 = 牛市(2), close > ma20 = 震荡偏多(1), 
    # close < ma20 < ma55 = 熊市(-2), close < ma20 = 震荡偏空(-1)
    market_ma20 = market_data['close'].rolling(20).mean()
    market_ma55 = market_data['close'].rolling(55).mean()
    
    def calc_market_regime(row):
        close = row['close']
        ma20 = row['_ma20']
        ma55 = row['_ma55']
        if pd.isna(ma20) or pd.isna(ma55):
            return 0
        if close > ma20 > ma55:
            return 2  # 牛市
        elif close > ma20:
            return 1  # 震荡偏多
        elif close < ma20 < ma55:
            return -2  # 熊市
        elif close < ma20:
            return -1  # 震荡偏空
        return 0
    
    market_data['_ma20'] = market_ma20
    market_data['_ma55'] = market_ma55
    market_data['market_regime'] = market_data.apply(calc_market_regime, axis=1)
    market_data = market_data.drop(columns=['_ma20', '_ma55'])
    
    # 7. 市场支撑/阻力位置（相对20日高低点）
    market_high_20d = market_data['close'].rolling(20).max()
    market_low_20d = market_data['close'].rolling(20).min()
    market_data['market_position_20d'] = (market_data['close'] - market_low_20d) / (market_high_20d - market_low_20d + 1e-8)
    
    # 准备合并的市场数据列
    market_cols = ['trade_date', 'market_pct_chg', 'market_return_34d', 
                   'market_volatility_34d', 'market_trend',
                   'market_momentum_5d', 'market_momentum_10d', 'market_momentum_20d',
                   'market_regime', 'market_position_20d']
    market_subset = market_data[market_cols].copy()
    
    # 合并市场数据到样本
    log.info("合并市场数据到样本...")
    df = pd.merge(df, market_subset, on='trade_date', how='left')
    
    # 计算超额收益（需要个股的 pct_chg）
    if 'pct_chg' in df.columns:
        # 5. 个股超额收益（个股涨跌幅 - 大盘涨跌幅）
        df['excess_return'] = df['pct_chg'] - df['market_pct_chg']
        
        # 6. 累计超额收益（34日滚动累计）
        # 注意：这里需要按 sample_id 分组计算，因为每个样本是34天的序列
        if 'sample_id' in df.columns:
            df['excess_return_cumsum'] = df.groupby('sample_id')['excess_return'].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        else:
            # 如果没有 sample_id，直接计算
            df['excess_return_cumsum'] = df['excess_return'].rolling(window, min_periods=1).sum()
    else:
        log.warning("数据中缺少 pct_chg 列，无法计算超额收益")
        df['excess_return'] = np.nan
        df['excess_return_cumsum'] = np.nan
    
    # ========== v2.5.5 新增：超额收益持续性 ==========
    if 'excess_return' in df.columns:
        # 8. 超额收益持续性（连续正超额收益天数）
        if 'sample_id' in df.columns:
            df['excess_return_consistency'] = df.groupby('sample_id')['excess_return'].transform(
                lambda x: (x > 0).rolling(10, min_periods=1).sum()
            )
        else:
            df['excess_return_consistency'] = (df['excess_return'] > 0).rolling(10, min_periods=1).sum()
    
    # 检查补充结果
    market_feature_cols = ['market_pct_chg', 'market_return_34d', 'market_volatility_34d', 
                          'market_trend', 'excess_return', 'excess_return_cumsum',
                          'market_momentum_5d', 'market_momentum_10d', 'market_momentum_20d',
                          'market_regime', 'market_position_20d', 'excess_return_consistency']
    
    filled_count = {}
    for col in market_feature_cols:
        if col in df.columns:
            filled = df[col].notna().sum()
            filled_count[col] = filled
    
    log.info(f"市场环境特征补充完成:")
    for col, count in filled_count.items():
        pct = count / len(df) * 100 if len(df) > 0 else 0
        log.info(f"  {col}: {count}/{len(df)} ({pct:.1f}%)")
    
    return df


def process_file(input_file: Path, output_file: Path, dm: DataManager):
    """
    处理单个文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        dm: DataManager 实例
    """
    log.info("="*80)
    log.info(f"处理文件: {input_file.name}")
    log.info("="*80)
    
    if not input_file.exists():
        log.error(f"文件不存在: {input_file}")
        return False
    
    # 读取数据
    log.info(f"读取数据: {input_file}")
    try:
        df = pd.read_csv(input_file)
        log.info(f"  原始数据: {len(df)} 条，{len(df.columns)} 列")
    except Exception as e:
        log.error(f"读取文件失败: {e}")
        return False
    
    # 检查是否已有市场环境特征
    market_cols = ['market_pct_chg', 'market_return_34d', 'market_volatility_34d', 
                   'market_trend', 'excess_return', 'excess_return_cumsum']
    
    existing_cols = [col for col in market_cols if col in df.columns]
    if existing_cols:
        log.info(f"检测到已有市场环境特征: {existing_cols}")
        # 检查是否都是 NaN
        all_nan = all(df[col].isna().all() for col in existing_cols)
        if not all_nan:
            log.warning("部分市场环境特征已有非空值，将覆盖")
        # 删除旧的特征列
        df = df.drop(columns=existing_cols)
        log.info(f"已删除旧的市场环境特征列")
    
    # 补充市场环境特征
    df = enrich_with_market_features(df, dm)
    
    if df is None or len(df) == 0:
        log.error("补充市场环境特征失败")
        return False
    
    # 保存结果
    log.info(f"保存结果: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 文件处理完成: {output_file}")
        log.info(f"  最终数据: {len(df)} 条，{len(df.columns)} 列")
        return True
    except Exception as e:
        log.error(f"保存文件失败: {e}")
        return False


def main():
    log.info("="*80)
    log.info("为训练数据补充市场环境特征")
    log.info("="*80)
    
    # 初始化 DataManager
    log.info("\n初始化 DataManager...")
    dm = DataManager()
    
    # 定义文件路径
    files_to_process = [
        {
            'input': PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv',
            'output': PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv',
            'name': '正样本'
        },
        {
            'input': PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv',
            'output': PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv',
            'name': '负样本'
        },
        {
            'input': PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v5.csv',
            'output': PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v5.csv',
            'name': '硬负样本'
        }
    ]
    
    # 处理每个文件
    results = []
    for file_info in files_to_process:
        success = process_file(file_info['input'], file_info['output'], dm)
        results.append({
            'name': file_info['name'],
            'success': success
        })
    
    # 总结
    log.info("\n" + "="*80)
    log.info("处理总结")
    log.info("="*80)
    
    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
        log.info(f"{result['name']}: {status}")
    
    success_count = sum(1 for r in results if r['success'])
    log.info(f"\n总计: {success_count}/{len(results)} 个文件处理成功")
    
    if success_count == len(results):
        log.success("\n✓ 所有文件处理完成！")
    else:
        log.warning("\n⚠️  部分文件处理失败，请检查日志")


if __name__ == '__main__':
    main()
