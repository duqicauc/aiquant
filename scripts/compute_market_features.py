#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场环境特征计算脚本
计算上证/沪深300的趋势、波动率、成交量等市场状态特征
供训练时作为全局特征加入

输出：data/training/features/market_features.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tushare_data_provider import TushareDataProvider
from src.utils.logger import log

OUTPUT = "data/training/features/market_features.csv"
START_DATE = "19990101"
END_DATE = "20260421"

def get_index_data(pro, ts_code, start, end):
    """拉取指数日线数据"""
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date'] = df['trade_date'].astype(str)
    return df

def compute_market_features(df):
    """基于指数数据计算市场环境特征"""
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['vol'] = df['vol'].astype(float)
    df['amount'] = df['amount'].astype(float)

    # 价格特征
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # 趋势得分 (0-1)
    df['trend_ma5'] = (df['close'] > df['ma5']).astype(float)
    df['trend_ma10'] = (df['close'] > df['ma10']).astype(float)
    df['trend_ma20'] = (df['close'] > df['ma20']).astype(float)
    df['trend_ma60'] = (df['close'] > df['ma60']).astype(float)
    df['trend_ma20_direction'] = (df['ma20'] > df['ma20'].shift(5)).astype(float)  # MA20是否向上

    # 综合趋势得分 (0-5)
    df['sh_trend_score'] = (df['trend_ma5'] + df['trend_ma10'] + df['trend_ma20'] +
                            df['trend_ma60'] + df['trend_ma20_direction'])

    # 波动率特征
    df['returns'] = df['close'].pct_change()
    df['volatility_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)   # 5日年化波动率
    df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252) # 20日年化波动率
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d'].replace(0, np.nan)  # 短期/长期波动比

    # 成交量特征
    df['vol_ma20'] = df['vol'].rolling(20).mean()
    df['volume_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)  # 量比
    df['amount_ma20'] = df['amount'].rolling(20).mean()
    df['amount_ratio'] = df['amount'] / df['amount_ma20'].replace(0, np.nan)

    # 涨跌特征
    df['days_up_5d'] = (df['returns'] > 0).rolling(5).sum()  # 近5日上涨天数
    df['days_up_20d'] = (df['returns'] > 0).rolling(20).sum()  # 近20日上涨天数
    df['max_drawdown_20d'] = (df['close'] / df['close'].rolling(20).max() - 1) * 100  # 20日最大回撤

    # 选择输出列
    feature_cols = [
        'trade_date',
        'close',
        'sh_trend_score',
        'trend_ma5', 'trend_ma10', 'trend_ma20', 'trend_ma60', 'trend_ma20_direction',
        'volatility_5d', 'volatility_20d', 'volatility_ratio',
        'volume_ratio', 'amount_ratio',
        'days_up_5d', 'days_up_20d',
        'max_drawdown_20d',
    ]

    return df[feature_cols].copy()

def main():
    log.info("=" * 80)
    log.info("市场环境特征计算")
    log.info("=" * 80)

    pro = TushareDataProvider().pro

    # 拉取上证指数
    log.info("拉取上证指数数据...")
    sh_df = get_index_data(pro, '000001.SH', START_DATE, END_DATE)
    log.info(f"上证指数记录: {len(sh_df)} 条")

    # 拉取沪深300
    log.info("拉取沪深300数据...")
    hs300_df = get_index_data(pro, '000300.SH', START_DATE, END_DATE)
    log.info(f"沪深300记录: {len(hs300_df)} 条")

    # 计算特征
    log.info("计算市场环境特征...")
    sh_features = compute_market_features(sh_df)

    # 重命名列，加上 sh_ 前缀
    rename_map = {c: f'sh_{c}' for c in sh_features.columns if c != 'trade_date'}
    sh_features = sh_features.rename(columns=rename_map)

    # 如果有沪深300数据，合并
    if not hs300_df.empty:
        hs300_features = compute_market_features(hs300_df)
        rename_map300 = {c: f'hs300_{c}' for c in hs300_features.columns if c != 'trade_date'}
        hs300_features = hs300_features.rename(columns=rename_map300)
        sh_features = sh_features.merge(hs300_features, on='trade_date', how='left')

    # 保存
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    sh_features.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

    log.info(f"\n市场环境特征已保存: {OUTPUT}")
    log.info(f"记录数: {len(sh_features)}")
    log.info(f"特征列: {list(sh_features.columns)}")

    # 统计
    log.info("\n市场环境统计:")
    log.info(f"  趋势得分范围: {sh_features['sh_sh_trend_score'].min():.1f} ~ {sh_features['sh_sh_trend_score'].max():.1f}")
    log.info(f"  平均波动率(20d): {sh_features['sh_volatility_20d'].mean():.2f}")
    log.info(f"  强牛天数(trend_score=5): {(sh_features['sh_sh_trend_score'] == 5).sum()}")
    log.info(f"  强熊天数(trend_score=0): {(sh_features['sh_sh_trend_score'] == 0).sum()}")

if __name__ == '__main__':
    main()
