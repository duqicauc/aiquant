#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将新计算的市场环境特征合并到现有训练数据中

输入：
- data/training/processed/feature_data_34d_v6.csv (正样本)
- data/training/features/negative_feature_data_v2_34d_v6.csv (负样本)
- data/training/features/hard_negative_feature_data_34d_v6.csv (硬负样本)
- data/training/features/market_features.csv (新市场环境特征)

输出（覆盖原文件，备份旧版本）：
- feature_data_34d_v6.csv (+29个市场环境特征)
- negative_feature_data_v2_34d_v6.csv (+29个市场环境特征)
- hard_negative_feature_data_34d_v6.csv (+29个市场环境特征)
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import log

MARKET_FEATURES = "data/training/features/market_features.csv"
FILES = {
    "positive": "data/training/processed/feature_data_34d_v6.csv",
    "negative": "data/training/features/negative_feature_data_v2_34d_v6.csv",
    "hard_negative": "data/training/features/hard_negative_feature_data_34d_v6.csv",
}

# 要合并的市场环境特征列（排除 trade_date）
MARKET_COLS = [
    'sh_sh_trend_score', 'sh_trend_ma5', 'sh_trend_ma10', 'sh_trend_ma20',
    'sh_trend_ma60', 'sh_trend_ma20_direction', 'sh_volatility_5d',
    'sh_volatility_20d', 'sh_volatility_ratio', 'sh_volume_ratio',
    'sh_amount_ratio', 'sh_days_up_5d', 'sh_days_up_20d', 'sh_max_drawdown_20d',
    'hs300_sh_trend_score', 'hs300_trend_ma5', 'hs300_trend_ma10',
    'hs300_trend_ma20', 'hs300_trend_ma60', 'hs300_trend_ma20_direction',
    'hs300_volatility_5d', 'hs300_volatility_20d', 'hs300_volatility_ratio',
    'hs300_volume_ratio', 'hs300_amount_ratio', 'hs300_days_up_5d',
    'hs300_days_up_20d', 'hs300_max_drawdown_20d',
]

def backup_file(path: Path):
    """备份原文件（复制，不移动）"""
    import shutil
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_path)
        log.info(f"  备份: {backup_path.name}")
        return backup_path
    return None

def merge_market_features(df_samples: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """按 trade_date 合并市场环境特征"""
    # 统一 trade_date 格式
    df_samples = df_samples.copy()
    df_market = df_market.copy()

    # 样本中的 trade_date 可能是 '2015-08-18' 或 '20150818'
    if pd.api.types.is_string_dtype(df_samples['trade_date']):
        # 已经是 YYYY-MM-DD 格式
        df_samples['trade_date_key'] = pd.to_datetime(df_samples['trade_date']).dt.strftime('%Y%m%d').astype(int)
    else:
        df_samples['trade_date_key'] = df_samples['trade_date'].astype(int)

    # market_features 中的 trade_date 是 int (20200102)
    df_market['trade_date_key'] = df_market['trade_date'].astype(int)

    # 选择需要的列
    merge_cols = ['trade_date_key'] + [c for c in MARKET_COLS if c in df_market.columns]
    df_market_merge = df_market[merge_cols].copy()

    # 合并
    df_merged = df_samples.merge(df_market_merge, on='trade_date_key', how='left')
    df_merged = df_merged.drop(columns=['trade_date_key'])

    # 填充缺失值（对于市场特征数据范围之外的日期）
    for col in MARKET_COLS:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    return df_merged

def main():
    log.info("=" * 80)
    log.info("市场环境特征合并")
    log.info("=" * 80)

    # 读取市场环境特征
    log.info(f"读取市场环境特征: {MARKET_FEATURES}")
    df_market = pd.read_csv(MARKET_FEATURES)
    log.info(f"  市场环境特征: {len(df_market)} 行, {len(df_market.columns)} 列")
    log.info(f"  日期范围: {df_market['trade_date'].min()} ~ {df_market['trade_date'].max()}")

    for sample_type, file_path in FILES.items():
        path = Path(file_path)
        if not path.exists():
            log.warning(f"文件不存在，跳过: {file_path}")
            continue

        log.info(f"\n处理 {sample_type}:")
        log.info(f"  文件: {file_path}")

        # 备份
        backup_file(path)

        # 读取样本
        df = pd.read_csv(file_path)
        original_cols = len(df.columns)
        log.info(f"  原始列数: {original_cols}")

        # 合并市场环境特征
        df_merged = merge_market_features(df, df_market)
        new_cols = len(df_merged.columns)
        log.info(f"  合并后列数: {new_cols} (+{new_cols - original_cols})")

        # 保存（覆盖原文件）
        df_merged.to_csv(file_path, index=False, encoding='utf-8-sig')
        log.info(f"  已保存: {file_path}")

    log.info("\n" + "=" * 80)
    log.info("✅ 市场环境特征合并完成!")
    log.info("=" * 80)

if __name__ == '__main__':
    main()
