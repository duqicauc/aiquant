#!/usr/bin/env python3
"""
多行时间序列展平工具

将 (sample_id, days_to_t1) 格式的多行数据展平为单行宽表。
用于 v3.x 系列模型的训练和预测。

Usage:
    from src.features.multits_flattener import flatten_multits
    df_flat = flatten_multits(df_multits, feature_cols)
"""
from typing import List, Optional

import pandas as pd

from src.utils.logger import log


def flatten_multits(
    df: pd.DataFrame,
    feature_cols: List[str],
    expected_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    """多行 -> 单行 (days_to_t1 作为 pivot)

    使用 set_index + unstack，零中间数据膨胀，内存效率高。
    trade_date 在 same sample 的不同 days_to_t1 行中不同，
    因此索引中只保留 sample_id + days_to_t1，metadata 通过 groupby 单独提取后 merge。

    Args:
        df: 多行 DataFrame，必须包含 sample_id, days_to_t1, label, ts_code, trade_date 列
        feature_cols: 需要展平的特征列名列表
        expected_days: 预期的 days_to_t1 值列表，默认从数据中推断

    Returns:
        展平后的宽表 DataFrame，每行一个 sample
    """
    if df.empty:
        log.warning("flatten_multits: 输入数据为空")
        return pd.DataFrame()

    if expected_days is None:
        expected_days = sorted(df["days_to_t1"].dropna().unique())

    n_samples = df["sample_id"].nunique()
    log.info(
        f"  展平: {len(df)} 行 -> {n_samples} 样本 × {len(feature_cols)} 特征 × {len(expected_days)} 天"
    )

    # 提取每个 sample 的 metadata
    # label 是固定的；trade_date 取 T1（days_to_t1 最大的那天，即 last）
    meta = (
        df.groupby("sample_id")
        .agg({"label": "first", "ts_code": "first", "trade_date": "last"})
        .reset_index()
    )

    # 仅对 sample_id + days_to_t1 做 set_index + unstack
    idx_cols = ["sample_id", "days_to_t1"]
    missing_idx = set(idx_cols) - set(df.columns)
    if missing_idx:
        raise ValueError(f"flatten_multits: 缺少必要列 {missing_idx}")

    df_indexed = df.set_index(idx_cols)[feature_cols]
    df_unstacked = df_indexed.unstack("days_to_t1")

    # 展平列名: (feat, day) -> feat_d{day}
    df_unstacked.columns = [
        f"{feat}_d{int(day)}" for feat, day in df_unstacked.columns
    ]

    # 合并 metadata
    result = meta.merge(df_unstacked.reset_index(), on="sample_id", how="left")

    log.info(f"  展平完成: {len(result)} 样本 × {len(result.columns) - 4} 维")
    return result
