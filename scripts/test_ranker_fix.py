#!/usr/bin/env python3
"""快速测试 LGB Ranker 修复 —— 只跑 Fold 1"""
import sys
sys.path.insert(0, "/Users/javaadu/Documents/GitHub/aiquant")

import pandas as pd
import numpy as np
from scripts.train_v3xx_models import (
    load_training_data, time_series_cv_splits, flatten_multits,
    train_lgb_ranker, evaluate_probs,
)

df, feature_cols = load_training_data()
expected_days = sorted(df["days_to_t1"].dropna().unique())
cv_splits = time_series_cv_splits(df)

train_df, val_df, test_df = cv_splits[0]
train_flat = flatten_multits(train_df, feature_cols, expected_days)

print(f"Fold 1 原始训练集: {train_df['sample_id'].nunique()} 样本 / {len(train_df)} 行")
print(f"展平后: {len(train_flat)} 样本")
print(f"正样本: {train_flat['label'].sum():.0f} / 负样本: {(train_flat['label']==0).sum():.0f}")
print(f"trade_date 数量: {train_flat['trade_date'].nunique()}")
print(f"每组大小范围: {train_flat.groupby('trade_date').size().min()} ~ {train_flat.groupby('trade_date').size().max()}")

# 统计每组内标签分布
group_labels = train_flat.groupby("trade_date")["label"].agg(["min", "max", "nunique"])
mixed_groups = (group_labels["min"] == 0) & (group_labels["max"] == 1)
print(f"同时含正负样本的组数: {mixed_groups.sum()} / {len(group_labels)}")

# 训练 Ranker
rank_model, rank_flat_cols = train_lgb_ranker(train_df, val_df, feature_cols, expected_days)
test_flat = flatten_multits(test_df, feature_cols, expected_days)
rank_scores = rank_model.predict(test_flat[rank_flat_cols])
rank_probs = 1 / (1 + np.exp(-rank_scores))

metrics = evaluate_probs(test_flat["label"].values, rank_probs, "LGB-Ranker:  ")
print(f"\nFold 1 Ranker AUC: {metrics[0]:.4f}")
