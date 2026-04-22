#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征分析与增强脚本

目标：
1. 分析特征重要性，识别低效特征
2. 添加新的高效特征
3. 输出优化后的特征集
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log


def load_model_and_features(version="v2.6.0"):
    """加载模型和特征名"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    return booster, feature_names


def analyze_feature_importance(booster, feature_names):
    """分析特征重要性"""
    importance_dict = booster.get_score(importance_type="gain")

    importance_list = []
    for i, feat in enumerate(feature_names):
        importance_list.append({"feature": feat, "importance": importance_dict.get(f"f{i}", 0)})

    df = pd.DataFrame(importance_list)
    df = df.sort_values("importance", ascending=False)

    total = df["importance"].sum()
    if total > 0:
        df["percentage"] = df["importance"] / total * 100
        df["cumulative"] = df["percentage"].cumsum()
    else:
        df["percentage"] = 0
        df["cumulative"] = 0

    return df


def identify_low_importance_features(df_importance, threshold=0.5):
    """识别低重要性特征"""
    low_features = df_importance[df_importance["percentage"] < threshold]["feature"].tolist()
    return low_features


def add_enhanced_features(df):
    """添加增强特征"""
    df = df.copy()
    n = len(df)

    log.info("添加增强特征...")
    added_features = []

    # 1. 换手率异常检测
    if "turnover_rate" in df.columns:
        tr = df["turnover_rate"]
        tr_mean = tr.rolling(20, min_periods=5).mean()
        tr_std = tr.rolling(20, min_periods=5).std()
        df["turnover_zscore"] = (tr - tr_mean) / (tr_std + 1e-8)
        added_features.append("turnover_zscore")

        # 换手率变化率
        df["turnover_change_rate"] = tr.pct_change(5)
        added_features.append("turnover_change_rate")

        # 换手率突增
        df["turnover_spike"] = (tr > tr_mean * 2).astype(int)
        added_features.append("turnover_spike")

    # 2. RSI-KDJ综合指标增强
    if "rsi_6" in df.columns and "kdj_j" in df.columns and "kdj_k" in df.columns:
        # RSI-KDJ金叉确认
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        added_features.append("rsi_kdj_golden_cross")

        # RSI-KDJ综合强度
        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        added_features.append("rsi_kdj_strength")

        # RSI超买超卖区间
        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        added_features.append("rsi_zone")

    # 3. 量价背离强度
    if "close" in df.columns and "vol" in df.columns:
        price_change_10d = df["close"].pct_change(10)
        vol_change_10d = df["vol"].pct_change(10)
        df["volume_price_divergence_strength"] = np.abs(price_change_10d - vol_change_10d)
        added_features.append("volume_price_divergence_strength")

        # 量价同向确认
        df["volume_price_confirm"] = ((price_change_10d > 0) == (vol_change_10d > 0)).astype(int)
        added_features.append("volume_price_confirm")

    # 4. 突破强度综合指标
    breakout_cols = [c for c in df.columns if "breakout_strength" in c]
    if len(breakout_cols) >= 2:
        df["breakout_strength_avg"] = df[breakout_cols].mean(axis=1)
        added_features.append("breakout_strength_avg")

        df["breakout_strength_max"] = df[breakout_cols].max(axis=1)
        added_features.append("breakout_strength_max")

    # 5. 均线多头排列强度
    ma_cols = ["ma5", "ma10", "ma_20d", "ma_34d", "ma_55d"]
    available_ma = [c for c in ma_cols if c in df.columns]
    if len(available_ma) >= 3:
        # 计算均线多头排列程度
        ma_values = df[available_ma].values
        ma_rank_score = np.zeros(len(df))
        for i in range(len(df)):
            row = ma_values[i]
            if not np.isnan(row).any():
                # 计算排序一致性
                sorted_idx = np.argsort(row)[::-1]  # 从大到小
                expected = np.arange(len(row))
                ma_rank_score[i] = 1 - np.abs(sorted_idx - expected).sum() / (len(row) * (len(row) - 1) / 2 + 1e-8)
        df["ma_alignment_score"] = ma_rank_score
        added_features.append("ma_alignment_score")

    # 6. 动量加速度
    if "momentum_10d" in df.columns:
        df["momentum_acceleration"] = df["momentum_10d"].diff(5)
        added_features.append("momentum_acceleration")

    # 7. 价格位置综合指标
    position_cols = [c for c in df.columns if "price_position" in c]
    if len(position_cols) >= 2:
        df["price_position_avg"] = df[position_cols].mean(axis=1)
        added_features.append("price_position_avg")

    # 8. 风险调整收益
    if "return_34d" in df.columns and "volatility_34d" in df.columns:
        df["sharpe_like_34d"] = df["return_34d"] / (df["volatility_34d"] + 1e-8)
        added_features.append("sharpe_like_34d")

    log.info(f"  添加了 {len(added_features)} 个新特征: {added_features}")

    return df, added_features


def process_training_data():
    """处理训练数据，添加增强特征"""
    log.info("=" * 80)
    log.info("特征分析与增强")
    log.info("=" * 80)

    # 1. 分析当前模型特征重要性
    log.info("\n1. 分析v2.6.0模型特征重要性...")
    booster, feature_names = load_model_and_features("v2.6.0")
    df_importance = analyze_feature_importance(booster, feature_names)

    # 显示Top 20
    log.info("\nTop 20 特征:")
    for i, row in df_importance.head(20).iterrows():
        log.info(f"  {row['feature']:40s} {row['percentage']:6.2f}%")

    # 2. 识别低重要性特征
    low_features = identify_low_importance_features(df_importance, threshold=0.3)
    log.info(f"\n2. 低重要性特征（<0.3%）: {len(low_features)} 个")

    # 3. 加载训练数据
    log.info("\n3. 加载训练数据...")
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    df_hard_neg = pd.read_csv(hard_neg_file)

    log.info(f"  正样本: {len(df_pos)} 行")
    log.info(f"  负样本: {len(df_neg)} 行")
    log.info(f"  硬负样本: {len(df_hard_neg)} 行")

    # 4. 添加增强特征
    log.info("\n4. 添加增强特征...")
    df_pos_enhanced, added_features = add_enhanced_features(df_pos)
    df_neg_enhanced, _ = add_enhanced_features(df_neg)
    df_hard_neg_enhanced, _ = add_enhanced_features(df_hard_neg)

    # 5. 保存增强后的数据
    output_dir = PROJECT_ROOT / "data" / "training" / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_output = output_dir / "feature_data_34d_v5_enhanced.csv"
    neg_output = output_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_output = output_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"

    df_pos_enhanced.to_csv(pos_output, index=False)
    df_neg_enhanced.to_csv(neg_output, index=False)
    df_hard_neg_enhanced.to_csv(hard_neg_output, index=False)

    log.success("\n5. 增强数据已保存:")
    log.info(f"  正样本: {pos_output}")
    log.info(f"  负样本: {neg_output}")
    log.info(f"  硬负样本: {hard_neg_output}")

    # 6. 保存特征分析结果
    analysis_output = output_dir / "feature_analysis_results.json"
    with open(analysis_output, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "low_importance_features": low_features,
                "added_features": added_features,
                "feature_importance": df_importance.to_dict("records"),
            },
            f,
            indent=2,
        )

    log.info(f"  分析结果: {analysis_output}")

    return added_features, low_features


if __name__ == "__main__":
    process_training_data()
