#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.2 CatBoost 单模型训练脚本

基于 v2.7.1-conservative 数据流程，核心改进：
1. ✅ 恢复 breakout 核心特征（v2.8.0/v2.9.1 错误排除的 6 个特征）
2. ✅ 统一数据来源（全部 enhanced/）
3. ✅ 概率校准（IsotonicRegression）
4. ✅ 只用 CatBoost（当前数据上最强模型）
5. ✅ 优化 CatBoost 超参（depth=8, iterations=800）

Usage:
    python scripts/train_v292_catboost.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import _SigmoidCalibration as PlattScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log

VERSION = "v2.9.2-catboost"


def load_training_data():
    """加载数据（统一 enhanced/ 目录）"""
    log.info("=" * 80)
    log.info("加载训练数据（统一 enhanced/）")
    log.info("=" * 80)

    enhanced_dir = PROJECT_ROOT / "data" / "training" / "enhanced"
    pos_file = enhanced_dir / "feature_data_34d_v5_enhanced.csv"
    neg_file = enhanced_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_file = enhanced_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0
    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0

    for df in [df_pos, df_neg, df_hard_neg]:
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(
                lambda x: (
                    f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                )
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    exclude_cols = {
        "label",
        "sample_id",
        "ts_code",
        "name",
        "t1_date",
        "t2_date",
        "trade_date",
        "list_date",
        "pattern_type",
        "days_to_t1",
    }
    common_cols = list(
        (set(df_pos.columns) - exclude_cols)
        & (set(df_neg.columns) - exclude_cols)
        & (set(df_hard_neg.columns) - exclude_cols)
    )

    n_pos = df_pos["sample_id"].nunique() if "sample_id" in df_pos.columns else len(df_pos)
    n_neg = df_neg["sample_id"].nunique() if "sample_id" in df_neg.columns else len(df_neg)
    n_hard = df_hard_neg["sample_id"].nunique() if "sample_id" in df_hard_neg.columns else len(df_hard_neg)
    hard_ratio = n_hard / (n_neg + n_hard) if (n_neg + n_hard) > 0 else 0

    log.info("\n样本统计:")
    log.info(f"  正样本:     {n_pos:>6,} 个 ({len(df_pos):>7,} 行)")
    log.info(f"  普通负样本: {n_neg:>6,} 个 ({len(df_neg):>7,} 行)")
    log.info(f"  硬负样本:   {n_hard:>6,} 个 ({len(df_hard_neg):>7,} 行)")
    log.info(f"  硬负比例:   {hard_ratio:.1%}")

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )

    log.info(f"\n数据加载完成: {len(df)} 条，特征数: {len(common_cols)}")
    return df, common_cols, hard_ratio


def get_feature_columns(df, feature_cols):
    """获取有效特征列 - 恢复 breakout 特征"""
    exclude_cols = [
        "ts_code",
        "name",
        "t1_date",
        "t2_date",
        "sample_id",
        "label",
        "trade_date",
        "weekly_return_1",
        "weekly_return_2",
        "weekly_return_3",
        "total_return_34d",
        "weekly_volume_1",
        "weekly_volume_2",
        "weekly_volume_3",
        "days_to_t1",
        # 注意: v2.9.2 恢复 breakout 特征（对比 v2.8.0/v2.9.1 的错误排除）
    ]

    valid_cols = [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    breakout_cols = [c for c in valid_cols if "breakout" in c or c in ["high_volume_breakout", "volume_price_match"]]
    log.info("\n特征筛选:")
    log.info(f"  原始特征数: {len(feature_cols)}")
    log.info(f"  有效特征数: {len(valid_cols)}")
    log.info(f"  保留的 breakout 相关特征: {len(breakout_cols)} 个")
    for c in breakout_cols[:10]:
        log.info(f"    - {c}")
    if len(breakout_cols) > 10:
        log.info(f"    ... 等共 {len(breakout_cols)} 个")

    return valid_cols


def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
    """按日期切分"""
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    cal_end = int(n_dates * (train_ratio + cal_ratio))

    train = df[df["trade_date"].dt.date.isin(set(unique_dates[:train_end]))].copy()
    cal = df[df["trade_date"].dt.date.isin(set(unique_dates[train_end:cal_end]))].copy()
    test = df[df["trade_date"].dt.date.isin(set(unique_dates[cal_end:]))].copy()

    log.info("\n时间序列划分:")
    log.info(f"  训练集: {len(train)} 行 ({len(set(unique_dates[:train_end]))} 天)")
    log.info(f"  校准集: {len(cal)} 行 ({len(set(unique_dates[train_end:cal_end]))} 天)")
    log.info(f"  测试集: {len(test)} 行 ({len(set(unique_dates[cal_end:]))} 天)")
    for name, subset in [("训练集", train), ("校准集", cal), ("测试集", test)]:
        log.info(f"  {name}正样本比例: {subset['label'].mean():.2%}")

    return train, cal, test


def train_catboost(X_train, y_train, X_val, y_val):
    """训练 CatBoost - 优化超参"""
    log.info("\n训练 CatBoost（优化超参）...")

    model = CatBoostClassifier(
        iterations=800,  # v2.7.1: 500 → 800
        depth=8,  # v2.7.1: 6 → 8（适度加深）
        learning_rate=0.08,  # v2.7.1: 0.1 → 0.08（配合更多轮次）
        l2_leaf_reg=1.0,  # v2.7.1: 3.0 → 1.0（减少正则化，数据量大不易过拟合）
        border_count=128,
        scale_pos_weight=1.5,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  ✓ CatBoost 验证集 AUC: {auc:.4f}")

    return model, auc


def calibrate_model(raw_pred, y_cal):
    """概率校准（Platt Scaling）"""
    log.info("\n概率校准（Platt Scaling / Sigmoid Calibration）...")
    calibrator = PlattScaler()
    calibrator.fit(raw_pred, y_cal)
    cal_pred = calibrator.predict(raw_pred)
    log.info(f"  Platt 参数: a={calibrator.a_:.4f}, b={calibrator.b_:.4f}")
    log.info(f"  校准前: mean={raw_pred.mean():.4f}, std={raw_pred.std():.4f}")
    log.info(f"  校准后: mean={cal_pred.mean():.4f}, std={cal_pred.std():.4f}")
    log.success("  ✓ 概率校准完成")
    return calibrator


def evaluate(y_true, y_pred, label="模型"):
    auc = roc_auc_score(y_true, y_pred)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    log.info(f"\n{label} 测试集性能:")
    log.info(f"  AUC:       {auc:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall:    {recall:.4f}")
    log.info(f"  F1:        {f1:.4f}")

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def save_model(model, feature_names, calibrator, metrics, hard_ratio):
    log.info(f"\n保存模型 {VERSION}...")
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model").mkdir(exist_ok=True)

    model.save_model(str(model_dir / "model" / "catboost.cbm"))

    with open(model_dir / "model" / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    import joblib

    joblib.dump(calibrator, str(model_dir / "model" / "calibrator.pkl"))

    with open(model_dir / "model" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "model": "catboost",
        "calibration_method": "platt_scaling",
        "hard_negative_ratio": hard_ratio,
        "description": "v2.9.2 CatBoost 单模型 - 恢复breakout特征+优化超参+概率校准",
        "improvements": [
            "恢复被 v2.8.0/v2.9.1 错误排除的 6 个 breakout 核心特征",
            "统一数据来源（全部 enhanced/）",
            "CatBoost 超参优化（depth=8, iterations=800, lr=0.08, l2=1.0）",
            "Platt Scaling (Sigmoid Calibration) 概率校准",
        ],
        "metrics": metrics,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info(f"训练 {VERSION}（CatBoost 单模型）")
    log.info("=" * 80)

    df, feature_cols, hard_ratio = load_training_data()
    feature_names = get_feature_columns(df, feature_cols)

    log.info("\n" + "=" * 80)
    log.info("时间序列划分")
    log.info("=" * 80)
    train_df, cal_df, test_df = time_series_split(df)

    X_train = train_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df["label"].values
    X_cal = cal_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_cal = cal_df["label"].values
    X_test = test_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test_df["label"].values

    log.info("\n" + "=" * 80)
    log.info("训练 CatBoost")
    log.info("=" * 80)
    model, val_auc = train_catboost(X_train, y_train, X_cal, y_cal)

    log.info("\n" + "=" * 80)
    log.info("概率校准")
    log.info("=" * 80)
    cal_pred_raw = model.predict_proba(X_cal)[:, 1]
    calibrator = calibrate_model(cal_pred_raw, y_cal)

    log.info("\n" + "=" * 80)
    log.info("测试集评估")
    log.info("=" * 80)

    test_pred_raw = model.predict_proba(X_test)[:, 1]
    metrics_raw = evaluate(y_test, test_pred_raw, "CatBoost (未校准)")

    test_pred_cal = calibrator.predict(test_pred_raw)
    metrics_cal = evaluate(y_test, test_pred_cal, "CatBoost (校准后)")

    save_model(model, feature_names, calibrator, metrics_cal, hard_ratio)

    # 对比
    log.info("\n" + "=" * 80)
    log.info("与历史版本对比")
    log.info("=" * 80)

    comparisons = {
        "v2.7.0-ensemble": {"auc": 0.9818, "precision": 0.8670, "recall": 0.8903, "f1": 0.8785},
        "v2.7.1-conservative": {"auc": 0.9554, "precision": 0.8021, "recall": 0.8139, "f1": 0.8079},
        "v2.9.1-ensemble": {"auc": 0.9660, "precision": 0.7891, "recall": 0.8486, "f1": 0.8178},
    }

    log.info(f"\n{'版本':<20} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    log.info("-" * 60)
    for name, m in comparisons.items():
        log.info(f"{name:<20} {m['auc']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    log.info(
        f"{'v2.9.2-catboost(当前)':<20} {metrics_cal['auc']:>8.4f} {metrics_cal['precision']:>10.4f} {metrics_cal['recall']:>8.4f} {metrics_cal['f1']:>8.4f}"
    )

    log.success(f"\n✓ {VERSION} 训练完成!")
    return metrics_cal


if __name__ == "__main__":
    main()
