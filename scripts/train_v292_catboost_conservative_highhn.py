#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.2-catboost-conservative-highhn 高硬负比例版

在保守版基础上，将硬负比例从 11.6% 提升到 18%，对标 v291 配置。

改进点：
- 合并 v291 的硬负样本 (data/training/features/hard_negative_feature_data_34d_v5.csv)
- 控制硬负比例在 18%（超过则随机下采样）
- 其余超参与保守版一致
"""

import json
import sys
import warnings
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

VERSION = "v2.9.2-catboost-conservative-highhn"


def load_training_data():
    """加载数据（enhanced/ + v291 硬负样本）"""
    log.info("=" * 80)
    log.info("加载训练数据（enhanced/ + v291 硬负样本）")
    log.info("=" * 80)

    enhanced_dir = PROJECT_ROOT / "data" / "training" / "enhanced"
    pos_file = enhanced_dir / "feature_data_34d_v5_enhanced.csv"
    neg_file = enhanced_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_file = enhanced_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"
    hard_neg_v291_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0
    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0

    # 加载 v291 的硬负样本
    df_hard_neg_v291 = pd.read_csv(hard_neg_v291_file)
    df_hard_neg_v291["label"] = 0

    for df in [df_pos, df_neg, df_hard_neg, df_hard_neg_v291]:
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(
                lambda x: (
                    f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                )
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    # 合并硬负样本
    df_hard_neg_all = pd.concat([df_hard_neg, df_hard_neg_v291], ignore_index=True)

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
        & (set(df_hard_neg_all.columns) - exclude_cols)
    )

    n_pos = df_pos["sample_id"].nunique() if "sample_id" in df_pos.columns else len(df_pos)
    n_neg = df_neg["sample_id"].nunique() if "sample_id" in df_neg.columns else len(df_neg)
    n_hard_all = (
        df_hard_neg_all["sample_id"].nunique() if "sample_id" in df_hard_neg_all.columns else len(df_hard_neg_all)
    )

    # 控制硬负比例不超过 18%
    target_hard_ratio = 0.18
    current_hard_count = df_hard_neg_all["sample_id"].nunique()
    current_neg_count = df_neg["sample_id"].nunique()
    max_hard = int(current_neg_count * target_hard_ratio / (1 - target_hard_ratio))

    if current_hard_count > max_hard:
        log.info(f"硬负样本过多: {current_hard_count}，下采样至 {max_hard} (目标占比 {target_hard_ratio*100:.0f}%)")
        keep_ids = df_hard_neg_all["sample_id"].drop_duplicates().sample(n=max_hard, random_state=42).tolist()
        df_hard_neg_all = df_hard_neg_all[df_hard_neg_all["sample_id"].isin(keep_ids)].copy()
        n_hard_all = max_hard

    hard_ratio = n_hard_all / (n_neg + n_hard_all) if (n_neg + n_hard_all) > 0 else 0

    log.info("\n样本统计:")
    log.info(f"  正样本:     {n_pos:>6,} 个 ({len(df_pos):>7,} 行)")
    log.info(f"  普通负样本: {n_neg:>6,} 个 ({len(df_neg):>7,} 行)")
    log.info(f"  硬负样本:   {n_hard_all:>6,} 个 ({len(df_hard_neg_all):>7,} 行)")
    log.info(f"  硬负比例:   {hard_ratio:.1%}")

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg_all[common_cols + ["label", "trade_date"]],
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
        "list_date",
        "pattern_type",
        "days_to_t1",
        "weekly_return_1",
        "weekly_return_2",
        "weekly_return_3",
        "total_return_34d",
        "weekly_volume_1",
        "weekly_volume_2",
        "weekly_volume_3",
        "breakout_high_10d",
        "breakout_high_20d",
        "breakout_ma10",
        "breakout_ma55",
        "high_volume_breakout",
        "volume_price_match",
    ]

    valid_cols = [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    return valid_cols


def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
    """严格按日期的时间序列划分"""
    df = df.copy()
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    cal_end = int(n_dates * (train_ratio + cal_ratio))

    train_dates = unique_dates[:train_end]
    cal_dates = unique_dates[train_end:cal_end]
    test_dates = unique_dates[cal_end:]

    df_train = df[df["trade_date"].dt.date.isin(train_dates)]
    df_cal = df[df["trade_date"].dt.date.isin(cal_dates)]
    df_test = df[df["trade_date"].dt.date.isin(test_dates)]

    log.info("\n时间序列划分:")
    log.info(f"  训练集: {len(train_dates)} 天, {len(df_train):,} 条")
    log.info(f"  校准集: {len(cal_dates)} 天, {len(df_cal):,} 条")
    log.info(f"  测试集: {len(test_dates)} 天, {len(df_test):,} 条")

    return df_train, df_cal, df_test


def train_catboost(X_train, y_train, X_val, y_val):
    """训练 CatBoost（保守超参）"""
    log.info("\n训练 CatBoost（保守超参）...")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=50,
        early_stopping_rounds=50,
        loss_function="Logloss",
        eval_metric="AUC",
        use_best_model=True,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=50,
    )

    log.success(f"CatBoost 训练完成: best_iteration={model.get_best_iteration()}")
    return model


def calibrate_model(model, X_val, y_val):
    """Platt Scaling 概率校准"""
    log.info("\n概率校准 (Platt Scaling)...")
    val_pred = model.predict_proba(X_val)[:, 1]

    calibrator = PlattScaler()
    calibrator.fit(val_pred, y_val)

    log.info(f"  Platt 参数: a={calibrator.a_:.4f}, b={calibrator.b_:.4f}")
    return calibrator


def evaluate(model, calibrator, X, y, name=""):
    """评估模型性能"""
    raw_pred = model.predict_proba(X)[:, 1]
    cal_pred = calibrator.predict(raw_pred)

    auc_raw = roc_auc_score(y, raw_pred)
    auc_cal = roc_auc_score(y, cal_pred)

    y_pred = (cal_pred > 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    log.info(f"\nCatBoost ({name}) 性能:")
    log.info(f"  AUC:       {auc_raw:.4f} (raw) / {auc_cal:.4f} (cal)")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall:    {recall:.4f}")
    log.info(f"  F1:        {f1:.4f}")

    return {"auc": auc_cal, "precision": precision, "recall": recall, "f1": f1}


def save_model(model, calibrator, feature_names, metrics, hard_ratio):
    """保存模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / "catboost.cbm"))

    import joblib

    joblib.dump(calibrator, str(model_dir / "calibrator.pkl"))

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    metadata = {
        "version": VERSION,
        "created_at": pd.Timestamp.now().isoformat(),
        "features_count": len(feature_names),
        "model": "catboost",
        "calibration_method": "platt_scaling",
        "hard_negative_ratio": float(hard_ratio),
        "description": f"{VERSION} - 保守超参+18%硬负比例",
        "improvements": [
            "保守 CatBoost 超参（depth=6, iterations=500, lr=0.05, l2=5.0）",
            "Platt Scaling (Sigmoid Calibration) 概率校准",
            "硬负比例提升到 18%（合并 v291 硬负样本）",
        ],
        "metrics": metrics,
    }
    with open(
        PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION / "metadata.json", "w"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info(f"训练 {VERSION}（CatBoost 单模型，18% 硬负比例）")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols, hard_ratio = load_training_data()

    # 2. 获取有效特征
    valid_features = get_feature_columns(df, feature_cols)
    log.info(f"有效特征: {len(valid_features)} 个")

    # 3. 时间序列划分
    df_train, df_cal, df_test = time_series_split(df)

    X_train = df_train[valid_features].astype(float).fillna(0)
    y_train = df_train["label"]
    X_cal = df_cal[valid_features].astype(float).fillna(0)
    y_cal = df_cal["label"]
    X_test = df_test[valid_features].astype(float).fillna(0)
    y_test = df_test["label"]

    # 4. 训练
    model = train_catboost(X_train, y_train, X_cal, y_cal)

    # 5. 概率校准
    calibrator = calibrate_model(model, X_cal, y_cal)

    # 6. 评估
    metrics = evaluate(model, calibrator, X_test, y_test, "测试集")

    # 7. 保存
    save_model(model, calibrator, valid_features, metrics, hard_ratio)

    log.info("\n" + "=" * 80)
    log.success(f"{VERSION} 训练完成!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
