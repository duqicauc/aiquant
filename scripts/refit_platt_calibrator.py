#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 v2.9.2-catboost 重新拟合 Platt Scaling 校准器

Usage:
    python scripts/refit_platt_calibrator.py
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration as PlattScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log

MODEL_VERSION = "v2.9.2-catboost"


def load_training_data():
    """加载数据（统一 enhanced/ 目录）"""
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
                lambda x: f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    exclude_cols = {"label", "sample_id", "ts_code", "name", "t1_date", "t2_date", "trade_date", "list_date", "pattern_type", "days_to_t1"}
    common_cols = list(
        (set(df_pos.columns) - exclude_cols) &
        (set(df_neg.columns) - exclude_cols) &
        (set(df_hard_neg.columns) - exclude_cols)
    )

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )

    log.info(f"数据加载完成: {len(df)} 条，特征数: {len(common_cols)}")
    return df, common_cols


def get_feature_columns(df, feature_cols):
    exclude_cols = [
        "ts_code", "name", "t1_date", "t2_date", "sample_id", "label", "trade_date",
        "weekly_return_1", "weekly_return_2", "weekly_return_3",
        "total_return_34d",
        "weekly_volume_1", "weekly_volume_2", "weekly_volume_3",
        "days_to_t1",
    ]

    valid_cols = [
        c for c in feature_cols
        if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    return valid_cols


def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    cal_end = int(n_dates * (train_ratio + cal_ratio))

    train = df[df["trade_date"].dt.date.isin(set(unique_dates[:train_end]))].copy()
    cal = df[df["trade_date"].dt.date.isin(set(unique_dates[train_end:cal_end]))].copy()
    test = df[df["trade_date"].dt.date.isin(set(unique_dates[cal_end:]))].copy()

    log.info(f"训练集: {len(train)} 行, 校准集: {len(cal)} 行, 测试集: {len(test)} 行")
    for name, subset in [("训练集", train), ("校准集", cal), ("测试集", test)]:
        log.info(f"  {name}正样本比例: {subset['label'].mean():.2%}")

    return train, cal, test


def main():
    log.info("=" * 80)
    log.info("重新拟合 Platt Scaling 校准器")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols = load_training_data()
    feature_names = get_feature_columns(df, feature_cols)

    # 2. 划分
    train_df, cal_df, test_df = time_series_split(df)

    # 3. 加载已训练的 CatBoost 模型和特征名（必须使用保存的顺序！）
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / MODEL_VERSION / "model"
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "catboost.cbm"))
    log.info("✓ 加载 CatBoost 模型")

    with open(model_dir / "feature_names.json", "r") as f:
        saved_feature_names = json.load(f)
    log.info(f"✓ 加载保存的特征名: {len(saved_feature_names)} 个")

    # 使用保存的特征顺序（关键！CatBoost 对特征顺序敏感）
    feature_names = saved_feature_names

    # 4. 在校准集上预测原始概率
    X_cal = cal_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_cal = cal_df["label"].values

    cal_raw_pred = model.predict_proba(X_cal)[:, 1]
    log.info(f"校准集原始概率: mean={cal_raw_pred.mean():.4f}, std={cal_raw_pred.std():.4f}, max={cal_raw_pred.max():.4f}")

    # 5. 拟合 Platt Scaling
    log.info("\n拟合 Platt Scaling (Sigmoid Calibration)...")
    platt = PlattScaler()
    platt.fit(cal_raw_pred, y_cal)
    log.info(f"Platt 参数: a={platt.a_:.4f}, b={platt.b_:.4f}")

    # 6. 对比 IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_raw_pred, y_cal)

    # 7. 在测试集上对比
    X_test = test_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test_df["label"].values
    test_raw_pred = model.predict_proba(X_test)[:, 1]

    test_platt = platt.predict(test_raw_pred)
    test_iso = iso.predict(test_raw_pred)

    from sklearn.metrics import roc_auc_score, brier_score_loss

    log.info("\n测试集对比:")
    log.info(f"  未校准 AUC:      {roc_auc_score(y_test, test_raw_pred):.4f}")
    log.info(f"  Platt AUC:       {roc_auc_score(y_test, test_platt):.4f}")
    log.info(f"  Isotonic AUC:    {roc_auc_score(y_test, test_iso):.4f}")
    log.info(f"  未校准 Brier:    {brier_score_loss(y_test, test_raw_pred):.4f}")
    log.info(f"  Platt Brier:     {brier_score_loss(y_test, test_platt):.4f}")
    log.info(f"  Isotonic Brier:  {brier_score_loss(y_test, test_iso):.4f}")

    # 检查 Platt 是否避免了极端压缩
    high_segment = test_raw_pred >= 0.99
    if high_segment.any():
        log.info(f"\n高分段 (raw_prob >= 0.99) 压缩对比:")
        log.info(f"  原始概率范围:     {test_raw_pred[high_segment].min():.4f} ~ {test_raw_pred[high_segment].max():.4f}")
        log.info(f"  Platt 输出范围:   {test_platt[high_segment].min():.4f} ~ {test_platt[high_segment].max():.4f}")
        log.info(f"  Isotonic 输出范围: {test_iso[high_segment].min():.4f} ~ {test_iso[high_segment].max():.4f}")

    # 8. 保存 Platt 校准器
    import joblib
    joblib.dump(platt, str(model_dir / "calibrator_platt.pkl"))
    log.success(f"✓ Platt Scaling 校准器已保存到 {model_dir / 'calibrator_platt.pkl'}")

    # 9. 备份旧校准器，替换为新校准器
    old_calib = model_dir / "calibrator.pkl"
    if old_calib.exists():
        old_calib.rename(model_dir / "calibrator_isotonic_backup.pkl")
        log.info("✓ 旧 IsotonicRegression 校准器已备份")

    joblib.dump(platt, str(model_dir / "calibrator.pkl"))
    log.success("✓ Platt Scaling 已成为默认校准器")

    log.success("\n完成！请重新运行预测脚本查看效果。")


if __name__ == "__main__":
    main()
