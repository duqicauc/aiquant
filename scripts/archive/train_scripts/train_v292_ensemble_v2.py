#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.2-ensemble-v2: 用 v292 数据复刻 v291 Ensemble 架构

改进点：
- 使用 v292 的训练数据（enhanced/ + v291 硬负样本，18% 硬负比例）
- 复刻 v291 的 Ensemble 架构（XGB + LGB + CatBoost，权重各33%）
- 使用 IsotonicRegression 概率校准（如 v291）
"""

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")
from src.utils.logger import log

VERSION = "v2.9.2-ensemble-v2"


def load_training_data():
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

    target_hard_ratio = 0.18
    current_hard_count = df_hard_neg_all["sample_id"].nunique()
    current_neg_count = df_neg["sample_id"].nunique()
    max_hard = int(current_neg_count * target_hard_ratio / (1 - target_hard_ratio))

    if current_hard_count > max_hard:
        keep_ids = df_hard_neg_all["sample_id"].drop_duplicates().sample(n=max_hard, random_state=42).tolist()
        df_hard_neg_all = df_hard_neg_all[df_hard_neg_all["sample_id"].isin(keep_ids)].copy()
        n_hard_all = max_hard

    hard_ratio = n_hard_all / (n_neg + n_hard_all) if (n_neg + n_hard_all) > 0 else 0

    log.info(f"正样本: {n_pos:,}, 普通负: {n_neg:,}, 硬负: {n_hard_all:,}, 比例: {hard_ratio:.1%}")

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg_all[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )
    return df, common_cols, hard_ratio


def get_feature_columns(df, feature_cols):
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
    return [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]


def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
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

    log.info(
        f"训练集: {len(train_dates)}天/{len(df_train):,}条, 校准集: {len(cal_dates)}天/{len(df_cal):,}条, 测试集: {len(test_dates)}天/{len(df_test):,}条"
    )
    return df_train, df_cal, df_test


def train_xgb(X_train, y_train, X_val, y_val):
    log.info("训练 XGBoost...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log.success(f"XGBoost 训练完成: best_iteration={model.best_iteration}")
    return model


def train_lgb(X_train, y_train, X_val, y_val):
    log.info("训练 LightGBM...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    log.success(f"LightGBM 训练完成: best_iteration={model.best_iteration_}")
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    log.info("训练 CatBoost...")
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=50,
        early_stopping_rounds=30,
        loss_function="Logloss",
        eval_metric="AUC",
        use_best_model=True,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    log.success(f"CatBoost 训练完成: best_iteration={model.get_best_iteration()}")
    return model


def calibrate_models(models, X_cal, y_cal):
    log.info("概率校准...")
    calibrators = {}
    for name, model in models.items():
        pred = model.predict_proba(X_cal)[:, 1]
        # 使用 IsotonicRegression（如 v291）
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(pred, y_cal)
        calibrators[name] = cal
        log.info(f"  {name}: 校准完成")
    return calibrators


def ensemble_predict(models, calibrators, weights, X):
    preds = []
    for name, model in models.items():
        raw_pred = model.predict_proba(X)[:, 1]
        cal_pred = calibrators[name].predict(raw_pred)
        preds.append(cal_pred * weights[name])
    return np.sum(preds, axis=0)


def evaluate(y_true, y_pred, name=""):
    auc = roc_auc_score(y_true, y_pred)
    y_bin = (y_pred > 0.5).astype(int)
    p = precision_score(y_true, y_bin, zero_division=0)
    r = recall_score(y_true, y_bin, zero_division=0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    log.info(f"{name}: AUC={auc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    return {"auc": auc, "precision": p, "recall": r, "f1": f1}


def save_models(models, calibrators, weights, feature_names, metrics):
    version_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION
    model_dir = version_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    models["xgboost"]._Booster.save_model(str(model_dir / "xgboost.ubj"))
    models["lightgbm"].booster_.save_model(str(model_dir / "lightgbm.txt"))
    models["catboost"].save_model(str(model_dir / "catboost.cbm"))

    import joblib

    for name, cal in calibrators.items():
        joblib.dump(cal, str(model_dir / f"calibrator_{name}.pkl"))

    with open(model_dir / "weights.json", "w") as f:
        json.dump(weights, f)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    metadata = {
        "version": VERSION,
        "created_at": pd.Timestamp.now().isoformat(),
        "features_count": len(feature_names),
        "model": "ensemble",
        "calibration_method": "isotonic_regression",
        "metrics": metrics,
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info(f"训练 {VERSION}（Ensemble: XGB+LGB+CatBoost）")
    log.info("=" * 80)

    df, feature_cols, hard_ratio = load_training_data()
    valid_features = get_feature_columns(df, feature_cols)
    log.info(f"有效特征: {len(valid_features)} 个")

    df_train, df_cal, df_test = time_series_split(df)

    X_train = df_train[valid_features].astype(float).fillna(0)
    y_train = df_train["label"]
    X_cal = df_cal[valid_features].astype(float).fillna(0)
    y_cal = df_cal["label"]
    X_test = df_test[valid_features].astype(float).fillna(0)
    y_test = df_test["label"]

    # 训练三个子模型
    models = {
        "xgboost": train_xgb(X_train, y_train, X_cal, y_cal),
        "lightgbm": train_lgb(X_train, y_train, X_cal, y_cal),
        "catboost": train_catboost(X_train, y_train, X_cal, y_cal),
    }

    # 概率校准
    calibrators = calibrate_models(models, X_cal, y_cal)

    # 评估子模型
    log.info("\n子模型测试集性能:")
    for name, model in models.items():
        pred = model.predict_proba(X_test)[:, 1]
        cal_pred = calibrators[name].predict(pred)
        evaluate(y_test, cal_pred, name)

    # Ensemble 评估
    log.info("\nEnsemble 测试集性能:")
    weights = {"xgboost": 1 / 3, "lightgbm": 1 / 3, "catboost": 1 / 3}
    ens_pred = ensemble_predict(models, calibrators, weights, X_test)
    metrics = evaluate(y_test, ens_pred, "ensemble")

    # 保存
    save_models(models, calibrators, weights, valid_features, metrics)

    log.info("\n" + "=" * 80)
    log.success(f"{VERSION} 训练完成!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
