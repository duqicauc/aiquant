#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.7.1 Conservative Upgrade - 保守升级训练脚本

基于 v2.7.0-ensemble 成功经验，修复 v2.8.0/v2.9.1 的退化问题：
1. ✅ 恢复 breakout 核心特征（v2.8.0/v2.9.1 错误排除的6个特征）
2. ✅ 统一数据来源（全部使用 enhanced/ 目录）
3. ✅ 加回概率校准（IsotonicRegression）
4. ✅ 权重策略优化（差异小时固定三等分）
5. ✅ 时间序列划分改为按日期切分（避免数据泄露）
6. ✅ 详细的样本统计和 hard negative 比例监控

Usage:
    python scripts/train_v271_conservative.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log

# =============================================================================
# 配置常量
# =============================================================================
VERSION = "v2.7.1-conservative"
TARGET_HARD_RATIO = 0.18  # 硬负样本目标比例上限
MAX_HARD_RATIO = 0.20  # 硬负样本绝对上限
WEIGHT_DIFF_THRESHOLD = 0.02  # 权重差异阈值，小于此值使用固定权重


def load_training_data():
    """加载增强后的训练数据（统一来源）"""
    log.info("=" * 80)
    log.info("加载训练数据（统一使用 enhanced/ 目录）")
    log.info("=" * 80)

    enhanced_dir = PROJECT_ROOT / "data" / "training" / "enhanced"

    pos_file = enhanced_dir / "feature_data_34d_v5_enhanced.csv"
    neg_file = enhanced_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_file = enhanced_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"

    # 检查文件存在性
    for f in [pos_file, neg_file, hard_neg_file]:
        if not f.exists():
            log.error(f"数据文件不存在: {f}")
            raise FileNotFoundError(f)

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0

    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0

    # 统一日期格式
    for df in [df_pos, df_neg, df_hard_neg]:
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(
                lambda x: (
                    f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                )
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    # 获取共同特征
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

    pos_cols = set(df_pos.columns) - exclude_cols
    neg_cols = set(df_neg.columns) - exclude_cols
    hard_cols = set(df_hard_neg.columns) - exclude_cols
    common_cols = list(pos_cols & neg_cols & hard_cols)

    # 样本统计（按 sample_id 去重）
    n_pos = df_pos["sample_id"].nunique() if "sample_id" in df_pos.columns else len(df_pos)
    n_neg = df_neg["sample_id"].nunique() if "sample_id" in df_neg.columns else len(df_neg)
    n_hard = df_hard_neg["sample_id"].nunique() if "sample_id" in df_hard_neg.columns else len(df_hard_neg)
    total_neg = n_neg + n_hard
    hard_ratio = n_hard / total_neg if total_neg > 0 else 0

    log.info("\n样本统计:")
    log.info(f"  正样本:     {n_pos:>6,} 个 ({len(df_pos):>7,} 行)")
    log.info(f"  普通负样本: {n_neg:>6,} 个 ({len(df_neg):>7,} 行)")
    log.info(f"  硬负样本:   {n_hard:>6,} 个 ({len(df_hard_neg):>7,} 行)")
    log.info(f"  硬负比例:   {hard_ratio:.1%} (目标: ≤{TARGET_HARD_RATIO:.0%})")

    # 硬负比例告警
    if hard_ratio > MAX_HARD_RATIO:
        log.warning(f"⚠️ 硬负比例 {hard_ratio:.1%} 超过上限 {MAX_HARD_RATIO:.0%}！")
    elif hard_ratio > TARGET_HARD_RATIO:
        log.warning(f"⚠️ 硬负比例 {hard_ratio:.1%} 超过目标 {TARGET_HARD_RATIO:.0%}")
    else:
        log.success("✓ 硬负比例在目标范围内")

    # 合并数据
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
    """获取有效特征列 - v2.7.0 原版（只排除未来函数）"""
    exclude_cols = [
        "ts_code",
        "name",
        "t1_date",
        "t2_date",
        "sample_id",
        "label",
        "trade_date",
        # 未来函数（未来数据）
        "weekly_return_1",
        "weekly_return_2",
        "weekly_return_3",
        "total_return_34d",
        "weekly_volume_1",
        "weekly_volume_2",
        "weekly_volume_3",
        "days_to_t1",
        # 注意：v2.7.0 原版没有排除 breakout 特征！
        # v2.8.0/v2.9.1 错误地排除了以下特征，我们在 v2.7.1 中恢复：
        # "breakout_high_10d", "breakout_high_20d", "breakout_ma10",
        # "breakout_ma55", "high_volume_breakout", "volume_price_match",
    ]

    valid_cols = [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    # 记录 breakout 特征是否被保留
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
    """严格按日期的时间序列划分（同一天的所有样本在同一集合）"""
    df = df.copy()

    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    cal_end = int(n_dates * (train_ratio + cal_ratio))

    train_dates = set(unique_dates[:train_end])
    cal_dates = set(unique_dates[train_end:cal_end])
    test_dates = set(unique_dates[cal_end:])

    train = df[df["trade_date"].dt.date.isin(train_dates)].copy()
    cal = df[df["trade_date"].dt.date.isin(cal_dates)].copy()
    test = df[df["trade_date"].dt.date.isin(test_dates)].copy()

    log.info("\n时间序列划分:")
    log.info(f"  训练集: {len(train_dates)} 天 ({unique_dates[0]} ~ {unique_dates[train_end-1]}), {len(train)} 行")
    log.info(f"  校准集: {len(cal_dates)} 天 ({unique_dates[train_end]} ~ {unique_dates[cal_end-1]}), {len(cal)} 行")
    log.info(f"  测试集: {len(test_dates)} 天 ({unique_dates[cal_end]} ~ {unique_dates[-1]}), {len(test)} 行")

    # 统计各集合的正负样本比例
    for name, subset in [("训练集", train), ("校准集", cal), ("测试集", test)]:
        pos_ratio = subset["label"].mean()
        log.info(f"  {name}正样本比例: {pos_ratio:.2%}")

    return train, cal, test


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """训练 XGBoost"""
    log.info("\n训练 XGBoost...")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "scale_pos_weight": 1.5,
        "random_state": 42,
        "tree_method": "hist",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  ✓ XGBoost 验证集 AUC: {auc:.4f} (best_iteration: {model.best_iteration})")

    return model, auc


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """训练 LightGBM"""
    log.info("\n训练 LightGBM...")

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": 1.5,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  ✓ LightGBM 验证集 AUC: {auc:.4f} (best_iteration: {model.best_iteration})")

    return model, auc


def train_catboost(X_train, y_train, X_val, y_val, feature_names):
    """训练 CatBoost"""
    log.info("\n训练 CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
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


def calibrate_model(ensemble_pred, y_cal):
    """概率校准（Isotonic Regression）"""
    log.info("\n概率校准（Isotonic Regression）...")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ensemble_pred, y_cal)

    cal_pred = calibrator.predict(ensemble_pred)
    log.info(f"  校准前: mean={ensemble_pred.mean():.4f}, std={ensemble_pred.std():.4f}")
    log.info(f"  校准后: mean={cal_pred.mean():.4f}, std={cal_pred.std():.4f}")
    log.success("  ✓ 概率校准完成")

    return calibrator


def compute_weights(xgb_auc, lgb_auc, cat_auc):
    """计算集成权重（差异小时固定三等分）"""
    aucs = {"xgboost": xgb_auc, "lightgbm": lgb_auc, "catboost": cat_auc}
    max_diff = max(aucs.values()) - min(aucs.values())

    if max_diff < WEIGHT_DIFF_THRESHOLD:
        log.info(f"\n模型AUC差异 {max_diff:.4f} < 阈值 {WEIGHT_DIFF_THRESHOLD}，使用固定三等分权重")
        weights = {"xgboost": 1 / 3, "lightgbm": 1 / 3, "catboost": 1 / 3}
    else:
        log.info(f"\n模型AUC差异 {max_diff:.4f} ≥ 阈值 {WEIGHT_DIFF_THRESHOLD}，使用AUC动态权重")
        total = sum(aucs.values())
        weights = {k: v / total for k, v in aucs.items()}

    log.info(f"  XGBoost:  {weights['xgboost']:.4f} (AUC={xgb_auc:.4f})")
    log.info(f"  LightGBM: {weights['lightgbm']:.4f} (AUC={lgb_auc:.4f})")
    log.info(f"  CatBoost: {weights['catboost']:.4f} (AUC={cat_auc:.4f})")

    return weights


def ensemble_predict(models, weights, X, feature_names, calibrator=None):
    """集成预测（可选校准）"""
    xgb_model, lgb_model, cat_model = models

    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    xgb_pred = xgb_model.predict(dmatrix)
    lgb_pred = lgb_model.predict(X)
    cat_pred = cat_model.predict_proba(X)[:, 1]

    ensemble_pred = weights["xgboost"] * xgb_pred + weights["lightgbm"] * lgb_pred + weights["catboost"] * cat_pred

    if calibrator is not None:
        ensemble_pred = calibrator.predict(ensemble_pred)

    return ensemble_pred, xgb_pred, lgb_pred, cat_pred


def evaluate_model(y_true, y_pred, model_name="模型"):
    """评估模型指标"""
    auc = roc_auc_score(y_true, y_pred)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    cm = confusion_matrix(y_true, y_pred_bin)

    log.info(f"\n{model_name} 测试集性能:")
    log.info(f"  AUC:       {auc:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall:    {recall:.4f}")
    log.info(f"  F1:        {f1:.4f}")
    log.info(f"  混淆矩阵:  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def save_model(models, weights, feature_names, calibrator, metrics, hard_ratio):
    """保存模型"""
    log.info(f"\n保存模型 {VERSION}...")

    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model").mkdir(exist_ok=True)

    xgb_model, lgb_model, cat_model = models

    # 保存各模型
    xgb_model.save_model(str(model_dir / "model" / "xgboost.json"))
    lgb_model.save_model(str(model_dir / "model" / "lightgbm.txt"))
    cat_model.save_model(str(model_dir / "model" / "catboost.cbm"))

    # 保存特征名
    with open(model_dir / "model" / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 保存权重
    with open(model_dir / "model" / "weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    # 保存校准器
    import joblib

    joblib.dump(calibrator, str(model_dir / "model" / "calibrator.pkl"))

    # 保存指标
    with open(model_dir / "model" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 元数据
    metadata = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "models": ["xgboost", "lightgbm", "catboost"],
        "weights": weights,
        "calibration_method": "isotonic_regression",
        "hard_negative_ratio": hard_ratio,
        "description": ("v2.7.1 保守升级 - 恢复breakout特征+统一数据来源+概率校准+权重优化"),
        "improvements_over_v270": [
            "恢复被 v2.8.0/v2.9.1 错误排除的6个 breakout 核心特征",
            "统一数据来源（全部使用 enhanced/ 目录）",
            "加回 IsotonicRegression 概率校准",
            "权重策略优化（差异小时固定三等分）",
            "时间序列划分改为按日期切分",
        ],
        "metrics": metrics,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info(f"训练 {VERSION}（保守升级）")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols, hard_ratio = load_training_data()

    # 2. 获取有效特征
    feature_names = get_feature_columns(df, feature_cols)

    # 3. 时间序列划分
    log.info("\n" + "=" * 80)
    log.info("时间序列划分")
    log.info("=" * 80)
    train_df, cal_df, test_df = time_series_split(df)

    X_train = train_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["label"]
    X_cal = cal_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_cal = cal_df["label"]
    X_test = test_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df["label"]

    # 4. 训练各模型
    log.info("\n" + "=" * 80)
    log.info("训练各基模型")
    log.info("=" * 80)

    xgb_model, xgb_auc = train_xgboost(X_train.values, y_train.values, X_cal.values, y_cal.values, feature_names)
    lgb_model, lgb_auc = train_lightgbm(X_train.values, y_train.values, X_cal.values, y_cal.values, feature_names)
    cat_model, cat_auc = train_catboost(X_train.values, y_train.values, X_cal.values, y_cal.values, feature_names)

    # 5. 计算权重
    log.info("\n" + "=" * 80)
    log.info("集成权重计算")
    log.info("=" * 80)
    weights = compute_weights(xgb_auc, lgb_auc, cat_auc)

    # 6. 在校准集上做集成预测并校准
    log.info("\n" + "=" * 80)
    log.info("集成预测与校准")
    log.info("=" * 80)

    models = (xgb_model, lgb_model, cat_model)
    cal_pred_raw, _, _, _ = ensemble_predict(models, weights, X_cal.values, feature_names)
    calibrator = calibrate_model(cal_pred_raw, y_cal.values)

    # 7. 在测试集上评估
    log.info("\n" + "=" * 80)
    log.info("测试集评估")
    log.info("=" * 80)

    # 各单模型评估
    dtest = xgb.DMatrix(X_test.values, feature_names=feature_names)
    xgb_test_pred = xgb_model.predict(dtest)
    lgb_test_pred = lgb_model.predict(X_test.values)
    cat_test_pred = cat_model.predict_proba(X_test.values)[:, 1]

    log.info(f"\n{'模型':<15} {'AUC':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    log.info("-" * 60)

    single_metrics = {}
    for name, pred in [("XGBoost", xgb_test_pred), ("LightGBM", lgb_test_pred), ("CatBoost", cat_test_pred)]:
        m = evaluate_model(y_test.values, pred, name)
        single_metrics[name] = m
        log.info(f"{name:<15} {m['auc']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # 集成模型评估（未校准）
    ensemble_pred_raw, _, _, _ = ensemble_predict(models, weights, X_test.values, feature_names)
    metrics_raw = evaluate_model(y_test.values, ensemble_pred_raw, "Ensemble (未校准)")
    log.info(
        f"{'Ensemble(原)':<15} {metrics_raw['auc']:>10.4f} {metrics_raw['precision']:>12.4f} {metrics_raw['recall']:>10.4f} {metrics_raw['f1']:>10.4f}"
    )

    # 集成模型评估（校准后）
    ensemble_pred_cal, _, _, _ = ensemble_predict(models, weights, X_test.values, feature_names, calibrator)
    metrics_cal = evaluate_model(y_test.values, ensemble_pred_cal, "Ensemble (校准后)")
    log.info(
        f"{'Ensemble(校)':<15} {metrics_cal['auc']:>10.4f} {metrics_cal['precision']:>12.4f} {metrics_cal['recall']:>10.4f} {metrics_cal['f1']:>10.4f}"
    )

    # 8. 保存模型（使用校准后的指标）
    save_model(models, weights, feature_names, calibrator, metrics_cal, hard_ratio)

    # 9. 与 v2.7.0-ensemble 对比
    log.info("\n" + "=" * 80)
    log.info("与 v2.7.0-ensemble 对比")
    log.info("=" * 80)

    v270_metrics = {"auc": 0.9818, "precision": 0.8670, "recall": 0.8903, "f1": 0.8785}

    log.info(f"\n{'指标':<12} {'v2.7.0':>10} {'v2.7.1':>10} {'变化':>10} {'状态':>8}")
    log.info("-" * 55)
    for key in ["auc", "precision", "recall", "f1"]:
        v270 = v270_metrics[key]
        v271 = metrics_cal[key]
        diff = (v271 - v270) * 100
        sign = "+" if diff >= 0 else ""
        status = "✅ 提升" if diff >= 0 else "❌ 下降"
        log.info(f"{key:<12} {v270:>10.4f} {v271:>10.4f} {sign}{diff:>9.2f}% {status:>8}")

    log.success(f"\n✓ {VERSION} 训练完成!")

    # 返回关键指标用于外部判断
    return metrics_cal


if __name__ == "__main__":
    main()
