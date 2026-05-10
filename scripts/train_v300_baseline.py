#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v300 基线模型训练 —— 方案A：精简回归路线

基于 v2.9.8 单 XGBoost，端到端重新训练，作为精简路线的验证基线。

核心设计：
1. 使用 v298 训练数据（v2.7.0 原始特征 + 34天多行增强）
2. 单 XGBoost（不集成）
3. Isotonic Regression 概率校准
4. 时间序列划分（65% train / 15% cal / 20% test）
5. 输出完整特征重要性，供后续降维参考

Usage:
    python scripts/train_v300_baseline.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
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

# ============================================================================
# 配置
# ============================================================================
SEED = 42
np.random.seed(SEED)

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "training" / "v298"
MODEL_FAMILY = "breakout_launch_scorer"
MODEL_VERSION = "v300-baseline"

# XGBoost 超参数（v2.7.0 优化参数，作为基线）
XGB_PARAMS = {
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
    "random_state": SEED,
    "tree_method": "hist",
}


# ============================================================================
# 数据加载
# ============================================================================
def load_training_data():
    """加载 v298 训练数据"""
    log.info("=" * 80)
    log.info("加载 v300 训练数据（来源: v298）")
    log.info("=" * 80)

    pos_file = DATA_DIR / "positive_features.csv"
    neg_file = DATA_DIR / "negative_features.csv"
    hard_neg_file = DATA_DIR / "hard_negative_features.csv"

    for f in [pos_file, neg_file, hard_neg_file]:
        if not f.exists():
            log.error(f"数据文件不存在: {f}")
            raise FileNotFoundError(f)

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    log.info(f"  正样本: {df_pos['sample_id'].nunique()} 个样本 / {len(df_pos)} 行")

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0
    log.info(f"  负样本: {df_neg['sample_id'].nunique()} 个样本 / {len(df_neg)} 行")

    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0
    log.info(f"  硬负样本: {df_hard_neg['sample_id'].nunique()} 个样本 / {len(df_hard_neg)} 行")

    # 共同特征
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
    common_cols = sorted(list(pos_cols & neg_cols & hard_cols))

    log.info(f"  共同特征: {len(common_cols)} 个")

    keep_cols = common_cols + ["label", "trade_date", "sample_id"]
    df = pd.concat(
        [
            df_pos[[c for c in keep_cols if c in df_pos.columns]],
            df_neg[[c for c in keep_cols if c in df_neg.columns]],
            df_hard_neg[[c for c in keep_cols if c in df_hard_neg.columns]],
        ],
        ignore_index=True,
    )

    total_samples = df["sample_id"].nunique()
    log.success(f"✓ 数据加载完成: {total_samples} 个样本 / {len(df)} 行")
    log.info(
        f"  样本构成: 正 {df_pos['sample_id'].nunique()} + 负 {df_neg['sample_id'].nunique()} + 硬负 {df_hard_neg['sample_id'].nunique()}"
    )

    return df, common_cols


# ============================================================================
# 特征选择
# ============================================================================
def get_feature_columns(df, feature_cols):
    """获取有效特征列"""
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
    ]

    # v298 中已标记为未使用的二值特征
    unused_binary = [
        "breakout_high_10d",
        "breakout_high_20d",
        "breakout_ma10",
        "breakout_ma55",
        "high_volume_breakout",
        "volume_price_match",
    ]
    exclude_cols.extend(unused_binary)

    valid_cols = [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    log.info(f"有效特征: {len(valid_cols)} 个（原始 {len(feature_cols)} 个）")
    return valid_cols


# ============================================================================
# 时间序列划分
# ============================================================================
def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
    """
    按 sample_id 的 T1 日期（最大 trade_date）统一划分
    确保同一样本的所有行进入同一 fold
    """
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    sample_t1 = df.groupby("sample_id")["trade_date"].max().reset_index()
    sample_t1.columns = ["sample_id", "t1_date"]
    sample_t1 = sample_t1.sort_values("t1_date")

    n_samples = len(sample_t1)
    train_end = int(n_samples * train_ratio)
    cal_end = int(n_samples * (train_ratio + cal_ratio))

    train_ids = set(sample_t1.iloc[:train_end]["sample_id"])
    cal_ids = set(sample_t1.iloc[train_end:cal_end]["sample_id"])
    test_ids = set(sample_t1.iloc[cal_end:]["sample_id"])

    train = df[df["sample_id"].isin(train_ids)].copy()
    cal = df[df["sample_id"].isin(cal_ids)].copy()
    test = df[df["sample_id"].isin(test_ids)].copy()

    log.info("\n时间序列划分:")
    log.info(f"  训练集: {len(train_ids)} 样本 / {len(train)} 行")
    log.info(f"  校准集: {len(cal_ids)} 样本 / {len(cal)} 行")
    log.info(f"  测试集: {len(test_ids)} 样本 / {len(test)} 行")

    return train, cal, test


# ============================================================================
# 模型训练
# ============================================================================
def train_model(X_train, y_train, X_val, y_val, feature_names):
    """训练 XGBoost"""
    log.info("\n训练 XGBoost 模型...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    log.success(f"✓ 训练完成, best_iteration: {booster.best_iteration}")
    return booster


# ============================================================================
# 概率校准
# ============================================================================
def calibrate_model(booster, X_cal, y_cal, feature_names):
    """Isotonic Regression 概率校准"""
    log.info("\n概率校准 (Isotonic Regression)...")

    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_probs = booster.predict(dcal)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    cal_probs = calibrator.predict(raw_probs)
    log.info(f"  校准前: mean={raw_probs.mean():.4f}, max={raw_probs.max():.4f}")
    log.info(f"  校准后: mean={cal_probs.mean():.4f}, max={cal_probs.max():.4f}")

    log.success("✓ 校准完成")
    return calibrator


# ============================================================================
# 评估
# ============================================================================
def evaluate(booster, calibrator, X_test, y_test, feature_names):
    """评估模型"""
    log.info("\n评估模型...")

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)

    # AUC
    auc_raw = roc_auc_score(y_test, raw_probs)
    auc_cal = roc_auc_score(y_test, cal_probs)
    log.info(f"  AUC (原始): {auc_raw:.4f}")
    log.info(f"  AUC (校准): {auc_cal:.4f}")

    # Brier Score
    brier = brier_score_loss(y_test, cal_probs)
    log.info(f"  Brier Score: {brier:.4f}")

    # 阈值分析
    log.info("\n不同阈值下的性能:")
    log.info(f"{'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    log.info("-" * 55)

    metrics_dict = {}
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        y_pred = (cal_probs >= thresh).astype(int)
        if y_pred.sum() > 0:
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            log.info(f"{thresh:<8.1f} {y_pred.sum():<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")
            if thresh == 0.5:
                metrics_dict = {"precision": p, "recall": r, "f1": f1}

    # 混淆矩阵
    y_pred_05 = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info("\n混淆矩阵（阈值=0.5）:")
    log.info("              预测负  预测正")
    log.info(f"  实际负      {cm[0,0]:<8} {cm[0,1]:<8}")
    log.info(f"  实际正      {cm[1,0]:<8} {cm[1,1]:<8}")

    return {
        "auc_raw": float(auc_raw),
        "auc_cal": float(auc_cal),
        "brier": float(brier),
        "precision": metrics_dict.get("precision"),
        "recall": metrics_dict.get("recall"),
        "f1": metrics_dict.get("f1"),
        "test_samples": len(X_test),
        "positive_samples": int(y_test.sum()),
    }


# ============================================================================
# 特征重要性
# ============================================================================
def extract_feature_importance(booster, feature_names):
    """提取并排序特征重要性"""
    importance = booster.get_score(importance_type="gain")
    total_gain = sum(importance.values())

    ranked = []
    for fname in feature_names:
        gain = importance.get(fname, 0)
        ranked.append(
            {
                "feature": fname,
                "gain": gain,
                "gain_pct": gain / total_gain * 100 if total_gain > 0 else 0,
            }
        )

    ranked = sorted(ranked, key=lambda x: x["gain"], reverse=True)
    return ranked


# ============================================================================
# 保存模型
# ============================================================================
def save_model(booster, calibrator, feature_names, metrics, importance_ranked):
    """保存模型和元数据"""
    model_dir = PROJECT_ROOT / "data" / "models" / MODEL_FAMILY / "versions" / MODEL_VERSION / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    booster.save_model(str(model_dir / "model.json"))

    # 特征名
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 校准器
    joblib.dump(calibrator, str(model_dir / "calibrator.pkl"))

    # 特征重要性
    with open(model_dir / "feature_importance.json", "w") as f:
        json.dump(importance_ranked[:50], f, indent=2, ensure_ascii=False)

    # 元数据
    metadata = {
        "version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "calibration_method": "isotonic_regression",
        "metrics": metrics,
        "top_10_features": [x["feature"] for x in importance_ranked[:10]],
        "description": "v300 baseline - 方案A精简回归路线，单XGBoost+Isotonic校准",
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"\n✓ 模型已保存到 {model_dir}")
    return model_dir


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("=" * 80)
    log.info(f"训练 {MODEL_VERSION} 基线模型 —— 方案A：精简回归路线")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols = load_training_data()

    # 2. 特征选择
    feature_names = get_feature_columns(df, feature_cols)
    log.info(f"\n最终特征数: {len(feature_names)}")

    # 3. 检查风险特征
    risk_features = [
        "max_drawdown_10d",
        "max_drawdown_20d",
        "max_drawdown_55d",
        "atr_14",
        "atr_ratio_14",
        "atr_expansion",
        "days_from_high_20d",
        "days_from_high_55d",
        "recovery_ratio_20d",
    ]
    available_risk = [f for f in risk_features if f in feature_names]
    log.info(f"风险特征: {len(available_risk)} / {len(risk_features)}")

    # 4. 时间序列划分
    log.info("\n" + "=" * 80)
    log.info("数据划分（时间序列方式）")
    log.info("=" * 80)

    train_df, cal_df, test_df = time_series_split(df)

    X_train = train_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["label"]
    X_cal = cal_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_cal = cal_df["label"]
    X_test = test_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df["label"]

    log.info(f"\n数据集: 训练 {len(X_train)} 行, 校准 {len(X_cal)} 行, 测试 {len(X_test)} 行")

    # 5. 训练模型
    booster = train_model(X_train, y_train, X_cal, y_cal, feature_names)

    # 6. 概率校准
    calibrator = calibrate_model(booster, X_cal, y_cal, feature_names)

    # 7. 评估
    metrics = evaluate(booster, calibrator, X_test, y_test, feature_names)

    # 8. 特征重要性
    importance_ranked = extract_feature_importance(booster, feature_names)
    log.info("\nTop 20 特征重要性 (by gain):")
    for i, item in enumerate(importance_ranked[:20], 1):
        log.info(f"  {i:2d}. {item['feature']:<30} {item['gain_pct']:>6.2f}%")

    # 9. 保存模型
    model_dir = save_model(booster, calibrator, feature_names, metrics, importance_ranked)

    # 10. 输出摘要
    log.info("\n" + "=" * 80)
    log.info("训练完成摘要")
    log.info("=" * 80)
    log.info(f"版本:        {MODEL_VERSION}")
    log.info(f"特征数:      {len(feature_names)}")
    log.info(f"训练样本:    {len(X_train)} 行")
    log.info(f"测试样本:    {metrics['test_samples']} 行")
    log.info(f"AUC (原始):  {metrics['auc_raw']:.4f}")
    log.info(f"AUC (校准):  {metrics['auc_cal']:.4f}")
    log.info(f"Brier:       {metrics['brier']:.4f}")
    log.info(f"Precision:   {metrics['precision']:.4f}")
    log.info(f"Recall:      {metrics['recall']:.4f}")
    log.info(f"F1:          {metrics['f1']:.4f}")
    log.info(f"模型路径:    {model_dir}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
