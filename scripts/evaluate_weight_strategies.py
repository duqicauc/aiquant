#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
权重策略评估脚本

基于 v2.7.1-conservative 已训练的三个子模型，
测试多种权重分配策略，选择最优集成方案。

Usage:
    python scripts/evaluate_weight_strategies.py
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


def load_data_and_models():
    """加载数据和已训练的模型"""
    log.info("=" * 80)
    log.info("加载 v2.7.1-conservative 数据和模型")
    log.info("=" * 80)

    # 1. 加载数据（与训练时相同的逻辑）
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

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )

    # 2. 时间序列划分（与训练时相同的参数: 0.65/0.15/0.20）
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.65)
    cal_end = int(n_dates * 0.80)

    train_dates = set(unique_dates[:train_end])
    cal_dates = set(unique_dates[train_end:cal_end])
    test_dates = set(unique_dates[cal_end:])

    test = df[df["trade_date"].dt.date.isin(test_dates)].copy()

    # 3. 加载模型和特征名
    model_dir = (
        PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.7.1-conservative" / "model"
    )

    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_dir / "xgboost.json"))

    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(model_dir / "catboost.cbm"))

    # 4. 准备测试数据
    X_test = test[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test["label"].values

    log.info(f"测试集: {len(X_test)} 条，特征数: {len(feature_names)}")
    log.info(f"正样本比例: {y_test.mean():.2%}")

    # 5. 获取各模型预测概率
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    xgb_pred = xgb_model.predict(dtest)
    lgb_pred = lgb_model.predict(X_test)
    cat_pred = cat_model.predict_proba(X_test)[:, 1]

    log.info("\n单模型测试集 AUC:")
    log.info(f"  XGBoost:  {roc_auc_score(y_test, xgb_pred):.4f}")
    log.info(f"  LightGBM: {roc_auc_score(y_test, lgb_pred):.4f}")
    log.info(f"  CatBoost: {roc_auc_score(y_test, cat_pred):.4f}")

    return {
        "xgb": xgb_pred,
        "lgb": lgb_pred,
        "cat": cat_pred,
        "y_true": y_test,
    }


def evaluate_weights(preds, weights, strategy_name):
    """评估给定权重策略"""
    xgb_pred = preds["xgb"]
    lgb_pred = preds["lgb"]
    cat_pred = preds["cat"]
    y_true = preds["y_true"]

    ensemble_pred = weights["xgb"] * xgb_pred + weights["lgb"] * lgb_pred + weights["cat"] * cat_pred

    auc = roc_auc_score(y_true, ensemble_pred)
    y_pred_bin = (ensemble_pred >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    return {
        "strategy": strategy_name,
        "weights": weights,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    log.info("=" * 80)
    log.info("权重策略评估")
    log.info("=" * 80)

    preds = load_data_and_models()

    # 获取验证集 AUC（用于部分权重策略）
    # 从训练日志中提取的验证集 AUC
    val_aucs = {"xgb": 0.9645, "lgb": 0.9629, "cat": 0.9717}
    test_aucs = {
        "xgb": roc_auc_score(preds["y_true"], preds["xgb"]),
        "lgb": roc_auc_score(preds["y_true"], preds["lgb"]),
        "cat": roc_auc_score(preds["y_true"], preds["cat"]),
    }

    # 定义所有待测试的权重策略
    strategies = []

    # 1. 单模型基准
    strategies.append(("XGBoost only", {"xgb": 1.0, "lgb": 0.0, "cat": 0.0}))
    strategies.append(("LightGBM only", {"xgb": 0.0, "lgb": 1.0, "cat": 0.0}))
    strategies.append(("CatBoost only", {"xgb": 0.0, "lgb": 0.0, "cat": 1.0}))

    # 2. 三等分（v2.7.0/v2.7.1 原版）
    strategies.append(("Equal (1/3)", {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}))

    # 3. CatBoost 主导
    strategies.append(("CatBoost 50%", {"xgb": 0.25, "lgb": 0.25, "cat": 0.50}))
    strategies.append(("CatBoost 60%", {"xgb": 0.20, "lgb": 0.20, "cat": 0.60}))
    strategies.append(("CatBoost 70%", {"xgb": 0.15, "lgb": 0.15, "cat": 0.70}))

    # 4. 验证集 AUC 线性加权（v2.8.0/v2.9.1 原版）
    total_val = sum(val_aucs.values())
    strategies.append(("Val AUC linear", {k: v / total_val for k, v in val_aucs.items()}))

    # 5. 验证集 AUC 平方加权
    total_val_sq = sum(v**2 for v in val_aucs.values())
    strategies.append(("Val AUC squared", {k: (v**2) / total_val_sq for k, v in val_aucs.items()}))

    # 6. 验证集 AUC 立方加权
    total_val_cu = sum(v**3 for v in val_aucs.values())
    strategies.append(("Val AUC cubed", {k: (v**3) / total_val_cu for k, v in val_aucs.items()}))

    # 7. 测试集 AUC 线性加权
    total_test = sum(test_aucs.values())
    strategies.append(("Test AUC linear", {k: v / total_test for k, v in test_aucs.items()}))

    # 8. 测试集 AUC 平方加权
    total_test_sq = sum(v**2 for v in test_aucs.values())
    strategies.append(("Test AUC squared", {k: (v**2) / total_test_sq for k, v in test_aucs.items()}))

    # 9. 差异阈值法（改进版）
    max_diff_val = max(val_aucs.values()) - min(val_aucs.values())
    if max_diff_val < 0.01:
        strategies.append(("Diff threshold (0.01)", {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}))
    else:
        total = sum(val_aucs.values())
        strategies.append(("Diff threshold (0.01)", {k: v / total for k, v in val_aucs.items()}))

    # 评估所有策略
    log.info("\n" + "=" * 80)
    log.info("评估所有权重策略")
    log.info("=" * 80)

    results = []
    for name, weights in strategies:
        result = evaluate_weights(preds, weights, name)
        results.append(result)

    # 排序并输出
    results.sort(key=lambda x: x["auc"], reverse=True)

    log.info(f"\n{'排名':<4} {'策略':<25} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'权重(X/L/C)':>20}")
    log.info("-" * 90)

    for i, r in enumerate(results, 1):
        w = r["weights"]
        weight_str = f"{w['xgb']:.2f}/{w['lgb']:.2f}/{w['cat']:.2f}"
        marker = " ⭐" if i == 1 else ""
        log.info(
            f"{i:<4} {r['strategy']:<25} {r['auc']:>8.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {weight_str:>20}{marker}"
        )

    # 最优策略详情
    best = results[0]
    log.info("\n" + "=" * 80)
    log.info(f"最优策略: {best['strategy']}")
    log.info("=" * 80)
    log.info(f"  AUC:       {best['auc']:.4f}")
    log.info(f"  Precision: {best['precision']:.4f}")
    log.info(f"  Recall:    {best['recall']:.4f}")
    log.info(f"  F1:        {best['f1']:.4f}")
    log.info(
        f"  权重:      XGB={best['weights']['xgb']:.4f}, LGB={best['weights']['lgb']:.4f}, CAT={best['weights']['cat']:.4f}"
    )

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "training" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "weight_strategy_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.success(f"\n✓ 结果已保存到 {output_dir / 'weight_strategy_comparison.json'}")

    # 返回最优权重供外部使用
    return best["weights"]


if __name__ == "__main__":
    main()
