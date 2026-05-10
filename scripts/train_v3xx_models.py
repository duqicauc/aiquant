#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3.x 系列模型训练脚本 —— 多行时间序列 + 展平法 + 排序模型

设计原则:
1. 数据一致性: 复用 v298 多行数据(34天×173维), 严格无泄露
2. 可复现性: 固定随机种子, 完整记录配置
3. 多模型对比: 单模型(XGB/LGB/Cat/LR) + 排序模型(LGB-Ranker) + 集成
4. CV安全: 市值中性化/下采样在fold内进行, 时间序列划分

Usage:
    python scripts/train_v3xx_models.py

Output:
    data/models/breakout_launch_scorer/versions/v3.0.0-comparison/
"""

import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log

# ============================================================================
# 配置
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data" / "training" / "v298"
MODEL_BASE_DIR = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions"
MODEL_VERSION = "v3.0.0-comparison"
SEED = 42

# 负样本下采样比例 (负:正 = 2.5:1, 硬负:正 = 0.5:1)
NEG_SAMPLING_RATIO = 2.5
HARD_NEG_SAMPLING_RATIO = 0.5

# 模型参数
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 1.5,
    "seed": SEED,
    "nthread": -1,
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "scale_pos_weight": 1.5,
    "seed": SEED,
    "verbose": -1,
}

CAT_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "verbose": False,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
}

LGB_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20, 50],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "seed": SEED,
    "verbose": -1,
}


# ============================================================================
# 数据加载
# ============================================================================
def load_training_data() -> Tuple[pd.DataFrame, List[str]]:
    """加载 v298 多行训练数据"""
    log.info("=" * 80)
    log.info("加载 v3.x 训练数据 (来源: v298 多行时间序列)")
    log.info("=" * 80)

    files = {
        "positive": ("positive_features.csv", 1),
        "negative": ("negative_features.csv", 0),
        "hard_negative": ("hard_negative_features.csv", 0),
    }

    dfs = []
    global_offset = 0
    for name, (fname, label) in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        df = pd.read_csv(path)
        df["label"] = label
        # 重新生成全局唯一 sample_id，避免不同类型间冲突
        n_samples = df["sample_id"].nunique()
        id_map = {old: global_offset + i for i, old in enumerate(sorted(df["sample_id"].unique()))}
        df["sample_id"] = df["sample_id"].map(id_map)
        global_offset += n_samples
        dfs.append(df)
        log.info(f"  {name}: {n_samples} 样本 / {len(df)} 行")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])

    # 共同特征列
    exclude = {"label", "sample_id", "ts_code", "name", "trade_date", "days_to_t1"}
    common_cols = sorted([c for c in df_all.columns if c not in exclude])

    log.success(f"数据加载完成: {df_all['sample_id'].nunique()} 样本 / {len(df_all)} 行, {len(common_cols)} 特征")
    return df_all, common_cols


# ============================================================================
# 时间序列 CV 划分
# ============================================================================
def time_series_cv_splits(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple]:
    """
    时间序列交叉验证划分
    按 sample_id 的 T1 日期(trade_date 最大值)排序,确保同一样本进入同一 fold
    """
    sample_t1 = (
        df.groupby("sample_id")
        .agg(
            {
                "trade_date": "max",
                "label": "first",
            }
        )
        .reset_index()
    )
    sample_t1.columns = ["sample_id", "t1_date", "label"]
    sample_t1 = sample_t1.sort_values("t1_date").reset_index(drop=True)

    n = len(sample_t1)
    fold_size = n // n_splits

    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n

        test_ids = set(sample_t1.iloc[test_start:test_end]["sample_id"])
        train_ids = set(sample_t1.iloc[:test_start]["sample_id"])

        if len(train_ids) == 0 or len(test_ids) == 0:
            continue

        train_df = df[df["sample_id"].isin(train_ids)].copy()
        test_df = df[df["sample_id"].isin(test_ids)].copy()

        # 从 train 中划分 val (训练集最后 15%)
        train_samples = sample_t1[sample_t1["sample_id"].isin(train_ids)].copy()
        val_split = int(len(train_samples) * 0.85)
        val_ids = set(train_samples.iloc[val_split:]["sample_id"])
        train_ids_final = set(train_samples.iloc[:val_split]["sample_id"])

        train_df_final = df[df["sample_id"].isin(train_ids_final)].copy()
        val_df = df[df["sample_id"].isin(val_ids)].copy()

        splits.append((train_df_final, val_df, test_df))
        log.info(
            f"  Fold {i+1}: 训练 {len(train_ids_final)} 样本 / 验证 {len(val_ids)} 样本 / 测试 {len(test_ids)} 样本"
        )

    return splits


# ============================================================================
# Fold 内预处理
# ============================================================================
def preprocess_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    neg_ratio: float = NEG_SAMPLING_RATIO,
    hard_ratio: float = HARD_NEG_SAMPLING_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fold 内预处理:
    1. 负样本分层下采样 (按市值匹配)
    """
    # 分离正负硬负
    train_pos = train_df[train_df["label"] == 1].copy()
    train_neg = train_df[train_df["label"] == 0].copy()

    # 下采样负样本
    n_pos = train_pos["sample_id"].nunique()
    target_neg = int(n_pos * neg_ratio)

    neg_samples = train_neg["sample_id"].unique()
    if len(neg_samples) > target_neg:
        np.random.seed(SEED)
        keep_neg = np.random.choice(neg_samples, target_neg, replace=False)
        train_neg = train_neg[train_neg["sample_id"].isin(keep_neg)].copy()

    train_balanced = pd.concat([train_pos, train_neg], ignore_index=True)

    log.info(f"    Fold 预处理后: 正 {train_pos['sample_id'].nunique()} / 负 {train_neg['sample_id'].nunique()} 样本")

    return train_balanced, val_df, test_df


# ============================================================================
# 展平: 多行 -> 单行 (34天 × N维 -> 展平向量)
# ============================================================================
def flatten_multits(df: pd.DataFrame, feature_cols: List[str], expected_days: List[int] = None) -> pd.DataFrame:
    """
    将多行时间序列展平为单行特征
    sample_id x days_to_t1 -> sample_id x (feature_day)
    """
    # 确保 days_to_t1 为数值
    df = df.copy()
    df["days_to_t1"] = pd.to_numeric(df["days_to_t1"], errors="coerce")

    if expected_days is None:
        expected_days = sorted(df["days_to_t1"].dropna().unique())

    # pivot: index=sample_id, columns=days_to_t1, values=feature
    flat_dfs = []
    for feat in feature_cols:
        pivot = df.pivot(index="sample_id", columns="days_to_t1", values=feat)
        # 确保所有 expected_days 列都存在
        for d in expected_days:
            if d not in pivot.columns:
                pivot[d] = 0.0
        pivot = pivot[expected_days]  # 固定列顺序
        pivot.columns = [f"{feat}_d{int(c)}" for c in pivot.columns]
        flat_dfs.append(pivot)

    flat = pd.concat(flat_dfs, axis=1)

    # 合并标签和元数据 (取每个 sample_id 的第一行)
    meta = (
        df.groupby("sample_id")
        .agg(
            {
                "label": "first",
                "trade_date": "max",
                "ts_code": "first",
            }
        )
        .reset_index()
    )

    flat = flat.merge(meta, on="sample_id", how="left")
    return flat.reset_index(drop=True)


# ============================================================================
# 模型训练辅助函数 (全部使用展平数据)
# ============================================================================
def train_xgb_flat(X_train, y_train, X_val, y_val, feature_names):
    """XGBoost 展平训练"""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model


def train_lgb_flat(X_train, y_train, X_val, y_val, feature_names):
    """LightGBM 展平训练"""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_cat_flat(X_train, y_train, X_val, y_val):
    """CatBoost 展平训练"""
    model = CatBoostClassifier(**CAT_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def train_lr_flat(X_train, y_train):
    """逻辑回归 (展平 + L2，快速收敛)"""
    model = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        max_iter=500,
        tol=1e-3,
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_lgb_ranker(df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: List[str], expected_days: List[int]):
    """LightGBM Ranker (按 trade_date 分组, 使用展平数据)

    注意: lambdarank 要求数据必须按 group 排序，group 数组描述每组大小。
    """
    train_flat = flatten_multits(df_train, feature_cols, expected_days)
    val_flat = flatten_multits(df_val, feature_cols, expected_days)
    flat_cols = [c for c in train_flat.columns if c not in {"sample_id", "label", "trade_date", "ts_code"}]

    # 按 trade_date 排序，确保数据顺序与 group 数组一致
    train_flat = train_flat.sort_values("trade_date").reset_index(drop=True)
    val_flat = val_flat.sort_values("trade_date").reset_index(drop=True)

    train_groups = train_flat.groupby("trade_date").size().values
    val_groups = val_flat.groupby("trade_date").size().values

    train_data = lgb.Dataset(
        train_flat[flat_cols],
        label=train_flat["label"],
        group=train_groups,
    )
    val_data = lgb.Dataset(
        val_flat[flat_cols],
        label=val_flat["label"],
        group=val_groups,
        reference=train_data,
    )

    model = lgb.train(
        LGB_RANK_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model, flat_cols


# ============================================================================
# 评估
# ============================================================================
def evaluate_probs(y_true, probs, prefix=""):
    """评估概率预测"""
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    # 找最佳 F1 阈值
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    pred_best = (probs >= best_thresh).astype(int)
    p = precision_score(y_true, pred_best, zero_division=0)
    r = recall_score(y_true, pred_best, zero_division=0)

    log.info(f"  {prefix}AUC={auc:.4f}  F1={best_f1:.4f}(t={best_thresh:.2f})  Brier={brier:.4f}  P={p:.4f}  R={r:.4f}")

    return {"auc": auc, "f1": best_f1, "brier": brier, "precision": p, "recall": r, "threshold": best_thresh}


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("\n" + "=" * 80)
    log.info(f"v3.x 系列模型训练 —— {MODEL_VERSION}")
    log.info("=" * 80)

    # 1. 加载数据
    df_all, feature_cols = load_training_data()

    # 2. 展平数据 (用于 LR 和展平树模型)
    log.info("\n展平多行时间序列...")
    expected_days = sorted(df_all["days_to_t1"].dropna().unique())
    df_flat = flatten_multits(df_all, feature_cols, expected_days)
    flat_feature_cols = [c for c in df_flat.columns if c not in {"sample_id", "label", "trade_date", "ts_code"}]
    log.info(f"展平后: {df_flat['sample_id'].nunique()} 样本 × {len(flat_feature_cols)} 维")

    # 3. 时间序列 CV
    log.info("\n时间序列 CV 划分...")
    cv_splits = time_series_cv_splits(df_all, n_splits=5)

    if not cv_splits:
        log.error("CV 划分失败")
        return

    # 4. 逐 fold 训练
    results = {name: [] for name in ["xgb_flat", "lgb_flat", "cat_flat", "lr_flat", "lgb_ranker"]}
    fold_ensemble_weights = []

    for fold_idx, (train_df, val_df, test_df) in enumerate(cv_splits):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_idx + 1}/{len(cv_splits)}")
        log.info(f"{'='*60}")

        # --- 预处理 ---
        train_bal, val_bal, test_bal = preprocess_fold(train_df, val_df, test_df, feature_cols)

        # 展平 fold 数据
        train_flat = flatten_multits(train_bal, feature_cols, expected_days)
        val_flat = flatten_multits(val_bal, feature_cols, expected_days)
        test_flat = flatten_multits(test_bal, feature_cols, expected_days)

        # 获取测试集样本级标签
        test_labels = test_bal.groupby("sample_id")["label"].first().reset_index()

        # --- 模型1: XGBoost 展平 ---
        log.info("  [XGB-Flat]")
        xgb_model = train_xgb_flat(
            train_flat[flat_feature_cols],
            train_flat["label"],
            val_flat[flat_feature_cols],
            val_flat["label"],
            flat_feature_cols,
        )
        xgb_probs = pd.DataFrame(
            {
                "sample_id": test_flat["sample_id"].values,
                "prob": xgb_model.predict(xgb.DMatrix(test_flat[flat_feature_cols], feature_names=flat_feature_cols)),
            }
        )
        xgb_metrics = evaluate_probs(
            test_labels.merge(xgb_probs, on="sample_id")["label"].values,
            test_labels.merge(xgb_probs, on="sample_id")["prob"].values,
            "XGB-Flat:    ",
        )
        results["xgb_flat"].append(xgb_metrics)

        # --- 模型2: LightGBM 展平 ---
        log.info("  [LGB-Flat]")
        lgb_model = train_lgb_flat(
            train_flat[flat_feature_cols],
            train_flat["label"],
            val_flat[flat_feature_cols],
            val_flat["label"],
            flat_feature_cols,
        )
        lgb_probs = pd.DataFrame(
            {
                "sample_id": test_flat["sample_id"].values,
                "prob": lgb_model.predict(test_flat[flat_feature_cols]),
            }
        )
        lgb_metrics = evaluate_probs(
            test_labels.merge(lgb_probs, on="sample_id")["label"].values,
            test_labels.merge(lgb_probs, on="sample_id")["prob"].values,
            "LGB-Flat:    ",
        )
        results["lgb_flat"].append(lgb_metrics)

        # --- 模型3: CatBoost 展平 ---
        log.info("  [CAT-Flat]")
        cat_model = train_cat_flat(
            train_flat[flat_feature_cols],
            train_flat["label"],
            val_flat[flat_feature_cols],
            val_flat["label"],
        )
        cat_probs = pd.DataFrame(
            {
                "sample_id": test_flat["sample_id"].values,
                "prob": cat_model.predict_proba(test_flat[flat_feature_cols])[:, 1],
            }
        )
        cat_metrics = evaluate_probs(
            test_labels.merge(cat_probs, on="sample_id")["label"].values,
            test_labels.merge(cat_probs, on="sample_id")["prob"].values,
            "CAT-Flat:    ",
        )
        results["cat_flat"].append(cat_metrics)

        # --- 模型4: 逻辑回归 展平 ---
        log.info("  [LR-Flat]")
        scaler_flat = StandardScaler()
        scaler_flat.fit(train_flat[flat_feature_cols])

        X_train_lr = scaler_flat.transform(train_flat[flat_feature_cols])
        X_val_lr = scaler_flat.transform(val_flat[flat_feature_cols])
        X_test_lr = scaler_flat.transform(test_flat[flat_feature_cols])

        lr_model = train_lr_flat(X_train_lr, train_flat["label"].values)
        lr_probs = pd.DataFrame(
            {
                "sample_id": test_flat["sample_id"].values,
                "prob": lr_model.predict_proba(X_test_lr)[:, 1],
            }
        )
        lr_metrics = evaluate_probs(
            test_labels.merge(lr_probs, on="sample_id")["label"].values,
            test_labels.merge(lr_probs, on="sample_id")["prob"].values,
            "LR-Flat:     ",
        )
        results["lr_flat"].append(lr_metrics)

        # --- 模型5: LightGBM Ranker ---
        # 使用原始未下采样数据，避免 lambdarank 因查询组内单一标签而失效
        log.info("  [LGB-Ranker]")
        rank_model, rank_flat_cols = train_lgb_ranker(train_df, val_df, feature_cols, expected_days)
        rank_test_flat = flatten_multits(test_bal, feature_cols, expected_days)
        rank_scores = rank_model.predict(rank_test_flat[rank_flat_cols])
        rank_probs = pd.DataFrame(
            {
                "sample_id": rank_test_flat["sample_id"].values,
                "prob": 1 / (1 + np.exp(-rank_scores)),
            }
        )
        rank_metrics = evaluate_probs(
            test_labels.merge(rank_probs, on="sample_id")["label"].values,
            test_labels.merge(rank_probs, on="sample_id")["prob"].values,
            "LGB-Ranker:  ",
        )
        results["lgb_ranker"].append(rank_metrics)

        # --- Fold 内集成 (简单平均) ---
        log.info("  [Ensemble-Avg]")
        ensemble_df = test_labels[["sample_id", "label"]].copy()
        ensemble_df = ensemble_df.merge(xgb_probs.rename(columns={"prob": "xgb"}), on="sample_id")
        ensemble_df = ensemble_df.merge(lgb_probs.rename(columns={"prob": "lgb"}), on="sample_id")
        ensemble_df = ensemble_df.merge(cat_probs.rename(columns={"prob": "cat"}), on="sample_id")
        ensemble_df = ensemble_df.merge(lr_probs.rename(columns={"prob": "lr"}), on="sample_id")
        ensemble_df = ensemble_df.merge(rank_probs.rename(columns={"prob": "rank"}), on="sample_id")

        ensemble_df["avg_prob"] = ensemble_df[["xgb", "lgb", "cat", "lr", "rank"]].mean(axis=1)
        ensemble_metrics = evaluate_probs(
            ensemble_df["label"].values,
            ensemble_df["avg_prob"].values,
            "Ensemble-Avg: ",
        )

    # 5. 汇总结果
    log.info("\n" + "=" * 80)
    log.info("跨 Fold 汇总")
    log.info("=" * 80)
    summary = {}
    for model_name, fold_results in results.items():
        if not fold_results:
            continue
        aucs = [r["auc"] for r in fold_results]
        f1s = [r["f1"] for r in fold_results]
        briers = [r["brier"] for r in fold_results]
        log.info(
            f"{model_name:20s} AUC={np.mean(aucs):.4f}(±{np.std(aucs):.4f})  F1={np.mean(f1s):.4f}  Brier={np.mean(briers):.4f}"
        )
        summary[model_name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)),
            "brier_mean": float(np.mean(briers)),
            "fold_results": fold_results,
        }

    # 6. 保存结果
    output_dir = MODEL_BASE_DIR / MODEL_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.success(f"\n结果已保存: {output_dir / 'comparison_results.json'}")
    log.info("v3.x 训练完成!")


if __name__ == "__main__":
    main()
