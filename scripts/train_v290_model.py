#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.0 集成模型训练脚本

在 v2.8.1 基础上，大幅扩充硬负样本（998 → 3,490个，3.5x），
解决模型在熊市中反向选股的问题。

训练数据来源：
- 正样本: feature_data_34d_v6.csv (Tushare指标 + 市场环境特征)
- 负样本: negative_feature_data_v2_34d_v6.csv
- 旧硬负: hard_negative_feature_data_34d_v6.csv
- 新硬负: hard_negative_feature_data_34d_v290.csv (2,492个，14x扩充)

Usage:
    python scripts/train_v290_model.py
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


def load_training_data():
    """加载 v2.9.0 训练数据"""
    log.info("加载 v2.9.0 训练数据...")

    pos_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "feature_data_34d_v5_enhanced.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_old_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "hard_negative_feature_data_34d_v5_enhanced.csv"
    hard_new_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0

    df_hard_old = pd.read_csv(hard_old_file)
    df_hard_old["label"] = 0

    df_hard_new = pd.read_csv(hard_new_file)
    df_hard_new["label"] = 0

    # 统一日期格式
    for df in [df_pos, df_neg, df_hard_old, df_hard_new]:
        if "trade_date" in df.columns:
            # 修复：整数格式(20190805)会被pd.to_datetime误解析为Unix时间戳
            # 先统一转为字符串再解析
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
    hard_old_cols = set(df_hard_old.columns) - exclude_cols
    hard_new_cols = set(df_hard_new.columns) - exclude_cols

    common_cols = list(pos_cols & neg_cols & hard_old_cols & hard_new_cols)

    # 样本统计
    log.info(f"正样本: {df_pos['sample_id'].nunique()} 个 ({len(df_pos)} 行)")
    log.info(f"负样本: {df_neg['sample_id'].nunique()} 个 ({len(df_neg)} 行)")
    log.info(f"旧硬负: {df_hard_old['sample_id'].nunique()} 个 ({len(df_hard_old)} 行)")
    log.info(f"新硬负: {df_hard_new['sample_id'].nunique()} 个 ({len(df_hard_new)} 行)")
    total_neg = df_neg["sample_id"].nunique() + df_hard_old["sample_id"].nunique() + df_hard_new["sample_id"].nunique()
    log.info(
        f"硬负样本占比: "
        f"{(df_hard_old['sample_id'].nunique() + df_hard_new['sample_id'].nunique()) / total_neg * 100:.1f}%"
    )

    # P1: 控制硬负样本比例不超过18%，超过则随机下采样
    target_hard_ratio = 0.18
    current_hard_count = df_hard_new["sample_id"].nunique()
    current_neg_count = df_neg["sample_id"].nunique() + df_hard_old["sample_id"].nunique()
    max_hard = int(current_neg_count * target_hard_ratio / (1 - target_hard_ratio))
    if current_hard_count > max_hard:
        log.warning(f"硬负样本过多: {current_hard_count}，下采样至 {max_hard} (目标占比 {target_hard_ratio*100:.0f}%)")
        keep_ids = df_hard_new["sample_id"].drop_duplicates().sample(n=max_hard, random_state=42).tolist()
        df_hard_new = df_hard_new[df_hard_new["sample_id"].isin(keep_ids)].copy()
        log.info(f"下采样后新硬负: {df_hard_new['sample_id'].nunique()} 个 ({len(df_hard_new)} 行)")

    # P1: 动态计算 scale_pos_weight
    total_neg_rows = len(df_neg) + len(df_hard_old) + len(df_hard_new)
    total_pos_rows = len(df_pos)
    dynamic_spw = total_neg_rows / total_pos_rows
    log.info(f"正负样本行数比: {total_neg_rows}/{total_pos_rows} = {dynamic_spw:.2f}")
    log.info(f"动态 scale_pos_weight: {dynamic_spw:.2f} (替代固定值 1.5)")

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_old[common_cols + ["label", "trade_date"]],
            df_hard_new[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )

    log.info(f"数据加载完成: {len(df)} 条，特征数: {len(common_cols)}")
    return df, common_cols, dynamic_spw


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

    train_dates = set(unique_dates[:train_end])
    cal_dates = set(unique_dates[train_end:cal_end])
    test_dates = set(unique_dates[cal_end:])

    train = df[df["trade_date"].dt.date.isin(train_dates)].copy()
    cal = df[df["trade_date"].dt.date.isin(cal_dates)].copy()
    test = df[df["trade_date"].dt.date.isin(test_dates)].copy()

    log.info(f"日期切分: 训练 {len(train_dates)}天, 校准 {len(cal_dates)}天, 测试 {len(test_dates)}天")
    log.info(f"样本切分: 训练 {len(train)}行, 校准 {len(cal)}行, 测试 {len(test)}行")

    return train, cal, test


def train_xgboost(X_train, y_train, X_val, y_val, feature_names, scale_pos_weight):
    """训练 XGBoost"""
    log.info("训练 XGBoost...")

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
        "scale_pos_weight": scale_pos_weight,
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
    log.info(f"  XGBoost 验证集 AUC: {auc:.4f}")

    return model, auc


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names, scale_pos_weight):
    """训练 LightGBM"""
    log.info("训练 LightGBM...")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
    }

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
    log.info(f"  LightGBM 验证集 AUC: {auc:.4f}")

    return model, auc


def train_catboost(X_train, y_train, X_val, y_val, feature_names):
    """训练 CatBoost"""
    log.info("训练 CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  CatBoost 验证集 AUC: {auc:.4f}")

    return model, auc


def evaluate_model(model, X_test, y_test, model_name, feature_names):
    """评估模型"""
    log.info(f"\n评估 {model_name}...")

    if model_name == "xgboost":
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        y_pred = model.predict(dtest)
    elif model_name == "lightgbm":
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        y_pred = model.predict_proba(X_test)[:, 1]

    y_pred_binary = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    log.info(f"  AUC: {auc:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall: {recall:.4f}")
    log.info(f"  F1: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred_binary)
    log.info(f"  混淆矩阵:\n{cm}")

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def main():
    log.info("=" * 80)
    log.info("v2.9.1 集成模型训练（硬负样本修正版：阈值收紧+trade_date修复+动态权重）")
    log.info("=" * 80)

    # 加载数据
    df, feature_cols, dynamic_spw = load_training_data()
    valid_features = get_feature_columns(df, feature_cols)
    log.info(f"有效特征: {len(valid_features)} 个")

    # 划分数据集
    train, cal, test = time_series_split(df)

    X_train = train[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train["label"]
    X_cal = cal[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_cal = cal["label"]
    X_test = test[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test["label"]

    # 训练三个模型
    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_cal, y_cal, valid_features, dynamic_spw)
    lgb_model, lgb_auc = train_lightgbm(X_train, y_train, X_cal, y_cal, valid_features, dynamic_spw)
    cat_model, cat_auc = train_catboost(X_train, y_train, X_cal, y_cal, valid_features)

    # 计算集成权重（基于验证集 AUC）
    total_auc = xgb_auc + lgb_auc + cat_auc
    weights = {
        "xgboost": xgb_auc / total_auc,
        "lightgbm": lgb_auc / total_auc,
        "catboost": cat_auc / total_auc,
    }

    log.info("\n集成权重:")
    log.info(f"  XGBoost: {weights['xgboost']:.4f}")
    log.info(f"  LightGBM: {weights['lightgbm']:.4f}")
    log.info(f"  CatBoost: {weights['catboost']:.4f}")

    # 评估集成模型
    log.info("\n评估集成模型...")
    pred_xgb = xgb_model.predict(xgb.DMatrix(X_test, feature_names=valid_features))
    pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    pred_cat = cat_model.predict_proba(X_test)[:, 1]

    ensemble_pred = pred_xgb * weights["xgboost"] + pred_lgb * weights["lightgbm"] + pred_cat * weights["catboost"]

    ensemble_binary = (ensemble_pred >= 0.5).astype(int)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    ensemble_precision = precision_score(y_test, ensemble_binary, zero_division=0)
    ensemble_recall = recall_score(y_test, ensemble_binary, zero_division=0)
    ensemble_f1 = f1_score(y_test, ensemble_binary, zero_division=0)

    log.info(f"  集成 AUC: {ensemble_auc:.4f}")
    log.info(f"  集成 Precision: {ensemble_precision:.4f}")
    log.info(f"  集成 Recall: {ensemble_recall:.4f}")
    log.info(f"  集成 F1: {ensemble_f1:.4f}")

    # 保存模型
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.9.1-ensemble" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(str(model_dir / "xgboost.json"))
    lgb_model.save_model(str(model_dir / "lightgbm.txt"))
    cat_model.save_model(str(model_dir / "catboost.cbm"))

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_features, f)

    with open(model_dir / "weights.json", "w") as f:
        json.dump(weights, f)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "xgboost_auc": xgb_auc,
                "lightgbm_auc": lgb_auc,
                "catboost_auc": cat_auc,
                "ensemble_auc": ensemble_auc,
                "ensemble_precision": ensemble_precision,
                "ensemble_recall": ensemble_recall,
                "ensemble_f1": ensemble_f1,
                "train_samples": len(train),
                "cal_samples": len(cal),
                "test_samples": len(test),
                "feature_count": len(valid_features),
                "hard_negative_ratio": (998 + 2492) / (7636 + 998 + 2492),
            },
            f,
            indent=2,
        )

    log.success(f"\n模型已保存到: {model_dir}")
    log.success("v2.9.1 训练完成！")


if __name__ == "__main__":
    main()
