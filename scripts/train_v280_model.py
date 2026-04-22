#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成学习模型训练脚本

实现XGBoost + LightGBM + CatBoost集成
使用加权投票策略，权重基于验证集表现
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
    """加载增强后的训练数据"""
    log.info("加载训练数据...")

    enhanced_dir = PROJECT_ROOT / "data" / "training" / "enhanced"

    pos_file = enhanced_dir / "feature_data_34d_v5_enhanced.csv"
    neg_file = enhanced_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_file = enhanced_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"

    if not pos_file.exists():
        pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
        neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
        hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0

    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0

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
    """时间序列划分"""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
    df = df.sort_values("trade_date")

    n = len(df)
    train_end = int(n * train_ratio)
    cal_end = int(n * (train_ratio + cal_ratio))

    train = df.iloc[:train_end]
    cal = df.iloc[train_end:cal_end]
    test = df.iloc[cal_end:]

    return train, cal, test


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """训练XGBoost模型"""
    log.info("训练XGBoost...")

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

    # 验证集AUC
    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  XGBoost 验证集 AUC: {auc:.4f}")

    return model, auc


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """训练LightGBM模型"""
    log.info("训练LightGBM...")

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "scale_pos_weight": 1.5,
        "random_state": 42,
        "verbose": -1,
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

    # 验证集AUC
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  LightGBM 验证集 AUC: {auc:.4f}")

    return model, auc


def train_catboost(X_train, y_train, X_val, y_val, feature_names):
    """训练CatBoost模型"""
    log.info("训练CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        border_count=128,
        scale_pos_weight=1.5,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # 验证集AUC
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"  CatBoost 验证集 AUC: {auc:.4f}")

    return model, auc


def ensemble_predict(models, weights, X, feature_names):
    """集成预测"""
    xgb_model, lgb_model, cat_model = models
    w_xgb, w_lgb, w_cat = weights

    # XGBoost预测
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    xgb_pred = xgb_model.predict(dmatrix)

    # LightGBM预测
    lgb_pred = lgb_model.predict(X)

    # CatBoost预测
    cat_pred = cat_model.predict_proba(X)[:, 1]

    # 加权平均
    ensemble_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred

    return ensemble_pred


def evaluate_ensemble(models, weights, X_test, y_test, feature_names):
    """评估集成模型"""
    log.info("评估集成模型...")

    # 集成预测
    ensemble_pred = ensemble_predict(models, weights, X_test, feature_names)

    # 计算指标
    auc = roc_auc_score(y_test, ensemble_pred)

    log.info("\n集成模型性能:")
    log.info(f"  AUC: {auc:.4f}")

    # 不同阈值
    log.info(f"\n{'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    log.info("-" * 60)

    metrics_dict = {}
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        y_pred = (ensemble_pred >= thresh).astype(int)
        if y_pred.sum() > 0:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            log.info(f"{thresh:<8.1f} {y_pred.sum():<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
            if thresh == 0.5:
                metrics_dict = {"precision": precision, "recall": recall, "f1": f1}

    # 混淆矩阵
    y_pred_05 = (ensemble_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info("\n混淆矩阵（阈值=0.5）:")
    log.info("              预测负  预测正")
    log.info(f"  实际负      {cm[0,0]:<8} {cm[0,1]:<8}")
    log.info(f"  实际正      {cm[1,0]:<8} {cm[1,1]:<8}")

    return {"auc": auc, **metrics_dict}


def save_ensemble_model(models, weights, feature_names, metrics):
    """保存集成模型"""
    version = "v2.8.0-ensemble"
    log.info(f"保存集成模型 {version}...")

    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version
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
        json.dump({"xgboost": weights[0], "lightgbm": weights[1], "catboost": weights[2]}, f, indent=2)

    # 元数据
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "models": ["xgboost", "lightgbm", "catboost"],
        "weights": {"xgboost": weights[0], "lightgbm": weights[1], "catboost": weights[2]},
        "description": "v2.8.0集成模型 - XGBoost+LightGBM+CatBoost加权投票 (增量更新数据)",
        "metrics": metrics,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 集成模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info("训练集成模型（XGBoost + LightGBM + CatBoost）")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols = load_training_data()
    feature_names = get_feature_columns(df, feature_cols)
    log.info(f"有效特征数: {len(feature_names)}")

    # 2. 时间序列划分
    train_df, cal_df, test_df = time_series_split(df)

    X_train = train_df[feature_names].fillna(0).values
    y_train = train_df["label"].values
    X_val = cal_df[feature_names].fillna(0).values
    y_val = cal_df["label"].values
    X_test = test_df[feature_names].fillna(0).values
    y_test = test_df["label"].values

    log.info(f"\n数据集: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")

    # 3. 训练各模型
    log.info("\n" + "=" * 80)
    log.info("训练各基模型")
    log.info("=" * 80)

    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    lgb_model, lgb_auc = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
    cat_model, cat_auc = train_catboost(X_train, y_train, X_val, y_val, feature_names)

    # 4. 计算权重（基于验证集AUC）
    total_auc = xgb_auc + lgb_auc + cat_auc
    w_xgb = xgb_auc / total_auc
    w_lgb = lgb_auc / total_auc
    w_cat = cat_auc / total_auc

    log.info("\n模型权重（基于验证集AUC）:")
    log.info(f"  XGBoost:  {w_xgb:.4f} (AUC={xgb_auc:.4f})")
    log.info(f"  LightGBM: {w_lgb:.4f} (AUC={lgb_auc:.4f})")
    log.info(f"  CatBoost: {w_cat:.4f} (AUC={cat_auc:.4f})")

    # 5. 评估集成模型
    log.info("\n" + "=" * 80)
    log.info("评估集成模型")
    log.info("=" * 80)

    models = (xgb_model, lgb_model, cat_model)
    weights = (w_xgb, w_lgb, w_cat)

    metrics = evaluate_ensemble(models, weights, X_test, y_test, feature_names)

    # 6. 保存模型
    save_ensemble_model(models, weights, feature_names, metrics)

    # 7. 对比
    log.info("\n" + "=" * 80)
    log.info("与单模型对比")
    log.info("=" * 80)

    # 单独评估各模型
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    xgb_pred = xgb_model.predict(dtest)
    lgb_pred = lgb_model.predict(X_test)
    cat_pred = cat_model.predict_proba(X_test)[:, 1]

    log.info(f"\n{'模型':<15} {'AUC':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    log.info("-" * 60)

    for name, pred in [("XGBoost", xgb_pred), ("LightGBM", lgb_pred), ("CatBoost", cat_pred)]:
        auc = roc_auc_score(y_test, pred)
        y_pred_bin = (pred >= 0.5).astype(int)
        precision = precision_score(y_test, y_pred_bin, zero_division=0)
        recall = recall_score(y_test, y_pred_bin, zero_division=0)
        f1 = f1_score(y_test, y_pred_bin, zero_division=0)
        log.info(f"{name:<15} {auc:>10.4f} {precision:>12.4f} {recall:>10.4f} {f1:>10.4f}")

    log.info(
        f"{'Ensemble':<15} {metrics['auc']:>10.4f} "
        f"{metrics['precision']:>12.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}"
    )

    log.success("\n✓ 集成模型训练完成!")


if __name__ == "__main__":
    main()
