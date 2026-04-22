#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超参数搜索脚本 - 基于v5数据快速验证不同参数组合

目标：找到最优的XGBoost超参数组合，提升实盘表现
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log


def load_training_data():
    """加载v5训练数据"""
    log.info("加载v5训练数据...")

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

    # 合并数据
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
    ]

    # 排除未使用的二值特征
    unused_binary_features = [
        "breakout_high_10d",
        "breakout_high_20d",
        "breakout_ma10",
        "breakout_ma55",
        "high_volume_breakout",
        "volume_price_match",
    ]
    exclude_cols.extend(unused_binary_features)

    valid_cols = [
        c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    return valid_cols


def time_series_split(df, train_ratio=0.65, cal_ratio=0.15):
    """时间序列划分"""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    n = len(df)
    train_end = int(n * train_ratio)
    cal_end = int(n * (train_ratio + cal_ratio))

    train_df = df.iloc[:train_end]
    cal_df = df.iloc[train_end:cal_end]
    test_df = df.iloc[cal_end:]

    return train_df, cal_df, test_df


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, params):
    """训练并评估模型"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练
    evals = [(dtrain, "train"), (dval, "val")]
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=30, verbose_eval=False)

    # 预测
    y_pred = bst.predict(dtest)

    # 评估
    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)

    # 使用0.5阈值
    y_pred_binary = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    return {
        "auc": auc,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_iteration": bst.best_iteration,
    }


def run_hyperparameter_search():
    """运行超参数搜索"""
    log.info("=" * 80)
    log.info("超参数搜索 - 基于v5数据")
    log.info("=" * 80)

    # 加载数据
    df, feature_cols = load_training_data()
    valid_features = get_feature_columns(df, feature_cols)
    log.info(f"有效特征数: {len(valid_features)}")

    # 时间序列划分
    train_df, cal_df, test_df = time_series_split(df)
    log.info(f"训练集: {len(train_df)}, 校准集: {len(cal_df)}, 测试集: {len(test_df)}")

    X_train = train_df[valid_features].fillna(0)
    y_train = train_df["label"]
    X_val = cal_df[valid_features].fillna(0)
    y_val = cal_df["label"]
    X_test = test_df[valid_features].fillna(0)
    y_test = test_df["label"]

    # 定义搜索空间
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "subsample": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.3, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    # 固定参数
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "min_child_weight": 5,
        "gamma": 0.1,
        "scale_pos_weight": 1.5,
        "random_state": 42,
        "tree_method": "hist",
    }

    # 快速搜索：只测试关键参数组合
    key_combinations = [
        # 当前配置（baseline）
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "colsample_bytree": 0.6,
            "subsample": 0.8,
            "reg_alpha": 0.3,
            "reg_lambda": 0.5,
        },
        # 更深的树
        {
            "max_depth": 8,
            "learning_rate": 0.03,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
        },
        # 更浅但更多树
        {
            "max_depth": 4,
            "learning_rate": 0.02,
            "colsample_bytree": 0.8,
            "subsample": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        },
        # 强正则化
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "colsample_bytree": 0.6,
            "subsample": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        },
        # 弱正则化
        {
            "max_depth": 6,
            "learning_rate": 0.1,
            "colsample_bytree": 0.8,
            "subsample": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        },
        # 平衡配置
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
        },
    ]

    results = []

    log.info(f"\n开始搜索 {len(key_combinations)} 种参数组合...")

    for i, combo in enumerate(key_combinations):
        params = {**base_params, **combo}

        log.info(f"\n[{i+1}/{len(key_combinations)}] 测试参数:")
        log.info(
            f"  max_depth={combo['max_depth']}, lr={combo['learning_rate']}, "
            f"colsample={combo['colsample_bytree']}, subsample={combo['subsample']}"
        )
        log.info(f"  reg_alpha={combo['reg_alpha']}, reg_lambda={combo['reg_lambda']}")

        metrics = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, params)

        log.info(
            f"  结果: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )

        results.append({"params": combo, "metrics": metrics})

    # 按F1排序
    results.sort(key=lambda x: x["metrics"]["f1"], reverse=True)

    log.info("\n" + "=" * 80)
    log.info("搜索结果排名（按F1）")
    log.info("=" * 80)

    for i, r in enumerate(results):
        m = r["metrics"]
        p = r["params"]
        log.info(
            f"\n#{i+1}: F1={m['f1']:.4f}, AUC={m['auc']:.4f}, "
            f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}"
        )
        log.info(
            f"    max_depth={p['max_depth']}, lr={p['learning_rate']}, "
            f"colsample={p['colsample_bytree']}, subsample={p['subsample']}"
        )
        log.info(f"    reg_alpha={p['reg_alpha']}, reg_lambda={p['reg_lambda']}, " f"best_iter={m['best_iteration']}")

    # 保存结果
    output_file = PROJECT_ROOT / "data" / "training" / "metrics" / "hyperparameter_search_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)

    log.success(f"\n结果已保存到: {output_file}")

    # 返回最佳参数
    best = results[0]
    log.info("\n最佳参数组合:")
    log.info(f"  {best['params']}")
    log.info(f"  F1={best['metrics']['f1']:.4f}, AUC={best['metrics']['auc']:.4f}")

    return best


if __name__ == "__main__":
    run_hyperparameter_search()
