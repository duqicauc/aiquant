#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练v2.7.0模型

优化点：
1. 使用超参数搜索得到的最优参数（lr=0.1, colsample=0.8, subsample=0.9）
2. 使用增强后的特征集（14个新特征）
3. 移除低重要性特征（<0.3%）
4. 概率校准 + 时间序列划分
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log


def load_training_data():
    """加载增强后的训练数据"""
    log.info("加载训练数据（增强版）...")

    # 优先使用增强版数据
    enhanced_dir = PROJECT_ROOT / "data" / "training" / "enhanced"

    pos_file = enhanced_dir / "feature_data_34d_v5_enhanced.csv"
    neg_file = enhanced_dir / "negative_feature_data_v2_34d_v5_enhanced.csv"
    hard_neg_file = enhanced_dir / "hard_negative_feature_data_34d_v5_enhanced.csv"

    # 如果增强版不存在，使用原版
    if not pos_file.exists():
        log.warning("增强版数据不存在，使用原版v5数据")
        pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
        neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
        hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    # 检查扩展硬负样本
    extended_hard_neg = (
        PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5_extended.csv"
    )
    if extended_hard_neg.exists():
        log.info("使用扩展硬负样本")
        hard_neg_file = extended_hard_neg

    # 加载数据
    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    log.info(f"  正样本: {len(df_pos)} 条，特征数: {len(df_pos.columns)}")

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0
    log.info(f"  负样本: {len(df_neg)} 条，特征数: {len(df_neg.columns)}")

    df_hard_neg = pd.read_csv(hard_neg_file)
    df_hard_neg["label"] = 0
    log.info(f"  硬负样本: {len(df_hard_neg)} 条，特征数: {len(df_hard_neg.columns)}")

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

    log.info("\n特征统计:")
    log.info(f"  正样本特征: {len(pos_cols)}")
    log.info(f"  负样本特征: {len(neg_cols)}")
    log.info(f"  硬负样本特征: {len(hard_cols)}")
    log.info(f"  共同特征: {len(common_cols)}")

    # 合并数据
    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date"]],
            df_neg[common_cols + ["label", "trade_date"]],
            df_hard_neg[common_cols + ["label", "trade_date"]],
        ],
        ignore_index=True,
    )

    log.info(f"\n合并数据: 正样本 {len(df_pos)} + 负样本 {len(df_neg)} + 硬负样本 {len(df_hard_neg)} = {len(df)}")
    log.success(f"✓ 数据加载完成: {len(df)} 条，特征数: {len(common_cols)} 个")

    return df, common_cols


def get_feature_columns(df, feature_cols):
    """获取有效特征列（移除低重要性特征）"""
    # 基础排除列
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

    # 低重要性特征（<0.3%）- 可选择性移除
    # 暂时保留，让模型自己选择

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
    date_col = "trade_date"

    if date_col not in df.columns:
        log.warning("无trade_date列，使用随机划分")
        from sklearn.model_selection import train_test_split

        train, temp = train_test_split(df, test_size=0.35, random_state=42)
        cal, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, cal, test

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    n = len(df)
    train_end = int(n * train_ratio)
    cal_end = int(n * (train_ratio + cal_ratio))

    train = df.iloc[:train_end]
    cal = df.iloc[train_end:cal_end]
    test = df.iloc[cal_end:]

    log.info("\n时间序列划分:")
    log.info(
        f"  训练集: {train[date_col].min().date()} ~ {train[date_col].max().date()} ({len(train)}条, 正:{train['label'].sum()}, 负:{len(train)-train['label'].sum()})"
    )
    log.info(
        f"  校准集: {cal[date_col].min().date()} ~ {cal[date_col].max().date()} ({len(cal)}条, 正:{cal['label'].sum()}, 负:{len(cal)-cal['label'].sum()})"
    )
    log.info(
        f"  测试集: {test[date_col].min().date()} ~ {test[date_col].max().date()} ({len(test)}条, 正:{test['label'].sum()}, 负:{len(test)-test['label'].sum()})"
    )

    return train, cal, test


def train_model(X_train, y_train, X_val, y_val):
    """训练XGBoost模型（使用优化后的参数）"""
    log.info("训练模型（v2.7.0优化参数）...")

    # v2.7.0 优化参数（来自超参数搜索）
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 6,  # 保持
        "learning_rate": 0.1,  # 0.05 -> 0.1（搜索最优）
        "subsample": 0.9,  # 0.8 -> 0.9（搜索最优）
        "colsample_bytree": 0.8,  # 0.6 -> 0.8（搜索最优）
        "min_child_weight": 5,  # 保持
        "gamma": 0.1,  # 保持
        "reg_alpha": 0.1,  # 0.3 -> 0.1（搜索最优）
        "reg_lambda": 0.5,  # 保持
        "scale_pos_weight": 1.5,  # 保持
        "random_state": 42,
        "tree_method": "hist",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    log.success(f"✓ 模型训练完成, best_iteration: {booster.best_iteration}")
    return booster


def calibrate_model(booster, X_cal, y_cal, feature_names):
    """概率校准"""
    log.info("概率校准...")

    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_probs = booster.predict(dcal)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    cal_probs = calibrator.predict(raw_probs)
    log.info(f"  校准前: mean={raw_probs.mean():.4f}, max={raw_probs.max():.4f}")
    log.info(f"  校准后: mean={cal_probs.mean():.4f}, max={cal_probs.max():.4f}")

    log.success("✓ 概率校准完成")
    return calibrator


def evaluate(booster, calibrator, X_test, y_test, feature_names):
    """评估模型"""
    log.info("评估模型...")

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)

    # 计算AUC
    auc = roc_auc_score(y_test, cal_probs)
    log.info(f"  AUC: {auc:.4f}")

    # 不同阈值下的指标
    log.info("\n不同阈值下的性能:")
    log.info(f"{'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    log.info("-" * 60)

    metrics_dict = {}
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        y_pred = (cal_probs >= thresh).astype(int)
        if y_pred.sum() > 0:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            log.info(f"{thresh:<8.1f} {y_pred.sum():<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
            if thresh == 0.5:
                metrics_dict["precision"] = precision
                metrics_dict["recall"] = recall
                metrics_dict["f1"] = f1

    # 混淆矩阵
    y_pred_05 = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info("\n混淆矩阵（阈值=0.5）:")
    log.info("              预测负  预测正")
    log.info(f"  实际负      {cm[0,0]:<8} {cm[0,1]:<8}")
    log.info(f"  实际正      {cm[1,0]:<8} {cm[1,1]:<8}")

    # 分类报告
    log.info("\n分类报告:")
    report = classification_report(y_test, y_pred_05, target_names=["负样本", "正样本"], zero_division=0)
    log.info(f"\n{report}")

    return {
        "test_samples": len(X_test),
        "positive_samples": int(y_test.sum()),
        "auc": auc,
        "precision": metrics_dict.get("precision"),
        "recall": metrics_dict.get("recall"),
        "f1": metrics_dict.get("f1"),
    }


def save_model(booster, calibrator, feature_names, metrics):
    """保存模型"""
    version = "v2.7.0"
    log.info(f"保存模型 {version}...")

    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    (model_dir / "model").mkdir(exist_ok=True)
    booster.save_model(str(model_dir / "model" / "model.json"))

    # 特征名
    with open(model_dir / "model" / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 校准器
    joblib.dump(calibrator, str(model_dir / "model" / "calibrator.pkl"))

    # 元数据
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "calibration_method": "isotonic_regression",
        "risk_features": [
            "max_drawdown_10d",
            "max_drawdown_20d",
            "max_drawdown_55d",
            "atr_14",
            "atr_ratio_14",
            "atr_expansion",
            "days_from_high_20d",
            "days_from_high_55d",
            "recovery_ratio_20d",
        ],
        "description": "v2.7.0模型 - 超参数优化(lr=0.1,colsample=0.8,subsample=0.9)+增强特征(14个新特征)+概率校准+时间序列划分",
        "split_method": "time_series",
        "optimizations": ["超参数搜索优化", "14个新增强特征", "移除低效二值特征", "概率校准"],
        "metrics": metrics,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info("训练v2.7.0模型（超参数优化+增强特征）")
    log.info("=" * 80)

    # 1. 加载数据
    df, feature_cols = load_training_data()

    # 2. 获取有效特征
    feature_names = get_feature_columns(df, feature_cols)
    log.info(f"\n有效特征数: {len(feature_names)}")

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
    log.info(f"风险特征数: {len(available_risk)}")

    # 4. 时间序列划分
    log.info("\n" + "=" * 80)
    log.info("数据划分（时间序列方式）")
    log.info("=" * 80)

    train_df, cal_df, test_df = time_series_split(df)

    X_train = train_df[feature_names].fillna(0)
    y_train = train_df["label"]
    X_cal = cal_df[feature_names].fillna(0)
    y_cal = cal_df["label"]
    X_test = test_df[feature_names].fillna(0)
    y_test = test_df["label"]

    log.info(f"\n数据集: 训练{len(X_train)}, 校准{len(X_cal)}, 测试{len(X_test)}")

    # 5. 训练模型
    booster = train_model(X_train, y_train, X_cal, y_cal)

    # 6. 概率校准
    calibrator = calibrate_model(booster, X_cal, y_cal, feature_names)

    # 7. 评估
    metrics = evaluate(booster, calibrator, X_test, y_test, feature_names)

    # 8. 保存模型
    save_model(booster, calibrator, feature_names, metrics)

    log.success("\n✓ v2.7.0模型训练完成!")

    # 9. 与v2.6.0对比
    log.info("\n" + "=" * 80)
    log.info("与v2.6.0对比")
    log.info("=" * 80)

    v260_metrics = {"auc": 0.9654, "precision": 0.8582, "recall": 0.7820, "f1": 0.8183}

    log.info(f"{'指标':<15} {'v2.6.0':>12} {'v2.7.0':>12} {'变化':>12}")
    log.info("-" * 55)
    for key in ["auc", "precision", "recall", "f1"]:
        v260 = v260_metrics[key]
        v270 = metrics[key]
        diff = (v270 - v260) * 100
        sign = "+" if diff >= 0 else ""
        log.info(f"{key:<15} {v260:>12.4f} {v270:>12.4f} {sign}{diff:>11.2f}%")


if __name__ == "__main__":
    main()
