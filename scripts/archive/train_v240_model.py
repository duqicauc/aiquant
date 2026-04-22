#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练v2.4.0模型 - 基于v4版本完整数据（144特征）

特点：
1. 基于v3版本139个特征 + 5个反追龙头新特征 = 144特征
2. 正样本已应用T1前约束筛选（return_34d <= 20%, volatility_34d <= 3%）
3. 时间序列划分（避免未来函数）
4. 概率校准（Isotonic Regression）

数据来源：
- 正样本: data/training/processed/feature_data_34d_v4.csv (43,370条)
- 负样本: data/training/features/negative_feature_data_v2_34d_v4.csv (218,337条)
- 硬负样本: data/training/features/hard_negative_feature_data_34d_v4.csv (33,932条)

使用方法：
  python scripts/train_v240_model.py
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
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log

VERSION = "v2.4.0"

# 元数据列（不作为特征）
META_COLS = ["sample_id", "trade_date", "name", "ts_code", "label"]


def load_training_data():
    """加载v4版本训练数据"""
    log.info("=" * 80)
    log.info("第一步：加载v4版本训练数据")
    log.info("=" * 80)

    # 文件路径
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v4.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v4.csv"
    hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v4.csv"

    # 检查文件存在
    for f in [pos_file, neg_file, hard_neg_file]:
        if not f.exists():
            log.error(f"文件不存在: {f}")
            log.error("请先运行: python scripts/add_anti_chasing_features.py")
            return None

    # 加载正样本
    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    log.info(f"  正样本(已筛选): {len(df_pos)} 条, {len(df_pos.columns)} 列")

    # 加载普通负样本
    df_neg = pd.read_csv(neg_file)
    if "label" not in df_neg.columns:
        df_neg["label"] = 0
    log.info(f"  普通负样本: {len(df_neg)} 条, {len(df_neg.columns)} 列")

    # 加载硬负样本
    df_hard_neg = pd.read_csv(hard_neg_file)
    if "label" not in df_hard_neg.columns:
        df_hard_neg["label"] = 0
    log.info(f"  硬负样本: {len(df_hard_neg)} 条, {len(df_hard_neg.columns)} 列")

    # 合并
    df = pd.concat([df_pos, df_neg, df_hard_neg], ignore_index=True)
    log.success(f"✓ 数据加载完成: {len(df)} 条")

    return df


def get_feature_columns(df):
    """获取特征列（排除元数据列）"""
    feature_cols = [col for col in df.columns if col not in META_COLS]
    # 只保留数值列
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols


def time_series_split(df, test_ratio=0.2, cal_ratio=0.15):
    """
    时间序列划分（避免未来函数）

    按trade_date排序，前面的数据用于训练，后面的用于测试
    """
    log.info("=" * 80)
    log.info("第二步：时间序列划分")
    log.info("=" * 80)

    # 确保有trade_date列
    if "trade_date" not in df.columns:
        log.warning("没有trade_date列，使用随机划分")
        return random_split(df, test_ratio, cal_ratio)

    # 按日期排序
    df = df.sort_values("trade_date").reset_index(drop=True)

    n = len(df)
    test_start = int(n * (1 - test_ratio))
    cal_start = int(n * (1 - test_ratio - cal_ratio))

    df_train = df.iloc[:cal_start]
    df_cal = df.iloc[cal_start:test_start]
    df_test = df.iloc[test_start:]

    log.info(f"  训练集: {len(df_train)} (正:{df_train['label'].sum()}, 负:{len(df_train)-df_train['label'].sum()})")
    log.info(f"  校准集: {len(df_cal)} (正:{df_cal['label'].sum()}, 负:{len(df_cal)-df_cal['label'].sum()})")
    log.info(f"  测试集: {len(df_test)} (正:{df_test['label'].sum()}, 负:{len(df_test)-df_test['label'].sum()})")

    # 显示日期范围
    log.info(f"  训练集日期: {df_train['trade_date'].min()} ~ {df_train['trade_date'].max()}")
    log.info(f"  校准集日期: {df_cal['trade_date'].min()} ~ {df_cal['trade_date'].max()}")
    log.info(f"  测试集日期: {df_test['trade_date'].min()} ~ {df_test['trade_date'].max()}")

    return df_train, df_cal, df_test


def random_split(df, test_ratio=0.2, cal_ratio=0.15):
    """随机划分（备用）"""

    df_train_full, df_test = train_test_split(df, test_size=test_ratio, random_state=42, stratify=df["label"])
    df_train, df_cal = train_test_split(
        df_train_full, test_size=cal_ratio / (1 - test_ratio), random_state=42, stratify=df_train_full["label"]
    )

    return df_train, df_cal, df_test


def train_model(X_train, y_train, X_val, y_val, feature_names):
    """训练XGBoost模型"""
    log.info("=" * 80)
    log.info("第三步：训练XGBoost模型")
    log.info("=" * 80)

    # 与v2.3.0一致的参数
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
        "min_child_weight": 10,
        "gamma": 0.3,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "scale_pos_weight": 1.5,
        "random_state": 42,
        "n_jobs": -1,
    }

    log.info("模型参数:")
    for k, v in params.items():
        log.info(f"  {k}: {v}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

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
    """概率校准（Isotonic Regression）"""
    log.info("=" * 80)
    log.info("第四步：概率校准")
    log.info("=" * 80)

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
    log.info("=" * 80)
    log.info("第五步：模型评估")
    log.info("=" * 80)

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)

    # 不同阈值下的准确率
    log.info("\n校准概率阈值分析:")
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        cal_high = cal_probs >= thresh
        if cal_high.sum() > 0:
            acc = y_test[cal_high].mean()
            log.info(f"  校准概率>={thresh}: {cal_high.sum()}个, 真实正确率{acc:.1%}")
        else:
            log.info(f"  校准概率>={thresh}: 0个")

    # 特征重要性
    importance = booster.get_score(importance_type="gain")
    importance_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in importance.items()]).sort_values(
        "importance", ascending=False
    )

    log.info("\n特征重要性 Top 20:")
    for idx, row in importance_df.head(20).iterrows():
        log.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # 检查新增特征的重要性
    new_features = ["price_range_pct", "close_vs_ma10_std", "days_near_ma10", "volume_shrink_ratio", "ma10_cross_count"]

    log.info("\nv2.4.0新增特征重要性:")
    for f in new_features:
        if f in importance:
            rank = list(importance_df["feature"]).index(f) + 1
            log.info(f"  {f:25s}: {importance[f]:.4f} (排名第{rank})")
        else:
            log.info(f"  {f:25s}: 未使用")

    return {
        "test_samples": len(X_test),
        "positive_samples": int(y_test.sum()),
        "feature_importance": importance_df.head(30).to_dict("records"),
    }


def save_model(booster, calibrator, feature_names, metrics):
    """保存模型到v2.4.0目录"""
    log.info("=" * 80)
    log.info(f"第六步：保存模型 {VERSION}")
    log.info("=" * 80)

    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION
    model_dir.mkdir(parents=True, exist_ok=True)

    # 模型目录
    (model_dir / "model").mkdir(exist_ok=True)

    # 保存XGBoost模型
    model_file = model_dir / "model" / "model.json"
    booster.save_model(str(model_file))
    log.success(f"✓ 模型已保存: {model_file}")

    # 保存特征名称
    feature_names_file = model_dir / "model" / "feature_names.json"
    with open(feature_names_file, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.success(f"✓ 特征名称已保存: {feature_names_file} ({len(feature_names)}个特征)")

    # 保存校准器
    calibrator_file = model_dir / "model" / "calibrator.pkl"
    joblib.dump(calibrator, str(calibrator_file))
    log.success(f"✓ 校准器已保存: {calibrator_file}")

    # 保存元数据
    metadata = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "calibration_method": "isotonic_regression",
        "base_version": "v3 (139 features)",
        "new_features": [
            "price_range_pct",
            "close_vs_ma10_std",
            "days_near_ma10",
            "volume_shrink_ratio",
            "ma10_cross_count",
        ],
        "anti_chasing_config": {"pre_t1_return_max": 20, "pre_t1_volatility_max": 3},
        "description": "v2.4.0反追龙头优化版：基于v3的139特征+5个新增特征，正样本应用T1前约束筛选",
    }

    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 元数据已保存: {metadata_file}")

    # 保存训练指标
    training_dir = model_dir / "training"
    training_dir.mkdir(exist_ok=True)

    metrics_file = training_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 训练指标已保存: {metrics_file}")

    log.info(f"\n模型目录: {model_dir}")


def main():
    log.info("=" * 80)
    log.info(f"训练 {VERSION} 模型 - 基于v4版本完整数据")
    log.info("=" * 80)
    log.info("")
    log.info("数据来源:")
    log.info("  - 基础特征: v3版本 (139个)")
    log.info("  - 新增特征: 5个反追龙头特征")
    log.info("  - 总特征数: 144个")
    log.info("")
    log.info("优化内容:")
    log.info("  1. 正样本T1前约束筛选 (return_34d <= 20%, volatility_34d <= 3%)")
    log.info("  2. 新增price_range_pct, close_vs_ma10_std, days_near_ma10等特征")
    log.info("  3. 时间序列划分（避免未来函数）")
    log.info("")

    # 1. 加载数据
    df = load_training_data()
    if df is None:
        return

    # 2. 获取特征列
    feature_cols = get_feature_columns(df)
    log.info(f"\n使用特征数: {len(feature_cols)}")

    # 3. 时间序列划分
    df_train, df_cal, df_test = time_series_split(df)

    # 4. 准备数据
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_cal = df_cal[feature_cols].values
    y_cal = df_cal["label"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # 处理NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_cal = np.nan_to_num(X_cal, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. 训练模型
    booster = train_model(X_train, y_train, X_cal, y_cal, feature_cols)

    # 6. 概率校准
    calibrator = calibrate_model(booster, X_cal, y_cal, feature_cols)

    # 7. 评估
    metrics = evaluate(booster, calibrator, X_test, y_test, feature_cols)

    # 8. 保存模型
    save_model(booster, calibrator, feature_cols, metrics)

    # 完成
    log.info("")
    log.info("=" * 80)
    log.success(f"✅ {VERSION} 模型训练完成！")
    log.info("=" * 80)
    log.info("")
    log.info("下一步:")
    log.info("  1. 运行预测: python scripts/predict_v240.py --date 20251212")
    log.info("  2. 评估效果: python scripts/evaluate_anti_chasing_effect.py")


if __name__ == "__main__":
    main()
