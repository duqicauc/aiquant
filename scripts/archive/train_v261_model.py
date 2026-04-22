#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练v2.6.1模型

特点：
1. 使用v6版本更丰富的样本数据（正样本+负样本）
2. 特征与v2.6.0一致（v5特征集）
3. 硬负样本使用v5版本
4. 其他训练参数与v2.6.0一致

注意：
- 正样本和负样本使用v6版本
- 硬负样本使用v5版本（已复制到v6路径）
- 训练前需先运行 scripts/align_v6_to_v5_features.py 确保特征对齐
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
    """
    加载训练数据（v2.6.1版本：使用v5数据）

    文件路径：
    - 正样本: data/training/processed/feature_data_34d_v5.csv
    - 负样本: data/training/features/negative_feature_data_v2_34d_v5.csv
    - 硬负样本: data/training/features/hard_negative_feature_data_34d_v5.csv

    注意：v6数据生成存在根本性bug（量数据缺失、high/low估算导致price_range_pct恒等于2.0），
    需要完全重新设计v6样本生成流程。暂时使用v5数据训练。
    """
    log.info("加载训练数据（v5版本 - v6数据生成有根本性bug待修复）...")

    # 使用v5数据（v6数据生成有根本性问题需要重新设计）
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    # 检查文件存在
    missing_files = []
    if not pos_file.exists():
        missing_files.append(f"正样本: {pos_file}")
    if not neg_file.exists():
        missing_files.append(f"负样本: {neg_file}")
    if not hard_neg_file.exists():
        missing_files.append(f"硬负样本: {hard_neg_file}")

    if missing_files:
        log.error("以下文件不存在:")
        for f in missing_files:
            log.error(f"  - {f}")
        log.error("请先运行: python scripts/align_v6_to_v5_features.py")
        raise FileNotFoundError("v6版本文件不完整")

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

    # 获取特征列（排除元数据列）
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
    }

    pos_cols = set(df_pos.columns) - exclude_cols
    neg_cols = set(df_neg.columns) - exclude_cols
    hard_cols = set(df_hard_neg.columns) - exclude_cols

    # 使用交集特征（v5版本应该已对齐）
    common_cols = pos_cols & neg_cols & hard_cols

    log.info("\n特征统计:")
    log.info(f"  正样本特征: {len(pos_cols)}")
    log.info(f"  负样本特征: {len(neg_cols)}")
    log.info(f"  硬负样本特征: {len(hard_cols)}")
    log.info(f"  共同特征: {len(common_cols)}")

    # 使用共同特征
    all_cols = list(common_cols) + [c for c in exclude_cols if c in df_pos.columns]

    df_pos = df_pos[[c for c in all_cols if c in df_pos.columns]]
    df_neg = df_neg[[c for c in all_cols if c in df_neg.columns]]
    df_hard_neg = df_hard_neg[[c for c in all_cols if c in df_hard_neg.columns]]

    # 合并数据
    df = pd.concat([df_pos, df_neg, df_hard_neg], ignore_index=True)
    log.info(f"\n合并数据: 正样本 {len(df_pos)} + 负样本 {len(df_neg)} + 硬负样本 {len(df_hard_neg)} = {len(df)}")
    log.success(f"✓ 数据加载完成: {len(df)} 条，特征数: {len(common_cols)} 个")

    return df


def get_feature_columns(df):
    """获取特征列（只保留数值列）"""
    # 基础排除列（v5数据正常，无需额外排除）
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
    ]  # 排除数据泄露特征

    # v2.5.4: 排除未使用的二值特征（已被连续强度特征替代）
    unused_binary_features = [
        "breakout_high_10d",  # 被 breakout_strength_10d 替代
        "breakout_high_20d",  # 被 breakout_strength_20d 替代
        "breakout_ma10",  # 被 price_vs_ma_34d 替代
        "breakout_ma55",  # 被 price_vs_ma_55d 替代
        "high_volume_breakout",  # 被 breakout_volume_strength 替代
        "volume_price_match",  # 被 volume_price_match_sum_10d 替代
        "price_down_vol_up",  # 与 volume_price_match 冗余
    ]
    exclude_cols.extend(unused_binary_features)

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols


def time_series_split(df, test_size=0.2, cal_size=0.15):
    """
    时间序列划分（避免未来函数）

    Args:
        df: 包含 trade_date 或 t1_date 的DataFrame
        test_size: 测试集比例
        cal_size: 校准集比例（从训练集中分出）

    Returns:
        train, cal, test: 三个DataFrame
    """
    # 确定日期列
    date_col = "trade_date" if "trade_date" in df.columns else "t1_date"

    if date_col not in df.columns:
        log.error("数据中缺少 trade_date 或 t1_date 列，无法进行时间序列划分")
        raise ValueError("缺少日期列")

    # 转换为日期类型
    if df[date_col].dtype != "datetime64[ns]":
        # 尝试多种日期格式
        try:
            # 先尝试标准格式 YYYY-MM-DD
            df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        except:
            try:
                # 再尝试 YYYYMMDD 格式
                df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
            except:
                # 最后使用自动解析
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 删除日期为空的记录
    df = df.dropna(subset=[date_col]).copy()

    if len(df) == 0:
        log.error("日期解析后数据为空，请检查日期列格式")
        raise ValueError("日期解析失败")

    # 按日期排序
    df = df.sort_values(date_col).reset_index(drop=True)

    # 计算划分点
    n = len(df)
    test_start = int(n * (1 - test_size))
    cal_start = int(n * (1 - test_size - cal_size))

    # 划分
    train = df.iloc[:cal_start].copy()
    cal = df.iloc[cal_start:test_start].copy()
    test = df.iloc[test_start:].copy()

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
    """训练XGBoost模型"""
    log.info("训练模型...")

    # v2.6.0 参数调优
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],  # 增加PR-AUC评估
        "max_depth": 6,  # 5 -> 6，允许更复杂的特征交互
        "learning_rate": 0.05,  # 保持
        "subsample": 0.8,  # 保持
        "colsample_bytree": 0.6,  # 0.8 -> 0.6，强制使用更多特征
        "min_child_weight": 5,  # 保持
        "gamma": 0.1,  # 保持
        "reg_alpha": 0.3,  # 0.5 -> 0.3，减少L1正则
        "reg_lambda": 0.5,  # 1.0 -> 0.5，减少L2正则
        "scale_pos_weight": 1.5,  # 保持适中，精确优先策略
        "random_state": 42,
        "n_jobs": -1,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # 增加轮数
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,  # 延长早停轮数
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
    """评估模型（完整指标）"""
    log.info("评估模型...")

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)

    # 计算AUC
    try:
        auc = roc_auc_score(y_test, cal_probs)
        log.info(f"  AUC: {auc:.4f}")
    except Exception as e:
        log.warning(f"  无法计算AUC: {e}")
        auc = None

    # 不同阈值下的指标（精确优先策略：关注高阈值）
    log.info("\n不同阈值下的性能（精确优先策略）:")
    log.info(f"{'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'准确率':<10}")
    log.info("-" * 70)

    metrics_dict = {}
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:  # 重点评估高阈值精确率
        y_pred = (cal_probs >= thresh).astype(int)
        if y_pred.sum() > 0:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            acc = (y_test[y_pred == 1] == 1).mean() if y_pred.sum() > 0 else 0
            log.info(f"{thresh:<8.1f} {y_pred.sum():<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {acc:<10.4f}")
            if thresh == 0.5:
                metrics_dict["precision"] = precision
                metrics_dict["recall"] = recall
                metrics_dict["f1"] = f1
        else:
            log.info(f"{thresh:<8.1f} {0:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    # 混淆矩阵（阈值0.5）
    y_pred_05 = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info("\n混淆矩阵（阈值=0.5）:")
    log.info("              预测负  预测正")
    log.info(f"  实际负      {cm[0,0]:<8} {cm[0,1]:<8}")
    log.info(f"  实际正      {cm[1,0]:<8} {cm[1,1]:<8}")

    # 分类报告
    log.info("\n分类报告（阈值=0.5）:")
    try:
        report = classification_report(y_test, y_pred_05, target_names=["负样本", "正样本"], zero_division=0)
        log.info(f"\n{report}")
    except Exception as e:
        log.warning(f"  无法生成分类报告: {e}")

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
    version = "v2.6.1"
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
        "description": "v2.6.1模型 - 使用v6更丰富的样本数据（正样本2倍+），特征与v2.6.0一致，硬负样本使用v5版本",
        "split_method": "time_series",
        "metrics": metrics,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("=" * 80)
    log.info("训练v2.6.1模型（v6更丰富样本+v5特征集+v5硬负样本）")
    log.info("=" * 80)

    # 加载数据
    df = load_training_data()

    # 特征
    feature_cols = get_feature_columns(df)
    log.info(f"特征数: {len(feature_cols)}")

    # 显示新增的风险特征
    risk_features = [f for f in feature_cols if any(k in f for k in ["drawdown", "atr", "days_from_high", "recovery"])]
    if risk_features:
        log.info(f"风险特征数: {len(risk_features)}")
        if len(risk_features) <= 10:
            log.info(f"  {risk_features}")
        else:
            log.info(f"  {risk_features[:10]}... (共{len(risk_features)}个)")

    # 显示233日均线特征
    ma233_features = [f for f in feature_cols if "233" in f]
    if ma233_features:
        log.info(f"233日均线特征数: {len(ma233_features)}")
        if len(ma233_features) <= 10:
            log.info(f"  {ma233_features}")
        else:
            log.info(f"  {ma233_features[:10]}... (共{len(ma233_features)}个)")

    # ⚠️ 修复：使用时间序列划分（避免未来函数）
    log.info("\n" + "=" * 80)
    log.info("数据划分（时间序列方式，避免未来函数）")
    log.info("=" * 80)
    train_df, cal_df, test_df = time_series_split(df, test_size=0.2, cal_size=0.15)

    # 准备特征和标签
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_cal = cal_df[feature_cols].values
    y_cal = cal_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # 处理NaN和无穷值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_cal = np.nan_to_num(X_cal, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"\n数据集: 训练{len(X_train)}, 校准{len(X_cal)}, 测试{len(X_test)}")

    # 训练
    booster = train_model(X_train, y_train, X_cal, y_cal)

    # 校准
    calibrator = calibrate_model(booster, X_cal, y_cal, feature_cols)

    # 评估
    metrics = evaluate(booster, calibrator, X_test, y_test, feature_cols)

    # 保存
    save_model(booster, calibrator, feature_cols, metrics)

    log.success("\n✓ v2.6.1模型训练完成!")


if __name__ == "__main__":
    main()
