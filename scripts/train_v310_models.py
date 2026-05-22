#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3.1.0 双模型训练脚本

支持 BreakoutScorer 和 BounceScorer 的训练：
1. 从样本CSV加载训练/验证/测试数据
2. 使用对应FeatureExtractor提取特征
3. 多行时间序列展平为单样本宽表
4. 时间序列交叉验证训练XGBoost
5. Isotonic Regression概率校准
6. 特征重要性分析（Top5占比检查）

Usage:
    python scripts/train_v310_models.py --model breakout [--skip_feature_extraction]
    python scripts/train_v310_models.py --model bounce [--skip_feature_extraction]
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# 模型配置
# ============================================================================
MODEL_CONFIG = {
    "breakout": {
        "name": "BreakoutScorer",
        "lookback_days": 34,
        "sample_dir": PROJECT_ROOT / "data" / "training" / "samples" / "v310" / "breakout",
        "model_dir": PROJECT_ROOT / "data" / "models" / "v310" / "breakout",
        "auc_threshold": 0.90,
        "sharpe_threshold": 0.8,
        "top5_gain_threshold": 15.0,
    },
    "bounce": {
        "name": "BounceScorer",
        "lookback_days": 40,  # v3.1.0重构: 64→40, 聚焦近期超跌
        "sample_dir": PROJECT_ROOT / "data" / "training" / "samples" / "v310" / "bounce",
        "model_dir": PROJECT_ROOT / "data" / "models" / "v310" / "bounce",
        "auc_threshold": 0.85,  # v3.1.0重构: 0.88→0.85, 现实预期
        "sharpe_threshold": 0.6,
        "top5_gain_threshold": 15.0,
    },
}

# XGBoost超参数（Breakout默认）
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr", "logloss"],
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "colsample_bylevel": 0.7,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

# Bounce v3.1.0重构超参数（更激进，避免欠拟合）
BOUNCE_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr", "logloss"],
    "max_depth": 6,  # 5→6
    "learning_rate": 0.05,  # 0.03→0.05
    "subsample": 0.9,  # 0.8→0.9
    "colsample_bytree": 0.5,  # 0.3→0.5, 让专属特征有机会被选中
    "colsample_bylevel": 0.7,
    "min_child_weight": 5,  # 10→5
    "gamma": 0.1,
    "reg_alpha": 0.1,  # 0.5→0.1
    "reg_lambda": 1.0,  # 2.0→1.0
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

SEED = 42
np.random.seed(SEED)


# ============================================================================
# 数据加载
# ============================================================================
def load_samples(sample_dir: Path, split: str) -> Optional[pd.DataFrame]:
    """加载指定划分（train/val/test）的样本"""
    split_dir = sample_dir / split
    if not split_dir.exists():
        log.warning(f"目录不存在: {split_dir}")
        return None

    files = {
        "positive": split_dir / "positive.csv",
        "negative": split_dir / "negative.csv",
        "hard_negative": split_dir / "hard_negative.csv",
    }

    dfs = []
    for sample_type, filepath in files.items():
        if not filepath.exists():
            log.warning(f"文件不存在: {filepath}")
            continue
        df = pd.read_csv(filepath)
        if df.empty:
            continue
        df["label"] = 1 if sample_type == "positive" else 0
        df["sample_type"] = sample_type
        dfs.append(df)
        log.info(f"  [{split}/{sample_type}]: {len(df)} 个样本")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"  [{split}] 合计: {len(combined)} 个样本")
    return combined


# ============================================================================
# 特征提取
# ============================================================================
def extract_features_for_model(model_name: str, samples_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """为样本提取特征"""
    if samples_df.empty:
        return pd.DataFrame()

    log.info(f"\n[{model_name}] 特征提取: {len(samples_df)} 个样本, lookback={lookback_days}天")

    if model_name == "breakout":
        from src.features.breakout_feature_extractor import BreakoutFeatureExtractor

        extractor = BreakoutFeatureExtractor(use_cache=True)
    else:
        from src.features.bounce_feature_extractor import BounceFeatureExtractor

        extractor = BounceFeatureExtractor(use_cache=True)

    # UnifiedFeatureExtractor.extract_for_samples 返回多行DataFrame
    df_features = extractor.extract_for_samples(
        samples_df=samples_df,
        lookback_days=lookback_days,
        batch_size=50,
    )

    log.success(f"  特征提取完成: {len(df_features)} 行")
    return df_features


# ============================================================================
# 特征展平
# ============================================================================
def flatten_features(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """将多行时间序列展平为单样本宽表"""
    from src.features.multits_flattener import flatten_multits

    if df_features.empty:
        return pd.DataFrame(), []

    # 确定特征列（排除metadata列）
    exclude_cols = {
        "sample_id",
        "ts_code",
        "name",
        "t1_date",
        "trade_date",
        "label",
        "days_to_t1",
        "list_date",
        "pattern_type",
        "sample_type",
        "quarter",
    }
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    log.info(f"\n展平特征: {len(feature_cols)} 个原始特征")

    # 展平
    df_flat = flatten_multits(df_features, feature_cols)

    if df_flat.empty:
        return pd.DataFrame(), []

    # 获取展平后的特征列
    flat_feature_cols = [c for c in df_flat.columns if c not in {"sample_id", "ts_code", "trade_date", "label"}]

    log.success(f"  展平完成: {len(df_flat)} 样本 × {len(flat_feature_cols)} 维")
    return df_flat, flat_feature_cols


# ============================================================================
# 数据准备
# ============================================================================
def prepare_data(df_flat: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间划分训练/验证/测试
    样本已按时间划分（train≤2022, val=2023, test=2024-2026），直接按trade_date划分
    """
    df = df_flat.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), errors="coerce")

    train_end = pd.to_datetime("20221231", format="%Y%m%d")
    val_end = pd.to_datetime("20231231", format="%Y%m%d")

    train_df = df[df["trade_date"] <= train_end].copy()
    val_df = df[(df["trade_date"] > train_end) & (df["trade_date"] <= val_end)].copy()
    test_df = df[df["trade_date"] > val_end].copy()

    log.info("\n时间划分:")
    train_min = train_df["trade_date"].min().date()
    train_max = train_df["trade_date"].max().date()
    log.info(f"  训练集: {len(train_df)} 样本 ({train_min} ~ {train_max})")
    log.info(
        f"  验证集: {len(val_df)} 样本 ({val_df['trade_date'].min().date()} ~ {val_df['trade_date'].max().date()})"
    )
    log.info(
        f"  测试集: {len(test_df)} 样本 ({test_df['trade_date'].min().date()} ~ {test_df['trade_date'].max().date()})"
    )

    return train_df, val_df, test_df


# ============================================================================
# 模型训练
# ============================================================================
def train_with_cv(X_train, y_train, X_val, y_val, feature_names, model_name: str):
    """训练XGBoost，带时间序列交叉验证"""
    params = BOUNCE_XGB_PARAMS if model_name.lower() == "bounce" else XGB_PARAMS
    log.info(f"\n[{model_name}] XGBoost训练...")
    log.info(f"  参数: {params}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    log.success(f"  训练完成, best_iteration: {booster.best_iteration}")
    return booster


# ============================================================================
# 概率校准
# ============================================================================
def calibrate_probabilities(booster, X_cal, y_cal, feature_names):
    """Isotonic Regression概率校准"""
    log.info("\n概率校准 (Isotonic Regression)...")

    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_probs = booster.predict(dcal)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    cal_probs = calibrator.predict(raw_probs)
    log.info(f"  校准前: mean={raw_probs.mean():.4f}, std={raw_probs.std():.4f}")
    log.info(f"  校准后: mean={cal_probs.mean():.4f}, std={cal_probs.std():.4f}")

    log.success("  校准完成")
    return calibrator


# ============================================================================
# 评估
# ============================================================================
def evaluate_model(booster, calibrator, X_test, y_test, feature_names) -> Dict:
    """评估模型性能"""
    log.info("\n模型评估...")

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
    metrics_dict = {}
    log.info(f"\n  {'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    log.info("  " + "-" * 55)
    for thresh in [0.7, 0.6, 0.5, 0.4, 0.3]:
        y_pred = (cal_probs >= thresh).astype(int)
        if y_pred.sum() > 0:
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            log.info(f"  {thresh:<8.1f} {y_pred.sum():<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")
            if thresh == 0.5:
                metrics_dict = {"precision": p, "recall": r, "f1": f1}

    # 混淆矩阵
    y_pred_05 = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info("\n  混淆矩阵（阈值=0.5）:")
    log.info(f"    预测负: {cm[0,0]:>6} | 预测正: {cm[0,1]:>6}")
    log.info(f"    实际负: {cm[0,0]:>6} | 实际正: {cm[1,1]:>6}")

    return {
        "auc_raw": float(auc_raw),
        "auc_cal": float(auc_cal),
        "brier": float(brier),
        **metrics_dict,
        "test_samples": len(X_test),
        "positive_samples": int(y_test.sum()),
    }


# ============================================================================
# 特征重要性分析
# ============================================================================
def analyze_feature_importance(booster, feature_names, top5_threshold: float = 15.0):
    """分析特征重要性，检查Top5占比"""
    log.info("\n特征重要性分析...")

    importance = booster.get_score(importance_type="gain")
    total_gain = sum(importance.values())

    if total_gain == 0:
        log.warning("  无法获取特征重要性")
        return [], 0.0

    ranked = []
    for fname in feature_names:
        gain = importance.get(fname, 0)
        ranked.append(
            {
                "feature": fname,
                "gain": gain,
                "gain_pct": gain / total_gain * 100,
            }
        )

    ranked = sorted(ranked, key=lambda x: x["gain"], reverse=True)

    # Top5占比
    top5_gain = sum(x["gain_pct"] for x in ranked[:5])
    log.info(f"  Top5特征占比: {top5_gain:.2f}% (门槛: ≤{top5_threshold}%)")

    log.info("\n  Top 15 特征:")
    for i, item in enumerate(ranked[:15], 1):
        marker = " ⚠️" if i <= 1 and item["gain_pct"] > 30 else ""
        log.info(f"  {i:2d}. {item['feature']:<40} {item['gain_pct']:>6.2f}%{marker}")

    # 检查门槛
    if top5_gain > top5_threshold:
        log.warning(
            f"  ⚠️ Top5占比过高 ({top5_gain:.1f}% > {top5_threshold}%)，建议增加colsample_bytree或删除高垄断特征"
        )
    else:
        log.success(f"  ✅ Top5占比达标 ({top5_gain:.1f}% ≤ {top5_threshold}%)")

    return ranked, top5_gain


# ============================================================================
# 保存模型
# ============================================================================
def save_model_artifacts(
    booster, calibrator, feature_names, metrics, importance_ranked, model_dir: Path, model_name: str, config: Dict
):
    """保存模型和相关文件"""
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = model_dir / f"{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    model_path = version_dir / "model.json"
    booster.save_model(str(model_path))

    # 校准器
    calibrator_path = version_dir / "calibrator.pkl"
    joblib.dump(calibrator, str(calibrator_path))

    # 特征名
    with open(version_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 特征重要性
    with open(version_dir / "feature_importance.json", "w") as f:
        json.dump(importance_ranked[:50], f, indent=2, ensure_ascii=False)

    # 元数据
    xgb_params = BOUNCE_XGB_PARAMS if model_name.lower() == "bounce" else XGB_PARAMS
    metadata = {
        "model_name": model_name,
        "version": "v310",
        "created_at": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "xgboost_params": xgb_params,
        "calibration_method": "isotonic_regression",
        "metrics": metrics,
        "top10_features": [x["feature"] for x in importance_ranked[:10]],
        "top5_gain_pct": sum(x["gain_pct"] for x in importance_ranked[:5]),
    }

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success(f"\n✓ 模型已保存到 {version_dir}")
    return version_dir


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="v3.1.0 双模型训练")
    parser.add_argument("--model", choices=["breakout", "bounce"], required=True, help="训练哪个模型")
    parser.add_argument("--skip_feature_extraction", action="store_true", help="跳过特征提取（假设已有特征CSV）")
    parser.add_argument("--feature_cache", type=str, default=None, help="特征缓存路径（跳过提取时读取）")
    args = parser.parse_args()

    config = MODEL_CONFIG[args.model]
    model_name = config["name"]

    log.info("=" * 80)
    log.info(f"v3.1.0 {model_name} 训练启动")
    log.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. 加载样本
    # ------------------------------------------------------------------
    log.info("\n[1/7] 加载样本...")
    train_samples = load_samples(config["sample_dir"], "train")
    val_samples = load_samples(config["sample_dir"], "val")
    test_samples = load_samples(config["sample_dir"], "test")

    if train_samples is None or train_samples.empty:
        log.error("训练样本为空，请先运行样本生成脚本")
        return

    all_samples = pd.concat([s for s in [train_samples, val_samples, test_samples] if s is not None], ignore_index=True)
    log.info(f"  总样本: {len(all_samples)} 个")

    # ------------------------------------------------------------------
    # 2. 特征提取
    # ------------------------------------------------------------------
    if not args.skip_feature_extraction:
        log.info("\n[2/7] 特征提取（耗时步骤）...")
        df_features = extract_features_for_model(args.model, all_samples, config["lookback_days"])

        if df_features.empty:
            log.error("特征提取失败")
            return

        # 保存特征缓存
        cache_path = config["model_dir"] / "features_cache.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(cache_path, index=False)
        log.info(f"  特征缓存: {cache_path}")
    else:
        cache_path = args.feature_cache or config["model_dir"] / "features_cache.csv"
        log.info(f"\n[2/7] 从缓存加载特征: {cache_path}")
        if not Path(cache_path).exists():
            log.error(f"特征缓存不存在: {cache_path}")
            return
        df_features = pd.read_csv(cache_path)

    # ------------------------------------------------------------------
    # 3. 特征展平
    # ------------------------------------------------------------------
    log.info("\n[3/7] 特征展平...")
    df_flat, feature_cols = flatten_features(df_features)

    if df_flat.empty:
        log.error("特征展平失败")
        return

    # ------------------------------------------------------------------
    # 4. 数据划分
    # ------------------------------------------------------------------
    log.info("\n[4/7] 数据划分...")
    train_df, val_df, test_df = prepare_data(df_flat, feature_cols)

    # 准备X/y
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test_df["label"].values

    log.info(f"\n  训练: {len(X_train)} 样本, 正例={y_train.sum()}")
    log.info(f"  验证: {len(X_val)} 样本, 正例={y_val.sum()}")
    log.info(f"  测试: {len(X_test)} 样本, 正例={y_test.sum()}")

    # ------------------------------------------------------------------
    # 5. 训练
    # ------------------------------------------------------------------
    log.info("\n[5/7] 模型训练...")
    booster = train_with_cv(X_train, y_train, X_val, y_val, feature_cols, model_name)

    # ------------------------------------------------------------------
    # 6. 概率校准（使用验证集）
    # ------------------------------------------------------------------
    log.info("\n[6/7] 概率校准...")
    calibrator = calibrate_probabilities(booster, X_val, y_val, feature_cols)

    # ------------------------------------------------------------------
    # 7. 评估
    # ------------------------------------------------------------------
    log.info("\n[7/7] 评估...")
    metrics = evaluate_model(booster, calibrator, X_test, y_test, feature_cols)

    # 特征重要性
    importance_ranked, top5_gain = analyze_feature_importance(
        booster, feature_cols, top5_threshold=config["top5_gain_threshold"]
    )

    # 保存
    version_dir = save_model_artifacts(
        booster, calibrator, feature_cols, metrics, importance_ranked, config["model_dir"], model_name, config
    )

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 80)
    log.info(f"{model_name} 训练完成")
    log.info("=" * 80)
    log.info(f"  特征维度:    {len(feature_cols)}")
    log.info(f"  训练样本:    {len(X_train)}")
    log.info(f"  测试样本:    {metrics['test_samples']}")
    log.info(f"  AUC (原始):  {metrics['auc_raw']:.4f}")
    log.info(f"  AUC (校准):  {metrics['auc_cal']:.4f}")
    log.info(f"  Brier:       {metrics['brier']:.4f}")
    log.info(f"  Precision:   {metrics.get('precision', 0):.4f}")
    log.info(f"  Recall:      {metrics.get('recall', 0):.4f}")
    log.info(f"  F1:          {metrics.get('f1', 0):.4f}")
    log.info(f"  Top5占比:    {top5_gain:.2f}%")
    log.info(f"  模型路径:    {version_dir}")
    log.info("=" * 80)

    # 门槛检查
    passed = True
    if metrics["auc_cal"] < config["auc_threshold"]:
        log.warning(f"⚠️ AUC未达标: {metrics['auc_cal']:.4f} < {config['auc_threshold']}")
        passed = False
    if top5_gain > config["top5_gain_threshold"]:
        log.warning(f"⚠️ Top5占比未达标: {top5_gain:.1f}% > {config['top5_gain_threshold']}%")
        passed = False

    if passed:
        log.success(f"✅ {model_name} 通过所有验收门槛！")
    else:
        log.warning(f"⚠️ {model_name} 部分门槛未通过，需调优")


if __name__ == "__main__":
    main()
