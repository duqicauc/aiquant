#!/usr/bin/env python3
"""
v3.0.0 XGB-Flat 最终模型训练
全量数据训练 + 时间序列验证早停，保存生产模型
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.multits_flattener import flatten_multits
from src.utils.logger import log

# ============================================================================
# 配置
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data" / "training" / "v298"
MODEL_BASE_DIR = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions"
MODEL_VERSION = "v3.0.0"
SEED = 42

NEG_SAMPLING_RATIO = 2.5  # 负:正

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


# ============================================================================
# 数据加载
# ============================================================================
def load_training_data() -> Tuple[pd.DataFrame, List[str]]:
    """加载 v298 多行训练数据"""
    log.info("=" * 80)
    log.info("加载 v3.0.0 训练数据")
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
        n_samples = df["sample_id"].nunique()
        id_map = {old: global_offset + i for i, old in enumerate(sorted(df["sample_id"].unique()))}
        df["sample_id"] = df["sample_id"].map(id_map)
        global_offset += n_samples
        dfs.append(df)
        log.info(f"  {name}: {n_samples} 样本 / {len(df)} 行")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])

    exclude = {"label", "sample_id", "ts_code", "name", "trade_date", "days_to_t1"}
    feature_cols = sorted([c for c in df_all.columns if c not in exclude])

    log.success(f"数据加载完成: {df_all['sample_id'].nunique()} 样本 / {len(df_all)} 行, {len(feature_cols)} 特征")
    return df_all, feature_cols


# ============================================================================
# 预处理
# ============================================================================
def balance_negatives(df: pd.DataFrame, neg_ratio: float = NEG_SAMPLING_RATIO, seed: int = SEED) -> pd.DataFrame:
    """负样本下采样"""
    pos = df[df["label"] == 1].copy()
    neg = df[df["label"] == 0].copy()

    n_pos = pos["sample_id"].nunique()
    target_neg = int(n_pos * neg_ratio)

    neg_samples = neg["sample_id"].unique()
    if len(neg_samples) > target_neg:
        np.random.seed(seed)
        keep_neg = np.random.choice(neg_samples, target_neg, replace=False)
        neg = neg[neg["sample_id"].isin(keep_neg)].copy()

    balanced = pd.concat([pos, neg], ignore_index=True)
    log.info(f"下采样后: 正 {pos['sample_id'].nunique()} / 负 {neg['sample_id'].nunique()} 样本")
    return balanced


# ============================================================================
# 展平
# ============================================================================
# ============================================================================
# 模型训练
# ============================================================================
def train_xgb_final(X_train, y_train, X_val, y_val, feature_names):
    """XGB-Flat 训练，带 early stopping"""
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
    log.info(f"  最佳迭代轮数: {model.best_iteration}, 最佳验证 AUC: {model.best_score:.4f}")
    return model


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info(f"\n{'='*80}")
    log.info("v3.0.0 XGB-Flat 最终模型训练")
    log.info(f"{'='*80}")

    # 1. 加载数据
    df_all, feature_cols = load_training_data()
    expected_days = sorted(df_all["days_to_t1"].dropna().unique())
    log.info(f"展平后维度: {len(feature_cols)} × {len(expected_days)} = {len(feature_cols) * len(expected_days)}")

    # 2. 负样本下采样
    df_bal = balance_negatives(df_all)

    # 3. 展平
    df_flat = flatten_multits(df_bal, feature_cols, expected_days)
    flat_cols = [c for c in df_flat.columns if c not in {"sample_id", "label", "trade_date", "ts_code"}]
    log.info(f"展平后: {len(df_flat)} 样本 × {len(flat_cols)} 维")

    # 4. 时间序列划分: 按 trade_date 排序，前 80% 训练，后 20% 验证
    df_flat = df_flat.sort_values("trade_date").reset_index(drop=True)
    split_idx = int(len(df_flat) * 0.8)
    train_flat = df_flat.iloc[:split_idx]
    val_flat = df_flat.iloc[split_idx:]
    log.info(f"时间序列划分: 训练 {len(train_flat)} / 验证 {len(val_flat)} 样本")
    log.info(f"  训练集日期范围: {train_flat['trade_date'].min()} ~ {train_flat['trade_date'].max()}")
    log.info(f"  验证集日期范围: {val_flat['trade_date'].min()} ~ {val_flat['trade_date'].max()}")

    X_train = train_flat[flat_cols].values
    y_train = train_flat["label"].values
    X_val = val_flat[flat_cols].values
    y_val = val_flat["label"].values

    # 5. 训练
    log.info("\n训练 XGB-Flat 最终模型...")
    model = train_xgb_final(X_train, y_train, X_val, y_val, flat_cols)

    # 6. 全量重新训练（使用最佳轮数）
    log.info("\n全量数据重新训练最终模型...")
    X_full = df_flat[flat_cols].values
    y_full = df_flat["label"].values
    dfull = xgb.DMatrix(X_full, label=y_full, feature_names=flat_cols)
    final_model = xgb.train(XGB_PARAMS, dfull, num_boost_round=model.best_iteration)

    # 7. 保存
    output_dir = MODEL_BASE_DIR / MODEL_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = output_dir / "xgb_flat_final.json"
    final_model.save_model(str(model_path))
    log.success(f"模型已保存: {model_path}")

    # 保存特征列表
    feature_path = output_dir / "feature_cols.json"
    with open(feature_path, "w") as f:
        json.dump(
            {
                "feature_cols": flat_cols,
                "expected_days": [int(d) for d in expected_days],
                "n_features": len(flat_cols),
                "n_days": len(expected_days),
                "model": "xgb_flat",
                "version": MODEL_VERSION,
                "seed": SEED,
                "best_iteration": int(model.best_iteration),
                "val_auc": float(model.best_score),
                "train_samples": int(len(train_flat)),
                "val_samples": int(len(val_flat)),
                "total_samples": int(len(df_flat)),
            },
            f,
            indent=2,
        )
    log.success(f"元数据已保存: {feature_path}")

    log.info(f"\n{'='*80}")
    log.info("v3.0.0 模型训练完成!")
    log.info(f"{'='*80}")


if __name__ == "__main__":
    main()
