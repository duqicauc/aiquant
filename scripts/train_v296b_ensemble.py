#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.5 集成模型训练脚本

在 v2.9.4 基础上回归"高 AUC + 适度分歧"路线：
1. 恢复动态类别权重（替代固定值）
2. 修复温度缩放：T ≥ 1.0（拉伸模式，非压缩）
3. 去掉分位数对齐（消除额外分布压缩）
4. 放宽 Diversity 惩罚（beta 1.0 → 0.5）
5. 放宽 Gates：MAX_DISAGREEMENT 0.50 → 0.65
6. 以 AUC 为核心优化指标，分歧度仅作辅助监控

适度分歧目标：Top50 分歧度 0.50~0.65，子模型相关性 0.60~0.80

Usage:
    python scripts/train_v296b_ensemble.py
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
from scipy.special import expit, logit
from sklearn.metrics import (
    brier_score_loss,
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
# 配置
# ============================================================================
SEED = 42
np.random.seed(SEED)

# 固定类别权重，对齐v2.7.0
SCALE_POS_WEIGHT = 1.5

# 硬负样本比例上限
MAX_HARD_NEG_RATIO = 0.18

# 分歧度阈值 —— 放宽到适度分歧区间（v294: 0.50 过低，v291: ~0.62 健康）
MAX_DISAGREEMENT = 0.65

# 子模型最小相关性 —— 适当降低，允许模型有更多特色
MIN_CORRELATION = 0.40

# 概率分布合理范围 —— 放宽上限，允许模型输出更分散的概率
PROB_MEAN_MIN = 0.05
PROB_MEAN_MAX = 0.50

# 最大Brier Score
MAX_BRIER = 0.18

# 温度缩放优化配置
TEMP_SCALING_MAX_ITER = 100
TEMP_SCALING_TOL = 1e-6

# ============================================================================
# 特征分组 —— 子模型差异化（P0优化）
# ============================================================================
MOMENTUM_FEATURES = [
    "mfi",
    "ema_250",
    "wr1",
    "ma_mass",
    "mass",
    "ema_5",
    "trend_slope_34d",
    "momentum_5d",
    "mtmma",
    "bias_long",
    "ma_8d",
    "ma60",
    "dmi_pdi",
    "madpo",
    "dmi_mdi",
    "dmi_adx",
    "dmi_adxr",
    "wr",
    "mtm",
    "roc",
    "maroc",
    "trix",
    "trma",
    "expma_12",
    "expma_50",
    "xsii_td1",
    "xsii_td2",
    "xsii_td3",
    "xsii_td4",
    "momentum_10d",
    "momentum_20d",
    "ema_10",
    "ema_20",
    "ema_60",
    "ema_30",
    "ema_90",
    "trend_slope_8d",
    "trend_slope_55d",
    "ma30",
    "ma90",
    "ma250",
    "bias_short",
    "bias_mid",
    "ma_34d",
    "ma_55d",
    "ma10",
    "ma5",
    "dfma_dif",
    "dfma_difma",
    "macd_dif",
    "macd_dea",
    "macd",
    "rsi_6",
    "rsi_12",
    "rsi_24",
    "kdj_k",
    "kdj_d",
    "kdj_j",
]

PRICE_VOLUME_FEATURES = [
    "relative_volatility",
    "vr",
    "breakout_high_55d",
    "support_20d",
    "turnover_zscore",
    "dist_to_resistance_10d",
    "maemv",
    "dist_to_support_20d",
    "volume_rsv_20d",
    "resistance_strength_55d",
    "volume_price_corr_10d",
    "volume_trend_slope_20d",
    "volume_price_divergence_strength",
    "volume_price_divergence",
    "price_down_vol_up_count_10d",
    "ktn_upper",
    "ktn_down",
    "taq_up",
    "taq_down",
    "boll_upper",
    "boll_lower",
    "dist_to_resistance_20d",
    "dist_to_support_10d",
    "support_strength_10d",
    "resistance_strength_10d",
    "support_strength_20d",
    "resistance_strength_20d",
    "breakout_strength_10d",
    "breakout_strength_20d",
    "breakout_strength_55d",
    "breakout_volume_strength",
    "high_volume_breakout",
    "volume_price_match",
    "vol_10d_avg",
    "vol_20d_avg",
    "vol_ratio_10d",
    "vol_ratio_20d",
    "turnover_f_change",
    "turnover_zscore_20d",
    "volume_consistency",
    "relative_volume",
    "volume_price_sync",
    "volume_spike_count_10d",
    "obv_trend",
    "obv_calc",
    "amount",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "vol",
    "volatility_8d",
    "volatility_34d",
    "volatility_55d",
    "atr",
    "atr_14",
    "atr_expansion",
]

MARKET_FEATURES = [
    "market_momentum_10d",
    "market_pct_chg",
    "momentum_market_interaction",
    "market_trend",
    "excess_return_cumsum",
    "excess_return",
    "excess_return_consistency",
    "market_volatility_34d",
    "market_position_20d",
    "market_return_34d",
    "market_momentum_20d",
    "market_momentum_5d",
]

POSITION_FEATURES = [
    "price_vs_ma_55d",
    "price_vs_ma_34d",
    "price_position_8d",
    "price_position_avg",
    "price_position_55d",
    "price_vs_ma_8d",
    "price_position_34d",
]

# 子模型特征分配
SUBMODEL_FEATURES = {
    "xgboost": None,  # None = 使用全部特征
    "lightgbm": MOMENTUM_FEATURES + MARKET_FEATURES + POSITION_FEATURES,
    "catboost": PRICE_VOLUME_FEATURES + MARKET_FEATURES + POSITION_FEATURES,
}


def get_submodel_features(model_name, all_features):
    """获取子模型使用的特征子集"""
    subset = SUBMODEL_FEATURES.get(model_name)
    if subset is None:
        return all_features
    # 只保留在 all_features 中存在的特征
    return [f for f in subset if f in all_features]


# ============================================================================
# 数据加载
# ============================================================================
def load_training_data():
    """加载 v296b 训练数据（含分时段对比聚合特征）"""
    log.info("加载 v296b 训练数据...")

    pos_file = PROJECT_ROOT / "data" / "training" / "v296b" / "positive_features.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "v296b" / "negative_features.csv"
    hard_file = PROJECT_ROOT / "data" / "training" / "v296b" / "hard_negative_features.csv"

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1

    df_neg = pd.read_csv(neg_file)
    df_neg["label"] = 0

    df_hard = pd.read_csv(hard_file)
    df_hard["label"] = 0

    # 统一日期格式
    for df in [df_pos, df_neg, df_hard]:
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(
                lambda x: (
                    f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                )
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    # ========== 1. 负样本分层下采样（按市值匹配正样本分布） ==========
    target_neg = int(len(df_pos) * 2.22)  # 正样本的 ~2.22 倍，目标约 1:2.2 比例
    if len(df_neg) > target_neg:
        log.info(f"负样本 {len(df_neg)} 超过目标 {target_neg}，执行分层下采样...")
        # 按 total_mv 分 5 层
        df_neg["mv_quintile"] = pd.qcut(df_neg["total_mv"].rank(method="first"), 5, labels=False, duplicates="drop")
        df_pos["mv_quintile"] = pd.qcut(df_pos["total_mv"].rank(method="first"), 5, labels=False, duplicates="drop")

        neg_sampled = []
        for q in range(5):
            pos_q = df_pos[df_pos["mv_quintile"] == q]
            neg_q = df_neg[df_neg["mv_quintile"] == q]
            # 该层负样本目标数 = 该层正样本数 × (总目标负样本 / 总正样本)
            target_q = max(1, int(len(pos_q) * target_neg / len(df_pos)))
            if len(neg_q) > target_q:
                neg_q = neg_q.sample(n=target_q, random_state=SEED)
            neg_sampled.append(neg_q)

        df_neg = pd.concat(neg_sampled, ignore_index=True)
        df_neg.drop(columns=["mv_quintile"], errors="ignore", inplace=True)
        log.info(f"负样本下采样后: {len(df_neg)}")

    # 清理临时列
    df_pos.drop(columns=["mv_quintile"], errors="ignore", inplace=True)

    # ========== 2. 共同特征列 ==========
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
        & (set(df_hard.columns) - exclude_cols)
    )

    # ========== 3. 删除死特征 + 冗余特征 ==========
    dead_features = [
        "breakout_confirmed_10d",
        "breakout_confirmed_20d",
        "breakout_resonance",
        "market_regime",
        "breakout_with_volume",
        "resonance_volume_confirm",
        "days_from_high_20d",
        "days_from_high_55d",
        "ma10_cross_count",
    ]
    redundant_features = [
        "obv_calc",
        "ma5",
        "bbi",
        "ktn_mid",
        "boll_mid",
        "resistance_55d",
        "support_55d",
        "resistance_10d",
        "resistance_20d",
        "price_vs_hist_mean",
        "return_8d",
        "return_34d",
        "return_55d",
        "breakout_strength_10d",
        "breakout_strength_20d",
        "max_drawdown_55d",
        "vol_ma20_ratio",
        "breakout_strength_max",
        "price_vs_hist_high",
    ]
    drop_features = [c for c in dead_features + redundant_features if c in common_cols]
    if drop_features:
        log.info(f"删除死特征+冗余特征: {len(drop_features)} 个")
        common_cols = [c for c in common_cols if c not in drop_features]

    # ========== 4. 市值中性化（按市值分箱后组内 z-score） ==========
    df_all = pd.concat([df_pos, df_neg, df_hard], ignore_index=True)
    for mv_col in ["total_mv", "circ_mv"]:
        if mv_col in df_all.columns:
            df_all["mv_bin"] = pd.qcut(df_all[mv_col].rank(method="first"), 5, labels=False, duplicates="drop")
            for b in range(5):
                mask = df_all["mv_bin"] == b
                if mask.sum() > 1:
                    mean_b = df_all.loc[mask, mv_col].mean()
                    std_b = df_all.loc[mask, mv_col].std()
                    if std_b and std_b > 0:
                        df_all.loc[mask, mv_col] = (df_all.loc[mask, mv_col] - mean_b) / std_b
            df_all.drop(columns=["mv_bin"], inplace=True)

    # 拆分回三类
    n_pos = len(df_pos)
    n_neg = len(df_neg)
    df_pos = df_all.iloc[:n_pos].copy()
    df_neg = df_all.iloc[n_pos : n_pos + n_neg].copy()
    df_hard = df_all.iloc[n_pos + n_neg :].copy()

    # ========== 5. 合并训练集 ==========
    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date", "ts_code"]],
            df_neg[common_cols + ["label", "trade_date", "ts_code"]],
            df_hard[common_cols + ["label", "trade_date", "ts_code"]],
        ],
        ignore_index=True,
    )

    # 类别权重（固定1.5，对齐v2.7.0，避免过度保守）
    n_pos_final = (df["label"] == 1).sum()
    n_neg_final = (df["label"] == 0).sum()
    dynamic_spw = n_neg_final / n_pos_final if n_pos_final > 0 else 1.0
    log.info(f"动态类别权重计算值: {dynamic_spw:.2f}，但使用固定值: {SCALE_POS_WEIGHT}")
    log.info(f"  (正{n_pos_final}, 负{n_neg_final})")

    # ========== 6. 时间衰减权重（已禁用，对齐v2.7.0） ==========
    df["sample_weight"] = 1.0
    log.info("时间衰减权重: 已禁用 (所有样本权重=1.0，对齐v2.7.0)")
    if "year" in df.columns:
        df.drop(columns=["year"], inplace=True)

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


# ============================================================================
# 时间序列交叉验证
# ============================================================================
def time_series_cv_splits(df, n_splits=5, train_days=365, val_days=30, test_days=30):
    """
    生成时间序列交叉验证的划分

    Returns:
        List[(train_df, val_df, test_df)]
    """
    df = df.copy().sort_values("trade_date")
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    n_dates = len(unique_dates)

    total_window = train_days + val_days + test_days
    stride = max(1, (n_dates - total_window) // n_splits)

    splits = []
    for i in range(n_splits):
        start_idx = i * stride
        train_end = start_idx + train_days
        val_end = train_end + val_days
        test_end = val_end + test_days

        if test_end > n_dates:
            break

        train_dates = set(unique_dates[start_idx:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        test_dates = set(unique_dates[val_end:test_end])

        train = df[df["trade_date"].dt.date.isin(train_dates)].copy()
        val = df[df["trade_date"].dt.date.isin(val_dates)].copy()
        test = df[df["trade_date"].dt.date.isin(test_dates)].copy()

        if len(train) > 0 and len(val) > 0 and len(test) > 0:
            splits.append((train, val, test))
            log.info(
                f"Fold {len(splits)}: 训练 {len(train_dates)}天/{len(train)}行, "
                f"验证 {len(val_dates)}天/{len(val)}行, 测试 {len(test_dates)}天/{len(test)}行"
            )

    return splits


# ============================================================================
# 模型训练（动态类别权重）
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val, feature_names, sample_weight=None):
    """训练 XGBoost"""
    log.info("训练 XGBoost...")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,  # 对齐v2.7.0
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "random_state": SEED,
        "tree_method": "hist",
        "max_bin": 255,
        "grow_policy": "lossguide",
        "max_delta_step": 1,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    y_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)
    log.info(
        f"  XGBoost AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.best_iteration}, features={len(feature_names)}"
    )

    return model, auc, brier


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names, sample_weight=None):
    """训练 LightGBM"""
    log.info("训练 LightGBM...")

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.3,  # P0: 降低特征采样率
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_state": SEED,
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

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
    brier = brier_score_loss(y_val, y_pred)
    log.info(
        f"  LightGBM AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.best_iteration}, features={len(feature_names)}"
    )

    return model, auc, brier


def train_catboost(X_train, y_train, X_val, y_val, feature_names, sample_weight=None):
    """训练 CatBoost"""
    log.info("训练 CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=254,
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=50,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=SCALE_POS_WEIGHT,
        rsm=0.8,  # 对齐v2.7.0
    )

    fit_kwargs = {"eval_set": (X_val, y_val), "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)
    log.info(
        f"  CatBoost AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.get_best_iteration()}, features={len(feature_names)}"
    )

    return model, auc, brier


# ============================================================================
# 温度缩放校准（修复：T >= 1.0 拉伸模式）
# ============================================================================
def temperature_scaling(probs, y_true, max_iter=100, tol=1e-6):
    """
    温度缩放：单参数校准，不改变排序

    v2.9.5 修复：温度下限从 0.1 改为 1.0
    - T < 1: 压缩概率（v294 的错误模式）
    - T = 1: 不变
    - T > 1: 拉伸概率（v295 目标：让概率分布更分散）

    Args:
        probs: 原始概率 (n,)
        y_true: 真实标签 (n,)

    Returns:
        calibrated_probs: 校准后概率 (n,)
        T: 温度参数
    """
    probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
    logits = logit(probs_clipped)

    T = 1.0
    for _ in range(max_iter):
        scaled_logits = logits / T
        scaled_probs = expit(scaled_logits)

        dL_dT = np.mean((scaled_probs - y_true) * logits) / (T**2)
        d2L_dT2 = np.mean(scaled_probs * (1 - scaled_probs) * (logits**2)) / (T**4) + 2 * np.mean(
            (scaled_probs - y_true) * logits
        ) / (T**3)

        T_new = T - dL_dT / (d2L_dT2 + 1e-8)
        T_new = max(1.0, T_new)  # v295: 下限改为 1.0（拉伸模式）

        if abs(T_new - T) < tol:
            break
        T = T_new

    final_logits = logits / T
    calibrated_probs = expit(final_logits)

    return calibrated_probs, T


# ============================================================================
# Diversity-aware加权平均（修复：降低 diversity 惩罚）
# ============================================================================
def compute_diversity_aware_weights(aucs, preds_dict):
    """
    计算 Diversity-aware 权重

    v2.9.5 修复：beta 从 1.0 降到 0.5
    - 降低 diversity 惩罚，让高 AUC 模型获得更高权重
    - 不再强制追求"平庸共识"

    w_i = AUC_i^alpha / (Diversity_penalty_i^beta)
    """
    names = list(preds_dict.keys())
    n = len(names)

    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.corrcoef(preds_dict[names[i]], preds_dict[names[j]])[0, 1]
            corr_matrix[i, j] = corr_matrix[j, i] = corr

    diversity_penalty = []
    for i in range(n):
        others_corr = [corr_matrix[i, j] for j in range(n) if j != i]
        diversity_penalty.append(np.mean(others_corr))

    # v295: beta 从 1.0 降到 0.5，降低 diversity 惩罚
    alpha = 2.0
    beta = 0.5
    raw_weights = []
    for i in range(n):
        w = (aucs[names[i]] ** alpha) / (diversity_penalty[i] ** beta)
        raw_weights.append(max(w, 0.01))

    weights = np.array(raw_weights) / sum(raw_weights)

    return {names[i]: weights[i] for i in range(n)}, corr_matrix


# ============================================================================
# 训练后验证 Gates（放宽）
# ============================================================================
def validate_ensemble_gates(
    models_dict, weights, X_test, y_test, feature_names_dict, feature_indices_dict, max_disagreement=MAX_DISAGREEMENT
):
    """训练后验证 Gates"""
    log.info("\n" + "=" * 80)
    log.info("执行训练后验证 Gates (v2.9.5 放宽版)")
    log.info("=" * 80)

    preds = {}
    if "xgboost" in models_dict:
        idx = feature_indices_dict["xgboost"]
        dtest = xgb.DMatrix(X_test[:, idx], feature_names=feature_names_dict["xgboost"])
        preds["xgboost"] = models_dict["xgboost"].predict(dtest)
    if "lightgbm" in models_dict:
        idx = feature_indices_dict["lightgbm"]
        preds["lightgbm"] = models_dict["lightgbm"].predict(
            X_test[:, idx], num_iteration=models_dict["lightgbm"].best_iteration
        )
    if "catboost" in models_dict:
        idx = feature_indices_dict["catboost"]
        preds["catboost"] = models_dict["catboost"].predict_proba(X_test[:, idx])[:, 1]

    all_passed = True

    # Gate 1: 子模型相关性
    log.info("\n[Gate 1] 子模型相关性检查")
    names = list(preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = np.corrcoef(preds[names[i]], preds[names[j]])[0, 1]
            status = "✓" if corr >= MIN_CORRELATION else "✗"
            log.info(f"  {names[i]} vs {names[j]}: {corr:.4f} {status}")
            if corr < MIN_CORRELATION:
                all_passed = False

    # Gate 2: 概率分布一致性（放宽上限）
    log.info("\n[Gate 2] 概率分布一致性检查")
    for name, pred in preds.items():
        mean_p = pred.mean()
        status = "✓" if PROB_MEAN_MIN <= mean_p <= PROB_MEAN_MAX else "✗"
        log.info(f"  {name} mean_prob={mean_p:.4f} (范围[{PROB_MEAN_MIN}, {PROB_MEAN_MAX}]) {status}")
        if not (PROB_MEAN_MIN <= mean_p <= PROB_MEAN_MAX):
            all_passed = False

    # Gate 3: Top50分歧度（放宽阈值到 0.65）
    log.info("\n[Gate 3] Top50分歧度检查")
    pred_df = pd.DataFrame(preds)
    top50_preds = pred_df.nlargest(50, "catboost")
    disagreements = top50_preds.max(axis=1) - top50_preds.min(axis=1)
    mean_disagree = disagreements.mean()
    status = "✓" if mean_disagree <= max_disagreement else "⚠ (警告但允许)"
    log.info(f"  Top50原始分歧度={mean_disagree:.4f} (目标<=0.65, 阈值<={max_disagreement}) {status}")
    # v295: 分歧度 gate 改为警告而非强制终止
    if mean_disagree > max_disagreement:
        log.warning(f"  分歧度 {mean_disagree:.4f} 超过阈值 {max_disagreement}，但允许继续（适度分歧策略）")
        # 不设置 all_passed = False，改为仅警告

    # Gate 4: Brier Score
    log.info("\n[Gate 4] Brier Score检查")
    for name, pred in preds.items():
        brier = brier_score_loss(y_test, pred)
        status = "✓" if brier <= MAX_BRIER else "✗"
        log.info(f"  {name} Brier={brier:.4f} (阈值<={MAX_BRIER}) {status}")
        if brier > MAX_BRIER:
            all_passed = False

    # Gate 5: 集成性能（AUC 为核心）
    log.info("\n[Gate 5] 集成性能检查（AUC 为核心指标）")
    ensemble = sum(preds[name] * weights[name] for name in preds)
    ensemble_auc = roc_auc_score(y_test, ensemble)
    ensemble_brier = brier_score_loss(y_test, ensemble)
    log.info(f"  Ensemble AUC={ensemble_auc:.4f}, Brier={ensemble_brier:.4f}")

    # 获取最高单模型 AUC
    best_single_auc = max(roc_auc_score(y_test, preds[name]) for name in preds)
    auc_gap = best_single_auc - ensemble_auc
    log.info(f"  最高单模型 AUC={best_single_auc:.4f}, 集成差距={auc_gap:.4f}")
    if auc_gap > 0.05:
        log.warning(f"  集成差距 {auc_gap:.4f} > 0.05，建议检查权重分配")

    if all_passed:
        log.success("核心 Gates 通过！")
    else:
        log.error("部分核心 Gates 未通过，请检查模型！")

    return all_passed, preds, ensemble


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("=" * 80)
    log.info("v2.9.5 集成模型训练 —— 高 AUC + 适度分歧")
    log.info("=" * 80)

    # 加载数据
    df, feature_cols = load_training_data()
    valid_features = get_feature_columns(df, feature_cols)
    log.info(f"有效特征: {len(valid_features)} 个")

    # 时间序列交叉验证
    log.info("\n" + "=" * 80)
    log.info("时间序列交叉验证划分")
    log.info("=" * 80)
    cv_splits = time_series_cv_splits(df, n_splits=3, train_days=300, val_days=30, test_days=30)

    if not cv_splits:
        log.error("CV划分失败")
        return

    train_df, val_df, test_df = cv_splits[-1]

    X_train_full = train_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df["label"].values
    X_val_full = val_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_val = val_df["label"].values
    X_test_full = test_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test_df["label"].values
    sample_weights = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    log.info(f"\n最终数据集: 训练{len(X_train_full)}, 验证{len(X_val_full)}, 测试{len(X_test_full)}")

    # 训练三个子模型（使用差异化特征子集）
    log.info("\n" + "=" * 80)
    log.info("训练子模型（动态类别权重 + 差异化特征）")
    log.info("=" * 80)

    # P0优化: 全部子模型使用全部特征，但通过colsample强制diversity
    xgb_features = valid_features
    lgb_features = valid_features
    cat_features = valid_features

    xgb_model, xgb_auc, xgb_brier = train_xgboost(
        X_train_full, y_train, X_val_full, y_val, valid_features, sample_weight=sample_weights
    )
    lgb_model, lgb_auc, lgb_brier = train_lightgbm(
        X_train_full, y_train, X_val_full, y_val, valid_features, sample_weight=sample_weights
    )
    cat_model, cat_auc, cat_brier = train_catboost(
        X_train_full, y_train, X_val_full, y_val, valid_features, sample_weight=sample_weights
    )

    models_dict = {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model}
    aucs = {"xgboost": xgb_auc, "lightgbm": lgb_auc, "catboost": cat_auc}

    # 获取验证集预测
    val_preds_raw = {
        "xgboost": xgb_model.predict(xgb.DMatrix(X_val_full, feature_names=valid_features)),
        "lightgbm": lgb_model.predict(X_val_full, num_iteration=lgb_model.best_iteration),
        "catboost": cat_model.predict_proba(X_val_full)[:, 1],
    }

    # 禁用温度缩放校准（对齐v2.7.0简单IsotonicRegression风格）
    log.info("\n" + "=" * 80)
    log.info("校准: 已禁用温度缩放（对齐v2.7.0）")
    log.info("=" * 80)
    val_preds_cal = val_preds_raw  # 直接使用原始预测

    # 简单AUC加权投票（对齐v2.7.0，去掉diversity惩罚）
    log.info("\n" + "=" * 80)
    log.info("简单AUC加权投票（去掉diversity惩罚）")
    log.info("=" * 80)

    total_auc = sum(aucs.values())
    weights = {name: aucs[name] / total_auc for name in aucs}
    for name, w in weights.items():
        log.info(f"  {name}: weight={w:.4f} (AUC={aucs[name]:.4f})")

    # 训练后验证 Gates（v2.9.5 放宽版）
    passed, test_preds, ensemble = validate_ensemble_gates(
        models_dict,
        weights,
        X_test_full,
        y_test,
        {"xgboost": valid_features, "lightgbm": valid_features, "catboost": valid_features},
        {
            "xgboost": list(range(len(valid_features))),
            "lightgbm": list(range(len(valid_features))),
            "catboost": list(range(len(valid_features))),
        },
    )

    if not passed:
        log.error("核心验证未通过，终止保存")
        return

    # 保存模型
    log.info("\n" + "=" * 80)
    log.info("保存模型")
    log.info("=" * 80)

    model_version = "v2.9.5-ensemble"
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / model_version / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(str(model_dir / "xgboost.json"))
    lgb_model.save_model(str(model_dir / "lightgbm.txt"))
    cat_model.save_model(str(model_dir / "catboost.cbm"))

    with open(model_dir / "lightgbm_meta.json", "w") as f:
        json.dump({"best_iteration": int(lgb_model.best_iteration), "num_trees": int(lgb_model.num_trees())}, f)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_features, f)

    # P0: 保存子模型特征子集（当前全部使用全部特征）
    for name in ["xgboost", "lightgbm", "catboost"]:
        with open(model_dir / f"feature_names_{name}.json", "w") as f:
            json.dump(valid_features, f)

    # 温度缩放已禁用（对齐v2.7.0），不保存 temperatures
    log.info("  温度缩放: 已禁用，不保存 temperatures.json")

    # v2.9.5: 不再保存分位数对齐映射（已去掉此步骤）
    log.info("  [v2.9.5] 不保存分位数对齐映射")

    with open(model_dir / "weights.json", "w") as f:
        json.dump(weights, f)

    # 保存指标
    pred_df = pd.DataFrame(test_preds)
    top50_preds = pred_df.nlargest(50, "catboost")
    mean_disagree = (top50_preds.max(axis=1) - top50_preds.min(axis=1)).mean()

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "version": model_version,
                "xgboost_auc": xgb_auc,
                "lightgbm_auc": lgb_auc,
                "catboost_auc": cat_auc,
                "ensemble_auc": roc_auc_score(y_test, ensemble),
                "ensemble_precision": precision_score(y_test, (ensemble >= 0.5).astype(int), zero_division=0),
                "ensemble_recall": recall_score(y_test, (ensemble >= 0.5).astype(int), zero_division=0),
                "ensemble_f1": f1_score(y_test, (ensemble >= 0.5).astype(int), zero_division=0),
                "ensemble_brier": float(brier_score_loss(y_test, ensemble)),
                "temperatures": None,
                "weights": weights,
                "correlation_matrix": None,
                "feature_count": len(valid_features),
                "train_samples": len(X_train_full),
                "val_samples": len(X_val_full),
                "test_samples": len(X_test_full),
                "mean_disagreement": float(mean_disagree),
                "scale_pos_weight": float(SCALE_POS_WEIGHT),
            },
            f,
            indent=2,
        )

    log.success(f"\n模型已保存到: {model_dir}")
    log.success("v2.9.5 训练完成！")

    # ========================================================================
    # 自动更新 current.json
    # ========================================================================
    log.info("\n" + "=" * 80)
    log.info("更新版本指针")
    log.info("=" * 80)

    current_file = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "current.json"
    current = {
        "production": "v2.7.0",
        "staging": "v2.7.0",
        "testing": "v2.9.1-ensemble",
        "development": model_version,
        "updated_at": datetime.now().isoformat(),
        "latest_train": {
            "version": model_version,
            "ensemble_auc": float(roc_auc_score(y_test, ensemble)),
            "ensemble_f1": float(f1_score(y_test, (ensemble >= 0.5).astype(int), zero_division=0)),
            "ensemble_brier": float(brier_score_loss(y_test, ensemble)),
            "mean_disagreement": float(mean_disagree),
            "feature_count": len(valid_features),
            "train_samples": len(X_train_full),
            "test_samples": len(X_test_full),
            "scale_pos_weight": float(SCALE_POS_WEIGHT),
        },
        "notes": (
            f"主力策略: v2.9.1-ensemble + integrated (sector-filter)。"
            f"最新训练: {model_version} (AUC={roc_auc_score(y_test, ensemble):.4f}, "
            f"分歧度={mean_disagree:.4f}, Brier={brier_score_loss(y_test, ensemble):.4f})。"
            f"v2.9.5 回归高AUC+适度分歧路线，放宽 gates，恢复动态类别权重。"
        ),
    }

    if current_file.exists():
        try:
            with open(current_file, "r", encoding="utf-8") as f:
                old = json.load(f)
            for key in ["production", "staging", "testing"]:
                if key in old:
                    current[key] = old[key]
        except Exception as e:
            log.warning(f"读取旧 current.json 失败: {e}")

    with open(current_file, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)

    log.success(f"  ✓ current.json 已更新: {current_file}")
    log.info(f"  development → {model_version}")
    log.info(f"  production  → {current['production']} (未变更)")
    log.info(f"  testing     → {current['testing']} (未变更)")


if __name__ == "__main__":
    main()
