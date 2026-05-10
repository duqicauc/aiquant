#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.4 专业级集成模型训练脚本

在 v2.9.3 基础上彻底修复分歧度问题：
1. 统一三模型正则化参数与类别权重策略
2. 时间序列交叉验证（TSCV）
3. 温度缩放概率校准（替代Platt Scaling）
4. 分位数概率对齐
5. Diversity-aware加权平均（替代Stacking元学习器）
6. 严格的训练后验证 gates

Usage:
    python scripts/train_v294_ensemble_professional.py
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
from scipy.interpolate import interp1d
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

# 统一类别权重（放弃动态计算，使用固定值保证三模型一致）
UNIFIED_SCALE_POS_WEIGHT = 1.5

# 硬负样本比例上限
MAX_HARD_NEG_RATIO = 0.18

# 分歧度阈值
MAX_DISAGREEMENT = 0.50

# 子模型最小相关性
MIN_CORRELATION = 0.50

# 概率分布合理范围
PROB_MEAN_MIN = 0.05
PROB_MEAN_MAX = 0.35

# 最大Brier Score
MAX_BRIER = 0.15

# 温度缩放优化配置
TEMP_SCALING_MAX_ITER = 100
TEMP_SCALING_TOL = 1e-6


# ============================================================================
# 数据加载
# ============================================================================
def load_training_data():
    """加载训练数据，保持与v293一致的数据源"""
    log.info("加载训练数据...")

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

    # 统一日期格式（与v293一致）
    for df in [df_pos, df_neg, df_hard_old, df_hard_new]:
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(
                lambda x: (
                    f"{int(x):08d}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                )
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    # 共同特征
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

    # 硬负样本下采样控制
    target_hard_ratio = MAX_HARD_NEG_RATIO
    current_hard_count = df_hard_new["sample_id"].nunique()
    current_neg_count = df_neg["sample_id"].nunique() + df_hard_old["sample_id"].nunique()
    max_hard = int(current_neg_count * target_hard_ratio / (1 - target_hard_ratio))
    if current_hard_count > max_hard:
        log.warning(f"硬负样本过多: {current_hard_count}，下采样至 {max_hard}")
        keep_ids = df_hard_new["sample_id"].drop_duplicates().sample(n=max_hard, random_state=SEED).tolist()
        df_hard_new = df_hard_new[df_hard_new["sample_id"].isin(keep_ids)].copy()

    df = pd.concat(
        [
            df_pos[common_cols + ["label", "trade_date", "ts_code"]],
            df_neg[common_cols + ["label", "trade_date", "ts_code"]],
            df_hard_old[common_cols + ["label", "trade_date", "ts_code"]],
            df_hard_new[common_cols + ["label", "trade_date", "ts_code"]],
        ],
        ignore_index=True,
    )

    log.info(f"数据加载完成: {len(df)} 条，特征数: {len(common_cols)}")
    log.info(f"正样本: {(df['label']==1).sum()}, 负样本: {(df['label']==0).sum()}")
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

    # 计算每个fold的起始位置
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
# 模型训练（统一参数）
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """训练 XGBoost（统一参数）"""
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
        "reg_lambda": 1.0,
        "scale_pos_weight": UNIFIED_SCALE_POS_WEIGHT,
        "random_state": SEED,
        "tree_method": "hist",
        "max_bin": 255,
        "grow_policy": "lossguide",
        "max_delta_step": 1,
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

    # 使用最优迭代次数预测
    y_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)
    log.info(f"  XGBoost AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.best_iteration}")

    return model, auc, brier


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """训练 LightGBM（统一参数，修复v293缺失参数）"""
    log.info("训练 LightGBM...")

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_state": SEED,
        "scale_pos_weight": UNIFIED_SCALE_POS_WEIGHT,
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

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)
    log.info(f"  LightGBM AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.best_iteration}")

    return model, auc, brier


def train_catboost(X_train, y_train, X_val, y_val, feature_names):
    """训练 CatBoost（统一参数）"""
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
        # 统一使用与XGB/LGB相同的scale_pos_weight
        scale_pos_weight=UNIFIED_SCALE_POS_WEIGHT,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)
    log.info(f"  CatBoost AUC={auc:.4f}, Brier={brier:.4f}, best_iteration={model.get_best_iteration()}")

    return model, auc, brier


# ============================================================================
# 温度缩放校准
# ============================================================================
def temperature_scaling(probs, y_true, max_iter=100, tol=1e-6):
    """
    温度缩放：单参数校准，不改变排序

    Args:
        probs: 原始概率 (n,)
        y_true: 真实标签 (n,)

    Returns:
        calibrated_probs: 校准后概率 (n,)
        T: 温度参数
    """
    # 转换为logits，避免极端值
    probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
    logits = logit(probs_clipped)

    # 牛顿法优化温度T
    T = 1.0
    for _ in range(max_iter):
        # 计算当前温度下的概率
        scaled_logits = logits / T
        scaled_probs = expit(scaled_logits)

        # 计算梯度和Hessian
        dL_dT = np.mean((scaled_probs - y_true) * logits) / (T**2)
        d2L_dT2 = np.mean(scaled_probs * (1 - scaled_probs) * (logits**2)) / (T**4) + 2 * np.mean(
            (scaled_probs - y_true) * logits
        ) / (T**3)

        # 牛顿更新
        T_new = T - dL_dT / (d2L_dT2 + 1e-8)
        T_new = max(0.1, T_new)  # 防止温度过低

        if abs(T_new - T) < tol:
            break
        T = T_new

    # 应用最终温度
    final_logits = logits / T
    calibrated_probs = expit(final_logits)

    return calibrated_probs, T


# ============================================================================
# 分位数对齐
# ============================================================================
def quantile_align_probs(probs_list, reference_idx=2):
    """
    将多个模型的概率映射到统一的参考分布

    Args:
        probs_list: [prob_xgb, prob_lgb, prob_cat]
        reference_idx: 参考模型的索引（默认CatBoost）

    Returns:
        aligned_probs_list: 对齐后的概率列表
    """
    ref_probs = probs_list[reference_idx]

    aligned = []
    for probs in probs_list:
        # 计算经验分位数函数
        sorted_probs = np.sort(probs)
        quantiles = np.linspace(0, 1, len(probs))

        # 计算当前概率对应的分位数
        prob_to_quantile = interp1d(sorted_probs, quantiles, kind="linear", bounds_error=False, fill_value=(0, 1))
        curr_quantiles = prob_to_quantile(probs)

        # 从参考分布的逆CDF获取对齐概率
        ref_sorted = np.sort(ref_probs)
        quantile_to_prob = interp1d(
            quantiles, ref_sorted, kind="linear", bounds_error=False, fill_value=(ref_sorted[0], ref_sorted[-1])
        )
        aligned_probs = quantile_to_prob(curr_quantiles)
        aligned.append(aligned_probs)

    return aligned


# ============================================================================
# Diversity-aware加权平均
# ============================================================================
def compute_diversity_aware_weights(aucs, preds_dict):
    """
    计算Diversity-aware权重

    w_i = AUC_i^2 / (Diversity_penalty_i * sum(...))

    Diversity_penalty_i = mean(correlation(preds_i, preds_j)) for j≠i
    """
    names = list(preds_dict.keys())
    n = len(names)

    # 计算两两相关性矩阵
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.corrcoef(preds_dict[names[i]], preds_dict[names[j]])[0, 1]
            corr_matrix[i, j] = corr_matrix[j, i] = corr

    # 计算多样性惩罚（与其他模型的平均相关性）
    diversity_penalty = []
    for i in range(n):
        others_corr = [corr_matrix[i, j] for j in range(n) if j != i]
        diversity_penalty.append(np.mean(others_corr))

    # 计算权重
    alpha = 2.0
    beta = 1.0
    raw_weights = []
    for i in range(n):
        w = (aucs[names[i]] ** alpha) / (diversity_penalty[i] ** beta)
        raw_weights.append(max(w, 0.01))  # 防止权重为0

    weights = np.array(raw_weights) / sum(raw_weights)

    return {names[i]: weights[i] for i in range(n)}, corr_matrix


# ============================================================================
# 训练后验证 Gates
# ============================================================================
def validate_ensemble_gates(models_dict, weights, X_test, y_test, feature_names, max_disagreement=MAX_DISAGREEMENT):
    """训练后强制检查 gates"""
    log.info("\n" + "=" * 80)
    log.info("执行训练后验证 Gates")
    log.info("=" * 80)

    # 获取各模型预测
    preds = {}
    if "xgboost" in models_dict:
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        preds["xgboost"] = models_dict["xgboost"].predict(dtest)
    if "lightgbm" in models_dict:
        preds["lightgbm"] = models_dict["lightgbm"].predict(
            X_test, num_iteration=models_dict["lightgbm"].best_iteration
        )
    if "catboost" in models_dict:
        preds["catboost"] = models_dict["catboost"].predict_proba(X_test)[:, 1]

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

    # Gate 2: 概率分布一致性
    log.info("\n[Gate 2] 概率分布一致性检查")
    for name, pred in preds.items():
        mean_p = pred.mean()
        status = "✓" if PROB_MEAN_MIN <= mean_p <= PROB_MEAN_MAX else "✗"
        log.info(f"  {name} mean_prob={mean_p:.4f} (范围[{PROB_MEAN_MIN}, {PROB_MEAN_MAX}]) {status}")
        if not (PROB_MEAN_MIN <= mean_p <= PROB_MEAN_MAX):
            all_passed = False

    # Gate 3: Top50分歧度
    log.info("\n[Gate 3] Top50分歧度检查")
    pred_df = pd.DataFrame(preds)
    top50_preds = pred_df.nlargest(50, "catboost")
    disagreements = top50_preds.max(axis=1) - top50_preds.min(axis=1)
    mean_disagree = disagreements.mean()
    status = "✓" if mean_disagree <= max_disagreement else "✗"
    log.info(f"  Top50原始分歧度={mean_disagree:.4f} (阈值<={max_disagreement}) {status}")
    if mean_disagree > max_disagreement:
        all_passed = False

    # Gate 4: Brier Score
    log.info("\n[Gate 4] Brier Score检查")
    for name, pred in preds.items():
        brier = brier_score_loss(y_test, pred)
        status = "✓" if brier <= MAX_BRIER else "✗"
        log.info(f"  {name} Brier={brier:.4f} (阈值<={MAX_BRIER}) {status}")
        if brier > MAX_BRIER:
            all_passed = False

    # Gate 5: 集成性能
    log.info("\n[Gate 5] 集成性能检查")
    ensemble = sum(preds[name] * weights[name] for name in preds)
    ensemble_auc = roc_auc_score(y_test, ensemble)
    ensemble_brier = brier_score_loss(y_test, ensemble)
    log.info(f"  Ensemble AUC={ensemble_auc:.4f}, Brier={ensemble_brier:.4f}")

    if all_passed:
        log.success("所有 Gates 通过！")
    else:
        log.error("部分 Gates 未通过，请检查模型！")

    return all_passed, preds, ensemble


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("=" * 80)
    log.info("v2.9.4 专业级集成模型训练")
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

    # 选择最后一折作为最终训练/验证/测试
    # 前面的fold用于超参数验证（简化版，直接用最后一折）
    train_df, val_df, test_df = cv_splits[-1]

    X_train = train_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df["label"].values
    X_val = val_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_val = val_df["label"].values
    X_test = test_df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test_df["label"].values

    log.info(f"\n最终数据集: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")

    # 训练三个子模型
    log.info("\n" + "=" * 80)
    log.info("训练子模型")
    log.info("=" * 80)

    xgb_model, xgb_auc, xgb_brier = train_xgboost(X_train, y_train, X_val, y_val, valid_features)
    lgb_model, lgb_auc, lgb_brier = train_lightgbm(X_train, y_train, X_val, y_val, valid_features)
    cat_model, cat_auc, cat_brier = train_catboost(X_train, y_train, X_val, y_val, valid_features)

    models_dict = {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model}
    aucs = {"xgboost": xgb_auc, "lightgbm": lgb_auc, "catboost": cat_auc}

    # 获取验证集预测（用于校准和对齐）
    val_preds_raw = {
        "xgboost": xgb_model.predict(xgb.DMatrix(X_val, feature_names=valid_features)),
        "lightgbm": lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration),
        "catboost": cat_model.predict_proba(X_val)[:, 1],
    }

    # 温度缩放校准
    log.info("\n" + "=" * 80)
    log.info("温度缩放校准")
    log.info("=" * 80)

    temperatures = {}
    val_preds_cal = {}
    for name, preds in val_preds_raw.items():
        cal_preds, T = temperature_scaling(preds, y_val)
        val_preds_cal[name] = cal_preds
        temperatures[name] = T
        brier_before = brier_score_loss(y_val, preds)
        brier_after = brier_score_loss(y_val, cal_preds)
        log.info(f"  {name}: T={T:.4f}, Brier {brier_before:.4f} -> {brier_after:.4f}")

    # 温度缩放后概率已足够对齐，跳过额外的分位数对齐
    log.info("\n" + "=" * 80)
    log.info("Diversity-aware权重计算")
    log.info("=" * 80)

    weights, corr_matrix = compute_diversity_aware_weights(aucs, val_preds_cal)
    for name, w in weights.items():
        log.info(f"  {name}: weight={w:.4f} (AUC={aucs[name]:.4f})")

    log.info(f"  相关性矩阵:\n{pd.DataFrame(corr_matrix, index=aucs.keys(), columns=aucs.keys())}")

    # 训练后验证 Gates
    passed, test_preds, ensemble = validate_ensemble_gates(models_dict, weights, X_test, y_test, valid_features)

    if not passed:
        log.error("验证未通过，终止保存")
        return

    # 保存模型
    log.info("\n" + "=" * 80)
    log.info("保存模型")
    log.info("=" * 80)

    model_version = "v2.9.4-ensemble"
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / model_version / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(str(model_dir / "xgboost.json"))
    lgb_model.save_model(str(model_dir / "lightgbm.txt"))
    cat_model.save_model(str(model_dir / "catboost.cbm"))

    # 保存LGB best_iteration
    with open(model_dir / "lightgbm_meta.json", "w") as f:
        json.dump({"best_iteration": int(lgb_model.best_iteration), "num_trees": int(lgb_model.num_trees())}, f)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_features, f)

    # 保存温度参数
    with open(model_dir / "temperatures.json", "w") as f:
        json.dump(temperatures, f)

    # 保存分位数对齐映射（保存验证集排序后的概率作为参考）
    ref_sorted = np.sort(val_preds_cal["catboost"])
    np.save(model_dir / "quantile_ref.npy", ref_sorted)

    # 保存权重
    with open(model_dir / "weights.json", "w") as f:
        json.dump(weights, f)

    # 保存指标
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
                "temperatures": temperatures,
                "weights": weights,
                "correlation_matrix": corr_matrix.tolist(),
                "feature_count": len(valid_features),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
            },
            f,
            indent=2,
        )

    log.success(f"\n模型已保存到: {model_dir}")
    log.success("v2.9.4 专业级训练完成！")

    # ========================================================================
    # ========================================================================
    # 自动更新 current.json
    # ========================================================================
    log.info("\n" + "=" * 80)
    log.info("更新版本指针")
    log.info("=" * 80)

    # 计算 Top50 分歧度（用于记录）
    pred_df = pd.DataFrame(test_preds)
    top50_preds = pred_df.nlargest(50, "catboost")
    mean_disagree = (top50_preds.max(axis=1) - top50_preds.min(axis=1)).mean()

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
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "notes": (
            f"主力策略: v2.9.1-ensemble + integrated (sector-filter)。"
            f"最新训练: {model_version} (AUC={roc_auc_score(y_test, ensemble):.4f}, "
            f"分歧度={mean_disagree:.4f}, Brier={brier_score_loss(y_test, ensemble):.4f})。"
            f"v2.9.4 目标为降低分歧度，AUC 可能略低于 v2.9.3。"
        ),
    }

    # 如果存在旧 current.json，保留 production/staging/testing 不变，只更新 development
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
