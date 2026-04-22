"""
Walk-Forward验证脚本

在多个时间窗口上测试模型的稳定性和泛化能力
- 滑动窗口方式训练和测试
- 验证模型在不同市场环境下的表现
- 避免过拟合，确保模型鲁棒性
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from src.utils.logger import log


def safe_to_datetime(date_value):
    """
    安全地将日期值转换为datetime类型

    处理以下情况：
    - 整数：如 20230101 -> 被错误解析为纳秒时间戳
    - 字符串：如 '20230101' -> 正常解析
    - datetime：直接返回
    """
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(date_value, errors="coerce")


def load_and_prepare_data(
    neg_version="v2", use_market_factors=True, use_tech_factors=False, use_advanced_factors=False
):
    """加载数据"""
    log.info("=" * 80)
    log.info("加载数据")
    log.info("=" * 80)

    # 加载正样本（使用新的目录结构）
    if use_advanced_factors:
        pos_file = "data/training/processed/feature_data_34d_advanced.csv"
        log.info("📊 使用带高级技术因子的特征文件(advanced)")
    elif use_tech_factors:
        pos_file = "data/training/processed/feature_data_34d_full.csv"
        log.info("📊 使用带新技术因子的特征文件(full)")
    elif use_market_factors:
        pos_file = "data/training/processed/feature_data_34d_with_market.csv"
        log.info("📊 使用带市场因子的特征文件")
    else:
        pos_file = "data/training/processed/feature_data_34d.csv"
        log.info("📊 使用基础特征文件")

    df_pos = pd.read_csv(pos_file)
    df_pos["label"] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")

    # 加载负样本（使用新的目录结构）
    if neg_version == "v2":
        if use_advanced_factors:
            neg_file = "data/training/features/negative_feature_data_v2_34d_advanced.csv"
        elif use_tech_factors:
            neg_file = "data/training/features/negative_feature_data_v2_34d_full.csv"
        elif use_market_factors:
            neg_file = "data/training/features/negative_feature_data_v2_34d_with_market.csv"
        else:
            neg_file = "data/training/features/negative_feature_data_v2_34d.csv"
    else:
        neg_file = "data/training/features/negative_feature_data_34d.csv"

    df_neg = pd.read_csv(neg_file)
    log.success(f"✓ 负样本加载完成: {len(df_neg)} 条 (版本: {neg_version})")

    # 合并
    df = pd.concat([df_pos, df_neg])
    log.info(f"✓ 数据合并完成: {len(df)} 条")
    log.info("")

    return df


def extract_features_with_time(df):
    """从34天的时序数据中提取统计特征（保留时间信息）"""
    log.info("=" * 80)
    log.info("特征工程")
    log.info("=" * 80)
    log.info("将34天时序数据转换为统计特征...")

    # 重新分配唯一的sample_id
    df["unique_sample_id"] = df.groupby(["ts_code", "label"]).ngroup()

    features = []
    sample_ids = df["unique_sample_id"].unique()

    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")

        sample_data = df[df["unique_sample_id"] == sample_id].sort_values("days_to_t1")

        if len(sample_data) < 20:  # 至少20天数据
            continue

        # 从数据中获取T1日期
        t1_row = sample_data.iloc[sample_data["days_to_t1"].abs().argmin()]
        t1_date = safe_to_datetime(t1_row["trade_date"])

        feature_dict = {
            "sample_id": sample_id,
            "ts_code": sample_data["ts_code"].iloc[0],
            "name": sample_data["name"].iloc[0],
            "label": int(sample_data["label"].iloc[0]),
            "t1_date": t1_date,
        }

        # 价格特征
        feature_dict["close_mean"] = sample_data["close"].mean()
        feature_dict["close_std"] = sample_data["close"].std()
        feature_dict["close_max"] = sample_data["close"].max()
        feature_dict["close_min"] = sample_data["close"].min()
        feature_dict["close_trend"] = (
            (sample_data["close"].iloc[-1] - sample_data["close"].iloc[0]) / sample_data["close"].iloc[0] * 100
        )

        # 涨跌幅特征
        feature_dict["pct_chg_mean"] = sample_data["pct_chg"].mean()
        feature_dict["pct_chg_std"] = sample_data["pct_chg"].std()
        feature_dict["pct_chg_sum"] = sample_data["pct_chg"].sum()
        feature_dict["positive_days"] = (sample_data["pct_chg"] > 0).sum()
        feature_dict["negative_days"] = (sample_data["pct_chg"] < 0).sum()
        feature_dict["max_gain"] = sample_data["pct_chg"].max()
        feature_dict["max_loss"] = sample_data["pct_chg"].min()

        # 量比特征
        if "volume_ratio" in sample_data.columns:
            feature_dict["volume_ratio_mean"] = sample_data["volume_ratio"].mean()
            feature_dict["volume_ratio_max"] = sample_data["volume_ratio"].max()
            feature_dict["volume_ratio_gt_2"] = (sample_data["volume_ratio"] > 2).sum()
            feature_dict["volume_ratio_gt_4"] = (sample_data["volume_ratio"] > 4).sum()

        # MACD特征
        if "macd" in sample_data.columns:
            macd_data = sample_data["macd"].dropna()
            if len(macd_data) > 0:
                feature_dict["macd_mean"] = macd_data.mean()
                feature_dict["macd_positive_days"] = (macd_data > 0).sum()
                feature_dict["macd_max"] = macd_data.max()

        # MA特征
        if "ma5" in sample_data.columns:
            feature_dict["ma5_mean"] = sample_data["ma5"].mean()
            feature_dict["price_above_ma5"] = (sample_data["close"] > sample_data["ma5"]).sum()

        if "ma10" in sample_data.columns:
            feature_dict["ma10_mean"] = sample_data["ma10"].mean()
            feature_dict["price_above_ma10"] = (sample_data["close"] > sample_data["ma10"]).sum()

        # 市值特征
        if "total_mv" in sample_data.columns:
            mv_data = sample_data["total_mv"].dropna()
            if len(mv_data) > 0:
                feature_dict["total_mv_mean"] = mv_data.mean()

        if "circ_mv" in sample_data.columns:
            circ_mv_data = sample_data["circ_mv"].dropna()
            if len(circ_mv_data) > 0:
                feature_dict["circ_mv_mean"] = circ_mv_data.mean()

        # 动量特征
        days = len(sample_data)
        if days >= 7:
            feature_dict["return_1w"] = (
                (sample_data["close"].iloc[-1] - sample_data["close"].iloc[-7]) / sample_data["close"].iloc[-7] * 100
            )
        if days >= 14:
            feature_dict["return_2w"] = (
                (sample_data["close"].iloc[-1] - sample_data["close"].iloc[-14]) / sample_data["close"].iloc[-14] * 100
            )

        # 市场因子特征（如果存在）
        if "market_pct_chg" in sample_data.columns:
            market_data = sample_data["market_pct_chg"].dropna()
            if len(market_data) > 0:
                feature_dict["market_pct_chg_mean"] = market_data.mean()

        if "market_return_34d" in sample_data.columns:
            market_return_data = sample_data["market_return_34d"].dropna()
            if len(market_return_data) > 0:
                feature_dict["market_return_34d_last"] = market_return_data.iloc[-1]

        if "market_volatility_34d" in sample_data.columns:
            market_vol_data = sample_data["market_volatility_34d"].dropna()
            if len(market_vol_data) > 0:
                feature_dict["market_volatility_34d_last"] = market_vol_data.iloc[-1]

        if "market_trend" in sample_data.columns:
            market_trend_data = sample_data["market_trend"].dropna()
            if len(market_trend_data) > 0:
                feature_dict["market_trend_last"] = market_trend_data.iloc[-1]

        if "excess_return" in sample_data.columns:
            excess_data = sample_data["excess_return"].dropna()
            if len(excess_data) > 0:
                feature_dict["excess_return_mean"] = excess_data.mean()
                feature_dict["excess_return_sum"] = excess_data.sum()
                feature_dict["excess_return_positive_days"] = (excess_data > 0).sum()

        if "excess_return_cumsum" in sample_data.columns:
            excess_cumsum_data = sample_data["excess_return_cumsum"].dropna()
            if len(excess_cumsum_data) > 0:
                feature_dict["excess_return_cumsum_last"] = excess_cumsum_data.iloc[-1]

        if "price_vs_hist_mean" in sample_data.columns:
            hist_mean_data = sample_data["price_vs_hist_mean"].dropna()
            if len(hist_mean_data) > 0:
                feature_dict["price_vs_hist_mean_last"] = hist_mean_data.iloc[-1]

        # 以下低效特征已剔除（重要性 < 阈值）:
        # - price_vs_hist_high_last: 0.0088
        # - volatility_vs_hist_last: 0.0064

        # ===== 新技术因子特征（full）=====
        # 换手率（自由流通股）
        if "turnover_rate_f" in sample_data.columns:
            turnover_data = sample_data["turnover_rate_f"].dropna()
            if len(turnover_data) > 0:
                feature_dict["turnover_rate_f_mean"] = turnover_data.mean()
                feature_dict["turnover_rate_f_max"] = turnover_data.max()
                feature_dict["turnover_rate_f_std"] = turnover_data.std()

        # 乖离率BIAS (bias_short/mid/long)
        if "bias_short" in sample_data.columns:
            bias_short = sample_data["bias_short"].dropna()
            if len(bias_short) > 0:
                feature_dict["bias_short_last"] = bias_short.iloc[-1]
                feature_dict["bias_short_mean"] = bias_short.mean()
        if "bias_mid" in sample_data.columns:
            bias_mid = sample_data["bias_mid"].dropna()
            if len(bias_mid) > 0:
                feature_dict["bias_mid_last"] = bias_mid.iloc[-1]
        if "bias_long" in sample_data.columns:
            bias_long = sample_data["bias_long"].dropna()
            if len(bias_long) > 0:
                feature_dict["bias_long_last"] = bias_long.iloc[-1]

        # EMA
        if "ema_5" in sample_data.columns and "ema_20" in sample_data.columns:
            ema5 = sample_data["ema_5"].dropna()
            ema20 = sample_data["ema_20"].dropna()
            if len(ema5) > 0 and len(ema20) > 0:
                feature_dict["ema_ratio_5_20"] = ema5.iloc[-1] / ema20.iloc[-1] if ema20.iloc[-1] != 0 else 1
                if len(sample_data["close"].dropna()) > 0:
                    close_last = sample_data["close"].dropna().iloc[-1]
                    feature_dict["price_vs_ema5"] = (
                        (close_last - ema5.iloc[-1]) / ema5.iloc[-1] * 100 if ema5.iloc[-1] != 0 else 0
                    )
                    feature_dict["price_vs_ema20"] = (
                        (close_last - ema20.iloc[-1]) / ema20.iloc[-1] * 100 if ema20.iloc[-1] != 0 else 0
                    )
        if "ema_60" in sample_data.columns:
            ema60 = sample_data["ema_60"].dropna()
            if len(ema60) > 0 and len(sample_data["close"].dropna()) > 0:
                close_last = sample_data["close"].dropna().iloc[-1]
                feature_dict["price_vs_ema60"] = (
                    (close_last - ema60.iloc[-1]) / ema60.iloc[-1] * 100 if ema60.iloc[-1] != 0 else 0
                )

        # KDJ
        if "kdj_k" in sample_data.columns:
            kdj_k = sample_data["kdj_k"].dropna()
            if len(kdj_k) > 0:
                feature_dict["kdj_k_last"] = kdj_k.iloc[-1]
                feature_dict["kdj_k_mean"] = kdj_k.mean()
        if "kdj_d" in sample_data.columns:
            kdj_d = sample_data["kdj_d"].dropna()
            if len(kdj_d) > 0:
                feature_dict["kdj_d_last"] = kdj_d.iloc[-1]
        if "kdj_j" in sample_data.columns:
            kdj_j = sample_data["kdj_j"].dropna()
            if len(kdj_j) > 0:
                feature_dict["kdj_j_last"] = kdj_j.iloc[-1]
                feature_dict["kdj_j_overbought"] = (kdj_j > 80).sum()
                feature_dict["kdj_j_oversold"] = (kdj_j < 20).sum()

        # 涨停统计 (is_limit_up)
        if "is_limit_up" in sample_data.columns:
            is_limit = sample_data["is_limit_up"].dropna()
            if len(is_limit) > 0:
                feature_dict["limit_up_count"] = is_limit.sum()

        # OBV
        if "obv" in sample_data.columns:
            obv = sample_data["obv"].dropna()
            if len(obv) > 0:
                feature_dict["obv_change"] = (
                    (obv.iloc[-1] - obv.iloc[0]) / abs(obv.iloc[0]) * 100 if obv.iloc[0] != 0 else 0
                )
                feature_dict["obv_trend"] = 1 if obv.iloc[-1] > obv.mean() else 0

        # 成交量与均量比 (vol_ma5_ratio/vol_ma20_ratio)
        if "vol_ma5_ratio" in sample_data.columns:
            vol_r5 = sample_data["vol_ma5_ratio"].dropna()
            if len(vol_r5) > 0:
                feature_dict["vol_ma5_ratio_mean"] = vol_r5.mean()
                feature_dict["vol_ma5_ratio_max"] = vol_r5.max()
        if "vol_ma20_ratio" in sample_data.columns:
            vol_r20 = sample_data["vol_ma20_ratio"].dropna()
            if len(vol_r20) > 0:
                feature_dict["vol_ma20_ratio_mean"] = vol_r20.mean()
                feature_dict["vol_ma20_ratio_max"] = vol_r20.max()

        # ===== 高级技术因子（advanced）=====
        # 动量因子
        for period in [5, 10, 20]:
            col = f"momentum_{period}d"
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_last"] = data.iloc[-1]
                    feature_dict[f"{col}_mean"] = data.mean()

        if "momentum_acceleration" in sample_data.columns:
            data = sample_data["momentum_acceleration"].dropna()
            if len(data) > 0:
                feature_dict["momentum_acceleration_last"] = data.iloc[-1]

        # 量价配合度
        if "volume_price_corr_10d" in sample_data.columns:
            data = sample_data["volume_price_corr_10d"].dropna()
            if len(data) > 0:
                feature_dict["volume_price_corr_last"] = data.iloc[-1]
        if "volume_price_match_sum_10d" in sample_data.columns:
            data = sample_data["volume_price_match_sum_10d"].dropna()
            if len(data) > 0:
                feature_dict["volume_price_match_sum"] = data.iloc[-1]

        # 多时间框架特征 (8d, 55d)
        for tf in [8, 55]:
            for metric in ["return", "price_vs_ma", "volatility", "price_position", "trend_slope"]:
                col = f"{metric}_{tf}d"
                if col in sample_data.columns:
                    data = sample_data[col].dropna()
                    if len(data) > 0:
                        feature_dict[f"{col}_last"] = data.iloc[-1]

        # 突破形态
        for period in [10, 20, 55]:
            col = f"breakout_high_{period}d"
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_sum"] = data.sum()

        for ma in [5, 10, 20, 55]:
            col = f"breakout_ma{ma}"
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_sum"] = data.sum()

        if "high_volume_breakout" in sample_data.columns:
            data = sample_data["high_volume_breakout"].dropna()
            if len(data) > 0:
                feature_dict["high_volume_breakout_sum"] = data.sum()

        if "consecutive_new_high" in sample_data.columns:
            data = sample_data["consecutive_new_high"].dropna()
            if len(data) > 0:
                feature_dict["consecutive_new_high_max"] = data.max()

        # 支撑阻力
        for period in [10, 20]:
            for metric in ["dist_to_support", "dist_to_resistance"]:
                col = f"{metric}_{period}d"
                if col in sample_data.columns:
                    data = sample_data[col].dropna()
                    if len(data) > 0:
                        feature_dict[f"{col}_last"] = data.iloc[-1]

            for metric in ["support_strength", "resistance_strength"]:
                col = f"{metric}_{period}d"
                if col in sample_data.columns:
                    data = sample_data[col].dropna()
                    if len(data) > 0:
                        feature_dict[f"{col}_last"] = data.iloc[-1]

        if "channel_width_20d" in sample_data.columns:
            data = sample_data["channel_width_20d"].dropna()
            if len(data) > 0:
                feature_dict["channel_width_last"] = data.iloc[-1]

        # 高级成交量
        for col in ["volume_trend_slope_10d", "volume_trend_slope_20d"]:
            if col in sample_data.columns:
                data = sample_data[col].dropna()
                if len(data) > 0:
                    feature_dict[f"{col}_last"] = data.iloc[-1]

        if "volume_breakout_count_20d" in sample_data.columns:
            data = sample_data["volume_breakout_count_20d"].dropna()
            if len(data) > 0:
                feature_dict["volume_breakout_count"] = data.iloc[-1]

        if "price_up_vol_down_count_10d" in sample_data.columns:
            data = sample_data["price_up_vol_down_count_10d"].dropna()
            if len(data) > 0:
                feature_dict["price_up_vol_down_count"] = data.iloc[-1]

        if "price_down_vol_up_count_10d" in sample_data.columns:
            data = sample_data["price_down_vol_up_count_10d"].dropna()
            if len(data) > 0:
                feature_dict["price_down_vol_up_count"] = data.iloc[-1]

        if "volume_rsv_20d" in sample_data.columns:
            data = sample_data["volume_rsv_20d"].dropna()
            if len(data) > 0:
                feature_dict["volume_rsv_last"] = data.iloc[-1]

        if "obv_trend" in sample_data.columns:
            data = sample_data["obv_trend"].dropna()
            if len(data) > 0:
                feature_dict["obv_trend_sum"] = data.sum()

        features.append(feature_dict)

    df_features = pd.DataFrame(features)
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info("")

    return df_features


def walk_forward_validation(df_features, n_splits=5, train_size=0.6):
    """
    Walk-forward验证

    Args:
        df_features: 特征DataFrame
        n_splits: 分割数量（时间窗口数）
        train_size: 训练集占比
    """
    log.info("=" * 80)
    log.info("Walk-Forward验证")
    log.info("=" * 80)
    log.info(f"时间窗口数: {n_splits}")
    log.info(f"训练集占比: {train_size*100}%")
    log.info("")

    # 按时间排序
    df_features = df_features.sort_values("t1_date").reset_index(drop=True)

    # 准备特征和标签
    feature_cols = [
        col for col in df_features.columns if col not in ["sample_id", "ts_code", "name", "label", "t1_date"]
    ]

    X = df_features[feature_cols]
    y = df_features["label"]
    dates = df_features["t1_date"]

    log.info("缺失值处理: 每个窗口使用训练集中位数填充（避免未来函数）")

    # 计算每个窗口的大小
    total_samples = len(df_features)
    window_size = total_samples // n_splits

    results = []

    for i in range(n_splits):
        log.info(f"\n{'='*80}")
        log.info(f"时间窗口 {i+1}/{n_splits}")
        log.info(f"{'='*80}")

        # 计算训练集和测试集的索引
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_splits - 1 else total_samples

        window_data = df_features.iloc[start_idx:end_idx]

        # 按时间划分训练集和测试集
        split_idx = int(len(window_data) * train_size)

        train_data = window_data.iloc[:split_idx]
        test_data = window_data.iloc[split_idx:]

        if len(test_data) < 10:  # 测试集太小，跳过
            log.warning(f"⚠️  测试集样本太少({len(test_data)}个)，跳过此窗口")
            continue

        # 准备训练集和测试集
        X_train = train_data[feature_cols].copy()
        y_train = train_data["label"]
        X_test = test_data[feature_cols].copy()
        y_test = test_data["label"]

        # 缺失值处理：使用训练集的中位数填充（避免未来函数）
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)  # 用训练集的统计量填充测试集

        train_dates = train_data["t1_date"]
        test_dates = test_data["t1_date"]

        log.info("\n时间范围:")
        log.info(f"  训练集: {train_dates.min().date()} 至 {train_dates.max().date()}")
        log.info(f"  测试集: {test_dates.min().date()} 至 {test_dates.max().date()}")

        log.info("\n样本分布:")
        log.info(f"  训练集: {len(X_train)} 个 (正:{y_train.sum()}, 负:{len(y_train)-y_train.sum()})")
        log.info(f"  测试集: {len(X_test)} 个 (正:{y_test.sum()}, 负:{len(y_test)-y_test.sum()})")

        # 计算类别权重（处理样本不均衡）
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        raw_weight = neg_count / pos_count if pos_count > 0 else 1.0
        # 限制权重范围在[0.5, 2.0]之间，避免过度补偿
        scale_pos_weight = max(0.5, min(2.0, raw_weight))

        # 训练模型
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # 处理样本不均衡
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train, verbose=False)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0

        log.info("\n性能指标:")
        log.info(f"  准确率: {accuracy*100:.2f}%")
        log.info(f"  精确率: {precision*100:.2f}%")
        log.info(f"  召回率: {recall*100:.2f}%")
        log.info(f"  F1-Score: {f1*100:.2f}%")
        log.info(f"  AUC-ROC: {auc:.4f}")

        # 记录结果
        results.append(
            {
                "window": i + 1,
                "train_start": train_dates.min().date().isoformat(),
                "train_end": train_dates.max().date().isoformat(),
                "test_start": test_dates.min().date().isoformat(),
                "test_end": test_dates.max().date().isoformat(),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc,
            }
        )

    return results


def analyze_results(results):
    """分析Walk-forward验证结果"""
    log.info("\n" + "=" * 80)
    log.info("Walk-Forward验证结果汇总")
    log.info("=" * 80)

    df_results = pd.DataFrame(results)

    # 计算统计量
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]

    log.info("\n各时间窗口表现:")
    for _, row in df_results.iterrows():
        log.info(f"\n窗口 {int(row['window'])}: {row['test_start']} 至 {row['test_end']}")
        log.info(f"  准确率: {row['accuracy']*100:.2f}%")
        log.info(f"  精确率: {row['precision']*100:.2f}%")
        log.info(f"  召回率: {row['recall']*100:.2f}%")
        log.info(f"  F1-Score: {row['f1_score']*100:.2f}%")
        log.info(f"  AUC: {row['auc']:.4f}")

    log.info("\n" + "=" * 80)
    log.info("统计摘要")
    log.info("=" * 80)

    for metric in metrics:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        min_val = df_results[metric].min()
        max_val = df_results[metric].max()

        metric_name = {
            "accuracy": "准确率",
            "precision": "精确率",
            "recall": "召回率",
            "f1_score": "F1-Score",
            "auc": "AUC-ROC",
        }[metric]

        log.info(f"\n{metric_name}:")
        log.info(f"  平均值: {mean_val*100:.2f}%")
        log.info(f"  标准差: {std_val*100:.2f}%")
        log.info(f"  最小值: {min_val*100:.2f}%")
        log.info(f"  最大值: {max_val*100:.2f}%")

    # 稳定性评估
    f1_std = df_results["f1_score"].std()
    if f1_std < 0.05:
        stability = "非常稳定 ⭐⭐⭐⭐⭐"
    elif f1_std < 0.10:
        stability = "稳定 ⭐⭐⭐⭐"
    elif f1_std < 0.15:
        stability = "一般 ⭐⭐⭐"
    else:
        stability = "不稳定 ⭐⭐"

    log.info(f"\n模型稳定性评估: {stability}")
    log.info(f"  (基于F1-Score标准差: {f1_std*100:.2f}%)")

    return df_results


def save_results(
    results_df, version=None, model_name="breakout_launch_scorer", use_advanced_factors=False, neg_version="v2"
):
    """
    保存验证结果

    Args:
        results_df: 验证结果 DataFrame
        version: 版本号（如 v1.5.0），指定后将保存到版本目录
        model_name: 模型名称
        use_advanced_factors: 是否使用高级因子
        neg_version: 负样本版本
    """
    # 构建结果字典
    results_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_windows": len(results_df),
        "config": {
            "use_advanced_factors": use_advanced_factors,
            "neg_version": neg_version,
            "n_splits": 5,
            "train_size": 0.6,
        },
        "summary": {
            "accuracy_mean": float(results_df["accuracy"].mean()),
            "accuracy_std": float(results_df["accuracy"].std()),
            "precision_mean": float(results_df["precision"].mean()),
            "precision_std": float(results_df["precision"].std()),
            "recall_mean": float(results_df["recall"].mean()),
            "recall_std": float(results_df["recall"].std()),
            "f1_score_mean": float(results_df["f1_score"].mean()),
            "f1_score_std": float(results_df["f1_score"].std()),
            "auc_mean": float(results_df["auc"].mean()),
            "auc_std": float(results_df["auc"].std()),
        },
        "stability": _evaluate_stability(results_df),
        "windows": results_df.to_dict("records"),
    }

    # 保存到默认位置
    output_file = "data/results/walk_forward_validation_results.json"
    os.makedirs("data/results", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    log.success(f"\n✓ 验证结果已保存: {output_file}")

    # 如果指定了版本号，同时保存到版本目录
    if version:
        save_to_version_directory(results_dict, results_df, version, model_name)


def _evaluate_stability(results_df):
    """评估模型稳定性"""
    f1_std = results_df["f1_score"].std()
    auc_std = results_df["auc"].std()

    if f1_std < 0.05:
        level = "excellent"
        description = "非常稳定 ⭐⭐⭐⭐⭐"
    elif f1_std < 0.10:
        level = "good"
        description = "稳定 ⭐⭐⭐⭐"
    elif f1_std < 0.15:
        level = "fair"
        description = "一般 ⭐⭐⭐"
    else:
        level = "poor"
        description = "不稳定 ⭐⭐"

    return {"level": level, "description": description, "f1_std": float(f1_std), "auc_std": float(auc_std)}


def save_to_version_directory(results_dict, results_df, version, model_name):
    """
    将验证结果保存到版本目录

    Args:
        results_dict: 验证结果字典
        results_df: 验证结果 DataFrame
        version: 版本号
        model_name: 模型名称
    """
    log.info("")
    log.info("-" * 60)
    log.info(f"📦 保存验证结果到版本目录: {model_name}/{version}")
    log.info("-" * 60)

    # 版本目录
    version_dir = f"data/models/{model_name}/versions/{version}"
    evaluation_dir = f"{version_dir}/evaluation"

    # 创建目录
    os.makedirs(evaluation_dir, exist_ok=True)

    # 1. 保存验证结果 JSON
    validation_file = f"{evaluation_dir}/walk_forward_validation.json"
    with open(validation_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    log.success(f"  ✓ 验证结果: {validation_file}")

    # 2. 保存验证结果 CSV（方便查看）
    csv_file = f"{evaluation_dir}/walk_forward_validation.csv"
    results_df.to_csv(csv_file, index=False)
    log.success(f"  ✓ 验证明细: {csv_file}")

    # 3. 生成验证报告
    report_file = f"{evaluation_dir}/validation_report.md"
    _generate_validation_report(results_dict, results_df, report_file, version)
    log.success(f"  ✓ 验证报告: {report_file}")

    # 4. 更新版本元数据
    _update_version_metadata(version_dir, results_dict)

    log.success(f"\n✅ 验证结果已保存到版本目录: {evaluation_dir}")


def _generate_validation_report(results_dict, results_df, report_file, version):
    """生成验证报告 Markdown"""
    summary = results_dict["summary"]
    stability = results_dict["stability"]

    report = f"""# Walk-Forward 验证报告

## 版本信息

| 属性 | 值 |
|------|-----|
| **版本号** | {version} |
| **验证时间** | {results_dict['timestamp']} |
| **时间窗口数** | {results_dict['n_windows']} |
| **训练集比例** | {results_dict['config']['train_size']*100:.0f}% |

## 性能摘要

| 指标 | 均值 | 标准差 |
|------|------|--------|
| **准确率** | {summary['accuracy_mean']*100:.2f}% | ±{summary['accuracy_std']*100:.2f}% |
| **精确率** | {summary['precision_mean']*100:.2f}% | ±{summary['precision_std']*100:.2f}% |
| **召回率** | {summary['recall_mean']*100:.2f}% | ±{summary['recall_std']*100:.2f}% |
| **F1-Score** | {summary['f1_score_mean']*100:.2f}% | ±{summary['f1_score_std']*100:.2f}% |
| **AUC-ROC** | {summary['auc_mean']:.4f} | ±{summary['auc_std']:.4f} |

## 稳定性评估

| 属性 | 值 |
|------|-----|
| **稳定性等级** | {stability['description']} |
| **F1 标准差** | {stability['f1_std']*100:.2f}% |
| **AUC 标准差** | {stability['auc_std']:.4f} |

## 各窗口详情

| 窗口 | 训练期 | 测试期 | 准确率 | 精确率 | 召回率 | F1 | AUC |
|------|--------|--------|--------|--------|--------|-----|-----|
"""

    for _, row in results_df.iterrows():
        report += f"| {int(row['window'])} | {row['train_start']}~{row['train_end']} | {row['test_start']}~{row['test_end']} | {row['accuracy']*100:.1f}% | {row['precision']*100:.1f}% | {row['recall']*100:.1f}% | {row['f1_score']*100:.1f}% | {row['auc']:.3f} |\n"

    report += """
## 结论

"""

    if stability["level"] == "excellent":
        report += "✅ **模型表现优秀**：在不同时间窗口上表现稳定，可以放心用于生产环境。\n"
    elif stability["level"] == "good":
        report += "✅ **模型表现良好**：在大多数时间窗口上表现稳定，建议持续监控。\n"
    elif stability["level"] == "fair":
        report += "⚠️ **模型表现一般**：在部分时间窗口上表现不稳定，建议进一步优化。\n"
    else:
        report += "❌ **模型表现不稳定**：在不同时间窗口上表现差异较大，不建议用于生产环境。\n"

    report += """
---
*由 walk_forward_validation.py 自动生成*
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


def _update_version_metadata(version_dir, results_dict):
    """更新版本元数据，添加验证结果"""
    metadata_file = f"{version_dir}/metadata.json"

    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # 添加验证结果
    metadata["validation"] = {
        "walk_forward": {
            "completed_at": results_dict["timestamp"],
            "n_windows": results_dict["n_windows"],
            "summary": results_dict["summary"],
            "stability": results_dict["stability"],
        }
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.success("  ✓ 已更新版本元数据")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward验证")
    parser.add_argument("--use-market-factors", action="store_true", help="使用带市场因子的特征文件")
    parser.add_argument("--use-tech-factors", action="store_true", help="使用带新技术因子的v2特征文件")
    parser.add_argument("--use-advanced-factors", action="store_true", help="使用带高级技术因子的特征文件")
    parser.add_argument("--neg-version", default="v2", choices=["v1", "v2"], help="负样本版本")
    # 版本管理参数
    parser.add_argument(
        "--version", type=str, default=None, help="模型版本号（如 v1.5.0），指定后将验证结果保存到版本目录"
    )
    parser.add_argument(
        "--model-name", type=str, default="breakout_launch_scorer", help="模型名称（默认: breakout_launch_scorer）"
    )
    args = parser.parse_args()

    # 配置
    NEG_VERSION = args.neg_version
    USE_ADVANCED_FACTORS = args.use_advanced_factors
    USE_TECH_FACTORS = args.use_tech_factors and not USE_ADVANCED_FACTORS
    USE_MARKET_FACTORS = args.use_market_factors or (not args.use_tech_factors and not USE_ADVANCED_FACTORS)

    log.info("=" * 80)
    log.info("Walk-Forward验证 - 多时间窗口模型稳定性测试")
    log.info("=" * 80)
    log.info(
        f"配置: 负样本版本={NEG_VERSION}, 市场因子={USE_MARKET_FACTORS}, 技术因子={USE_TECH_FACTORS}, 高级因子={USE_ADVANCED_FACTORS}"
    )
    if args.version:
        log.info(f"版本: {args.version} (结果将保存到版本目录)")
    log.info("")

    try:
        # 1. 加载数据
        df = load_and_prepare_data(
            neg_version=NEG_VERSION,
            use_market_factors=USE_MARKET_FACTORS,
            use_tech_factors=USE_TECH_FACTORS,
            use_advanced_factors=USE_ADVANCED_FACTORS,
        )

        # 2. 特征工程
        df_features = extract_features_with_time(df)

        # 3. Walk-forward验证
        results = walk_forward_validation(df_features, n_splits=5, train_size=0.6)

        # 4. 分析结果
        results_df = analyze_results(results)

        # 5. 保存结果（支持版本管理）
        save_results(
            results_df,
            version=args.version,
            model_name=args.model_name,
            use_advanced_factors=USE_ADVANCED_FACTORS,
            neg_version=NEG_VERSION,
        )

        log.info("\n" + "=" * 80)
        log.success("✅ Walk-Forward验证完成！")
        log.info("=" * 80)

        if args.version:
            log.info("")
            log.info("💡 验证结果已保存到版本目录:")
            log.info(f"   data/models/{args.model_name}/versions/{args.version}/evaluation/")

    except Exception as e:
        log.error(f"✗ Walk-Forward验证出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
