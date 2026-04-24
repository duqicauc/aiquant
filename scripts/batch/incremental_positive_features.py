#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增量正样本特征提取（v280）

只为新增的正样本（不在现有 feature_data 中的）提取34天特征，
添加增强特征，并追加到现有 v5 enhanced 文件。
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为新增数据添加增强特征（与 feature_analysis_and_enhancement.py 一致）
    不依赖旧模型，直接计算。
    """
    df = df.copy()
    n = len(df)
    added_features = []

    # 1. 换手率异常检测
    if "turnover_rate" in df.columns:
        tr = df["turnover_rate"]
        tr_mean = tr.rolling(20, min_periods=5).mean()
        tr_std = tr.rolling(20, min_periods=5).std()
        df["turnover_zscore"] = (tr - tr_mean) / (tr_std + 1e-8)
        added_features.append("turnover_zscore")

        df["turnover_change_rate"] = tr.pct_change(5)
        added_features.append("turnover_change_rate")

        df["turnover_spike"] = (tr > tr_mean * 2).astype(int)
        added_features.append("turnover_spike")

    # 2. RSI-KDJ综合指标增强
    if "rsi_6" in df.columns and "kdj_j" in df.columns and "kdj_k" in df.columns:
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        added_features.append("rsi_kdj_golden_cross")

        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        added_features.append("rsi_kdj_strength")

        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        added_features.append("rsi_zone")

    # 3. 量价背离强度
    if "close" in df.columns and "vol" in df.columns:
        price_change_10d = df["close"].pct_change(10)
        vol_change_10d = df["vol"].pct_change(10)
        df["volume_price_divergence_strength"] = np.abs(price_change_10d - vol_change_10d)
        added_features.append("volume_price_divergence_strength")

        df["volume_price_confirm"] = ((price_change_10d > 0) == (vol_change_10d > 0)).astype(int)
        added_features.append("volume_price_confirm")

    # 4. 突破强度综合指标
    breakout_cols = [c for c in df.columns if "breakout_strength" in c]
    if len(breakout_cols) >= 2:
        df["breakout_strength_avg"] = df[breakout_cols].mean(axis=1)
        added_features.append("breakout_strength_avg")
        df["breakout_strength_max"] = df[breakout_cols].max(axis=1)
        added_features.append("breakout_strength_max")

    # 5. 均线多头排列强度
    ma_cols = ["ma5", "ma10", "ma_20d", "ma_34d", "ma_55d"]
    available_ma = [c for c in ma_cols if c in df.columns]
    if len(available_ma) >= 3:
        ma_values = df[available_ma].values
        ma_rank_score = np.zeros(len(df))
        for i in range(len(df)):
            row = ma_values[i]
            if not np.isnan(row).any():
                sorted_idx = np.argsort(row)[::-1]
                expected = np.arange(len(row))
                ma_rank_score[i] = 1 - np.abs(sorted_idx - expected).sum() / (len(row) * (len(row) - 1) / 2 + 1e-8)
        df["ma_alignment_score"] = ma_rank_score
        added_features.append("ma_alignment_score")

    # 6. 动量加速度（如果v5中不存在）
    if "momentum_acceleration" not in df.columns and "momentum_10d" in df.columns:
        df["momentum_acceleration"] = df["momentum_10d"].diff(5)
        added_features.append("momentum_acceleration")

    # 7. 价格位置综合指标
    position_cols = [c for c in df.columns if "price_position" in c]
    if len(position_cols) >= 2:
        df["price_position_avg"] = df[position_cols].mean(axis=1)
        added_features.append("price_position_avg")

    # 8. 风险调整收益
    if "return_34d" in df.columns and "volatility_34d" in df.columns:
        df["sharpe_like_34d"] = df["return_34d"] / (df["volatility_34d"] + 1e-8)
        added_features.append("sharpe_like_34d")

    log.info(f"  添加了 {len(added_features)} 个增强特征: {added_features}")
    return df


def main():
    log.info("=" * 80)
    log.info("v2.8.0 增量正样本特征提取")
    log.info("=" * 80)

    dm = DataManager()
    screener = PositiveSampleScreener(dm)

    # 1. 读取现有正样本列表
    existing_samples_file = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
    df_existing = pd.read_csv(existing_samples_file)
    log.info(f"现有正样本: {len(df_existing)} 条")

    # 2. 扫描新增正样本
    log.info("\n扫描新增正样本 (20251227 ~ 20260414)...")
    df_new = screener.screen_all_stocks(start_date="20251227", end_date="20260414")
    log.info(f"扫描到候选: {len(df_new)} 条")

    # 3. 去重
    existing_keys = set(zip(df_existing["ts_code"], df_existing["t1_date"].astype(str)))
    df_new["t1_date_str"] = df_new["t1_date"].astype(str)
    mask = ~df_new.apply(lambda r: (r["ts_code"], r["t1_date_str"]) in existing_keys, axis=1)
    df_new = df_new[mask].drop(columns=["t1_date_str"]).reset_index(drop=True)
    log.info(f"实际新增: {len(df_new)} 条")

    if df_new.empty:
        log.info("无新增正样本，跳过")
        return

    # 4. 分配新的 sample_id
    max_existing_id = int(df_existing["sample_id"].max()) if "sample_id" in df_existing.columns else 3254
    df_new.index = range(max_existing_id + 1, max_existing_id + 1 + len(df_new))
    log.info(f"新 sample_id 范围: {max_existing_id + 1} ~ {max_existing_id + len(df_new)}")

    # 5. 提取特征
    log.info("\n提取新增样本的34天特征...")
    df_new_features = screener.extract_features(df_new, lookback_days=34)

    if df_new_features.empty:
        log.error("特征提取失败！")
        return

    # 6. 数据质量处理
    log.info("数据质量处理...")
    numeric_cols = [
        "close", "pct_chg", "total_mv", "circ_mv", "ma5", "ma10",
        "volume_ratio", "macd_dif", "macd_dea", "macd", "rsi_6", "rsi_12", "rsi_24",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_new_features.columns]
    df_new_features[numeric_cols] = df_new_features.groupby("sample_id")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )

    min_days = 30
    days_per_sample = df_new_features.groupby("sample_id").size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    invalid_samples = days_per_sample[days_per_sample < min_days]
    if len(invalid_samples) > 0:
        log.warning(f"过滤 {len(invalid_samples)} 个数据不足样本")
        df_new_features = df_new_features[df_new_features["sample_id"].isin(valid_samples)]

    log.info(f"有效新增特征: {len(df_new_features)} 行, {df_new_features['sample_id'].nunique()} 个样本")

    # 7. 添加增强特征
    log.info("\n添加增强特征...")
    df_new_features = add_enhanced_features(df_new_features)

    # 8. 读取现有 v5 enhanced 文件
    enhanced_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "feature_data_34d_v5_enhanced.csv"
    df_existing_features = pd.read_csv(enhanced_file)
    log.info(f"现有 enhanced: {len(df_existing_features)} 行, {len(df_existing_features.columns)} 列")

    # 9. 对齐列并追加
    # 确保新增数据有所有现有列
    for col in df_existing_features.columns:
        if col not in df_new_features.columns:
            log.warning(f"新增数据缺少列: {col}，填充NaN")
            df_new_features[col] = np.nan

    common_cols = list(df_existing_features.columns)
    df_combined = pd.concat([
        df_existing_features[common_cols],
        df_new_features[common_cols]
    ], ignore_index=True)

    # 10. 保存
    backup_file = enhanced_file.parent / f"feature_data_34d_v5_enhanced_backup_{datetime.now().strftime('%Y%m%d')}.csv"
    df_existing_features.to_csv(backup_file, index=False)
    log.info(f"原文件已备份: {backup_file}")

    df_combined.to_csv(enhanced_file, index=False)
    log.success(f"✓ 追加完成: {len(df_combined)} 行 (新增 {len(df_new_features)} 行)")

    # 11. 同步更新 positive_samples.csv
    df_new.to_csv(
        PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples_v280.csv",
        index=False
    )
    log.success("✓ 正样本列表已更新")


if __name__ == "__main__":
    main()
