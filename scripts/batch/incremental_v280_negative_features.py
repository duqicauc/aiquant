#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 增量负样本特征提取

为新增正样本生成对应的负样本，提取完整特征后追加到 v5 enhanced 负样本文件。

流程：
1. 读取现有正/负样本列表
2. 确定新增正样本（2025-12-27 ~ 2026-04-14）
3. 用 NegativeSampleScreenerV2 生成负样本
4. extract_features(lookback_days=70)
5. 补充缺失列（open/high/low/vol/turnover_rate 等）
6. enrich_features_v6.py 计算高级特征
7. calculate_missing_features + add_features_inplace 补充 v5 缺失特征
8. 截取 days_to_t1 ∈ [-34, -1] 的行
9. 添加增强特征
10. 追加到 v5 enhanced 负样本文件
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log

# Import enrich_features_v6 functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from enrich_features_v6 import (
    calculate_basic_features,
    calculate_breakout_features,
    add_market_features,
    calculate_interaction_features,
    get_market_data,
)

# Import additional feature calculators to reach full v5 165-column set
from scripts.data_prep.align_all_sample_features import calculate_missing_features
from scripts.regenerate_v6_negative_features import add_features_inplace


def supplement_missing_columns(df: pd.DataFrame, dm: DataManager, cols: list) -> pd.DataFrame:
    """从 DataManager 补充缺失列（按 sample_id 分组调用 get_complete_data）"""
    if not cols:
        return df

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    groups = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        ts_code = group["ts_code"].iloc[0]
        start_date = group["trade_date"].min().strftime("%Y%m%d")
        end_date = group["trade_date"].max().strftime("%Y%m%d")

        try:
            df_complete = dm.get_complete_data(ts_code, start_date, end_date)
            if not df_complete.empty:
                df_complete["trade_date"] = pd.to_datetime(df_complete["trade_date"])
                merge_cols = [c for c in cols if c in df_complete.columns and c not in group.columns]
                if merge_cols:
                    group = pd.merge(group, df_complete[["trade_date"] + merge_cols], on="trade_date", how="left")
        except Exception as e:
            log.warning(f"补充缺失列失败 {ts_code}: {e}")

        groups.append(group)

    return pd.concat(groups, ignore_index=True)


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加增强特征（与 feature_analysis_and_enhancement.py 一致）"""
    df = df.copy()
    added_features = []

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

    if "rsi_6" in df.columns and "kdj_j" in df.columns and "kdj_k" in df.columns:
        df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
        added_features.append("rsi_kdj_golden_cross")
        df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
        added_features.append("rsi_kdj_strength")
        df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))
        added_features.append("rsi_zone")

    if "close" in df.columns and "vol" in df.columns:
        price_change_10d = df["close"].pct_change(10)
        vol_change_10d = df["vol"].pct_change(10)
        df["volume_price_divergence_strength"] = np.abs(price_change_10d - vol_change_10d)
        added_features.append("volume_price_divergence_strength")
        df["volume_price_confirm"] = ((price_change_10d > 0) == (vol_change_10d > 0)).astype(int)
        added_features.append("volume_price_confirm")

    breakout_cols = [c for c in df.columns if "breakout_strength" in c]
    if len(breakout_cols) >= 2:
        df["breakout_strength_avg"] = df[breakout_cols].mean(axis=1)
        added_features.append("breakout_strength_avg")
        df["breakout_strength_max"] = df[breakout_cols].max(axis=1)
        added_features.append("breakout_strength_max")

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

    if "momentum_10d" in df.columns and "momentum_acceleration" not in df.columns:
        df["momentum_acceleration"] = df["momentum_10d"].diff(5)
        added_features.append("momentum_acceleration")

    position_cols = [c for c in df.columns if "price_position" in c]
    if len(position_cols) >= 2:
        df["price_position_avg"] = df[position_cols].mean(axis=1)
        added_features.append("price_position_avg")

    if "return_34d" in df.columns and "volatility_34d" in df.columns:
        df["sharpe_like_34d"] = df["return_34d"] / (df["volatility_34d"] + 1e-8)
        added_features.append("sharpe_like_34d")

    log.info(f"  添加了 {len(added_features)} 个增强特征")
    return df


def main():
    log.info("=" * 80)
    log.info("v2.8.0 增量负样本特征提取")
    log.info("=" * 80)

    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)

    # 1. 读取现有正样本和负样本列表
    existing_pos_file = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
    existing_neg_file = PROJECT_ROOT / "data" / "training" / "samples" / "negative_samples_v2.csv"

    df_existing_pos = pd.read_csv(existing_pos_file)
    log.info(f"现有正样本: {len(df_existing_pos)} 条")

    if existing_neg_file.exists():
        df_existing_neg = pd.read_csv(existing_neg_file)
        log.info(f"现有负样本列表: {len(df_existing_neg)} 条")
    else:
        df_existing_neg = pd.DataFrame()
        log.warning("未找到现有负样本列表文件")

    # 2. 确定新增正样本（日期 >= 2025-12-27）
    log.info("\n确定新增正样本 (>= 20251227)...")
    df_existing_pos["t1_date"] = df_existing_pos["t1_date"].astype(str)
    df_new_pos = df_existing_pos[df_existing_pos["t1_date"] >= "20251227"].copy()
    log.info(f"新增正样本: {len(df_new_pos)} 条")

    if df_new_pos.empty:
        log.info("无新增正样本，跳过负样本生成")
        return

    # 3. 生成负样本（每个新增正样本对应 2 个负样本）
    log.info("\n生成负样本...")
    df_new_neg = screener.screen_negative_samples(
        positive_samples_df=df_new_pos,
        samples_per_positive=2,
        random_seed=42,
    )
    log.info(f"生成负样本: {len(df_new_neg)} 条")

    if df_new_neg.empty:
        log.error("负样本生成失败")
        return

    # 4. 分配新的 sample_id
    enhanced_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "negative_feature_data_v2_34d_v5_enhanced.csv"
    if enhanced_file.exists():
        df_existing_features = pd.read_csv(enhanced_file, usecols=["sample_id"])
        max_existing_id = int(df_existing_features["sample_id"].max())
    else:
        max_existing_id = 0

    df_new_neg = df_new_neg.reset_index(drop=True)
    df_new_neg["sample_id"] = range(max_existing_id + 1, max_existing_id + 1 + len(df_new_neg))
    log.info(f"新 sample_id 范围: {max_existing_id + 1} ~ {max_existing_id + len(df_new_neg)}")

    # 5. 提取 70 天基础特征
    log.info("\n提取新增负样本的70天基础特征...")
    df_features_70d = screener.extract_features(df_new_neg, lookback_days=70)

    if df_features_70d.empty:
        log.error("负样本特征提取失败")
        return

    log.info(f"70天基础特征: {len(df_features_70d)} 行, {len(df_features_70d.columns)} 列")

    # 6. 数据质量处理
    numeric_cols = [
        "close", "pct_chg", "total_mv", "circ_mv", "ma5", "ma10",
        "volume_ratio", "macd_dif", "macd_dea", "macd", "rsi_6", "rsi_12", "rsi_24",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_features_70d.columns]
    df_features_70d[numeric_cols] = df_features_70d.groupby("sample_id")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # 6.5 补充缺失的基础列（open/high/low/vol/turnover_rate 等）
    log.info("\n补充缺失的基础列...")
    missing_base_cols = [c for c in ["open", "high", "low", "vol", "turnover_rate", "pre_close", "change", "amount"] if c not in df_features_70d.columns]
    if missing_base_cols:
        log.info(f"  需要补充: {missing_base_cols}")
        df_features_70d = supplement_missing_columns(df_features_70d, dm, missing_base_cols)
        log.info(f"  补充后列数: {len(df_features_70d.columns)}")

    # 7. 运行 v6 特征工程
    log.info("\n运行 v6 特征工程...")

    df_market = get_market_data(dm, "20250101", "20261231")
    log.info(f"  市场数据: {len(df_market)} 条")

    log.info("  计算基础技术特征...")
    df_features_70d = calculate_basic_features(df_features_70d)
    log.info(f"  基础特征后: {len(df_features_70d.columns)} 列")

    log.info("  添加突破特征...")
    df_features_70d = calculate_breakout_features(df_features_70d)

    log.info("  添加市场环境特征...")
    df_features_70d = add_market_features(df_features_70d, df_market)

    log.info("  添加交互特征...")
    df_features_70d = calculate_interaction_features(df_features_70d)

    log.info(f"  v6 特征工程完成: {len(df_features_70d.columns)} 列")

    # 7.5 补充 v5 缺失特征（对齐到完整 165 列）
    log.info("\n补充 v5 缺失特征...")

    v5_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    if v5_file.exists():
        v5_cols = list(pd.read_csv(v5_file, nrows=0).columns)
    else:
        # fallback to positive v5 columns
        v5_cols = list(pd.read_csv(PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv", nrows=0).columns)

    missing_cols = [c for c in v5_cols if c not in df_features_70d.columns]
    log.info(f"  需要补充 {len(missing_cols)} 个特征")

    if missing_cols:
        log.info("  按样本分组计算缺失特征（calculate_missing_features）...")
        groups = []
        for sample_id, group in df_features_70d.groupby("sample_id", sort=False):
            group = group.sort_values("trade_date").reset_index(drop=True)
            group = calculate_missing_features(group, missing_cols)
            groups.append(group)
        df_features_70d = pd.concat(groups, ignore_index=True)
        log.info(f"  补充后: {len(df_features_70d.columns)} 列")

    log.info("  运行 add_features_inplace...")
    df_features_70d = add_features_inplace(df_features_70d)
    log.info(f"  最终特征数: {len(df_features_70d.columns)} 列")

    # 8. 截取最后 34 天
    log.info("\n截取最后34天（转换为v5格式）...")
    df_features_34d = df_features_70d[
        df_features_70d["days_to_t1"] >= -34
    ].copy()
    log.info(f"截取后: {len(df_features_34d)} 行, {len(df_features_34d.columns)} 列")

    # 9. 添加增强特征
    log.info("\n添加增强特征...")
    df_features_34d = add_enhanced_features(df_features_34d)

    # 10. 读取现有 enhanced 文件并对齐列
    if enhanced_file.exists():
        df_existing_features = pd.read_csv(enhanced_file)
        log.info(f"现有 enhanced: {len(df_existing_features)} 行, {len(df_existing_features.columns)} 列")
    else:
        log.warning("未找到现有 enhanced 文件，将创建新文件")
        df_existing_features = pd.DataFrame()

    # 对齐列
    if not df_existing_features.empty:
        for col in df_existing_features.columns:
            if col not in df_features_34d.columns:
                log.warning(f"新增数据缺少列: {col}，填充NaN")
                df_features_34d[col] = np.nan

        common_cols = list(df_existing_features.columns)
        df_combined = pd.concat([
            df_existing_features[common_cols],
            df_features_34d[common_cols]
        ], ignore_index=True)
    else:
        df_combined = df_features_34d.copy()

    # 11. 保存
    backup_file = enhanced_file.parent / f"negative_feature_data_v2_34d_v5_enhanced_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if enhanced_file.exists():
        df_existing_features.to_csv(backup_file, index=False)
        log.info(f"原文件已备份: {backup_file}")

    df_combined.to_csv(enhanced_file, index=False)
    log.success(f"✓ 追加完成: {len(df_combined)} 行 (新增 {len(df_features_34d)} 行)")

    # 12. 同步更新负样本列表
    df_all_neg = pd.concat([df_existing_neg, df_new_neg], ignore_index=True)
    df_all_neg.to_csv(existing_neg_file, index=False)
    log.success(f"✓ 负样本列表已更新: {len(df_all_neg)} 条")

    # 13. 统计
    log.info("\n" + "=" * 80)
    log.info("增量负样本更新统计")
    log.info("=" * 80)
    log.info(f"新增负样本: {len(df_new_neg)} 个")
    log.info(f"新增特征行: {len(df_features_34d)} 行")
    log.info(f"合并后总样本: {df_combined['sample_id'].nunique()} 个")
    log.info(f"合并后总行数: {len(df_combined)} 行")
    log.info(f"合并后总列数: {len(df_combined.columns)} 列")


if __name__ == "__main__":
    main()
