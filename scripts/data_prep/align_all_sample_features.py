#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一对齐所有样本的特征（v5版本）

目标：让正样本、负样本、硬负样本都具有相同的特征集
- 计算三个样本的特征并集
- 把所有样本都对齐到这个并集

输出：统一输出到v5版本文件
- 正样本: data/training/processed/feature_data_34d_v5.csv
- 负样本: data/training/features/negative_feature_data_v2_34d_v5.csv
- 硬负样本: data/training/features/hard_negative_feature_data_34d_v5.csv
"""
import sys
import warnings
from pathlib import Path
from datetime import timedelta
import time

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


# 配置
BATCH_SIZE = 100


def calculate_missing_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """
    计算缺失的特征（本地计算）

    Args:
        df: 个股日线数据
        required_features: 需要的特征列表

    Returns:
        添加了缺失特征的DataFrame
    """
    df = df.copy()
    n = len(df)

    if n < 5:
        return df

    # ========== 1. 均线相关 ==========
    # EMA
    for period in [5, 10, 20, 60]:
        col = f"ema_{period}"
        if col in required_features and col not in df.columns and n >= period:
            df[col] = df["close"].ewm(span=period, adjust=False).mean()

    # MA（如果不存在）
    for period in [5, 10, 20]:
        col = f"ma{period}" if period != 5 else "ma5"
        if col not in df.columns and n >= period:
            df[col] = df["close"].rolling(period).mean()

    # ========== 2. 量比 ==========
    if "vol" in df.columns:
        if "vol_ma5_ratio" in required_features and "vol_ma5_ratio" not in df.columns and n >= 5:
            df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
        if "vol_ma20_ratio" in required_features and "vol_ma20_ratio" not in df.columns and n >= 20:
            df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)
        if "volume_shrink_ratio" in required_features and "volume_shrink_ratio" not in df.columns and n >= 20:
            df["volume_shrink_ratio"] = df["vol"].rolling(5).mean() / (df["vol"].rolling(20).mean() + 1e-8)

    # ========== 3. 乖离率 ==========
    for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
        if name in required_features and name not in df.columns and n >= period:
            ma = df["close"].rolling(period).mean()
            df[name] = (df["close"] - ma) / ma * 100

    # ========== 4. ATR相关 ==========
    if "atr_14" in required_features and "atr_14" not in df.columns and n >= 14:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

    if "atr_ratio_14" in required_features and "atr_ratio_14" not in df.columns:
        if "atr_14" in df.columns:
            df["atr_ratio_14"] = df["atr_14"] / (df["close"] + 1e-8) * 100

    if "atr_expansion" in required_features and "atr_expansion" not in df.columns:
        if "atr_14" in df.columns and n >= 20:
            df["atr_expansion"] = df["atr_14"] / (df["atr_14"].rolling(20).mean() + 1e-8)

    # ========== 5. 最大回撤 ==========
    for period in [10, 20, 55]:
        col = f"max_drawdown_{period}d"
        if col in required_features and col not in df.columns and n >= period:
            rolling_max = df["close"].rolling(period).max()
            df[col] = (df["close"] - rolling_max) / rolling_max * 100

    # ========== 6. 距离高点天数 ==========
    for period in [20, 55]:
        col = f"days_from_high_{period}d"
        if col in required_features and col not in df.columns and n >= period:
            days_list = []
            for i in range(n):
                if i < period:
                    days_list.append(np.nan)
                else:
                    window = df["close"].iloc[i - period + 1 : i + 1]
                    max_idx = window.idxmax()
                    days_list.append(i - max_idx)
            df[col] = days_list

    # ========== 7. 回撤恢复比 ==========
    if "recovery_ratio_20d" in required_features and "recovery_ratio_20d" not in df.columns and n >= 20:
        if "max_drawdown_20d" in df.columns:
            rolling_max = df["close"].rolling(20).max()
            rolling_min = df["close"].rolling(20).min()
            df["recovery_ratio_20d"] = (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-8)

    # ========== 8. 涨停标志 ==========
    if "is_limit_up" in required_features and "is_limit_up" not in df.columns:
        if "pct_chg" in df.columns:
            df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # ========== 9. 换手率 ==========
    # turnover_rate_f 需要从daily_basic获取，这里跳过，设为NaN
    if "turnover_rate_f" in required_features and "turnover_rate_f" not in df.columns:
        df["turnover_rate_f"] = np.nan

    # ========== 10. OBV ==========
    if "obv" in required_features and "obv" not in df.columns and "vol" in df.columns:
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()

    # ========== 10.5 KDJ ==========
    if any(k in required_features and k not in df.columns for k in ["kdj_k", "kdj_d", "kdj_j"]) and n >= 9:
        low_min = df["low"].rolling(9).min()
        high_max = df["high"].rolling(9).max()
        rsv = (df["close"] - low_min) / (high_max - low_min + 1e-8) * 100

        # 计算K、D、J
        kdj_k = rsv.ewm(com=2, adjust=False).mean()  # K线，使用EMA平滑，alpha=1/3
        kdj_d = kdj_k.ewm(com=2, adjust=False).mean()  # D线
        kdj_j = 3 * kdj_k - 2 * kdj_d  # J线

        if "kdj_k" in required_features and "kdj_k" not in df.columns:
            df["kdj_k"] = kdj_k
        if "kdj_d" in required_features and "kdj_d" not in df.columns:
            df["kdj_d"] = kdj_d
        if "kdj_j" in required_features and "kdj_j" not in df.columns:
            df["kdj_j"] = kdj_j

    # ========== 11. 价格区间和波动 ==========
    if "price_range_pct" in required_features and "price_range_pct" not in df.columns and n >= 20:
        df["price_range_pct"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["close"] * 100

    if "volatility_vs_hist" in required_features and "volatility_vs_hist" not in df.columns and n >= 60:
        vol_20 = (
            df["pct_chg"].rolling(20).std() if "pct_chg" in df.columns else df["close"].pct_change().rolling(20).std()
        )
        vol_60 = (
            df["pct_chg"].rolling(60).std() if "pct_chg" in df.columns else df["close"].pct_change().rolling(60).std()
        )
        df["volatility_vs_hist"] = vol_20 / (vol_60 + 1e-8)

    # ========== 12. 价格相对历史位置 ==========
    if "price_vs_hist_high" in required_features and "price_vs_hist_high" not in df.columns and n >= 55:
        hist_high = df["close"].rolling(55).max()
        df["price_vs_hist_high"] = (df["close"] - hist_high) / hist_high * 100

    if "price_vs_hist_mean" in required_features and "price_vs_hist_mean" not in df.columns and n >= 55:
        hist_mean = df["close"].rolling(55).mean()
        df["price_vs_hist_mean"] = (df["close"] - hist_mean) / hist_mean * 100

    # ========== 13. MA10相关 ==========
    if "close_vs_ma10_std" in required_features and "close_vs_ma10_std" not in df.columns and n >= 20:
        ma10 = df["close"].rolling(10).mean()
        diff = df["close"] - ma10
        df["close_vs_ma10_std"] = diff / (diff.rolling(20).std() + 1e-8)

    if "days_near_ma10" in required_features and "days_near_ma10" not in df.columns and n >= 10:
        ma10 = df["close"].rolling(10).mean()
        near_ma10 = (abs(df["close"] - ma10) / df["close"] < 0.02).astype(int)
        df["days_near_ma10"] = near_ma10.rolling(10).sum()

    if "ma10_cross_count" in required_features and "ma10_cross_count" not in df.columns and n >= 20:
        ma10 = df["close"].rolling(10).mean()
        cross = ((df["close"] > ma10) != (df["close"].shift(1) > ma10.shift(1))).astype(int)
        df["ma10_cross_count"] = cross.rolling(20).sum()

    # ========== 14. 市场相关（需要从外部获取，这里设为NaN或0） ==========
    for col in [
        "market_pct_chg",
        "market_return_34d",
        "market_volatility_34d",
        "excess_return",
        "excess_return_cumsum",
        "market_trend",
    ]:
        if col in required_features and col not in df.columns:
            df[col] = np.nan

    return df


def process_sample_batch(
    dm: DataManager, features_df: pd.DataFrame, sample_ids: list, required_features: list, max_lookback: int = 60
) -> list:
    """
    批量处理样本，补充缺失特征
    """
    results = []

    for sample_id in sample_ids:
        sample_data = features_df[features_df["sample_id"] == sample_id].copy()

        if sample_data.empty:
            continue

        ts_code = sample_data["ts_code"].iloc[0]

        # 检查需要补充哪些特征
        existing_cols = set(sample_data.columns)
        missing_cols = [c for c in required_features if c not in existing_cols]

        if not missing_cols:
            # 无需补充
            results.append(sample_data)
            continue

        try:
            # 获取日期范围
            min_date = pd.to_datetime(sample_data["trade_date"]).min()
            max_date = pd.to_datetime(sample_data["trade_date"]).max()

            extended_start = (min_date - timedelta(days=max_lookback + 30)).strftime("%Y%m%d")
            end_date = max_date.strftime("%Y%m%d")

            # 获取日线数据
            df_daily = dm.get_daily_data(ts_code, extended_start, end_date)

            if df_daily is None or df_daily.empty:
                # 无法获取数据，填充NaN
                for col in missing_cols:
                    sample_data[col] = np.nan
                results.append(sample_data)
                continue

            # 计算缺失特征
            df_with_features = calculate_missing_features(df_daily, missing_cols)
            df_with_features["trade_date"] = pd.to_datetime(df_with_features["trade_date"])

            # 只取需要补充的列
            cols_to_add = [c for c in missing_cols if c in df_with_features.columns]

            if cols_to_add:
                sample_data["trade_date"] = pd.to_datetime(sample_data["trade_date"])
                merged = pd.merge(
                    sample_data, df_with_features[["trade_date"] + cols_to_add], on="trade_date", how="left"
                )
                results.append(merged)
            else:
                # 无法计算的特征，填充NaN
                for col in missing_cols:
                    sample_data[col] = np.nan
                results.append(sample_data)

        except Exception:
            # 出错时填充NaN
            for col in missing_cols:
                sample_data[col] = np.nan
            results.append(sample_data)

    return results


def align_sample_features(
    dm: DataManager,
    input_file: Path,
    output_file: Path,
    checkpoint_file: Path,
    required_features: list,
    batch_size: int = 100,
):
    """
    对齐样本特征（带断点续传）
    """
    log.info(f"处理文件: {input_file.name}")

    # 加载数据
    df = pd.read_csv(input_file)
    # 处理日期格式
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")

    log.info(f"  加载: {len(df)} 条, 特征数: {len(df.columns)}")

    # 检查缺失的特征
    existing_cols = set(df.columns)
    missing_cols = [c for c in required_features if c not in existing_cols]

    if not missing_cols:
        log.success("  所有特征都已存在，无需处理")
        df.to_csv(output_file, index=False)
        return

    log.info(f"  需要补充 {len(missing_cols)} 个特征")

    # 获取所有样本ID
    all_sample_ids = df["sample_id"].unique().tolist()
    total_samples = len(all_sample_ids)

    # 检查断点
    processed_ids = set()
    processed_results = []

    if checkpoint_file.exists():
        log.info("  发现断点，加载已处理数据...")
        df_checkpoint = pd.read_csv(checkpoint_file)
        df_checkpoint["trade_date"] = pd.to_datetime(df_checkpoint["trade_date"], format="mixed", errors="coerce")
        processed_ids = set(df_checkpoint["sample_id"].unique())
        processed_results.append(df_checkpoint)
        log.info(f"  已处理: {len(processed_ids)} 个样本")

    # 筛选待处理样本
    remaining_ids = [sid for sid in all_sample_ids if sid not in processed_ids]
    log.info(f"  待处理: {len(remaining_ids)} 个样本")

    if not remaining_ids:
        log.success("  所有样本已处理完成")
        if processed_results:
            final_df = pd.concat(processed_results, ignore_index=True)
            final_df.to_csv(output_file, index=False)
        return

    # 批量处理
    batch_results = processed_results.copy()

    for i in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[i : i + batch_size]
        current_batch = i // batch_size + 1
        total_batches = (len(remaining_ids) + batch_size - 1) // batch_size

        log.info(f"  批次 {current_batch}/{total_batches}")

        batch_df = df[df["sample_id"].isin(batch_ids)]
        batch_result = process_sample_batch(dm, batch_df, batch_ids, missing_cols)

        if batch_result:
            batch_df_result = pd.concat(batch_result, ignore_index=True)
            batch_results.append(batch_df_result)

            # 保存断点
            checkpoint_df = pd.concat(batch_results, ignore_index=True)
            checkpoint_df.to_csv(checkpoint_file, index=False)

        # 进度
        progress = (len(processed_ids) + i + len(batch_ids)) / total_samples * 100
        log.info(f"    进度: {progress:.1f}%")

        time.sleep(0.3)

    # 保存最终结果
    if batch_results:
        final_df = pd.concat(batch_results, ignore_index=True)
        final_df = final_df.ffill().bfill()
        final_df.to_csv(output_file, index=False)

        log.success(f"  完成！特征数: {len(final_df.columns)}")

        # 清理断点
        if checkpoint_file.exists():
            checkpoint_file.unlink()


def main():
    log.info("=" * 80)
    log.info("统一对齐所有样本特征（统一输出v5版本）")
    log.info("=" * 80)

    # 输入文件路径（各自最新版本）
    pos_input = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    neg_input = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    hard_neg_input = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v4.csv"

    # 输出文件路径（统一v5版本）
    pos_output = pos_input  # 覆盖原文件
    neg_output = neg_input  # 覆盖原文件
    hard_neg_output = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    # 检查文件存在
    if not pos_input.exists():
        log.error(f"正样本文件不存在: {pos_input}")
        return
    if not neg_input.exists():
        log.error(f"负样本文件不存在: {neg_input}")
        return
    if not hard_neg_input.exists():
        log.error(f"硬负样本文件不存在: {hard_neg_input}")
        return

    # [步骤1] 分析特征并集
    log.info("\n[步骤1] 分析特征并集...")
    df_pos = pd.read_csv(pos_input)
    df_neg = pd.read_csv(neg_input)
    df_hard = pd.read_csv(hard_neg_input)

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
    hard_cols = set(df_hard.columns) - exclude_cols

    log.info(f"  正样本v5特征数: {len(pos_cols)}")
    log.info(f"  负样本v5特征数: {len(neg_cols)}")
    log.info(f"  硬负样本v4特征数: {len(hard_cols)}")

    # 特征并集（目标特征集）
    union_features = pos_cols | neg_cols | hard_cols
    target_features = sorted(list(union_features))
    log.info(f"  特征并集: {len(target_features)}")

    # 各自缺失的特征
    pos_missing = union_features - pos_cols
    neg_missing = union_features - neg_cols
    hard_missing = union_features - hard_cols

    log.info(f"\n  正样本需补充: {len(pos_missing)} 个特征")
    if pos_missing:
        log.info(f"    {sorted(pos_missing)}")
    log.info(f"  负样本需补充: {len(neg_missing)} 个特征")
    if neg_missing:
        log.info(f"    {sorted(neg_missing)}")
    log.info(f"  硬负样本需补充: {len(hard_missing)} 个特征")
    if hard_missing:
        log.info(f"    {sorted(hard_missing)}")

    # 初始化DataManager
    log.info("\n[步骤2] 初始化数据管理器...")
    dm = DataManager(source="tushare")
    log.success("✓ 初始化完成")

    # 处理正样本
    if pos_missing:
        log.info("\n[步骤3] 对齐正样本特征...")
        pos_checkpoint = PROJECT_ROOT / "data" / "training" / "processed" / ".checkpoint_align_pos.csv"
        align_sample_features(dm, pos_input, pos_output, pos_checkpoint, target_features, BATCH_SIZE)
    else:
        log.info("\n[步骤3] 正样本特征已完整，跳过")

    # 处理负样本
    if neg_missing:
        log.info("\n[步骤4] 对齐负样本特征...")
        neg_checkpoint = PROJECT_ROOT / "data" / "training" / "features" / ".checkpoint_align_neg.csv"
        align_sample_features(dm, neg_input, neg_output, neg_checkpoint, target_features, BATCH_SIZE)
    else:
        log.info("\n[步骤4] 负样本特征已完整，跳过")

    # 处理硬负样本
    if hard_missing:
        log.info("\n[步骤5] 对齐硬负样本特征...")
        hard_checkpoint = PROJECT_ROOT / "data" / "training" / "features" / ".checkpoint_align_hard.csv"
        align_sample_features(dm, hard_neg_input, hard_neg_output, hard_checkpoint, target_features, BATCH_SIZE)
    else:
        # 如果硬负样本不需要补充，直接复制为v5
        log.info("\n[步骤5] 硬负样本特征已完整，直接保存为v5...")
        df_hard.to_csv(hard_neg_output, index=False)
        log.success(f"  ✓ 已保存: {hard_neg_output.name}")

    # 验证结果
    log.info("\n[步骤6] 验证特征对齐...")
    df_pos = pd.read_csv(pos_output)
    df_neg = pd.read_csv(neg_output)
    df_hard = pd.read_csv(hard_neg_output)

    pos_cols = set(df_pos.columns) - exclude_cols
    neg_cols = set(df_neg.columns) - exclude_cols
    hard_cols = set(df_hard.columns) - exclude_cols

    common_all = pos_cols & neg_cols & hard_cols

    log.info(f"  正样本v5特征数: {len(pos_cols)}")
    log.info(f"  负样本v5特征数: {len(neg_cols)}")
    log.info(f"  硬负样本v5特征数: {len(hard_cols)}")
    log.info(f"  共同特征数: {len(common_all)}")

    # 检查差异
    pos_only = pos_cols - neg_cols - hard_cols
    neg_only = neg_cols - pos_cols - hard_cols
    hard_only = hard_cols - pos_cols - neg_cols

    if pos_only or neg_only or hard_only:
        log.warning("仍有特征差异:")
        if pos_only:
            log.warning(f"  正样本独有: {len(pos_only)} - {sorted(pos_only)}")
        if neg_only:
            log.warning(f"  负样本独有: {len(neg_only)} - {sorted(neg_only)}")
        if hard_only:
            log.warning(f"  硬负样本独有: {len(hard_only)} - {sorted(hard_only)}")
    else:
        log.success("✓ 所有样本特征完全对齐！")

    log.info("\n" + "=" * 80)
    log.success("✅ 特征对齐完成！所有文件统一使用v5版本")
    log.info("=" * 80)
    log.info("\n输出文件:")
    log.info(f"  - {pos_output}")
    log.info(f"  - {neg_output}")
    log.info(f"  - {hard_neg_output}")
    log.info("\n下一步: python scripts/train_v250_model.py")


if __name__ == "__main__":
    main()
