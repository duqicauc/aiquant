#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
断点续传版负样本数据准备脚本

特点：
1. 支持断点续传，网络中断后重新运行可继续
2. 按T1日期分组处理，每10个日期保存一次checkpoint
3. 完成后自动清理checkpoint文件
"""
import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


# 配置
BATCH_SIZE = 10  # 每10个T1日期保存一次checkpoint
CHECKPOINT_SAMPLES = PROJECT_ROOT / "data" / "training" / "samples" / ".checkpoint_negative_samples.csv"
CHECKPOINT_FEATURES = PROJECT_ROOT / "data" / "training" / "features" / ".checkpoint_negative_features.csv"

POSITIVE_SAMPLES_FILE = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
OUTPUT_NEGATIVE_SAMPLES = PROJECT_ROOT / "data" / "training" / "samples" / "negative_samples_v2.csv"
OUTPUT_NEGATIVE_FEATURES = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v3.csv"
OUTPUT_STATS = PROJECT_ROOT / "data" / "training" / "samples" / "negative_sample_statistics_v2.json"


def screen_negative_samples_with_checkpoint(dm, positive_samples_df, samples_per_positive=2, random_seed=42):
    """
    带断点续传的负样本筛选
    """
    log.info("=" * 80)
    log.info("负样本筛选（断点续传版）")
    log.info("=" * 80)

    # 确保目录存在
    CHECKPOINT_SAMPLES.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_seed)

    # 获取所有有效股票列表
    all_stocks = dm.get_stock_list(list_status="L")
    all_stocks = all_stocks[
        ~all_stocks["name"].str.contains("ST|退", na=False) & ~all_stocks["ts_code"].str.endswith(".BJ")
    ].copy()
    all_stocks["list_date"] = pd.to_datetime(all_stocks["list_date"])

    log.info(f"有效股票池: {len(all_stocks)} 只")

    # 获取正样本的股票代码集合
    positive_stocks = set(positive_samples_df["ts_code"].unique())
    available_stocks = all_stocks[~all_stocks["ts_code"].isin(positive_stocks)]
    log.info(f"可用负样本股票池: {len(available_stocks)} 只")

    # 加载checkpoint
    processed_t1_dates = set()
    existing_samples = []

    if CHECKPOINT_SAMPLES.exists():
        log.info("发现断点文件，加载已处理的数据...")
        df_checkpoint = pd.read_csv(CHECKPOINT_SAMPLES)
        processed_t1_dates = set(df_checkpoint["t1_date"].unique())
        existing_samples.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(df_checkpoint)} 个负样本 (来自 {len(processed_t1_dates)} 个T1日期)")

    # 按T1日期分组
    t1_groups = positive_samples_df.groupby("t1_date")
    all_t1_dates = list(t1_groups.groups.keys())
    pending_t1_dates = [d for d in all_t1_dates if d not in processed_t1_dates]

    log.info(f"T1日期总数: {len(all_t1_dates)}")
    log.info(f"待处理T1日期: {len(pending_t1_dates)}")

    if len(pending_t1_dates) == 0:
        log.success("所有T1日期已处理完成！")
        if existing_samples:
            return pd.concat(existing_samples, ignore_index=True)
        return pd.DataFrame()

    # 分批处理
    all_new_samples = []
    min_days_before_t1 = 180
    sample_id_offset = len(existing_samples[0]) if existing_samples else 0

    for batch_start in range(0, len(pending_t1_dates), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_t1_dates))
        batch_dates = pending_t1_dates[batch_start:batch_end]

        log.info(f"\n处理批次: T1日期 {batch_start+1}-{batch_end}/{len(pending_t1_dates)}")

        batch_samples = []
        for t1_date in batch_dates:
            group = t1_groups.get_group(t1_date)
            num_positive = len(group)
            num_needed = num_positive * samples_per_positive

            t1_datetime = pd.to_datetime(str(t1_date))
            eligible_stocks = available_stocks[
                available_stocks["list_date"] < t1_datetime - timedelta(days=min_days_before_t1)
            ]

            # 市值分层抽样：按正样本的市值分布来抽样负样本
            if len(eligible_stocks) > 0 and len(group) > 0:
                # 获取正样本的市值（如果有）
                try:
                    # 尝试从正样本数据中获取市值信息
                    # 如果没有，则使用简单的随机抽样
                    if "circ_mv" in group.columns:
                        pos_mv_values = group["circ_mv"].dropna()
                        if len(pos_mv_values) > 0:
                            # 将正样本市值分为5档
                            pos_mv_bins = pd.qcut(pos_mv_values, q=5, duplicates="drop", retbins=True)[1]

                            # 为每档正样本匹配相同市值档位的负样本
                            selected_list = []
                            for _, pos_row in group.iterrows():
                                pos_mv = pos_row.get("circ_mv", None)
                                if pd.notna(pos_mv):
                                    # 找到对应的市值档位
                                    mv_bin_idx = np.digitize([pos_mv], pos_mv_bins)[0] - 1
                                    mv_bin_idx = max(0, min(mv_bin_idx, len(pos_mv_bins) - 2))

                                    # 从相同市值档位中抽取
                                    mv_min = pos_mv_bins[mv_bin_idx] if mv_bin_idx >= 0 else 0
                                    mv_max = (
                                        pos_mv_bins[mv_bin_idx + 1]
                                        if mv_bin_idx + 1 < len(pos_mv_bins)
                                        else float("inf")
                                    )

                                    # 获取该档位的股票（需要先获取市值数据）
                                    # 简化处理：使用市值范围在0.5-2倍之间的股票
                                    mv_range_stocks = eligible_stocks.copy()
                                    # 如果没有市值列，使用随机抽样
                                    if "circ_mv" not in mv_range_stocks.columns:
                                        selected_list.extend(
                                            mv_range_stocks.sample(
                                                n=min(samples_per_positive, len(mv_range_stocks)),
                                                random_state=random_seed,
                                            ).to_dict("records")
                                        )
                                    else:
                                        mv_range_stocks = mv_range_stocks[
                                            (mv_range_stocks["circ_mv"] >= pos_mv * 0.5)
                                            & (mv_range_stocks["circ_mv"] <= pos_mv * 2.0)
                                        ]
                                        if len(mv_range_stocks) > 0:
                                            selected_list.extend(
                                                mv_range_stocks.sample(
                                                    n=min(samples_per_positive, len(mv_range_stocks)),
                                                    random_state=random_seed,
                                                ).to_dict("records")
                                            )
                                else:
                                    # 如果没有市值信息，随机抽取
                                    selected_list.extend(
                                        eligible_stocks.sample(
                                            n=min(samples_per_positive, len(eligible_stocks)), random_state=random_seed
                                        ).to_dict("records")
                                    )

                            if len(selected_list) >= num_needed:
                                selected = pd.DataFrame(selected_list[:num_needed])
                            else:
                                # 如果分层抽样不够，补充随机抽样
                                remaining = num_needed - len(selected_list)
                                additional = eligible_stocks[
                                    ~eligible_stocks["ts_code"].isin([s["ts_code"] for s in selected_list])
                                ].sample(n=min(remaining, len(eligible_stocks)), random_state=random_seed)
                                selected = pd.concat([pd.DataFrame(selected_list), additional], ignore_index=True)
                        else:
                            # 正样本没有市值信息，使用随机抽样
                            selected = eligible_stocks.sample(
                                n=min(num_needed, len(eligible_stocks)), random_state=random_seed
                            )
                    else:
                        # 没有市值列，使用随机抽样
                        selected = eligible_stocks.sample(
                            n=min(num_needed, len(eligible_stocks)), random_state=random_seed
                        )
                except Exception as e:
                    log.warning(f"  T1={t1_date}: 市值分层抽样失败，使用随机抽样: {e}")
                    selected = eligible_stocks.sample(n=min(num_needed, len(eligible_stocks)), random_state=random_seed)
            else:
                selected = pd.DataFrame()

            if len(selected) > 0:

                for idx, stock in selected.iterrows():
                    sample_id_offset += 1
                    batch_samples.append(
                        {
                            "sample_id": sample_id_offset,
                            "ts_code": stock["ts_code"],
                            "name": stock["name"],
                            "t1_date": t1_date,
                            "list_date": (
                                stock["list_date"].strftime("%Y%m%d") if pd.notna(stock["list_date"]) else None
                            ),
                            "label": 0,  # 负样本标签
                        }
                    )

        # 保存批次结果
        if batch_samples:
            df_batch = pd.DataFrame(batch_samples)
            all_new_samples.append(df_batch)

            # 合并并保存checkpoint
            all_data = existing_samples + all_new_samples
            df_checkpoint = pd.concat(all_data, ignore_index=True)
            df_checkpoint.to_csv(CHECKPOINT_SAMPLES, index=False, encoding="utf-8-sig")

            log.info(f"  💾 checkpoint已保存: 累计 {len(df_checkpoint)} 个负样本")

        progress = (batch_end / len(pending_t1_dates)) * 100
        log.info(f"  📊 进度: {progress:.1f}%")

    # 合并最终结果
    all_data = existing_samples + all_new_samples
    if all_data:
        df_samples = pd.concat(all_data, ignore_index=True)
        return df_samples

    return pd.DataFrame()


def extract_features_with_checkpoint(dm, df_negative_samples, lookback_days=34):
    """
    带断点续传的负样本特征提取
    """
    log.info("\n" + "=" * 80)
    log.info("负样本特征提取（断点续传版）")
    log.info("=" * 80)

    # 确保目录存在
    CHECKPOINT_FEATURES.parent.mkdir(parents=True, exist_ok=True)

    # 加载checkpoint
    processed_sample_ids = set()
    existing_features = []

    if CHECKPOINT_FEATURES.exists():
        log.info("发现断点文件，加载已处理的特征...")
        df_checkpoint = pd.read_csv(CHECKPOINT_FEATURES)
        processed_sample_ids = set(df_checkpoint["sample_id"].unique())
        existing_features.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(processed_sample_ids)} 个样本的特征")

    # 过滤待处理样本
    pending_samples = df_negative_samples[~df_negative_samples["sample_id"].isin(processed_sample_ids)].copy()
    log.info(f"待处理样本: {len(pending_samples)} 个")

    if len(pending_samples) == 0:
        log.success("所有样本特征已提取完成！")
        if existing_features:
            return pd.concat(existing_features, ignore_index=True)
        return pd.DataFrame()

    # 分批处理
    all_new_features = []
    batch_size = 100
    error_count = 0

    for batch_start in range(0, len(pending_samples), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_samples))
        batch_samples = pending_samples.iloc[batch_start:batch_end]

        log.info(f"\n提取批次: 样本 {batch_start+1}-{batch_end}/{len(pending_samples)}")

        batch_features = []
        for _, row in batch_samples.iterrows():
            try:
                # 获取T1前lookback_days天的日线数据
                t1_date = str(row["t1_date"])
                end_date = t1_date
                start_date = (pd.to_datetime(t1_date) - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

                df_daily = dm.get_daily_data(row["ts_code"], start_date, end_date)

                if df_daily is None or len(df_daily) < lookback_days:
                    error_count += 1
                    continue

                # 只取T1前的数据
                df_daily = df_daily[df_daily["trade_date"] < t1_date].tail(lookback_days)

                if len(df_daily) < 20:  # 至少需要20天数据
                    error_count += 1
                    continue

                # 添加标识列
                df_daily = df_daily.copy()
                df_daily["sample_id"] = row["sample_id"]
                df_daily["name"] = row["name"]
                df_daily["t1_date"] = t1_date
                df_daily["label"] = 0

                # 计算days_to_t1
                df_daily["days_to_t1"] = range(len(df_daily), 0, -1)

                batch_features.append(df_daily)

            except Exception:
                error_count += 1
                continue

        # 保存批次结果
        if batch_features:
            df_batch = pd.concat(batch_features, ignore_index=True)
            all_new_features.append(df_batch)

            # 合并并保存checkpoint
            all_data = existing_features + all_new_features
            df_checkpoint = pd.concat(all_data, ignore_index=True)
            df_checkpoint.to_csv(CHECKPOINT_FEATURES, index=False, encoding="utf-8-sig")

            log.info(f"  💾 checkpoint已保存: 累计 {df_checkpoint['sample_id'].nunique()} 个样本")

        progress = (batch_end / len(pending_samples)) * 100
        log.info(f"  📊 进度: {progress:.1f}% | 错误: {error_count}")

    # 合并最终结果
    all_data = existing_features + all_new_features
    if all_data:
        df_features = pd.concat(all_data, ignore_index=True)
        return df_features

    return pd.DataFrame()


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("负样本数据准备（断点续传版）")
    log.info("=" * 80)
    log.info("💡 提示：网络中断后重新运行相同命令即可继续")
    log.info("")

    # 配置
    SAMPLES_PER_POSITIVE = 2
    RANDOM_SEED = 42

    # 1. 加载正样本
    log.info("[步骤1] 加载正样本...")
    if not POSITIVE_SAMPLES_FILE.exists():
        log.error(f"正样本文件不存在: {POSITIVE_SAMPLES_FILE}")
        log.error("请先运行 python scripts/prepare_positive_samples_checkpoint.py")
        return

    df_positive = pd.read_csv(POSITIVE_SAMPLES_FILE)
    log.success(f"✓ 加载正样本: {len(df_positive)} 个")

    # 2. 初始化
    log.info("\n[步骤2] 初始化...")
    dm = DataManager()
    log.success("✓ 初始化完成")

    # 3. 筛选负样本
    log.info("\n[步骤3] 筛选负样本...")
    df_negative = screen_negative_samples_with_checkpoint(
        dm, df_positive, samples_per_positive=SAMPLES_PER_POSITIVE, random_seed=RANDOM_SEED
    )

    if df_negative.empty:
        log.error("未找到负样本！")
        return

    # 保存负样本列表
    df_negative.to_csv(OUTPUT_NEGATIVE_SAMPLES, index=False, encoding="utf-8-sig")
    log.success(f"✓ 负样本列表已保存: {OUTPUT_NEGATIVE_SAMPLES}")

    # 清理样本checkpoint
    if CHECKPOINT_SAMPLES.exists():
        os.remove(CHECKPOINT_SAMPLES)
        log.info("✓ 样本checkpoint已清理")

    # 4. 提取特征
    log.info("\n[步骤4] 提取负样本特征...")
    df_features = extract_features_with_checkpoint(dm, df_negative, lookback_days=34)

    if df_features.empty:
        log.error("特征提取失败！")
        return

    # 数据质量处理
    log.info("\n数据质量处理...")
    min_days = 30
    days_per_sample = df_features.groupby("sample_id").size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    df_features = df_features[df_features["sample_id"].isin(valid_samples)]

    log.info(f"有效样本数: {df_features['sample_id'].nunique()}")

    # 保存特征数据
    df_features.to_csv(OUTPUT_NEGATIVE_FEATURES, index=False, encoding="utf-8-sig")
    log.success(f"✓ 负样本特征已保存: {OUTPUT_NEGATIVE_FEATURES}")

    # 清理特征checkpoint
    if CHECKPOINT_FEATURES.exists():
        os.remove(CHECKPOINT_FEATURES)
        log.info("✓ 特征checkpoint已清理")

    # 保存统计信息
    stats = {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "V2 - 同周期其他股票法（断点续传版）",
        "total_negative_samples": len(df_negative),
        "total_positive_samples": len(df_positive),
        "samples_per_positive": SAMPLES_PER_POSITIVE,
        "negative_feature_records": len(df_features),
        "feature_samples": int(df_features["sample_id"].nunique()),
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")

    # 完成
    log.info("\n" + "=" * 80)
    log.success("✅ 负样本数据准备完成！")
    log.info("=" * 80)
    log.info("\n生成的文件:")
    log.info(f"  1. 负样本列表: {OUTPUT_NEGATIVE_SAMPLES}")
    log.info(f"  2. 负样本特征: {OUTPUT_NEGATIVE_FEATURES}")
    log.info("\n下一步: 运行 python scripts/train_v250_model.py")


if __name__ == "__main__":
    main()
