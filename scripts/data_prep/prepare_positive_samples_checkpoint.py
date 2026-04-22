#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
断点续传版正样本数据准备脚本

特点：
1. 支持断点续传，网络中断后重新运行可继续
2. 每处理100只股票保存一次checkpoint
3. 完成后自动清理checkpoint文件
4. 特征提取也支持断点续传
"""
import sys
import os
import warnings
from pathlib import Path

import pandas as pd

# 过滤 pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log
from config.settings import settings


# 配置
BATCH_SIZE = 100  # 每批处理100只股票
CHECKPOINT_SAMPLES = PROJECT_ROOT / "data" / "training" / "samples" / ".checkpoint_positive.csv"
CHECKPOINT_FEATURES = PROJECT_ROOT / "data" / "training" / "processed" / ".checkpoint_features.csv"
SAMPLES_FILE = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
FEATURES_FILE = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v3.csv"


def screen_positive_samples_with_checkpoint(dm, screener, start_date, end_date):
    """
    带断点续传的正样本扫描
    """
    log.info("=" * 80)
    log.info("正样本扫描（断点续传版）")
    log.info("=" * 80)

    # 确保目录存在
    CHECKPOINT_SAMPLES.parent.mkdir(parents=True, exist_ok=True)

    # 1. 获取股票列表
    stock_list = dm.get_stock_list(list_status="L")

    # 过滤ST、退市整理、北交所
    stock_list = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False) & ~stock_list["ts_code"].str.endswith(".BJ")
    ].copy()

    # 确保list_date是datetime类型
    if stock_list["list_date"].dtype in ["int64", "float64"]:
        stock_list["list_date"] = pd.to_datetime(stock_list["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif stock_list["list_date"].dtype == "object":
        stock_list["list_date"] = pd.to_datetime(stock_list["list_date"], format="%Y%m%d", errors="coerce")
    else:
        stock_list["list_date"] = pd.to_datetime(stock_list["list_date"], errors="coerce")

    total_stocks = len(stock_list)
    log.info(f"有效股票总数: {total_stocks}")

    # 2. 加载checkpoint
    processed_stocks = set()
    existing_samples = []

    if CHECKPOINT_SAMPLES.exists():
        log.info("发现断点文件，加载已处理的数据...")
        df_checkpoint = pd.read_csv(CHECKPOINT_SAMPLES)
        processed_stocks = set(df_checkpoint["ts_code"].unique())
        existing_samples.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(df_checkpoint)} 个样本 (来自 {len(processed_stocks)} 只股票)")

    # 3. 过滤待处理股票
    pending_stocks = stock_list[~stock_list["ts_code"].isin(processed_stocks)].copy()
    log.info(f"待处理股票: {len(pending_stocks)} 只")

    if len(pending_stocks) == 0:
        log.success("所有股票已处理完成！")
        if existing_samples:
            return pd.concat(existing_samples, ignore_index=True)
        return pd.DataFrame()

    # 4. 分批处理
    all_new_samples = []
    batch_count = 0
    error_count = 0

    for batch_start in range(0, len(pending_stocks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_stocks))
        batch_stocks = pending_stocks.iloc[batch_start:batch_end]
        batch_count += 1

        log.info(f"\n处理批次 {batch_count}: 股票 {batch_start+1}-{batch_end}/{len(pending_stocks)}")

        batch_samples = []
        for idx, row in batch_stocks.iterrows():
            ts_code = row["ts_code"]
            name = row["name"]
            list_date = row["list_date"]

            # 确保list_date是pd.Timestamp类型
            if not isinstance(list_date, pd.Timestamp):
                if isinstance(list_date, str):
                    list_date = pd.to_datetime(list_date, format="%Y%m%d", errors="coerce")
                else:
                    list_date = pd.to_datetime(list_date, errors="coerce")

            try:
                # 调用筛选器的内部方法筛选单只股票
                samples = screener._screen_single_stock(ts_code, name, list_date, start_date, end_date)

                if samples:
                    batch_samples.extend(samples)
                    log.success(f"  ✓ {ts_code} {name}: 找到 {len(samples)} 个样本")

            except Exception as e:
                error_count += 1
                log.warning(f"  ✗ {ts_code} {name}: {e}")
                continue

        # 保存批次结果到checkpoint
        if batch_samples:
            df_batch = pd.DataFrame(batch_samples)
            all_new_samples.append(df_batch)

            # 合并所有数据并保存checkpoint
            all_data = existing_samples + all_new_samples
            df_checkpoint = pd.concat(all_data, ignore_index=True)
            df_checkpoint.to_csv(CHECKPOINT_SAMPLES, index=False, encoding="utf-8-sig")

            log.info(f"  💾 checkpoint已保存: 累计 {len(df_checkpoint)} 个样本")

        # 显示进度
        progress = (batch_end / len(pending_stocks)) * 100
        total_samples = sum(len(df) for df in existing_samples + all_new_samples)
        log.info(f"  📊 进度: {progress:.1f}% | 累计样本: {total_samples} | 错误: {error_count}")

    # 5. 合并最终结果
    all_data = existing_samples + all_new_samples
    if all_data:
        df_samples = pd.concat(all_data, ignore_index=True)

        # 保存最终结果
        df_samples.to_csv(SAMPLES_FILE, index=False, encoding="utf-8-sig")
        log.success(f"✓ 正样本已保存: {SAMPLES_FILE}")

        # 清理checkpoint
        if CHECKPOINT_SAMPLES.exists():
            os.remove(CHECKPOINT_SAMPLES)
            log.info("✓ checkpoint文件已清理")

        return df_samples

    return pd.DataFrame()


def extract_features_with_checkpoint(screener, df_samples, lookback_days=34):
    """
    带断点续传的特征提取
    """
    log.info("\n" + "=" * 80)
    log.info("特征提取（断点续传版）")
    log.info("=" * 80)

    # 确保目录存在
    CHECKPOINT_FEATURES.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载checkpoint
    processed_sample_ids = set()
    existing_features = []

    if CHECKPOINT_FEATURES.exists():
        log.info("发现断点文件，加载已处理的特征...")
        df_checkpoint = pd.read_csv(CHECKPOINT_FEATURES)
        processed_sample_ids = set(df_checkpoint["sample_id"].unique())
        existing_features.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(processed_sample_ids)} 个样本的特征")

    # 2. 为每个样本添加sample_id
    df_samples = df_samples.copy()
    df_samples["sample_id"] = range(len(df_samples))

    # 3. 过滤待处理样本
    pending_samples = df_samples[~df_samples["sample_id"].isin(processed_sample_ids)].copy()
    log.info(f"待处理样本: {len(pending_samples)} 个")

    if len(pending_samples) == 0:
        log.success("所有样本特征已提取完成！")
        if existing_features:
            return pd.concat(existing_features, ignore_index=True)
        return pd.DataFrame()

    # 4. 分批处理
    all_new_features = []
    batch_size = 50  # 每50个样本保存一次
    error_count = 0

    for batch_start in range(0, len(pending_samples), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_samples))
        batch_samples = pending_samples.iloc[batch_start:batch_end]

        log.info(f"\n提取批次: 样本 {batch_start+1}-{batch_end}/{len(pending_samples)}")

        batch_features = []
        for _, row in batch_samples.iterrows():
            try:
                # 提取单个样本的特征
                features = screener._extract_single_sample_features(
                    row["ts_code"], row["name"], row["t1_date"], lookback_days, row["sample_id"]
                )

                if features is not None and not features.empty:
                    batch_features.append(features)

            except Exception as e:
                error_count += 1
                log.warning(f"  ✗ 样本 {row['sample_id']} ({row['ts_code']}): {e}")
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

        # 显示进度
        progress = (batch_end / len(pending_samples)) * 100
        log.info(f"  📊 进度: {progress:.1f}% | 错误: {error_count}")

    # 5. 合并最终结果
    all_data = existing_features + all_new_features
    if all_data:
        df_features = pd.concat(all_data, ignore_index=True)

        # 数据质量处理
        log.info("\n数据质量处理...")

        # 过滤数据不足的样本
        min_days = 30
        days_per_sample = df_features.groupby("sample_id").size()
        valid_samples = days_per_sample[days_per_sample >= min_days].index
        df_features = df_features[df_features["sample_id"].isin(valid_samples)]

        log.info(f"有效样本数: {df_features['sample_id'].nunique()}")

        # 保存最终结果
        df_features.to_csv(FEATURES_FILE, index=False, encoding="utf-8-sig")
        log.success(f"✓ 特征数据已保存: {FEATURES_FILE}")

        # 清理checkpoint
        if CHECKPOINT_FEATURES.exists():
            os.remove(CHECKPOINT_FEATURES)
            log.info("✓ checkpoint文件已清理")

        return df_features

    return pd.DataFrame()


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("正样本数据准备（断点续传版）")
    log.info("=" * 80)
    log.info("💡 提示：网络中断后重新运行相同命令即可继续")
    log.info("")

    # 配置
    START_DATE = settings.get("data.sample_preparation.start_date", "20000101")
    END_DATE = settings.get("data.sample_preparation.end_date", None)

    log.info(f"日期范围: {START_DATE} - {END_DATE or '今天'}")

    # 初始化
    log.info("\n[步骤1] 初始化...")
    dm = DataManager(source="tushare")

    positive_criteria = settings.get("data.sample_preparation.positive_criteria", {})
    screener_config = {
        "consecutive_weeks": positive_criteria.get("consecutive_weeks", 3),
        "total_return_threshold": positive_criteria.get("total_return_threshold", 50),
        "max_return_threshold": positive_criteria.get("max_return_threshold", 70),
        "min_listing_days": positive_criteria.get("min_listing_days", 180),
        "pre_t1_return_max": positive_criteria.get("pre_t1_return_max", 25),
        "pre_t1_volatility_max": positive_criteria.get("pre_t1_volatility_max", 4),
        "enable_anti_chasing": positive_criteria.get("enable_anti_chasing", True),
    }

    screener = PositiveSampleScreener(dm, config=screener_config)
    log.success("✓ 初始化完成")

    # 步骤2: 正样本扫描
    log.info("\n[步骤2] 正样本扫描...")
    df_samples = screen_positive_samples_with_checkpoint(dm, screener, START_DATE, END_DATE)

    if df_samples.empty:
        log.error("未找到符合条件的正样本！")
        return

    log.info("\n正样本统计:")
    log.info(f"  总样本数: {len(df_samples)}")
    log.info(f"  股票数量: {df_samples['ts_code'].nunique()}")
    log.info(f"  平均总涨幅: {df_samples['total_return'].mean():.2f}%")

    # 步骤3: 特征提取
    log.info("\n[步骤3] 特征提取...")
    df_features = extract_features_with_checkpoint(screener, df_samples, lookback_days=34)

    if df_features.empty:
        log.error("特征提取失败！")
        return

    log.info("\n特征数据统计:")
    log.info(f"  总记录数: {len(df_features)}")
    log.info(f"  样本数: {df_features['sample_id'].nunique()}")

    # 完成
    log.info("\n" + "=" * 80)
    log.success("✅ 正样本数据准备完成！")
    log.info("=" * 80)
    log.info("\n生成的文件:")
    log.info(f"  1. 正样本列表: {SAMPLES_FILE}")
    log.info(f"  2. 特征数据: {FEATURES_FILE}")
    log.info("\n下一步: 运行 python scripts/prepare_negative_samples_checkpoint.py")


if __name__ == "__main__":
    main()
