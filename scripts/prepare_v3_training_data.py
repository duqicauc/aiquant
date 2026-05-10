#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3.0 训练数据准备脚本

使用 UnifiedFeatureExtractor 统一提取正/负/硬负样本特征，
确保三类样本使用完全相同的计算逻辑、相同的 Tushare 因子来源、相同的特征集合。

Usage:
    python scripts/prepare_v3_training_data.py

Output:
    data/training/v3/positive_features.csv
    data/training/v3/negative_features.csv
    data/training/v3/hard_negative_features.csv
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.features.unified_feature_extractor import FeatureValidator, UnifiedFeatureExtractor
from src.utils.logger import log

# ============================================================================
# 配置
# ============================================================================
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_SAMPLES_FILE = SAMPLES_DIR / "positive_samples.csv"
NEGATIVE_SAMPLES_FILE = SAMPLES_DIR / "negative_samples_v2.csv"
HARD_NEGATIVE_SAMPLES_FILE = SAMPLES_DIR / "hard_negatives_v291.csv"

LOOKBACK_DAYS = 34


def load_positive_samples() -> pd.DataFrame:
    """加载正样本列表"""
    log.info("=" * 80)
    log.info("加载正样本...")
    df = pd.read_csv(POSITIVE_SAMPLES_FILE)
    log.info(f"正样本: {len(df)} 条")

    # 添加 sample_id（如果没有）
    if "sample_id" not in df.columns:
        df["sample_id"] = range(len(df))

    return df


def load_negative_samples() -> pd.DataFrame:
    """加载负样本列表"""
    log.info("=" * 80)
    log.info("加载负样本...")
    df = pd.read_csv(NEGATIVE_SAMPLES_FILE)
    log.info(f"负样本: {len(df)} 条")

    # 负样本CSV中已有 sample_id（从1开始）
    if "sample_id" not in df.columns:
        df["sample_id"] = range(len(df))

    return df


def load_hard_negative_samples() -> pd.DataFrame:
    """加载硬负样本列表"""
    log.info("=" * 80)
    log.info("加载硬负样本...")
    if not HARD_NEGATIVE_SAMPLES_FILE.exists():
        log.error(f"硬负样本文件不存在: {HARD_NEGATIVE_SAMPLES_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(HARD_NEGATIVE_SAMPLES_FILE)
    log.info(f"硬负样本: {len(df)} 条")

    # 硬负样本没有 sample_id，需要生成
    if "sample_id" not in df.columns:
        df["sample_id"] = range(len(df))

    # 硬负样本没有 name 列，尝试补充
    if "name" not in df.columns:
        df["name"] = ""

    return df


def process_sample_type(extractor: UnifiedFeatureExtractor, samples_df: pd.DataFrame, label: int, output_name: str):
    """处理一种样本类型并保存"""
    if samples_df.empty:
        log.warning(f"{output_name}: 样本为空，跳过")
        return None

    sample_type = "positive" if label == 1 else ("hard_negative" if "hard" in output_name else "negative")
    log.info("=" * 80)
    log.info(f"开始处理: {output_name} (label={label})")
    log.info("=" * 80)

    # 提取特征
    df_features = extractor.extract_for_samples(samples_df, lookback_days=LOOKBACK_DAYS, label=label)

    if df_features.empty:
        log.error(f"{output_name}: 特征提取失败")
        return None

    # 校验
    try:
        FeatureValidator.validate(df_features, sample_type=sample_type)
    except ValueError as e:
        log.error(f"{output_name}: 校验失败 - {e}")
        # 即使校验失败也保存，供排查
        output_path = OUTPUT_DIR / f"{output_name}_INVALID.csv"
        df_features.to_csv(output_path, index=False)
        log.info(f"已保存失败数据供排查: {output_path}")
        return None

    # 保存
    output_path = OUTPUT_DIR / f"{output_name}.csv"
    df_features.to_csv(output_path, index=False)
    log.success(f"已保存: {output_path}")

    return output_path


def main():
    log.info("=" * 80)
    log.info("v3.0 训练数据准备")
    log.info("=" * 80)
    log.info(f"输出目录: {OUTPUT_DIR}")
    log.info(f"回看天数: {LOOKBACK_DAYS}")
    log.info("")

    # 初始化提取器
    extractor = UnifiedFeatureExtractor(use_cache=True)

    results = {}

    # 1. 正样本
    df_pos = load_positive_samples()
    results["positive"] = process_sample_type(extractor, df_pos, label=1, output_name="positive_features")

    # 2. 负样本
    df_neg = load_negative_samples()
    results["negative"] = process_sample_type(extractor, df_neg, label=0, output_name="negative_features")

    # 3. 硬负样本
    df_hard = load_hard_negative_samples()
    results["hard_negative"] = process_sample_type(extractor, df_hard, label=0, output_name="hard_negative_features")

    # 汇总
    log.info("\n" + "=" * 80)
    log.info("处理完成汇总")
    log.info("=" * 80)
    for k, v in results.items():
        status = "✅ " + str(v) if v else "❌ 失败"
        log.info(f"  {k}: {status}")
    log.info("=" * 80)

    # 如果全部成功，提示下一步
    if all(v is not None for v in results.values()):
        log.success("\n✅ 全部样本特征提取成功！")
        log.info("下一步: 运行训练脚本")
    else:
        log.warning("\n⚠️ 部分样本处理失败，请检查日志")


if __name__ == "__main__":
    main()
