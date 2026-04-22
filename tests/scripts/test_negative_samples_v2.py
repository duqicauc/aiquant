"""
负样本筛选功能测试脚本 V2 - 同周期其他股票法

快速测试V2方案是否正常工作
"""

import sys
import os
import warnings
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.data_manager import DataManager
from src.strategy.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log


def test_v2_screening():
    """测试V2筛选功能"""
    log.info("=" * 80)
    log.info("测试：V2负样本筛选（同周期其他股票法）")
    log.info("=" * 80)

    # 加载正样本
    try:
        df_positive = pd.read_csv("data/processed/positive_samples.csv")
        log.success(f"✓ 加载正样本: {len(df_positive)} 个")
    except FileNotFoundError:
        log.error("✗ 未找到正样本文件，请先运行 prepare_positive_samples.py")
        return False

    # 只用前10个正样本测试
    df_positive_test = df_positive.head(10)
    log.info(f"\n测试用正样本数: {len(df_positive_test)} 个")

    # 初始化筛选器
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)

    # 筛选负样本（每个正样本对应1个负样本）
    log.info("\n开始筛选负样本...")

    df_negative = screener.screen_negative_samples(
        positive_samples_df=df_positive_test, samples_per_positive=1, random_seed=42
    )

    if df_negative.empty:
        log.warning("⚠️  未找到负样本")
        return False

    log.success(f"✓ 找到 {len(df_negative)} 个负样本")

    # 显示结果
    log.info("\n负样本预览：")
    print(df_negative)

    return df_negative


def test_v2_feature_extraction(df_negative):
    """测试V2特征提取"""
    log.info("\n" + "=" * 80)
    log.info("测试：V2负样本特征提取")
    log.info("=" * 80)

    # 初始化筛选器
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)

    # 提取特征
    df_features = screener.extract_features(df_negative)

    if df_features.empty:
        log.warning("⚠️  特征提取失败")
        return False

    log.success(f"✓ 提取特征: {len(df_features)} 条")

    # 显示结果
    log.info("\n特征数据预览（前5条）：")
    available_cols = [
        col
        for col in [
            "sample_id",
            "trade_date",
            "name",
            "ts_code",
            "close",
            "pct_chg",
            "volume_ratio",
            "ma5",
            "ma10",
            "label",
        ]
        if col in df_features.columns
    ]

    print(df_features[available_cols].head())

    # 验证标签
    unique_labels = df_features["label"].unique()
    log.info(f"\n标签检查: {unique_labels}")

    if len(unique_labels) == 1 and unique_labels[0] == 0:
        log.success("✓ 所有负样本标签正确（label=0）")
    else:
        log.warning(f"⚠️  标签异常: {unique_labels}")

    return True


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("负样本筛选功能测试 V2 - 同周期其他股票法")
    log.info("=" * 80)
    log.info("")
    log.info("说明：本测试将执行以下步骤：")
    log.info("  1. 筛选10个负样本（快速测试）")
    log.info("  2. 提取负样本特征数据")
    log.info("")
    log.info("=" * 80)

    # 测试1：负样本筛选
    df_negative = test_v2_screening()
    if df_negative is False or (isinstance(df_negative, pd.DataFrame) and df_negative.empty):
        log.error("\n✗ 测试1失败，请检查筛选逻辑")
        return

    # 测试2：特征提取
    success = test_v2_feature_extraction(df_negative)
    if not success:
        log.error("\n✗ 测试2失败，请检查特征提取逻辑")
        return

    # 测试完成
    log.info("\n" + "=" * 80)
    log.success("✅ 所有测试通过！")
    log.info("=" * 80)
    log.info("")
    log.info("✨ V2方案优势：")
    log.info("  - 筛选速度快（不需要计算特征统计）")
    log.info("  - 实现简单")
    log.info("  - 数据量充足")
    log.info("  - 更接近实际场景")
    log.info("")
    log.info("下一步：")
    log.info("  1. 运行完整的V2负样本筛选：")
    log.info("     python scripts/prepare_negative_samples_v2.py")
    log.info("")
    log.info("  2. 对比V1和V2效果：")
    log.info("     - V1: 基于特征统计（已实现）")
    log.info("     - V2: 同周期其他股票（新方案）")
    log.info("")
    log.info("  3. 训练两个模型对比效果")
    log.info("")
    log.info("📚 详细对比: docs/NEGATIVE_SAMPLE_COMPARISON.md")
    log.info("")


if __name__ == "__main__":
    main()
