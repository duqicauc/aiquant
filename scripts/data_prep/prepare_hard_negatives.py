"""
硬负样本数据准备脚本

筛选"接近但未达标"的股票作为硬负样本：
- 34日涨幅在20-45%之间（接近正样本的50%阈值）
- 这些股票特征与正样本相似，用于提高模型区分能力

使用方法：
    python scripts/prepare_hard_negatives.py
"""

import sys
import os
import warnings
import pandas as pd
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.data_manager import DataManager
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener
from src.utils.logger import log


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("硬负样本数据准备")
    log.info("=" * 80)

    # 配置参数
    MIN_RETURN = 20.0  # 最小34日涨幅
    MAX_RETURN = 45.0  # 最大34日涨幅（低于正样本的50%）
    SAMPLES_PER_DATE = 3  # 每个T1日期采样的硬负样本数量
    RANDOM_SEED = 42

    POSITIVE_SAMPLES_FILE = "data/training/samples/positive_samples.csv"
    OUTPUT_HARD_SAMPLES = "data/training/samples/hard_negative_samples.csv"
    OUTPUT_HARD_FEATURES = "data/training/features/hard_negative_feature_data_34d.csv"
    OUTPUT_STATS = "data/training/samples/hard_negative_statistics.json"

    log.info("\n当前设置：")
    log.info(f"  方法: 硬负样本筛选（34日涨幅{MIN_RETURN}%-{MAX_RETURN}%）")
    log.info(f"  正样本文件: {POSITIVE_SAMPLES_FILE}")
    log.info(f"  每T1日期采样: {SAMPLES_PER_DATE} 只")
    log.info(f"  随机种子: {RANDOM_SEED}")
    log.info("")

    # 1. 加载正样本数据
    log.info("=" * 80)
    log.info("第一步：加载正样本数据")
    log.info("=" * 80)

    try:
        df_positive_samples = pd.read_csv(POSITIVE_SAMPLES_FILE)
        log.success(f"✓ 正样本加载成功: {len(df_positive_samples)} 个")
    except Exception as e:
        log.error(f"✗ 加载正样本数据失败: {e}")
        return

    # 2. 初始化数据管理器和筛选器
    log.info("\n" + "=" * 80)
    log.info("第二步：初始化硬负样本筛选器")
    log.info("=" * 80)

    dm = DataManager()
    screener = HardNegativeSampleScreener(dm)

    log.success("✓ 筛选器初始化完成")

    # 3. 筛选硬负样本
    log.info("\n" + "=" * 80)
    log.info("第三步：筛选硬负样本")
    log.info("=" * 80)

    df_hard_samples = screener.screen_hard_negatives(
        positive_samples_df=df_positive_samples,
        min_return=MIN_RETURN,
        max_return=MAX_RETURN,
        samples_per_date=SAMPLES_PER_DATE,
        random_seed=RANDOM_SEED,
    )

    if df_hard_samples.empty:
        log.error("✗ 未找到硬负样本")
        return

    # 4. 提取硬负样本特征
    log.info("\n" + "=" * 80)
    log.info("第四步：提取硬负样本特征数据")
    log.info("=" * 80)

    df_hard_features = screener.extract_features(df_hard_samples)

    if df_hard_features.empty:
        log.error("✗ 特征提取失败")
        return

    # 4.1 数据质量处理
    log.info("\n[步骤4.1] 数据质量处理...")

    # 定义数值列
    numeric_cols = [
        "close",
        "pct_chg",
        "total_mv",
        "circ_mv",
        "ma5",
        "ma10",
        "volume_ratio",
        "macd_dif",
        "macd_dea",
        "macd",
        "rsi_6",
        "rsi_12",
        "rsi_24",
    ]
    numeric_cols = [col for col in numeric_cols if col in df_hard_features.columns]

    # 缺失值填充
    df_hard_features[numeric_cols] = df_hard_features.groupby("sample_id")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # 过滤数据不足的样本
    min_days = 30
    days_per_sample = df_hard_features.groupby("sample_id").size()
    valid_samples = days_per_sample[days_per_sample >= min_days].index
    df_hard_features = df_hard_features[df_hard_features["sample_id"].isin(valid_samples)]

    # 同步过滤样本列表
    valid_sample_ids = df_hard_features["sample_id"].unique()
    df_hard_samples = df_hard_samples[df_hard_samples.index.isin(valid_sample_ids)]

    log.success("✓ 数据质量处理完成")

    # 5. 保存结果
    log.info("\n" + "=" * 80)
    log.info("第五步：保存结果")
    log.info("=" * 80)

    # 保存硬负样本列表
    df_hard_samples.to_csv(OUTPUT_HARD_SAMPLES, index=False)
    log.success(f"✓ 硬负样本列表已保存: {OUTPUT_HARD_SAMPLES}")

    # 保存硬负样本特征数据
    df_hard_features.to_csv(OUTPUT_HARD_FEATURES, index=False)
    log.success(f"✓ 硬负样本特征数据已保存: {OUTPUT_HARD_FEATURES}")

    # 保存统计信息
    stats = {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "硬负样本筛选（34日涨幅接近阈值）",
        "total_hard_samples": len(df_hard_samples),
        "total_positive_samples": len(df_positive_samples),
        "hard_feature_records": len(df_hard_features),
        "feature_samples": int(df_hard_features["sample_id"].nunique()),
        "return_range": {"min": MIN_RETURN, "max": MAX_RETURN},
        "samples_per_date": SAMPLES_PER_DATE,
        "random_seed": RANDOM_SEED,
        "return_statistics": {
            "mean": float(df_hard_samples["return_34d"].mean()) if "return_34d" in df_hard_samples.columns else None,
            "median": (
                float(df_hard_samples["return_34d"].median()) if "return_34d" in df_hard_samples.columns else None
            ),
            "min": float(df_hard_samples["return_34d"].min()) if "return_34d" in df_hard_samples.columns else None,
            "max": float(df_hard_samples["return_34d"].max()) if "return_34d" in df_hard_samples.columns else None,
        },
        "files": {
            "hard_samples": OUTPUT_HARD_SAMPLES,
            "hard_features": OUTPUT_HARD_FEATURES,
            "positive_samples": POSITIVE_SAMPLES_FILE,
        },
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 统计报告已保存: {OUTPUT_STATS}")

    # 6. 最终总结
    log.info("\n" + "=" * 80)
    log.success("✅ 硬负样本数据准备完成！")
    log.info("=" * 80)
    log.info("")
    log.info(f"  1. 硬负样本列表: {OUTPUT_HARD_SAMPLES}")
    log.info(f"  2. 硬负样本特征: {OUTPUT_HARD_FEATURES}")
    log.info(f"  3. 统计报告: {OUTPUT_STATS}")
    log.info("")
    log.info("📊 数据统计：")
    log.info(f"  正样本数: {len(df_positive_samples)}")
    log.info(f"  硬负样本数: {len(df_hard_samples)}")
    log.info(f"  硬负样本特征: {len(df_hard_features)} 条")
    if "return_34d" in df_hard_samples.columns:
        log.info("\n📈 34日涨幅分布：")
        log.info(f"  均值: {df_hard_samples['return_34d'].mean():.2f}%")
        log.info(f"  中位数: {df_hard_samples['return_34d'].median():.2f}%")
    log.info("")
    log.info("💡 硬负样本特点：")
    log.info("  - 34日涨幅在20-45%之间（接近正样本的50%阈值）")
    log.info("  - 特征与正样本相似，但未达到正样本标准")
    log.info("  - 用于提高模型区分能力，减少过拟合")
    log.info("")
    log.info("下一步：")
    log.info("  运行 python scripts/train_optimized_model.py 训练优化版模型")
    log.info("")


if __name__ == "__main__":
    main()
