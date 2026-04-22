#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.4.0特征扩充脚本

功能：
1. 读取v3版本的特征数据（139列）
2. 计算并添加5个反追龙头新特征
3. 对正样本应用T1前约束筛选
4. 输出为v4版本特征文件

新增特征：
- days_near_ma10: MA10附近天数
- close_vs_ma10_std: 价格偏离MA10的标准差
- price_range_pct: 34天振幅百分比
- volume_shrink_ratio: 缩量比
- ma10_cross_count: 穿越MA10次数
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log

# 配置
PRE_T1_RETURN_MAX = 20  # T1前34天涨幅上限(%)
PRE_T1_VOLATILITY_MAX = 3  # T1前日均波动率上限(%)


def calculate_anti_chasing_features(df):
    """
    计算反追龙头相关的新特征

    由于v3数据是已经聚合好的特征（每个样本一行），部分特征需要基于已有列计算
    """
    df = df.copy()

    # 1. price_range_pct: 34天振幅百分比
    # 使用已有的 high_34d 和 low_34d
    if "high_34d" in df.columns and "low_34d" in df.columns:
        df["price_range_pct"] = np.where(df["low_34d"] > 0, (df["high_34d"] - df["low_34d"]) / df["low_34d"] * 100, 0)
    else:
        df["price_range_pct"] = 0

    # 2. 从已有特征推导盘整相关指标
    # close_vs_ma10_std: 使用 bias_short (短期乖离率) 的绝对值作为近似
    if "bias_short" in df.columns:
        df["close_vs_ma10_std"] = df["bias_short"].abs()
    else:
        df["close_vs_ma10_std"] = 0

    # 3. days_near_ma10: 使用 price_position_34d 推导
    # price_position_34d 范围是 0-100（百分比），接近50说明价格在区间中间
    # 这里用 1 - abs(price_position_34d/100 - 0.5) * 2 来近似
    if "price_position_34d" in df.columns:
        # 先归一化到0-1，然后转换为0-34的天数估计
        position_norm = df["price_position_34d"] / 100.0  # 归一化到0-1
        df["days_near_ma10"] = (1 - (position_norm - 0.5).abs() * 2) * 34
        df["days_near_ma10"] = df["days_near_ma10"].clip(0, 34)
    else:
        df["days_near_ma10"] = 17  # 默认中间值

    # 4. volume_shrink_ratio: 使用已有的量能特征推导
    # vol_ma5_ratio / vol_ma20_ratio 可以反映近期量能相对远期的变化
    if "vol_ma5_ratio" in df.columns and "vol_ma20_ratio" in df.columns:
        df["volume_shrink_ratio"] = np.where(df["vol_ma20_ratio"] > 0, df["vol_ma5_ratio"] / df["vol_ma20_ratio"], 1)
    else:
        df["volume_shrink_ratio"] = 1

    # 5. ma10_cross_count: 使用 volatility 和 bias 推导
    # 波动率高且乖离率在0附近反复，说明频繁穿越均线
    if "volatility_34d" in df.columns and "bias_short" in df.columns:
        # 波动率高 + 乖离率小 = 频繁穿越
        volatility_norm = df["volatility_34d"] / (df["volatility_34d"].mean() + 1e-10)
        bias_small = (df["bias_short"].abs() < 3).astype(float)
        df["ma10_cross_count"] = (volatility_norm * bias_small * 10).clip(0, 34)
    else:
        df["ma10_cross_count"] = 5  # 默认值

    return df


def filter_positive_samples(df):
    """
    应用T1前约束筛选正样本

    条件：
    - return_34d <= PRE_T1_RETURN_MAX (20%)
    - volatility_34d <= PRE_T1_VOLATILITY_MAX (3%)
    """
    original_count = len(df)

    # 筛选条件
    mask = pd.Series(True, index=df.index)

    if "return_34d" in df.columns:
        mask &= df["return_34d"] <= PRE_T1_RETURN_MAX

    if "volatility_34d" in df.columns:
        mask &= df["volatility_34d"] <= PRE_T1_VOLATILITY_MAX

    df_filtered = df[mask].copy()
    filtered_count = len(df_filtered)

    log.info(f"  正样本筛选: {original_count} -> {filtered_count} ({filtered_count/original_count*100:.1f}%)")
    log.info(f"  筛选条件: return_34d <= {PRE_T1_RETURN_MAX}%, volatility_34d <= {PRE_T1_VOLATILITY_MAX}%")

    return df_filtered


def process_feature_file(input_path, output_path, is_positive=False):
    """
    处理单个特征文件

    Args:
        input_path: 输入文件路径（v3版本）
        output_path: 输出文件路径（v4版本）
        is_positive: 是否是正样本（需要应用筛选）
    """
    log.info(f"\n处理: {input_path.name}")

    # 读取数据
    df = pd.read_csv(input_path)
    log.info(f"  读取: {len(df)} 行, {len(df.columns)} 列")

    # 计算新特征
    df = calculate_anti_chasing_features(df)

    # 如果是正样本，应用T1前约束筛选
    if is_positive:
        df = filter_positive_samples(df)

    # 保存
    df.to_csv(output_path, index=False)
    log.success(f"  保存: {output_path.name} ({len(df)} 行, {len(df.columns)} 列)")

    return df


def main():
    log.info("=" * 80)
    log.info("v2.4.0 特征扩充脚本")
    log.info("=" * 80)
    log.info("")
    log.info("基于v3版本数据（139列），增加5个反追龙头特征")
    log.info("新增特征: price_range_pct, close_vs_ma10_std, days_near_ma10,")
    log.info("         volume_shrink_ratio, ma10_cross_count")
    log.info("")

    # 文件路径
    pos_v3 = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v3.csv"
    neg_v3 = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v3.csv"
    hard_neg_v3 = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v3.csv"

    pos_v4 = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v4.csv"
    neg_v4 = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v4.csv"
    hard_neg_v4 = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v4.csv"

    # 检查输入文件
    for f in [pos_v3, neg_v3, hard_neg_v3]:
        if not f.exists():
            log.error(f"输入文件不存在: {f}")
            return

    # 处理正样本（应用T1前约束筛选）
    log.info("=" * 80)
    log.info("Phase 1: 处理正样本（含筛选）")
    log.info("=" * 80)
    df_pos = process_feature_file(pos_v3, pos_v4, is_positive=True)

    # 处理普通负样本
    log.info("\n" + "=" * 80)
    log.info("Phase 2: 处理普通负样本")
    log.info("=" * 80)
    df_neg = process_feature_file(neg_v3, neg_v4, is_positive=False)

    # 处理硬负样本
    log.info("\n" + "=" * 80)
    log.info("Phase 3: 处理硬负样本")
    log.info("=" * 80)
    df_hard_neg = process_feature_file(hard_neg_v3, hard_neg_v4, is_positive=False)

    # 汇总
    log.info("\n" + "=" * 80)
    log.info("汇总")
    log.info("=" * 80)
    log.info(f"  正样本(筛选后): {len(df_pos)} 条")
    log.info(f"  普通负样本: {len(df_neg)} 条")
    log.info(f"  硬负样本: {len(df_hard_neg)} 条")
    log.info(f"  总计: {len(df_pos) + len(df_neg) + len(df_hard_neg)} 条")
    log.info(f"  特征数: {len(df_pos.columns)} 列")

    # 显示新增特征统计
    new_features = ["price_range_pct", "close_vs_ma10_std", "days_near_ma10", "volume_shrink_ratio", "ma10_cross_count"]

    log.info("\n新增特征统计（正样本）:")
    for feat in new_features:
        if feat in df_pos.columns:
            log.info(f"  {feat}: mean={df_pos[feat].mean():.2f}, std={df_pos[feat].std():.2f}")

    log.info("\n" + "=" * 80)
    log.success("✅ 特征扩充完成！")
    log.info("=" * 80)
    log.info("")
    log.info("输出文件:")
    log.info(f"  {pos_v4}")
    log.info(f"  {neg_v4}")
    log.info(f"  {hard_neg_v4}")


if __name__ == "__main__":
    main()
