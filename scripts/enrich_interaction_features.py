#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为训练数据补充特征交互项

新增特征：
1. breakout_with_volume: 突破 x 量能交互
2. momentum_market_interaction: 动量 x 市场环境交互
3. rsi_kdj_divergence: RSI 与 KDJ 背离
4. trend_consistency: 短期长期趋势一致性
5. volume_price_divergence: 量价背离

处理文件：
1. data/training/processed/feature_data_34d_v5.csv（正样本）
2. data/training/features/negative_feature_data_v2_34d_v5.csv（负样本）
3. data/training/features/hard_negative_feature_data_34d_v5.csv（硬负样本）
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


def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征交互项

    Args:
        df: 样本数据

    Returns:
        添加了交互特征的DataFrame
    """
    df = df.copy()

    # ========== 1. 突破 x 量能交互 ==========
    # breakout_with_volume = breakout_strength_20d * breakout_volume_ratio
    if "breakout_strength_20d" in df.columns and "breakout_volume_ratio" in df.columns:
        df["breakout_with_volume"] = df["breakout_strength_20d"] * df["breakout_volume_ratio"]
        log.info("  ✓ 计算 breakout_with_volume")
    else:
        df["breakout_with_volume"] = np.nan
        log.warning("  ⚠ 缺少 breakout_strength_20d 或 breakout_volume_ratio")

    # ========== 2. 动量 x 市场环境交互 ==========
    # momentum_market_interaction = return_20d * market_trend
    if "return_20d" in df.columns and "market_trend" in df.columns:
        df["momentum_market_interaction"] = df["return_20d"] * df["market_trend"]
        log.info("  ✓ 计算 momentum_market_interaction")
    elif "pct_chg" in df.columns and "market_trend" in df.columns:
        # 如果没有 return_20d，使用 pct_chg 的滚动和
        if "sample_id" in df.columns:
            df["_return_20d"] = df.groupby("sample_id")["pct_chg"].transform(
                lambda x: x.rolling(20, min_periods=1).sum()
            )
        else:
            df["_return_20d"] = df["pct_chg"].rolling(20, min_periods=1).sum()
        df["momentum_market_interaction"] = df["_return_20d"] * df["market_trend"]
        df = df.drop(columns=["_return_20d"])
        log.info("  ✓ 计算 momentum_market_interaction（使用 pct_chg 滚动和）")
    else:
        df["momentum_market_interaction"] = np.nan
        log.warning("  ⚠ 缺少动量或市场趋势特征")

    # ========== 3. RSI 与 KDJ 背离 ==========
    # rsi_kdj_divergence = rsi_6 - kdj_j
    if "rsi_6" in df.columns and "kdj_j" in df.columns:
        df["rsi_kdj_divergence"] = df["rsi_6"] - df["kdj_j"]
        log.info("  ✓ 计算 rsi_kdj_divergence")
    else:
        df["rsi_kdj_divergence"] = np.nan
        log.warning("  ⚠ 缺少 rsi_6 或 kdj_j")

    # ========== 4. 短期长期趋势一致性 ==========
    # trend_consistency = sign(trend_slope_8d) == sign(trend_slope_34d)
    if "trend_slope_8d" in df.columns and "trend_slope_34d" in df.columns:
        df["trend_consistency"] = (np.sign(df["trend_slope_8d"]) == np.sign(df["trend_slope_34d"])).astype(int)
        log.info("  ✓ 计算 trend_consistency")
    else:
        df["trend_consistency"] = np.nan
        log.warning("  ⚠ 缺少 trend_slope_8d 或 trend_slope_34d")

    # ========== 5. 量价背离 ==========
    # volume_price_divergence = sign(pct_chg) != sign(volume_change)
    if "pct_chg" in df.columns:
        if "volume_change" in df.columns:
            df["volume_price_divergence"] = (np.sign(df["pct_chg"]) != np.sign(df["volume_change"])).astype(int)
            log.info("  ✓ 计算 volume_price_divergence")
        elif "vol" in df.columns:
            # 计算成交量变化
            if "sample_id" in df.columns:
                df["_vol_change"] = df.groupby("sample_id")["vol"].transform(lambda x: x.pct_change())
            else:
                df["_vol_change"] = df["vol"].pct_change()
            df["volume_price_divergence"] = (np.sign(df["pct_chg"]) != np.sign(df["_vol_change"])).astype(int)
            df = df.drop(columns=["_vol_change"])
            log.info("  ✓ 计算 volume_price_divergence（使用 vol 计算）")
        else:
            df["volume_price_divergence"] = np.nan
            log.warning("  ⚠ 缺少成交量变化特征")
    else:
        df["volume_price_divergence"] = np.nan
        log.warning("  ⚠ 缺少 pct_chg")

    # ========== 6. 额外的交互特征 ==========

    # 6.1 突破强度 x RSI（超买超卖中的突破）
    if "breakout_strength_20d" in df.columns and "rsi_6" in df.columns:
        df["breakout_rsi_interaction"] = df["breakout_strength_20d"] * (df["rsi_6"] - 50) / 50
        log.info("  ✓ 计算 breakout_rsi_interaction")

    # 6.2 市场波动率 x 个股波动率（相对波动）
    if "market_volatility_34d" in df.columns and "volatility_34d" in df.columns:
        df["relative_volatility"] = df["volatility_34d"] / (df["market_volatility_34d"] + 1e-8)
        log.info("  ✓ 计算 relative_volatility")

    # 6.3 突破共振 x 量能（多信号确认）
    if "breakout_resonance" in df.columns and "breakout_volume_strength" in df.columns:
        df["resonance_volume_confirm"] = df["breakout_resonance"] * df["breakout_volume_strength"]
        log.info("  ✓ 计算 resonance_volume_confirm")

    return df


def process_file(input_file: Path) -> bool:
    """
    处理单个文件

    Args:
        input_file: 输入文件路径

    Returns:
        是否成功
    """
    log.info("=" * 80)
    log.info(f"处理文件: {input_file.name}")
    log.info("=" * 80)

    if not input_file.exists():
        log.error(f"文件不存在: {input_file}")
        return False

    # 读取数据
    log.info(f"读取数据: {input_file}")
    try:
        df = pd.read_csv(input_file)
        log.info(f"  原始数据: {len(df)} 条，{len(df.columns)} 列")
    except Exception as e:
        log.error(f"读取文件失败: {e}")
        return False

    # 检查是否已有交互特征
    interaction_cols = [
        "breakout_with_volume",
        "momentum_market_interaction",
        "rsi_kdj_divergence",
        "trend_consistency",
        "volume_price_divergence",
        "breakout_rsi_interaction",
        "relative_volatility",
        "resonance_volume_confirm",
    ]

    existing_cols = [col for col in interaction_cols if col in df.columns]
    if existing_cols:
        log.info(f"检测到已有交互特征: {existing_cols}")
        df = df.drop(columns=existing_cols)
        log.info("已删除旧的交互特征列")

    # 计算交互特征
    log.info("计算特征交互项...")
    df = calculate_interaction_features(df)

    if df is None or len(df) == 0:
        log.error("计算交互特征失败")
        return False

    # 检查新增特征
    log.info("\n新增交互特征统计:")
    for col in interaction_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100 if len(df) > 0 else 0

            if df[col].dtype in ["float64", "int64"]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                log.info(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%), mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                log.info(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")

    # 保存结果
    log.info(f"\n保存结果: {input_file}")
    try:
        df.to_csv(input_file, index=False, encoding="utf-8-sig")
        log.success(f"✓ 文件处理完成: {input_file}")
        log.info(f"  最终数据: {len(df)} 条，{len(df.columns)} 列")
        return True
    except Exception as e:
        log.error(f"保存文件失败: {e}")
        return False


def main():
    log.info("=" * 80)
    log.info("为训练数据补充特征交互项")
    log.info("=" * 80)

    # 定义文件路径
    files_to_process = [
        PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv",
        PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv",
        PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv",
    ]

    # 处理每个文件
    results = []
    for file_path in files_to_process:
        success = process_file(file_path)
        results.append({"name": file_path.name, "success": success})

    # 总结
    log.info("\n" + "=" * 80)
    log.info("处理总结")
    log.info("=" * 80)

    for result in results:
        status = "✓ 成功" if result["success"] else "✗ 失败"
        log.info(f"{result['name']}: {status}")

    success_count = sum(1 for r in results if r["success"])
    log.info(f"\n总计: {success_count}/{len(results)} 个文件处理成功")

    if success_count == len(results):
        log.success("\n✓ 所有文件处理完成！")
    else:
        log.warning("\n⚠️  部分文件处理失败，请检查日志")


if __name__ == "__main__":
    main()
