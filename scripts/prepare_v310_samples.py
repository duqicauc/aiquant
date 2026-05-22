#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3.1.0 双模型样本生成脚本（时间范围扩展版）

同时生成 BreakoutScorer 和 BounceScorer 的训练样本：
1. Breakout正样本: 平台整理(10-30d, BOLL带宽<8%, 振幅<15%) + 放量突破(>高点+2%, 成交量>1.5x) + 确认上涨(3d≥5%)
2. Bounce正样本: 深度回调(20-60d, 回落≥20%, RSI<35) + 止跌迹象(下影线≥1.5% / 放量) + 确认反弹(3d≥5%)
3. 负样本: 各模型各自的非目标形态股票
4. 硬负样本: Breakout(假突破/冲高回落) + Bounce(下跌中继/弱势反弹/无量反弹)

时间划分（严格时间外）：
  训练集: 2015-2020 (学规律)
  验证集: 2021-2022 (调超参)
  测试集: 2023-2024 (最终评估，模型不可见)

Usage:
    python scripts/prepare_v310_samples.py [--model breakout|bounce|both]

Output:
    data/training/samples/v310/breakout/train/{positive,negative,hard_negative}.csv
    data/training/samples/v310/breakout/val/{positive,negative,hard_negative}.csv
    data/training/samples/v310/breakout/test/{positive,negative,hard_negative}.csv
    data/training/samples/v310/bounce/train/...
    data/training/samples/v310/bounce/val/...
    data/training/samples/v310/bounce/test/...
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening import BreakoutSampleScreener, BounceSampleScreener

# ============================================================================
# 配置
# ============================================================================
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples" / "v310"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# 时间范围（扩展至10年）
DEFAULT_START_DATE = "20150101"
DEFAULT_END_DATE = "20261231"  # 扩展至2026年，纳入最新数据

# 严格时间外划分（避免未来函数泄漏）
# 方案A: 训练到2022（纳入注册制改革后数据），验证2023，测试2024-2026
TRAIN_END = "20221231"   # 训练截止（8年: 2015-2022）
VAL_END = "20231231"     # 验证截止（1年: 2023）
# TEST: 2024-01-01 ~ 2026-12-31（2.5年，含最新数据）

# 各模型目标样本数（全时间范围生成后按划分拆分）
BREAKOUT_CONFIG = {
    "positive_max": None,      # None = 不限制，后续按季度降采样
    "negative_target": 12000,
    "hard_negative_target": 1500,
}

BOUNCE_CONFIG = {
    "positive_max": None,
    "negative_target": 20000,
    "hard_negative_target": 2500,
}


# ============================================================================
# 时间划分
# ============================================================================
def split_by_time(df: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END):
    """
    按时间严格划分训练/验证/测试集

    Args:
        df: 含 t1_date 列的DataFrame
        train_end: 训练集截止日期 (YYYYMMDD)
        val_end: 验证集截止日期 (YYYYMMDD)

    Returns:
        (train_df, val_df, test_df)
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    df = df.copy()
    df["t1_date"] = pd.to_datetime(df["t1_date"].astype(str), format="%Y%m%d", errors="coerce")

    train_df = df[df["t1_date"] <= pd.to_datetime(train_end, format="%Y%m%d")].copy()
    val_df = df[
        (df["t1_date"] > pd.to_datetime(train_end, format="%Y%m%d")) &
        (df["t1_date"] <= pd.to_datetime(val_end, format="%Y%m%d"))
    ].copy()
    test_df = df[df["t1_date"] > pd.to_datetime(val_end, format="%Y%m%d")].copy()

    # 转回字符串格式
    for d in [train_df, val_df, test_df]:
        if not d.empty:
            d["t1_date"] = d["t1_date"].dt.strftime("%Y%m%d")

    return train_df, val_df, test_df


def save_split_samples(df_pos, df_neg, df_hard, model_name: str):
    """生成并保存按时间划分的样本"""
    splits = {
        "positive": df_pos,
        "negative": df_neg,
        "hard_negative": df_hard,
    }

    for split_name, split_df in [("train", None), ("val", None), ("test", None)]:
        split_dir = SAMPLES_DIR / model_name / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    for sample_type, df in splits.items():
        if df is None or df.empty:
            log.warning(f"{model_name} {sample_type}: 空样本，跳过划分")
            continue

        train_df, val_df, test_df = split_by_time(df)

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if split_df.empty:
                continue
            path = SAMPLES_DIR / model_name / split_name / f"{sample_type}.csv"
            split_df.to_csv(path, index=False)

        # 报告分布
        log.info(f"  {sample_type}: 训练={len(train_df)} 验证={len(val_df)} 测试={len(test_df)}")


# ============================================================================
# Breakout 样本生成
# ============================================================================
def generate_breakout_samples(start_date: str, end_date: str):
    """生成Breakout模型样本"""
    log.info("=" * 80)
    log.info("BreakoutScorer 样本生成")
    log.info(f"  时间范围: {start_date} ~ {end_date}")
    log.info(f"  划分: 训练≤{TRAIN_END} | 验证{TRAIN_END}~{VAL_END} | 测试>{VAL_END}")
    log.info("=" * 80)

    dm = DataManager(use_cache=True)
    screener = BreakoutSampleScreener(data_manager=dm)

    # 正样本
    log.info("\n[1/3] Breakout 正样本...")
    df_pos = screener.screen_positive_samples(
        start_date=start_date,
        end_date=end_date,
        max_samples=BREAKOUT_CONFIG["positive_max"],
    )
    if df_pos.empty:
        log.error("Breakout正样本生成失败")
        return None, None, None

    _save_raw(df_pos, "breakout_positive_raw.csv", "正样本(原始)")

    # 负样本
    log.info("\n[2/3] Breakout 负样本...")
    df_neg = screener.screen_negative_samples(
        start_date=start_date,
        end_date=end_date,
        positive_df=df_pos,
        target_count=BREAKOUT_CONFIG["negative_target"],
    )
    _save_raw(df_neg, "breakout_negative_raw.csv", "负样本(原始)")

    # 硬负样本
    log.info("\n[3/3] Breakout 硬负样本...")
    df_hard = screener.screen_hard_negative_samples(
        start_date=start_date,
        end_date=end_date,
        target_count=BREAKOUT_CONFIG["hard_negative_target"],
    )
    _save_raw(df_hard, "breakout_hard_negative_raw.csv", "硬负样本(原始)")

    # 时间划分
    log.info("\n[4/4] Breakout 时间划分...")
    save_split_samples(df_pos, df_neg, df_hard, "breakout")

    _validate_model_samples("Breakout", df_pos, df_neg, df_hard)
    return df_pos, df_neg, df_hard


# ============================================================================
# Bounce 样本生成
# ============================================================================
def generate_bounce_samples(start_date: str, end_date: str):
    """生成Bounce模型样本"""
    log.info("\n" + "=" * 80)
    log.info("BounceScorer 样本生成")
    log.info(f"  时间范围: {start_date} ~ {end_date}")
    log.info(f"  划分: 训练≤{TRAIN_END} | 验证{TRAIN_END}~{VAL_END} | 测试>{VAL_END}")
    log.info("=" * 80)

    dm = DataManager(use_cache=True)
    screener = BounceSampleScreener(data_manager=dm)

    # 正样本
    log.info("\n[1/3] Bounce 正样本...")
    df_pos = screener.screen_positive_samples(
        start_date=start_date,
        end_date=end_date,
        max_samples=BOUNCE_CONFIG["positive_max"],
    )
    if df_pos.empty:
        log.error("Bounce正样本生成失败")
        return None, None, None

    _save_raw(df_pos, "bounce_positive_raw.csv", "正样本(原始)")

    # 负样本
    log.info("\n[2/3] Bounce 负样本...")
    df_neg = screener.screen_negative_samples(
        start_date=start_date,
        end_date=end_date,
        positive_df=df_pos,
        target_count=BOUNCE_CONFIG["negative_target"],
    )
    _save_raw(df_neg, "bounce_negative_raw.csv", "负样本(原始)")

    # 硬负样本
    log.info("\n[3/3] Bounce 硬负样本...")
    df_hard = screener.screen_hard_negative_samples(
        start_date=start_date,
        end_date=end_date,
        target_count=BOUNCE_CONFIG["hard_negative_target"],
    )
    _save_raw(df_hard, "bounce_hard_negative_raw.csv", "硬负样本(原始)")

    # 时间划分
    log.info("\n[4/4] Bounce 时间划分...")
    save_split_samples(df_pos, df_neg, df_hard, "bounce")

    _validate_model_samples("Bounce", df_pos, df_neg, df_hard)
    return df_pos, df_neg, df_hard


# ============================================================================
# 辅助函数
# ============================================================================
def _save_raw(df: pd.DataFrame, filename: str, label: str):
    """保存原始全时间范围样本"""
    if df.empty:
        log.warning(f"{label}: 0 个 (空)")
        return
    path = SAMPLES_DIR / filename
    df.to_csv(path, index=False)
    log.success(f"{label}: {len(df)} 个 -> {path}")

    # 时间分布
    df_copy = df.copy()
    df_copy["t1_date"] = pd.to_datetime(df_copy["t1_date"].astype(str), format="%Y%m%d", errors="coerce")
    df_copy["year"] = df_copy["t1_date"].dt.year
    yearly = df_copy["year"].value_counts().sort_index()
    log.info(f"  年度分布: {dict(yearly)}")

    # 硬负类型分布
    if "hard_negative_type" in df.columns:
        type_dist = df["hard_negative_type"].value_counts().to_dict()
        log.info(f"  硬负类型: {type_dist}")


def _validate_model_samples(model_name: str, df_pos, df_neg, df_hard):
    """验证单个模型的样本分布"""
    log.info(f"\n--- {model_name} 全量样本验证 ---")

    pos_count = len(df_pos) if df_pos is not None else 0
    neg_count = len(df_neg) if df_neg is not None else 0
    hard_count = len(df_hard) if df_hard is not None else 0
    total = pos_count + neg_count + hard_count

    if total == 0:
        log.error(f"{model_name}: 无样本")
        return

    neg_pos_ratio = neg_count / pos_count if pos_count > 0 else 0
    hard_ratio = hard_count / total

    log.info(f"  正样本: {pos_count}")
    log.info(f"  负样本: {neg_count} (负/正比例: {neg_pos_ratio:.2f})")
    log.info(f"  硬负样本: {hard_count} (占比: {hard_ratio:.1%})")
    log.info(f"  总样本: {total}")

    # 时间划分后各集统计
    for df, name in [(df_pos, "positive"), (df_neg, "negative"), (df_hard, "hard_negative")]:
        if df is None or df.empty:
            continue
        train_df, val_df, test_df = split_by_time(df)
        log.info(f"  {name}: 训练={len(train_df)} 验证={len(val_df)} 测试={len(test_df)}")

    # 门槛检查
    if pos_count < 500:
        log.warning(f"  ⚠️ 正样本数量不足 (目标 ≥ 500)")
    if neg_pos_ratio < 1.5:
        log.warning(f"  ⚠️ 负/正比例偏低 (目标 ≥ 1.5)")
    if hard_ratio < 0.05 or hard_ratio > 0.25:
        log.warning(f"  ⚠️ 硬负比例异常 (目标 5%-25%)")


def main():
    parser = argparse.ArgumentParser(description="v3.1.0 双模型样本生成（扩展时间范围）")
    parser.add_argument(
        "--model", choices=["breakout", "bounce", "both"], default="both",
        help="生成哪个模型的样本 (default: both)"
    )
    parser.add_argument(
        "--start_date", default=DEFAULT_START_DATE,
        help=f"样本起始日期 (default: {DEFAULT_START_DATE})"
    )
    parser.add_argument(
        "--end_date", default=DEFAULT_END_DATE,
        help=f"样本结束日期 (default: {DEFAULT_END_DATE})"
    )
    parser.add_argument(
        "--train_end", default=TRAIN_END,
        help=f"训练集截止日期 (default: {TRAIN_END})"
    )
    parser.add_argument(
        "--val_end", default=VAL_END,
        help=f"验证集截止日期 (default: {VAL_END})"
    )
    args = parser.parse_args()

    train_end = args.train_end
    val_end = args.val_end

    log.info("=" * 80)
    log.info("v3.1.0 双模型样本生成启动（扩展时间范围）")
    log.info(f"  生成范围: {args.start_date} ~ {args.end_date}")
    log.info(f"  训练集: ≤ {train_end}")
    log.info(f"  验证集: {train_end} ~ {val_end}")
    log.info(f"  测试集: > {val_end}")
    log.info(f"  输出目录: {SAMPLES_DIR}")
    log.info("=" * 80)

    if args.model in ("breakout", "both"):
        generate_breakout_samples(args.start_date, args.end_date)

    if args.model in ("bounce", "both"):
        generate_bounce_samples(args.start_date, args.end_date)

    log.info("\n" + "=" * 80)
    log.info("v3.1.0 样本生成完成")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
