#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.8 v2.7.0 原始特征多行增强特征提取脚本

基于 v295 样本，使用 UnifiedFeatureExtractor 统一提取三类样本特征，
并运行强制校验（NaN=0%, 列一致性, Inf检查等）。

原则: 所有数据基于真实API，不模拟任何数据。

Usage:
    python scripts/extract_v298_features.py

Input:
    data/training/samples/positive_samples_v295.csv
    data/training/samples/negative_samples_v295.csv
    data/training/samples/hard_negatives_v295.csv

Output:
    data/training/v298/positive_features.csv
    data/training/v298/negative_features.csv
    data/training/v298/hard_negative_features.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.features.unified_feature_extractor import FeatureValidator, UnifiedFeatureExtractor
from src.features.quality_checker import DataQualityChecker
from src.utils.logger import log

# ============================================================================
# 配置
# ============================================================================
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "v298"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 120

# ============================================================================
# 加载样本
# ============================================================================
def load_sample(name: str, label: int) -> pd.DataFrame:
    """加载样本文件"""
    path = SAMPLES_DIR / f"{name}_v295.csv"
    if not path.exists():
        log.error(f"样本文件不存在: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"加载 {name}: {len(df)} 条")

    # 确保有 sample_id
    if "sample_id" not in df.columns:
        df["sample_id"] = range(len(df))

    # 确保有 name 列（硬负可能没有）
    if "name" not in df.columns:
        df["name"] = ""

    return df


# ============================================================================
# 特征提取
# ============================================================================
# v2.7.0 原始特征列（173个）
V27_FEATURES = [
    'ma10', 'price_position_55d', 'return_55d', 'support_20d', 'breakout_ma10',
    'resistance_55d', 'price_vs_ma_55d', 'low_34d', 'trend_slope_34d', 'price_vs_ma_34d',
    'vol', 'dist_to_support_20d', 'volume_trend_slope_10d', 'obv_calc', 'breakout_high_55d',
    'total_mv', 'ma_8d', 'high_volume_breakout', 'support_strength_10d', 'macd_dea',
    'ma_34d', 'volume_ratio', 'turnover_rate', 'volume_rsv_20d', 'breakout_ma5',
    'volume_trend_slope_20d', 'momentum_10d', 'volume_change', 'volume_price_corr_10d',
    'close', 'price_down_vol_up_count_10d', 'price_down_vol_up', 'price_vs_ma_8d',
    'low_55d', 'support_55d', 'resistance_20d', 'volume_price_match_sum_10d',
    'volume_price_corr_20d', 'breakout_ma55', 'high', 'trend_slope_8d',
    'volume_breakout_count_20d', 'breakout_high_10d', 'high_8d', 'low_8d', 'open',
    'change', 'resistance_strength_10d', 'price_position_34d', 'pct_chg', 'high_34d',
    'rsi_12', 'macd', 'low', 'volatility_34d', 'trend_slope_55d', 'momentum_5d',
    'return_8d', 'dist_to_support_55d', 'obv_ma10', 'breakout_ma20',
    'dist_to_resistance_55d', 'obv_trend', 'momentum_20d', 'ma5', 'support_strength_20d',
    'return_34d', 'channel_width_20d', 'resistance_10d', 'circ_mv', 'price_change',
    'high_55d', 'consecutive_new_high', 'volume_price_match', 'price_position_8d',
    'price_up_vol_down_count_10d', 'support_10d', 'resistance_strength_20d', 'ma_10d',
    'dist_to_resistance_20d', 'volatility_55d', 'ma_5d', 'momentum_acceleration',
    'ma_55d', 'support_strength_55d', 'price_up_vol_down', 'dist_to_resistance_10d',
    'amount', 'rsi_6', 'pre_close', 'ma_20d', 'breakout_volume_ratio',
    'breakout_high_20d', 'dist_to_support_10d', 'macd_dif', 'rsi_24', 'volatility_8d',
    'resistance_strength_55d', 'price_vs_hist_mean', 'price_vs_hist_high',
    'volatility_vs_hist', 'turnover_rate_f', 'bias_short', 'bias_mid', 'bias_long',
    'ema_5', 'ema_10', 'ema_20', 'ema_60', 'obv', 'vol_ma5_ratio', 'vol_ma20_ratio',
    'is_limit_up', 'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_55d',
    'atr_14', 'atr_ratio_14', 'atr_expansion', 'days_from_high_20d', 'days_from_high_55d',
    'recovery_ratio_20d', 'price_range_pct', 'close_vs_ma10_std', 'days_near_ma10',
    'volume_shrink_ratio', 'ma10_cross_count', 'kdj_d', 'kdj_j', 'kdj_k',
    'prev_high_20d', 'prev_high_55d', 'prev_high_10d', 'breakout_with_volume',
    'momentum_market_interaction', 'rsi_kdj_divergence', 'trend_consistency',
    'volume_price_divergence', 'breakout_rsi_interaction', 'relative_volatility',
    'resonance_volume_confirm', 'market_pct_chg', 'market_return_34d',
    'market_volatility_34d', 'market_trend', 'market_momentum_5d',
    'market_momentum_10d', 'market_momentum_20d', 'market_regime',
    'market_position_20d', 'excess_return', 'excess_return_cumsum',
    'excess_return_consistency', 'breakout_strength_10d', 'breakout_strength_20d',
    'breakout_strength_55d', 'breakout_volume_strength', 'breakout_confirmed_10d',
    'breakout_confirmed_20d', 'breakout_resonance', 'turnover_zscore',
    'turnover_change_rate', 'turnover_spike', 'rsi_kdj_golden_cross',
    'rsi_kdj_strength', 'rsi_zone', 'volume_price_divergence_strength',
    'volume_price_confirm', 'breakout_strength_avg', 'breakout_strength_max',
    'ma_alignment_score', 'price_position_avg', 'sharpe_like_34d',
]

META_COLS = ["sample_id", "ts_code", "name", "trade_date", "days_to_t1", "label"]


def _filter_v27_features(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 v2.7.0 的原始特征列 + 元数据列"""
    keep_cols = []
    for c in META_COLS + V27_FEATURES:
        if c in df.columns:
            keep_cols.append(c)
    missing_features = set(V27_FEATURES) - set(df.columns)
    if missing_features:
        log.warning(f"v2.7.0 特征缺失 {len(missing_features)} 个: {list(missing_features)[:10]}")
    log.info(f"保留 {len(keep_cols)} 列（元数据 {len([c for c in keep_cols if c in META_COLS])} + 特征 {len([c for c in keep_cols if c in V27_FEATURES])}）")
    return df[keep_cols].copy()


def _extract_34d_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取每个样本最近 34 天的所有行（对齐 v2.7.0 的 34d 回看窗口）。
    所有行共享同一 label。
    """
    if df.empty:
        return df
    if "sample_id" not in df.columns or "days_to_t1" not in df.columns:
        log.warning("缺少 sample_id 或 days_to_t1 列")
        return df

    # 只保留 days_to_t1 >= -34 的行（最近34天）
    df = df.copy()
    df["days_to_t1"] = pd.to_numeric(df["days_to_t1"], errors="coerce")
    df_filtered = df[(df["days_to_t1"] >= -34) & (df["days_to_t1"] <= -1)].copy()

    log.info(f"时间序列 {len(df)} 行 -> 34天回看 {len(df_filtered)} 行")
    return df_filtered.reset_index(drop=True)


def _clean_features_multits(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """
    多行时间序列清洗：丢弃全NaN列 + 用0填充剩余NaN
    （不丢弃含NaN行，因为历史日期缺失是正常的）
    """
    original_n = len(df)

    # 丢弃在所有行上都NaN的列
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        log.info(f"{sample_type}: 丢弃 {len(all_nan_cols)} 个全NaN列")
        df = df.drop(columns=all_nan_cols)

    # 用0填充剩余NaN（树模型可以处理0）
    nan_before = df.isna().sum().sum()
    if nan_before > 0:
        df = df.fillna(0)
        log.info(f"{sample_type}: 填充 {nan_before} 个NaN为0")

    # 检查Inf
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df.values).sum()
    if inf_count > 0:
        log.warning(f"{sample_type}: 发现 {inf_count} 个Inf值，替换为0")
        df = df.replace([np.inf, -np.inf], 0)

    log.success(f"{sample_type}: 清洗后 {len(df)} 行 × {len(df.columns)} 列")
    return df


def extract_features_for_type(samples_df: pd.DataFrame, sample_type: str, label: int) -> pd.DataFrame:
    """为单一类型样本提取特征并校验（v2.9.7: 时间序列多行增强）"""
    if samples_df.empty:
        log.warning(f"{sample_type}: 样本为空，跳过")
        return pd.DataFrame()

    log.info("")
    log.info("=" * 80)
    log.info(f"开始提取: {sample_type} (label={label})")
    log.info("=" * 80)

    extractor = UnifiedFeatureExtractor(use_cache=True)
    df_features = extractor.extract_for_samples(samples_df, lookback_days=LOOKBACK_DAYS, label=label)

    if df_features.empty:
        log.error(f"{sample_type}: 特征提取失败")
        return pd.DataFrame()

    log.success(f"{sample_type}: 提取完成，{len(df_features)} 行 × {len(df_features.columns)} 列")

    # ========== v2.9.8: v2.7.0 原始特征 + 34天多行增强 ==========
    log.info(f"{sample_type}: 提取34天回看多行数据...")
    df_34d = _extract_34d_rows(df_features)
    log.info(f"{sample_type}: 只保留 v2.7.0 原始特征列...")
    df_34d = _filter_v27_features(df_34d)

    if df_34d.empty:
        log.error(f"{sample_type}: 34天提取后无数据")
        return pd.DataFrame()
    # =============================================

    # 清洗
    df_34d = _clean_features_multits(df_34d, sample_type)
    if df_34d.empty:
        log.error(f"{sample_type}: 清洗后无有效数据")
        return pd.DataFrame()

    log.success(f"{sample_type}: 最终 {len(df_34d)} 行 × {len(df_34d.columns)} 列")

    return df_34d


# ============================================================================
# 综合质量检查
# ============================================================================
def run_quality_check(df_pos: pd.DataFrame, df_neg: pd.DataFrame, df_hard: pd.DataFrame) -> bool:
    """运行 DataQualityChecker 综合质量检查"""
    log.info("")
    log.info("=" * 80)
    log.info("运行 DataQualityChecker 综合质量检查")
    log.info("=" * 80)

    checker = DataQualityChecker()

    report = checker.check_all(df_pos, df_neg, df_hard)
    status = report.status

    if status == "pass":
        log.success("质量检查通过")
    elif status == "warning":
        log.warning("质量检查有警告")
    else:
        log.error("质量检查未通过")

    # 保存报告
    report_path = OUTPUT_DIR / "quality_report.json"
    checker.save_report(report, report_path)
    log.info(f"报告已保存: {report_path}")

    return status != "fail"


# ============================================================================
# 跨样本一致性检查
# ============================================================================
def check_cross_consistency(df_pos: pd.DataFrame, df_neg: pd.DataFrame, df_hard: pd.DataFrame) -> bool:
    """检查三类样本的特征列是否完全一致"""
    log.info("")
    log.info("=" * 80)
    log.info("跨样本特征一致性检查")
    log.info("=" * 80)

    cols_pos = set(df_pos.columns)
    cols_neg = set(df_neg.columns)
    cols_hard = set(df_hard.columns)

    common = cols_pos & cols_neg & cols_hard
    only_pos = cols_pos - cols_neg - cols_hard
    only_neg = cols_neg - cols_pos - cols_hard
    only_hard = cols_hard - cols_pos - cols_neg

    log.info(f"正样本列数: {len(cols_pos)}")
    log.info(f"负样本列数: {len(cols_neg)}")
    log.info(f"硬负样本列数: {len(cols_hard)}")
    log.info(f"共同列数: {len(common)}")

    if only_pos:
        log.warning(f"仅正样本有的列: {sorted(only_pos)}")
    if only_neg:
        log.warning(f"仅负样本有的列: {sorted(only_neg)}")
    if only_hard:
        log.warning(f"仅硬负样本有的列: {sorted(only_hard)}")

    if cols_pos == cols_neg == cols_hard:
        log.success("✅ 三类样本特征列完全一致")
        return True
    else:
        log.error("❌ 三类样本特征列不一致！")
        return False


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("=" * 80)
    log.info("v2.9.8 v2.7.0 原始特征多行增强特征提取")
    log.info("原则: 所有数据基于真实API，不模拟任何数据")
    log.info("=" * 80)

    # 1. 加载样本
    df_pos = load_sample("positive_samples", label=1)
    df_neg = load_sample("negative_samples", label=0)
    df_hard = load_sample("hard_negatives", label=0)

    if df_pos.empty:
        log.error("正样本为空，终止")
        return

    # 2. 提取特征（多行时间序列增强）
    df_pos_features = extract_features_for_type(df_pos, "positive", 1)
    df_neg_features = extract_features_for_type(df_neg, "negative", 0)
    df_hard_features = extract_features_for_type(df_hard, "hard_negative", 0)

    # 如果任何一类失败，终止
    if df_pos_features.empty or df_neg_features.empty or df_hard_features.empty:
        log.error("部分样本特征提取失败，终止")
        return

    log.info("")
    log.info("=" * 80)
    log.info("多行数据汇总")
    log.info("=" * 80)
    log.info(f"  正样本: {len(df_pos_features)} 行 × {len(df_pos_features.columns)} 列")
    log.info(f"  负样本: {len(df_neg_features)} 行 × {len(df_neg_features.columns)} 列")
    log.info(f"  硬负样本: {len(df_hard_features)} 行 × {len(df_hard_features.columns)} 列")

    # 5. 死特征检测（记录但不自动丢弃，根因分析后手动修复）
    log.info("")
    log.info("=" * 80)
    log.info("死特征检测（全局零值率>99%且唯一值<=2）—— 仅记录，不自动丢弃")
    log.info("=" * 80)

    all_features = pd.concat([df_pos_features, df_neg_features, df_hard_features], ignore_index=True)
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns
    dead_cols = []
    for col in numeric_cols:
        if col in {"label", "sample_id"}:
            continue
        zero_ratio = (all_features[col] == 0).mean()
        unique_count = all_features[col].nunique()
        if zero_ratio > 0.99 and unique_count <= 2:
            dead_cols.append(col)
            log.warning(f"发现死特征: '{col}' (零值率={zero_ratio*100:.1f}%, 唯一值={unique_count})")

    if dead_cols:
        log.info(f"共发现 {len(dead_cols)} 个死特征，已记录待修复")
        log.info(f"  死特征列表: {dead_cols}")
        log.info(f"  说明: 这些特征在当前样本中几乎无变化，需分析根因后决定是否修复计算逻辑或手动移除")
    else:
        log.info("未发现死特征")

    log.success("死特征检测完成")
    log.info(f"  正样本: {len(df_pos_features)} 行 × {len(df_pos_features.columns)} 列")
    log.info(f"  负样本: {len(df_neg_features)} 行 × {len(df_neg_features.columns)} 列")
    log.info(f"  硬负样本: {len(df_hard_features)} 行 × {len(df_hard_features.columns)} 列")

    # 6. FeatureValidator 强制校验
    for name, df in [("positive", df_pos_features), ("negative", df_neg_features), ("hard_negative", df_hard_features)]:
        log.info(f"{name}: 运行 FeatureValidator...")
        try:
            FeatureValidator.validate(df, sample_type=name)
            log.success(f"{name}: FeatureValidator 通过")
        except ValueError as e:
            log.error(f"{name}: FeatureValidator 失败 - {e}")
            invalid_path = OUTPUT_DIR / f"{name}_features_INVALID.csv"
            df.to_csv(invalid_path, index=False)
            log.info(f"已保存失败数据: {invalid_path}")
            return

    # 7. 跨样本一致性检查
    consistent = check_cross_consistency(df_pos_features, df_neg_features, df_hard_features)
    if not consistent:
        log.error("特征列不一致，终止")
        return

    # 8. 综合质量检查
    quality_pass = run_quality_check(df_pos_features, df_neg_features, df_hard_features)
    if not quality_pass:
        log.error("质量检查未通过，终止")
        return

    # 5. 保存
    log.info("")
    log.info("=" * 80)
    log.info("保存特征数据")
    log.info("=" * 80)

    pos_path = OUTPUT_DIR / "positive_features.csv"
    neg_path = OUTPUT_DIR / "negative_features.csv"
    hard_path = OUTPUT_DIR / "hard_negative_features.csv"

    df_pos_features.to_csv(pos_path, index=False)
    df_neg_features.to_csv(neg_path, index=False)
    df_hard_features.to_csv(hard_path, index=False)

    log.success(f"正样本特征: {pos_path} ({len(df_pos_features)} 行)")
    log.success(f"负样本特征: {neg_path} ({len(df_neg_features)} 行)")
    log.success(f"硬负样本特征: {hard_path} ({len(df_hard_features)} 行)")

    # 汇总
    total_rows = len(df_pos_features) + len(df_neg_features) + len(df_hard_features)
    log.info("")
    log.info("=" * 80)
    log.success("✅ 全部特征提取完成！")
    log.info(f"  总样本-日期记录: {total_rows}")
    log.info(f"  特征维度: {len(df_pos_features.columns)} 列")
    log.info("  下一步: 合并训练数据并训练模型")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
