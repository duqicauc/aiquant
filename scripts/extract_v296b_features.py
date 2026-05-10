#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.6b 分时段对比聚合特征提取脚本

基于 v295 样本，使用 UnifiedFeatureExtractor 统一提取三类样本特征，
并运行强制校验（NaN=0%, 列一致性, Inf检查等）。

原则: 所有数据基于真实API，不模拟任何数据。

Usage:
    python scripts/extract_v296b_features.py

Input:
    data/training/samples/positive_samples_v295.csv
    data/training/samples/negative_samples_v295.csv
    data/training/samples/hard_negatives_v295.csv

Output:
    data/training/v296b/positive_features.csv
    data/training/v296b/negative_features.csv
    data/training/v296b/hard_negative_features.csv
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
from src.features.time_series_aggregator import TimeSeriesAggregator
from src.utils.logger import log

# ============================================================================
# 配置
# ============================================================================
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "v296b"
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
def _extract_t1_rows(df: pd.DataFrame) -> pd.DataFrame:
    """从时间序列特征中提取每组最后一行(T1日期的特征)"""
    if df.empty:
        return df
    if "sample_id" not in df.columns:
        log.warning("缺少 sample_id 列，无法提取T1行")
        return df
    # 按 sample_id 分组，取每组 trade_date 最大的那行
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    t1_rows = df.loc[df.groupby("sample_id")["trade_date"].idxmax()].copy()
    return t1_rows.reset_index(drop=True)


def _clean_features_phase1(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """
    第一阶段清洗：提取T1行 + 丢弃全NaN列
    （不丢弃高NaN列，留给全局统一处理）
    """
    original_n = len(df)
    t1_df = _extract_t1_rows(df)
    log.info(f"{sample_type}: 时间序列 {original_n} 行 -> T1行 {len(t1_df)} 行")

    # 丢弃全NaN列
    all_nan_cols = t1_df.columns[t1_df.isna().all()].tolist()
    if all_nan_cols:
        log.info(f"{sample_type}: 丢弃 {len(all_nan_cols)} 个全NaN列: {all_nan_cols[:10]}{'...' if len(all_nan_cols) > 10 else ''}")
        t1_df = t1_df.drop(columns=all_nan_cols)

    return t1_df


def _clean_features_phase2(df: pd.DataFrame, sample_type: str, global_high_nan_cols: list) -> pd.DataFrame:
    """
    第二阶段清洗：丢弃全局高NaN列 + 丢弃剩余含NaN的样本行
    """
    # 丢弃全局高NaN列(>10%)
    existing = [c for c in global_high_nan_cols if c in df.columns]
    if existing:
        log.info(f"{sample_type}: 丢弃 {len(existing)} 个全局高NaN列(>10%): {existing[:10]}{'...' if len(existing) > 10 else ''}")
        df = df.drop(columns=existing)

    # 丢弃剩余含NaN的样本行
    nan_per_row = df.isna().sum(axis=1)
    bad_rows = nan_per_row[nan_per_row > 0]
    if not bad_rows.empty:
        log.warning(f"{sample_type}: {len(bad_rows)} 个样本仍有NaN，将被丢弃")
        nan_cols = df.isna().mean()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        for col, rate in nan_cols.head(5).items():
            log.warning(f"  {col}: {rate*100:.1f}% NaN")
        df = df.dropna()

    return df


def extract_features_for_type(samples_df: pd.DataFrame, sample_type: str, label: int) -> pd.DataFrame:
    """为单一类型样本提取特征并校验（v2.9.6: 增加时间序列聚合特征）"""
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

    # ========== 新增：时间序列聚合特征 ==========
    log.info(f"{sample_type}: 计算时间序列聚合统计特征...")
    aggregator = TimeSeriesAggregator()
    df_agg = aggregator.aggregate(df_features)

    # 提取T1行
    t1_df = _extract_t1_rows(df_features)
    log.info(f"{sample_type}: 时间序列 {len(df_features)} 行 -> T1行 {len(t1_df)} 行")

    # 将聚合特征合并到T1行
    if not df_agg.empty:
        t1_df = aggregator.merge_with_t1(t1_df, df_agg)
    # =============================================

    # 第一阶段清洗：丢弃全NaN列
    all_nan_cols = t1_df.columns[t1_df.isna().all()].tolist()
    if all_nan_cols:
        log.info(f"{sample_type}: 丢弃 {len(all_nan_cols)} 个全NaN列: {all_nan_cols[:10]}{'...' if len(all_nan_cols) > 10 else ''}")
        t1_df = t1_df.drop(columns=all_nan_cols)

    if t1_df.empty:
        log.error(f"{sample_type}: 第一阶段清洗后无有效数据")
        return pd.DataFrame()

    log.success(f"{sample_type}: 第一阶段清洗后 {len(t1_df)} 行 × {len(t1_df.columns)} 列")

    return t1_df


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
    log.info("v2.9.6b 分时段对比聚合特征提取")
    log.info("原则: 所有数据基于真实API，不模拟任何数据")
    log.info("=" * 80)

    # 1. 加载样本
    df_pos = load_sample("positive_samples", label=1)
    df_neg = load_sample("negative_samples", label=0)
    df_hard = load_sample("hard_negatives", label=0)

    if df_pos.empty:
        log.error("正样本为空，终止")
        return

    # 2. 提取特征（第一阶段清洗：提取T1行 + 丢弃全NaN列）
    df_pos_features = extract_features_for_type(df_pos, "positive", 1)
    df_neg_features = extract_features_for_type(df_neg, "negative", 0)
    df_hard_features = extract_features_for_type(df_hard, "hard_negative", 0)

    # 如果任何一类失败，终止
    if df_pos_features.empty or df_neg_features.empty or df_hard_features.empty:
        log.error("部分样本特征提取失败，终止")
        return

    # 3. 全局高NaN列计算（合并三类样本T1特征，统一计算NaN率）
    log.info("")
    log.info("=" * 80)
    log.info("全局高NaN列计算（合并三类样本统一评估）")
    log.info("=" * 80)

    all_t1 = pd.concat([df_pos_features, df_neg_features, df_hard_features], ignore_index=True)
    global_nan_rates = all_t1.isna().mean()
    global_high_nan_cols = global_nan_rates[global_nan_rates > 0.10].index.tolist()

    if global_high_nan_cols:
        log.info(f"全局高NaN列(>10%): {global_high_nan_cols}")
        for col in global_high_nan_cols:
            pos_rate = df_pos_features[col].isna().mean() if col in df_pos_features.columns else 0
            neg_rate = df_neg_features[col].isna().mean() if col in df_neg_features.columns else 0
            hard_rate = df_hard_features[col].isna().mean() if col in df_hard_features.columns else 0
            log.info(f"  {col}: 全局={global_nan_rates[col]*100:.1f}%, 正={pos_rate*100:.1f}%, 负={neg_rate*100:.1f}%, 硬负={hard_rate*100:.1f}%")
    else:
        log.info("未发现全局高NaN列")

    # 4. 第二阶段清洗：对每类样本统一丢弃全局高NaN列 + 丢弃含NaN行
    df_pos_features = _clean_features_phase2(df_pos_features, "positive", global_high_nan_cols)
    df_neg_features = _clean_features_phase2(df_neg_features, "negative", global_high_nan_cols)
    df_hard_features = _clean_features_phase2(df_hard_features, "hard_negative", global_high_nan_cols)

    if df_pos_features.empty or df_neg_features.empty or df_hard_features.empty:
        log.error("第二阶段清洗后部分样本无有效数据，终止")
        return

    log.success("第二阶段清洗完成")
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
