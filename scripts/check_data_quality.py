#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据质量检查脚本

检查正负样本特征是否对齐，确保训练数据质量
如果特征不对齐，返回非零退出码
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def check_feature_alignment(pos_file: Path, neg_file: Path, max_missing_allowed: int = 3) -> bool:
    """
    检查正负样本特征对齐
    
    Args:
        pos_file: 正样本文件路径
        neg_file: 负样本文件路径
        max_missing_allowed: 允许的最大缺失特征数
    
    Returns:
        True 如果通过检查，False 否则
    """
    log.info("="*80)
    log.info("数据质量检查 - 特征对齐")
    log.info("="*80)
    
    # 检查文件存在
    if not pos_file.exists():
        log.error(f"正样本文件不存在: {pos_file}")
        return False
    
    if not neg_file.exists():
        log.error(f"负样本文件不存在: {neg_file}")
        return False
    
    # 加载数据
    log.info("加载数据...")
    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    
    log.info(f"  正样本: {len(df_pos)} 条, 特征数: {len(df_pos.columns)}")
    log.info(f"  负样本: {len(df_neg)} 条, 特征数: {len(df_neg.columns)}")
    
    # 排除非特征列
    exclude_cols = {'label', 'sample_id', 'ts_code', 'name', 't1_date', 't2_date', 
                    'trade_date', 'list_date', 'pattern_type'}
    
    pos_cols = set(df_pos.columns) - exclude_cols
    neg_cols = set(df_neg.columns) - exclude_cols
    
    common_cols = pos_cols & neg_cols
    pos_only = pos_cols - neg_cols
    neg_only = neg_cols - pos_cols
    
    log.info(f"\n特征统计:")
    log.info(f"  正样本特征数: {len(pos_cols)}")
    log.info(f"  负样本特征数: {len(neg_cols)}")
    log.info(f"  共同特征数: {len(common_cols)}")
    log.info(f"  正样本独有: {len(pos_only)}")
    log.info(f"  负样本独有: {len(neg_only)}")
    
    # 检查关键特征
    key_features = ['circ_mv', 'total_mv', 'rsi_6', 'rsi_12', 'rsi_24', 
                    'macd_dif', 'macd_dea', 'macd', 'ma5', 'ma10']
    
    log.info(f"\n关键特征检查:")
    missing_key_features = []
    for feat in key_features:
        in_pos = feat in pos_cols
        in_neg = feat in neg_cols
        if in_pos and in_neg:
            status = "✓"
        elif in_pos:
            status = "⚠ 负样本缺失"
            missing_key_features.append(feat)
        elif in_neg:
            status = "⚠ 正样本缺失"
        else:
            status = "✗ 都缺失"
        log.info(f"  {feat}: {status}")
    
    # 输出差异详情
    if pos_only:
        log.warning(f"\n正样本独有特征 ({len(pos_only)}个):")
        for col in sorted(pos_only):
            log.info(f"  - {col}")
    
    if neg_only:
        log.warning(f"\n负样本独有特征 ({len(neg_only)}个):")
        for col in sorted(neg_only):
            log.info(f"  - {col}")
    
    # 判断是否通过
    if len(pos_only) <= max_missing_allowed and len(neg_only) <= max_missing_allowed:
        log.success(f"\n✓ 特征对齐检查通过！")
        log.info(f"  允许的最大差异: {max_missing_allowed} 个特征")
        log.info(f"  实际差异: 正样本独有 {len(pos_only)}, 负样本独有 {len(neg_only)}")
        return True
    else:
        log.error(f"\n✗ 特征对齐检查未通过！")
        log.error(f"  允许的最大差异: {max_missing_allowed} 个特征")
        log.error(f"  实际差异: 正样本独有 {len(pos_only)}, 负样本独有 {len(neg_only)}")
        
        if missing_key_features:
            log.error(f"\n缺失的关键特征: {missing_key_features}")
            log.error(f"请运行: python scripts/fix_negative_sample_features.py")
        
        return False


def check_sample_quality(pos_file: Path, neg_file: Path) -> bool:
    """
    检查样本质量
    """
    log.info("\n" + "="*80)
    log.info("数据质量检查 - 样本质量")
    log.info("="*80)
    
    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    
    # 检查样本数量
    pos_samples = df_pos['sample_id'].nunique()
    neg_samples = df_neg['sample_id'].nunique()
    
    log.info(f"\n样本数量:")
    log.info(f"  正样本: {pos_samples}")
    log.info(f"  负样本: {neg_samples}")
    log.info(f"  正负比例: 1:{neg_samples/pos_samples:.2f}")
    
    # 检查缺失值
    pos_missing = df_pos.isnull().sum().sum()
    neg_missing = df_neg.isnull().sum().sum()
    
    pos_missing_pct = pos_missing / (len(df_pos) * len(df_pos.columns)) * 100
    neg_missing_pct = neg_missing / (len(df_neg) * len(df_neg.columns)) * 100
    
    log.info(f"\n缺失值:")
    log.info(f"  正样本: {pos_missing} ({pos_missing_pct:.2f}%)")
    log.info(f"  负样本: {neg_missing} ({neg_missing_pct:.2f}%)")
    
    # 检查是否有严重问题
    if pos_missing_pct > 10 or neg_missing_pct > 10:
        log.warning("⚠ 缺失值比例较高，可能影响模型训练")
    else:
        log.success("✓ 缺失值比例正常")
    
    return True


def check_hard_negative_alignment(pos_file: Path, hard_neg_file: Path, max_missing_allowed: int = 3) -> bool:
    """
    检查硬负样本特征对齐
    """
    log.info("\n" + "="*80)
    log.info("数据质量检查 - 硬负样本特征对齐")
    log.info("="*80)
    
    if not hard_neg_file.exists():
        log.warning("硬负样本文件不存在，跳过检查")
        return True
    
    # 加载数据
    df_pos = pd.read_csv(pos_file)
    df_hard_neg = pd.read_csv(hard_neg_file)
    
    log.info(f"  正样本: {len(df_pos)} 条, 特征数: {len(df_pos.columns)}")
    log.info(f"  硬负样本: {len(df_hard_neg)} 条, 特征数: {len(df_hard_neg.columns)}")
    
    # 排除非特征列
    exclude_cols = {'label', 'sample_id', 'ts_code', 'name', 't1_date', 't2_date', 
                    'trade_date', 'list_date', 'pattern_type'}
    
    pos_cols = set(df_pos.columns) - exclude_cols
    hard_neg_cols = set(df_hard_neg.columns) - exclude_cols
    
    common_cols = pos_cols & hard_neg_cols
    pos_only = pos_cols - hard_neg_cols
    hard_neg_only = hard_neg_cols - pos_cols
    
    log.info(f"\n特征统计:")
    log.info(f"  正样本特征数: {len(pos_cols)}")
    log.info(f"  硬负样本特征数: {len(hard_neg_cols)}")
    log.info(f"  共同特征数: {len(common_cols)}")
    log.info(f"  正样本独有: {len(pos_only)}")
    log.info(f"  硬负样本独有: {len(hard_neg_only)}")
    
    if pos_only:
        log.warning(f"\n正样本独有特征 ({len(pos_only)}个):")
        for col in sorted(list(pos_only))[:20]:
            log.info(f"  - {col}")
        if len(pos_only) > 20:
            log.info(f"  ... (共{len(pos_only)}个)")
    
    if hard_neg_only:
        log.warning(f"\n硬负样本独有特征 ({len(hard_neg_only)}个):")
        for col in sorted(list(hard_neg_only))[:20]:
            log.info(f"  - {col}")
        if len(hard_neg_only) > 20:
            log.info(f"  ... (共{len(hard_neg_only)}个)")
    
    # 判断是否通过
    if len(pos_only) <= max_missing_allowed:
        log.success(f"\n✓ 硬负样本特征对齐检查通过！")
        log.info(f"  允许的最大差异: {max_missing_allowed} 个特征")
        log.info(f"  实际差异: 正样本独有 {len(pos_only)}, 硬负样本独有 {len(hard_neg_only)}")
        return True
    else:
        log.error(f"\n✗ 硬负样本特征对齐检查未通过！")
        log.error(f"  允许的最大差异: {max_missing_allowed} 个特征")
        log.error(f"  实际差异: 正样本独有 {len(pos_only)}, 硬负样本独有 {len(hard_neg_only)}")
        log.error(f"\n请运行优化特征工程脚本处理硬负样本:")
        log.error(f"  python scripts/add_advanced_factors_optimized.py")
        return False


def main():
    log.info("="*80)
    log.info("v2.5.0 数据质量检查（包含硬负样本）")
    log.info("="*80)
    
    # ✅ 优先使用对齐后的版本 (_aligned.csv)
    pos_file_aligned = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5_aligned.csv'
    pos_file_v5 = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    
    if pos_file_aligned.exists():
        pos_file = pos_file_aligned
        log.info(f"✓ 使用对齐后的正样本: {pos_file.name}")
    elif pos_file_v5.exists():
        pos_file = pos_file_v5
        log.warning(f"⚠️  对齐版本不存在，使用v5: {pos_file.name}")
    else:
        log.error("✗ 正样本文件不存在！")
        sys.exit(1)
    
    # 负样本：优先对齐版本 > 修复版本 > v5
    neg_file_aligned = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5_aligned.csv'
    neg_file_fixed = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5_fixed.csv'
    neg_file_v5 = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv'
    
    if neg_file_aligned.exists():
        neg_file = neg_file_aligned
        log.info(f"✓ 使用对齐后的负样本: {neg_file.name}")
    elif neg_file_fixed.exists():
        neg_file = neg_file_fixed
        log.warning(f"⚠️  对齐版本不存在，使用修复版: {neg_file.name}")
    elif neg_file_v5.exists():
        neg_file = neg_file_v5
        log.warning(f"⚠️  修复版不存在，使用v5: {neg_file.name}")
    else:
        log.error("✗ 负样本文件不存在！")
        sys.exit(1)
    
    # 硬负样本文件（v4是最全的，不需要对齐）
    hard_neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v4.csv'
    if not hard_neg_file.exists():
        hard_neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v3.csv'
    
    log.info(f"  硬负样本: {hard_neg_file.name}")
    
    # 检查正负样本特征对齐
    alignment_ok = check_feature_alignment(pos_file, neg_file, max_missing_allowed=3)
    
    # 检查硬负样本特征对齐
    hard_neg_alignment_ok = check_hard_negative_alignment(pos_file, hard_neg_file, max_missing_allowed=3)
    
    # 检查样本质量
    quality_ok = check_sample_quality(pos_file, neg_file)
    
    # 总结
    log.info("\n" + "="*80)
    log.info("检查结果总结")
    log.info("="*80)
    
    all_ok = alignment_ok and hard_neg_alignment_ok and quality_ok
    
    if all_ok:
        log.success("✓ 所有检查通过！可以开始训练")
        sys.exit(0)
    else:
        log.error("✗ 检查未通过！请先修复问题")
        if not alignment_ok:
            log.error("  - 正负样本特征对齐失败，请运行: python scripts/align_all_sample_features.py")
        if not hard_neg_alignment_ok:
            log.error("  - 硬负样本特征对齐失败，请运行: python scripts/align_all_sample_features.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
