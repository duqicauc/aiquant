#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.5.0训练数据质量评估脚本

评估内容：
1. 样本分布统计（正负样本比例、时间分布、股票分布）
2. 特征完整性检查（缺失值、无穷值、重复特征）
3. 特征质量分析（方差、相关性、异常值比例）
4. 数据一致性检查（v3和v4合并后的一致性）
5. 生成详细的质量评估报告
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log


def check_file_exists(file_path):
    """检查文件是否存在"""
    if not file_path.exists():
        log.error(f"文件不存在: {file_path}")
        return False
    return True


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        log.success(f"✓ 加载成功: {file_path.name}")
        log.info(f"  行数: {len(df):,}, 列数: {len(df.columns)}")
        return df
    except Exception as e:
        log.error(f"加载失败: {file_path.name}, 错误: {e}")
        return None


def evaluate_sample_distribution(df_pos, df_neg):
    """评估样本分布"""
    log.info("\n" + "="*80)
    log.info("📊 1. 样本分布统计")
    log.info("="*80)
    
    stats = {}
    
    # 1.1 正负样本数量
    pos_samples = df_pos['sample_id'].nunique()
    neg_samples = df_neg['sample_id'].nunique()
    total_samples = pos_samples + neg_samples
    
    log.info(f"\n样本数量:")
    log.info(f"  正样本: {pos_samples:,} ({pos_samples/total_samples*100:.1f}%)")
    log.info(f"  负样本: {neg_samples:,} ({neg_samples/total_samples*100:.1f}%)")
    log.info(f"  总计: {total_samples:,}")
    log.info(f"  正负比例: 1:{neg_samples/pos_samples:.2f}")
    
    stats['sample_count'] = {
        'positive': pos_samples,
        'negative': neg_samples,
        'total': total_samples,
        'ratio': f"1:{neg_samples/pos_samples:.2f}"
    }
    
    # 1.2 时间分布
    log.info(f"\n时间分布:")
    for label, df in [('正样本', df_pos), ('负样本', df_neg)]:
        if 't1_date' in df.columns:
            t1_dates = pd.to_datetime(df.groupby('sample_id')['t1_date'].first(), format='%Y%m%d')
            log.info(f"  {label}:")
            log.info(f"    T1日期范围: {t1_dates.min().date()} ~ {t1_dates.max().date()}")
            log.info(f"    时间跨度: {(t1_dates.max() - t1_dates.min()).days} 天")
            
            # 按年份统计
            year_counts = t1_dates.dt.year.value_counts().sort_index()
            log.info(f"    年份分布:")
            for year, count in year_counts.items():
                log.info(f"      {year}: {count} 个样本")
    
    # 1.3 股票分布
    log.info(f"\n股票分布:")
    pos_stocks = df_pos.groupby('sample_id')['ts_code'].first().nunique()
    neg_stocks = df_neg.groupby('sample_id')['ts_code'].first().nunique()
    log.info(f"  正样本涉及股票数: {pos_stocks}")
    log.info(f"  负样本涉及股票数: {neg_stocks}")
    
    # 检查是否有重叠
    pos_ts_codes = set(df_pos.groupby('sample_id')['ts_code'].first())
    neg_ts_codes = set(df_neg.groupby('sample_id')['ts_code'].first())
    overlap = pos_ts_codes & neg_ts_codes
    log.info(f"  正负样本股票重叠数: {len(overlap)}")
    
    stats['stock_distribution'] = {
        'positive_stocks': pos_stocks,
        'negative_stocks': neg_stocks,
        'overlap': len(overlap)
    }
    
    # 1.4 每个样本的天数统计
    log.info(f"\n每个样本的天数统计:")
    for label, df in [('正样本', df_pos), ('负样本', df_neg)]:
        days_per_sample = df.groupby('sample_id').size()
        log.info(f"  {label}:")
        log.info(f"    平均天数: {days_per_sample.mean():.1f}")
        log.info(f"    中位数: {days_per_sample.median():.0f}")
        log.info(f"    最小: {days_per_sample.min()}, 最大: {days_per_sample.max()}")
        log.info(f"    标准差: {days_per_sample.std():.1f}")
        
        # 检查异常样本（天数过少）
        insufficient = (days_per_sample < 30).sum()
        if insufficient > 0:
            log.warning(f"    ⚠️  天数<30的样本: {insufficient} 个 ({insufficient/len(days_per_sample)*100:.1f}%)")
    
    return stats


def evaluate_feature_completeness(df_pos, df_neg):
    """评估特征完整性"""
    log.info("\n" + "="*80)
    log.info("🔍 2. 特征完整性检查")
    log.info("="*80)
    
    stats = {}
    
    # 2.1 特征列对比
    pos_cols = set(df_pos.columns)
    neg_cols = set(df_neg.columns)
    common_cols = pos_cols & neg_cols
    pos_only = pos_cols - neg_cols
    neg_only = neg_cols - pos_cols
    
    log.info(f"\n特征列对比:")
    log.info(f"  正样本特征数: {len(pos_cols)}")
    log.info(f"  负样本特征数: {len(neg_cols)}")
    log.info(f"  共同特征数: {len(common_cols)}")
    
    if pos_only:
        log.warning(f"  ⚠️  仅正样本有: {len(pos_only)} 个特征")
        log.info(f"      {list(pos_only)[:5]}...")
    
    if neg_only:
        log.warning(f"  ⚠️  仅负样本有: {len(neg_only)} 个特征")
        log.info(f"      {list(neg_only)[:5]}...")
    
    stats['feature_columns'] = {
        'positive_features': len(pos_cols),
        'negative_features': len(neg_cols),
        'common_features': len(common_cols),
        'positive_only': len(pos_only),
        'negative_only': len(neg_only)
    }
    
    # 2.2 缺失值检查
    log.info(f"\n缺失值检查:")
    for label, df in [('正样本', df_pos), ('负样本', df_neg)]:
        missing = df.isnull().sum()
        missing_features = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing_features) > 0:
            log.warning(f"  {label}: {len(missing_features)} 个特征有缺失值")
            log.info(f"    缺失最多的特征:")
            for feat, count in missing_features.head(5).items():
                pct = count / len(df) * 100
                log.info(f"      {feat}: {count:,} ({pct:.2f}%)")
        else:
            log.success(f"  {label}: ✓ 无缺失值")
    
    # 2.3 无穷值检查
    log.info(f"\n无穷值检查:")
    for label, df in [('正样本', df_pos), ('负样本', df_neg)]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            log.warning(f"  {label}: {len(inf_counts)} 个特征有无穷值")
            sorted_inf = sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info(f"    无穷值最多的特征:")
            for feat, count in sorted_inf:
                log.info(f"      {feat}: {count:,}")
        else:
            log.success(f"  {label}: ✓ 无无穷值")
    
    return stats


def evaluate_feature_quality(df_pos, df_neg):
    """评估特征质量"""
    log.info("\n" + "="*80)
    log.info("📈 3. 特征质量分析")
    log.info("="*80)
    
    stats = {}
    
    # 排除非特征列
    exclude_cols = ['ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label', 
                    'trade_date', 'weekly_return_1', 'weekly_return_2', 'weekly_return_3',
                    'total_return_34d', 'weekly_volume_1', 'weekly_volume_2', 'weekly_volume_3']
    
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    log.info(f"\n特征统计:")
    log.info(f"  总特征数: {len(feature_cols)}")
    log.info(f"  数值特征数: {len(numeric_features)}")
    
    # 3.1 零方差特征
    log.info(f"\n零方差特征检查:")
    zero_var_features = []
    for col in numeric_features:
        var = df[col].var()
        if pd.isna(var) or var < 1e-10:
            zero_var_features.append(col)
    
    if zero_var_features:
        log.warning(f"  ⚠️  发现 {len(zero_var_features)} 个零方差特征:")
        log.info(f"      {zero_var_features[:10]}")
        if len(zero_var_features) > 10:
            log.info(f"      ... 还有 {len(zero_var_features)-10} 个")
    else:
        log.success(f"  ✓ 无零方差特征")
    
    stats['zero_variance_features'] = len(zero_var_features)
    
    # 3.2 高相关性特征
    log.info(f"\n高相关性特征检查（相关系数>0.95）:")
    df_clean = df[numeric_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    # 随机抽样以加速计算（如果特征太多）
    if len(numeric_features) > 100:
        sample_features = np.random.choice(numeric_features, 100, replace=False)
        df_sample = df_clean[sample_features]
        log.info(f"  （从{len(numeric_features)}个特征中抽样100个进行相关性分析）")
    else:
        df_sample = df_clean
    
    corr_matrix = df_sample.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column] > 0.95]
        for idx in high_corr.index:
            high_corr_pairs.append((column, idx, high_corr[idx]))
    
    if high_corr_pairs:
        log.warning(f"  ⚠️  发现 {len(high_corr_pairs)} 对高相关特征")
        log.info(f"    前5对:")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            log.info(f"      {feat1} <-> {feat2}: {corr:.3f}")
    else:
        log.success(f"  ✓ 无高相关特征对")
    
    stats['high_correlation_pairs'] = len(high_corr_pairs)
    
    # 3.3 异常值比例
    log.info(f"\n异常值比例检查（使用3σ原则）:")
    outlier_features = {}
    for col in numeric_features[:20]:  # 检查前20个特征
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            outliers = np.abs(values - mean) > 3 * std
            outlier_pct = outliers.sum() / len(values) * 100
            if outlier_pct > 5:  # 超过5%认为异常
                outlier_features[col] = outlier_pct
    
    if outlier_features:
        log.warning(f"  ⚠️  {len(outlier_features)} 个特征的异常值比例>5%")
        sorted_outliers = sorted(outlier_features.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info(f"    异常值最多的特征:")
        for feat, pct in sorted_outliers:
            log.info(f"      {feat}: {pct:.1f}%")
    else:
        log.success(f"  ✓ 特征异常值比例正常")
    
    stats['high_outlier_features'] = len(outlier_features)
    
    return stats


def evaluate_label_separation(df_pos, df_neg):
    """评估正负样本的特征区分度"""
    log.info("\n" + "="*80)
    log.info("🎯 4. 正负样本特征区分度")
    log.info("="*80)
    
    # 排除非特征列
    exclude_cols = ['ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label', 
                    'trade_date', 'weekly_return_1', 'weekly_return_2', 'weekly_return_3',
                    'total_return_34d', 'weekly_volume_1', 'weekly_volume_2', 'weekly_volume_3']
    
    # 找出共同的特征列
    pos_cols = set(df_pos.columns) - set(exclude_cols)
    neg_cols = set(df_neg.columns) - set(exclude_cols)
    common_features = list(pos_cols & neg_cols)
    
    # 只保留数值型特征
    numeric_features = []
    for col in common_features:
        if df_pos[col].dtype in [np.float64, np.int64] and df_neg[col].dtype in [np.float64, np.int64]:
            numeric_features.append(col)
    
    log.info(f"\n共同数值特征数: {len(numeric_features)}")
    log.info(f"\n抽样检查前10个特征的区分度:")
    log.info(f"{'特征名':<30} {'正样本均值':<15} {'负样本均值':<15} {'差异倍数':<10}")
    log.info("-" * 75)
    
    separation_scores = []
    for col in numeric_features[:10]:
        pos_mean = df_pos[col].mean()
        neg_mean = df_neg[col].mean()
        
        if abs(neg_mean) > 1e-10:
            ratio = abs(pos_mean / neg_mean)
        else:
            ratio = float('inf') if abs(pos_mean) > 1e-10 else 1.0
        
        separation_scores.append(ratio)
        
        log.info(f"{col:<30} {pos_mean:>14.4f} {neg_mean:>14.4f} {ratio:>9.2f}x")
    
    avg_separation = np.mean([s for s in separation_scores if s != float('inf')])
    log.info(f"\n平均特征差异倍数: {avg_separation:.2f}x")
    
    return {'average_separation': avg_separation}


def check_data_consistency(df_pos, df_neg):
    """检查数据一致性"""
    log.info("\n" + "="*80)
    log.info("✅ 5. 数据一致性检查")
    log.info("="*80)
    
    issues = []
    
    # 5.1 检查label列
    if 'label' in df_pos.columns:
        pos_label_check = df_pos['label'].unique()
        if len(pos_label_check) > 1 or (len(pos_label_check) == 1 and pos_label_check[0] != 1):
            issues.append(f"正样本label列值异常: {pos_label_check}")
    
    if 'label' in df_neg.columns:
        neg_label_check = df_neg['label'].unique()
        if len(neg_label_check) > 1 or (len(neg_label_check) == 1 and neg_label_check[0] != 0):
            issues.append(f"负样本label列值异常: {neg_label_check}")
    
    # 5.2 检查sample_id唯一性和连续性
    pos_sample_ids = df_pos.groupby('sample_id').size()
    neg_sample_ids = df_neg.groupby('sample_id').size()
    
    log.info(f"\nsample_id检查:")
    log.info(f"  正样本sample_id数量: {len(pos_sample_ids)}")
    log.info(f"  负样本sample_id数量: {len(neg_sample_ids)}")
    
    # 检查是否有重复的sample_id
    overlap_ids = set(pos_sample_ids.index) & set(neg_sample_ids.index)
    if overlap_ids:
        issues.append(f"正负样本有重复的sample_id: {len(overlap_ids)}个")
        log.warning(f"  ⚠️  正负样本有重复的sample_id: {len(overlap_ids)}个")
    else:
        log.success(f"  ✓ 正负样本sample_id无重复")
    
    # 5.3 检查必要列是否存在
    required_cols = ['ts_code', 'sample_id', 'trade_date']
    for col in required_cols:
        if col not in df_pos.columns:
            issues.append(f"正样本缺少必要列: {col}")
        if col not in df_neg.columns:
            issues.append(f"负样本缺少必要列: {col}")
    
    # 5.4 检查数据类型
    log.info(f"\n数据类型检查:")
    numeric_cols = df_pos.select_dtypes(include=[np.number]).columns
    log.info(f"  正样本数值列数: {len(numeric_cols)}")
    log.info(f"  负样本数值列数: {len(df_neg.select_dtypes(include=[np.number]).columns)}")
    
    if issues:
        log.warning(f"\n⚠️  发现 {len(issues)} 个一致性问题:")
        for issue in issues:
            log.warning(f"    - {issue}")
        return {'issues': issues, 'passed': False}
    else:
        log.success(f"\n✓ 数据一致性检查通过")
        return {'issues': [], 'passed': True}


def generate_quality_report(all_stats, output_dir):
    """生成质量评估报告"""
    log.info("\n" + "="*80)
    log.info("📄 生成质量评估报告")
    log.info("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON报告
    report = {
        'evaluation_time': datetime.now().isoformat(),
        'statistics': all_stats,
        'summary': {
            'total_samples': all_stats.get('sample_distribution', {}).get('sample_count', {}).get('total', 0),
            'feature_count': all_stats.get('feature_completeness', {}).get('feature_columns', {}).get('common_features', 0),
            'quality_issues': sum([
                all_stats.get('feature_quality', {}).get('zero_variance_features', 0),
                all_stats.get('feature_quality', {}).get('high_correlation_pairs', 0),
                all_stats.get('feature_quality', {}).get('high_outlier_features', 0),
            ]),
            'consistency_passed': all_stats.get('consistency', {}).get('passed', False)
        }
    }
    
    json_file = output_dir / 'training_data_quality_report.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.success(f"✓ JSON报告已保存: {json_file}")
    
    # Markdown报告
    md_file = output_dir / 'training_data_quality_report.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# v2.5.0 训练数据质量评估报告\n\n")
        f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 评估总览\n\n")
        summary = report['summary']
        f.write(f"- **总样本数**: {summary['total_samples']:,}\n")
        f.write(f"- **特征数量**: {summary['feature_count']}\n")
        f.write(f"- **质量问题数**: {summary['quality_issues']}\n")
        f.write(f"- **一致性检查**: {'✅ 通过' if summary['consistency_passed'] else '❌ 未通过'}\n\n")
        
        f.write("## 2. 样本分布\n\n")
        sample_stats = all_stats.get('sample_distribution', {}).get('sample_count', {})
        f.write(f"- 正样本: {sample_stats.get('positive', 0):,}\n")
        f.write(f"- 负样本: {sample_stats.get('negative', 0):,}\n")
        f.write(f"- 正负比例: {sample_stats.get('ratio', 'N/A')}\n\n")
        
        f.write("## 3. 特征质量\n\n")
        feature_quality = all_stats.get('feature_quality', {})
        f.write(f"- 零方差特征: {feature_quality.get('zero_variance_features', 0)}\n")
        f.write(f"- 高相关特征对: {feature_quality.get('high_correlation_pairs', 0)}\n")
        f.write(f"- 高异常值特征: {feature_quality.get('high_outlier_features', 0)}\n\n")
        
        f.write("## 4. 数据一致性\n\n")
        consistency = all_stats.get('consistency', {})
        if consistency.get('passed', False):
            f.write("✅ 所有一致性检查通过\n\n")
        else:
            f.write("❌ 发现以下问题:\n\n")
            for issue in consistency.get('issues', []):
                f.write(f"- {issue}\n")
            f.write("\n")
        
        f.write("## 5. 建议\n\n")
        if summary['quality_issues'] == 0 and summary['consistency_passed']:
            f.write("✅ 数据质量良好，可以直接用于模型训练\n")
        else:
            f.write("⚠️ 建议处理以下问题后再进行训练:\n\n")
            if feature_quality.get('zero_variance_features', 0) > 0:
                f.write("- 移除零方差特征\n")
            if feature_quality.get('high_correlation_pairs', 0) > 10:
                f.write("- 考虑移除高度相关的特征以减少冗余\n")
            if not summary['consistency_passed']:
                f.write("- 修复数据一致性问题\n")
    
    log.success(f"✓ Markdown报告已保存: {md_file}")
    
    return json_file, md_file


def main():
    log.info("="*80)
    log.info("v2.5.0 训练数据质量评估")
    log.info("="*80)
    
    # 文件路径
    pos_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv'
    output_dir = PROJECT_ROOT / 'data' / 'training' / 'quality_reports'
    
    # 检查文件
    if not check_file_exists(pos_file) or not check_file_exists(neg_file):
        log.error("\n❌ 请先运行合并脚本生成v5数据:")
        log.error("   python scripts/merge_v3_v4_data.py")
        return
    
    # 加载数据
    log.info("\n" + "="*80)
    log.info("📥 加载数据")
    log.info("="*80)
    
    df_pos = load_data(pos_file)
    df_neg = load_data(neg_file)
    
    if df_pos is None or df_neg is None:
        log.error("数据加载失败！")
        return
    
    # 执行评估
    all_stats = {}
    
    # 1. 样本分布
    all_stats['sample_distribution'] = evaluate_sample_distribution(df_pos, df_neg)
    
    # 2. 特征完整性
    all_stats['feature_completeness'] = evaluate_feature_completeness(df_pos, df_neg)
    
    # 3. 特征质量
    all_stats['feature_quality'] = evaluate_feature_quality(df_pos, df_neg)
    
    # 4. 特征区分度
    all_stats['label_separation'] = evaluate_label_separation(df_pos, df_neg)
    
    # 5. 数据一致性
    all_stats['consistency'] = check_data_consistency(df_pos, df_neg)
    
    # 6. 生成报告
    json_file, md_file = generate_quality_report(all_stats, output_dir)
    
    # 总结
    log.info("\n" + "="*80)
    log.info("📋 评估总结")
    log.info("="*80)
    
    quality_issues = all_stats.get('feature_quality', {}).get('zero_variance_features', 0) + \
                    all_stats.get('feature_quality', {}).get('high_correlation_pairs', 0) + \
                    all_stats.get('feature_quality', {}).get('high_outlier_features', 0)
    
    consistency_passed = all_stats.get('consistency', {}).get('passed', False)
    
    if quality_issues == 0 and consistency_passed:
        log.success("\n✅ 数据质量评估通过！可以进行模型训练。")
    else:
        log.warning(f"\n⚠️  发现 {quality_issues} 个质量问题")
        if not consistency_passed:
            log.warning("⚠️  数据一致性检查未通过")
        log.info("\n建议查看详细报告后决定是否需要数据清洗")
    
    log.info("\n报告文件:")
    log.info(f"  - JSON: {json_file}")
    log.info(f"  - Markdown: {md_file}")
    
    log.info("\n下一步: python scripts/train_v250_model.py")


if __name__ == '__main__':
    main()
