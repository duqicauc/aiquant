#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.5.3模型特征重要性分布，并与v2.5.2对比
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def load_model(version):
    """加载模型"""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version / 'model'
    
    # 加载模型
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'model.json'))
    
    # 加载特征名
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # 加载元数据
    metadata_file = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return booster, feature_names, metadata


def get_feature_importance(booster, feature_names):
    """获取特征重要性"""
    # 获取重要性分数
    importance_dict = booster.get_score(importance_type='gain')
    
    # 转换为DataFrame
    importance_list = []
    for feat in feature_names:
        importance_list.append({
            'feature': feat,
            'importance': importance_dict.get(f'f{feature_names.index(feat)}', 0)
        })
    
    df = pd.DataFrame(importance_list)
    df = df.sort_values('importance', ascending=False)
    
    # 计算占比
    total = df['importance'].sum()
    if total > 0:
        df['percentage'] = df['importance'] / total * 100
        df['cumulative'] = df['percentage'].cumsum()
    else:
        df['percentage'] = 0
        df['cumulative'] = 0
    
    return df


def categorize_feature(feature_name):
    """特征分类"""
    # 先检查市场环境特征（必须在其他检查之前，避免被误分类）
    if 'market' in feature_name or 'excess' in feature_name:
        return '市场环境'
    elif 'circ_mv' in feature_name or 'total_mv' in feature_name or 'amount' in feature_name:
        return '市值/流动性'
    elif 'rsi' in feature_name:
        return 'RSI指标'
    elif 'ma' in feature_name or 'ema' in feature_name:
        return '均线特征'
    elif 'bias' in feature_name:
        return '乖离率'
    elif 'kdj' in feature_name:
        return 'KDJ指标'
    elif 'atr' in feature_name or 'drawdown' in feature_name or 'recovery' in feature_name or 'days_from_high' in feature_name:
        return '风险特征'
    elif 'breakout' in feature_name:
        return '突破特征'
    elif 'vol' in feature_name or 'volume' in feature_name or 'obv' in feature_name:
        return '量价特征'
    elif 'momentum' in feature_name or 'return' in feature_name:
        return '动量特征'
    elif '233' in feature_name:
        return '233日均线'
    else:
        return '其他'


def analyze_feature_importance(df_imp, version):
    """分析特征重要性"""
    log.info(f"\n{'='*80}")
    log.info(f"{version} 特征重要性分析")
    log.info(f"{'='*80}")
    
    # 添加分类
    df_imp['category'] = df_imp['feature'].apply(categorize_feature)
    
    # Top 20特征
    log.info(f"\nTop 20 特征:")
    log.info(f"{'排名':<6} {'特征':<30} {'重要性':<12} {'占比':<10} {'累计占比':<10} {'类别':<15}")
    log.info("-" * 90)
    
    top20 = df_imp.head(20)
    for idx, row in top20.iterrows():
        log.info(f"{row.name+1:<6} {row['feature']:<30} {row['importance']:<12.2f} {row['percentage']:<10.2f}% {row['cumulative']:<10.2f}% {row['category']:<15}")
    
    # 按类别统计
    log.info(f"\n按类别统计:")
    category_stats = df_imp.groupby('category').agg({
        'importance': 'sum',
        'percentage': 'sum',
        'feature': 'count'
    }).sort_values('percentage', ascending=False)
    category_stats.columns = ['总重要性', '总占比%', '特征数']
    
    log.info(f"\n{'类别':<20} {'特征数':<10} {'总重要性':<15} {'总占比%':<10}")
    log.info("-" * 60)
    for cat, row in category_stats.iterrows():
        log.info(f"{cat:<20} {row['特征数']:<10} {row['总重要性']:<15.2f} {row['总占比%']:<10.2f}%")
    
    # 使用的特征数（重要性>0）
    used_features = len(df_imp[df_imp['importance'] > 0])
    log.info(f"\n使用的特征数: {used_features} / {len(df_imp)} ({used_features/len(df_imp)*100:.1f}%)")
    
    # Top 7特征占比
    top7_pct = df_imp.head(7)['percentage'].sum()
    top1_pct = df_imp.iloc[0]['percentage'] if len(df_imp) > 0 else 0
    log.info(f"Top 1特征占比: {top1_pct:.2f}%")
    log.info(f"Top 7特征占比: {top7_pct:.2f}%")
    
    # 核心业务特征占比
    breakout_pct = df_imp[df_imp['category'] == '突破特征']['percentage'].sum()
    volume_pct = df_imp[df_imp['category'] == '量价特征']['percentage'].sum()
    momentum_pct = df_imp[df_imp['category'] == '动量特征']['percentage'].sum()
    market_pct = df_imp[df_imp['category'] == '市场环境']['percentage'].sum()
    log.info(f"\n核心业务特征占比:")
    log.info(f"  突破特征: {breakout_pct:.2f}%")
    log.info(f"  量价特征: {volume_pct:.2f}%")
    log.info(f"  动量特征: {momentum_pct:.2f}%")
    log.info(f"  市场环境: {market_pct:.2f}%")
    
    return df_imp, category_stats


def compare_models(df_v252, df_v253):
    """对比两个模型"""
    log.info(f"\n{'='*80}")
    log.info("v2.5.2 vs v2.5.3 对比分析")
    log.info(f"{'='*80}")
    
    # 合并数据
    df_merge = pd.merge(
        df_v252[['feature', 'importance', 'percentage', 'category']].rename(columns={
            'importance': 'importance_v252',
            'percentage': 'percentage_v252',
            'category': 'category_v252'
        }),
        df_v253[['feature', 'importance', 'percentage', 'category']].rename(columns={
            'importance': 'importance_v253',
            'percentage': 'percentage_v253',
            'category': 'category_v253'
        }),
        on='feature',
        how='outer'
    ).fillna(0)
    
    # 统一category（优先使用v2.5.3的分类）
    df_merge['category'] = df_merge['category_v253'].where(
        df_merge['category_v253'] != 0, 
        df_merge['category_v252']
    )
    
    # 计算变化
    df_merge['importance_change'] = df_merge['importance_v253'] - df_merge['importance_v252']
    df_merge['percentage_change'] = df_merge['percentage_v253'] - df_merge['percentage_v252']
    
    # Top 20变化最大的特征
    log.info(f"\n重要性变化最大的Top 20特征:")
    log.info(f"{'特征':<30} {'v2.5.2占比':<12} {'v2.5.3占比':<12} {'变化':<10} {'类别':<15}")
    log.info("-" * 85)
    
    top_changes = df_merge.nlargest(20, 'percentage_change')[['feature', 'percentage_v252', 'percentage_v253', 'percentage_change', 'category']].copy()
    for idx, row in top_changes.iterrows():
        log.info(f"{row['feature']:<30} {row['percentage_v252']:<12.2f}% {row['percentage_v253']:<12.2f}% {row['percentage_change']:<10.2f}% {row['category']:<15}")
    
    # 按类别对比
    log.info(f"\n按类别对比（占比变化）:")
    log.info(f"{'类别':<20} {'v2.5.2占比':<12} {'v2.5.3占比':<12} {'变化':<10}")
    log.info("-" * 60)
    
    cat_v252 = df_v252.groupby('category')['percentage'].sum().sort_values(ascending=False)
    cat_v253 = df_v253.groupby('category')['percentage'].sum().sort_values(ascending=False)
    
    all_cats = set(cat_v252.index) | set(cat_v253.index)
    for cat in sorted(all_cats, key=lambda x: cat_v253.get(x, 0), reverse=True):
        pct_v252 = cat_v252.get(cat, 0)
        pct_v253 = cat_v253.get(cat, 0)
        change = pct_v253 - pct_v252
        log.info(f"{cat:<20} {pct_v252:<12.2f}% {pct_v253:<12.2f}% {change:<10.2f}%")
    
    # 检查市场环境特征
    market_features = df_v253[df_v253['category'] == '市场环境']
    if len(market_features) > 0:
        log.info(f"\n市场环境特征重要性:")
        log.info(f"{'特征':<30} {'重要性':<12} {'占比':<10}")
        log.info("-" * 55)
        for idx, row in market_features.iterrows():
            log.info(f"{row['feature']:<30} {row['importance']:<12.2f} {row['percentage']:<10.2f}%")
        total_market_pct = market_features['percentage'].sum()
        log.info(f"\n市场环境特征总占比: {total_market_pct:.2f}%")
    else:
        log.warning("⚠️  v2.5.3中未发现市场环境特征")
    
    return df_merge


def main():
    log.info("="*80)
    log.info("评估v2.5.3模型特征重要性分布")
    log.info("="*80)
    
    # 加载模型
    log.info("\n加载v2.5.2模型...")
    booster_v252, features_v252, metadata_v252 = load_model('v2.5.2')
    
    log.info("\n加载v2.5.3模型...")
    booster_v253, features_v253, metadata_v253 = load_model('v2.5.3')
    
    # 获取特征重要性
    log.info("\n提取特征重要性...")
    df_imp_v252 = get_feature_importance(booster_v252, features_v252)
    df_imp_v253 = get_feature_importance(booster_v253, features_v253)
    
    # 分析
    df_imp_v252, cat_stats_v252 = analyze_feature_importance(df_imp_v252, 'v2.5.2')
    df_imp_v253, cat_stats_v253 = analyze_feature_importance(df_imp_v253, 'v2.5.3')
    
    # 对比
    df_compare = compare_models(df_imp_v252, df_imp_v253)
    
    # 性能对比
    log.info(f"\n{'='*80}")
    log.info("模型性能对比")
    log.info(f"{'='*80}")
    log.info(f"{'指标':<20} {'v2.5.2':<15} {'v2.5.3':<15} {'变化':<10}")
    log.info("-" * 60)
    log.info(f"{'AUC':<20} {metadata_v252['metrics']['auc']:<15.4f} {metadata_v253['metrics']['auc']:<15.4f} {metadata_v253['metrics']['auc'] - metadata_v252['metrics']['auc']:<+10.4f}")
    log.info(f"{'Precision':<20} {metadata_v252['metrics']['precision']:<15.4f} {metadata_v253['metrics']['precision']:<15.4f} {metadata_v253['metrics']['precision'] - metadata_v252['metrics']['precision']:<+10.4f}")
    log.info(f"{'Recall':<20} {metadata_v252['metrics']['recall']:<15.4f} {metadata_v253['metrics']['recall']:<15.4f} {metadata_v253['metrics']['recall'] - metadata_v252['metrics']['recall']:<+10.4f}")
    log.info(f"{'F1':<20} {metadata_v252['metrics']['f1']:<15.4f} {metadata_v253['metrics']['f1']:<15.4f} {metadata_v253['metrics']['f1'] - metadata_v252['metrics']['f1']:<+10.4f}")
    
    # 保存结果
    output_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / 'v2.5.3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_imp_v253.to_csv(output_dir / 'feature_importance_v253.csv', index=False)
    df_compare.to_csv(output_dir / 'feature_importance_comparison.csv', index=False)
    
    log.success("\n✓ 评估完成！")
    log.info(f"结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
