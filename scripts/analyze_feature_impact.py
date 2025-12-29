#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析特征对预测概率的影响，找出导致差异的关键特征
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
from scripts.detailed_feature_comparison import extract_features_for_stock


def analyze_feature_impact():
    """分析特征对预测概率的影响"""
    log.info("="*80)
    log.info("特征影响分析")
    log.info("="*80)
    
    # 加载模型和特征重要性
    new_model_path = 'data/models/breakout_launch_scorer/versions/v1.2.0/model/model.json'
    old_model_path = 'data/training/models/xgboost_timeseries_v2_20251225_205905.json'
    
    new_booster = xgb.Booster()
    new_booster.load_model(new_model_path)
    
    old_booster = xgb.Booster()
    old_booster.load_model(old_model_path)
    
    feature_names = new_booster.feature_names
    
    # 加载特征重要性（从旧模型的metrics文件）
    metrics_file = 'data/training/metrics/xgboost_timeseries_v2_metrics.json'
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    feature_importance = {item['feature']: item['importance'] for item in metrics['feature_importance']}
    
    log.info(f"\n特征重要性（Top 10）:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:10]:
        log.info(f"  {feat}: {imp:.4f}")
    
    # 加载详细特征对比结果
    detail_file = 'data/prediction/comparison/detailed_feature_comparison_20251225.csv'
    if not os.path.exists(detail_file):
        log.error(f"详细特征对比文件不存在: {detail_file}")
        log.error("请先运行: python scripts/detailed_feature_comparison.py")
        return
    
    df = pd.read_csv(detail_file)
    
    log.info(f"\n分析股票数量: {len(df)}")
    log.info(f"平均概率差异: {df['diff_vs_csv'].mean():.6f}")
    
    # 分析每个特征对预测概率的影响
    log.info("\n" + "="*80)
    log.info("特征值差异分析（Top 10重要特征）")
    log.info("="*80)
    
    for feat, imp in sorted_features[:10]:
        feat_col = f'feat_{feat}'
        if feat_col in df.columns:
            values = df[feat_col].values
            log.info(f"\n{feat} (重要性: {imp:.4f}):")
            log.info(f"  值范围: {values.min():.4f} ~ {values.max():.4f}")
            log.info(f"  平均值: {values.mean():.4f}")
            log.info(f"  标准差: {values.std():.4f}")
            
            # 检查是否有异常值
            if values.std() > 0:
                cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                if cv > 1.0:
                    log.warning(f"  ⚠️  变异系数较大: {cv:.2f}，可能存在异常值")
    
    # 分析特征与概率差异的相关性
    log.info("\n" + "="*80)
    log.info("特征与概率差异的相关性分析")
    log.info("="*80)
    
    correlations = []
    for feat in feature_names:
        feat_col = f'feat_{feat}'
        if feat_col in df.columns:
            corr = df['diff_vs_csv'].corr(df[feat_col])
            if not pd.isna(corr):
                correlations.append({
                    'feature': feat,
                    'correlation': corr,
                    'importance': feature_importance.get(feat, 0)
                })
    
    df_corr = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    
    log.info(f"\n与概率差异相关性最高的特征（Top 10）:")
    for idx, row in df_corr.head(10).iterrows():
        log.info(f"  {row['feature']}: 相关性={row['correlation']:.4f}, 重要性={row['importance']:.4f}")
    
    # 保存相关性分析结果
    output_file = 'data/prediction/comparison/feature_correlation_analysis_20251225.csv'
    df_corr.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 特征相关性分析结果已保存: {output_file}")


if __name__ == '__main__':
    analyze_feature_impact()

