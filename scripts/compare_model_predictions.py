#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比新旧模型对相同特征值的预测结果
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
from scripts.analyze_feature_differences import extract_features_for_stock


def compare_models():
    """对比新旧模型对相同特征值的预测"""
    log.info("="*80)
    log.info("新旧模型预测对比（使用相同特征值）")
    log.info("="*80)
    
    # 加载新旧模型
    new_model_path = 'data/models/breakout_launch_scorer/versions/v1.2.0/model/model.json'
    old_model_path = 'data/training/models/xgboost_timeseries_v2_20251225_205905.json'
    
    new_booster = xgb.Booster()
    new_booster.load_model(new_model_path)
    
    old_booster = xgb.Booster()
    old_booster.load_model(old_model_path)
    
    feature_names = new_booster.feature_names
    log.info(f"特征数量: {len(feature_names)}")
    
    # 选择几只股票进行测试
    test_stocks = ['300668.SZ', '002163.SZ', '688323.SH', '600990.SH', '002935.SZ']
    
    results = []
    for ts_code in test_stocks:
        log.info(f"\n测试 {ts_code}...")
        
        # 提取特征
        features = extract_features_for_stock(ts_code, '20251225')
        if features is None:
            log.warning(f"  {ts_code}: 无法提取特征")
            continue
        
        # 构建特征向量
        feature_vector = []
        for feat_name in feature_names:
            feature_vector.append(features.get(feat_name, 0))
        
        # 使用新旧模型分别预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        new_prob = new_booster.predict(dmatrix)[0]
        old_prob = old_booster.predict(dmatrix)[0]
        
        diff = new_prob - old_prob
        
        results.append({
            'ts_code': ts_code,
            'name': features.get('name', ''),
            'new_prob': new_prob,
            'old_prob': old_prob,
            'diff': diff,
            'abs_diff': abs(diff)
        })
        
        log.info(f"  新模型概率: {new_prob:.6f}")
        log.info(f"  旧模型概率: {old_prob:.6f}")
        log.info(f"  差异: {diff:.6f}")
    
    # 总结
    df_results = pd.DataFrame(results)
    log.info("\n" + "="*80)
    log.info("对比结果总结")
    log.info("="*80)
    log.info(f"\n平均概率差异: {df_results['diff'].mean():.6f}")
    log.info(f"最大概率差异: {df_results['abs_diff'].max():.6f}")
    log.info(f"最小概率差异: {df_results['abs_diff'].min():.6f}")
    
    log.info("\n详细结果:")
    log.info(df_results.to_string(index=False))
    
    # 如果差异很大，说明模型本身不同
    if df_results['abs_diff'].max() > 0.01:
        log.warning("\n⚠️  模型预测差异较大，可能是模型本身不同（训练数据或训练过程不同）")
    else:
        log.success("\n✓ 模型预测基本一致，差异在可接受范围内")


if __name__ == '__main__':
    compare_models()

