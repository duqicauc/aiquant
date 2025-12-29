#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用迁移前的旧xgboost模型进行预测
直接使用旧路径：data/training/models/xgboost_timeseries_v2_20251225_205905.json
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log
from scripts.score_current_stocks import get_all_stocks, _calculate_features_from_df


def load_old_model():
    """加载旧模型（直接从旧路径）"""
    old_model_path = 'data/training/models/xgboost_timeseries_v2_20251225_205905.json'
    
    if not os.path.exists(old_model_path):
        raise FileNotFoundError(f"旧模型文件不存在: {old_model_path}")
    
    log.info(f"加载旧模型: {old_model_path}")
    
    # 加载XGBoost Booster
    booster = xgb.Booster()
    booster.load_model(str(old_model_path))
    
    # 从metrics文件获取特征名称
    metrics_file = 'data/training/metrics/xgboost_timeseries_v2_metrics.json'
    feature_names = None
    
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        if 'feature_importance' in metrics:
            feature_names = [item['feature'] for item in metrics['feature_importance']]
            log.info(f"✓ 从metrics文件加载特征名称: {len(feature_names)} 个特征")
    
    # 如果无法从metrics获取，尝试从模型获取
    if feature_names is None:
        if hasattr(booster, 'feature_names'):
            feature_names = booster.feature_names
        elif hasattr(booster, 'feature_names_'):
            feature_names = booster.feature_names_
    
    if feature_names is None:
        log.warning("无法获取特征名称，将使用默认顺序")
        # 使用已知的特征顺序（从旧模型训练脚本中获取）
        feature_names = [
            'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
            'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum',
            'positive_days', 'negative_days', 'max_gain', 'max_loss',
            'volume_ratio_mean', 'volume_ratio_max', 'volume_ratio_gt_2', 'volume_ratio_gt_4',
            'macd_mean', 'macd_positive_days', 'macd_max',
            'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10',
            'total_mv_mean', 'circ_mv_mean',
            'return_1w', 'return_2w'
        ]
        log.info(f"使用默认特征顺序: {len(feature_names)} 个特征")
    
    class ModelWrapper:
        def __init__(self, booster, feature_names, model_path):
            self.booster = booster
            self.feature_names = feature_names
            self.model_path = model_path
            self.model_name = 'xgboost_timeseries'
            self.model_version = 'v2_20251225_205905'
        
        def predict(self, dmatrix):
            """预测概率"""
            return self.booster.predict(dmatrix, output_margin=False, validate_features=False)
    
    log.success("✓ 旧模型加载成功")
    return ModelWrapper(booster, feature_names, old_model_path)


def main():
    """主函数"""
    log.info("="*80)
    log.info("使用旧xgboost模型预测20251225")
    log.info("="*80)
    log.info("")
    log.info("模型路径: data/training/models/xgboost_timeseries_v2_20251225_205905.json")
    log.info("")
    
    # 1. 加载旧模型
    log.info("="*80)
    log.info("第一步：加载旧模型")
    log.info("="*80)
    try:
        model = load_old_model()
        log.info(f"  模型名称: {model.model_name}")
        log.info(f"  模型版本: {model.model_version}")
        log.info(f"  特征数量: {len(model.feature_names)}")
        log.info("")
    except Exception as e:
        log.error(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 初始化数据管理器
    log.info("="*80)
    log.info("第二步：初始化数据管理器")
    log.info("="*80)
    dm = DataManager()
    log.success("✓ 数据管理器初始化完成")
    log.info("")
    
    # 3. 获取所有符合条件的股票
    log.info("="*80)
    log.info("第三步：获取股票列表")
    log.info("="*80)
    target_date = datetime(2025, 12, 25)
    valid_stocks = get_all_stocks(dm, target_date=target_date.strftime('%Y%m%d'))
    log.success(f"✓ 获取到 {len(valid_stocks)} 只符合条件的股票")
    log.info("")
    
    # 4. 对所有股票进行评分
    log.info("="*80)
    log.info("第四步：对所有股票进行评分")
    log.info("="*80)
    log.info("注意：这可能需要较长时间（约10-30分钟）")
    log.info("")
    
    from scripts.score_current_stocks import score_all_stocks
    
    df_scores = score_all_stocks(
        dm, 
        model, 
        valid_stocks, 
        max_stocks=len(valid_stocks),
        target_date=target_date.strftime('%Y%m%d')
    )
    
    if df_scores is None or len(df_scores) == 0:
        log.error("✗ 评分结果为空")
        return
    
    log.success(f"✓ 成功评分 {len(df_scores)} 只股票")
    log.info("")
    
    # 5. 生成推荐结果
    log.info("="*80)
    log.info("第五步：生成推荐结果")
    log.info("="*80)
    
    from scripts.score_current_stocks import analyze_and_output_results, save_results
    
    TOP_N = 50
    df_top = analyze_and_output_results(df_scores, top_n=min(TOP_N, len(df_scores)))
    
    # 6. 保存结果
    log.info("")
    log.info("="*80)
    log.info("第六步：保存结果")
    log.info("="*80)
    
    # 使用旧模型的标识
    model_name = 'xgboost_timeseries'
    model_version = 'v2_20251225_205905'
    
    save_results(
        df_scores,
        df_top,
        top_n=TOP_N,  # 明确指定top_n参数
        model_name=model_name,
        model_version=model_version,
        target_date=target_date.strftime('%Y%m%d')
    )
    
    log.info("")
    log.info("="*80)
    log.success("✅ 预测完成！")
    log.info("="*80)
    log.info("")
    log.info("结果文件:")
    log.info(f"  全量评分: data/prediction/results/stock_scores_20251225_{model_name}_{model_version}.csv")
    log.info(f"  Top 50: data/prediction/results/top_50_stocks_20251225_{model_name}_{model_version}.csv")
    log.info(f"  报告: data/prediction/results/prediction_report_20251225_{model_name}_{model_version}.txt")
    log.info("")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.warning("预测被用户中断")
    except Exception as e:
        log.error(f"预测过程出错: {e}")
        import traceback
        traceback.print_exc()

