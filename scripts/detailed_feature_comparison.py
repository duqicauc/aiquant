#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细对比新旧模型预测时使用的特征值差异
找出导致预测概率不同的关键特征
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

from src.data.data_manager import DataManager
from src.utils.logger import log
from scripts.score_current_stocks import load_model, _calculate_features_from_df


def extract_features_for_stock(ts_code, target_date='20251225'):
    """提取单只股票的特征值（使用当前预测逻辑）"""
    dm = DataManager()
    
    # 获取股票名称
    stock_list = dm.get_stock_list()
    stock_info = stock_list[stock_list['ts_code'] == ts_code]
    if stock_info.empty:
        return None
    name = stock_info.iloc[0]['name']
    
    # 获取日线数据
    target_dt = pd.to_datetime(target_date, format='%Y%m%d')
    end_date = target_dt.strftime('%Y%m%d')
    start_date = (target_dt - pd.Timedelta(days=68)).strftime('%Y%m%d')
    
    df = dm.get_daily_data(
        stock_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    
    if df is None or len(df) < 20:
        return None
    
    # 确保trade_date是datetime类型
    if 'trade_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # 取最近的34天
    df = df.tail(34).sort_values('trade_date')
    if len(df) < 20:
        return None
    
    # 尝试获取Tushare技术因子（与训练时一致）
    try:
        end_date_str = df['trade_date'].max()
        start_date_str = df['trade_date'].min()
        if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            end_date_str = end_date_str.strftime('%Y%m%d')
            start_date_str = start_date_str.strftime('%Y%m%d')
        else:
            end_date_str = str(end_date_str).replace('-', '')
            start_date_str = str(start_date_str).replace('-', '')
        
        df_factor = dm.get_stk_factor(ts_code, start_date_str, end_date_str)
        if not df_factor.empty:
            if 'trade_date' in df_factor.columns:
                if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
                else:
                    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
            
            df = pd.merge(
                df,
                df_factor[['trade_date', 'macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']],
                on='trade_date',
                how='left'
            )
    except Exception as e:
        pass
    
    # 合并daily_basic数据
    try:
        basic_data = dm.get_daily_basic(ts_code, end_date, end_date)
        if not basic_data.empty:
            basic_row = basic_data.iloc[0]
            if 'total_mv' not in df.columns and 'total_mv' in basic_row:
                df['total_mv'] = basic_row['total_mv']
            if 'circ_mv' not in df.columns and 'circ_mv' in basic_row:
                df['circ_mv'] = basic_row['circ_mv']
            if 'volume_ratio' in basic_row and pd.notna(basic_row['volume_ratio']):
                if 'volume_ratio' not in df.columns:
                    df['volume_ratio'] = basic_row['volume_ratio']
                else:
                    df['volume_ratio'] = df['volume_ratio'].fillna(basic_row['volume_ratio'])
    except Exception as e:
        pass
    
    # 计算特征
    features = _calculate_features_from_df(df, ts_code, name, debug_log=None)
    
    return features


def analyze_feature_differences():
    """详细分析特征差异"""
    log.info("="*80)
    log.info("详细特征差异分析")
    log.info("="*80)
    
    # 加载新旧模型的预测结果
    new_file = 'data/prediction/results/top_50_stocks_20251225_breakout_launch_scorer_v1.2.0.csv'
    old_file = 'data/prediction/results/top_50_stocks_20251225_232545.csv'
    
    df_new = pd.read_csv(new_file, encoding='utf-8-sig')
    df_old = pd.read_csv(old_file, encoding='utf-8-sig')
    
    # 找出差异最大的股票
    common_stocks = set(df_new['股票代码']) & set(df_old['股票代码'])
    
    comparison = []
    for stock in common_stocks:
        new_row = df_new[df_new['股票代码'] == stock].iloc[0]
        old_row = df_old[df_old['股票代码'] == stock].iloc[0]
        prob_diff = abs(new_row['牛股概率'] - old_row['牛股概率'])
        comparison.append({
            'ts_code': stock,
            'name': new_row['股票名称'],
            'new_prob': new_row['牛股概率'],
            'old_prob': old_row['牛股概率'],
            'prob_diff': prob_diff,
        })
    
    df_comp = pd.DataFrame(comparison).sort_values('prob_diff', ascending=False)
    
    # 选择差异最大的5只股票进行详细分析
    sample_stocks = df_comp.head(5)['ts_code'].tolist()
    
    log.info(f"\n选择以下股票进行详细特征分析（概率差异最大）:")
    for idx, row in df_comp.head(5).iterrows():
        log.info(f"  {row['ts_code']} {row['name']}: 新={row['new_prob']:.6f}, 旧={row['old_prob']:.6f}, 差异={row['prob_diff']:.6f}")
    
    # 加载模型
    new_model = load_model(version='v1.2.0')
    old_model_path = 'data/training/models/xgboost_timeseries_v2_20251225_205905.json'
    old_booster = xgb.Booster()
    old_booster.load_model(old_model_path)
    
    feature_names = new_model.feature_names
    log.info(f"\n模型特征数量: {len(feature_names)}")
    
    # 分析每只股票的特征
    results = []
    for ts_code in sample_stocks:
        log.info(f"\n{'='*80}")
        log.info(f"分析 {ts_code}...")
        log.info(f"{'='*80}")
        
        # 提取特征
        features = extract_features_for_stock(ts_code, '20251225')
        if features is None:
            log.warning(f"  {ts_code}: 无法提取特征")
            continue
        
        # 构建特征向量
        feature_vector = []
        missing_features = []
        for feat_name in feature_names:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            else:
                feature_vector.append(0)
                missing_features.append(feat_name)
        
        if missing_features:
            log.warning(f"  缺失特征: {missing_features}")
        
        # 使用新旧模型分别预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        new_prob = new_model.predict(dmatrix)[0]
        old_prob = old_booster.predict(dmatrix)[0]
        
        # 获取旧模型预测结果（从CSV）
        old_row = df_old[df_old['股票代码'] == ts_code]
        old_csv_prob = old_row.iloc[0]['牛股概率'] if not old_row.empty else None
        
        log.info(f"\n预测结果:")
        log.info(f"  新模型（v1.2.0）: {new_prob:.6f}")
        log.info(f"  旧模型（相同特征）: {old_prob:.6f}")
        old_csv_str = f"{old_csv_prob:.6f}" if old_csv_prob is not None else "N/A"
        diff_str = f"{new_prob - old_csv_prob:.6f}" if old_csv_prob is not None else "N/A"
        log.info(f"  旧模型（CSV记录）: {old_csv_str}")
        log.info(f"  差异（新-旧CSV）: {diff_str}")
        
        # 如果使用相同特征值，新旧模型预测一致，但CSV记录不同
        # 说明旧模型预测时使用的特征值不同
        if abs(new_prob - old_prob) < 0.0001 and old_csv_prob and abs(new_prob - old_csv_prob) > 0.001:
            log.warning(f"  ⚠️  使用相同特征值时，新旧模型预测一致，但与CSV记录不同")
            log.warning(f"     说明：旧模型预测时使用的特征值与我们当前提取的不同")
        
        # 保存特征值
        result = {
            'ts_code': ts_code,
            'name': features.get('name', ''),
            'new_prob': new_prob,
            'old_prob_model': old_prob,
            'old_prob_csv': old_csv_prob,
            'diff_vs_model': new_prob - old_prob,
            'diff_vs_csv': new_prob - old_csv_prob if old_csv_prob else None,
        }
        
        # 添加所有特征值
        for feat_name in feature_names:
            result[f'feat_{feat_name}'] = features.get(feat_name, 0)
        
        results.append(result)
        
        # 显示关键特征值
        log.info(f"\n关键特征值:")
        key_features = ['close_trend', 'pct_chg_sum', 'return_1w', 'return_2w', 
                       'positive_days', 'negative_days', 'volume_ratio_mean', 
                       'macd_mean', 'total_mv_mean', 'circ_mv_mean']
        for feat in key_features:
            if feat in features:
                log.info(f"  {feat}: {features[feat]:.4f}")
    
    # 保存详细结果
    df_results = pd.DataFrame(results)
    output_file = 'data/prediction/comparison/detailed_feature_comparison_20251225.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 详细特征对比结果已保存: {output_file}")
    
    # 分析特征重要性
    log.info("\n" + "="*80)
    log.info("特征差异分析总结")
    log.info("="*80)
    
    if len(results) > 0:
        log.info(f"\n分析股票数量: {len(results)}")
        log.info(f"平均概率差异（vs旧模型CSV）: {df_results['diff_vs_csv'].mean():.6f}")
        log.info(f"最大概率差异: {df_results['diff_vs_csv'].abs().max():.6f}")
        
        # 如果使用相同特征值时新旧模型预测一致，说明问题在于特征值不同
        avg_diff_vs_model = df_results['diff_vs_model'].abs().mean()
        if avg_diff_vs_model < 0.0001:
            log.success("\n✓ 使用相同特征值时，新旧模型预测完全一致")
            log.warning("⚠️  但预测结果与旧模型CSV记录不同，说明：")
            log.warning("   旧模型预测时使用的特征值与我们当前提取的特征值不同")
            log.warning("   可能原因：")
            log.warning("   1. 数据获取时间不同（数据有更新）")
            log.warning("   2. 特征计算逻辑在旧模型预测时不同")
            log.warning("   3. Tushare技术因子获取失败，使用了不同的计算方式")


if __name__ == '__main__':
    analyze_feature_differences()

