#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析新旧模型预测结果的中间特征差异
找出具体哪个特征导致预测概率不同
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log
from scripts.score_current_stocks import load_model, _calculate_features_from_df


def extract_features_for_stock(ts_code, target_date='20251225'):
    """提取单只股票的特征值"""
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
    """分析特征差异"""
    log.info("="*80)
    log.info("特征差异分析")
    log.info("="*80)
    
    # 加载新旧模型的预测结果
    new_file = 'data/prediction/results/top_50_stocks_20251225_breakout_launch_scorer_v1.2.0.csv'
    old_file = 'data/prediction/results/top_50_stocks_20251225_232545.csv'
    
    df_new = pd.read_csv(new_file, encoding='utf-8-sig')
    df_old = pd.read_csv(old_file, encoding='utf-8-sig')
    
    # 找出共同股票
    common_stocks = set(df_new['股票代码']) & set(df_old['股票代码'])
    log.info(f"\n共同股票: {len(common_stocks)} 只")
    
    # 找出差异最大的股票（概率差异大）
    comparison = []
    for stock in common_stocks:
        new_row = df_new[df_new['股票代码'] == stock].iloc[0]
        old_row = df_old[df_old['股票代码'] == stock].iloc[0]
        prob_diff = new_row['牛股概率'] - old_row['牛股概率']
        comparison.append({
            'ts_code': stock,
            'name': new_row['股票名称'],
            'new_prob': new_row['牛股概率'],
            'old_prob': old_row['牛股概率'],
            'prob_diff': prob_diff,
            'new_rank': df_new[df_new['股票代码'] == stock].index[0] + 1,
            'old_rank': df_old[df_old['股票代码'] == stock].index[0] + 1,
        })
    
    df_comp = pd.DataFrame(comparison).sort_values('prob_diff', key=abs, ascending=False)
    
    log.info(f"\n概率差异最大的10只股票:")
    log.info(df_comp.head(10).to_string(index=False))
    
    # 选择几只股票进行详细特征分析
    sample_stocks = df_comp.head(5)['ts_code'].tolist()
    sample_stocks.extend(df_comp.tail(2)['ts_code'].tolist())  # 也分析差异小的
    
    log.info(f"\n详细分析以下股票的特征: {sample_stocks}")
    
    # 加载模型以获取特征名称
    model = load_model(version='v1.2.0')
    feature_names = model.feature_names if hasattr(model, 'feature_names') and model.feature_names else None
    
    if feature_names is None:
        log.warning("无法获取模型特征名称，使用默认特征列表")
        feature_names = [
            'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
            'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum', 
            'positive_days', 'negative_days', 'max_gain', 'max_loss',
            'volume_ratio_mean', 'volume_ratio_max', 'volume_ratio_gt_2', 'volume_ratio_gt_4',
            'macd_mean', 'macd_positive_days', 'macd_max',
            'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10',
            'total_mv_mean', 'circ_mv_mean', 'return_1w', 'return_2w'
        ]
    
    log.info(f"模型特征数量: {len(feature_names)}")
    
    # 提取每只股票的特征
    results = []
    for ts_code in sample_stocks:
        log.info(f"\n分析 {ts_code}...")
        features = extract_features_for_stock(ts_code, '20251225')
        
        if features is None:
            log.warning(f"  {ts_code}: 无法提取特征")
            continue
        
        # 构建特征向量（按照模型要求的顺序）
        feature_vector = []
        missing_features = []
        for feat_name in feature_names:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            else:
                feature_vector.append(0)
                missing_features.append(feat_name)
        
        if missing_features:
            log.warning(f"  {ts_code}: 缺失特征 {missing_features}")
        
        # 使用模型预测
        import xgboost as xgb
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        prob = model.predict(dmatrix)[0]
        
        # 获取旧模型的预测结果
        old_row = df_old[df_old['股票代码'] == ts_code]
        old_prob = old_row.iloc[0]['牛股概率'] if not old_row.empty else None
        
        result = {
            'ts_code': ts_code,
            'name': features.get('name', ''),
            'new_prob': prob,
            'old_prob': old_prob,
            'prob_diff': prob - old_prob if old_prob else None,
        }
        
        # 添加所有特征值
        for feat_name in feature_names:
            result[f'feat_{feat_name}'] = features.get(feat_name, 0)
        
        results.append(result)
        
        old_prob_str = f"{old_prob:.6f}" if old_prob is not None else "N/A"
        prob_diff_str = f"{prob-old_prob:.6f}" if old_prob is not None else "N/A"
        log.info(f"  {ts_code}: 新模型概率={prob:.6f}, 旧模型概率={old_prob_str}, 差异={prob_diff_str}")
    
    # 保存详细结果
    df_results = pd.DataFrame(results)
    output_file = 'data/prediction/comparison/feature_analysis_20251225.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"\n✓ 详细特征分析结果已保存: {output_file}")
    
    # 分析特征差异
    if len(results) > 1:
        log.info("\n" + "="*80)
        log.info("特征值统计（前5只差异最大的股票）")
        log.info("="*80)
        
        df_analysis = df_results.head(5)
        for feat_name in feature_names:
            feat_col = f'feat_{feat_name}'
            if feat_col in df_analysis.columns:
                values = df_analysis[feat_col].values
                log.info(f"\n{feat_name}:")
                log.info(f"  值范围: {values.min():.4f} ~ {values.max():.4f}")
                log.info(f"  平均值: {values.mean():.4f}")
                log.info(f"  标准差: {values.std():.4f}")
                if len(values) > 1:
                    log.info(f"  变异系数: {values.std()/abs(values.mean()) if values.mean() != 0 else 'N/A'}")


if __name__ == '__main__':
    analyze_feature_differences()

