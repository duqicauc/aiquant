#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比v2.3.0和v2.4.0模型的预测效果

功能：
1. 使用v2.4.0模型预测2025年12月12日的股票
2. 计算到2026年1月5日的实际收益
3. 与v2.3.0模型（或v2.3.1）的预测结果对比
4. 按Top10股票质量进行评价
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_model(version):
    """加载指定版本的模型"""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version / 'model'
    
    if not model_dir.exists():
        log.error(f"模型目录不存在: {model_dir}")
        return None, None, None
    
    # 加载模型
    model_file = model_dir / 'model.json'
    booster = xgb.Booster()
    booster.load_model(str(model_file))
    log.success(f"✓ {version} 模型已加载")
    
    # 加载特征名称
    feature_names_file = model_dir / 'feature_names.json'
    with open(feature_names_file, 'r') as f:
        feature_names = json.load(f)
    
    # 加载校准器（如果存在）
    calibrator_file = model_dir / 'calibrator.pkl'
    calibrator = None
    if calibrator_file.exists():
        calibrator = joblib.load(str(calibrator_file))
        log.info(f"  校准器: 已加载")
    
    return booster, feature_names, calibrator


def get_valid_stocks(dm, target_date):
    """获取有效股票列表"""
    stock_list = dm.get_stock_list()
    
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y%m%d')
    
    valid_stocks = []
    for _, stock in stock_list.iterrows():
        name = stock['name']
        ts_code = stock['ts_code']
        
        # 排除规则
        if 'ST' in name or '*' in name:
            continue
        if ts_code.endswith('.BJ'):
            continue
        if '退' in name:
            continue
        
        # 检查上市天数
        list_date = stock.get('list_date', '')
        if list_date:
            try:
                days_since_list = (target_date - pd.to_datetime(list_date)).days
                if days_since_list < 180:
                    continue
            except:
                pass
        
        valid_stocks.append(stock)
    
    return pd.DataFrame(valid_stocks)


def extract_features(dm, ts_code, lookback_days=34, target_date=None):
    """提取单只股票的特征（与训练时保持一致）"""
    try:
        if target_date:
            if isinstance(target_date, str):
                t1 = datetime.strptime(target_date, '%Y%m%d')
            else:
                t1 = target_date
        else:
            t1 = datetime.now()
        
        end_date = t1.strftime('%Y%m%d')
        start_date = (t1 - timedelta(days=lookback_days * 2)).strftime('%Y%m%d')
        
        df = dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
        
        if df is None or len(df) < 20:
            return None
        
        df = df.tail(lookback_days).sort_values('trade_date')
        
        if len(df) < 20:
            return None
        
        # 计算技术指标
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        if 'volume_ratio' not in df.columns:
            df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['vol'] / df['vol_ma5']
        if 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = (ema12 - ema26) * 2
        
        features = {}
        features['latest_close'] = df['close'].iloc[-1]
        
        # 原有特征
        features['close_mean'] = df['close'].mean()
        features['close_std'] = df['close'].std()
        features['close_max'] = df['close'].max()
        features['close_min'] = df['close'].min()
        features['close_trend'] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        if 'pct_chg' in df.columns:
            pct_chg = df['pct_chg'].dropna()
            if len(pct_chg) > 0:
                features['pct_chg_mean'] = pct_chg.mean()
                features['pct_chg_std'] = pct_chg.std()
                features['pct_chg_sum'] = pct_chg.sum()
                features['positive_days'] = (pct_chg > 0).sum()
                features['negative_days'] = (pct_chg < 0).sum()
                features['max_gain'] = pct_chg.max()
                features['max_loss'] = pct_chg.min()
        
        if 'volume_ratio' in df.columns:
            features['volume_ratio_mean'] = df['volume_ratio'].mean()
            features['volume_ratio_max'] = df['volume_ratio'].max()
            features['volume_ratio_gt_2'] = (df['volume_ratio'] > 2).sum()
            features['volume_ratio_gt_4'] = (df['volume_ratio'] > 4).sum()
        
        if 'macd' in df.columns:
            macd_data = df['macd'].dropna()
            if len(macd_data) > 0:
                features['macd_mean'] = macd_data.mean()
                features['macd_positive_days'] = (macd_data > 0).sum()
                features['macd_max'] = macd_data.max()
        
        if 'ma5' in df.columns:
            features['ma5_mean'] = df['ma5'].mean()
            features['price_above_ma5'] = (df['close'] > df['ma5']).sum()
        
        if 'ma10' in df.columns:
            features['ma10_mean'] = df['ma10'].mean()
            features['price_above_ma10'] = (df['close'] > df['ma10']).sum()
        
        if 'total_mv' in df.columns:
            mv_data = df['total_mv'].dropna()
            if len(mv_data) > 0:
                features['total_mv_mean'] = mv_data.mean()
        
        if 'circ_mv' in df.columns:
            circ_mv_data = df['circ_mv'].dropna()
            if len(circ_mv_data) > 0:
                features['circ_mv_mean'] = circ_mv_data.mean()
        
        days = len(df)
        if days >= 7:
            features['return_1w'] = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
        if days >= 14:
            features['return_2w'] = (df['close'].iloc[-1] - df['close'].iloc[-14]) / df['close'].iloc[-14] * 100
        
        # v2.4.0新增特征（v2.3.0可能没有，需要兼容处理）
        close_max = df['close'].max()
        close_min = df['close'].min()
        if close_min > 0:
            features['price_range_34d'] = (close_max - close_min) / close_min * 100
        
        if 'ma10' in df.columns:
            ma10_data = df['ma10'].dropna()
            close_data = df['close'].dropna()
            if len(ma10_data) > 0 and len(close_data) > 0:
                close_vs_ma10 = close_data / ma10_data.reindex(close_data.index)
                close_vs_ma10 = close_vs_ma10.dropna()
                if len(close_vs_ma10) > 0:
                    features['close_vs_ma10_std'] = close_vs_ma10.std()
                    features['days_near_ma10'] = ((close_vs_ma10 - 1).abs() < 0.03).sum()
        
        if 'vol' in df.columns and days >= 20:
            vol_data = df['vol'].dropna()
            if len(vol_data) >= 20:
                vol_first_half = vol_data.iloc[:len(vol_data)//2].mean()
                vol_last_half = vol_data.iloc[len(vol_data)//2:].mean()
                if vol_first_half > 0:
                    features['volume_shrink_ratio'] = vol_last_half / vol_first_half
        
        if close_max > 0 and close_min > 0:
            latest_close = df['close'].iloc[-1]
            features['price_vs_34d_high'] = latest_close / close_max
            features['price_vs_34d_low'] = latest_close / close_min
            if close_max != close_min:
                features['price_position_34d'] = (latest_close - close_min) / (close_max - close_min)
        
        if 'pct_chg' in df.columns:
            pct_chg_data = df['pct_chg'].dropna()
            if len(pct_chg_data) > 0:
                features['volatility_34d'] = pct_chg_data.abs().mean()
        
        return features
        
    except Exception as e:
        return None


def predict_stocks(booster, calibrator, feature_names, dm, stocks, target_date, top_n=50):
    """对股票进行预测"""
    log.info(f"\n预测股票 (共{len(stocks)}只)...")
    
    predictions = []
    processed = 0
    
    for idx, stock in stocks.iterrows():
        ts_code = stock['ts_code']
        name = stock['name']
        
        processed += 1
        if processed % 200 == 0:
            log.info(f"进度: {processed}/{len(stocks)} ({processed/len(stocks)*100:.1f}%)")
        
        features = extract_features(dm, ts_code, lookback_days=34, target_date=target_date)
        
        if features is None:
            continue
        
        # 构建特征向量（兼容不同版本的特征）
        feature_vector = []
        for name in feature_names:
            if name in features:
                feature_vector.append(features[name])
            else:
                feature_vector.append(0)  # 缺失特征用0填充
        
        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        raw_prob = booster.predict(dmatrix)[0]
        
        # 校准
        if calibrator is not None:
            cal_prob = calibrator.predict([raw_prob])[0]
        else:
            cal_prob = raw_prob
        
        predictions.append({
            'ts_code': ts_code,
            'name': name,
            'raw_probability': raw_prob,
            'calibrated_probability': cal_prob,
            'latest_close': features.get('latest_close', 0)
        })
    
    # 排序
    df_predictions = pd.DataFrame(predictions)
    df_predictions = df_predictions.sort_values('calibrated_probability', ascending=False)
    
    return df_predictions.head(top_n)


def calculate_actual_return(dm, ts_code, start_date, end_date):
    """计算从start_date到end_date的实际收益率"""
    try:
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, '%Y%m%d')
        else:
            start = start_date
        
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y%m%d')
        else:
            end = end_date
        
        # 获取起始价格（start_date当天或之前最近一天）
        start_df = dm.get_daily_data(ts_code, start.strftime('%Y%m%d'), start.strftime('%Y%m%d'), adjust='qfq')
        if start_df is None or len(start_df) == 0:
            # 如果当天没有数据，往前找
            for i in range(1, 10):
                check_date = (start - timedelta(days=i)).strftime('%Y%m%d')
                start_df = dm.get_daily_data(ts_code, check_date, check_date, adjust='qfq')
                if start_df is not None and len(start_df) > 0:
                    break
        
        if start_df is None or len(start_df) == 0:
            return None
        
        start_price = start_df['close'].iloc[-1]
        
        # 获取结束价格（end_date当天或之前最近一天）
        end_df = dm.get_daily_data(ts_code, end.strftime('%Y%m%d'), end.strftime('%Y%m%d'), adjust='qfq')
        if end_df is None or len(end_df) == 0:
            # 如果当天没有数据，往前找
            for i in range(1, 10):
                check_date = (end - timedelta(days=i)).strftime('%Y%m%d')
                end_df = dm.get_daily_data(ts_code, check_date, check_date, adjust='qfq')
                if end_df is not None and len(end_df) > 0:
                    break
        
        if end_df is None or len(end_df) == 0:
            return None
        
        end_price = end_df['close'].iloc[-1]
        
        if start_price <= 0:
            return None
        
        return (end_price - start_price) / start_price * 100
        
    except Exception as e:
        return None


def evaluate_predictions(df_predictions, dm, start_date, end_date, version_name):
    """评估预测结果的实际收益"""
    log.info(f"\n评估 {version_name} 预测结果...")
    
    results = []
    
    for idx, row in df_predictions.iterrows():
        ts_code = row['ts_code']
        actual_return = calculate_actual_return(dm, ts_code, start_date, end_date)
        
        if actual_return is not None:
            results.append({
                'ts_code': ts_code,
                'name': row['name'],
                'predicted_probability': row['calibrated_probability'],
                'actual_return': actual_return,
                'start_price': None,  # 可以后续补充
                'end_price': None
            })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        log.warning("无法计算任何股票的实际收益")
        return None
    
    return df_results


def compare_top10(df_v23, df_v24):
    """对比Top10股票的质量"""
    log.info("")
    log.info("="*80)
    log.info("Top10股票质量对比")
    log.info("="*80)
    
    top10_v23 = df_v23.head(10) if df_v23 is not None and len(df_v23) > 0 else pd.DataFrame()
    top10_v24 = df_v24.head(10) if df_v24 is not None and len(df_v24) > 0 else pd.DataFrame()
    
    if len(top10_v23) == 0 and len(top10_v24) == 0:
        log.warning("两个版本都没有有效的Top10结果")
        return
    
    # v2.3.0 Top10
    if len(top10_v23) > 0:
        log.info("\n【v2.3.0 Top10】")
        log.info(f"  平均收益率: {top10_v23['actual_return'].mean():.2f}%")
        log.info(f"  中位数收益率: {top10_v23['actual_return'].median():.2f}%")
        log.info(f"  正收益股票数: {(top10_v23['actual_return'] > 0).sum()}/{len(top10_v23)}")
        log.info(f"  平均收益率>10%: {(top10_v23['actual_return'] > 10).sum()} 只")
        log.info(f"  平均收益率>20%: {(top10_v23['actual_return'] > 20).sum()} 只")
        log.info(f"  最大收益率: {top10_v23['actual_return'].max():.2f}%")
        log.info(f"  最小收益率: {top10_v23['actual_return'].min():.2f}%")
        
        log.info("\n  详细列表:")
        for idx, row in top10_v23.iterrows():
            log.info(f"    {row['name']:10s} ({row['ts_code']}): {row['actual_return']:6.2f}% (预测概率: {row['predicted_probability']:.2%})")
    
    # v2.4.0 Top10
    if len(top10_v24) > 0:
        log.info("\n【v2.4.0 Top10】")
        log.info(f"  平均收益率: {top10_v24['actual_return'].mean():.2f}%")
        log.info(f"  中位数收益率: {top10_v24['actual_return'].median():.2f}%")
        log.info(f"  正收益股票数: {(top10_v24['actual_return'] > 0).sum()}/{len(top10_v24)}")
        log.info(f"  平均收益率>10%: {(top10_v24['actual_return'] > 10).sum()} 只")
        log.info(f"  平均收益率>20%: {(top10_v24['actual_return'] > 20).sum()} 只")
        log.info(f"  最大收益率: {top10_v24['actual_return'].max():.2f}%")
        log.info(f"  最小收益率: {top10_v24['actual_return'].min():.2f}%")
        
        log.info("\n  详细列表:")
        for idx, row in top10_v24.iterrows():
            log.info(f"    {row['name']:10s} ({row['ts_code']}): {row['actual_return']:6.2f}% (预测概率: {row['predicted_probability']:.2%})")
    
    # 对比分析
    if len(top10_v23) > 0 and len(top10_v24) > 0:
        log.info("\n【对比分析】")
        
        mean_v23 = top10_v23['actual_return'].mean()
        mean_v24 = top10_v24['actual_return'].mean()
        improvement = mean_v24 - mean_v23
        
        log.info(f"  平均收益率: v2.3.0={mean_v23:.2f}%, v2.4.0={mean_v24:.2f}%, 提升={improvement:.2f}%")
        
        median_v23 = top10_v23['actual_return'].median()
        median_v24 = top10_v24['actual_return'].median()
        log.info(f"  中位数收益率: v2.3.0={median_v23:.2f}%, v2.4.0={median_v24:.2f}%")
        
        positive_v23 = (top10_v23['actual_return'] > 0).sum()
        positive_v24 = (top10_v24['actual_return'] > 0).sum()
        log.info(f"  正收益股票数: v2.3.0={positive_v23}/10, v2.4.0={positive_v24}/10")
        
        high_return_v23 = (top10_v23['actual_return'] > 20).sum()
        high_return_v24 = (top10_v24['actual_return'] > 20).sum()
        log.info(f"  高收益(>20%)股票数: v2.3.0={high_return_v23}/10, v2.4.0={high_return_v24}/10")
        
        # 综合评分
        score_v23 = (
            mean_v23 * 0.4 +  # 平均收益权重40%
            median_v23 * 0.2 +  # 中位数收益权重20%
            positive_v23 * 5 +  # 正收益数权重（每只5分）
            high_return_v23 * 10  # 高收益数权重（每只10分）
        )
        
        score_v24 = (
            mean_v24 * 0.4 +
            median_v24 * 0.2 +
            positive_v24 * 5 +
            high_return_v24 * 10
        )
        
        log.info("\n【综合评分】")
        log.info(f"  v2.3.0: {score_v23:.2f} 分")
        log.info(f"  v2.4.0: {score_v24:.2f} 分")
        log.info(f"  提升: {score_v24 - score_v23:.2f} 分 ({((score_v24 - score_v23) / abs(score_v23) * 100) if score_v23 != 0 else 0:.1f}%)")
        
        if score_v24 > score_v23:
            log.success(f"✅ v2.4.0 表现优于 v2.3.0")
        elif score_v24 < score_v23:
            log.warning(f"⚠️  v2.4.0 表现不如 v2.3.0")
        else:
            log.info(f"➡️  v2.4.0 与 v2.3.0 表现相当")


def main():
    parser = argparse.ArgumentParser(description='对比v2.3.0和v2.4.0模型预测效果')
    parser.add_argument('--predict-date', type=str, default='20251212', help='预测日期(YYYYMMDD)')
    parser.add_argument('--evaluate-date', type=str, default='20260105', help='评估日期(YYYYMMDD)')
    parser.add_argument('--top', type=int, default=10, help='Top N股票数量')
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("v2.3.0 vs v2.4.0 模型预测效果对比")
    log.info("="*80)
    log.info(f"预测日期: {args.predict_date}")
    log.info(f"评估日期: {args.evaluate_date}")
    log.info(f"对比Top: {args.top}")
    log.info("")
    
    # 初始化数据管理器
    log.info("[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    
    # 加载模型
    log.info("\n[步骤2] 加载模型...")
    
    # 尝试加载v2.3.1，如果不存在则加载v2.3.0
    v23_version = 'v2.3.1'
    booster_v23, feature_names_v23, calibrator_v23 = load_model(v23_version)
    if booster_v23 is None:
        v23_version = 'v2.3.0'
        log.info(f"v2.3.1不存在，尝试加载v2.3.0...")
        booster_v23, feature_names_v23, calibrator_v23 = load_model(v23_version)
    
    booster_v24, feature_names_v24, calibrator_v24 = load_model('v2.4.0')
    
    if booster_v23 is None or booster_v24 is None:
        log.error("无法加载模型")
        return
    
    # 获取有效股票
    log.info("\n[步骤3] 获取有效股票...")
    predict_date = datetime.strptime(args.predict_date, '%Y%m%d')
    stocks = get_valid_stocks(dm, predict_date)
    log.info(f"  有效股票数: {len(stocks)}")
    
    # 预测
    log.info("\n[步骤4] 使用v2.3.0模型预测...")
    df_predictions_v23 = predict_stocks(
        booster_v23, calibrator_v23, feature_names_v23, 
        dm, stocks, args.predict_date, top_n=50
    )
    
    log.info("\n[步骤5] 使用v2.4.0模型预测...")
    df_predictions_v24 = predict_stocks(
        booster_v24, calibrator_v24, feature_names_v24,
        dm, stocks, args.predict_date, top_n=50
    )
    
    # 评估实际收益
    log.info("\n[步骤6] 评估实际收益...")
    df_results_v23 = evaluate_predictions(
        df_predictions_v23, dm, args.predict_date, args.evaluate_date, v23_version
    )
    df_results_v24 = evaluate_predictions(
        df_predictions_v24, dm, args.predict_date, args.evaluate_date, 'v2.4.0'
    )
    
    # 保存结果
    log.info("\n[步骤7] 保存结果...")
    output_dir = PROJECT_ROOT / 'data' / 'prediction' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if df_results_v23 is not None:
        output_file_v23 = output_dir / f'{v23_version}_predictions_{args.predict_date}_evaluated_{args.evaluate_date}.csv'
        df_results_v23.to_csv(output_file_v23, index=False, encoding='utf-8-sig')
        log.success(f"✓ {v23_version} 结果已保存: {output_file_v23}")
    
    if df_results_v24 is not None:
        output_file_v24 = output_dir / f'v2.4.0_predictions_{args.predict_date}_evaluated_{args.evaluate_date}.csv'
        df_results_v24.to_csv(output_file_v24, index=False, encoding='utf-8-sig')
        log.success(f"✓ v2.4.0 结果已保存: {output_file_v24}")
    
    # 对比Top10
    compare_top10(df_results_v23, df_results_v24)
    
    log.info("")
    log.info("="*80)
    log.success("✅ 对比评估完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()

