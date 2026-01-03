"""
模型评估脚本 - 预测2025年9月19日并评估后续收益

确保训练数据不包含2025年9月19日及之后的数据
用于公平评估v2.0.0和v2.1.0模型的实际预测效果

使用方法：
    python scripts/evaluate_models_20250919.py
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

from src.data.data_manager import DataManager
from src.utils.logger import log


# 评估日期配置
PREDICTION_DATE = '20250919'  # 预测日期
CUTOFF_DATE = pd.to_datetime('2025-09-18')  # 训练数据截止日期（预测日前一天）
EVAL_WEEKS = [1, 2, 3, 4]  # 评估周数


def safe_to_datetime(date_value):
    """安全地将日期值转换为datetime类型"""
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(date_value, errors='coerce')


def load_training_data_before_cutoff(use_advanced_features: bool = False):
    """
    加载训练数据，确保只使用截止日期之前的数据
    """
    log.info(f"加载训练数据（截止日期: {CUTOFF_DATE.date()}）")
    
    if use_advanced_features:
        pos_file = 'data/training/processed/feature_data_34d_advanced.csv'
        neg_file = 'data/training/features/negative_feature_data_v2_34d_advanced.csv'
        hard_neg_file = 'data/training/features/hard_negative_feature_data_34d_advanced.csv'
    else:
        pos_file = 'data/training/processed/feature_data_34d.csv'
        neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
        hard_neg_file = 'data/training/features/hard_negative_feature_data_34d.csv'
    
    # 加载正样本
    df_pos = pd.read_csv(pos_file)
    df_pos['label'] = 1
    
    # 加载负样本
    df_neg = pd.read_csv(neg_file)
    df_neg['label'] = 0
    
    # 加载硬负样本
    if os.path.exists(hard_neg_file):
        df_hard_neg = pd.read_csv(hard_neg_file)
        df_hard_neg['label'] = 0
        df = pd.concat([df_pos, df_neg, df_hard_neg])
    else:
        df = pd.concat([df_pos, df_neg])
    
    # 过滤：只保留截止日期之前的数据
    # 需要按sample_id分组，取每个样本的最后一天（T1日期）
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    
    # 获取每个样本的T1日期
    sample_t1_dates = df.groupby(['sample_id', 'label'])['trade_date'].max().reset_index()
    sample_t1_dates.columns = ['sample_id', 'label', 't1_date']
    
    # 筛选截止日期之前的样本
    valid_samples = sample_t1_dates[sample_t1_dates['t1_date'] <= CUTOFF_DATE]
    valid_sample_keys = set(zip(valid_samples['sample_id'], valid_samples['label']))
    
    # 过滤数据
    df['sample_key'] = list(zip(df['sample_id'], df['label']))
    df_filtered = df[df['sample_key'].isin(valid_sample_keys)].drop(columns=['sample_key'])
    
    log.info(f"原始样本数: {len(sample_t1_dates)}")
    log.info(f"过滤后样本数: {len(valid_samples)}")
    log.info(f"  - 正样本: {len(valid_samples[valid_samples['label']==1])}")
    log.info(f"  - 负样本: {len(valid_samples[valid_samples['label']==0])}")
    
    return df_filtered


def extract_features(df, use_advanced: bool = False):
    """提取特征"""
    log.info("提取特征...")
    
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    # 识别数值列
    meta_cols = ['sample_id', 'unique_sample_id', 'trade_date', 'ts_code', 'name', 
                 'label', 'sample_type', 'days_to_t1', 'sample_key']
    numeric_cols = [col for col in df.columns if col not in meta_cols]
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 1000 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df[df['unique_sample_id'] == sample_id].sort_values('trade_date')
        
        if len(sample_data) < 20:
            continue
        
        t1_date = sample_data['trade_date'].max()
        
        feature_dict = {
            'sample_id': sample_id,
            'ts_code': sample_data['ts_code'].iloc[0],
            'name': sample_data['name'].iloc[0] if 'name' in sample_data.columns else '',
            'label': int(sample_data['label'].iloc[0]),
            't1_date': t1_date,
        }
        
        # 计算特征
        for col in numeric_cols:
            if col in sample_data.columns:
                col_data = pd.to_numeric(sample_data[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    if col.endswith('_last') or col.endswith('_sum') or col.endswith('_count'):
                        feature_dict[col] = col_data.iloc[-1]
                    elif col.endswith('_mean'):
                        feature_dict[col] = col_data.mean()
                    elif col.endswith('_std'):
                        feature_dict[col] = col_data.std()
                    elif col.endswith('_max'):
                        feature_dict[col] = col_data.max()
                    elif col.endswith('_min'):
                        feature_dict[col] = col_data.min()
                    else:
                        feature_dict[col] = col_data.mean()
        
        features.append(feature_dict)
    
    df_features = pd.DataFrame(features)
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    
    return df_features


def train_model(df_features, model_params):
    """训练模型"""
    # 时间序列划分（80%训练，20%验证）
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    n_train = int(len(df_features) * 0.8)
    
    df_train = df_features.iloc[:n_train]
    df_val = df_features.iloc[n_train:]
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['sample_id', 'label', 't1_date', 'ts_code', 'name']]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['label']
    
    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['label']
    
    # 删除非数值列
    non_numeric = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        X_train = X_train.drop(columns=non_numeric)
        X_val = X_val.drop(columns=non_numeric)
    
    log.info(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")
    log.info(f"特征数: {len(X_train.columns)}")
    
    # 训练
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 验证集评估
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    
    log.success(f"✓ 模型训练完成，验证集AUC: {auc:.4f}")
    
    return model, list(X_train.columns), auc


def predict_stocks(model, feature_names, dm, prediction_date):
    """
    预测指定日期的股票
    """
    log.info(f"预测 {prediction_date} 的股票...")
    
    # 获取所有股票
    stock_list = dm.get_stock_list(list_status='L')
    
    # 过滤ST和北交所
    stock_list = stock_list[~stock_list['name'].str.contains('ST', na=False, case=False)]
    stock_list = stock_list[~stock_list['ts_code'].str.endswith('.BJ')]
    stock_list = stock_list[~stock_list['name'].str.contains('退', na=False)]
    
    log.info(f"有效股票: {len(stock_list)} 只")
    
    predictions = []
    pred_date = pd.to_datetime(prediction_date)
    start_date = (pred_date - timedelta(days=60)).strftime('%Y%m%d')
    end_date = prediction_date
    
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        if (idx + 1) % 500 == 0:
            log.info(f"进度: {idx+1}/{len(stock_list)}, 已预测: {len(predictions)}")
        
        try:
            # 获取数据
            df = dm.get_complete_data(ts_code, start_date, end_date)
            if df.empty or len(df) < 20:
                continue
            
            df = df.tail(34)
            
            # 计算特征
            feature_dict = {}
            for col in feature_names:
                if col in df.columns:
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(col_data) > 0:
                        if col.endswith('_last') or col.endswith('_sum') or col.endswith('_count'):
                            feature_dict[col] = col_data.iloc[-1]
                        elif col.endswith('_mean'):
                            feature_dict[col] = col_data.mean()
                        elif col.endswith('_std'):
                            feature_dict[col] = col_data.std()
                        elif col.endswith('_max'):
                            feature_dict[col] = col_data.max()
                        elif col.endswith('_min'):
                            feature_dict[col] = col_data.min()
                        else:
                            feature_dict[col] = col_data.mean()
            
            # 准备特征向量
            X = pd.DataFrame([feature_dict])[feature_names].fillna(0)
            
            # 预测
            prob = model.predict_proba(X)[0, 1]
            
            predictions.append({
                'ts_code': ts_code,
                'name': name,
                'probability': prob,
                'prediction_date': prediction_date,
                'close_price': df['close'].iloc[-1]
            })
            
        except Exception as e:
            continue
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions = df_predictions.sort_values('probability', ascending=False)
    
    log.success(f"✓ 预测完成: {len(df_predictions)} 只股票")
    
    return df_predictions


def evaluate_predictions(df_predictions, dm, eval_weeks):
    """
    评估预测结果
    """
    log.info("="*80)
    log.info("评估预测结果")
    log.info("="*80)
    
    pred_date = pd.to_datetime(df_predictions['prediction_date'].iloc[0])
    
    # 获取后续价格
    results = []
    
    for idx, row in df_predictions.iterrows():
        ts_code = row['ts_code']
        
        try:
            # 获取未来数据
            future_start = (pred_date + timedelta(days=1)).strftime('%Y%m%d')
            future_end = (pred_date + timedelta(days=max(eval_weeks)*7+10)).strftime('%Y%m%d')
            
            df_future = dm.get_daily_data(ts_code, future_start, future_end, adjust='qfq')
            
            if df_future.empty:
                continue
            
            result = {
                'ts_code': row['ts_code'],
                'name': row['name'],
                'probability': row['probability'],
                'pred_price': row['close_price']
            }
            
            # 计算各周收益
            df_future = df_future.sort_values('trade_date')
            
            for week in eval_weeks:
                target_days = week * 5  # 交易日
                if len(df_future) >= target_days:
                    future_price = df_future.iloc[target_days-1]['close']
                    return_pct = (future_price - row['close_price']) / row['close_price'] * 100
                    result[f'{week}w_price'] = future_price
                    result[f'{week}w_return'] = return_pct
                else:
                    result[f'{week}w_price'] = None
                    result[f'{week}w_return'] = None
            
            results.append(result)
            
        except Exception as e:
            continue
    
    df_results = pd.DataFrame(results)
    
    return df_results


def print_evaluation_report(df_results, model_name, top_n=50):
    """
    打印评估报告
    """
    log.info("")
    log.info("="*80)
    log.info(f"{model_name} - Top {top_n} 预测评估报告")
    log.info("="*80)
    
    df_top = df_results.head(top_n)
    
    for week in EVAL_WEEKS:
        col = f'{week}w_return'
        if col in df_top.columns:
            valid = df_top[col].dropna()
            if len(valid) > 0:
                avg_return = valid.mean()
                win_rate = (valid > 0).sum() / len(valid) * 100
                max_return = valid.max()
                min_return = valid.min()
                
                log.info(f"\n{week}周后评估 ({len(valid)}只有效):")
                log.info(f"  平均收益率: {avg_return:.2f}%")
                log.info(f"  胜率: {win_rate:.1f}%")
                log.info(f"  最大收益: {max_return:.2f}%")
                log.info(f"  最大亏损: {min_return:.2f}%")
    
    # 概率区间分析
    log.info("\n概率区间收益分析（4周）:")
    bins = [(0.99, 1.0), (0.98, 0.99), (0.97, 0.98), (0.95, 0.97), (0.90, 0.95)]
    for low, high in bins:
        subset = df_results[(df_results['probability'] >= low) & (df_results['probability'] < high)]
        if len(subset) > 0 and '4w_return' in subset.columns:
            valid = subset['4w_return'].dropna()
            if len(valid) > 0:
                avg = valid.mean()
                win = (valid > 0).sum() / len(valid) * 100
                log.info(f"  [{low*100:.0f}%-{high*100:.0f}%): {len(valid)}只, 收益{avg:.2f}%, 胜率{win:.1f}%")


def main():
    """主函数"""
    log.info("="*80)
    log.info("模型评估 - 预测2025年9月19日")
    log.info("="*80)
    log.info(f"预测日期: {PREDICTION_DATE}")
    log.info(f"训练数据截止: {CUTOFF_DATE.date()}")
    log.info(f"评估周数: {EVAL_WEEKS}")
    log.info("")
    
    # 模型参数
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 3,
        'min_child_weight': 10,
        'gamma': 0.3,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'scale_pos_weight': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 30,
    }
    
    dm = DataManager()
    
    # ===================== v2.0.0 基础特征版 =====================
    log.info("="*80)
    log.info("训练 v2.0.0 基础特征版（27个特征）")
    log.info("="*80)
    
    df_train_basic = load_training_data_before_cutoff(use_advanced_features=False)
    df_features_basic = extract_features(df_train_basic, use_advanced=False)
    model_basic, features_basic, auc_basic = train_model(df_features_basic, model_params)
    
    log.info("\n预测股票...")
    df_pred_basic = predict_stocks(model_basic, features_basic, dm, PREDICTION_DATE)
    
    log.info("\n评估预测结果...")
    df_eval_basic = evaluate_predictions(df_pred_basic, dm, EVAL_WEEKS)
    
    print_evaluation_report(df_eval_basic, "v2.0.0 基础特征版", top_n=50)
    
    # 保存结果
    output_dir = Path('data/prediction/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_eval_basic.to_csv(output_dir / 'v2.0.0_eval_20250919.csv', index=False)
    
    # ===================== v2.1.0 高级特征版 =====================
    log.info("\n" + "="*80)
    log.info("训练 v2.1.0 高级特征版（125个特征）")
    log.info("="*80)
    
    df_train_adv = load_training_data_before_cutoff(use_advanced_features=True)
    df_features_adv = extract_features(df_train_adv, use_advanced=True)
    model_adv, features_adv, auc_adv = train_model(df_features_adv, model_params)
    
    log.info("\n预测股票...")
    df_pred_adv = predict_stocks(model_adv, features_adv, dm, PREDICTION_DATE)
    
    log.info("\n评估预测结果...")
    df_eval_adv = evaluate_predictions(df_pred_adv, dm, EVAL_WEEKS)
    
    print_evaluation_report(df_eval_adv, "v2.1.0 高级特征版", top_n=50)
    
    # 保存结果
    df_eval_adv.to_csv(output_dir / 'v2.1.0_eval_20250919.csv', index=False)
    
    # ===================== 对比总结 =====================
    log.info("\n" + "="*80)
    log.info("模型对比总结")
    log.info("="*80)
    
    log.info(f"\n验证集AUC:")
    log.info(f"  v2.0.0 基础特征版: {auc_basic:.4f}")
    log.info(f"  v2.1.0 高级特征版: {auc_adv:.4f}")
    
    # 4周收益对比
    log.info(f"\nTop 50 四周收益对比:")
    
    basic_4w = df_eval_basic.head(50)['4w_return'].dropna()
    adv_4w = df_eval_adv.head(50)['4w_return'].dropna()
    
    if len(basic_4w) > 0:
        log.info(f"  v2.0.0: 平均{basic_4w.mean():.2f}%, 胜率{(basic_4w>0).sum()/len(basic_4w)*100:.1f}%")
    if len(adv_4w) > 0:
        log.info(f"  v2.1.0: 平均{adv_4w.mean():.2f}%, 胜率{(adv_4w>0).sum()/len(adv_4w)*100:.1f}%")
    
    log.info("\n" + "="*80)
    log.success("✅ 评估完成！")
    log.info("="*80)
    log.info(f"\n结果已保存:")
    log.info(f"  - {output_dir / 'v2.0.0_eval_20250919.csv'}")
    log.info(f"  - {output_dir / 'v2.1.0_eval_20250919.csv'}")


if __name__ == '__main__':
    main()

