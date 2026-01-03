"""
优化版模型训练脚本 v2.1.0 - 高级特征版

三重优化：
1. 防过拟合的模型参数（低复杂度、强正则化）
2. 硬负样本（34日涨幅20-45%）
3. 113个高级技术因子特征

使用方法：
    python scripts/train_optimized_model_advanced.py
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score
)

from src.utils.logger import log


def safe_to_datetime(date_value):
    """安全地将日期值转换为datetime类型"""
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(date_value, errors='coerce')


def load_and_prepare_data(use_hard_negatives: bool = True):
    """
    加载并准备训练数据（高级特征版）
    """
    log.info("="*80)
    log.info("第一步：加载高级特征数据")
    log.info("="*80)
    
    # 加载正样本（高级特征版）
    df_pos = pd.read_csv('data/training/processed/feature_data_34d_advanced.csv')
    df_pos['label'] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条，{df_pos['sample_id'].nunique()} 个样本")
    
    # 加载普通负样本（高级特征版）
    neg_file = 'data/training/features/negative_feature_data_v2_34d_advanced.csv'
    df_neg = pd.read_csv(neg_file)
    df_neg['label'] = 0
    df_neg['sample_type'] = 'random_negative'
    log.success(f"✓ 普通负样本加载完成: {len(df_neg)} 条，{df_neg['sample_id'].nunique()} 个样本")
    
    # 加载硬负样本（高级特征版）
    if use_hard_negatives:
        hard_neg_file = 'data/training/features/hard_negative_feature_data_34d_advanced.csv'
        
        if os.path.exists(hard_neg_file):
            df_hard_neg = pd.read_csv(hard_neg_file)
            df_hard_neg['label'] = 0
            df_hard_neg['sample_type'] = 'hard_negative'
            log.success(f"✓ 硬负样本加载完成: {len(df_hard_neg)} 条，{df_hard_neg['sample_id'].nunique()} 个样本")
            
            df = pd.concat([df_pos, df_neg, df_hard_neg])
            log.info(f"\n数据合并完成:")
            log.info(f"  - 正样本: {len(df_pos)} 条 ({df_pos['sample_id'].nunique()} 个)")
            log.info(f"  - 普通负样本: {len(df_neg)} 条 ({df_neg['sample_id'].nunique()} 个)")
            log.info(f"  - 硬负样本: {len(df_hard_neg)} 条 ({df_hard_neg['sample_id'].nunique()} 个)")
            log.info(f"  - 总计: {len(df)} 条")
            log.info(f"  - 特征数: {len(df.columns)} 列")
        else:
            log.warning(f"硬负样本文件不存在: {hard_neg_file}")
            df = pd.concat([df_pos, df_neg])
    else:
        df = pd.concat([df_pos, df_neg])
    
    return df


def extract_features_with_time(df):
    """
    提取特征（高级版本 - 保留更多技术因子）
    """
    log.info("="*80)
    log.info("第二步：特征工程（高级版）")
    log.info("="*80)
    
    # 重新分配唯一的sample_id
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    # 获取所有列名
    all_columns = df.columns.tolist()
    
    # 识别数值型特征列（排除元数据列）
    meta_cols = ['sample_id', 'unique_sample_id', 'trade_date', 'ts_code', 'name', 
                 'label', 'sample_type', 'days_to_t1']
    numeric_cols = [col for col in all_columns if col not in meta_cols]
    
    log.info(f"识别到 {len(numeric_cols)} 个数值特征列")
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')
        
        if len(sample_data) < 20:
            continue
        
        t1_row = sample_data.iloc[sample_data['days_to_t1'].abs().argmin()]
        t1_date = safe_to_datetime(t1_row['trade_date'])
        
        feature_dict = {
            'sample_id': sample_id,
            'ts_code': sample_data['ts_code'].iloc[0],
            'name': sample_data['name'].iloc[0] if 'name' in sample_data.columns else '',
            'label': int(sample_data['label'].iloc[0]),
            't1_date': t1_date,
        }
        
        # 对每个数值列计算统计特征
        for col in numeric_cols:
            if col in sample_data.columns:
                col_data = pd.to_numeric(sample_data[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    # 使用最后一个值（T1时刻的值）
                    if col.endswith('_last') or col.endswith('_sum') or col.endswith('_count'):
                        feature_dict[col] = col_data.iloc[-1]
                    # 对于均值类特征，直接取均值
                    elif col.endswith('_mean'):
                        feature_dict[col] = col_data.mean()
                    # 对于标准差类特征，直接取标准差
                    elif col.endswith('_std'):
                        feature_dict[col] = col_data.std()
                    # 对于最大值类特征
                    elif col.endswith('_max'):
                        feature_dict[col] = col_data.max()
                    # 对于最小值类特征
                    elif col.endswith('_min'):
                        feature_dict[col] = col_data.min()
                    # 其他情况取均值
                    else:
                        feature_dict[col] = col_data.mean()
        
        features.append(feature_dict)
    
    df_features = pd.DataFrame(features)
    
    # 统计特征数量
    feature_cols = [col for col in df_features.columns 
                   if col not in ['sample_id', 'ts_code', 'name', 'label', 't1_date']]
    
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info(f"✓ 特征维度: {len(feature_cols)} 个特征")
    
    return df_features


def timeseries_split(df_features, train_ratio=0.8):
    """时间序列划分"""
    log.info("="*80)
    log.info("第三步：时间序列划分")
    log.info("="*80)
    
    df_features['t1_date'] = df_features['t1_date'].apply(safe_to_datetime)
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    min_date = df_features['t1_date'].min()
    max_date = df_features['t1_date'].max()
    log.info(f"数据时间范围: {min_date.date()} 至 {max_date.date()}")
    
    n_train = int(len(df_features) * train_ratio)
    train_end_date = df_features.iloc[n_train]['t1_date']
    test_start_date = df_features.iloc[n_train + 1]['t1_date']
    
    train_mask = df_features['t1_date'] <= train_end_date
    test_mask = df_features['t1_date'] >= test_start_date
    
    df_train = df_features[train_mask]
    df_test = df_features[test_mask]
    
    log.info(f"\n时间划分:")
    log.info(f"  训练集: {df_train['t1_date'].min().date()} 至 {df_train['t1_date'].max().date()}")
    log.info(f"  测试集: {df_test['t1_date'].min().date()} 至 {df_test['t1_date'].max().date()}")
    log.info(f"\n样本划分:")
    log.info(f"  训练集: {len(df_train)} 个样本 (正:{(df_train['label']==1).sum()}, 负:{(df_train['label']==0).sum()})")
    log.info(f"  测试集: {len(df_test)} 个样本 (正:{(df_test['label']==1).sum()}, 负:{(df_test['label']==0).sum()})")
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['sample_id', 'label', 't1_date', 'ts_code', 'name']]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['label']
    train_dates = df_train['t1_date']
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test['label']
    test_dates = df_test['t1_date']
    
    # 删除非数值列
    non_numeric_cols = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        log.info(f"删除非数值列: {list(non_numeric_cols)}")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)
    
    log.info(f"\n最终特征数: {len(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def train_model(X_train, y_train, X_test, y_test, params: dict):
    """训练模型"""
    log.info("="*80)
    log.info("第四步：训练XGBoost模型")
    log.info("="*80)
    
    log.info("模型参数:")
    for key, value in params.items():
        log.info(f"  {key}: {value}")
    
    log.info("\n开始训练...")
    
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    log.success("✓ 模型训练完成！")
    
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    log.info("="*80)
    log.info("第五步：模型评估")
    log.info("="*80)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    log.info("\n分类报告:")
    report = classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本'],
        output_dict=True
    )
    print(classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本']
    ))
    
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"\nAUC-ROC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    log.info("\n混淆矩阵:")
    log.info(f"  真负例(TN): {cm[0,0]:4d}  |  假正例(FP): {cm[0,1]:4d}")
    log.info(f"  假负例(FN): {cm[1,0]:4d}  |  真正例(TP): {cm[1,1]:4d}")
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['正样本']['precision'],
        'recall': report['正样本']['recall'],
        'f1_score': report['正样本']['f1-score'],
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, y_prob


def analyze_probability_distribution(y_test, y_prob):
    """分析概率分布"""
    log.info("="*80)
    log.info("概率分布分析")
    log.info("="*80)
    
    df_analysis = pd.DataFrame({
        'label': y_test.values,
        'probability': y_prob
    })
    
    bins = [0, 0.5, 0.8, 0.9, 0.95, 0.975, 0.985, 0.99, 1.0]
    df_analysis['prob_bin'] = pd.cut(df_analysis['probability'], bins=bins)
    
    log.info("\n各概率区间的样本分布:")
    for bin_range in sorted(df_analysis['prob_bin'].dropna().unique()):
        subset = df_analysis[df_analysis['prob_bin'] == bin_range]
        if len(subset) > 0:
            positive_rate = subset['label'].mean() * 100
            log.info(f"  {bin_range}: {len(subset)} 样本, 正样本率: {positive_rate:.1f}%")


def main():
    """主函数"""
    log.info("="*80)
    log.info("优化版模型训练 v2.1.0 - 高级特征版")
    log.info("="*80)
    log.info("三重优化: 防过拟合参数 + 硬负样本 + 113个高级技术因子")
    log.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("")
    
    # 创建输出目录
    output_dir = Path('data/models/breakout_launch_scorer/versions/v2.1.0')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'model').mkdir(exist_ok=True)
    (output_dir / 'training').mkdir(exist_ok=True)
    
    # 防过拟合参数（与v2.0.0相同）
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
    
    # 1. 加载数据
    df = load_and_prepare_data(use_hard_negatives=True)
    
    # 2. 特征工程
    df_features = extract_features_with_time(df)
    
    # 3. 时间序列划分
    X_train, X_test, y_train, y_test, train_dates, test_dates = timeseries_split(df_features)
    
    # 4. 训练模型
    model = train_model(X_train, y_train, X_test, y_test, model_params)
    
    # 5. 评估模型
    metrics, y_prob = evaluate_model(model, X_test, y_test)
    
    # 6. 概率分布分析
    analyze_probability_distribution(y_test, y_prob)
    
    # 7. 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log.info("\n" + "="*80)
    log.info("特征重要性 Top 15:")
    log.info("="*80)
    for idx, row in feature_importance.head(15).iterrows():
        log.info(f"  {row['feature']:35s}: {row['importance']:.4f}")
    
    # 8. 保存模型和指标
    model_path = output_dir / 'model' / 'model.json'
    model.get_booster().save_model(str(model_path))
    log.success(f"✓ 模型已保存: {model_path}")
    
    # 保存特征名称
    feature_names_file = output_dir / 'model' / 'feature_names.json'
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        json.dump(list(X_train.columns), f, indent=2, ensure_ascii=False)
    
    # 保存指标
    metrics_file = output_dir / 'training' / 'metrics.json'
    metrics['model_file'] = str(model_path)
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics['train_date_range'] = f"{train_dates.min().date()} to {train_dates.max().date()}"
    metrics['test_date_range'] = f"{test_dates.min().date()} to {test_dates.max().date()}"
    metrics['params'] = model_params
    metrics['feature_count'] = len(X_train.columns)
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 指标已保存: {metrics_file}")
    
    # 9. 最终总结
    log.info("\n" + "="*80)
    log.success("✅ v2.1.0 高级特征版模型训练完成！")
    log.info("="*80)
    log.info(f"\n模型性能:")
    log.info(f"  特征数: {len(X_train.columns)}")
    log.info(f"  准确率: {metrics['accuracy']:.4f}")
    log.info(f"  精确率: {metrics['precision']:.4f}")
    log.info(f"  召回率: {metrics['recall']:.4f}")
    log.info(f"  F1分数: {metrics['f1_score']:.4f}")
    log.info(f"  AUC: {metrics['auc']:.4f}")
    log.info(f"\n模型文件: {model_path}")
    log.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

