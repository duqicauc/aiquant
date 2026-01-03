"""
优化版模型训练脚本

主要优化：
1. 添加硬负样本（34日涨幅20-45%的股票）
2. 使用防过拟合的模型参数
3. 与原模型对比评估

使用方法：
    python scripts/train_optimized_model.py
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve
)

from src.data.data_manager import DataManager
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log


def safe_to_datetime(date_value):
    """安全地将日期值转换为datetime类型"""
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(str(int(date_value)), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(date_value, errors='coerce')


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_and_prepare_data(use_hard_negatives: bool = True, hard_neg_ratio: float = 0.5):
    """
    加载并准备训练数据
    
    Args:
        use_hard_negatives: 是否使用硬负样本
        hard_neg_ratio: 硬负样本与正样本的比例
    
    Returns:
        合并后的DataFrame
    """
    log.info("="*80)
    log.info("第一步：加载数据")
    log.info("="*80)
    
    # 加载正样本（位于processed目录）
    df_pos = pd.read_csv('data/training/processed/feature_data_34d.csv')
    df_pos['label'] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")
    
    # 加载普通负样本（位于features目录）
    neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    df_neg = pd.read_csv(neg_file)
    df_neg['label'] = 0
    df_neg['sample_type'] = 'random_negative'
    log.success(f"✓ 普通负样本加载完成: {len(df_neg)} 条")
    
    # 加载或生成硬负样本
    if use_hard_negatives:
        hard_neg_file = 'data/training/features/hard_negative_feature_data_34d.csv'
        
        if os.path.exists(hard_neg_file):
            df_hard_neg = pd.read_csv(hard_neg_file)
            df_hard_neg['label'] = 0
            df_hard_neg['sample_type'] = 'hard_negative'
            log.success(f"✓ 硬负样本加载完成: {len(df_hard_neg)} 条")
        else:
            log.warning(f"硬负样本文件不存在: {hard_neg_file}")
            log.info("请先运行 prepare_hard_negatives.py 生成硬负样本")
            df_hard_neg = pd.DataFrame()
        
        # 合并
        if not df_hard_neg.empty:
            df = pd.concat([df_pos, df_neg, df_hard_neg])
            log.info(f"\n数据合并完成:")
            log.info(f"  - 正样本: {len(df_pos)} 条")
            log.info(f"  - 普通负样本: {len(df_neg)} 条")
            log.info(f"  - 硬负样本: {len(df_hard_neg)} 条")
            log.info(f"  - 总计: {len(df)} 条")
        else:
            df = pd.concat([df_pos, df_neg])
            log.info(f"\n数据合并完成（无硬负样本）:")
            log.info(f"  - 正样本: {len(df_pos)} 条")
            log.info(f"  - 负样本: {len(df_neg)} 条")
    else:
        df = pd.concat([df_pos, df_neg])
        log.info(f"\n数据合并完成:")
        log.info(f"  - 正样本: {len(df_pos)} 条")
        log.info(f"  - 负样本: {len(df_neg)} 条")
    
    return df


def extract_features_with_time(df):
    """
    提取特征（保留时间信息）
    """
    log.info("="*80)
    log.info("第二步：特征工程（保留时间信息）")
    log.info("="*80)
    
    # 重新分配唯一的sample_id
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    # 获取T1日期映射
    df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
    t1_date_map = dict(zip(
        df_positive_samples.index,
        df_positive_samples['t1_date'].apply(safe_to_datetime)
    ))
    
    if os.path.exists('data/training/samples/negative_samples_v2.csv'):
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples_v2.csv')
    else:
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples.csv')
    
    max_positive_id = df_positive_samples.index.max()
    for idx, row in df_negative_samples.iterrows():
        t1_date_map[max_positive_id + 1 + idx] = safe_to_datetime(row['t1_date'])
    
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
            'name': sample_data['name'].iloc[0],
            'label': int(sample_data['label'].iloc[0]),
            't1_date': t1_date,
        }
        
        # 价格特征
        feature_dict['close_mean'] = sample_data['close'].mean()
        feature_dict['close_std'] = sample_data['close'].std()
        feature_dict['close_max'] = sample_data['close'].max()
        feature_dict['close_min'] = sample_data['close'].min()
        feature_dict['close_trend'] = (
            (sample_data['close'].iloc[-1] - sample_data['close'].iloc[0]) / 
            sample_data['close'].iloc[0] * 100
        )
        
        # 涨跌幅特征
        feature_dict['pct_chg_mean'] = sample_data['pct_chg'].mean()
        feature_dict['pct_chg_std'] = sample_data['pct_chg'].std()
        feature_dict['pct_chg_sum'] = sample_data['pct_chg'].sum()
        feature_dict['positive_days'] = (sample_data['pct_chg'] > 0).sum()
        feature_dict['negative_days'] = (sample_data['pct_chg'] < 0).sum()
        feature_dict['max_gain'] = sample_data['pct_chg'].max()
        feature_dict['max_loss'] = sample_data['pct_chg'].min()
        
        # 量比特征
        if 'volume_ratio' in sample_data.columns:
            feature_dict['volume_ratio_mean'] = sample_data['volume_ratio'].mean()
            feature_dict['volume_ratio_max'] = sample_data['volume_ratio'].max()
            feature_dict['volume_ratio_gt_2'] = (sample_data['volume_ratio'] > 2).sum()
            feature_dict['volume_ratio_gt_4'] = (sample_data['volume_ratio'] > 4).sum()
        
        # MACD特征
        if 'macd' in sample_data.columns:
            macd_data = sample_data['macd'].dropna()
            if len(macd_data) > 0:
                feature_dict['macd_mean'] = macd_data.mean()
                feature_dict['macd_positive_days'] = (macd_data > 0).sum()
                feature_dict['macd_max'] = macd_data.max()
        
        # MA特征
        if 'ma5' in sample_data.columns:
            feature_dict['ma5_mean'] = sample_data['ma5'].mean()
            feature_dict['price_above_ma5'] = (
                sample_data['close'] > sample_data['ma5']
            ).sum()
        
        if 'ma10' in sample_data.columns:
            feature_dict['ma10_mean'] = sample_data['ma10'].mean()
            feature_dict['price_above_ma10'] = (
                sample_data['close'] > sample_data['ma10']
            ).sum()
        
        # 市值特征
        if 'total_mv' in sample_data.columns:
            mv_data = sample_data['total_mv'].dropna()
            if len(mv_data) > 0:
                feature_dict['total_mv_mean'] = mv_data.mean()
        
        if 'circ_mv' in sample_data.columns:
            circ_mv_data = sample_data['circ_mv'].dropna()
            if len(circ_mv_data) > 0:
                feature_dict['circ_mv_mean'] = circ_mv_data.mean()
        
        # 动量特征
        days = len(sample_data)
        if days >= 7:
            feature_dict['return_1w'] = (
                (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-7]) /
                sample_data['close'].iloc[-7] * 100
            )
        if days >= 14:
            feature_dict['return_2w'] = (
                (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-14]) /
                sample_data['close'].iloc[-14] * 100
            )
        
        features.append(feature_dict)
    
    df_features = pd.DataFrame(features)
    
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info(f"✓ 特征维度: {len(df_features.columns) - 5} 个特征")
    
    return df_features


def timeseries_split(df_features, train_ratio=0.8):
    """时间序列划分"""
    log.info("="*80)
    log.info("第三步：时间序列划分")
    log.info("="*80)
    
    # 确保t1_date是datetime类型
    df_features['t1_date'] = df_features['t1_date'].apply(safe_to_datetime)
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 显示时间范围
    min_date = df_features['t1_date'].min()
    max_date = df_features['t1_date'].max()
    log.info(f"数据时间范围: {min_date.date()} 至 {max_date.date()}")
    
    # 划分
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
    
    # 准备特征和标签
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
    
    # 训练（支持早停）
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    if params.get('early_stopping_rounds'):
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
    else:
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
    
    log.success("✓ 模型训练完成！")
    
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    log.info("="*80)
    log.info("第五步：模型评估")
    log.info("="*80)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 分类报告
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
    
    # AUC
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"\nAUC-ROC: {auc:.4f}")
    
    # 混淆矩阵
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
    
    # 按概率区间分析
    bins = [0, 0.5, 0.8, 0.9, 0.95, 0.975, 0.985, 0.99, 1.0]
    df_analysis['prob_bin'] = pd.cut(df_analysis['probability'], bins=bins)
    
    log.info("\n各概率区间的样本分布:")
    for bin_range in df_analysis['prob_bin'].unique():
        if pd.isna(bin_range):
            continue
        subset = df_analysis[df_analysis['prob_bin'] == bin_range]
        if len(subset) > 0:
            positive_rate = subset['label'].mean() * 100
            log.info(f"  {bin_range}: {len(subset)} 样本, 正样本率: {positive_rate:.1f}%")


def main():
    """主函数"""
    log.info("="*80)
    log.info("优化版模型训练 - 防过拟合 + 硬负样本")
    log.info("="*80)
    log.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("")
    
    # 创建输出目录
    output_dir = Path('data/models/breakout_launch_scorer/versions/v2.0.0')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'model').mkdir(exist_ok=True)
    (output_dir / 'training').mkdir(exist_ok=True)
    
    # 加载配置
    config = load_config('config/models/breakout_launch_scorer_v2.yaml')
    
    # 防过拟合参数
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
    log.info("特征重要性 Top 10:")
    log.info("="*80)
    for idx, row in feature_importance.head(10).iterrows():
        log.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
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
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 指标已保存: {metrics_file}")
    
    # 9. 最终总结
    log.info("\n" + "="*80)
    log.success("✅ 优化版模型训练完成！")
    log.info("="*80)
    log.info(f"\n模型性能:")
    log.info(f"  准确率: {metrics['accuracy']:.4f}")
    log.info(f"  精确率: {metrics['precision']:.4f}")
    log.info(f"  召回率: {metrics['recall']:.4f}")
    log.info(f"  F1分数: {metrics['f1_score']:.4f}")
    log.info(f"  AUC: {metrics['auc']:.4f}")
    log.info(f"\n模型文件: {model_path}")
    log.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

