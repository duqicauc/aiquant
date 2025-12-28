"""
Walk-Forward验证脚本

在多个时间窗口上测试模型的稳定性和泛化能力
- 滑动窗口方式训练和测试
- 验证模型在不同市场环境下的表现
- 避免过拟合，确保模型鲁棒性
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
from src.utils.logger import log


def load_and_prepare_data(neg_version='v2'):
    """加载数据"""
    log.info("="*80)
    log.info("加载数据")
    log.info("="*80)
    
    # 加载正样本
    df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
    df_pos['label'] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")
    
    # 加载负样本
    if neg_version == 'v2':
        neg_file = 'data/processed/negative_feature_data_v2_34d.csv'
    else:
        neg_file = 'data/processed/negative_feature_data_34d.csv'
    
    df_neg = pd.read_csv(neg_file)
    log.success(f"✓ 负样本加载完成: {len(df_neg)} 条 (版本: {neg_version})")
    
    # 合并
    df = pd.concat([df_pos, df_neg])
    log.info(f"✓ 数据合并完成: {len(df)} 条")
    log.info("")
    
    return df


def extract_features_with_time(df):
    """从34天的时序数据中提取统计特征（保留时间信息）"""
    log.info("="*80)
    log.info("特征工程")
    log.info("="*80)
    log.info("将34天时序数据转换为统计特征...")
    
    # 重新分配唯一的sample_id
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')
        
        if len(sample_data) < 20:  # 至少20天数据
            continue
        
        # 从数据中获取T1日期
        t1_row = sample_data.iloc[sample_data['days_to_t1'].abs().argmin()]
        t1_date = pd.to_datetime(t1_row['trade_date'])
        
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
    log.info("")
    
    return df_features


def walk_forward_validation(df_features, n_splits=5, train_size=0.6):
    """
    Walk-forward验证
    
    Args:
        df_features: 特征DataFrame
        n_splits: 分割数量（时间窗口数）
        train_size: 训练集占比
    """
    log.info("="*80)
    log.info("Walk-Forward验证")
    log.info("="*80)
    log.info(f"时间窗口数: {n_splits}")
    log.info(f"训练集占比: {train_size*100}%")
    log.info("")
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 准备特征和标签
    feature_cols = [col for col in df_features.columns 
                    if col not in ['sample_id', 'ts_code', 'name', 'label', 't1_date']]
    
    X = df_features[feature_cols].fillna(0)
    y = df_features['label']
    dates = df_features['t1_date']
    
    # 计算每个窗口的大小
    total_samples = len(df_features)
    window_size = total_samples // n_splits
    
    results = []
    
    for i in range(n_splits):
        log.info(f"\n{'='*80}")
        log.info(f"时间窗口 {i+1}/{n_splits}")
        log.info(f"{'='*80}")
        
        # 计算训练集和测试集的索引
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_splits - 1 else total_samples
        
        window_data = df_features.iloc[start_idx:end_idx]
        
        # 按时间划分训练集和测试集
        split_idx = int(len(window_data) * train_size)
        
        train_data = window_data.iloc[:split_idx]
        test_data = window_data.iloc[split_idx:]
        
        if len(test_data) < 10:  # 测试集太小，跳过
            log.warning(f"⚠️  测试集样本太少({len(test_data)}个)，跳过此窗口")
            continue
        
        # 准备训练集和测试集
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['label']
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['label']
        
        train_dates = train_data['t1_date']
        test_dates = test_data['t1_date']
        
        log.info(f"\n时间范围:")
        log.info(f"  训练集: {train_dates.min().date()} 至 {train_dates.max().date()}")
        log.info(f"  测试集: {test_dates.min().date()} 至 {test_dates.max().date()}")
        
        log.info(f"\n样本分布:")
        log.info(f"  训练集: {len(X_train)} 个 (正:{y_train.sum()}, 负:{len(y_train)-y_train.sum()})")
        log.info(f"  测试集: {len(X_test)} 个 (正:{y_test.sum()}, 负:{len(y_test)-y_test.sum()})")
        
        # 训练模型
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0
        
        log.info(f"\n性能指标:")
        log.info(f"  准确率: {accuracy*100:.2f}%")
        log.info(f"  精确率: {precision*100:.2f}%")
        log.info(f"  召回率: {recall*100:.2f}%")
        log.info(f"  F1-Score: {f1*100:.2f}%")
        log.info(f"  AUC-ROC: {auc:.4f}")
        
        # 记录结果
        results.append({
            'window': i + 1,
            'train_start': train_dates.min().date().isoformat(),
            'train_end': train_dates.max().date().isoformat(),
            'test_start': test_dates.min().date().isoformat(),
            'test_end': test_dates.max().date().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        })
    
    return results


def analyze_results(results):
    """分析Walk-forward验证结果"""
    log.info("\n" + "="*80)
    log.info("Walk-Forward验证结果汇总")
    log.info("="*80)
    
    df_results = pd.DataFrame(results)
    
    # 计算统计量
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    log.info("\n各时间窗口表现:")
    for _, row in df_results.iterrows():
        log.info(f"\n窗口 {int(row['window'])}: {row['test_start']} 至 {row['test_end']}")
        log.info(f"  准确率: {row['accuracy']*100:.2f}%")
        log.info(f"  精确率: {row['precision']*100:.2f}%")
        log.info(f"  召回率: {row['recall']*100:.2f}%")
        log.info(f"  F1-Score: {row['f1_score']*100:.2f}%")
        log.info(f"  AUC: {row['auc']:.4f}")
    
    log.info("\n" + "="*80)
    log.info("统计摘要")
    log.info("="*80)
    
    for metric in metrics:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        min_val = df_results[metric].min()
        max_val = df_results[metric].max()
        
        metric_name = {
            'accuracy': '准确率',
            'precision': '精确率',
            'recall': '召回率',
            'f1_score': 'F1-Score',
            'auc': 'AUC-ROC'
        }[metric]
        
        log.info(f"\n{metric_name}:")
        log.info(f"  平均值: {mean_val*100:.2f}%")
        log.info(f"  标准差: {std_val*100:.2f}%")
        log.info(f"  最小值: {min_val*100:.2f}%")
        log.info(f"  最大值: {max_val*100:.2f}%")
    
    # 稳定性评估
    f1_std = df_results['f1_score'].std()
    if f1_std < 0.05:
        stability = "非常稳定 ⭐⭐⭐⭐⭐"
    elif f1_std < 0.10:
        stability = "稳定 ⭐⭐⭐⭐"
    elif f1_std < 0.15:
        stability = "一般 ⭐⭐⭐"
    else:
        stability = "不稳定 ⭐⭐"
    
    log.info(f"\n模型稳定性评估: {stability}")
    log.info(f"  (基于F1-Score标准差: {f1_std*100:.2f}%)")
    
    return df_results


def save_results(results_df):
    """保存验证结果"""
    output_file = 'data/results/walk_forward_validation_results.json'
    os.makedirs('data/results', exist_ok=True)
    
    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_windows': len(results_df),
        'summary': {
            'accuracy_mean': float(results_df['accuracy'].mean()),
            'accuracy_std': float(results_df['accuracy'].std()),
            'precision_mean': float(results_df['precision'].mean()),
            'precision_std': float(results_df['precision'].std()),
            'recall_mean': float(results_df['recall'].mean()),
            'recall_std': float(results_df['recall'].std()),
            'f1_score_mean': float(results_df['f1_score'].mean()),
            'f1_score_std': float(results_df['f1_score'].std()),
            'auc_mean': float(results_df['auc'].mean()),
            'auc_std': float(results_df['auc'].std()),
        },
        'windows': results_df.to_dict('records')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    log.success(f"\n✓ 验证结果已保存: {output_file}")


def main():
    """主函数"""
    log.info("="*80)
    log.info("Walk-Forward验证 - 多时间窗口模型稳定性测试")
    log.info("="*80)
    log.info("")
    
    try:
        # 1. 加载数据
        df = load_and_prepare_data(neg_version='v2')
        
        # 2. 特征工程
        df_features = extract_features_with_time(df)
        
        # 3. Walk-forward验证
        results = walk_forward_validation(df_features, n_splits=5, train_size=0.6)
        
        # 4. 分析结果
        results_df = analyze_results(results)
        
        # 5. 保存结果
        save_results(results_df)
        
        log.info("\n" + "="*80)
        log.success("✅ Walk-Forward验证完成！")
        log.info("="*80)
        
    except Exception as e:
        log.error(f"✗ Walk-Forward验证出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

