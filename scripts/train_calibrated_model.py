#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练带概率校准的模型 v2.2.0

优化内容：
1. 概率校准 - 使用Isotonic Regression校准过于自信的概率输出
2. 风险过滤 - 集成风险评分到预测流程
3. 保持之前的防过拟合参数和硬负样本

目标：解决模型输出概率过于自信（Top50全在94%+）但实际胜率只有44%的问题
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import joblib

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log


def load_training_data():
    """加载训练数据（使用高级特征）"""
    log.info("加载训练数据...")
    
    # 正样本
    pos_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_advanced.csv'
    df_pos = pd.read_csv(pos_file)
    df_pos['label'] = 1
    log.info(f"  正样本: {len(df_pos)} 条")
    
    # 负样本
    neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_advanced.csv'
    df_neg = pd.read_csv(neg_file)
    df_neg['label'] = 0
    log.info(f"  普通负样本: {len(df_neg)} 条")
    
    # 硬负样本
    hard_neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_advanced.csv'
    if hard_neg_file.exists():
        df_hard_neg = pd.read_csv(hard_neg_file)
        df_hard_neg['label'] = 0
        log.info(f"  硬负样本: {len(df_hard_neg)} 条")
        df_neg = pd.concat([df_neg, df_hard_neg], ignore_index=True)
    
    # 合并
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    log.success(f"✓ 数据加载完成: {len(df)} 条 (正:{len(df_pos)}, 负:{len(df_neg)})")
    
    return df


def get_feature_columns(df):
    """获取特征列"""
    exclude_cols = ['ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label', 
                    'trade_date', 'weekly_return_1', 'weekly_return_2', 'weekly_return_3', 
                    'total_return_34d', 'weekly_volume_1', 'weekly_volume_2', 'weekly_volume_3']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 确保只有数值列
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    return numeric_cols


def train_base_model(X_train, y_train, X_val, y_val):
    """训练基础XGBoost模型（防过拟合参数）"""
    log.info("训练基础模型...")
    
    # 防过拟合参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'min_child_weight': 10,
        'gamma': 0.3,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'scale_pos_weight': 1.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=watchlist,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    log.success(f"✓ 基础模型训练完成, best_iteration: {booster.best_iteration}")
    
    return booster


def calibrate_probabilities(booster, X_cal, y_cal, feature_names):
    """
    使用Isotonic Regression校准概率
    
    原理：
    - 模型输出的概率可能不准确（如94%概率但只有44%正确）
    - Isotonic Regression学习一个单调映射函数
    - 将模型输出概率映射到真实概率
    """
    log.info("进行概率校准...")
    
    # 获取未校准的概率
    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_probs = booster.predict(dcal)
    
    # 分析校准前的概率分布
    log.info(f"  校准前概率分布:")
    log.info(f"    均值: {raw_probs.mean():.4f}")
    log.info(f"    中位数: {np.median(raw_probs):.4f}")
    log.info(f"    最大值: {raw_probs.max():.4f}")
    log.info(f"    最小值: {raw_probs.min():.4f}")
    
    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(y_cal, raw_probs, n_bins=10, strategy='uniform')
    
    log.info(f"  校准曲线 (预测概率 vs 真实概率):")
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        log.info(f"    bin {i}: 预测{pp:.3f} → 真实{pt:.3f}")
    
    # 训练Isotonic Regression校准器
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_cal)
    
    # 校准后的概率
    calibrated_probs = calibrator.predict(raw_probs)
    
    log.info(f"  校准后概率分布:")
    log.info(f"    均值: {calibrated_probs.mean():.4f}")
    log.info(f"    中位数: {np.median(calibrated_probs):.4f}")
    log.info(f"    最大值: {calibrated_probs.max():.4f}")
    log.info(f"    最小值: {calibrated_probs.min():.4f}")
    
    # 计算校准后的曲线
    prob_true_cal, prob_pred_cal = calibration_curve(y_cal, calibrated_probs, n_bins=10, strategy='uniform')
    
    log.info(f"  校准后曲线:")
    for i, (pt, pp) in enumerate(zip(prob_true_cal, prob_pred_cal)):
        log.info(f"    bin {i}: 预测{pp:.3f} → 真实{pt:.3f}")
    
    log.success("✓ 概率校准完成")
    
    return calibrator


def evaluate_model(booster, calibrator, X_test, y_test, feature_names):
    """评估模型"""
    log.info("评估模型...")
    
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    
    # 原始概率
    raw_probs = booster.predict(dtest)
    
    # 校准后概率
    cal_probs = calibrator.predict(raw_probs)
    
    # 使用0.5阈值评估
    raw_preds = (raw_probs >= 0.5).astype(int)
    cal_preds = (cal_probs >= 0.5).astype(int)
    
    # 原始模型指标
    raw_acc = (raw_preds == y_test).mean()
    raw_precision = y_test[raw_preds == 1].mean() if raw_preds.sum() > 0 else 0
    raw_recall = raw_preds[y_test == 1].mean() if y_test.sum() > 0 else 0
    
    # 校准后指标
    cal_acc = (cal_preds == y_test).mean()
    cal_precision = y_test[cal_preds == 1].mean() if cal_preds.sum() > 0 else 0
    cal_recall = cal_preds[y_test == 1].mean() if y_test.sum() > 0 else 0
    
    log.info(f"\n原始模型 vs 校准后模型:")
    log.info(f"  | 指标 | 原始 | 校准后 |")
    log.info(f"  |------|------|--------|")
    log.info(f"  | 准确率 | {raw_acc:.4f} | {cal_acc:.4f} |")
    log.info(f"  | 精确率 | {raw_precision:.4f} | {cal_precision:.4f} |")
    log.info(f"  | 召回率 | {raw_recall:.4f} | {cal_recall:.4f} |")
    log.info(f"  | 预测正例数 | {raw_preds.sum()} | {cal_preds.sum()} |")
    
    # 高概率样本分析
    log.info(f"\n高概率样本分析:")
    
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        raw_high = raw_probs >= thresh
        cal_high = cal_probs >= thresh
        
        raw_acc_high = y_test[raw_high].mean() if raw_high.sum() > 0 else 0
        cal_acc_high = y_test[cal_high].mean() if cal_high.sum() > 0 else 0
        
        log.info(f"  概率>={thresh}: 原始{raw_high.sum()}个(真实{raw_acc_high:.1%}) vs 校准后{cal_high.sum()}个(真实{cal_acc_high:.1%})")
    
    return {
        'raw_accuracy': raw_acc,
        'raw_precision': raw_precision,
        'raw_recall': raw_recall,
        'calibrated_accuracy': cal_acc,
        'calibrated_precision': cal_precision,
        'calibrated_recall': cal_recall
    }


def save_model(booster, calibrator, feature_names, metrics, version='v2.2.0'):
    """保存模型"""
    log.info(f"保存模型 {version}...")
    
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存XGBoost模型
    model_path = model_dir / 'model' / 'model.json'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    
    # 保存特征名
    feature_path = model_dir / 'model' / 'feature_names.json'
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # 保存校准器
    calibrator_path = model_dir / 'model' / 'calibrator.pkl'
    joblib.dump(calibrator, str(calibrator_path))
    
    # 保存指标
    metrics_path = model_dir / 'training' / 'metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存元数据
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'calibration_method': 'isotonic_regression',
        'risk_filter_threshold': 0.7,
        'description': '带概率校准和风险过滤的优化模型'
    }
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("="*80)
    log.info("训练带概率校准的模型 v2.2.0")
    log.info("="*80)
    
    # 1. 加载数据
    df = load_training_data()
    
    # 2. 准备特征
    feature_cols = get_feature_columns(df)
    log.info(f"特征数: {len(feature_cols)}")
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. 划分数据集
    # 三分：训练集、校准集、测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    
    log.info(f"\n数据集划分:")
    log.info(f"  训练集: {len(X_train)} (正:{y_train.sum()}, 负:{len(y_train)-y_train.sum()})")
    log.info(f"  校准集: {len(X_cal)} (正:{y_cal.sum()}, 负:{len(y_cal)-y_cal.sum()})")
    log.info(f"  测试集: {len(X_test)} (正:{y_test.sum()}, 负:{len(y_test)-y_test.sum()})")
    
    # 4. 训练基础模型
    booster = train_base_model(X_train, y_train, X_cal, y_cal)
    
    # 5. 概率校准
    calibrator = calibrate_probabilities(booster, X_cal, y_cal, feature_cols)
    
    # 6. 评估
    metrics = evaluate_model(booster, calibrator, X_test, y_test, feature_cols)
    
    # 7. 保存模型
    save_model(booster, calibrator, feature_cols, metrics, version='v2.2.0')
    
    log.info("\n" + "="*80)
    log.info("训练完成!")
    log.info("="*80)
    
    log.info("""
后续使用方式：

1. 加载模型和校准器：
   booster = xgb.Booster()
   booster.load_model('model.json')
   calibrator = joblib.load('calibrator.pkl')

2. 预测时：
   raw_prob = booster.predict(dmatrix)      # 原始概率
   calibrated_prob = calibrator.predict(raw_prob)  # 校准后概率

3. 结合风险过滤（阈值0.7）：
   final_score = calibrated_prob * risk_score
   selected = final_score[risk_score >= 0.7]
""")


if __name__ == '__main__':
    main()
