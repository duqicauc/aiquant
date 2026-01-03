#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练v2.3.0模型

特点：
1. 包含风险特征（最大回撤、ATR、回撤恢复）
2. 概率校准
3. 防过拟合参数
4. 硬负样本
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log


def load_training_data():
    """加载带风险特征的训练数据"""
    log.info("加载训练数据...")
    
    # 正样本（带风险特征）
    pos_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3.csv'
    df_pos = pd.read_csv(pos_file)
    df_pos['label'] = 1
    log.info(f"  正样本: {len(df_pos)} 条")
    
    # 负样本
    neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v3.csv'
    df_neg = pd.read_csv(neg_file)
    df_neg['label'] = 0
    log.info(f"  普通负样本: {len(df_neg)} 条")
    
    # 硬负样本
    hard_neg_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v3.csv'
    if hard_neg_file.exists():
        df_hard_neg = pd.read_csv(hard_neg_file)
        df_hard_neg['label'] = 0
        log.info(f"  硬负样本: {len(df_hard_neg)} 条")
        df_neg = pd.concat([df_neg, df_hard_neg], ignore_index=True)
    
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    log.success(f"✓ 数据加载完成: {len(df)} 条")
    
    return df


def get_feature_columns(df):
    """获取特征列（只保留数值列）"""
    exclude_cols = ['ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label', 
                    'trade_date', 'weekly_return_1', 'weekly_return_2', 'weekly_return_3', 
                    'total_return_34d', 'weekly_volume_1', 'weekly_volume_2', 'weekly_volume_3']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    return numeric_cols


def train_model(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    log.info("训练模型...")
    
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
    
    booster = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    log.success(f"✓ 模型训练完成, best_iteration: {booster.best_iteration}")
    return booster


def calibrate_model(booster, X_cal, y_cal, feature_names):
    """概率校准"""
    log.info("概率校准...")
    
    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_probs = booster.predict(dcal)
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_cal)
    
    cal_probs = calibrator.predict(raw_probs)
    log.info(f"  校准前: mean={raw_probs.mean():.4f}, max={raw_probs.max():.4f}")
    log.info(f"  校准后: mean={cal_probs.mean():.4f}, max={cal_probs.max():.4f}")
    
    log.success("✓ 概率校准完成")
    return calibrator


def evaluate(booster, calibrator, X_test, y_test, feature_names):
    """评估模型"""
    log.info("评估模型...")
    
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)
    
    # 评估
    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        cal_high = cal_probs >= thresh
        acc = y_test[cal_high].mean() if cal_high.sum() > 0 else 0
        log.info(f"  校准概率>={thresh}: {cal_high.sum()}个, 真实正确率{acc:.1%}")
    
    return {
        'test_samples': len(X_test),
        'positive_samples': y_test.sum(),
        'auc': None  # 可以添加AUC计算
    }


def save_model(booster, calibrator, feature_names, metrics):
    """保存模型"""
    version = 'v2.3.0'
    log.info(f"保存模型 {version}...")
    
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'breakout_launch_scorer' / 'versions' / version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型
    (model_dir / 'model').mkdir(exist_ok=True)
    booster.save_model(str(model_dir / 'model' / 'model.json'))
    
    # 特征名
    with open(model_dir / 'model' / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # 校准器
    joblib.dump(calibrator, str(model_dir / 'model' / 'calibrator.pkl'))
    
    # 元数据
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'calibration_method': 'isotonic_regression',
        'risk_features': [
            'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_55d',
            'atr_14', 'atr_ratio_14', 'atr_expansion',
            'days_from_high_20d', 'days_from_high_55d', 'recovery_ratio_20d'
        ],
        'description': '带风险特征+概率校准的优化模型'
    }
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 模型已保存到 {model_dir}")


def main():
    log.info("="*80)
    log.info("训练v2.3.0模型（含风险特征+概率校准）")
    log.info("="*80)
    
    # 加载数据
    df = load_training_data()
    
    # 特征
    feature_cols = get_feature_columns(df)
    log.info(f"特征数: {len(feature_cols)}")
    
    # 显示新增的风险特征
    risk_features = [f for f in feature_cols if any(k in f for k in ['drawdown', 'atr', 'days_from_high', 'recovery'])]
    log.info(f"风险特征: {risk_features}")
    
    X = df[feature_cols].values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 划分数据集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    
    log.info(f"\n数据集: 训练{len(X_train)}, 校准{len(X_cal)}, 测试{len(X_test)}")
    
    # 训练
    booster = train_model(X_train, y_train, X_cal, y_cal)
    
    # 校准
    calibrator = calibrate_model(booster, X_cal, y_cal, feature_cols)
    
    # 评估
    metrics = evaluate(booster, calibrator, X_test, y_test, feature_cols)
    
    # 保存
    save_model(booster, calibrator, feature_cols, metrics)
    
    log.success("\n✓ v2.3.0模型训练完成!")


if __name__ == '__main__':
    main()

