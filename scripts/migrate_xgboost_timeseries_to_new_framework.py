#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 xgboost_timeseries_v2_20251225_205905.json 模型迁移到新框架
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
from src.models.lifecycle.iterator import ModelIterator
import xgboost as xgb


def extract_feature_names_from_model(model_path):
    """从模型文件中提取特征名称"""
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        
        # XGBoost模型的特征名称存储在feature_names属性中
        if hasattr(booster, 'feature_names'):
            return booster.feature_names
        
        # 如果没有feature_names，尝试从feature_names_属性获取
        if hasattr(booster, 'feature_names_'):
            return booster.feature_names_
        
        # 如果都没有，从metrics文件中获取
        log.warning("模型文件中没有特征名称，尝试从metrics文件获取")
        return None
    except Exception as e:
        log.warning(f"从模型提取特征名称失败: {e}")
        return None


def extract_feature_names_from_metrics(metrics_file):
    """从metrics文件中提取特征名称"""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        if 'feature_importance' in metrics:
            feature_names = [item['feature'] for item in metrics['feature_importance']]
            return feature_names
        
        return None
    except Exception as e:
        log.warning(f"从metrics文件提取特征名称失败: {e}")
        return None


def migrate_model():
    """迁移模型到新框架"""
    log.info("="*80)
    log.info("迁移 xgboost_timeseries_v2_20251225_205905.json 到新框架")
    log.info("="*80)
    log.info("")
    
    # 1. 定义源文件路径
    old_model_path = Path('data/training/models/xgboost_timeseries_v2_20251225_205905.json')
    old_metrics_path = Path('data/training/metrics/xgboost_timeseries_v2_metrics.json')
    old_charts_dir = Path('data/training/charts')
    
    # 检查源文件是否存在
    if not old_model_path.exists():
        log.error(f"模型文件不存在: {old_model_path}")
        return False
    
    if not old_metrics_path.exists():
        log.error(f"指标文件不存在: {old_metrics_path}")
        return False
    
    log.success("✓ 源文件检查完成")
    log.info("")
    
    # 2. 读取metrics文件
    with open(old_metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    log.info("从metrics文件读取的信息:")
    log.info(f"  准确率: {metrics.get('accuracy', 'N/A'):.4f}")
    log.info(f"  精确率: {metrics.get('precision', 'N/A'):.4f}")
    log.info(f"  召回率: {metrics.get('recall', 'N/A'):.4f}")
    log.info(f"  F1分数: {metrics.get('f1_score', 'N/A'):.4f}")
    log.info(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
    log.info(f"  训练日期范围: {metrics.get('train_date_range', 'N/A')}")
    log.info(f"  测试日期范围: {metrics.get('test_date_range', 'N/A')}")
    log.info("")
    
    # 3. 提取特征名称
    log.info("提取特征名称...")
    feature_names = extract_feature_names_from_model(old_model_path)
    
    if feature_names is None:
        log.warning("从模型文件无法提取特征名称，尝试从metrics文件提取...")
        feature_names = extract_feature_names_from_metrics(old_metrics_path)
    
    if feature_names is None:
        log.error("无法提取特征名称，迁移失败")
        return False
    
    log.success(f"✓ 提取到 {len(feature_names)} 个特征名称")
    log.info("")
    
    # 4. 创建新框架版本
    model_name = 'breakout_launch_scorer'
    version = 'v1.0.0-legacy'  # 使用legacy标识这是迁移的旧模型
    
    iterator = ModelIterator(model_name)
    
    # 检查版本是否已存在
    if version in iterator.list_versions():
        log.warning(f"版本 {version} 已存在，将覆盖")
        # 删除旧版本
        version_path = iterator.versions_path / version
        if version_path.exists():
            shutil.rmtree(version_path)
    
    # 创建新版本
    iterator.create_version(version)
    version_path = iterator.versions_path / version
    
    log.success(f"✓ 创建版本: {version}")
    log.info("")
    
    # 5. 复制模型文件
    log.info("复制模型文件...")
    model_dir = version_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    new_model_path = model_dir / "model.json"
    shutil.copy2(old_model_path, new_model_path)
    log.success(f"✓ 模型文件已复制: {new_model_path}")
    
    # 6. 保存特征名称
    feature_names_file = model_dir / "feature_names.json"
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 特征名称已保存: {feature_names_file}")
    log.info("")
    
    # 7. 复制charts文件（如果存在）
    if old_charts_dir.exists():
        log.info("复制charts文件...")
        new_charts_dir = version_path / "charts"
        if new_charts_dir.exists():
            shutil.rmtree(new_charts_dir)
        shutil.copytree(old_charts_dir, new_charts_dir)
        log.success(f"✓ Charts文件已复制: {new_charts_dir}")
        log.info("")
    
    # 8. 创建metadata.json
    log.info("创建metadata.json...")
    
    # 从metrics构建metadata结构
    metadata = {
        "version": version,
        "model_name": model_name,
        "status": "production",  # 标记为生产版本
        "created_at": metrics.get('timestamp', datetime.now().isoformat()),
        "created_by": "migration_script",
        "parent_version": None,
        "migration_source": {
            "original_model": str(old_model_path),
            "original_metrics": str(old_metrics_path),
            "migration_date": datetime.now().isoformat(),
            "note": "从 xgboost_timeseries_v2_20251225_205905.json 迁移"
        },
        "metrics": {
            "training": {
                "accuracy": metrics.get('accuracy', 0),
                "precision": metrics.get('precision', 0),
                "recall": metrics.get('recall', 0),
                "f1": metrics.get('f1_score', 0),
                "auc": metrics.get('auc', 0)
            },
            "validation": {
                "accuracy": metrics.get('accuracy', 0),
                "precision": metrics.get('precision', 0),
                "recall": metrics.get('recall', 0),
                "f1": metrics.get('f1_score', 0),
                "auc": metrics.get('auc', 0)
            },
            "test": {
                "accuracy": metrics.get('accuracy', 0),
                "precision": metrics.get('precision', 0),
                "recall": metrics.get('recall', 0),
                "f1": metrics.get('f1_score', 0),
                "auc": metrics.get('auc', 0),
                "confusion_matrix": metrics.get('confusion_matrix', [])
            }
        },
        "changes": {
            "note": "从旧框架迁移到新框架，保持模型和指标不变"
        },
        "notes": f"从 xgboost_timeseries_v2_20251225_205905.json 迁移。{metrics.get('note', '')}",
        "display_name": f"突破起爆评分模型 {version} (Legacy)",
        "description": "从旧框架 xgboost_timeseries 迁移的模型，训练数据范围: " + metrics.get('train_date_range', 'N/A'),
        "config": {
            "model": {
                "name": model_name,
                "display_name": "突破起爆评分模型",
                "description": "基于技术指标识别股票起爆点，预测未来3周强势上涨概率的XGBoost选股模型",
                "type": "xgboost",
                "version": "v1.0.0",
                "status": "production"
            },
            "model_params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42
            },
            "training": {
                "validation_split": 0.2,
                "time_series_split": True
            }
        },
        "training": {
            "started_at": metrics.get('timestamp', datetime.now().isoformat()),
            "completed_at": metrics.get('timestamp', datetime.now().isoformat()),
            "duration_seconds": 0,
            "samples": {
                "train": 0,  # metrics中没有这个信息
                "test": 0
            },
            "hyperparameters": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42
            },
            "train_date_range": metrics.get('train_date_range', 'N/A'),
            "test_date_range": metrics.get('test_date_range', 'N/A')
        }
    }
    
    # 添加特征重要性
    if 'feature_importance' in metrics:
        metadata['feature_importance'] = metrics['feature_importance']
    
    metadata_file = version_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ Metadata已创建: {metadata_file}")
    log.info("")
    
    # 9. 验证迁移
    log.info("="*80)
    log.info("验证迁移结果")
    log.info("="*80)
    
    # 验证模型文件
    if new_model_path.exists():
        log.success(f"✓ 模型文件存在: {new_model_path}")
        
        # 尝试加载模型
        try:
            booster = xgb.Booster()
            booster.load_model(str(new_model_path))
            log.success("✓ 模型可以正常加载")
        except Exception as e:
            log.error(f"✗ 模型加载失败: {e}")
            return False
    else:
        log.error(f"✗ 模型文件不存在: {new_model_path}")
        return False
    
    # 验证特征名称文件
    if feature_names_file.exists():
        log.success(f"✓ 特征名称文件存在: {feature_names_file}")
        with open(feature_names_file, 'r', encoding='utf-8') as f:
            loaded_features = json.load(f)
        log.info(f"  特征数量: {len(loaded_features)}")
    else:
        log.error(f"✗ 特征名称文件不存在: {feature_names_file}")
        return False
    
    # 验证metadata文件
    if metadata_file.exists():
        log.success(f"✓ Metadata文件存在: {metadata_file}")
    else:
        log.error(f"✗ Metadata文件不存在: {metadata_file}")
        return False
    
    log.info("")
    log.info("="*80)
    log.success("✅ 模型迁移完成！")
    log.info("="*80)
    log.info("")
    log.info("迁移信息:")
    log.info(f"  旧模型: {old_model_path}")
    log.info(f"  新版本: {version}")
    log.info(f"  新路径: {version_path}")
    log.info("")
    log.info("使用方法:")
    log.info(f"  python scripts/score_current_stocks.py --version {version}")
    log.info("")
    
    return True


if __name__ == '__main__':
    try:
        success = migrate_model()
        if success:
            log.info("迁移成功完成！")
        else:
            log.error("迁移失败，请检查日志")
            sys.exit(1)
    except Exception as e:
        log.error(f"迁移过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

