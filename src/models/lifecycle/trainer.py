"""
模型训练器
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.logger import log
from src.models.lifecycle.iterator import ModelIterator
from src.visualization.training_visualizer import TrainingVisualizer


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_name: str, config_path: str = None):
        self.model_name = model_name
        self.iterator = ModelIterator(model_name)
        
        # 加载配置
        if config_path is None:
            config_path = f"config/models/{model_name}.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置路径
        self.base_path = Path(f"data/models/{model_name}")
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def train_version(
        self,
        version: str = None,
        neg_version: str = 'v2'
    ):
        """训练指定版本"""
        if version is None:
            # 创建新版本
            existing_versions = self.iterator.list_versions()
            if existing_versions:
                latest = existing_versions[-1]
                # 递增版本号
                version = self._increment_version(latest)
            else:
                version = 'v1.0.0'
            
            self.iterator.create_version(version)
        else:
            # 如果指定了版本，检查是否存在，不存在则创建
            try:
                self.iterator.get_version_info(version)
                log.info(f"版本 {version} 已存在，将重新训练并覆盖")
            except ValueError:
                # 版本不存在，创建新版本
                self.iterator.create_version(version)
                log.info(f"创建新版本: {version}")
        
        log.info("="*80)
        log.info(f"训练模型: {self.model_name} 版本: {version}")
        log.info("="*80)
        
        # 1. 加载数据
        df = self._load_and_prepare_data(neg_version)
        
        # 2. 特征工程
        df_features = self._extract_features(df)
        
        # 3. 时间序列划分
        X_train, X_test, y_train, y_test, train_dates, test_dates = self._timeseries_split(df_features)
        
        # 4. 训练模型
        model, metrics = self._train_model(X_train, y_train, X_test, y_test)
        
        # 4.5. 生成可视化图表
        self._generate_visualizations(
            model, X_train, X_test, y_train, y_test,
            train_dates, test_dates, version
        )
        
        # 5. 保存模型（包含特征名称）
        feature_cols = list(X_train.columns)
        training_start_time = datetime.now()
        self._save_model(model, metrics, version, train_dates, test_dates, feature_cols)
        training_end_time = datetime.now()
        
        # 6. 更新版本元数据（包含完整信息）
        # 重新组织 metrics 结构
        metrics_structured = {
            'training': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'auc': metrics['auc']
            },
            'validation': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'auc': metrics['auc']
            },
            'test': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'auc': metrics['auc'],
                'confusion_matrix': metrics.get('confusion_matrix', [])
            }
        }
        
        # 获取模型配置信息
        model_config = self.config.get('model', {})
        display_name = model_config.get('display_name', self.model_name)
        description = model_config.get('description', '')
        
        # 更新版本元数据
        self.iterator.update_version_metadata(
            version,
            display_name=f"{display_name} {version}",
            description=description,
            config=self.config,
            metrics=metrics_structured,
            training={
                'started_at': training_start_time.isoformat(),
                'completed_at': training_end_time.isoformat(),
                'duration_seconds': int((training_end_time - training_start_time).total_seconds()),
                'samples': {
                    'train': len(X_train),
                    'test': len(X_test)
                },
                'hyperparameters': self.config.get('model_params', {}),
                'train_date_range': f"{train_dates.min().date()} to {train_dates.max().date()}",
                'test_date_range': f"{test_dates.min().date()} to {test_dates.max().date()}"
            }
        )
        
        log.success(f"✅ 模型训练完成！版本: {version}")
        return model, metrics
    
    def _load_and_prepare_data(self, neg_version='v2'):
        """加载并准备训练数据"""
        log.info("加载数据...")
        
        # 加载正样本
        df_pos = pd.read_csv('data/training/features/feature_data_34d.csv')
        df_pos['label'] = 1
        log.success(f"✓ 正样本: {len(df_pos)} 条")
        
        # 加载负样本（检查多个可能的位置）
        if neg_version == 'v2':
            # 先检查features目录
            if os.path.exists('data/training/features/negative_feature_data_v2_34d.csv'):
                neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
            elif os.path.exists('data/training/samples/negative_feature_data_v2_34d.csv'):
                neg_file = 'data/training/samples/negative_feature_data_v2_34d.csv'
            else:
                raise FileNotFoundError("未找到负样本特征文件 negative_feature_data_v2_34d.csv")
        else:
            if os.path.exists('data/training/features/negative_feature_data_34d.csv'):
                neg_file = 'data/training/features/negative_feature_data_34d.csv'
            elif os.path.exists('data/training/samples/negative_feature_data_34d.csv'):
                neg_file = 'data/training/samples/negative_feature_data_34d.csv'
            else:
                raise FileNotFoundError("未找到负样本特征文件 negative_feature_data_34d.csv")
        
        df_neg = pd.read_csv(neg_file)
        log.success(f"✓ 负样本: {len(df_neg)} 条")
        
        # 合并
        df = pd.concat([df_pos, df_neg])
        return df
    
    def _extract_features(self, df):
        """提取特征（复用现有逻辑）"""
        log.info("提取特征...")
        
        # 重新分配唯一的sample_id
        df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
        
        features = []
        sample_ids = df['unique_sample_id'].unique()
        
        # 获取T1日期映射
        df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
        t1_date_map = dict(zip(
            df_positive_samples.index,
            pd.to_datetime(df_positive_samples['t1_date'])
        ))
        
        if os.path.exists('data/training/samples/negative_samples_v2.csv'):
            df_negative_samples = pd.read_csv('data/training/samples/negative_samples_v2.csv')
        else:
            df_negative_samples = pd.read_csv('data/training/samples/negative_samples.csv')
        
        max_positive_id = df_positive_samples.index.max()
        for idx, row in df_negative_samples.iterrows():
            t1_date_map[max_positive_id + 1 + idx] = pd.to_datetime(row['t1_date'])
        
        for i, sample_id in enumerate(sample_ids):
            if (i + 1) % 500 == 0:
                log.info(f"进度: {i+1}/{len(sample_ids)}")
            
            sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')
            
            if len(sample_data) < 20:
                continue
            
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
            else:
                feature_dict['volume_ratio_mean'] = 0
                feature_dict['volume_ratio_max'] = 0
                feature_dict['volume_ratio_gt_2'] = 0
                feature_dict['volume_ratio_gt_4'] = 0
            
            # MACD特征
            if 'macd' in sample_data.columns:
                macd_data = sample_data['macd'].dropna()
                if len(macd_data) > 0:
                    feature_dict['macd_mean'] = macd_data.mean()
                    feature_dict['macd_positive_days'] = (macd_data > 0).sum()
                    feature_dict['macd_max'] = macd_data.max()
                else:
                    feature_dict['macd_mean'] = 0
                    feature_dict['macd_positive_days'] = 0
                    feature_dict['macd_max'] = 0
            else:
                feature_dict['macd_mean'] = 0
                feature_dict['macd_positive_days'] = 0
                feature_dict['macd_max'] = 0
            
            # MA特征
            if 'ma5' in sample_data.columns:
                feature_dict['ma5_mean'] = sample_data['ma5'].mean()
                feature_dict['price_above_ma5'] = (sample_data['close'] > sample_data['ma5']).sum()
            else:
                feature_dict['ma5_mean'] = 0
                feature_dict['price_above_ma5'] = 0
            
            if 'ma10' in sample_data.columns:
                feature_dict['ma10_mean'] = sample_data['ma10'].mean()
                feature_dict['price_above_ma10'] = (sample_data['close'] > sample_data['ma10']).sum()
            else:
                feature_dict['ma10_mean'] = 0
                feature_dict['price_above_ma10'] = 0
            
            # 市值特征
            if 'total_mv' in sample_data.columns:
                mv_data = sample_data['total_mv'].dropna()
                if len(mv_data) > 0:
                    feature_dict['total_mv_mean'] = mv_data.mean()
                else:
                    feature_dict['total_mv_mean'] = 0
            else:
                feature_dict['total_mv_mean'] = 0
            
            if 'circ_mv' in sample_data.columns:
                circ_mv_data = sample_data['circ_mv'].dropna()
                if len(circ_mv_data) > 0:
                    feature_dict['circ_mv_mean'] = circ_mv_data.mean()
                else:
                    feature_dict['circ_mv_mean'] = 0
            else:
                feature_dict['circ_mv_mean'] = 0
            
            # 动量特征（分段收益率）
            days = len(sample_data)
            if days >= 7:
                feature_dict['return_1w'] = (
                    (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-7]) /
                    sample_data['close'].iloc[-7] * 100
                )
            else:
                feature_dict['return_1w'] = 0
            
            if days >= 14:
                feature_dict['return_2w'] = (
                    (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-14]) /
                    sample_data['close'].iloc[-14] * 100
                )
            else:
                feature_dict['return_2w'] = 0
            
            features.append(feature_dict)
        
        df_features = pd.DataFrame(features)
        log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
        return df_features
    
    def _timeseries_split(self, df_features):
        """时间序列划分"""
        df_features['t1_date'] = pd.to_datetime(df_features['t1_date'])
        df_features = df_features.sort_values('t1_date').reset_index(drop=True)
        
        # 使用配置中的划分方式
        train_end_date = self.config.get('training', {}).get('train_end_date')
        test_start_date = self.config.get('training', {}).get('test_start_date')
        
        if train_end_date is None:
            n_train = int(len(df_features) * 0.8)
            train_end_date = df_features.iloc[n_train]['t1_date']
            test_start_date = df_features.iloc[n_train + 1]['t1_date']
        else:
            train_end_date = pd.to_datetime(train_end_date)
            test_start_date = pd.to_datetime(test_start_date)
        
        train_mask = df_features['t1_date'] <= train_end_date
        test_mask = df_features['t1_date'] >= test_start_date
        
        df_train = df_features[train_mask]
        df_test = df_features[test_mask]
        
        log.info(f"训练集: {len(df_train)} 个样本")
        log.info(f"测试集: {len(df_test)} 个样本")
        
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
            X_train = X_train.drop(columns=non_numeric_cols)
            X_test = X_test.drop(columns=non_numeric_cols)
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def _train_model(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        log.info("训练模型...")
        
        model_params = self.config.get('model_params', {})
        model = xgb.XGBClassifier(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 评估
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, target_names=['负样本', '正样本'], output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['正样本']['precision'],
            'recall': report['正样本']['recall'],
            'f1_score': report['正样本']['f1-score'],
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
        
        log.success("✓ 模型训练完成")
        log.info(f"准确率: {metrics['accuracy']:.2%}")
        log.info(f"AUC: {metrics['auc']:.4f}")
        
        return model, metrics
    
    def _save_model(self, model, metrics, version, train_dates, test_dates, feature_cols=None):
        """保存模型"""
        version_path = self.iterator.versions_path / version
        model_path = version_path / "model" / "model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.get_booster().save_model(str(model_path))
        log.success(f"✓ 模型已保存: {model_path}")
        
        # 保存特征名称
        if feature_cols is not None:
            feature_names_file = version_path / "model" / "feature_names.json"
            with open(feature_names_file, 'w', encoding='utf-8') as f:
                json.dump(feature_cols, f, indent=2, ensure_ascii=False)
            log.success(f"✓ 特征名称已保存: {feature_names_file}")
        
        # 保存指标
        metrics_file = version_path / "training" / "metrics.json"
        metrics['model_file'] = str(model_path)
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['train_date_range'] = f"{train_dates.min().date()} to {train_dates.max().date()}"
        metrics['test_date_range'] = f"{test_dates.min().date()} to {test_dates.max().date()}"
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        log.success(f"✓ 指标已保存: {metrics_file}")
    
    def _generate_visualizations(
        self, model, X_train, X_test, y_train, y_test,
        train_dates, test_dates, version
    ):
        """生成训练过程可视化图表"""
        try:
            log.info("="*80)
            log.info("生成训练可视化图表")
            log.info("="*80)
            
            visualizer = TrainingVisualizer(
                output_dir=f"data/models/{self.model_name}/versions/{version}/charts"
            )
            
            # 1. 样本质量可视化（正样本和负样本）
            df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
            visualizer.visualize_sample_quality(
                df_positive_samples, 
                save_prefix="positive_sample_quality"
            )
            
            # 负样本
            if os.path.exists('data/training/samples/negative_samples_v2.csv'):
                df_negative_samples = pd.read_csv('data/training/samples/negative_samples_v2.csv')
                visualizer.visualize_sample_quality(
                    df_negative_samples,
                    save_prefix="negative_sample_quality"
                )
            
            # 2. 特征质量评估可视化
            visualizer.visualize_feature_quality(
                X_train, y_train, X_test, y_test,
                model_name=f"{self.model_name}_{version}"
            )
            
            # 3. 因子重要性可视化
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            
            visualizer.visualize_feature_importance(
                feature_importance,
                model_name=f"{self.model_name}_{version}",
                top_n=20
            )
            
            # 4. 模型训练过程可视化
            visualizer.visualize_training_process(
                model, X_train, y_train, X_test, y_test,
                model_name=f"{self.model_name}_{version}"
            )
            
            # 5. 模型结果评测可视化
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            visualizer.visualize_model_results(
                y_test, y_pred, y_prob,
                model_name=f"{self.model_name}_{version}"
            )
            
            # 6. 生成索引页面
            visualizer.generate_index_page(model_name=f"{self.model_name}_{version}")
            
            log.success("✓ 可视化图表生成完成")
            log.info(f"📊 查看图表: open data/models/{self.model_name}/versions/{version}/charts/index.html")
            
        except Exception as e:
            log.warning(f"生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _increment_version(self, version: str) -> str:
        """递增版本号"""
        version = version.lstrip('v')
        parts = version.split('.')
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
        return 'v' + '.'.join(parts)

