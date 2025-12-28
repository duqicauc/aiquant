"""
左侧潜力牛股模型核心类

整合样本筛选、特征工程、模型训练和预测的完整流程
"""
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from src.utils.logger import log
from .left_positive_screener import LeftPositiveSampleScreener
from .left_negative_screener import LeftNegativeSampleScreener
from .left_feature_engineering import LeftBreakoutFeatureEngineering


class LeftBreakoutModel:
    """左侧潜力牛股模型"""

    def __init__(self, data_manager, config: Dict = None):
        """
        初始化左侧模型

        Args:
            data_manager: 数据管理器实例
            config: 模型配置字典
        """
        self.dm = data_manager
        self.config = config or self._get_default_config()

        # 初始化组件（传递配置）
        self.positive_screener = LeftPositiveSampleScreener(data_manager, self.config)
        self.negative_screener = LeftNegativeSampleScreener(data_manager, self.config)
        self.feature_engineer = LeftBreakoutFeatureEngineering()

        # 模型和数据
        self.model = None
        self.feature_columns = []
        self.model_metrics = {}

        # 路径配置
        self.model_dir = self.config.get('save', {}).get('directory', 'data/models/left_breakout')
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'model': {
                'type': 'xgboost',
                'version': 'v1',
                'parameters': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'training': {
                    'test_size': 0.2,
                    'time_series_split': True,
                    'n_splits': 5
                },
                'save': {
                    'directory': 'data/models/left_breakout',
                    'auto_backup': True
                }
            },
            'sample_preparation': {
                'start_date': '20000101',
                'end_date': None,
                'lookback_days': 34,
                'look_forward_days': 45,
                'look_forward_days': 45
            }
        }

    def prepare_samples(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备正负样本数据

        Args:
            force_refresh: 是否强制重新生成样本

        Returns:
            (正样本DataFrame, 负样本DataFrame)
        """
        log.info("开始准备左侧潜力牛股样本...")

        # 检查是否已有缓存的样本数据
        positive_file = 'data/training/samples/left_positive_samples.csv'
        negative_file = 'data/training/samples/left_negative_samples.csv'

        if not force_refresh and os.path.exists(positive_file) and os.path.exists(negative_file):
            log.info("发现缓存的样本数据，正在加载...")
            try:
                positive_samples = pd.read_csv(positive_file)
                negative_samples = pd.read_csv(negative_file)
                log.info(f"加载完成：{len(positive_samples)} 个正样本，{len(negative_samples)} 个负样本")
                return positive_samples, negative_samples
            except Exception as e:
                log.warning(f"加载缓存样本失败: {e}")

        # 重新生成样本
        log.info("重新生成样本数据...")

        # 1. 生成正样本
        positive_samples = self.positive_screener.screen_all_stocks(
            start_date=self.config['sample_preparation']['start_date'],
            end_date=self.config['sample_preparation']['end_date'],
            look_forward_days=self.config['sample_preparation']['look_forward_days']
        )

        if positive_samples.empty:
            log.error("未找到任何正样本")
            return pd.DataFrame(), pd.DataFrame()

        log.info(f"生成正样本: {len(positive_samples)} 个")

        # 2. 生成负样本
        negative_samples = self.negative_screener.screen_negative_samples(
            positive_samples=positive_samples,
            start_date=self.config['sample_preparation']['start_date'],
            end_date=self.config['sample_preparation']['end_date'],
            look_forward_days=self.config['sample_preparation']['look_forward_days']
        )

        log.info(f"生成负样本: {len(negative_samples)} 个")

        # 3. 保存样本数据
        os.makedirs('data/training/samples', exist_ok=True)
        positive_samples.to_csv(positive_file, index=False)
        negative_samples.to_csv(negative_file, index=False)

        log.info("样本数据保存完成")

        return positive_samples, negative_samples

    def extract_features(self, positive_samples: pd.DataFrame, negative_samples: pd.DataFrame) -> pd.DataFrame:
        """
        从样本数据中提取特征

        Args:
            positive_samples: 正样本DataFrame
            negative_samples: 负样本DataFrame

        Returns:
            特征DataFrame
        """
        log.info("开始特征提取...")

        # 合并正负样本
        # 确保正样本有label字段（值为1），负样本有label字段（值为0）
        if 'label' not in positive_samples.columns:
            positive_samples['label'] = 1
        else:
            positive_samples['label'] = 1  # 确保正样本标签为1
        
        if 'label' not in negative_samples.columns:
            negative_samples['label'] = 0
        else:
            negative_samples['label'] = 0  # 确保负样本标签为0
        
        all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
        all_samples['unique_sample_id'] = range(len(all_samples))

        log.info(f"总样本数: {len(all_samples)} (正样本: {len(positive_samples)}, 负样本: {len(negative_samples)})")

        # 为每个样本提取34天特征数据
        feature_data_list = []

        failed_count = 0
        success_count = 0
        
        for idx, sample in all_samples.iterrows():
            if idx % 50 == 0:
                log.info(f"处理样本 {idx + 1}/{len(all_samples)} (成功: {success_count}, 失败: {failed_count})")

            try:
                # 获取该样本的34天原始数据
                sample_features = self._extract_single_sample_raw_data(sample)
                if not sample_features.empty:
                    feature_data_list.append(sample_features)
                    success_count += 1
                else:
                    failed_count += 1
                    log.debug(f"样本 {sample['ts_code']} {sample['t0_date']} 返回空数据，跳过")

            except Exception as e:
                failed_count += 1
                error_msg = str(e)[:200] if str(e) else type(e).__name__
                log.warning(f"样本 {sample['ts_code']} {sample['t0_date']} 特征提取失败: {error_msg}")
                # 如果是API错误，等待一下再继续
                if "API" in error_msg or "ERROR" in error_msg or "调用失败" in error_msg:
                    import time
                    time.sleep(1)  # 短暂等待，避免连续失败
                continue
        
        log.info(f"特征提取完成: 成功 {success_count} 个，失败 {failed_count} 个")

        if not feature_data_list:
            log.error("没有成功提取到任何特征数据")
            return pd.DataFrame()

        # 合并所有样本的特征数据
        raw_feature_data = pd.concat(feature_data_list, ignore_index=True)
        log.info(f"原始特征数据行数: {len(raw_feature_data)}")

        # 使用特征工程器提取统计特征
        final_features = self.feature_engineer.extract_features(raw_feature_data)

        # 保存特征数据
        os.makedirs('data/training/features', exist_ok=True)
        final_features.to_csv('data/training/features/left_breakout_features.csv', index=False)

        log.info(f"特征提取完成，最终特征维度: {len(final_features)} 行 × {len(final_features.columns)} 列")

        return final_features

    def _extract_single_sample_raw_data(self, sample) -> pd.DataFrame:
        """
        提取单个样本的34天原始数据

        Args:
            sample: 样本记录

        Returns:
            34天原始数据的DataFrame
        """
        ts_code = sample['ts_code']
        t0_date = sample['t0_date']

        try:
            # 计算数据获取的时间范围（T0前34天到T0）
            end_date = str(t0_date)  # 确保是字符串格式

            # 计算大约34个交易日对应的日历天数（大约45-50天）
            import datetime
            end_dt = datetime.datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - datetime.timedelta(days=60)  # 多取一些天数以确保有足够交易日
            start_date = start_dt.strftime('%Y%m%d')

            # 获取交易日历（带重试）
            try:
                calendar_df = self.dm.get_trade_calendar(start_date, end_date)
            except Exception as e:
                log.debug(f"获取交易日历失败 {ts_code} {t0_date}: {e}")
                return pd.DataFrame()
                
            if calendar_df.empty:
                return pd.DataFrame()

            # 筛选交易日
            trading_days = calendar_df[calendar_df['is_open'] == 1]['cal_date'].sort_values().tolist()
            if len(trading_days) < 20:  # 最少需要20天数据
                return pd.DataFrame()

            # 取最近的34个交易日
            recent_trading_days = trading_days[-34:] if len(trading_days) >= 34 else trading_days
            start_date = recent_trading_days[0]

            # 获取日线数据和技术指标（带错误处理）
            try:
                df = self.dm.get_complete_data(ts_code, start_date, end_date)
            except Exception as e:
                error_msg = str(e)[:200] if str(e) else type(e).__name__
                log.debug(f"获取日线数据失败 {ts_code} {t0_date}: {error_msg}")
                return pd.DataFrame()
                
            if df.empty or len(df) < 20:
                return pd.DataFrame()

            # 获取技术因子数据（可选，失败不影响）
            try:
                df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
                if not df_factor.empty:
                    df = pd.merge(df, df_factor, on='trade_date', how='left')
            except Exception as e:
                log.debug(f"获取技术因子失败 {ts_code} {t0_date}，继续使用基础数据: {e}")

            # 添加样本标识和标签
            df['unique_sample_id'] = sample['unique_sample_id']
            df['ts_code'] = ts_code
            df['name'] = sample['name']
            df['t0_date'] = t0_date
            df['label'] = sample['label']

            # 添加days_to_t1字段（距离T0的天数）
            df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
            # 修复：t0_date可能是int类型，需要先转换为字符串再解析
            t0_dt = pd.to_datetime(str(t0_date), format='%Y%m%d')
            df['days_to_t1'] = (df['trade_date_dt'] - t0_dt).dt.days

            # 只保留T0前的34天数据
            df = df[df['days_to_t1'] <= 0].tail(34).reset_index(drop=True)

            return df

        except Exception as e:
            error_msg = str(e)[:200] if str(e) else type(e).__name__
            log.debug(f"提取样本 {ts_code} {t0_date} 原始数据失败: {error_msg}")
            return pd.DataFrame()

    def train_model(self, features_df: pd.DataFrame) -> Dict:
        """
        训练XGBoost模型

        Args:
            features_df: 特征DataFrame

        Returns:
            训练结果字典
        """
        log.info("开始训练左侧潜力牛股模型...")

        if features_df.empty:
            log.error("特征数据为空，无法训练模型")
            return {}

        # 准备训练数据
        # 先清理掉label为NaN的样本
        features_df = features_df.dropna(subset=['label'])
        if features_df.empty:
            log.error("清理NaN后特征数据为空，无法训练模型")
            return {}
        
        feature_cols = [col for col in features_df.columns
                       if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]

        X = features_df[feature_cols].values
        y = features_df['label'].values

        # 确保y是整数类型，没有NaN
        y = y.astype(int)
        
        # 清理特征数据中的inf和NaN值
        import numpy as np
        # 替换inf为NaN
        X = np.where(np.isinf(X), np.nan, X)
        # 替换NaN为0（或者使用中位数填充）
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 检查是否有异常值
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            log.warning("特征数据中仍有异常值，使用更严格的清理")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        log.info(f"特征数据清理完成: {X.shape}, 异常值已处理")
        
        log.info(f"训练数据维度: {X.shape[0]} 样本 × {X.shape[1]} 特征")
        log.info(f"正样本比例: {np.mean(y):.3f}")

        # 时间序列分割
        if self.config['model']['training']['time_series_split']:
            tscv = TimeSeriesSplit(n_splits=self.config['model']['training']['n_splits'])
            splits = list(tscv.split(X))

            # 使用最后一个分割作为训练/测试集
            train_idx, test_idx = splits[-1]
        else:
            # 简单分割（非时间序列）
            split_point = int(len(X) * (1 - self.config['model']['training']['test_size']))
            train_idx = np.arange(split_point)
            test_idx = np.arange(split_point, len(X))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        log.info(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

        # 训练模型
        model_params = self.config['model']['parameters']
        self.model = xgb.XGBClassifier(**model_params)

        log.info("开始模型训练...")
        self.model.fit(X_train, y_train)

        # 评估模型
        train_metrics = self._evaluate_model(self.model, X_train, y_train, "训练集")
        test_metrics = self._evaluate_model(self.model, X_test, y_test, "测试集")

        # 分析特征重要性
        log.info("📊 分析特征重要性...")
        feature_importance = self._analyze_feature_importance(feature_cols)
        
        # 保存特征重要性
        importance_path = os.path.join(self.model_dir, f"feature_importance_{self.config['model']['version']}.csv")
        feature_importance.to_csv(importance_path, index=False, encoding='utf-8')
        log.info(f"特征重要性已保存至: {importance_path}")
        
        # 显示Top 20重要特征
        log.info("\n" + "="*60)
        log.info("🏆 Top 20 重要特征:")
        log.info("="*60)
        for idx, row in feature_importance.head(20).iterrows():
            log.info(f"  {idx+1:2d}. {row['feature']:30s}: {row['importance']:.6f} ({row['importance_pct']:.2f}%)")
        log.info("="*60)

        # 保存模型
        model_path = os.path.join(self.model_dir, f"left_breakout_{self.config['model']['version']}.joblib")
        joblib.dump(self.model, model_path)
        log.info(f"模型已保存至: {model_path}")

        # 保存特征列名
        self.feature_columns = feature_cols
        feature_cols_path = os.path.join(self.model_dir, f"feature_columns_{self.config['model']['version']}.txt")
        with open(feature_cols_path, 'w') as f:
            f.write('\n'.join(feature_cols))

        # 编译训练结果
        training_results = {
            'model_path': model_path,
            'feature_columns_path': feature_cols_path,
            'feature_importance_path': importance_path,
            'feature_columns': feature_cols,
            'feature_importance': feature_importance.to_dict('records'),
            'top_features': feature_importance.head(20).to_dict('records'),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 保存训练报告
        self._save_training_report(training_results)

        log.info("模型训练完成")
        return training_results

    def _analyze_feature_importance(self, feature_cols: List[str]) -> pd.DataFrame:
        """
        分析特征重要性
        
        Args:
            feature_cols: 特征列名列表
            
        Returns:
            特征重要性DataFrame，包含特征名、重要性分数、百分比和排名
        """
        if self.model is None:
            log.error("模型未训练")
            return pd.DataFrame()
        
        try:
            # 获取特征重要性分数
            importance_scores = self.model.feature_importances_
            
            # 计算总重要性（用于计算百分比）
            total_importance = np.sum(importance_scores)
            
            # 构建DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance_scores,
                'importance_pct': (importance_scores / total_importance * 100) if total_importance > 0 else 0
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            # 添加排名
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            return importance_df
            
        except Exception as e:
            log.error(f"分析特征重要性失败: {e}")
            return pd.DataFrame()

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
        """
        评估模型性能

        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 标签向量
            dataset_name: 数据集名称

        Returns:
            评估指标字典
        """
        try:
            # 预测
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 计算指标
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y, y_pred_proba)
            }

            log.info(f"{dataset_name}评估结果:")
            for metric, value in metrics.items():
                log.info(f"  {metric}: {value:.4f}")

            return metrics

        except Exception as e:
            log.error(f"模型评估失败: {e}")
            return {}

    def _save_training_report(self, training_results: Dict):
        """保存训练报告"""
        try:
            report_path = os.path.join(self.model_dir, f"training_report_{self.config['model']['version']}.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("左侧潜力牛股模型训练报告\n")
                f.write("="*80 + "\n\n")

                f.write(f"训练时间: {training_results['training_time']}\n")
                f.write(f"模型版本: {self.config['model']['version']}\n")
                f.write(f"模型路径: {training_results['model_path']}\n\n")

                f.write("数据统计:\n")
                f.write(f"  训练样本: {training_results['train_samples']}\n")
                f.write(f"  测试样本: {training_results['test_samples']}\n")
                f.write(f"  特征数量: {len(training_results['feature_columns'])}\n\n")

                f.write("训练集性能:\n")
                for metric, value in training_results['train_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")

                f.write("\n测试集性能:\n")
                for metric, value in training_results['test_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")

                f.write("\n模型参数:\n")
                for param, value in self.config['model']['parameters'].items():
                    f.write(f"  {param}: {value}\n")

                f.write("\n特征列表:\n")
                for i, feature in enumerate(training_results['feature_columns'], 1):
                    f.write(f"  {i:2d}. {feature}\n")

                # 添加特征重要性分析
                if 'feature_importance' in training_results and training_results['feature_importance']:
                    f.write("\n" + "="*80 + "\n")
                    f.write("特征重要性分析 (Top 20)\n")
                    f.write("="*80 + "\n")
                    f.write(f"{'排名':<6} {'特征名':<35} {'重要性分数':<15} {'占比(%)':<10}\n")
                    f.write("-"*80 + "\n")
                    
                    top_features = training_results.get('top_features', [])
                    for feat in top_features[:20]:
                        f.write(f"{feat.get('rank', 0):<6} {feat.get('feature', ''):<35} "
                               f"{feat.get('importance', 0):<15.6f} {feat.get('importance_pct', 0):<10.2f}\n")
                    
                    f.write("\n特征重要性统计:\n")
                    if top_features:
                        total_importance = sum(f.get('importance', 0) for f in top_features[:20])
                        f.write(f"  Top 20特征总重要性: {total_importance:.6f}\n")
                        f.write(f"  Top 20特征占比: {sum(f.get('importance_pct', 0) for f in top_features[:20]):.2f}%\n")

            log.info(f"训练报告已保存至: {report_path}")

        except Exception as e:
            log.error(f"保存训练报告失败: {e}")

    def load_model(self, version: str = None) -> bool:
        """
        加载已训练的模型

        Args:
            version: 模型版本，默认使用配置中的版本

        Returns:
            是否加载成功
        """
        if version is None:
            version = self.config['model']['version']

        try:
            model_path = os.path.join(self.model_dir, f"left_breakout_{version}.joblib")
            feature_cols_path = os.path.join(self.model_dir, f"feature_columns_{version}.txt")

            if not os.path.exists(model_path):
                log.error(f"模型文件不存在: {model_path}")
                return False

            # 加载模型
            self.model = joblib.load(model_path)
            log.info(f"模型已加载: {model_path}")

            # 加载特征列
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]
                log.info(f"特征列已加载: {len(self.feature_columns)} 个特征")
            else:
                log.warning(f"特征列文件不存在: {feature_cols_path}")

            return True

        except Exception as e:
            log.error(f"加载模型失败: {e}")
            return False

    def predict_stocks(self, stock_features: pd.DataFrame) -> pd.DataFrame:
        """
        对股票进行预测评分

        Args:
            stock_features: 股票特征DataFrame

        Returns:
            预测结果DataFrame
        """
        if self.model is None:
            log.error("模型未加载，无法进行预测")
            return pd.DataFrame()

        if stock_features.empty:
            log.warning("输入特征数据为空")
            return pd.DataFrame()

        try:
            # 准备特征数据
            available_features = [col for col in self.feature_columns if col in stock_features.columns]

            if len(available_features) == 0:
                log.error("没有可用的特征列")
                return pd.DataFrame()

            if len(available_features) < len(self.feature_columns):
                log.warning(f"缺少 {len(self.feature_columns) - len(available_features)} 个特征列")

            X = stock_features[available_features].values

            # 预测概率
            probabilities = self.model.predict_proba(X)[:, 1]

            # 构建结果
            results = stock_features[['ts_code', 'name']].copy()
            results['probability'] = probabilities
            results['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 按概率排序
            results = results.sort_values('probability', ascending=False).reset_index(drop=True)

            log.info(f"预测完成，共 {len(results)} 只股票")

            return results

        except Exception as e:
            log.error(f"预测失败: {e}")
            return pd.DataFrame()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            log.error("模型未加载")
            return pd.DataFrame()

        try:
            importance_scores = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance_scores
            })

            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

            return importance_df

        except Exception as e:
            log.error(f"获取特征重要性失败: {e}")
            return pd.DataFrame()
