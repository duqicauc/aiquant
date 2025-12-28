"""
左侧潜力牛股模型 - Walk-Forward滚动验证

评估模型在不同市场周期的稳定性和预测能力
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from src.utils.logger import log
from .left_model import LeftBreakoutModel


class LeftBreakoutValidator:
    """左侧潜力牛股模型验证器"""

    def __init__(self, left_model: LeftBreakoutModel):
        """
        初始化验证器

        Args:
            left_model: 左侧模型实例
        """
        self.model = left_model
        self.validation_results = {}

    def walk_forward_validation(
        self,
        features_df: pd.DataFrame,
        n_splits: int = 5,
        min_train_samples: int = 1000
    ) -> Dict:
        """
        执行Walk-Forward滚动验证

        Args:
            features_df: 特征DataFrame（按时间排序）
            n_splits: 验证折数
            min_train_samples: 最少训练样本数

        Returns:
            验证结果字典
        """
        log.info(f"开始Walk-Forward滚动验证 (折数: {n_splits})...")

        if features_df.empty or len(features_df) < min_train_samples:
            log.error(f"样本数量不足: {len(features_df)} < {min_train_samples}")
            return {}

        # 确保数据按时间排序（假设t0_date是时间字段）
        features_df = features_df.sort_values('t0_date').reset_index(drop=True)

        # 准备特征数据
        feature_cols = [col for col in features_df.columns
                       if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]

        X = features_df[feature_cols].values
        y = features_df['label'].values
        dates = features_df['t0_date'].values

        log.info(f"验证数据: {len(X)} 样本 × {len(feature_cols)} 特征")

        # 计算每折的大小
        fold_size = len(X) // n_splits
        results = []

        for fold in range(n_splits):
            log.info(f"\n执行第 {fold + 1}/{n_splits} 折验证...")

            # 计算训练和测试集的索引
            if fold == n_splits - 1:
                # 最后折使用剩余所有数据
                test_start = fold * fold_size
                train_end = test_start
                test_end = len(X)
            else:
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size
                train_end = test_start

            # 确保训练集有足够样本
            if train_end < min_train_samples:
                train_end = min(min_train_samples, test_start)

            # 分割数据
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            test_dates = dates[test_start:test_end]

            log.info(f"  训练集: {len(X_train)} 样本 ({dates[0]} - {dates[train_end-1]})")
            log.info(f"  测试集: {len(X_test)} 样本 ({test_dates[0]} - {test_dates[-1]})")

            if len(X_train) < 100 or len(X_test) < 50:
                log.warning(f"  样本数量不足，跳过此折")
                continue

            # 训练模型
            fold_model = self._train_fold_model(X_train, y_train)

            # 评估模型
            fold_metrics = self._evaluate_fold_model(fold_model, X_test, y_test)

            # 添加时间信息
            fold_metrics.update({
                'fold': fold + 1,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_period': f"{dates[0]} - {dates[train_end-1]}",
                'test_period': f"{test_dates[0]} - {test_dates[-1]}",
                'test_positive_ratio': np.mean(y_test)
            })

            results.append(fold_metrics)

            log.info(f"  准确率: {fold_metrics['accuracy']:.4f}, AUC: {fold_metrics['auc_roc']:.4f}")

        # 计算总体统计
        validation_summary = self._calculate_validation_summary(results)

        # 保存验证结果
        self.validation_results = {
            'summary': validation_summary,
            'fold_results': results,
            'validation_config': {
                'n_splits': n_splits,
                'total_samples': len(X),
                'feature_count': len(feature_cols),
                'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        self._save_validation_report()

        log.info("Walk-Forward验证完成")
        return self.validation_results

    def _train_fold_model(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """
        训练单折模型

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            训练好的模型
        """
        try:
            model_params = self.model.config['model']['parameters'].copy()

            # 为验证调整参数（减少过拟合）
            model_params.update({
                'n_estimators': 50,  # 减少树的数量
                'max_depth': 4,      # 减少树深度
                'learning_rate': 0.05  # 降低学习率
            })

            fold_model = xgb.XGBClassifier(**model_params)
            fold_model.fit(X_train, y_train)

            return fold_model

        except Exception as e:
            log.error(f"训练折模型失败: {e}")
            return None

    def _evaluate_fold_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估单折模型

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        try:
            # 预测
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 计算指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }

            return metrics

        except Exception as e:
            log.error(f"评估折模型失败: {e}")
            return {}

    def _calculate_validation_summary(self, fold_results: List[Dict]) -> Dict:
        """
        计算验证总体统计

        Args:
            fold_results: 各折结果列表

        Returns:
            总体统计字典
        """
        if not fold_results:
            return {}

        try:
            # 提取各指标
            accuracies = [r['accuracy'] for r in fold_results if 'accuracy' in r]
            precisions = [r['precision'] for r in fold_results if 'precision' in r]
            recalls = [r['recall'] for r in fold_results if 'recall' in r]
            f1_scores = [r['f1_score'] for r in fold_results if 'f1_score' in r]
            aucs = [r['auc_roc'] for r in fold_results if 'auc_roc' in r]

            # 计算统计量
            summary = {}

            for metric_name, values in [
                ('accuracy', accuracies),
                ('precision', precisions),
                ('recall', recalls),
                ('f1_score', f1_scores),
                ('auc_roc', aucs)
            ]:
                if values:
                    summary[f'{metric_name}_mean'] = np.mean(values)
                    summary[f'{metric_name}_std'] = np.std(values)
                    summary[f'{metric_name}_min'] = np.min(values)
                    summary[f'{metric_name}_max'] = np.max(values)

                    # 稳定性评分（标准差的倒数，越小越稳定）
                    if np.mean(values) > 0:
                        summary[f'{metric_name}_stability'] = 1 / (np.std(values) + 0.001)  # 加小常数避免除零

            # 整体评估
            if aucs:
                avg_auc = np.mean(aucs)
                auc_stability = 1 / (np.std(aucs) + 0.001)

                if avg_auc > 0.75 and auc_stability > 10:
                    summary['overall_rating'] = '优秀'
                elif avg_auc > 0.70 and auc_stability > 5:
                    summary['overall_rating'] = '良好'
                elif avg_auc > 0.65:
                    summary['overall_rating'] = '合格'
                else:
                    summary['overall_rating'] = '需改进'

            return summary

        except Exception as e:
            log.error(f"计算验证统计失败: {e}")
            return {}

    def _save_validation_report(self):
        """保存验证报告"""
        try:
            report_dir = os.path.join(self.model.model_dir, 'validation')
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(report_dir, f'walk_forward_validation_{timestamp}.json')

            # 保存JSON格式的结果
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

            # 保存文本格式的简要报告
            txt_report_file = os.path.join(report_dir, f'validation_report_{timestamp}.txt')

            with open(txt_report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("左侧潜力牛股模型 - Walk-Forward验证报告\n")
                f.write("="*80 + "\n\n")

                config = self.validation_results.get('validation_config', {})
                f.write(f"验证时间: {config.get('validation_time', 'N/A')}\n")
                f.write(f"总样本数: {config.get('total_samples', 0)}\n")
                f.write(f"特征数量: {config.get('feature_count', 0)}\n")
                f.write(f"验证折数: {config.get('n_splits', 0)}\n\n")

                summary = self.validation_results.get('summary', {})
                if summary:
                    f.write("总体性能统计:\n")
                    f.write("-"*40 + "\n")

                    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
                    for metric in metrics:
                        mean_key = f'{metric}_mean'
                        std_key = f'{metric}_std'
                        stability_key = f'{metric}_stability'

                        if mean_key in summary:
                            mean_val = summary[mean_key]
                            std_val = summary.get(std_key, 0)
                            stability_val = summary.get(stability_key, 0)

                            f.write(f"{metric.capitalize():20}: {mean_val:.4f} ± {std_val:.4f} (稳定性: {stability_val:.2f})")
                        f.write("\n")

                    rating = summary.get('overall_rating', 'N/A')
                    f.write(f"整体评级: {rating}\n\n")

                # 各折详细结果
                fold_results = self.validation_results.get('fold_results', [])
                if fold_results:
                    f.write("各折详细结果:\n")
                    f.write("-"*80 + "\n")
                    f.write("<10")
                    f.write("-"*80 + "\n")

                    for result in fold_results:
                        f.write("2d"
                               "<12"
                               "<12"
                               "<8.4f"
                               "<8.4f"
                               "<8.4f"
                               "<8.4f"
                               "<8.4f"
                               "\n")

            log.info(f"验证报告已保存至: {txt_report_file}")

        except Exception as e:
            log.error(f"保存验证报告失败: {e}")

    def time_series_cross_validation(
        self,
        features_df: pd.DataFrame,
        initial_train_size: float = 0.6,
        test_size: float = 0.2,
        step_size: float = 0.1
    ) -> Dict:
        """
        时间序列交叉验证（另一种验证方法）

        Args:
            features_df: 特征DataFrame
            initial_train_size: 初始训练集比例
            test_size: 测试集比例
            step_size: 每次滑动的步长比例

        Returns:
            交叉验证结果
        """
        log.info("开始时间序列交叉验证...")

        if features_df.empty:
            return {}

        # 确保数据按时间排序
        features_df = features_df.sort_values('t0_date').reset_index(drop=True)

        # 准备数据
        feature_cols = [col for col in features_df.columns
                       if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]

        X = features_df[feature_cols].values
        y = features_df['label'].values
        dates = features_df['t0_date'].values

        total_samples = len(X)
        initial_train = int(total_samples * initial_train_size)
        test_samples = int(total_samples * test_size)
        step_samples = int(total_samples * step_size)

        results = []
        current_train_end = initial_train

        fold = 1
        while current_train_end + test_samples <= total_samples:
            # 定义训练和测试集
            train_end = current_train_end
            test_start = train_end
            test_end = min(test_start + test_samples, total_samples)

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            test_dates = dates[test_start:test_end]

            log.info(f"交叉验证第 {fold} 轮:")
            log.info(f"  训练集: {len(X_train)} 样本 ({dates[0]} - {dates[train_end-1]})")
            log.info(f"  测试集: {len(X_test)} 样本 ({test_dates[0]} - {test_dates[-1]})")

            # 训练和评估
            fold_model = self._train_fold_model(X_train, y_train)
            fold_metrics = self._evaluate_fold_model(fold_model, X_test, y_test)

            fold_metrics.update({
                'fold': fold,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_period': f"{dates[0]} - {dates[train_end-1]}",
                'test_period': f"{test_dates[0]} - {test_dates[-1]}"
            })

            results.append(fold_metrics)

            # 滑动窗口
            current_train_end += step_samples
            fold += 1

        # 计算统计
        cv_summary = self._calculate_validation_summary(results)

        cv_results = {
            'summary': cv_summary,
            'fold_results': results,
            'cv_config': {
                'method': 'time_series_cv',
                'initial_train_size': initial_train_size,
                'test_size': test_size,
                'step_size': step_size,
                'total_folds': len(results),
                'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        log.info("时间序列交叉验证完成")
        return cv_results

    def robustness_test(
        self,
        features_df: pd.DataFrame,
        n_bootstraps: int = 50,
        sample_fraction: float = 0.8
    ) -> Dict:
        """
        鲁棒性测试（自助采样）

        Args:
            features_df: 特征DataFrame
            n_bootstraps: 自助采样次数
            sample_fraction: 每次采样的比例

        Returns:
            鲁棒性测试结果
        """
        log.info(f"开始鲁棒性测试 (自助采样 {n_bootstraps} 次)...")

        if features_df.empty:
            return {}

        # 准备数据
        feature_cols = [col for col in features_df.columns
                       if col not in ['unique_sample_id', 'ts_code', 'name', 't0_date', 'label']]

        X = features_df[feature_cols].values
        y = features_df['label'].values

        bootstrap_results = []

        for i in range(n_bootstraps):
            if (i + 1) % 10 == 0:
                log.info(f"自助采样进度: {i + 1}/{n_bootstraps}")

            # 自助采样
            n_samples = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), size=n_samples, replace=True)

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 训练模型
            bootstrap_model = self._train_fold_model(X_bootstrap, y_bootstrap)

            # 在全量数据上测试
            bootstrap_metrics = self._evaluate_fold_model(bootstrap_model, X, y)
            bootstrap_metrics['bootstrap_id'] = i + 1

            bootstrap_results.append(bootstrap_metrics)

        # 计算鲁棒性统计
        robustness_stats = self._calculate_bootstrap_statistics(bootstrap_results)

        robustness_results = {
            'statistics': robustness_stats,
            'bootstrap_results': bootstrap_results,
            'robustness_config': {
                'n_bootstraps': n_bootstraps,
                'sample_fraction': sample_fraction,
                'total_samples': len(X),
                'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        # 保存鲁棒性测试结果
        self._save_robustness_report(robustness_results)

        log.info("鲁棒性测试完成")
        return robustness_results

    def _calculate_bootstrap_statistics(self, bootstrap_results: List[Dict]) -> Dict:
        """
        计算自助采样的统计结果

        Args:
            bootstrap_results: 自助采样结果列表

        Returns:
            统计结果字典
        """
        if not bootstrap_results:
            return {}

        try:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            stats = {}

            for metric in metrics:
                values = [r[metric] for r in bootstrap_results if metric in r]

                if values:
                    stats[f'{metric}_mean'] = np.mean(values)
                    stats[f'{metric}_std'] = np.std(values)
                    stats[f'{metric}_95_ci_lower'] = np.percentile(values, 2.5)
                    stats[f'{metric}_95_ci_upper'] = np.percentile(values, 97.5)
                    stats[f'{metric}_robustness'] = 1 / (np.std(values) + 0.001)  # 鲁棒性评分

            return stats

        except Exception as e:
            log.error(f"计算自助采样统计失败: {e}")
            return {}

    def _save_robustness_report(self, robustness_results: Dict):
        """保存鲁棒性测试报告"""
        try:
            report_dir = os.path.join(self.model.model_dir, 'validation')
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(report_dir, f'robustness_test_{timestamp}.txt')

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("左侧潜力牛股模型 - 鲁棒性测试报告\n")
                f.write("="*80 + "\n\n")

                config = robustness_results.get('robustness_config', {})
                f.write(f"测试时间: {config.get('test_time', 'N/A')}\n")
                f.write(f"自助采样次数: {config.get('n_bootstraps', 0)}\n")
                f.write(f"采样比例: {config.get('sample_fraction', 0)}\n")
                f.write(f"总样本数: {config.get('total_samples', 0)}\n\n")

                stats = robustness_results.get('statistics', {})
                if stats:
                    f.write("鲁棒性统计结果:\n")
                    f.write("-"*60 + "\n")
                    f.write("<15")
                    f.write("-"*60 + "\n")

                    for key, value in stats.items():
                        if key.endswith('_mean'):
                            metric = key.replace('_mean', '')
                            mean_val = value
                            std_val = stats.get(f'{metric}_std', 0)
                            ci_lower = stats.get(f'{metric}_95_ci_lower', 0)
                            ci_upper = stats.get(f'{metric}_95_ci_upper', 0)
                            robustness = stats.get(f'{metric}_robustness', 0)

                            f.write("<15"
                                   "<10.4f"
                                   "<10.4f"
                                   "<10.4f"
                                   "<10.4f"
                                   "<10.2f"
                                   "\n")

            log.info(f"鲁棒性测试报告已保存至: {report_file}")

        except Exception as e:
            log.error(f"保存鲁棒性测试报告失败: {e}")
