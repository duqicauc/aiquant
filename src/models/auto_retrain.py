#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动重训练评估器 (Auto Retrain Evaluator)

A/B 对比新旧模型，自动判断是否触发重训练及模型替换。
支持增量训练评估和全量训练评估两种模式。

Usage:
    from src.models.auto_retrain import AutoRetrainEvaluator
    evaluator = AutoRetrainEvaluator()
    result = evaluator.evaluate(
        new_train_df, new_test_df,
        current_model_path="data/models/v293/ensemble",
        retrain_script="scripts/train_v293_ensemble_calibrated.py",
    )
    if result["should_replace"]:
        evaluator.deploy(result["new_model_path"])
"""

import json
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def _get_current_model_dir() -> Path:
    """读取 current.json 获取当前生产模型目录"""
    current_json = PROJECT_ROOT / "data" / "models" / "current.json"
    version = "v2.8.0-ensemble"
    if current_json.exists():
        try:
            data = json.loads(current_json.read_text(encoding="utf-8"))
            version = data.get("current_version", version)
        except Exception:
            pass
    return (
        PROJECT_ROOT
        / "data"
        / "models"
        / "breakout_launch_scorer"
        / "versions"
        / version
        / "model"
    )


@dataclass
class ModelMetrics:
    """模型评估指标"""
    auc: float = 0.0
    brier: float = 1.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    pos_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "pos_rate": round(self.pos_rate, 4),
        }


class AutoRetrainEvaluator:
    """自动重训练评估器

    参数:
        min_improvement: 触发模型替换的最小提升（AUC 绝对提升）
        min_test_samples: 测试集最小样本数
        current_model_dir: 当前生产模型目录
    """

    def __init__(
        self,
        min_improvement: float = 0.005,
        min_test_samples: int = 500,
        current_model_dir: Optional[Path] = None,
    ):
        self.min_improvement = min_improvement
        self.min_test_samples = min_test_samples
        self.current_model_dir = current_model_dir or _get_current_model_dir()
        self.history: List[dict] = []
        self._predictor = None

    # ==================== 模型评估 ====================

    def _load_current_models(self) -> Dict[str, any]:
        """加载当前生产模型"""
        models = {}
        model_files = {
            "xgb": self.current_model_dir / "xgboost.json",
            "lgb": self.current_model_dir / "lightgbm.txt",
            "cat": self.current_model_dir / "catboost.cbm",
        }
        for name, path in model_files.items():
            if path.exists():
                try:
                    if name == "xgb":
                        import xgboost as xgb
                        models[name] = xgb.Booster()
                        models[name].load_model(str(path))
                    elif name == "lgb":
                        import lightgbm as lgb
                        models[name] = lgb.Booster(model_file=str(path))
                    elif name == "cat":
                        from catboost import CatBoostClassifier
                        models[name] = CatBoostClassifier()
                        models[name].load_model(str(path))
                    log.info(f"加载当前模型: {name}")
                except Exception as e:
                    log.warning(f"加载 {name} 失败: {e}")
        return models

    def _load_meta_learner(self):
        """加载 Stacking 元学习器"""
        path = self.current_model_dir / "meta_learner.pkl"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                log.warning(f"加载 meta_learner 失败: {e}")
        return None

    def _load_calibrators(self) -> Dict[str, any]:
        """加载校准器"""
        calibrators = {}
        for name in ["xgb", "lgb", "cat"]:
            path = self.current_model_dir / f"calibrator_{name}.pkl"
            if path.exists():
                try:
                    calibrators[name] = joblib.load(path)
                except Exception as e:
                    log.warning(f"加载 calibrator_{name} 失败: {e}")
        return calibrators

    def _get_predictor(self):
        """懒加载 EnsemblePredictor"""
        if self._predictor is None:
            from src.prediction.predictor import EnsemblePredictor
            # 从路径推断版本名
            version = self.current_model_dir.parent.name
            self._predictor = EnsemblePredictor(model_version=version)
        return self._predictor

    def _predict_current(self, X: pd.DataFrame) -> np.ndarray:
        """使用当前生产模型预测"""
        try:
            predictor = self._get_predictor()
            ensemble_prob, _, _ = predictor.predict(X)
            return ensemble_prob
        except Exception as e:
            log.error(f"当前模型预测失败: {e}")
            return np.zeros(len(X))

    def _evaluate_model(self, y_true: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
        """评估模型性能"""
        y_pred = (y_prob >= 0.5).astype(int)
        m = ModelMetrics()
        m.auc = roc_auc_score(y_true, y_prob)
        m.brier = brier_score_loss(y_true, y_prob)
        m.precision = precision_score(y_true, y_pred, zero_division=0)
        m.recall = recall_score(y_true, y_pred, zero_division=0)
        m.f1 = f1_score(y_true, y_pred, zero_division=0)
        m.accuracy = accuracy_score(y_true, y_pred)
        m.pos_rate = y_prob.mean()
        return m

    # ==================== A/B 对比 ====================

    def evaluate(
        self,
        test_df: pd.DataFrame,
        new_model_predict_fn: Optional[callable] = None,
        new_model_path: Optional[Path] = None,
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
    ) -> dict:
        """A/B 对比评估

        参数:
            test_df: 测试数据
            new_model_predict_fn: 新模型的预测函数(y_prob = fn(X_test))
            new_model_path: 新模型路径（如果预测函数为 None，尝试加载）
            label_col: 标签列
            feature_cols: 特征列，None 则自动推断

        返回:
            包含对比结果和是否替换建议的字典
        """
        if test_df is None or len(test_df) < self.min_test_samples:
            return {
                "status": "error",
                "message": f"测试集样本不足: {len(test_df) if test_df is not None else 0} < {self.min_test_samples}",
            }

        if label_col not in test_df.columns:
            return {"status": "error", "message": f"缺少标签列: {label_col}"}

        y_test = test_df[label_col].values

        # 推断特征列
        if feature_cols is None:
            exclude = {
                label_col, "ts_code", "trade_date", "name", "sample_id",
                "future_high", "future_close_ret", "future_low",
                "future_max_drawdown", "max_return", "label_close",
            }
            feature_cols = [c for c in test_df.columns if c not in exclude]

        X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())

        # 评估旧模型
        log.info("评估当前生产模型...")
        old_prob = self._predict_current(X_test)
        old_metrics = self._evaluate_model(y_test, old_prob)

        # 评估新模型
        new_metrics = None
        new_prob = None
        if new_model_predict_fn is not None:
            log.info("评估新模型（自定义预测函数）...")
            new_prob = new_model_predict_fn(X_test)
            new_metrics = self._evaluate_model(y_test, new_prob)
        elif new_model_path and new_model_path.exists():
            log.info(f"评估新模型: {new_model_path}...")
            # 这里可以扩展为加载新模型并预测
            # 简化起见，假设新模型是同样的ensemble格式
            new_prob = self._predict_with_path(X_test, new_model_path)
            new_metrics = self._evaluate_model(y_test, new_prob)
        else:
            log.warning("无新模型，仅评估旧模型")

        # 对比
        result = {
            "status": "ok",
            "test_samples": len(y_test),
            "old_model": old_metrics.to_dict(),
            "new_model": new_metrics.to_dict() if new_metrics else None,
        }

        if new_metrics:
            delta_auc = new_metrics.auc - old_metrics.auc
            delta_brier = old_metrics.brier - new_metrics.brier  # brier越低越好
            should_replace = delta_auc >= self.min_improvement

            result["comparison"] = {
                "delta_auc": round(delta_auc, 4),
                "delta_brier": round(delta_brier, 4),
                "should_replace": should_replace,
                "reason": (
                    f"新模型 AUC {'+' if delta_auc >= 0 else ''}{delta_auc:.4f}, "
                    f"Brier {'+' if delta_brier >= 0 else ''}{delta_brier:.4f}"
                ),
            }

            self.history.append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "old_auc": old_metrics.auc,
                "new_auc": new_metrics.auc,
                "delta_auc": delta_auc,
                "should_replace": should_replace,
            })
        else:
            result["comparison"] = None

        return result

    def _predict_with_path(self, X: pd.DataFrame, model_path: Path) -> np.ndarray:
        """从路径加载并预测（简化版，实际应与新模型训练脚本保持一致）"""
        # TODO: 根据新模型格式实现加载逻辑
        return np.zeros(len(X))

    # ==================== 模型部署 ====================

    def deploy(self, new_model_dir: Path, backup: bool = True) -> Path:
        """部署新模型到生产环境

        参数:
            new_model_dir: 新模型目录
            backup: 是否备份旧模型
        """
        if not new_model_dir.exists():
            raise ValueError(f"新模型目录不存在: {new_model_dir}")

        if backup:
            backup_dir = (
                PROJECT_ROOT
                / "data"
                / "models_backup"
                / f"v293_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.current_model_dir.exists():
                shutil.copytree(self.current_model_dir, backup_dir)
                log.info(f"旧模型已备份: {backup_dir}")

        # 替换模型文件
        for src in new_model_dir.glob("*"):
            dst = self.current_model_dir / src.name
            if src.is_file():
                shutil.copy2(str(src), str(dst))
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(str(src), str(dst))

        log.info(f"新模型已部署: {self.current_model_dir}")
        return self.current_model_dir

    def save_report(self, result: dict, path: Optional[Path] = None) -> Path:
        """保存评估报告"""
        if path is None:
            path = (
                PROJECT_ROOT
                / "data"
                / "training"
                / "retrain_eval"
                / f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"评估报告已保存: {path}")
        return path


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自动重训练评估")
    parser.add_argument("--test-csv", required=True, help="测试集 CSV")
    parser.add_argument("--new-model", help="新模型目录")
    parser.add_argument("--min-improvement", type=float, default=0.005, help="最小提升阈值")
    args = parser.parse_args()

    df_test = pd.read_csv(args.test_csv)
    evaluator = AutoRetrainEvaluator(min_improvement=args.min_improvement)
    result = evaluator.evaluate(
        df_test,
        new_model_path=Path(args.new_model) if args.new_model else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
