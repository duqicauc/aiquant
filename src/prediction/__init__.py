"""预测模块"""

from src.prediction.predictor import EnsemblePredictor
from src.prediction.catboost_predictor import CatBoostPredictor

__all__ = ["EnsemblePredictor", "CatBoostPredictor"]
