#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一预测器

标准化预测流程：
1. 获取数据（TushareDataProvider）
2. 计算特征（FeatureEngineer）
3. 模型预测（加载 v2.8.0 集成模型）
4. 输出结果

Usage:
    from src.prediction.predictor import EnsemblePredictor
    predictor = EnsemblePredictor()
    df_result = predictor.predict_date("20260422", lookback_days=70)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.data.tushare_data_provider import TushareDataProvider
from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import log

PROJECT_ROOT = Path(__file__).parent.parent.parent


class EnsemblePredictor:
    """v2.8.x 集成预测器"""

    DEFAULT_MODEL_DIR = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "breakout_launch_scorer"
        / "versions"
        / "v2.8.0-ensemble"
        / "model"
    )

    def __init__(self, model_version: str = None):
        """
        Args:
            model_version: 模型版本，如 "v2.8.0-ensemble" 或 "v2.8.1-ensemble"
        """
        if model_version:
            self.MODEL_DIR = (
                PROJECT_ROOT
                / "data"
                / "models"
                / "breakout_launch_scorer"
                / "versions"
                / model_version
                / "model"
            )
        else:
            self.MODEL_DIR = self.DEFAULT_MODEL_DIR

        self.data_provider = TushareDataProvider()
        self.feature_engineer = FeatureEngineer()
        self.models, self.weights, self.feature_names = self._load_models()

    def _load_models(self) -> Tuple[Dict, dict, List[str]]:
        """加载集成模型"""
        models = {}

        xgb_model = xgb.Booster()
        xgb_model.load_model(str(self.MODEL_DIR / "xgboost.json"))
        models["xgboost"] = xgb_model
        log.info("  加载 XGBoost 模型")

        lgb_model = lgb.Booster(model_file=str(self.MODEL_DIR / "lightgbm.txt"))
        models["lightgbm"] = lgb_model
        log.info("  加载 LightGBM 模型")

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(self.MODEL_DIR / "catboost.cbm"))
        models["catboost"] = cat_model
        log.info("  加载 CatBoost 模型")

        with open(self.MODEL_DIR / "feature_names.json", "r") as f:
            feature_names = json.load(f)
        log.info(f"  特征数: {len(feature_names)}")

        with open(self.MODEL_DIR / "weights.json", "r") as f:
            weights = json.load(f)
        log.info(
            f"  权重: XGB={weights['xgboost']:.4f}, LGB={weights['lightgbm']:.4f}, CAT={weights['catboost']:.4f}"
        )

        return models, weights, feature_names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """对特征矩阵进行预测"""
        # 对齐特征
        X_aligned = pd.DataFrame(index=X.index)
        missing_cols = []
        for col in self.feature_names:
            if col in X.columns:
                X_aligned[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                missing_cols.append(col)
                X_aligned[col] = 0.0

        if missing_cols:
            log.warning(f"缺失特征 {len(missing_cols)} 个: {missing_cols[:10]}...")

        X_aligned = X_aligned.astype(float).fillna(0)

        # XGBoost
        dmatrix = xgb.DMatrix(X_aligned, feature_names=self.feature_names)
        pred_xgb = self.models["xgboost"].predict(dmatrix)

        # LightGBM
        pred_lgb = self.models["lightgbm"].predict(X_aligned)

        # CatBoost
        pred_cat = self.models["catboost"].predict_proba(X_aligned)[:, 1]

        # 加权集成
        ensemble = (
            pred_xgb * self.weights["xgboost"]
            + pred_lgb * self.weights["lightgbm"]
            + pred_cat * self.weights["catboost"]
        )

        return ensemble

    def predict_date(self, prediction_date: str, lookback_days: int = 70) -> pd.DataFrame:
        """预测单个日期

        Args:
            prediction_date: 预测日期 YYYYMMDD
            lookback_days: 回看天数

        Returns:
            预测结果 DataFrame（含 prob, ts_code, name 等）
        """
        pred_dt = datetime.strptime(prediction_date, "%Y%m%d")
        start_dt = pred_dt - timedelta(days=lookback_days + 30)
        start_date = start_dt.strftime("%Y%m%d")
        end_date = prediction_date

        log.info("=" * 80)
        log.info(f"预测日期: {prediction_date}")
        log.info(f"数据范围: {start_date} ~ {end_date}")
        log.info("=" * 80)

        # 1. 获取原始数据
        df_raw = self.data_provider.fetch_date_range(start_date, end_date)
        if df_raw.empty:
            log.error("无数据")
            return pd.DataFrame()

        # 2. 准备特征工程（生成 sample_id 等辅助列）
        df_raw = df_raw.copy()
        df_raw["sample_id"] = df_raw.groupby("ts_code").ngroup()
        df_raw["name"] = ""
        df_raw["days_to_t1"] = 0

        # 3. 获取市场环境数据
        df_market = self.data_provider.fetch_market_index(start_date, end_date)

        # 4. 计算特征
        df_features = self.feature_engineer.compute_all_features(df_raw, df_market)

        # 5. 取预测日期的数据
        pred_date_dt = pd.to_datetime(prediction_date)
        df_pred = df_features[df_features["trade_date"] == pred_date_dt].copy()

        if df_pred.empty:
            log.error(f"预测日期 {prediction_date} 无数据")
            return pd.DataFrame()

        log.info(f"预测样本: {len(df_pred)} 只股票")

        # 6. 模型预测
        df_pred["prob"] = self.predict(df_pred)
        df_pred["prob_xgb"] = self.models["xgboost"].predict(
            xgb.DMatrix(
                df_pred[[c for c in self.feature_names if c in df_pred.columns]].fillna(0).astype(float),
                feature_names=self.feature_names,
            )
        )
        df_pred["prob_lgb"] = self.models["lightgbm"].predict(
            df_pred[[c for c in self.feature_names if c in df_pred.columns]].fillna(0).astype(float)
        )
        df_pred["prob_cat"] = self.models["catboost"].predict_proba(
            df_pred[[c for c in self.feature_names if c in df_pred.columns]].fillna(0).astype(float)
        )[:, 1]

        # 7. 排序
        df_pred = df_pred.sort_values("prob", ascending=False).reset_index(drop=True)
        df_pred["rank"] = range(1, len(df_pred) + 1)

        log.success(
            f"预测完成: Top1={df_pred['prob'].iloc[0]:.4f}, Top50均值={df_pred['prob'].iloc[:50].mean():.4f}"
        )

        return df_pred

    def predict_range(
        self, start_date: str, end_date: str, lookback_days: int = 70
    ) -> Dict[str, pd.DataFrame]:
        """预测日期范围（批量优化版）

        优化策略：
        1. 一次性获取整个日期范围的原始数据
        2. 一次性计算所有特征
        3. 对每一天直接过滤已计算的特征进行预测

        Returns:
            {date_str: df_result, ...}
        """
        trade_dates = self.data_provider.get_trade_dates(start_date, end_date)
        if not trade_dates:
            log.warning("无交易日")
            return {}

        # 计算扩展的数据范围（包含 lookback）
        first_dt = datetime.strptime(trade_dates[0], "%Y%m%d")
        extended_start = (first_dt - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

        log.info("=" * 80)
        log.info(f"批量预测: {start_date} ~ {end_date}, {len(trade_dates)} 个交易日")
        log.info(f"数据范围: {extended_start} ~ {end_date}")
        log.info("=" * 80)

        # 1. 一次性获取原始数据
        df_raw = self.data_provider.fetch_date_range(extended_start, end_date)
        if df_raw.empty:
            log.error("无数据")
            return {}

        # 2. 准备特征工程
        df_raw = df_raw.copy()
        df_raw["sample_id"] = df_raw.groupby("ts_code").ngroup()
        df_raw["name"] = ""
        df_raw["days_to_t1"] = 0

        # 3. 获取市场环境数据
        df_market = self.data_provider.fetch_market_index(extended_start, end_date)

        # 4. 一次性计算所有特征
        df_features = self.feature_engineer.compute_all_features(df_raw, df_market)

        # 5. 对每一天进行预测
        results = {}
        for date in trade_dates:
            try:
                pred_dt = pd.to_datetime(date)
                df_pred = df_features[df_features["trade_date"] == pred_dt].copy()

                if df_pred.empty:
                    log.warning(f"{date} 无预测数据")
                    continue

                # 模型预测
                df_pred["prob"] = self.predict(df_pred)

                # 获取各模型概率
                X = df_pred[[c for c in self.feature_names if c in df_pred.columns]].fillna(0).astype(float)
                df_pred["prob_xgb"] = self.models["xgboost"].predict(xgb.DMatrix(X, feature_names=self.feature_names))
                df_pred["prob_lgb"] = self.models["lightgbm"].predict(X)
                df_pred["prob_cat"] = self.models["catboost"].predict_proba(X)[:, 1]

                # 排序
                df_pred = df_pred.sort_values("prob", ascending=False).reset_index(drop=True)
                df_pred["rank"] = range(1, len(df_pred) + 1)

                results[date] = df_pred
                log.info(
                    f"  {date}: {len(df_pred)} 只, Top1={df_pred['prob'].iloc[0]:.4f}, Top50均值={df_pred['prob'].iloc[:50].mean():.4f}"
                )
            except Exception as e:
                log.error(f"预测 {date} 失败: {e}")

        log.success(f"批量预测完成: {len(results)} 天")
        return results

    def save_results(self, df: pd.DataFrame, prediction_date: str, output_dir: Path):
        """保存预测结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cols = ["rank", "ts_code", "name", "prob", "prob_xgb", "prob_lgb", "prob_cat",
                "close", "pct_chg", "turnover_rate", "total_mv"]
        cols = [c for c in cols if c in df.columns]

        top50 = df.head(50)[cols]
        top50_file = output_dir / f"predictions_{prediction_date}_top50.csv"
        top50.to_csv(top50_file, index=False)
        log.info(f"  Top50: {top50_file}")

        top100 = df.head(100)[cols]
        top100_file = output_dir / f"predictions_{prediction_date}_top100.csv"
        top100.to_csv(top100_file, index=False)
        log.info(f"  Top100: {top100_file}")

        all_file = output_dir / f"predictions_{prediction_date}_all.csv"
        df[cols].to_csv(all_file, index=False)
        log.info(f"  全市场: {all_file}")
