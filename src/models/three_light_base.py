#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3L 评分模型基类

为 short_term_scorer 和 long_term_scorer 提供通用基础设施：
- 标签生成（基于未来收益率）
- 数据加载（ArcticDB 优先）
- LightGBM 训练 + 时间序列 CV
- Platt 校准（sigmoid 拟合）
- 模型保存/加载（与现有模型目录结构兼容）

Usage:
    from src.models.three_light_base import ThreeLightBase
    class ShortTermScorer(ThreeLightBase):
        LOOKFORWARD_DAYS = 5
        ...
"""

import json
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log


class ThreeLightBase:
    """3L 评分模型基类"""

    # 子类必须覆盖
    MODEL_NAME: str = ""
    LOOKFORWARD_DAYS: int = 0
    RETURN_THRESHOLD: float = 0.0
    MAX_DRAWDOWN_THRESHOLD: float = None  # 可选的回撤约束
    EXCESS_RETURN: bool = False  # 是否使用超额收益（相对于大盘）

    # 默认 LightGBM 参数
    DEFAULT_LGB_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    def __init__(
        self,
        model_version: str = "v1.0.0",
        data_provider: Optional[ArcticDataProvider] = None,
    ):
        self.model_version = model_version
        self.model_dir = (
            PROJECT_ROOT
            / "data"
            / "models"
            / self.MODEL_NAME
            / "versions"
            / model_version
            / "model"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_provider = data_provider or ArcticDataProvider()
        self.model = None
        self.feature_names: List[str] = []
        self.calibrator = None

    # ==================== 标签生成 ====================

    def generate_labels(
        self,
        start_date: str,
        end_date: str,
        min_price: float = 2.0,
        max_price: float = 500.0,
    ) -> pd.DataFrame:
        """生成训练标签

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            min_price: 最小股价过滤
            max_price: 最大股价过滤

        Returns:
            DataFrame with columns: ts_code, trade_date, close, label, ...
        """
        log.info(f"[{self.MODEL_NAME}] 生成标签: {start_date} ~ {end_date}")

        # 向后扩展日期以确保有足够未来数据
        end_dt = pd.to_datetime(end_date)
        extended_end = (end_dt + pd.Timedelta(days=self.LOOKFORWARD_DAYS * 2)).strftime("%Y%m%d")

        df = self._load_daily_data(start_date, extended_end)
        if df.empty:
            log.warning("无数据")
            return pd.DataFrame()

        df = df[(df["close"] >= min_price) & (df["close"] <= max_price)]

        # 计算未来收益
        df = self._compute_forward_returns(df)
        if df.empty:
            log.warning("未来收益计算失败")
            return pd.DataFrame()

        df = self._filter_valid_samples(df)
        df = self._assign_labels(df)

        # 只保留原始日期范围
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]

        pos = df["label"].sum()
        neg = len(df) - pos
        pos_rate = pos / len(df) * 100 if len(df) > 0 else 0
        log.info(f"[{self.MODEL_NAME}] 标签完成: 总{len(df)} 正{pos} 负{neg} 正样本率{pos_rate:.2f}%")
        return df

    def _load_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """从 ArcticDB 加载日线数据"""
        try:
            df = self.data_provider.read_daily_ohlcv(start_date, end_date)
            if not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                return df
        except Exception as e:
            log.warning(f"ArcticDB 读取失败: {e}")
        return pd.DataFrame()

    def _compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算未来 N 日收益指标"""
        results = []
        grouped = df.groupby("ts_code", sort=False)

        for ts_code, g in grouped:
            g = g.sort_values("trade_date").copy()
            if len(g) < self.LOOKFORWARD_DAYS + 1:
                continue

            n = self.LOOKFORWARD_DAYS
            # 未来 N 日收盘价收益
            g["future_close_ret"] = g["close"].shift(-n) / g["close"] - 1
            # 未来 N 日最大涨幅（基于最高价）
            g["future_high"] = g["high"].shift(-1).rolling(n, min_periods=1).max()
            g["max_return"] = g["future_high"] / g["close"] - 1
            # 未来 N 日最大回撤
            g["future_low"] = g["low"].shift(-1).rolling(n, min_periods=1).min()
            g["future_max_drawdown"] = g["future_low"] / g["close"] - 1

            # 如果需要超额收益，加载大盘数据
            if self.EXCESS_RETURN:
                g = self._add_excess_return(g, df)

            results.append(g)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _add_excess_return(self, df_stock: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
        """计算相对于大盘的超额收益"""
        # 取上证指数
        df_market = df_all[df_all["ts_code"] == "000001.SH"].copy()
        if df_market.empty:
            df_stock["future_market_ret"] = 0.0
            df_stock["future_excess_ret"] = df_stock["future_close_ret"]
            return df_stock

        df_market = df_market.sort_values("trade_date")
        df_market["future_market_ret"] = df_market["close"].shift(-self.LOOKFORWARD_DAYS) / df_market["close"] - 1
        market_map = df_market.set_index("trade_date")["future_market_ret"].to_dict()

        df_stock["future_market_ret"] = df_stock["trade_date"].map(market_map).fillna(0.0)
        df_stock["future_excess_ret"] = df_stock["future_close_ret"] - df_stock["future_market_ret"]
        return df_stock

    def _filter_valid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤有效样本"""
        df = df[(df["vol"] > 0) & (df["amount"] > 0)]
        df = df[~df["ts_code"].str.match(r"^[89]", na=False)]
        try:
            st_codes = self.data_provider.get_st_stock_codes()
            df = df[~df["ts_code"].isin(st_codes)]
        except Exception:
            pass
        df = df[df["future_close_ret"].notna()]
        return df

    def _assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据阈值分配二元标签"""
        df = df.copy()

        if self.EXCESS_RETURN:
            ret_col = "future_excess_ret"
        else:
            ret_col = "future_close_ret"

        # 基础标签：收益达标
        label = df[ret_col] >= self.RETURN_THRESHOLD

        # 回撤约束（如果配置了）
        if self.MAX_DRAWDOWN_THRESHOLD is not None:
            dd_ok = df["future_max_drawdown"] >= self.MAX_DRAWDOWN_THRESHOLD
            label = label & dd_ok

        df["label"] = label.astype(int)
        return df

    # ==================== 特征提取（子类覆盖） ====================

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取特征 —— 子类必须覆盖"""
        raise NotImplementedError("子类必须实现 extract_features")

    # ==================== 训练 ====================

    def train(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        lgb_params: Optional[Dict] = None,
        n_splits: int = 5,
    ) -> Dict:
        """训练 LightGBM 模型 + Platt 校准

        Args:
            df_features: 包含特征和标签的 DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            lgb_params: LightGBM 参数（覆盖默认值）
            n_splits: 时间序列交叉验证折数

        Returns:
            metrics dict with auc, best_iteration, feature_importance, etc.
        """
        log.info(f"[{self.MODEL_NAME}] 开始训练，样本数={len(df_features)}, 特征数={len(feature_cols)}")

        params = {**self.DEFAULT_LGB_PARAMS, **(lgb_params or {})}

        # 清理特征数据
        X = df_features[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        y = df_features[label_col].values

        # 自动处理类别不平衡
        # 注意：scale_pos_weight 会扭曲概率分布，导致 Platt 校准失效
        # 仅在模型不用于概率解释时启用（如纯排序场景）
        # 3L 模型需要输出真实概率，因此默认不启用
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            ratio = n_neg / n_pos
            if ratio > 20 or ratio < 0.05:
                # 极端不平衡时启用，但记录警告
                params["scale_pos_weight"] = np.sqrt(ratio)
                log.warning(f"[{self.MODEL_NAME}] 极端类别不平衡，启用 scale_pos_weight={np.sqrt(ratio):.2f} (正{n_pos}/负{n_neg})，概率解释可能失真")

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.zeros(len(y))
        fold_aucs = []
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=params.get("n_estimators", 200),
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 20), verbose=False)],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = preds

            auc = roc_auc_score(y_val, preds)
            fold_aucs.append(auc)
            best_iterations.append(model.best_iteration)
            log.info(f"  Fold {fold+1}/{n_splits}: AUC={auc:.4f}, best_iter={model.best_iteration}")

        # 全量训练（使用平均 best_iteration）
        avg_best_iter = int(np.mean(best_iterations))
        final_params = params.copy()
        final_params["n_estimators"] = avg_best_iter
        # 全量训练不需要 early stopping
        final_params.pop("early_stopping_rounds", None)

        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(final_params, train_data, num_boost_round=avg_best_iter)
        self.feature_names = feature_cols

        # OOF AUC
        oof_auc = roc_auc_score(y, oof_preds)
        log.info(f"[{self.MODEL_NAME}] OOF AUC={oof_auc:.4f}, Fold AUCs={fold_aucs}")

        # Platt 校准（用 OOF 预测拟合 sigmoid）
        self.calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self.calibrator.fit(oof_preds.reshape(-1, 1), y)
        calibrated = self.calibrator.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        cal_auc = roc_auc_score(y, calibrated)
        log.info(f"[{self.MODEL_NAME}] 校准后 AUC={cal_auc:.4f}")

        # 特征重要性
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        metrics = {
            "oof_auc": float(oof_auc),
            "calibrated_auc": float(cal_auc),
            "fold_aucs": [float(a) for a in fold_aucs],
            "best_iteration": avg_best_iter,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
            "top_features": importance.head(20).to_dict("records"),
        }

        # 保存 OOF 预测（用于后续分层胜率分析）
        self._oof_df = pd.DataFrame({
            "raw_prob": oof_preds,
            "calibrated_prob": calibrated,
            "label": y,
        })
        # 如果有原始索引信息，也保存
        if "ts_code" in df_features.columns:
            self._oof_df["ts_code"] = df_features["ts_code"].values
        if "trade_date" in df_features.columns:
            self._oof_df["trade_date"] = df_features["trade_date"].values

        return metrics

    # ==================== 预测 ====================

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（经过校准）"""
        if self.model is None:
            self.load_model()

        # 对齐特征
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                X_aligned[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X_aligned[col] = 0.0

        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

        raw_probs = self.model.predict(X_aligned)

        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            probs = raw_probs

        return np.clip(probs, 0.0, 1.0)

    # ==================== 保存 / 加载 ====================

    def save_model(self, metrics: Dict) -> None:
        """保存模型、特征名、校准器、指标"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 保存 LightGBM 模型
        model_path = self.model_dir / "lightgbm.txt"
        self.model.save_model(str(model_path))

        # 保存特征名
        with open(self.model_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

        # 保存校准器
        if self.calibrator is not None:
            joblib.dump(self.calibrator, self.model_dir / "calibrator.pkl")

        # 保存指标
        with open(self.model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 保存元数据
        metadata = {
            "model_name": self.MODEL_NAME,
            "version": self.model_version,
            "created_at": datetime.now().isoformat(),
            "lookforward_days": self.LOOKFORWARD_DAYS,
            "return_threshold": self.RETURN_THRESHOLD,
            "max_drawdown_threshold": self.MAX_DRAWDOWN_THRESHOLD,
            "excess_return": self.EXCESS_RETURN,
            "n_features": len(self.feature_names),
        }
        with open(self.model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 保存 OOF 预测（用于回测分析）
        if hasattr(self, "_oof_df") and self._oof_df is not None:
            oof_path = self.model_dir / "oof_predictions.parquet"
            try:
                self._oof_df.to_parquet(oof_path, index=False)
                log.info(f"[{self.MODEL_NAME}] OOF 预测已保存: {oof_path}")
            except Exception as e:
                log.warning(f"[{self.MODEL_NAME}] OOF 预测保存失败: {e}")

        log.success(f"[{self.MODEL_NAME}] 模型已保存到 {self.model_dir}")

    def load_model(self) -> None:
        """加载模型、特征名、校准器"""
        model_path = self.model_dir / "lightgbm.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = lgb.Booster(model_file=str(model_path))

        with open(self.model_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        calibrator_path = self.model_dir / "calibrator.pkl"
        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
        else:
            self.calibrator = None

        log.info(f"[{self.MODEL_NAME}] 模型已加载: {model_path}")

    def model_exists(self) -> bool:
        """检查模型文件是否存在"""
        return (self.model_dir / "lightgbm.txt").exists()
