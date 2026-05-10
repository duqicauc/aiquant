#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长期质量评分模型 (Long-Term Scorer)

预测未来 120 个交易日跑赢大盘（超额收益 >= 10%）的概率。
特征以基本面估值 + 长期趋势为主，模型为 LightGBM 单模型 + Platt 校准。

Usage:
    from src.models.long_term_scorer import LongTermScorer
    scorer = LongTermScorer()
    df_features = scorer.prepare_training_data("20220101", "20241231")
    metrics = scorer.train(df_features, feature_cols=scorer.feature_cols)
    scorer.save_model(metrics)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.three_light_base import ThreeLightBase
from src.utils.logger import log


class LongTermScorer(ThreeLightBase):
    """长期质量评分模型"""

    MODEL_NAME = "long_term_scorer"
    LOOKFORWARD_DAYS = 120
    RETURN_THRESHOLD = 0.20
    MAX_DRAWDOWN_THRESHOLD = -0.15
    EXCESS_RETURN = True  # 使用超额收益（相对大盘）

    FEATURE_COLS = [
        "pe", "pb",
        "pe_industry_zscore", "pb_industry_zscore",
        "total_mv_log", "circ_mv_log",
        "turnover_rate", "volume_ratio",
        "return_20d", "return_60d", "return_120d",
        "volatility_60d", "volatility_120d",
        "close_ma60_ratio", "close_ma120_ratio",
        "max_drawdown_60d",
        "trend_strength_60d",
    ]

    def __init__(self, model_version: str = "v1.0.0", data_provider=None):
        super().__init__(model_version=model_version, data_provider=data_provider)
        self.feature_cols = self.FEATURE_COLS.copy()
        self._industry_map = None

    def _load_industry_map(self) -> dict:
        """加载股票行业映射"""
        if self._industry_map is not None:
            return self._industry_map
        try:
            df_basic = self.data_provider.read_stock_basic()
            self._industry_map = df_basic.set_index("ts_code")["industry"].to_dict()
        except Exception:
            self._industry_map = {}
        return self._industry_map

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """准备训练数据"""
        df_labels = self.generate_labels(start_date, end_date)
        if df_labels.empty:
            return pd.DataFrame()

        df_features = self.extract_features_for_training(df_labels)
        if df_features.empty:
            return pd.DataFrame()

        df_merged = df_features.merge(
            df_labels[["ts_code", "trade_date", "label", "future_excess_ret", "future_close_ret"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        df_merged = df_merged[df_merged["label"].notna()]
        return df_merged

    def extract_features_for_training(self, df_labels: pd.DataFrame) -> pd.DataFrame:
        """提取长期特征（批量优化版）"""
        log.info(f"[{self.MODEL_NAME}] 提取特征（批量），样本数={len(df_labels)}")

        dates = pd.to_datetime(df_labels["trade_date"].unique())
        min_date = (dates.min() - pd.Timedelta(days=150)).strftime("%Y%m%d")
        max_date = dates.max().strftime("%Y%m%d")

        # 加载数据
        df_ohlcv = self.data_provider.read_daily_ohlcv(min_date, max_date)
        if df_ohlcv.empty:
            log.warning("ohlcv 数据为空")
            return pd.DataFrame()
        if isinstance(df_ohlcv.index, pd.DatetimeIndex):
            df_ohlcv = df_ohlcv.reset_index()
        df_ohlcv["trade_date"] = pd.to_datetime(df_ohlcv["trade_date"])

        df_basic = self.data_provider.read_daily_basic(min_date, max_date)
        if not df_basic.empty and isinstance(df_basic.index, pd.DatetimeIndex):
            df_basic = df_basic.reset_index()
        if not df_basic.empty:
            df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])

        industry_map = self._load_industry_map()

        # === 批量计算 ohlcv 长期特征 ===
        df_o = df_ohlcv.sort_values(["ts_code", "trade_date"]).copy()
        g = df_o.groupby("ts_code")
        df_o["return_1d"] = g["close"].pct_change()
        df_o["return_20d"] = g["close"].pct_change(periods=20)
        df_o["return_60d"] = g["close"].pct_change(periods=60)
        df_o["return_120d"] = g["close"].pct_change(periods=120)
        df_o["volatility_60d"] = g["return_1d"].transform(lambda x: x.rolling(60, min_periods=30).std())
        df_o["volatility_120d"] = g["return_1d"].transform(lambda x: x.rolling(120, min_periods=60).std())
        df_o["ma60"] = g["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())
        df_o["ma120"] = g["close"].transform(lambda x: x.rolling(120, min_periods=60).mean())
        df_o["close_ma60_ratio"] = df_o["close"] / df_o["ma60"] - 1
        df_o["close_ma120_ratio"] = df_o["close"] / df_o["ma120"] - 1

        # 60 日最大回撤
        def rolling_max_drawdown(x):
            rolling_max = x.expanding(min_periods=5).max()
            dd = (x - rolling_max) / rolling_max
            return dd.min()
        df_o["max_drawdown_60d"] = g["close"].transform(lambda x: x.rolling(60, min_periods=30).apply(rolling_max_drawdown, raw=False))

        # 趋势强度（60 日线性回归斜率 / 残差标准差）
        def trend_strength(x):
            x = np.array(x)
            if len(x) < 10:
                return 0
            y = x
            xi = np.arange(len(y))
            slope = np.polyfit(xi, y, 1)[0]
            resid = y - np.polyval(np.polyfit(xi, y, 1), xi)
            se = np.std(resid) / np.sqrt(len(y)) if len(y) > 1 else 1e-6
            return slope / max(se, 1e-6)
        df_o["trend_strength_60d"] = g["close"].transform(lambda x: x.rolling(60, min_periods=30).apply(trend_strength, raw=False))

        # === 批量计算 basic 特征 ===
        if not df_basic.empty:
            df_b = df_basic.copy()
            df_b["industry"] = df_b["ts_code"].map(industry_map)

            # 行业 z-score（按日期+行业）
            def zscore(group):
                group = group.copy()
                for col in ["pe", "pb"]:
                    if col in group.columns:
                        mean = group[col].mean()
                        std = group[col].std()
                        if std and std > 0:
                            group[f"{col}_industry_zscore"] = (group[col] - mean) / std
                        else:
                            group[f"{col}_industry_zscore"] = 0
                return group
            df_b = df_b.groupby(["trade_date", "industry"], group_keys=False).apply(zscore)

            # 市值对数
            for col in ["total_mv", "circ_mv"]:
                if col in df_b.columns:
                    df_b[f"{col}_log"] = np.log(df_b[col].clip(lower=1))
        else:
            df_b = pd.DataFrame()

        # === 与标签表 merge ===
        df_labels["trade_date"] = pd.to_datetime(df_labels["trade_date"])

        # merge ohlcv 特征
        ohlcv_cols = ["ts_code", "trade_date", "return_20d", "return_60d", "return_120d",
                      "volatility_60d", "volatility_120d", "close_ma60_ratio", "close_ma120_ratio",
                      "max_drawdown_60d", "trend_strength_60d"]
        df_merged = df_labels[["ts_code", "trade_date"]].merge(
            df_o[[c for c in ohlcv_cols if c in df_o.columns]],
            on=["ts_code", "trade_date"],
            how="left",
        )

        # merge basic
        if not df_b.empty:
            basic_cols = ["ts_code", "trade_date", "pe", "pb",
                          "pe_industry_zscore", "pb_industry_zscore",
                          "total_mv_log", "circ_mv_log",
                          "turnover_rate", "volume_ratio"]
            df_merged = df_merged.merge(
                df_b[[c for c in basic_cols if c in df_b.columns]],
                on=["ts_code", "trade_date"],
                how="left",
            )

        # 填充缺失值
        for col in self.FEATURE_COLS:
            if col not in df_merged.columns:
                df_merged[col] = 0.0
            else:
                df_merged[col] = df_merged[col].fillna(0)

        return df_merged

    def _compute_industry_percentiles(self, df_basic: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
        """计算 PE/PB 的行业分位数"""
        if df_basic.empty:
            return df_basic

        df_basic = df_basic.copy()
        df_basic["industry"] = df_basic["ts_code"].map(industry_map)

        # 按日期+行业分组，计算 PE/PB 的 zscore
        def zscore(group):
            group = group.copy()
            for col in ["pe", "pb"]:
                if col in group.columns:
                    mean = group[col].mean()
                    std = group[col].std()
                    if std and std > 0:
                        group[f"{col}_industry_zscore"] = (group[col] - mean) / std
                    else:
                        group[f"{col}_industry_zscore"] = 0
            return group

        df_basic = df_basic.groupby(["trade_date", "industry"], group_keys=False).apply(zscore)
        return df_basic

    def _compute_single_features(
        self,
        o: pd.DataFrame,
        b: pd.DataFrame,
        industry: str,
    ) -> Optional[pd.Series]:
        """计算单只股票的长期特征"""
        feats = {}
        close = o["close"].values

        # --- 基本面（来自 basic） ---
        if not b.empty:
            bl = b.iloc[-1]
            feats["pe"] = bl.get("pe", np.nan)
            feats["pb"] = bl.get("pb", np.nan)
            feats["pe_industry_zscore"] = bl.get("pe_industry_zscore", 0)
            feats["pb_industry_zscore"] = bl.get("pb_industry_zscore", 0)
            mv = bl.get("total_mv", np.nan)
            feats["total_mv_log"] = np.log(mv) if pd.notna(mv) and mv > 0 else 0
            cmv = bl.get("circ_mv", np.nan)
            feats["circ_mv_log"] = np.log(cmv) if pd.notna(cmv) and cmv > 0 else 0
            feats["turnover_rate"] = bl.get("turnover_rate", 0)
            feats["volume_ratio"] = bl.get("volume_ratio", 1.0)
        else:
            feats.update({
                "pe": np.nan, "pb": np.nan,
                "pe_industry_zscore": 0, "pb_industry_zscore": 0,
                "total_mv_log": 0, "circ_mv_log": 0,
                "turnover_rate": 0, "volume_ratio": 1.0,
            })

        # --- 长期动量 ---
        if len(close) >= 21:
            feats["return_20d"] = close[-1] / close[-21] - 1
        else:
            feats["return_20d"] = 0
        if len(close) >= 61:
            feats["return_60d"] = close[-1] / close[-61] - 1
        else:
            feats["return_60d"] = 0
        if len(close) >= 121:
            feats["return_120d"] = close[-1] / close[-121] - 1
        else:
            feats["return_120d"] = 0

        # --- 波动率 ---
        pct = pd.Series(close).pct_change().dropna()
        if len(pct) >= 60:
            feats["volatility_60d"] = pct.tail(60).std()
            feats["volatility_120d"] = pct.tail(min(120, len(pct))).std()
        else:
            feats["volatility_60d"] = 0
            feats["volatility_120d"] = 0

        # --- MA 位置 ---
        if len(close) >= 60:
            ma60 = close[-60:].mean()
            feats["close_ma60_ratio"] = close[-1] / ma60 - 1
        else:
            feats["close_ma60_ratio"] = 0
        if len(close) >= 120:
            ma120 = close[-120:].mean()
            feats["close_ma120_ratio"] = close[-1] / ma120 - 1
        else:
            feats["close_ma120_ratio"] = 0

        # --- 最大回撤（过去 60 日） ---
        if len(close) >= 60:
            rolling_max = pd.Series(close[-60:]).cummax()
            drawdown = (close[-60:] - rolling_max) / rolling_max
            feats["max_drawdown_60d"] = drawdown.min()
        else:
            feats["max_drawdown_60d"] = 0

        # --- 趋势强度（60 日斜率 / 标准误） ---
        if len(close) >= 60:
            y = close[-60:]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            resid = y - np.polyval(np.polyfit(x, y, 1), x)
            se = np.std(resid) / np.sqrt(len(y)) if len(y) > 1 else 1e-6
            feats["trend_strength_60d"] = slope / max(se, 1e-6)
        else:
            feats["trend_strength_60d"] = 0

        # 填充 NaN
        for k, v in feats.items():
            if pd.isna(v):
                feats[k] = 0

        return pd.Series(feats)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对外接口"""
        return df
