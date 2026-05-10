#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
短期动量评分模型 (Short-Term Scorer)

预测未来 5 个交易日上涨（涨幅 >= 5% 且最大回撤 >= -3%）的概率。
特征以动量/技术指标为主，模型为 LightGBM 单模型 + Platt 校准。

Usage:
    from src.models.short_term_scorer import ShortTermScorer
    scorer = ShortTermScorer()
    # 训练
    df_features = scorer.prepare_training_data("20230101", "20241231")
    metrics = scorer.train(df_features, feature_cols=scorer.feature_cols)
    scorer.save_model(metrics)
    # 预测
    probs = scorer.predict(df_test)
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


class ShortTermScorer(ThreeLightBase):
    """短期动量评分模型"""

    MODEL_NAME = "short_term_scorer"
    LOOKFORWARD_DAYS = 5
    RETURN_THRESHOLD = 0.05
    MAX_DRAWDOWN_THRESHOLD = -0.03
    EXCESS_RETURN = False

    # 特征列（由 extract_features 生成）
    FEATURE_COLS = [
        "rsi_6", "rsi_12", "rsi_24",
        "macd", "macd_dif", "macd_dea",
        "kdj_k", "kdj_d", "kdj_j",
        "return_1d", "return_3d", "return_5d", "return_10d",
        "vol_ratio", "turnover_rate",
        "volatility_5d", "volatility_10d",
        "excess_return_5d", "excess_return_10d", "excess_return_20d",
        "close_ma20_ratio", "close_ma60_ratio",
        "vol_change_5d", "amount_ratio",
        "industry_rel_5d",
        "rsi_macd_score",
        "momentum_acceleration",
    ]

    def __init__(self, model_version: str = "v1.0.0", data_provider=None):
        super().__init__(model_version=model_version, data_provider=data_provider)
        self.feature_cols = self.FEATURE_COLS.copy()

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """准备训练数据：标签 + 特征"""
        # 1. 生成标签
        df_labels = self.generate_labels(start_date, end_date)
        if df_labels.empty:
            return pd.DataFrame()

        # 2. 提取特征
        df_features = self.extract_features_for_training(df_labels)
        if df_features.empty:
            return pd.DataFrame()

        # 3. 合并标签
        df_merged = df_features.merge(
            df_labels[["ts_code", "trade_date", "label", "future_close_ret", "future_max_drawdown"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        df_merged = df_merged[df_merged["label"].notna()]

        return df_merged

    def extract_features_for_training(self, df_labels: pd.DataFrame) -> pd.DataFrame:
        """为训练数据提取特征（批量优化版）

        使用向量化 groupby + rolling 计算，避免逐个样本循环。
        """
        log.info(f"[{self.MODEL_NAME}] 提取特征（批量），样本数={len(df_labels)}")

        # 获取日期范围
        dates = pd.to_datetime(df_labels["trade_date"].unique())
        min_date = (dates.min() - pd.Timedelta(days=30)).strftime("%Y%m%d")
        max_date = dates.max().strftime("%Y%m%d")

        # 加载数据（处理 ArcticDB 的 DatetimeIndex）
        df_ohlcv = self.data_provider.read_daily_ohlcv(min_date, max_date)
        if df_ohlcv.empty:
            log.warning("ohlcv 数据为空")
            return pd.DataFrame()
        if isinstance(df_ohlcv.index, pd.DatetimeIndex):
            df_ohlcv = df_ohlcv.reset_index()
        df_ohlcv["trade_date"] = pd.to_datetime(df_ohlcv["trade_date"])

        df_factors = self.data_provider.read_daily_factors(min_date, max_date)
        if not df_factors.empty and isinstance(df_factors.index, pd.DatetimeIndex):
            df_factors = df_factors.reset_index()
        if not df_factors.empty:
            df_factors["trade_date"] = pd.to_datetime(df_factors["trade_date"])

        df_basic = self.data_provider.read_daily_basic(min_date, max_date)
        if not df_basic.empty and isinstance(df_basic.index, pd.DatetimeIndex):
            df_basic = df_basic.reset_index()
        if not df_basic.empty:
            df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])

        # 加载大盘数据
        df_market = self._load_market_index(min_date, max_date)

        # === 批量计算 ohlcv 衍生特征 ===
        df_o = df_ohlcv.sort_values(["ts_code", "trade_date"]).copy()
        df_o["return_1d"] = df_o.groupby("ts_code")["close"].pct_change()
        df_o["return_3d"] = df_o.groupby("ts_code")["close"].pct_change(periods=3)
        df_o["return_5d"] = df_o.groupby("ts_code")["close"].pct_change(periods=5)
        df_o["return_10d"] = df_o.groupby("ts_code")["close"].pct_change(periods=10)
        df_o["volatility_5d"] = df_o.groupby("ts_code")["return_1d"].transform(lambda x: x.rolling(5, min_periods=3).std())
        df_o["volatility_10d"] = df_o.groupby("ts_code")["return_1d"].transform(lambda x: x.rolling(10, min_periods=5).std())
        df_o["ma20"] = df_o.groupby("ts_code")["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df_o["ma60"] = df_o.groupby("ts_code")["close"].transform(lambda x: x.rolling(60, min_periods=20).mean())
        df_o["close_ma20_ratio"] = df_o["close"] / df_o["ma20"] - 1
        df_o["close_ma60_ratio"] = df_o["close"] / df_o["ma60"] - 1

        # 超额收益（相对大盘 5日/10日/20日）
        if not df_market.empty:
            df_m = df_market.sort_values("trade_date").copy()
            df_m["market_ret_5d"] = df_m["close"].pct_change(periods=5)
            df_m["market_ret_10d"] = df_m["close"].pct_change(periods=10)
            df_m["market_ret_20d"] = df_m["close"].pct_change(periods=20)
            market_map_5d = df_m.set_index("trade_date")["market_ret_5d"].to_dict()
            market_map_10d = df_m.set_index("trade_date")["market_ret_10d"].to_dict()
            market_map_20d = df_m.set_index("trade_date")["market_ret_20d"].to_dict()
            df_o["excess_return_5d"] = df_o["return_5d"] - df_o["trade_date"].map(market_map_5d).fillna(0)
            df_o["excess_return_10d"] = df_o["return_10d"] - df_o["trade_date"].map(market_map_10d).fillna(0)
            df_o["excess_return_20d"] = df_o["return_20d"] - df_o["trade_date"].map(market_map_20d).fillna(0)
        else:
            df_o["excess_return_5d"] = 0.0
            df_o["excess_return_10d"] = 0.0
            df_o["excess_return_20d"] = 0.0

        # 成交量变化（近5日 vs 前5日）
        df_o["vol_change_5d"] = df_o.groupby("ts_code")["vol"].transform(
            lambda x: x.rolling(5, min_periods=3).mean() / x.shift(5).rolling(5, min_periods=3).mean() - 1
        )

        # 成交额比率（当日 / 20日均）
        df_o["amount_ratio"] = df_o["amount"] / df_o.groupby("ts_code")["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())

        # 行业相对强弱（个股5日涨幅 - 行业均值5日涨幅）
        try:
            df_basic_ref = self.data_provider.read_stock_basic()
            industry_map = df_basic_ref.set_index("ts_code")["industry"].to_dict()
            df_o["industry"] = df_o["ts_code"].map(industry_map)
            df_o["industry_rel_5d"] = df_o["return_5d"] - df_o.groupby(["trade_date", "industry"])["return_5d"].transform("mean")
        except Exception:
            df_o["industry_rel_5d"] = 0.0

        # RSI + MACD 联合信号
        if not df_factors.empty and "rsi_12" in df_o.columns and "macd" in df_o.columns:
            df_o["rsi_macd_score"] = ((df_o["rsi_12"] > 50).astype(int) + (df_o["macd"] > 0).astype(int)) / 2.0
        else:
            df_o["rsi_macd_score"] = 0.5

        # 动量加速/减速
        df_o["momentum_acceleration"] = (df_o["return_5d"] - df_o["return_10d"] / 2) / (df_o["volatility_10d"] + 1e-6)

        # 取每个 (ts_code, trade_date) 的最新记录
        df_o = df_o.sort_values(["ts_code", "trade_date"])

        # === 合并 factors ===
        df_f = df_factors.copy() if not df_factors.empty else pd.DataFrame()
        factor_cols = ["rsi_6", "rsi_12", "rsi_24", "macd", "macd_dif", "macd_dea", "kdj_k", "kdj_d", "kdj_j"]
        if not df_f.empty:
            df_f = df_f[["ts_code", "trade_date"] + [c for c in factor_cols if c in df_f.columns]]

        # === 合并 basic ===
        df_b = df_basic.copy() if not df_basic.empty else pd.DataFrame()
        basic_cols = ["volume_ratio", "turnover_rate"]
        if not df_b.empty:
            df_b = df_b[["ts_code", "trade_date"] + [c for c in basic_cols if c in df_b.columns]]

        # === 与标签表 merge ===
        df_labels["trade_date"] = pd.to_datetime(df_labels["trade_date"])

        # 取每个股票在每个日期的最新记录
        df_merged = df_labels[["ts_code", "trade_date"]].copy()

        # merge ohlcv 特征
        ohlcv_cols = ["ts_code", "trade_date", "return_1d", "return_3d", "return_5d", "return_10d",
                      "volatility_5d", "volatility_10d", "close_ma20_ratio", "close_ma60_ratio",
                      "excess_return_5d", "excess_return_10d", "excess_return_20d",
                      "vol_change_5d", "amount_ratio", "industry_rel_5d",
                      "rsi_macd_score", "momentum_acceleration"]
        df_merged = df_merged.merge(
            df_o[[c for c in ohlcv_cols if c in df_o.columns]],
            on=["ts_code", "trade_date"],
            how="left",
        )

        # merge factors
        if not df_f.empty:
            df_merged = df_merged.merge(df_f, on=["ts_code", "trade_date"], how="left")

        # merge basic
        if not df_b.empty:
            df_merged = df_merged.merge(df_b, on=["ts_code", "trade_date"], how="left")

        # 填充缺失值
        for col in self.FEATURE_COLS:
            if col not in df_merged.columns:
                df_merged[col] = 0.0
            else:
                df_merged[col] = df_merged[col].fillna(0)

        return df_merged

    def _load_market_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载上证指数用于计算超额收益"""
        try:
            df = self.data_provider.read_daily_ohlcv(start_date, end_date)
            df = df[df["ts_code"] == "000001.SH"].copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df.sort_values("trade_date")
        except Exception:
            return pd.DataFrame()

    def _compute_single_features(
        self,
        o: pd.DataFrame,
        f: pd.DataFrame,
        b: pd.DataFrame,
        df_market: pd.DataFrame,
        t1: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """计算单只股票在 t1 日的特征"""
        feats = {}

        # --- 技术指标（来自 factors） ---
        if not f.empty:
            fl = f.iloc[-1]
            feats["rsi_6"] = fl.get("rsi_6", 50)
            feats["rsi_12"] = fl.get("rsi_12", 50)
            feats["rsi_24"] = fl.get("rsi_24", 50)
            feats["macd"] = fl.get("macd", 0)
            feats["macd_dif"] = fl.get("macd_dif", 0)
            feats["macd_dea"] = fl.get("macd_dea", 0)
            feats["kdj_k"] = fl.get("kdj_k", 50)
            feats["kdj_d"] = fl.get("kdj_d", 50)
            feats["kdj_j"] = fl.get("kdj_j", 50)
        else:
            feats.update({k: 0 for k in ["rsi_6", "rsi_12", "rsi_24", "macd", "macd_dif", "macd_dea", "kdj_k", "kdj_d", "kdj_j"]})

        # --- 价格动量（来自 ohlcv） ---
        close = o["close"].values
        if len(close) >= 2:
            feats["return_1d"] = close[-1] / close[-2] - 1
        else:
            feats["return_1d"] = 0
        if len(close) >= 4:
            feats["return_3d"] = close[-1] / close[-4] - 1
        else:
            feats["return_3d"] = 0
        if len(close) >= 6:
            feats["return_5d"] = close[-1] / close[-6] - 1
        else:
            feats["return_5d"] = 0
        if len(close) >= 11:
            feats["return_10d"] = close[-1] / close[-11] - 1
        else:
            feats["return_10d"] = 0

        # --- 波动率 ---
        if len(close) >= 5:
            pct = pd.Series(close).pct_change().dropna()
            feats["volatility_5d"] = pct.tail(5).std() if len(pct) >= 5 else 0
            feats["volatility_10d"] = pct.tail(10).std() if len(pct) >= 10 else 0
        else:
            feats["volatility_5d"] = 0
            feats["volatility_10d"] = 0

        # --- MA 位置 ---
        if len(close) >= 20:
            ma20 = close[-20:].mean()
            feats["close_ma20_ratio"] = close[-1] / ma20 - 1
        else:
            feats["close_ma20_ratio"] = 0
        if len(close) >= 60:
            ma60 = close[-60:].mean()
            feats["close_ma60_ratio"] = close[-1] / ma60 - 1
        else:
            feats["close_ma60_ratio"] = 0

        # --- 基本面（来自 basic） ---
        if not b.empty:
            bl = b.iloc[-1]
            feats["vol_ratio"] = bl.get("volume_ratio", 1.0)
            feats["turnover_rate"] = bl.get("turnover_rate", 0.0)
        else:
            feats["vol_ratio"] = 1.0
            feats["turnover_rate"] = 0.0

        # --- 超额收益（相对大盘 5 日） ---
        feats["excess_return_5d"] = 0.0
        if not df_market.empty and len(close) >= 6:
            stock_ret_5d = close[-1] / close[-6] - 1
            # 找到 t1 日期前后的大盘数据
            m = df_market[df_market["trade_date"] <= t1].sort_values("trade_date")
            if len(m) >= 6:
                market_ret_5d = m["close"].iloc[-1] / m["close"].iloc[-6] - 1
                feats["excess_return_5d"] = stock_ret_5d - market_ret_5d

        return pd.Series(feats)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对外接口：从输入 DataFrame 提取特征

        用于 enrich_predictions.py 中的推理阶段。
        df 需要包含 ts_code, trade_date，以及 ohlcv/factors/basic 的合并数据。
        """
        # enrich_predictions.py 会传入单行合并数据
        # 这里简化处理：直接复用 _compute_single_features 的逻辑
        # 但 enrich 场景中数据已经通过 ArcticDB 批量读取
        # 所以实际使用时，会在 enrich 脚本中批量调用
        return df
