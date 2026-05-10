#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一特征工程模块

训练与预测共用同一套特征计算逻辑。
原则：
1. Tushare 已提供的技术指标（MACD/KDJ/RSI/EMA/MA/OBV/BIAS/ATR）直接使用
2. 仅对 Tushare 没有的自定义特征进行本地计算
3. 所有自定义特征基于 Tushare 因子或基础 OHLCV 计算

Usage:
    from src.features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df_features = fe.compute_all_features(df_raw, df_market)
"""

import warnings

import numpy as np
import pandas as pd

from src.utils.logger import log

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """统一特征工程器"""

    # Tushare 已提供、不应被覆盖的指标 (stk_factor_pro Batch 1)
    TUSHARE_INDICATORS = {
        # MA
        "ma5", "ma10", "ma_20d", "ma30", "ma60", "ma90", "ma250",
        # EMA
        "ema_5", "ema_10", "ema_20", "ema_30", "ema_60", "ema_90", "ema_250",
        # MACD
        "macd", "macd_dea", "macd_dif",
        # RSI
        "rsi_6", "rsi_12", "rsi_24",
        # KDJ
        "kdj_k", "kdj_d", "kdj_j",
        # OBV
        "obv",
        # BIAS
        "bias_short", "bias_mid", "bias_long",
        # ATR
        "atr",
        # BOLL
        "boll_upper", "boll_mid", "boll_lower",
        # CCI
        "cci",
        # DMI
        "dmi_pdi", "dmi_mdi", "dmi_adx", "dmi_adxr",
        # WR
        "wr", "wr1",
        # MFI
        "mfi",
        # MTM
        "mtm", "mtmma",
        # ROC
        "roc", "maroc",
        # PSY
        "psy", "psyma",
        # VR
        "vr",
        # CR
        "cr",
        # BRAR
        "brar_br", "brar_ar",
        # EMV
        "emv", "maemv",
        # BBI
        "bbi",
        # DPO
        "dpo", "madpo",
        # DFMA
        "dfma_dif", "dfma_difma",
        # KTN
        "ktn_upper", "ktn_mid", "ktn_down",
        # TAQ (海龟)
        "taq_up", "taq_mid", "taq_down",
        # TRI
        "trix", "trma",
        # MASS
        "mass", "ma_mass",
        # EXPMA
        "expma_12", "expma_50",
        # ASI
        "asi", "asit",
        # XSII
        "xsii_td1", "xsii_td2", "xsii_td3", "xsii_td4",
    }

    def __init__(self):
        pass

    # ==================== 基础特征 ====================

    def _calc_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术特征（按股票分组）"""
        df = df.copy()

        def calc_per_stock(g):
            g = g.sort_values("trade_date").copy()

            if "close" not in g.columns:
                return g

            # 波动率（基于 pct_chg）
            if "pct_chg" in g.columns:
                g["volatility_8d"] = g["pct_chg"].rolling(8, min_periods=3).std()
                g["volatility_34d"] = g["pct_chg"].rolling(34, min_periods=10).std()
                g["volatility_55d"] = g["pct_chg"].rolling(55, min_periods=20).std()

            # 动量
            g["momentum_5d"] = g["close"].pct_change(5) * 100
            g["momentum_10d"] = g["close"].pct_change(10) * 100
            g["momentum_20d"] = g["close"].pct_change(20) * 100

            # 高低点
            g["high_8d"] = g["close"].rolling(8, min_periods=3).max()
            g["low_8d"] = g["close"].rolling(8, min_periods=3).min()
            g["high_34d"] = g["close"].rolling(34, min_periods=10).max()
            g["low_34d"] = g["close"].rolling(34, min_periods=10).min()
            g["high_55d"] = g["close"].rolling(55, min_periods=20).max()
            g["low_55d"] = g["close"].rolling(55, min_periods=20).min()
            g["high_10d"] = g["close"].rolling(10, min_periods=5).max()
            g["high_20d"] = g["close"].rolling(20, min_periods=10).max()

            # 价格位置
            g["price_position_8d"] = np.where(
                g["high_8d"] > g["low_8d"], (g["close"] - g["low_8d"]) / (g["high_8d"] - g["low_8d"]), 0.5
            )
            g["price_position_34d"] = np.where(
                g["high_34d"] > g["low_34d"], (g["close"] - g["low_34d"]) / (g["high_34d"] - g["low_34d"]), 0.5
            )
            g["price_position_55d"] = np.where(
                g["high_55d"] > g["low_55d"], (g["close"] - g["low_55d"]) / (g["high_55d"] - g["low_55d"]), 0.5
            )

            # MA（Tushare 已提供 ma5/ma10/ma_20d，只补缺失的）
            if "ma5" not in g.columns:
                g["ma5"] = g["close"].rolling(5, min_periods=3).mean()
            if "ma10" not in g.columns:
                g["ma10"] = g["close"].rolling(10, min_periods=5).mean()
            g["ma_34d"] = g["close"].rolling(34, min_periods=10).mean()
            g["ma_55d"] = g["close"].rolling(55, min_periods=20).mean()

            g["price_vs_ma_8d"] = (
                (g["close"] - g["close"].rolling(8, min_periods=3).mean())
                / g["close"].rolling(8, min_periods=3).mean()
                * 100
            )
            g["price_vs_ma_34d"] = (g["close"] - g["ma_34d"]) / g["ma_34d"] * 100
            g["price_vs_ma_55d"] = (g["close"] - g["ma_55d"]) / g["ma_55d"] * 100

            # 趋势斜率
            g["trend_slope_8d"] = g["close"].diff(8) / g["close"].shift(8) * 100
            g["trend_slope_34d"] = g["close"].diff(34) / g["close"].shift(34) * 100
            g["trend_slope_55d"] = g["close"].diff(55) / g["close"].shift(55) * 100

            # 收益率
            g["return_8d"] = g["close"].pct_change(8) * 100
            g["return_34d"] = g["close"].pct_change(34) * 100
            g["return_55d"] = g["close"].pct_change(55) * 100

            # KDJ（Tushare 已提供则跳过）
            if "kdj_k" not in g.columns:
                low_9 = g["close"].rolling(9, min_periods=5).min()
                high_9 = g["close"].rolling(9, min_periods=5).max()
                rsv = np.where(high_9 > low_9, (g["close"] - low_9) / (high_9 - low_9) * 100, 50)
                g["kdj_k"] = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
            if "kdj_d" not in g.columns:
                g["kdj_d"] = pd.Series(g["kdj_k"]).ewm(com=2, adjust=False).mean().values
            if "kdj_j" not in g.columns:
                g["kdj_j"] = 3 * g["kdj_k"] - 2 * g["kdj_d"]

            # 突破标识
            if "high_10d" in g.columns:
                g["breakout_high_10d"] = (g["close"] > g["high_10d"].shift(1)).astype(int)
                g["breakout_high_20d"] = (g["close"] > g["high_20d"].shift(1)).astype(int)
                g["breakout_high_55d"] = (g["close"] > g["high_55d"].shift(1)).astype(int)
            g["breakout_ma5"] = (g["close"] > g["ma5"]).astype(int)
            g["breakout_ma10"] = (g["close"] > g["ma10"]).astype(int)

            # 支撑/阻力
            g["support_10d"] = g["close"].rolling(10, min_periods=5).min()
            g["resistance_10d"] = g["close"].rolling(10, min_periods=5).max()
            g["support_20d"] = g["close"].rolling(20, min_periods=10).min()
            g["resistance_20d"] = g["close"].rolling(20, min_periods=10).max()

            g["support_strength_10d"] = np.where(
                g["support_10d"] > 0, (g["close"] - g["support_10d"]) / g["support_10d"] * 100, 0
            )
            g["resistance_strength_10d"] = np.where(
                g["resistance_10d"] > 0, (g["resistance_10d"] - g["close"]) / g["resistance_10d"] * 100, 0
            )
            g["support_strength_20d"] = np.where(
                g["support_20d"] > 0, (g["close"] - g["support_20d"]) / g["support_20d"] * 100, 0
            )
            g["resistance_strength_20d"] = np.where(
                g["resistance_20d"] > 0, (g["resistance_20d"] - g["close"]) / g["resistance_20d"] * 100, 0
            )

            return g

        group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
        if group_key in df.columns:
            df = df.groupby(group_key, group_keys=False).apply(calc_per_stock)
        else:
            df = calc_per_stock(df)
        return df

    # ==================== 突破特征 ====================

    def _calc_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算突破强度特征"""
        df = df.copy()

        if "close" in df.columns and "high_10d" in df.columns:
            df["breakout_strength_10d"] = np.where(
                df["high_10d"] > 0, (df["close"] - df["high_10d"]) / df["high_10d"] * 100, 0
            )
        else:
            df["breakout_strength_10d"] = 0

        if "close" in df.columns and "high_20d" in df.columns:
            df["breakout_strength_20d"] = np.where(
                df["high_20d"] > 0, (df["close"] - df["high_20d"]) / df["high_20d"] * 100, 0
            )
        else:
            df["breakout_strength_20d"] = 0

        if "close" in df.columns and "high_55d" in df.columns:
            df["breakout_strength_55d"] = np.where(
                df["high_55d"] > 0, (df["close"] - df["high_55d"]) / df["high_55d"] * 100, 0
            )
        else:
            df["breakout_strength_55d"] = 0

        # 放量突破强度
        if "vol" in df.columns:
            group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
            if group_key in df.columns:
                vol_ma20 = df.groupby(group_key)["vol"].transform(lambda x: x.rolling(20, min_periods=10).mean())
            else:
                vol_ma20 = df["vol"].rolling(20, min_periods=10).mean()
            df["breakout_volume_strength"] = np.where(vol_ma20 > 0, df["vol"] / vol_ma20, 1)
        elif "volume_ratio" in df.columns:
            df["breakout_volume_strength"] = df["volume_ratio"].fillna(1)
        else:
            df["breakout_volume_strength"] = 1

        df["breakout_confirmed_10d"] = np.where(df["breakout_strength_10d"] > 0, 1, 0)
        df["breakout_confirmed_20d"] = np.where(df["breakout_strength_20d"] > 0, 1, 0)

        df["breakout_resonance"] = (
            (df["breakout_strength_10d"] > 0).astype(int)
            + (df["breakout_strength_20d"] > 0).astype(int)
            + (df["breakout_strength_55d"] > 0).astype(int)
        )

        return df

    # ==================== 市场环境特征 ====================

    def _calc_market(self, df: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
        """添加市场环境特征"""
        df = df.copy()

        market_cols = [
            "market_pct_chg", "market_return_34d", "market_volatility_34d",
            "market_trend", "market_momentum_5d", "market_momentum_10d",
            "market_momentum_20d", "market_regime", "market_position_20d",
        ]

        # 检查是否已有市场特征
        existing = [c for c in market_cols if c in df.columns]
        if len(existing) >= 5:
            for col in market_cols:
                if col not in df.columns:
                    df[col] = 0
            if "excess_return" not in df.columns and "pct_chg" in df.columns:
                df["excess_return"] = df["pct_chg"] - df.get("market_pct_chg", 0)
            if "excess_return_cumsum" not in df.columns and "excess_return" in df.columns:
                group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
                if group_key in df.columns:
                    df["excess_return_cumsum"] = df.groupby(group_key)["excess_return"].cumsum()
                else:
                    df["excess_return_cumsum"] = df["excess_return"].cumsum()
            if "excess_return_consistency" not in df.columns:
                df["excess_return_consistency"] = np.where(df.get("excess_return", 0) > 0, 1, 0)
            return df

        if df_market.empty:
            for col in market_cols:
                df[col] = 0
            df["excess_return"] = df.get("pct_chg", 0)
            df["excess_return_cumsum"] = 0
            df["excess_return_consistency"] = 0
            return df

        # 合并市场数据
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df_market = df_market.copy()
        df_market["trade_date"] = pd.to_datetime(df_market["trade_date"])

        # 如果 df_market 只有原始行情数据，本地计算市场特征
        if "market_pct_chg" not in df_market.columns and "pct_chg" in df_market.columns:
            df_market["market_pct_chg"] = df_market["pct_chg"]

        if "market_return_34d" not in df_market.columns and "close" in df_market.columns:
            df_market["market_return_34d"] = df_market["close"].pct_change(34) * 100

        if "market_volatility_34d" not in df_market.columns and "pct_chg" in df_market.columns:
            df_market["market_volatility_34d"] = df_market["pct_chg"].rolling(34, min_periods=10).std()

        if "market_momentum_5d" not in df_market.columns and "close" in df_market.columns:
            df_market["market_momentum_5d"] = df_market["close"].pct_change(5) * 100

        if "market_momentum_10d" not in df_market.columns and "close" in df_market.columns:
            df_market["market_momentum_10d"] = df_market["close"].pct_change(10) * 100

        if "market_momentum_20d" not in df_market.columns and "close" in df_market.columns:
            df_market["market_momentum_20d"] = df_market["close"].pct_change(20) * 100

        if "market_trend" not in df_market.columns and "close" in df_market.columns:
            ma5 = df_market["close"].rolling(5, min_periods=3).mean()
            ma10 = df_market["close"].rolling(10, min_periods=5).mean()
            df_market["market_trend"] = np.where(ma5 > ma10, 1, np.where(ma5 < ma10, -1, 0))

        if "market_regime" not in df_market.columns:
            df_market["market_regime"] = 0

        if "market_position_20d" not in df_market.columns and "close" in df_market.columns:
            high_20 = df_market["close"].rolling(20, min_periods=10).max()
            low_20 = df_market["close"].rolling(20, min_periods=10).min()
            df_market["market_position_20d"] = np.where(
                high_20 > low_20, (df_market["close"] - low_20) / (high_20 - low_20), 0.5
            )

        # 只保留需要的列进行合并
        merge_cols = ["trade_date"] + [c for c in market_cols if c in df_market.columns]
        df = df.merge(df_market[merge_cols], on="trade_date", how="left")

        for col in market_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)

        if "pct_chg" in df.columns:
            df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]
        else:
            df["excess_return"] = 0

        group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
        if group_key in df.columns:
            df["excess_return_cumsum"] = df.groupby(group_key)["excess_return"].cumsum()
        else:
            df["excess_return_cumsum"] = df["excess_return"].cumsum()

        df["excess_return_consistency"] = np.where(df["excess_return"] > 0, 1, 0)

        return df

    # ==================== 交互特征 ====================

    def _calc_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算交互特征"""
        df = df.copy()

        breakout_signal = (df.get("breakout_strength_10d", 0) > 0) | (df.get("breakout_strength_20d", 0) > 0)
        volume_signal = df.get("breakout_volume_strength", 1) > 1.5
        df["breakout_with_volume"] = (breakout_signal & volume_signal).astype(int)

        momentum = df.get("momentum_20d", df.get("pct_chg", 0))
        market_trend = df.get("market_trend", 0)
        df["momentum_market_interaction"] = momentum * market_trend

        rsi = df.get("rsi_6", 50)
        kdj_j = df.get("kdj_j", 50)
        df["rsi_kdj_divergence"] = np.abs(rsi - kdj_j)

        ma5 = df.get("ma5", df.get("close", 0))
        ma10 = df.get("ma10", ma5)
        df["trend_consistency"] = np.where(
            (ma5 > ma10) & (df.get("pct_chg", 0) > 0), 1,
            np.where((ma5 < ma10) & (df.get("pct_chg", 0) < 0), 1, 0)
        )

        pct_chg = df.get("pct_chg", pd.Series([0] * len(df)))
        vol = df.get("vol", pd.Series([1] * len(df)))
        vol_chg = vol / vol.shift(1).fillna(1) - 1
        df["volume_price_divergence"] = np.where(
            (pct_chg > 0) & (vol_chg < -0.2), 1,
            np.where((pct_chg < 0) & (vol_chg > 0.2), -1, 0)
        )

        breakout = df.get("breakout_strength_20d", 0)
        df["breakout_rsi_interaction"] = breakout * (100 - rsi) / 100

        vol_34d = df.get("volatility_34d", df.get("pct_chg", pd.Series([0] * len(df))).rolling(34).std())
        market_vol = df.get("market_volatility_34d", 1)
        df["relative_volatility"] = np.where(market_vol > 0, vol_34d / market_vol, 1)

        resonance = df.get("breakout_resonance", 0)
        vol_confirm = (df.get("breakout_volume_strength", 1) > 1.2).astype(int)
        df["resonance_volume_confirm"] = resonance * vol_confirm

        return df

    # ==================== 补充特征 ====================

    def _calc_supplementary(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算补充特征"""
        df = df.copy()
        n = len(df)
        if n < 5:
            return df

        close = df["close"].values if "close" in df.columns else np.ones(n)
        pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(n)

        # 确保基础列
        for col, default in [("high", close * 1.01), ("low", close * 0.99), ("open", close),
                              ("change", np.diff(close, prepend=close[0])),
                              ("amount", df.get("vol", 0) * close),
                              ("pre_close", np.roll(close, 1)),
                              ("price_change", np.diff(close, prepend=close[0])),
                              ("vol", 0), ("turnover_rate", 0), ("turnover_rate_f", df.get("turnover_rate", 0))]:
            if col not in df.columns:
                df[col] = default

        group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
        groups = df.groupby(group_key, group_keys=False) if group_key in df.columns else None

        def safe_rolling(series, window, func="mean", min_periods=None):
            if min_periods is None:
                min_periods = max(1, window // 2)
            if func == "mean":
                return series.rolling(window, min_periods=min_periods).mean()
            elif func == "std":
                return series.rolling(window, min_periods=min_periods).std()
            elif func == "max":
                return series.rolling(window, min_periods=min_periods).max()
            elif func == "min":
                return series.rolling(window, min_periods=min_periods).min()
            elif func == "sum":
                return series.rolling(window, min_periods=min_periods).sum()
            return series

        # EMA（Tushare 已提供则跳过）
        for period in [5, 10, 20, 60]:
            col = f"ema_{period}"
            if col not in df.columns:
                if groups is not None:
                    df[col] = groups["close"].transform(lambda x: x.ewm(span=period, adjust=False, min_periods=1).mean())
                else:
                    df[col] = df["close"].ewm(span=period, adjust=False, min_periods=1).mean()

        # MA 自定义周期
        for period, name in [(5, "ma_5d"), (8, "ma_8d"), (10, "ma_10d"), (20, "ma_20d")]:
            if name not in df.columns:
                if groups is not None:
                    df[name] = groups["close"].transform(lambda x: safe_rolling(x, period, "mean"))
                else:
                    df[name] = safe_rolling(df["close"], period, "mean")

        # 乖离率（Tushare 已提供则跳过）
        for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
            if name not in df.columns:
                if groups is not None:
                    ma = groups["close"].transform(lambda x: safe_rolling(x, period, "mean"))
                else:
                    ma = safe_rolling(df["close"], period, "mean")
                df[name] = (df["close"] - ma) / (ma + 1e-8) * 100

        # ATR
        if "atr_14" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift(1))
            low_close = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr_14"] = safe_rolling(tr, 14, "mean")

        if "atr_ratio_14" not in df.columns:
            df["atr_ratio_14"] = df["atr_14"] / (df["close"] + 1e-8) * 100

        if "atr_expansion" not in df.columns:
            df["atr_expansion"] = df["atr_14"] / (safe_rolling(df["atr_14"], 20, "mean") + 1e-8)

        # 最大回撤
        for period in [10, 20, 55]:
            col = f"max_drawdown_{period}d"
            if col not in df.columns:
                if groups is not None:
                    rolling_max = groups["close"].transform(lambda x: safe_rolling(x, period, "max"))
                else:
                    rolling_max = safe_rolling(df["close"], period, "max")
                df[col] = (df["close"] - rolling_max) / (rolling_max + 1e-8) * 100

        # 距高点天数
        for period in [20, 55]:
            col = f"days_from_high_{period}d"
            if col not in df.columns:
                df[col] = 0

        # 恢复比率
        if "recovery_ratio_20d" not in df.columns:
            if groups is not None:
                rolling_max = groups["close"].transform(lambda x: safe_rolling(x, 20, "max"))
                rolling_min = groups["close"].transform(lambda x: safe_rolling(x, 20, "min"))
            else:
                rolling_max = safe_rolling(df["close"], 20, "max")
                rolling_min = safe_rolling(df["close"], 20, "min")
            df["recovery_ratio_20d"] = np.where(
                rolling_max > rolling_min, (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-8), 0.5
            )

        # 通道宽度
        if "channel_width_20d" not in df.columns:
            if groups is not None:
                high_20 = groups["close"].transform(lambda x: safe_rolling(x, 20, "max"))
                low_20 = groups["close"].transform(lambda x: safe_rolling(x, 20, "min"))
            else:
                high_20 = safe_rolling(df["close"], 20, "max")
                low_20 = safe_rolling(df["close"], 20, "min")
            df["channel_width_20d"] = (high_20 - low_20) / (df["close"] + 1e-8) * 100

        # 价格区间
        if "price_range_pct" not in df.columns:
            df["price_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-8) * 100

        # MA10 相关
        if "close_vs_ma10_std" not in df.columns:
            ma10 = df.get("ma10", safe_rolling(df["close"], 10, "mean"))
            diff = df["close"] - ma10
            diff_std = safe_rolling(diff, 10, "std")
            df["close_vs_ma10_std"] = diff / (diff_std + 1e-8)

        if "days_near_ma10" not in df.columns:
            ma10 = df.get("ma10", safe_rolling(df["close"], 10, "mean"))
            near_ma10 = (abs(df["close"] - ma10) / (df["close"] + 1e-8) < 0.02).astype(int)
            df["days_near_ma10"] = safe_rolling(near_ma10, 10, "sum")

        if "ma10_cross_count" not in df.columns:
            df["ma10_cross_count"] = 0

        # 量比
        vol = df.get("vol", pd.Series([1] * n))
        if "vol_ma5_ratio" not in df.columns:
            if groups is not None:
                df["vol_ma5_ratio"] = vol / (groups["vol"].transform(lambda x: safe_rolling(x, 5, "mean")) + 1e-8)
            else:
                df["vol_ma5_ratio"] = vol / (safe_rolling(vol, 5, "mean") + 1e-8)
        if "vol_ma20_ratio" not in df.columns:
            if groups is not None:
                df["vol_ma20_ratio"] = vol / (groups["vol"].transform(lambda x: safe_rolling(x, 20, "mean")) + 1e-8)
            else:
                df["vol_ma20_ratio"] = vol / (safe_rolling(vol, 20, "mean") + 1e-8)
        if "volume_shrink_ratio" not in df.columns:
            vol_ma5 = safe_rolling(vol, 5, "mean")
            vol_ma20 = safe_rolling(vol, 20, "mean")
            df["volume_shrink_ratio"] = vol_ma5 / (vol_ma20 + 1e-8)
        if "volume_change" not in df.columns:
            df["volume_change"] = vol.pct_change()

        # 放量突破
        vol_ratio = df.get("volume_ratio", vol / (safe_rolling(vol, 5, "mean") + 1e-8))
        pct_chg_series = df["pct_chg"]
        if "high_volume_breakout" not in df.columns:
            df["high_volume_breakout"] = ((vol_ratio > 2) & (pct_chg_series > 0)).astype(int)
        if "breakout_volume_ratio" not in df.columns:
            breakout = pct_chg_series > 3
            df["breakout_volume_ratio"] = np.where(breakout, vol_ratio, 0)

        # 突破 MA
        if "breakout_ma20" not in df.columns:
            ma20 = df.get("ma_20d", safe_rolling(df["close"], 20, "mean"))
            df["breakout_ma20"] = (df["close"] > ma20).astype(int)
        if "breakout_ma55" not in df.columns:
            if groups is not None:
                ma55 = groups["close"].transform(lambda x: safe_rolling(x, 55, "mean"))
            else:
                ma55 = safe_rolling(df["close"], 55, "mean")
            df["breakout_ma55"] = (df["close"] > ma55).astype(int)

        # 支撑阻力距离
        for period in [10, 20, 55]:
            ds_col = f"dist_to_support_{period}d"
            dr_col = f"dist_to_resistance_{period}d"
            if groups is not None:
                support = groups["close"].transform(lambda x: safe_rolling(x, period, "min"))
                resistance = groups["close"].transform(lambda x: safe_rolling(x, period, "max"))
            else:
                support = safe_rolling(df["close"], period, "min")
                resistance = safe_rolling(df["close"], period, "max")
            if ds_col not in df.columns:
                df[ds_col] = (df["close"] - support) / (df["close"] + 1e-8) * 100
            if dr_col not in df.columns:
                df[dr_col] = (resistance - df["close"]) / (df["close"] + 1e-8) * 100

        # 支撑阻力强度 55d
        if "support_strength_55d" not in df.columns:
            if groups is not None:
                support = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))
            else:
                support = safe_rolling(df["close"], 55, "min")
            df["support_strength_55d"] = (df["close"] - support) / (support + 1e-8) * 100
        if "resistance_strength_55d" not in df.columns:
            if groups is not None:
                resistance = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))
            else:
                resistance = safe_rolling(df["close"], 55, "max")
            df["resistance_strength_55d"] = (resistance - df["close"]) / (resistance + 1e-8) * 100

        # 55d 支撑阻力值
        if "support_55d" not in df.columns:
            if groups is not None:
                df["support_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))
            else:
                df["support_55d"] = safe_rolling(df["close"], 55, "min")
        if "resistance_55d" not in df.columns:
            if groups is not None:
                df["resistance_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))
            else:
                df["resistance_55d"] = safe_rolling(df["close"], 55, "max")

        # 高低点
        if "high_55d" not in df.columns:
            if groups is not None:
                df["high_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))
            else:
                df["high_55d"] = safe_rolling(df["close"], 55, "max")
        if "low_55d" not in df.columns:
            if groups is not None:
                df["low_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))
            else:
                df["low_55d"] = safe_rolling(df["close"], 55, "min")

        # 其他特征
        if "consecutive_new_high" not in df.columns:
            if groups is not None:
                high_10 = groups["close"].transform(lambda x: safe_rolling(x, 10, "max"))
            else:
                high_10 = safe_rolling(df["close"], 10, "max")
            new_high = (df["close"] >= high_10).astype(int)
            df["consecutive_new_high"] = safe_rolling(new_high, 5, "sum")

        if "momentum_acceleration" not in df.columns:
            mom = df["close"].pct_change(5)
            df["momentum_acceleration"] = mom.diff()

        if "is_limit_up" not in df.columns:
            df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

        # OBV（Tushare 已提供则跳过）
        if "obv" not in df.columns:
            obv_sign = np.sign(df["pct_chg"]).fillna(0)
            df["obv"] = (obv_sign * vol).cumsum()
        if "obv_calc" not in df.columns:
            df["obv_calc"] = df["obv"]
        if "obv_ma10" not in df.columns:
            df["obv_ma10"] = safe_rolling(df["obv"], 10, "mean")
        if "obv_trend" not in df.columns:
            df["obv_trend"] = np.sign(df["obv"] - df["obv"].shift(5)).fillna(0)

        # 价格变化相关
        if "price_down_vol_up" not in df.columns:
            df["price_down_vol_up"] = ((df["pct_chg"] < 0) & (vol > vol.shift(1))).astype(int)
        if "price_down_vol_up_count_10d" not in df.columns:
            df["price_down_vol_up_count_10d"] = safe_rolling(df["price_down_vol_up"], 10, "sum")
        if "price_up_vol_down" not in df.columns:
            df["price_up_vol_down"] = ((df["pct_chg"] > 0) & (vol < vol.shift(1))).astype(int)
        if "price_up_vol_down_count_10d" not in df.columns:
            df["price_up_vol_down_count_10d"] = safe_rolling(df["price_up_vol_down"], 10, "sum")

        # 量价相关性
        for period in [10, 20]:
            col = f"volume_price_corr_{period}d"
            if col not in df.columns:
                df[col] = df["close"].rolling(period, min_periods=5).corr(vol)

        # 量价匹配
        if "volume_price_match" not in df.columns:
            vol_up = vol > vol.shift(1)
            price_up = df["pct_chg"] > 0
            df["volume_price_match"] = (vol_up.values == price_up.values).astype(int)
        if "volume_price_match_sum_10d" not in df.columns:
            df["volume_price_match_sum_10d"] = safe_rolling(df["volume_price_match"], 10, "sum")

        # 量能突破
        if "volume_breakout_count_20d" not in df.columns:
            vol_breakout = (vol > safe_rolling(vol, 20, "mean") * 2).astype(int)
            df["volume_breakout_count_20d"] = safe_rolling(vol_breakout, 20, "sum")

        if "volume_rsv_20d" not in df.columns:
            if groups is not None:
                vol_low = groups["vol"].transform(lambda x: safe_rolling(x, 20, "min"))
                vol_high = groups["vol"].transform(lambda x: safe_rolling(x, 20, "max"))
            else:
                vol_low = safe_rolling(vol, 20, "min")
                vol_high = safe_rolling(vol, 20, "max")
            df["volume_rsv_20d"] = np.where(vol_high > vol_low, (vol - vol_low) / (vol_high - vol_low + 1e-8), 0.5)

        if "volume_trend_slope_10d" not in df.columns:
            if groups is not None:
                df["volume_trend_slope_10d"] = groups["vol"].transform(lambda x: x.rolling(10, min_periods=5).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=True))
            else:
                df["volume_trend_slope_10d"] = vol.rolling(10, min_periods=5).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=True)
        if "volume_trend_slope_20d" not in df.columns:
            if groups is not None:
                df["volume_trend_slope_20d"] = groups["vol"].transform(lambda x: x.rolling(20, min_periods=10).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=True))
            else:
                df["volume_trend_slope_20d"] = vol.rolling(20, min_periods=10).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=True)

        # 历史价格位置
        if "price_vs_hist_high" not in df.columns:
            hist_high = safe_rolling(df["close"], 55, "max")
            df["price_vs_hist_high"] = (df["close"] - hist_high) / hist_high * 100
        if "price_vs_hist_mean" not in df.columns:
            hist_mean = safe_rolling(df["close"], 55, "mean")
            df["price_vs_hist_mean"] = (df["close"] - hist_mean) / hist_mean * 100

        # 波动率历史对比
        if "volatility_vs_hist" not in df.columns and "pct_chg" in df.columns:
            vol_20 = df["pct_chg"].rolling(20).std()
            vol_60 = df["pct_chg"].rolling(60).std()
            df["volatility_vs_hist"] = vol_20 / (vol_60 + 1e-8)

        # 前高
        if "prev_high_10d" not in df.columns and "high_10d" in df.columns:
            df["prev_high_10d"] = df["high_10d"].shift(1)
        if "prev_high_20d" not in df.columns and "high_20d" in df.columns:
            df["prev_high_20d"] = df["high_20d"].shift(1)
        if "prev_high_55d" not in df.columns and "high_55d" in df.columns:
            df["prev_high_55d"] = df["high_55d"].shift(1)

        return df

    # ==================== 增强特征 ====================

    def _calc_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算增强特征"""
        df = df.copy()

        if "turnover_rate" in df.columns:
            tr = df["turnover_rate"]
            tr_mean = tr.rolling(20, min_periods=5).mean()
            tr_std = tr.rolling(20, min_periods=5).std()
            df["turnover_zscore"] = (tr - tr_mean) / (tr_std + 1e-8)
            df["turnover_change_rate"] = tr.pct_change(5)
            df["turnover_spike"] = (tr > tr_mean * 2).astype(int)

        if "rsi_6" in df.columns and "kdj_j" in df.columns and "kdj_k" in df.columns:
            df["rsi_kdj_golden_cross"] = ((df["rsi_6"] > 50) & (df["kdj_j"] > df["kdj_k"])).astype(int)
            df["rsi_kdj_strength"] = (df["rsi_6"] / 100 + df["kdj_j"] / 100) / 2
            df["rsi_zone"] = np.where(df["rsi_6"] > 70, 1, np.where(df["rsi_6"] < 30, -1, 0))

        if "close" in df.columns and "vol" in df.columns:
            price_change_10d = df["close"].pct_change(10)
            vol_change_10d = df["vol"].pct_change(10)
            df["volume_price_divergence_strength"] = np.abs(price_change_10d - vol_change_10d)
            df["volume_price_confirm"] = ((price_change_10d > 0) == (vol_change_10d > 0)).astype(int)

        breakout_cols = [c for c in df.columns if "breakout_strength" in c]
        if len(breakout_cols) >= 2:
            df["breakout_strength_avg"] = df[breakout_cols].mean(axis=1)
            df["breakout_strength_max"] = df[breakout_cols].max(axis=1)

        ma_cols = ["ma5", "ma10", "ma_20d", "ma_34d", "ma_55d"]
        available_ma = [c for c in ma_cols if c in df.columns]
        if len(available_ma) >= 3:
            ma_values = df[available_ma].values
            ma_rank_score = np.zeros(len(df))
            for i in range(len(df)):
                row = ma_values[i]
                if not np.isnan(row).any():
                    sorted_idx = np.argsort(row)[::-1]
                    expected = np.arange(len(row))
                    ma_rank_score[i] = 1 - np.abs(sorted_idx - expected).sum() / (len(row) * (len(row) - 1) / 2 + 1e-8)
            df["ma_alignment_score"] = ma_rank_score

        if "momentum_10d" in df.columns and "momentum_acceleration" not in df.columns:
            df["momentum_acceleration"] = df["momentum_10d"].diff(5)

        position_cols = [c for c in df.columns if "price_position" in c]
        if len(position_cols) >= 2:
            df["price_position_avg"] = df[position_cols].mean(axis=1)

        if "return_34d" in df.columns and "volatility_34d" in df.columns:
            df["sharpe_like_34d"] = df["return_34d"] / (df["volatility_34d"] + 1e-8)

        return df

    # ==================== 主入口 ====================

    def compute_all_features(
        self, df_raw: pd.DataFrame, df_market: pd.DataFrame
    ) -> pd.DataFrame:
        """计算全部特征

        Args:
            df_raw: 原始数据（含 OHLCV + Tushare 技术指标）
            df_market: 市场环境数据

        Returns:
            完整特征 DataFrame
        """
        log.info("=" * 50)
        log.info("开始特征工程...")
        n_stocks = df_raw["ts_code"].nunique() if "ts_code" in df_raw.columns else 1
        log.info(f"股票数: {n_stocks}, 总行数: {len(df_raw)}, 初始列数: {len(df_raw.columns)}")

        df = df_raw.copy()

        # 1. 基础特征
        log.info("[1/6] 计算基础技术特征...")
        df = self._calc_basic(df)
        log.info(f"  列数: {len(df.columns)}")

        # 2. 突破特征
        log.info("[2/6] 计算突破特征...")
        df = self._calc_breakout(df)
        log.info(f"  列数: {len(df.columns)}")

        # 3. 市场环境特征
        log.info("[3/6] 计算市场环境特征...")
        df = self._calc_market(df, df_market)
        log.info(f"  列数: {len(df.columns)}")

        # 4. 交互特征
        log.info("[4/6] 计算交互特征...")
        df = self._calc_interaction(df)
        log.info(f"  列数: {len(df.columns)}")

        # 5. 补充特征
        log.info("[5/6] 计算补充特征...")
        df = self._calc_supplementary(df)
        log.info(f"  列数: {len(df.columns)}")

        # 6. 增强特征
        log.info("[6/6] 计算增强特征...")
        df = self._calc_enhanced(df)
        log.info(f"  列数: {len(df.columns)}")

        log.success(f"特征工程完成: {len(df.columns)} 列")
        return df
