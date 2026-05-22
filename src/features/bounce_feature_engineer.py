"""
BounceFeatureEngineer — v3.1.0 超跌反弹模型专属特征工程

在基础特征（FeatureEngineer）之上，添加识别"深度回调+止跌反弹"形态的专属特征。

核心原则:
1. 所有特征严格使用 T1 之前的数据
2. 复用 FeatureEngineer 的基础特征计算
3. 专属特征聚焦于：回调深度、超卖程度、止跌信号、量能萎缩

Author: AIQuant Team
Version: 3.1.0
"""

import warnings

import numpy as np
import pandas as pd

from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import log

warnings.filterwarnings("ignore")


class BounceFeatureEngineer:
    """
    超跌反弹模型特征工程器

    专属特征维度:
    - 回调深度: drawback_pct, days_since_high, max_drawback_depth
    - 超卖程度: rsi_oversold_depth, cci_extreme, wr_extreme
    - 止跌信号: lower_shadow_ratio, hammer_pattern, doji_near_support
    - 量能萎缩: volume_contraction_ratio, volume_vs_drawback_corr
    - 背离信号: macd_divergence_bull, rsi_divergence_bull
    - 支撑测试: support_test_count, boll_lower_distance
    """

    def __init__(self):
        self.base_engineer = FeatureEngineer()

    def compute_all_features(self, df_raw: pd.DataFrame, df_market: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算全部特征（基础 + Bounce专属）

        Args:
            df_raw: 原始数据（含 OHLCV + Tushare 技术指标）
            df_market: 市场环境数据（可选）

        Returns:
            完整特征 DataFrame
        """
        if df_market is None:
            df_market = pd.DataFrame()

        # 1. 基础特征
        log.info("[Bounce] 计算基础特征...")
        df = self.base_engineer.compute_all_features(df_raw, df_market)

        # 2. Bounce专属特征
        log.info("[Bounce] 计算专属特征...")
        df = self._calc_bounce_specific(df)

        log.success(f"[Bounce] 特征工程完成: {len(df.columns)} 列")
        return df

    # ------------------------------------------------------------------
    # Bounce 专属特征
    # ------------------------------------------------------------------

    def _calc_bounce_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Bounce专属特征"""
        df = df.copy()
        group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
        has_group = group_key in df.columns

        if has_group:
            results = []
            for name, group in df.groupby(group_key, sort=False):
                results.append(self._calc_per_stock_bounce(group))
            df = pd.concat(results, ignore_index=True)
        else:
            df = self._calc_per_stock_bounce(df)

        return df

    def _calc_per_stock_bounce(self, g: pd.DataFrame) -> pd.DataFrame:
        """单只股票Bounce专属特征计算"""
        g = g.sort_values("trade_date").copy()

        if "close" not in g.columns or len(g) < 20:
            return g

        close = g["close"]
        high = g["high"]
        low = g["low"]
        open_p = g["open"]
        vol = g["vol"] if "vol" in g.columns else pd.Series([0] * len(g), index=g.index)

        # ========== 回调深度特征 ==========

        # 1. 60日回调幅度 = (60日high - close) / 60日high * 100
        high_60d = high.rolling(60, min_periods=20).max()
        g["drawback_pct_60d"] = np.where(high_60d > 0, (high_60d - close) / high_60d * 100, 0)

        # 2. 20日回调幅度（短期回调）
        high_20d = high.rolling(20, min_periods=10).max()
        g["drawback_pct_20d"] = np.where(high_20d > 0, (high_20d - close) / high_20d * 100, 0)

        # 3. 距高点天数 — 距离最近高点的交易日数
        g["days_since_high_60d"] = self._days_since_high(close, high, window=60)

        # 4. 最大回撤深度(60日) — 从高点到当前的最大回落
        cummax = close.cummax()
        g["max_drawdown_depth"] = np.where(cummax > 0, (cummax - close) / cummax * 100, 0)

        # ========== 超卖程度特征 ==========

        # 5. RSI超卖深度 = max(0, 35 - rsi_6) — 正值表示超卖程度
        if "rsi_6" in g.columns:
            g["rsi_oversold_depth"] = np.maximum(0, 35 - g["rsi_6"])
            g["rsi_oversold_flag"] = (g["rsi_6"] < 30).astype(int)
            g["rsi_extreme_zone"] = np.where(g["rsi_6"] < 20, 2, np.where(g["rsi_6"] < 30, 1, 0))

        # 6. CCI极端值（<-100为超卖）
        if "cci" in g.columns:
            g["cci_oversold_depth"] = np.maximum(0, -100 - g["cci"])

        # 7. WR极端值（>80为超卖）
        if "wr" in g.columns:
            g["wr_oversold_depth"] = np.maximum(0, g["wr"] - 80)

        # ========== 止跌信号特征 ==========

        # 8. 下影线比例 = (min(close, open) - low) / close * 100
        g["lower_shadow_ratio"] = np.where(
            close > 0, (np.minimum(close, open_p) - low) / close * 100, 0
        )

        # 9. 上影线比例（冲高回落信号，负相关）
        g["upper_shadow_ratio"] = np.where(
            close > 0, (high - np.maximum(close, open_p)) / close * 100, 0
        )

        # 10. 锤子线形态（下影线 > 实体2倍）
        body = np.abs(close - open_p)
        g["hammer_pattern"] = np.where(
            (body > 0) & (g["lower_shadow_ratio"] > body / close * 100 * 2), 1, 0
        )

        # 11. 十字星（实体极小，预示变盘）
        g["doji_pattern"] = np.where(body / close * 100 < 0.5, 1, 0)

        # 12. 连续下跌后阳线（止跌确认）
        g["bull_after_decline"] = ((close > open_p) & (close.shift(1) < open_p.shift(1)) & (close.shift(2) < open_p.shift(2))).astype(int)

        # ========== 量能萎缩特征 ==========

        # 13. 成交量萎缩比 = vol_ma5 / vol_ma20（<1表示萎缩）
        vol_ma5 = vol.rolling(5, min_periods=3).mean()
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        g["volume_contraction_ratio"] = np.where(vol_ma20 > 0, vol_ma5 / vol_ma20, 1)

        # 14. 回调期成交量趋势（下跌过程中成交量是否萎缩）
        g["volume_trend_decline"] = vol.rolling(20, min_periods=10).apply(
            lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 5 else 0, raw=True
        )

        # 15. 地量信号（成交量创20日新低）
        vol_low_20d = vol.rolling(20, min_periods=10).min()
        g["extreme_low_volume"] = (vol <= vol_low_20d * 1.05).astype(int)

        # ========== 背离信号特征 ==========

        # 16. MACD底背离 — 价格创新低但MACD未创新低
        if "macd" in g.columns:
            g["macd_divergence_bull"] = self._detect_macd_divergence(close, g["macd"], window=20)

        # 17. RSI底背离 — 价格创新低但RSI未创新低
        if "rsi_6" in g.columns:
            g["rsi_divergence_bull"] = self._detect_rsi_divergence(close, g["rsi_6"], window=20)

        # ========== 支撑测试特征 ==========

        # 18. 支撑位测试次数(20日) — 低点被接近但未跌破的次数
        low_20d = low.rolling(20, min_periods=10).min()
        support_threshold = low_20d * 1.02  # 低点上方2%
        g["support_test_count_20d"] = ((low <= support_threshold) & (close > low_20d)).rolling(20, min_periods=10).sum()

        # 19. 距BOLL下轨距离 = (close - boll_lower) / boll_mid * 100
        if "boll_lower" in g.columns and "boll_mid" in g.columns:
            g["boll_lower_distance"] = np.where(
                g["boll_mid"] > 0, (close - g["boll_lower"]) / g["boll_mid"] * 100, 0
            )
            g["boll_lower_touch"] = (close <= g["boll_lower"] * 1.02).astype(int)

        # 20. 价格相对60日低点位置 = (close - 60日low) / (60日high - 60日low)
        low_60d = low.rolling(60, min_periods=20).min()
        g["price_vs_60d_range"] = np.where(
            high_60d > low_60d, (close - low_60d) / (high_60d - low_60d), 0.5
        )

        # 21. KDJ超卖金叉（K上穿D且都在超卖区）
        if "kdj_k" in g.columns and "kdj_d" in g.columns:
            g["kdj_golden_cross_oversold"] = (
                (g["kdj_k"] > g["kdj_d"]) &
                (g["kdj_k"].shift(1) <= g["kdj_d"].shift(1)) &
                (g["kdj_k"] < 30)
            ).astype(int)

        # 22. 波动率收缩（回调后期波动率下降，预示变盘）
        if "pct_chg" in g.columns:
            vol_5d = g["pct_chg"].rolling(5, min_periods=3).std()
            vol_20d = g["pct_chg"].rolling(20, min_periods=10).std()
            g["volatility_contraction"] = np.where(vol_20d > 0, vol_5d / vol_20d, 1)

        # ========== v3.1.0 新增专属特征 ==========

        # 23. 回调速度 = 回落幅度 / 回调天数（急跌 vs 缓跌）
        days_since_high = g["days_since_high_60d"].replace(0, np.nan)
        g["drawback_velocity"] = np.where(
            days_since_high > 0, g["drawback_pct_60d"] / days_since_high, 0
        )

        # 24. 反弹准备度综合得分（多信号共振）
        rsi_score = (g["rsi_6"] < 35).astype(int) if "rsi_6" in g.columns else 0
        shadow_score = (g["lower_shadow_ratio"] >= 2.5).astype(int)
        vol_contract_score = (g["volume_contraction_ratio"] < 0.8).astype(int)
        divergence_score = g["macd_divergence_bull"] if "macd_divergence_bull" in g.columns else 0
        g["bounce_readiness"] = rsi_score + shadow_score + vol_contract_score + divergence_score

        # 25. 支撑位密度（20日低点集中度，标准差倒数）
        low_20d_roll = low.rolling(20, min_periods=10)
        support_std = low_20d_roll.std()
        g["support_density_20d"] = np.where(support_std > 0, 1 / support_std, 0)

        # 26. 量能干涸度 = vol_ma5 / vol_ma60（地量信号）
        vol_ma60 = vol.rolling(60, min_periods=20).mean()
        g["volume_dryness"] = np.where(vol_ma60 > 0, vol_ma5 / vol_ma60, 1)

        # 27. 价格相对MA20偏离度
        if "ma20" in g.columns:
            g["price_vs_ma20_distance"] = np.where(
                g["ma20"] > 0, (close - g["ma20"]) / g["ma20"] * 100, 0
            )

        # 28. 连阴天数（恐慌程度）
        bearish = (close < open_p).astype(int)
        g["consecutive_down_days"] = bearish.groupby(
            (bearish != bearish.shift()).cumsum()
        ).cumsum() * bearish

        # 29. 反转强度指数 = (下影线% + 实体阳线%) / 前5日平均振幅
        body_pct = np.where(close > 0, (close - open_p) / close * 100, 0)
        amplitude_5d = ((high - low) / close * 100).rolling(5, min_periods=3).mean()
        g["reversal_strength_index"] = np.where(
            amplitude_5d > 0, (g["lower_shadow_ratio"] + np.maximum(body_pct, 0)) / amplitude_5d, 0
        )

        return g

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _days_since_high(close: pd.Series, high: pd.Series, window: int) -> pd.Series:
        """计算距离最近高点的交易日数"""
        result = pd.Series(0, index=close.index, dtype=float)
        for i in range(len(close)):
            if i == 0:
                continue
            # 往前找window天内的高点位置
            start = max(0, i - window + 1)
            window_high_idx = high.iloc[start:i+1].idxmax()
            if window_high_idx in close.index:
                idx_pos = close.index.get_loc(window_high_idx)
                result.iloc[i] = i - idx_pos
        return result

    @staticmethod
    def _detect_macd_divergence(close: pd.Series, macd: pd.Series, window: int) -> pd.Series:
        """
        检测MACD底背离（价格创新低，MACD未创新低）
        返回: 1=底背离, 0=无背离, -1=顶背离
        """
        result = pd.Series(0, index=close.index)
        for i in range(window, len(close)):
            # 当前window内低点
            curr_low = close.iloc[i - window:i + 1].min()
            curr_macd_low = macd.iloc[i - window:i + 1].min()
            # 前一个window内低点
            if i - 2 * window >= 0:
                prev_low = close.iloc[i - 2 * window:i - window + 1].min()
                prev_macd_low = macd.iloc[i - 2 * window:i - window + 1].min()
                # 底背离: 价格更低，MACD更高
                if curr_low < prev_low and curr_macd_low > prev_macd_low:
                    result.iloc[i] = 1
                # 顶背离
                elif curr_low > prev_low and curr_macd_low < prev_macd_low:
                    result.iloc[i] = -1
        return result

    @staticmethod
    def _detect_rsi_divergence(close: pd.Series, rsi: pd.Series, window: int) -> pd.Series:
        """
        检测RSI底背离（价格创新低，RSI未创新低）
        返回: 1=底背离, 0=无背离
        """
        result = pd.Series(0, index=close.index)
        for i in range(window, len(close)):
            curr_low = close.iloc[i - window:i + 1].min()
            curr_rsi_low = rsi.iloc[i - window:i + 1].min()
            if i - 2 * window >= 0:
                prev_low = close.iloc[i - 2 * window:i - window + 1].min()
                prev_rsi_low = rsi.iloc[i - 2 * window:i - window + 1].min()
                if curr_low < prev_low and curr_rsi_low > prev_rsi_low:
                    result.iloc[i] = 1
        return result
