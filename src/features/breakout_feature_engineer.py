"""
BreakoutFeatureEngineer — v3.1.0 突破识别模型专属特征工程

在基础特征（FeatureEngineer）之上，添加识别"平台整理+放量突破"形态的专属特征。

核心原则:
1. 所有特征严格使用 T1 之前的数据
2. 复用 FeatureEngineer 的基础特征计算
3. 专属特征聚焦于：平台识别、突破强度、量能确认

Author: AIQuant Team
Version: 3.1.0
"""

import warnings

import numpy as np
import pandas as pd

from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import log

warnings.filterwarnings("ignore")


class BreakoutFeatureEngineer:
    """
    突破识别模型特征工程器

    专属特征维度:
    - 平台识别: boll_bandwidth, platform_squeeze_days, platform_amplitude
    - 突破强度: breakout_strength_vs_high, breakout_gap
    - 量能确认: volume_breakout_ratio, platform_volume_trend
    - 前期状态: pre_breakout_volatility, resistance_touch_count, pre_t1_trend
    """

    def __init__(self):
        self.base_engineer = FeatureEngineer()

    def compute_all_features(self, df_raw: pd.DataFrame, df_market: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算全部特征（基础 + Breakout专属）

        Args:
            df_raw: 原始数据（含 OHLCV + Tushare 技术指标）
            df_market: 市场环境数据（可选）

        Returns:
            完整特征 DataFrame
        """
        if df_market is None:
            df_market = pd.DataFrame()

        # 1. 基础特征
        log.info("[Breakout] 计算基础特征...")
        df = self.base_engineer.compute_all_features(df_raw, df_market)

        # 2. Breakout专属特征
        log.info("[Breakout] 计算专属特征...")
        df = self._calc_breakout_specific(df)

        log.success(f"[Breakout] 特征工程完成: {len(df.columns)} 列")
        return df

    # ------------------------------------------------------------------
    # Breakout 专属特征
    # ------------------------------------------------------------------

    def _calc_breakout_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Breakout专属特征"""
        df = df.copy()
        group_key = "sample_id" if "sample_id" in df.columns else "ts_code"
        has_group = group_key in df.columns

        # 按股票分组计算（使用 pd.concat 避免 groupby.apply 列丢失问题）
        if has_group:
            results = []
            for name, group in df.groupby(group_key, sort=False):
                results.append(self._calc_per_stock_breakout(group))
            df = pd.concat(results, ignore_index=True)
        else:
            df = self._calc_per_stock_breakout(df)

        return df

    def _calc_per_stock_breakout(self, g: pd.DataFrame) -> pd.DataFrame:
        """单只股票Breakout专属特征计算"""
        g = g.sort_values("trade_date").copy()

        if "close" not in g.columns or len(g) < 20:
            return g

        close = g["close"]
        high = g["high"]
        low = g["low"]
        vol = g["vol"] if "vol" in g.columns else pd.Series([0] * len(g), index=g.index)

        # ========== 平台识别特征 ==========

        # 1. BOLL带宽(20日) = (upper - lower) / mid * 100
        ma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        g["boll_bandwidth_20d"] = np.where(ma20 > 0, (upper - lower) / ma20 * 100, 0)

        # 2. 平台振幅(10/20/30日) = (high - low) / low * 100
        g["platform_amplitude_10d"] = self._rolling_amplitude(close, high, low, 10)
        g["platform_amplitude_20d"] = self._rolling_amplitude(close, high, low, 20)
        g["platform_amplitude_30d"] = self._rolling_amplitude(close, high, low, 30)

        # 3. 平台整理天数 — 连续N日振幅<阈值（滑动窗口内连续低振幅天数）
        amplitude_10d = g["platform_amplitude_10d"]
        g["platform_squeeze_days"] = self._count_squeeze_days(amplitude_10d, threshold=5.0)

        # ========== 突破强度特征 ==========

        # 4. 突破强度(相对20日高点) = (close - high_20d.shift(1)) / high_20d.shift(1) * 100
        high_20d = high.rolling(20, min_periods=10).max()
        prev_high_20d = high_20d.shift(1)
        g["breakout_strength_vs_20d_high"] = np.where(
            prev_high_20d > 0, (close - prev_high_20d) / prev_high_20d * 100, 0
        )

        # 5. 突破强度(相对55日高点)
        high_55d = high.rolling(55, min_periods=20).max()
        prev_high_55d = high_55d.shift(1)
        g["breakout_strength_vs_55d_high"] = np.where(
            prev_high_55d > 0, (close - prev_high_55d) / prev_high_55d * 100, 0
        )

        # 6. 跳空突破缺口 = (open - prev_high) / prev_high * 100
        g["breakout_gap"] = np.where(
            prev_high_20d > 0, (g["open"] - prev_high_20d) / prev_high_20d * 100, 0
        )

        # ========== 量能确认特征 ==========

        # 7. 成交量突破比 = vol / vol_ma20
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        g["volume_breakout_ratio"] = np.where(vol_ma20 > 0, vol / vol_ma20, 1)

        # 8. 平台期成交量趋势（平台期10日内成交量斜率）
        g["platform_volume_trend"] = vol.rolling(10, min_periods=5).apply(
            lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 3 else 0, raw=True
        )

        # 9. 成交量变异系数（平台期）
        vol_mean_10d = vol.rolling(10, min_periods=5).mean()
        vol_std_10d = vol.rolling(10, min_periods=5).std()
        g["volume_cv_10d"] = np.where(vol_mean_10d > 0, vol_std_10d / vol_mean_10d, 0)

        # ========== 前期状态特征 ==========

        # 10. 突破前波动率(20日) / 波动率(60日) — 波动率压缩信号
        if "pct_chg" in g.columns:
            vol_20d = g["pct_chg"].rolling(20, min_periods=10).std()
            vol_60d = g["pct_chg"].rolling(60, min_periods=20).std()
            g["pre_breakout_volatility_ratio"] = np.where(vol_60d > 0, vol_20d / vol_60d, 1)

        # 11. 阻力位触碰次数(30日) — 高点被接近但未突破的次数
        g["resistance_touch_count_30d"] = self._count_resistance_touches(close, high, window=30, threshold_pct=2.0)

        # 12. 突破前趋势斜率(20日) = 价格变化率
        g["pre_t1_trend_20d"] = close.diff(20) / close.shift(20) * 100

        # 13. 突破前RSI(6) — 确认突破时RSI不过热(<70)
        if "rsi_6" in g.columns:
            g["rsi_at_breakout_zone"] = np.where(g["rsi_6"] > 70, 1, np.where(g["rsi_6"] < 30, -1, 0))

        # 14. 均线多头排列得分(5/10/20/34)
        ma_cols = ["ma5", "ma10", "ma_20d", "ma_34d"]
        available = [c for c in ma_cols if c in g.columns]
        if len(available) >= 3:
            g["ma_bull_alignment"] = self._calc_ma_alignment(g, available)

        # 15. 价格相对平台位置 = (close - platform_low) / (platform_high - platform_low)
        platform_high_20d = high.rolling(20, min_periods=10).max()
        platform_low_20d = low.rolling(20, min_periods=10).min()
        g["price_vs_platform_position"] = np.where(
            platform_high_20d > platform_low_20d,
            (close - platform_low_20d) / (platform_high_20d - platform_low_20d),
            0.5
        )

        return g

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_amplitude(close, high, low, window: int) -> pd.Series:
        """计算N日振幅 = (high_max - low_min) / low_min * 100"""
        high_max = high.rolling(window, min_periods=window // 2).max()
        low_min = low.rolling(window, min_periods=window // 2).min()
        return np.where(low_min > 0, (high_max - low_min) / low_min * 100, 0)

    @staticmethod
    def _count_squeeze_days(amplitude: pd.Series, threshold: float = 5.0) -> pd.Series:
        """
        计算连续低振幅天数（平台整理信号）
        滑动窗口内连续振幅<阈值的最大天数
        """
        is_squeeze = amplitude < threshold
        # 计算连续True的长度（反向）
        result = pd.Series(0, index=amplitude.index)
        for i in range(len(amplitude)):
            if pd.isna(amplitude.iloc[i]):
                continue
            count = 0
            for j in range(i, max(0, i - 30), -1):
                if is_squeeze.iloc[j]:
                    count += 1
                else:
                    break
            result.iloc[i] = count
        return result

    @staticmethod
    def _count_resistance_touches(close, high, window: int, threshold_pct: float) -> pd.Series:
        """
        计算N日内阻力位被触碰但未突破的次数
        条件: 当日high >= 窗口内high_max * (1 - threshold_pct/100) 且 close < high_max
        """
        high_max = high.rolling(window, min_periods=window // 2).max()
        threshold = high_max * (1 - threshold_pct / 100)

        is_touch = (high >= threshold) & (close < high_max)
        # 滚动计数（过去window天内满足条件的次数）
        return is_touch.rolling(window, min_periods=window // 2).sum()

    @staticmethod
    def _calc_ma_alignment(g: pd.DataFrame, ma_cols: list) -> pd.Series:
        """计算均线多头排列得分(0-1)"""
        ma_values = g[ma_cols].values
        score = pd.Series(0.0, index=g.index)
        for i in range(len(g)):
            row = ma_values[i]
            if np.isnan(row).any():
                continue
            sorted_idx = np.argsort(row)[::-1]  # 从大到小
            expected = np.arange(len(row))       # [0,1,2,3] 表示 ma5 > ma10 > ma20 > ma34
            # 完美多头排列时 sorted_idx = [0,1,2,3], 与 expected 完全一致
            max_disorder = len(row) * (len(row) - 1) / 2
            disorder = np.abs(sorted_idx - expected).sum()
            score.iloc[i] = 1 - disorder / (max_disorder + 1e-8)
        return score
