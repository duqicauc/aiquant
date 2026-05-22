"""
BreakoutFeatureEngineer 单元测试

测试核心专属特征计算逻辑。
"""

import numpy as np
import pandas as pd
import pytest

from src.features.breakout_feature_engineer import BreakoutFeatureEngineer


class TestBreakoutSpecificFeatures:
    """测试Breakout专属特征"""

    def test_boll_bandwidth(self):
        """BOLL带宽计算"""
        np.random.seed(42)
        close = 10 + np.random.randn(30) * 0.2
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=30),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "open": close - 0.05,
            "vol": [1000] * 30,
            "pct_chg": [0.5] * 30,
        })
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "boll_bandwidth_20d" in result.columns
        assert result["boll_bandwidth_20d"].iloc[-1] < 10

    def test_platform_amplitude(self):
        """平台振幅计算"""
        close = np.linspace(10, 11, 20)
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close - 0.1,
            "vol": [1000] * 20,
            "pct_chg": [0.5] * 20,
        })
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "platform_amplitude_10d" in result.columns
        assert result["platform_amplitude_10d"].iloc[-1] > 5

    def test_breakout_strength(self):
        """突破强度计算"""
        close = np.concatenate([np.linspace(10, 12, 20), [13.0]])
        high = np.concatenate([np.linspace(10.2, 12.2, 20), [13.2]])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=21),
            "ts_code": "000001.SZ",
            "close": close,
            "high": high,
            "low": close - 0.2,
            "open": close - 0.1,
            "vol": [1000] * 21,
            "pct_chg": [0.5] * 21,
        })
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "breakout_strength_vs_20d_high" in result.columns
        assert result["breakout_strength_vs_20d_high"].iloc[-1] > 0

    def test_volume_breakout_ratio(self):
        """成交量突破比"""
        vol = [1000] * 19 + [3000]
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": [10.0] * 20,
            "high": [10.2] * 20,
            "low": [9.8] * 20,
            "open": [10.0] * 20,
            "vol": vol,
            "pct_chg": [0.0] * 20,
        })
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "volume_breakout_ratio" in result.columns
        assert result["volume_breakout_ratio"].iloc[-1] > 2

    def test_ma_alignment(self):
        """均线多头排列得分"""
        close = np.cumsum(np.ones(40) * 0.1) + 10
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=40),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "open": close - 0.05,
            "vol": [1000] * 40,
            "pct_chg": [0.5] * 40,
        })
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma_20d"] = df["close"].rolling(20).mean()
        df["ma_34d"] = df["close"].rolling(34).mean()
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "ma_bull_alignment" in result.columns
        assert result["ma_bull_alignment"].iloc[-1] > 0.8

    def test_resistance_touch_count(self):
        """阻力位触碰次数"""
        close = np.array([10.0] * 15 + [10.5, 10.5, 10.5, 10.5, 10.5])
        high = np.array([10.3] * 15 + [10.8, 10.8, 10.8, 10.8, 10.8])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": close,
            "high": high,
            "low": close - 0.3,
            "open": close - 0.1,
            "vol": [1000] * 20,
            "pct_chg": [0.0] * 20,
        })
        fe = BreakoutFeatureEngineer()
        result = fe._calc_per_stock_breakout(df)
        assert "resistance_touch_count_30d" in result.columns
        assert result["resistance_touch_count_30d"].iloc[-1] >= 3
