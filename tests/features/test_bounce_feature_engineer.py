"""
BounceFeatureEngineer 单元测试

测试核心专属特征计算逻辑。
"""

import numpy as np
import pandas as pd
import pytest

from src.features.bounce_feature_engineer import BounceFeatureEngineer


class TestBounceSpecificFeatures:
    """测试Bounce专属特征"""

    def test_drawback_pct(self):
        """回调幅度计算"""
        close = np.concatenate([np.linspace(20, 10, 40), [10.5]])
        high = np.concatenate([np.linspace(20, 10, 40) + 0.5, [11.0]])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=41),
            "ts_code": "000001.SZ",
            "close": close,
            "high": high,
            "low": close - 0.5,
            "open": close - 0.2,
            "vol": [1000] * 41,
            "pct_chg": [-1.0] * 41,
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "drawback_pct_60d" in result.columns
        # 从高点20回落到10.5，回调幅度约47.5%
        assert result["drawback_pct_60d"].iloc[-1] > 40

    def test_rsi_oversold_depth(self):
        """RSI超卖深度"""
        np.random.seed(42)
        # 连续下跌使RSI降低
        close = np.concatenate([np.linspace(20, 10, 30), np.linspace(10, 9, 10)])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=40),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close - 0.1,
            "vol": [1000] * 40,
            "pct_chg": [-1.0] * 40,
            "rsi_6": [20.0] * 40,  # 模拟RSI=20（超卖）
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "rsi_oversold_depth" in result.columns
        # RSI=20, depth = max(0, 35-20) = 15
        assert result["rsi_oversold_depth"].iloc[-1] == 15

    def test_lower_shadow_ratio(self):
        """下影线比例"""
        close = [10.0] * 15 + [10.5, 10.2, 10.0, 9.5, 9.0]
        high = [10.2] * 15 + [10.7, 10.4, 10.2, 9.8, 9.5]
        low = [9.8] * 15 + [10.3, 9.5, 9.0, 8.5, 8.0]
        open_p = [10.1] * 15 + [10.6, 10.3, 10.1, 9.8, 9.2]
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": close,
            "high": high,
            "low": low,
            "open": open_p,
            "vol": [1000] * 20,
            "pct_chg": [0.0] * 20,
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "lower_shadow_ratio" in result.columns
        # 最后一日: open=9.8, close=9.5, low=8.5
        # min(close,open)=9.5, lower_shadow=(9.5-8.5)/9.5*100=10.53%
        assert result["lower_shadow_ratio"].iloc[-1] > 10

    def test_hammer_pattern(self):
        """锤子线形态"""
        close = [10.0] * 18 + [10.5, 10.5]
        high = [10.2] * 18 + [10.2, 10.6]
        low = [9.8] * 18 + [8.0, 9.8]  # 倒数第二日低点很低（下影线长）
        open_p = [10.0] * 18 + [10.0, 10.0]
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": close,
            "high": high,
            "low": low,
            "open": open_p,
            "vol": [1000] * 20,
            "pct_chg": [0.0] * 20,
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "hammer_pattern" in result.columns
        # 倒数第二日(index=18): close=10.5, open=10.0, low=8.0
        # body=0.5, lower_shadow=(10.0-8.0)/10.5*100=19.05%
        # threshold = body/close*100*2 = 0.5/10.5*100*2 = 9.52%
        # 19.05 > 9.52, 满足条件
        assert result["hammer_pattern"].iloc[18] == 1

    def test_volume_contraction(self):
        """成交量萎缩比"""
        vol = [2000] * 15 + [500] * 5  # 后期成交量萎缩
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
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "volume_contraction_ratio" in result.columns
        # 后期 vol=500, vol_ma5=500, vol_ma20≈1625, ratio≈0.3
        assert result["volume_contraction_ratio"].iloc[-1] < 0.5

    def test_macd_divergence(self):
        """MACD底背离检测"""
        # 价格创新低，MACD未创新低（需要至少40行数据，因为window=20）
        close1 = np.linspace(20, 12, 20)  # 第一阶段下跌
        close2 = np.linspace(11.5, 6.3, 25)  # 第二阶段更低，但MACD抬升
        close = np.concatenate([close1, close2])
        macd1 = np.linspace(2.0, 0.2, 20)
        macd2 = np.linspace(0.1, 0.5, 25)  # MACD抬升
        macd = np.concatenate([macd1, macd2])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=45),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close - 0.1,
            "vol": [1000] * 45,
            "pct_chg": [0.0] * 45,
            "macd": macd,
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "macd_divergence_bull" in result.columns
        # 最后几日价格创新低但MACD抬升，应有底背离信号
        assert result["macd_divergence_bull"].iloc[-1] == 1

    def test_support_test_count(self):
        """支撑位测试次数"""
        close = np.array([10.0] * 15 + [10.1, 10.0, 10.1, 10.0, 10.1])
        low = np.array([9.8] * 15 + [9.9, 9.8, 9.9, 9.8, 9.9])
        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "ts_code": "000001.SZ",
            "close": close,
            "high": close + 0.2,
            "low": low,
            "open": close - 0.1,
            "vol": [1000] * 20,
            "pct_chg": [0.0] * 20,
        })
        fe = BounceFeatureEngineer()
        result = fe._calc_per_stock_bounce(df)
        assert "support_test_count_20d" in result.columns
        # 后5日多次测试9.8支撑位
        assert result["support_test_count_20d"].iloc[-1] >= 2
