"""
BounceSampleScreener 单元测试

测试核心判定逻辑，不依赖真实 DataManager（使用 mock）。
"""

import numpy as np
import pandas as pd
import pytest

from src.models.screening.bounce_sample_screener import BounceSampleScreener


class MockDataManager:
    """Mock DataManager"""

    def get_stock_list(self, **kwargs):
        return pd.DataFrame({"ts_code": [], "name": [], "list_date": []})

    def get_daily_data(self, *args, **kwargs):
        return pd.DataFrame()

    def get_suspend_info(self, *args, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def screener():
    dm = MockDataManager()
    return BounceSampleScreener(data_manager=dm)


# ============================================================================
# _calc_drawback
# ============================================================================
class TestCalcDrawback:
    """测试回调深度计算"""

    def test_significant_drawback(self):
        """明显回调"""
        high = [20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5]
        low = [h - 0.3 for h in high]
        close = [h - 0.15 for h in high]
        df = pd.DataFrame({"high": high, "low": low, "close": close})

        screener = BounceSampleScreener(MockDataManager(), config={"drawback_min_days": 5})
        info = screener._calc_drawback(df)
        assert info is not None
        assert info["peak_price"] == 20.0
        assert info["trough_price"] == 15.2  # low的最小值
        assert abs(info["drawback_pct"] - 24.0) < 1.0  # (20-15.2)/20*100 = 24%

    def test_mild_drawback(self):
        """轻微回调"""
        high = [10.2] * 20
        low = [9.8] * 20
        close = [10.0] * 20
        df = pd.DataFrame({"high": high, "low": low, "close": close})

        screener = BounceSampleScreener(MockDataManager())
        info = screener._calc_drawback(df)
        assert info is not None
        assert abs(info["drawback_pct"] - 3.92) < 0.1  # (10.2-9.8)/10.2*100

    def test_insufficient_days(self):
        """天数不足"""
        df = pd.DataFrame({"high": [10.0], "low": [9.0], "close": [9.5]})
        screener = BounceSampleScreener(MockDataManager(), config={"drawback_min_days": 5})
        info = screener._calc_drawback(df)
        assert info is None

    def test_zero_peak(self):
        """峰值为0（异常）"""
        df = pd.DataFrame({"high": [0.0] * 10, "low": [0.0] * 10, "close": [0.0] * 10})
        screener = BounceSampleScreener(MockDataManager())
        info = screener._calc_drawback(df)
        assert info is None


# ============================================================================
# _calc_lower_shadow
# ============================================================================
class TestCalcLowerShadow:
    """测试下影线计算"""

    def test_bullish_candle(self):
        """阳线（close > open）"""
        row = pd.Series({"open": 10.0, "close": 11.0, "low": 9.5})
        shadow = BounceSampleScreener._calc_lower_shadow(row)
        # min(close, open) = 10.0, low = 9.5, 下影线 = (10.0 - 9.5) / 11.0 * 100 = 4.55%
        assert abs(shadow - 4.55) < 0.01

    def test_bearish_candle(self):
        """阴线（close < open）"""
        row = pd.Series({"open": 11.0, "close": 10.0, "low": 9.5})
        shadow = BounceSampleScreener._calc_lower_shadow(row)
        # min(close, open) = 10.0, 下影线 = (10.0 - 9.5) / 10.0 * 100 = 5.0%
        assert abs(shadow - 5.0) < 0.01

    def test_doji(self):
        """十字星"""
        row = pd.Series({"open": 10.0, "close": 10.0, "low": 9.8})
        shadow = BounceSampleScreener._calc_lower_shadow(row)
        # min = 10.0, 下影线 = (10.0 - 9.8) / 10.0 * 100 = 2.0%
        assert abs(shadow - 2.0) < 0.01

    def test_zero_close(self):
        """收盘价为0（异常）"""
        row = pd.Series({"open": 10.0, "close": 0.0, "low": 9.5})
        shadow = BounceSampleScreener._calc_lower_shadow(row)
        assert shadow == 0.0

    def test_no_lower_shadow(self):
        """无下影线（low == min(close, open)）"""
        row = pd.Series({"open": 10.0, "close": 11.0, "low": 10.0})
        shadow = BounceSampleScreener._calc_lower_shadow(row)
        assert shadow == 0.0


# ============================================================================
# _calc_rsi
# ============================================================================
class TestCalcRSI:
    """测试RSI计算"""

    def test_strong_uptrend(self):
        """强上升趋势 — RSI应接近100"""
        closes = list(np.linspace(10, 20, 20))  # 连续上涨
        last_close = 21.0
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), last_close, period=14)
        assert rsi is not None
        assert rsi > 70  # 强上升趋势RSI高

    def test_strong_downtrend(self):
        """强下降趋势 — RSI应接近0"""
        closes = list(np.linspace(20, 10, 20))  # 连续下跌
        last_close = 9.0
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), last_close, period=14)
        assert rsi is not None
        assert rsi < 30  # 强下降趋势RSI低

    def test_sideways(self):
        """横盘 — RSI应在40-60之间"""
        np.random.seed(42)
        closes = list(10 + np.random.randn(20) * 0.1)
        last_close = 10.0
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), last_close, period=14)
        assert rsi is not None
        assert 30 < rsi < 70

    def test_insufficient_data(self):
        """数据不足"""
        closes = [10.0, 11.0]
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), 12.0, period=14)
        assert rsi is None

    def test_all_gains(self):
        """全部上涨 — RSI = 100"""
        closes = list(range(10, 30))
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), 30.0, period=14)
        assert rsi is not None
        assert rsi == 100.0

    def test_all_losses(self):
        """全部下跌 — RSI = 0"""
        closes = list(range(30, 10, -1))
        rsi = BounceSampleScreener._calc_rsi(pd.DataFrame({"close": closes}), 9.0, period=14)
        assert rsi is not None
        assert rsi == 0.0


# ============================================================================
# _has_stop_fall_sign
# ============================================================================
class TestHasStopFallSign:
    """测试止跌迹象判定"""

    def test_lower_shadow_sign(self):
        """下影线足够长"""
        t1_row = pd.Series({"open": 10.0, "close": 10.5, "low": 9.5, "vol": 1000})
        draw_df = pd.DataFrame({"vol": [1000] * 10, "close": [10.0] * 10})

        screener = BounceSampleScreener(MockDataManager(), config={"lower_shadow_min": 1.5})
        assert screener._has_stop_fall_sign(t1_row, draw_df) is True

    def test_rsi_oversold(self):
        """RSI超卖"""
        # 连续下跌导致RSI低
        closes = list(np.linspace(20, 10, 20))
        t1_row = pd.Series({"open": 10.0, "close": 9.5, "low": 9.3, "vol": 1000})
        draw_df = pd.DataFrame({"vol": [1000] * 20, "close": closes})

        screener = BounceSampleScreener(MockDataManager(), config={"rsi_oversold_max": 35})
        assert screener._has_stop_fall_sign(t1_row, draw_df) is True

    def test_volume_expansion(self):
        """放量止跌"""
        t1_row = pd.Series({"open": 10.0, "close": 10.2, "low": 9.9, "vol": 2000})
        draw_df = pd.DataFrame({"vol": [800] * 10, "close": [10.0] * 10})  # 前5日均量800

        screener = BounceSampleScreener(MockDataManager())
        assert screener._has_stop_fall_sign(t1_row, draw_df) is True

    def test_no_sign(self):
        """无任何止跌迹象"""
        # 价格横盘波动，RSI≈50；无下影线；无量放大
        np.random.seed(42)
        closes = list(10 + np.random.randn(20) * 0.2)  # 小幅波动，RSI中性
        t1_row = pd.Series({"open": 10.0, "close": 10.0, "low": 10.0, "vol": 1000})  # 无下影线
        draw_df = pd.DataFrame({"vol": [1000] * 20, "close": closes})

        screener = BounceSampleScreener(MockDataManager(), config={"lower_shadow_min": 2.0})
        assert screener._has_stop_fall_sign(t1_row, draw_df) is False


# ============================================================================
# _calc_confirm_return
# ============================================================================
class TestCalcConfirmReturn:
    """测试确认涨幅计算（仅用于标签）"""

    def test_positive_confirm(self):
        """确认上涨"""
        df = pd.DataFrame({
            "open": [10.0, 10.2, 10.5, 11.0],
            "close": [10.2, 10.5, 11.0, 11.5],
        })
        ret = BounceSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        # open=10.0, 3天后close=11.5, 涨幅15%
        assert ret is not None
        assert abs(ret - 15.0) < 0.01

    def test_negative_confirm(self):
        """确认下跌"""
        df = pd.DataFrame({
            "open": [10.0, 9.8, 9.5, 9.0],
            "close": [9.8, 9.5, 9.0, 8.5],
        })
        ret = BounceSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        assert ret is not None
        assert ret < 0

    def test_insufficient_data(self):
        """数据不足"""
        df = pd.DataFrame({"open": [10.0], "close": [10.0]})
        ret = BounceSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        assert ret is None


# ============================================================================
# _temporal_downsample
# ============================================================================
class TestTemporalDownsample:
    """测试时间均匀化降采样"""

    def test_downsample(self):
        """降采样应减少数量"""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        df = pd.DataFrame({"t1_date": dates.strftime("%Y%m%d"), "value": range(500)})
        result = BounceSampleScreener._temporal_downsample(df, samples_per_quarter=30)
        assert len(result) < 500

    def test_empty(self):
        """空DataFrame"""
        df = pd.DataFrame({"t1_date": []})
        result = BounceSampleScreener._temporal_downsample(df)
        assert result.empty


# ============================================================================
# 初始化参数
# ============================================================================
class TestInitialization:
    """测试初始化参数"""

    def test_default_config(self):
        """默认配置"""
        screener = BounceSampleScreener(MockDataManager())
        assert screener.drawback_min_days == 20
        assert screener.drawback_max_days == 60
        assert screener.drawback_min_pct == 20.0
        assert screener.rsi_oversold_max == 35.0
        assert screener.lower_shadow_min == 1.5
        assert screener.confirm_min_return == 5.0
        assert screener.confirm_days == 3

    def test_custom_config(self):
        """自定义配置"""
        config = {
            "drawback_min_days": 30,
            "drawback_max_days": 90,
            "drawback_min_pct": 25.0,
            "rsi_oversold_max": 30.0,
        }
        screener = BounceSampleScreener(MockDataManager(), config=config)
        assert screener.drawback_min_days == 30
        assert screener.drawback_max_days == 90
        assert screener.drawback_min_pct == 25.0
        assert screener.rsi_oversold_max == 30.0
