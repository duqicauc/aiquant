"""
BreakoutSampleScreener 单元测试

测试核心判定逻辑，不依赖真实 DataManager（使用 mock）。
"""

import numpy as np
import pandas as pd
import pytest

from src.models.screening.breakout_sample_screener import BreakoutSampleScreener


class MockDataManager:
    """Mock DataManager，仅返回空 DataFrame"""

    def get_stock_list(self, **kwargs):
        return pd.DataFrame({"ts_code": [], "name": [], "list_date": []})

    def get_daily_data(self, *args, **kwargs):
        return pd.DataFrame()

    def get_suspend_info(self, *args, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def screener():
    dm = MockDataManager()
    return BreakoutSampleScreener(data_manager=dm)


# ============================================================================
# _calc_boll_bandwidth
# ============================================================================
class TestCalcBollBandwidth:
    """测试 BOLL 带宽计算"""

    def test_normal_case(self):
        """正常情况：价格稳定波动"""
        close = [10.0, 10.5, 10.2, 10.8, 10.4, 10.6, 10.3, 10.7, 10.5, 10.9,
                 10.6, 10.8, 10.4, 10.7, 10.5, 10.9, 10.6, 10.8, 10.4, 10.7]
        df = pd.DataFrame({"close": close})
        bw = BreakoutSampleScreener._calc_boll_bandwidth(df, period=20)
        # BOLL带宽应该是较小的正值（价格在10附近小幅波动）
        assert 0 < bw < 20, f"BOLL带宽异常: {bw}"

    def test_flat_price(self):
        """价格完全不变 — BOLL带宽 ≈ 0"""
        df = pd.DataFrame({"close": [10.0] * 20})
        bw = BreakoutSampleScreener._calc_boll_bandwidth(df, period=20)
        assert bw == 0.0, f"价格不变时BOLL带宽应为0, 实际: {bw}"

    def test_insufficient_data(self):
        """数据不足时回退到可用数据量"""
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        bw = BreakoutSampleScreener._calc_boll_bandwidth(df, period=20)
        # 用3天数据计算，标准差应该较大
        assert bw > 0

    def test_empty_dataframe(self):
        """空DataFrame应返回异常大值"""
        df = pd.DataFrame({"close": []})
        bw = BreakoutSampleScreener._calc_boll_bandwidth(df, period=20)
        assert bw == 999.0


# ============================================================================
# _is_platform_consolidation
# ============================================================================
class TestIsPlatformConsolidation:
    """测试平台整理判定"""

    def test_true_platform(self):
        """典型平台整理：振幅小，BOLL带宽小"""
        # 价格围绕10小幅波动，振幅约5%
        np.random.seed(42)
        base = 10.0
        close = base + np.random.randn(15) * 0.15
        high = close + np.abs(np.random.randn(15) * 0.1)
        low = close - np.abs(np.random.randn(15) * 0.1)
        df = pd.DataFrame({"close": close, "high": high, "low": low})

        screener = BreakoutSampleScreener(MockDataManager(), config={"platform_amplitude_max": 15, "boll_bandwidth_max": 8})
        assert screener._is_platform_consolidation(df) is True

    def test_too_wide_amplitude(self):
        """振幅过大，不是平台"""
        close = np.linspace(10, 20, 15)  # 涨幅100%
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame({"close": close, "high": high, "low": low})

        screener = BreakoutSampleScreener(MockDataManager())
        assert screener._is_platform_consolidation(df) is False

    def test_too_few_days(self):
        """天数不足"""
        df = pd.DataFrame({"close": [10.0] * 5, "high": [10.5] * 5, "low": [9.5] * 5})
        screener = BreakoutSampleScreener(MockDataManager(), config={"platform_min_days": 10})
        assert screener._is_platform_consolidation(df) is False

    def test_boll_bandwidth_too_wide(self):
        """BOLL带宽过大"""
        close = np.concatenate([np.linspace(10, 15, 8), np.linspace(15, 10, 7)])
        high = close + 0.3
        low = close - 0.3
        df = pd.DataFrame({"close": close, "high": high, "low": low})

        screener = BreakoutSampleScreener(MockDataManager(), config={"boll_bandwidth_max": 1.0})
        assert screener._is_platform_consolidation(df) is False


# ============================================================================
# _is_volume_breakout
# ============================================================================
class TestIsVolumeBreakout:
    """测试放量突破判定"""

    def test_true_breakout(self):
        """典型放量突破"""
        platform_df = pd.DataFrame({
            "high": [10.2] * 10,
            "vol": [1000] * 10,
        })
        t1_row = pd.Series({"close": 10.5, "vol": 2000})  # 突破2.9%，放量2x

        screener = BreakoutSampleScreener(MockDataManager(), config={"breakout_threshold": 2.0, "volume_ratio_min": 1.5})
        assert screener._is_volume_breakout(t1_row, platform_df) is True

    def test_close_not_high_enough(self):
        """收盘价未突破高点"""
        platform_df = pd.DataFrame({"high": [10.2] * 10, "vol": [1000] * 10})
        t1_row = pd.Series({"close": 10.2, "vol": 2000})  # 刚好等于高点，未突破

        screener = BreakoutSampleScreener(MockDataManager())
        assert screener._is_volume_breakout(t1_row, platform_df) is False

    def test_volume_not_enough(self):
        """成交量未放量"""
        platform_df = pd.DataFrame({"high": [10.2] * 10, "vol": [1000] * 10})
        t1_row = pd.Series({"close": 10.5, "vol": 1000})  # 突破但无量

        screener = BreakoutSampleScreener(MockDataManager())
        assert screener._is_volume_breakout(t1_row, platform_df) is False

    def test_zero_volume(self):
        """平台均量为0"""
        platform_df = pd.DataFrame({"high": [10.2] * 10, "vol": [0] * 10})
        t1_row = pd.Series({"close": 10.5, "vol": 2000})

        screener = BreakoutSampleScreener(MockDataManager())
        assert screener._is_volume_breakout(t1_row, platform_df) is False


# ============================================================================
# _calc_pre_t1_return
# ============================================================================
class TestCalcPreT1Return:
    """测试T1前涨幅计算"""

    def test_normal_increase(self):
        """正常上涨情况"""
        df = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]})
        ret = BreakoutSampleScreener._calc_pre_t1_return(df, t1_idx=5, lookback=5)
        # T1前5天到T1前1天: 100 -> 108, 涨幅8%
        assert ret is not None
        assert abs(ret - 8.0) < 0.01

    def test_normal_decrease(self):
        """正常下跌情况"""
        # T1_idx=5, lookback=5: start_idx=0, start_price=110 (idx=0), end_price=102 (idx=4, T1前一天)
        df = pd.DataFrame({"close": [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]})
        ret = BreakoutSampleScreener._calc_pre_t1_return(df, t1_idx=5, lookback=5)
        assert ret is not None
        # (102 - 110) / 110 * 100 = -7.27%
        assert abs(ret - (-7.27)) < 0.1

    def test_insufficient_data(self):
        """数据不足"""
        df = pd.DataFrame({"close": [100.0, 101.0]})
        ret = BreakoutSampleScreener._calc_pre_t1_return(df, t1_idx=1, lookback=5)
        assert ret is None

    def test_zero_start_price(self):
        """起始价格为0"""
        df = pd.DataFrame({"close": [0.0, 100.0, 102.0, 104.0, 106.0, 108.0]})
        ret = BreakoutSampleScreener._calc_pre_t1_return(df, t1_idx=5, lookback=5)
        assert ret is None


# ============================================================================
# _calc_confirm_return
# ============================================================================
class TestCalcConfirmReturn:
    """测试确认涨幅计算（仅用于标签）"""

    def test_positive_return(self):
        """确认上涨"""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 105.0],
            "close": [101.0, 102.0, 105.0, 108.0],
        })
        ret = BreakoutSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        # T1 open=100, 3天后 close=108, 涨幅8%
        assert ret is not None
        assert abs(ret - 8.0) < 0.01

    def test_negative_return(self):
        """确认下跌（负标签）"""
        df = pd.DataFrame({
            "open": [100.0, 99.0, 98.0, 95.0],
            "close": [99.0, 98.0, 95.0, 92.0],
        })
        ret = BreakoutSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        assert ret is not None
        assert ret < 0

    def test_insufficient_future_data(self):
        """未来数据不足"""
        df = pd.DataFrame({"open": [100.0], "close": [101.0]})
        ret = BreakoutSampleScreener._calc_confirm_return(df, t1_idx=0, days=3)
        assert ret is None

    def test_t1_idx_at_end(self):
        """T1在数据末尾"""
        df = pd.DataFrame({"open": [100.0, 101.0], "close": [101.0, 102.0]})
        ret = BreakoutSampleScreener._calc_confirm_return(df, t1_idx=1, days=3)
        assert ret is None


# ============================================================================
# _temporal_downsample
# ============================================================================
class TestTemporalDownsample:
    """测试时间均匀化降采样"""

    def test_downsample_reduces_count(self):
        """降采样应减少样本数"""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        df = pd.DataFrame({"t1_date": dates.strftime("%Y%m%d"), "value": range(1000)})
        result = BreakoutSampleScreener._temporal_downsample(df, samples_per_quarter=50)
        assert len(result) < 1000

    def test_preserves_all_quarters(self):
        """保留所有季度的样本"""
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="W")
        df = pd.DataFrame({"t1_date": dates.strftime("%Y%m%d"), "value": range(len(dates))})
        result = BreakoutSampleScreener._temporal_downsample(df, samples_per_quarter=10)
        # 至少8个季度，每个季度应有样本
        assert len(result) >= 8

    def test_empty_dataframe(self):
        """空DataFrame"""
        df = pd.DataFrame({"t1_date": []})
        result = BreakoutSampleScreener._temporal_downsample(df)
        assert result.empty


# ============================================================================
# 初始化参数
# ============================================================================
class TestInitialization:
    """测试初始化参数"""

    def test_default_config(self):
        """默认配置"""
        screener = BreakoutSampleScreener(MockDataManager())
        assert screener.platform_min_days == 10
        assert screener.platform_max_days == 30
        assert screener.platform_amplitude_max == 15.0
        assert screener.boll_bandwidth_max == 8.0
        assert screener.breakout_threshold == 2.0
        assert screener.volume_ratio_min == 1.5
        assert screener.confirm_min_return == 5.0
        assert screener.pre_t1_return_max == 15.0

    def test_custom_config(self):
        """自定义配置"""
        config = {
            "platform_min_days": 15,
            "platform_max_days": 45,
            "breakout_threshold": 3.0,
            "volume_ratio_min": 2.0,
        }
        screener = BreakoutSampleScreener(MockDataManager(), config=config)
        assert screener.platform_min_days == 15
        assert screener.platform_max_days == 45
        assert screener.breakout_threshold == 3.0
        assert screener.volume_ratio_min == 2.0
