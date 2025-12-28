"""
股票图表可视化测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.visualization.stock_chart import StockChartVisualizer


class TestStockChartVisualizer:
    """股票图表可视化器测试类"""

    @pytest.fixture
    def mock_data_manager(self):
        """模拟DataManager"""
        mock_dm = Mock()
        
        # 创建示例数据
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        mock_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 120),
            'high': np.random.uniform(15, 25, 120),
            'low': np.random.uniform(8, 18, 120),
            'close': np.random.uniform(10, 20, 120),
            'vol': np.random.uniform(1000000, 10000000, 120),
        })
        
        mock_dm.get_daily_data.return_value = mock_data
        return mock_dm

    @pytest.fixture
    def visualizer(self, mock_data_manager):
        """创建可视化器实例"""
        with patch('src.visualization.stock_chart.DataManager', return_value=mock_data_manager):
            viz = StockChartVisualizer()
            viz.dm = mock_data_manager
            return viz

    @pytest.fixture
    def sample_report(self):
        """示例体检报告"""
        return {
            'basic_info': {
                'name': '测试股票',
                'ts_code': '000001.SZ'
            },
            'overall_score': 7.5,
            'recommendation': '买入',
            'technical_analysis': {
                'trend': {
                    'alignment_score': 8.0
                },
                'volume_analysis': {
                    'pv_score': 7.0
                }
            },
            'fundamental_analysis': {
                'financial_score': 6.5
            },
            'model_prediction': {
                'score': 7.8
            },
            'risk_assessment': {
                'risk_score': 6.0
            }
        }

    @pytest.mark.unit
    def test_init(self, visualizer, mock_data_manager):
        """测试初始化"""
        assert visualizer.dm == mock_data_manager

    @pytest.mark.unit
    def test_create_comprehensive_chart_success(self, visualizer, sample_report):
        """测试成功创建综合图表"""
        fig = visualizer.create_comprehensive_chart(
            stock_code='000001.SZ',
            report=sample_report,
            days=120
        )
        
        assert fig is not None
        # 检查图表是否有数据
        assert len(fig.data) > 0

    @pytest.mark.unit
    def test_create_comprehensive_chart_empty_data(self, visualizer, sample_report):
        """测试数据为空的情况"""
        visualizer.dm.get_daily_data.return_value = pd.DataFrame()
        
        fig = visualizer.create_comprehensive_chart(
            stock_code='000001.SZ',
            report=sample_report,
            days=120
        )
        
        # 应该返回空图表
        assert fig is not None

    @pytest.mark.unit
    def test_identify_buy_sell_points(self, visualizer, sample_report):
        """测试识别买卖点"""
        # 创建有明确信号的数据
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        df = pd.DataFrame({
            'trade_date': dates,
            'open': np.random.uniform(10, 20, 50),
            'high': np.random.uniform(15, 25, 50),
            'low': np.random.uniform(8, 18, 50),
            'close': np.linspace(10, 20, 50),  # 上升趋势
            'vol': np.random.uniform(1000000, 10000000, 50),
        })
        
        points = visualizer._identify_buy_sell_points(df, sample_report)
        
        assert 'buy_points' in points
        assert 'sell_points' in points
        assert isinstance(points['buy_points'], list)
        assert isinstance(points['sell_points'], list)

    @pytest.mark.unit
    def test_calculate_macd_series(self, visualizer):
        """测试计算MACD序列"""
        prices = pd.Series(np.linspace(10, 20, 50))
        
        macd_data = visualizer._calculate_macd_series(prices)
        
        assert 'dif' in macd_data
        assert 'dea' in macd_data
        assert 'macd' in macd_data
        assert len(macd_data['dif']) == len(prices)

    @pytest.mark.unit
    def test_calculate_rsi_series(self, visualizer):
        """测试计算RSI序列"""
        prices = pd.Series(np.linspace(10, 20, 50))
        
        rsi = visualizer._calculate_rsi_series(prices)
        
        assert len(rsi) == len(prices)
        assert all(0 <= val <= 100 for val in rsi if not np.isnan(val))

    @pytest.mark.unit
    def test_create_indicators_heatmap(self, visualizer, sample_report):
        """测试创建指标热力图"""
        fig = visualizer.create_indicators_heatmap(sample_report)
        
        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.unit
    def test_create_indicators_heatmap_minimal_report(self, visualizer):
        """测试使用最小报告创建热力图"""
        minimal_report = {
            'technical_analysis': {
                'trend': {'alignment_score': 5.0}
            }
        }
        
        fig = visualizer.create_indicators_heatmap(minimal_report)
        
        assert fig is not None

    @pytest.mark.unit
    def test_get_color(self, visualizer):
        """测试根据评分获取颜色"""
        assert visualizer._get_color(9.0) == '#28a745'  # 绿色
        assert visualizer._get_color(7.0) == '#ffc107'  # 黄色
        assert visualizer._get_color(5.0) == '#ff9800'  # 橙色
        assert visualizer._get_color(3.0) == '#dc3545'  # 红色

    @pytest.mark.unit
    def test_identify_buy_sell_points_exception_handling(self, visualizer, sample_report):
        """测试买卖点识别异常处理"""
        # 创建无效数据
        invalid_df = pd.DataFrame({
            'trade_date': [datetime.now()],
            'close': [10.0],
        })
        
        # 应该能处理异常而不崩溃
        points = visualizer._identify_buy_sell_points(invalid_df, sample_report)
        
        assert 'buy_points' in points
        assert 'sell_points' in points
