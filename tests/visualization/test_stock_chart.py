"""
股票图表可视化测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.visualization.stock_chart import StockChartVisualizer


class TestStockChartVisualizer:
    """StockChartVisualizer测试类"""
    
    @pytest.fixture
    def visualizer(self, mock_data_manager):
        """创建可视化器实例"""
        viz = StockChartVisualizer()
        viz.dm = mock_data_manager
        return viz
    
    @pytest.fixture
    def sample_report(self):
        """创建示例体检报告"""
        return {
            'stock_code': '000001.SZ',
            'check_time': '2024-01-01',
            'basic_info': {
                'name': '平安银行',
                'industry': '银行',
            },
            'technical_analysis': {
                'macd': {'signal': 'bullish'},
                'rsi': 65.0,
                'ma_arrangement': '多头排列',
            },
            'fundamental_analysis': {
                'pe': 8.5,
                'pb': 0.8,
            },
            'model_prediction': {
                'probability': 0.75,
                'score': 85.0,
            },
            'overall_score': 85.0,
            'recommendation': '买入',
        }
    
    def test_init(self, visualizer):
        """测试初始化"""
        assert visualizer.dm is not None
    
    def test_create_comprehensive_chart_empty_data(self, visualizer, sample_report):
        """测试空数据创建图表"""
        # 模拟返回空数据
        visualizer.dm.get_daily_data.return_value = pd.DataFrame()
        
        fig = visualizer.create_comprehensive_chart(
            stock_code='999999.SZ',
            report=sample_report,
            days=120
        )
        
        # 应该返回空的Figure
        assert fig is not None
    
    def test_create_comprehensive_chart_structure(self, visualizer, sample_report):
        """测试创建综合图表结构"""
        # 创建模拟数据
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        df = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 12, 120),
            'high': np.random.uniform(11, 13, 120),
            'low': np.random.uniform(9, 11, 120),
            'close': np.random.uniform(10, 12, 120),
            'vol': np.random.uniform(1000000, 5000000, 120),
        })
        visualizer.dm.get_daily_data.return_value = df
        
        fig = visualizer.create_comprehensive_chart(
            stock_code='000001.SZ',
            report=sample_report,
            days=120
        )
        
        # 应该返回plotly Figure对象
        assert fig is not None
        # 检查是否有子图
        assert hasattr(fig, 'data')
    
    def test_identify_buy_sell_points(self, visualizer, sample_report):
        """测试识别买卖点"""
        # 创建模拟数据
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        df = pd.DataFrame({
            'trade_date': dates,
            'close': np.random.uniform(10, 12, 120),
            'vol': np.random.uniform(1000000, 5000000, 120),
        })
        
        points = visualizer._identify_buy_sell_points(df, sample_report)
        
        assert isinstance(points, dict)
        assert 'buy_points' in points
        assert 'sell_points' in points
        assert isinstance(points['buy_points'], list)
        assert isinstance(points['sell_points'], list)
    
    def test_calculate_macd_series(self, visualizer):
        """测试计算MACD序列"""
        prices = pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15])
        macd_data = visualizer._calculate_macd_series(prices)
        
        assert isinstance(macd_data, dict)
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'hist' in macd_data
    
    def test_calculate_rsi_series(self, visualizer):
        """测试计算RSI序列"""
        prices = pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15])
        rsi = visualizer._calculate_rsi_series(prices, period=14)
        
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)
        # RSI应该在0-100之间
        assert np.all((rsi >= 0) & (rsi <= 100)) or np.all(np.isnan(rsi))
    
    def test_create_indicators_heatmap(self, visualizer, sample_report):
        """测试创建指标热力图"""
        fig = visualizer.create_indicators_heatmap(sample_report)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_get_color(self, visualizer):
        """测试获取颜色"""
        # 测试不同分数对应的颜色
        color1 = visualizer._get_color(90.0)  # 高分
        color2 = visualizer._get_color(50.0)  # 中分
        color3 = visualizer._get_color(20.0)  # 低分
        
        assert isinstance(color1, str)
        assert isinstance(color2, str)
        assert isinstance(color3, str)
        # 高分应该是绿色系，低分应该是红色系
        assert 'green' in color1.lower() or 'blue' in color1.lower()
        assert 'red' in color3.lower() or 'orange' in color3.lower()

