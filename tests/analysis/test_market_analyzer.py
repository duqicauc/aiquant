"""
市场分析器测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.analysis.market_analyzer import MarketAnalyzer


class TestMarketAnalyzer:
    """MarketAnalyzer测试类"""
    
    def test_init(self):
        """测试初始化"""
        analyzer = MarketAnalyzer()
        assert analyzer.dm is not None
        assert hasattr(analyzer, 'MAJOR_INDICES')
        assert len(analyzer.MAJOR_INDICES) > 0
    
    def test_major_indices(self):
        """测试主要指数定义"""
        analyzer = MarketAnalyzer()
        assert '000001.SH' in analyzer.MAJOR_INDICES
        assert '399001.SZ' in analyzer.MAJOR_INDICES
        assert '399006.SZ' in analyzer.MAJOR_INDICES
    
    def test_analyze_market_structure(self, mock_data_manager):
        """测试分析市场结构"""
        analyzer = MarketAnalyzer()
        analyzer.dm = mock_data_manager
        
        # 模拟返回空数据
        mock_data_manager.get_daily_data.return_value = pd.DataFrame()
        mock_data_manager.batch_get_daily_basic.return_value = pd.DataFrame()
        
        result = analyzer.analyze_market(days=30)
        
        assert isinstance(result, dict)
        assert 'analysis_date' in result
        assert 'indices_analysis' in result
        assert 'market_breadth' in result
        assert 'market_sentiment' in result
        assert 'market_state' in result
        assert 'market_score' in result
        assert 'recommendations' in result
    
    def test_determine_market_state(self):
        """测试市场状态判断"""
        analyzer = MarketAnalyzer()
        
        # 测试牛市状态
        report = {
            'indices_analysis': {'average_score': 75},
            'market_breadth': {'up_ratio': 70},
            'market_sentiment': {'fear_greed_index': 70}
        }
        state, score = analyzer._determine_market_state(report)
        assert isinstance(state, str)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

