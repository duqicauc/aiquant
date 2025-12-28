"""
市场分析器真实数据测试
"""
import pytest
import os
from datetime import datetime, timedelta


@pytest.mark.api
@pytest.mark.slow
class TestMarketAnalyzerReal:
    """MarketAnalyzer真实数据测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """检查是否有Tushare Token"""
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            pytest.skip("需要设置TUSHARE_TOKEN环境变量")
    
    def test_analyze_market_real(self):
        """测试市场分析（真实数据）"""
        from src.analysis.market_analyzer import MarketAnalyzer
        
        analyzer = MarketAnalyzer()
        result = analyzer.analyze_market(days=30)
        
        assert isinstance(result, dict)
        assert 'market_state' in result
        assert 'market_score' in result
        assert isinstance(result['market_score'], (int, float))
        assert 0 <= result['market_score'] <= 100
    
    def test_analyze_indices_real(self):
        """测试指数分析（真实数据）"""
        from src.analysis.market_analyzer import MarketAnalyzer
        
        analyzer = MarketAnalyzer()
        result = analyzer._analyze_indices(days=30)
        
        assert isinstance(result, dict)
        # 应该至少有一个指数的分析结果
        assert len(result) > 0

