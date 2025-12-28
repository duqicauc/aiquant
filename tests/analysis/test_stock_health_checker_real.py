"""
股票健康检查器真实数据测试
"""
import pytest
import os


@pytest.mark.api
@pytest.mark.slow
class TestStockHealthCheckerReal:
    """StockHealthChecker真实数据测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """检查是否有Tushare Token"""
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            pytest.skip("需要设置TUSHARE_TOKEN环境变量")
    
    def test_check_stock_real(self):
        """测试股票健康检查（真实数据）"""
        from src.analysis.stock_health_checker import StockHealthChecker
        
        checker = StockHealthChecker()
        result = checker.check_stock('000001.SZ', days=60)
        
        assert isinstance(result, dict)
        assert 'stock_code' in result
        assert result['stock_code'] == '000001.SZ'
        assert 'overall_score' in result
        assert isinstance(result['overall_score'], (int, float))
        assert 0 <= result['overall_score'] <= 100
    
    def test_get_basic_info_real(self):
        """测试获取基本信息（真实数据）"""
        from src.analysis.stock_health_checker import StockHealthChecker
        
        checker = StockHealthChecker()
        result = checker._get_basic_info('000001.SZ')
        
        assert isinstance(result, dict)
        assert 'name' in result or 'ts_code' in result

