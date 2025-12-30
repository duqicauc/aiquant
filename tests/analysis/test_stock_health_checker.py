"""
股票健康检查器测试
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.analysis.stock_health_checker import StockHealthChecker


class TestStockHealthChecker:
    """StockHealthChecker测试类"""
    
    def test_init(self):
        """测试初始化"""
        checker = StockHealthChecker()
        assert checker.dm is not None
    
    def test_check_stock_structure(self, mock_data_manager):
        """测试检查股票结构"""
        checker = StockHealthChecker()
        checker.dm = mock_data_manager
        
        # 模拟返回数据
        mock_data_manager.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['平安银行'],
        })
        mock_data_manager.get_daily_data.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'close': [10.0],
        })
        
        result = checker.check_stock('000001.SZ', days=30)
        
        assert isinstance(result, dict)
        assert 'stock_code' in result
        assert 'check_time' in result
        assert 'basic_info' in result
        assert 'technical_analysis' in result
        assert 'fundamental_analysis' in result
        assert 'model_prediction' in result
        assert 'risk_assessment' in result
        assert 'overall_score' in result
        assert 'recommendation' in result

