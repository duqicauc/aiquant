"""
筛选器真实数据测试
"""
import pytest
import os
import pandas as pd
from datetime import datetime, timedelta


@pytest.mark.api
@pytest.mark.slow
class TestScreeningReal:
    """筛选器真实数据测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """检查是否有Tushare Token"""
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            pytest.skip("需要设置TUSHARE_TOKEN环境变量")
    
    def test_positive_screener_real(self):
        """测试正样本筛选器（真实数据）"""
        from src.data.data_manager import DataManager
        from src.strategy.screening.positive_sample_screener import PositiveSampleScreener
        
        dm = DataManager(source='tushare')
        screener = PositiveSampleScreener(dm)
        
        # 只测试最近30天的数据，避免耗时过长
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = screener.screen_all_stocks(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有找到正样本，也应该返回DataFrame
    
    def test_financial_filter_real(self):
        """测试财务筛选器（真实数据）"""
        from src.data.data_manager import DataManager
        from src.strategy.screening.financial_filter import FinancialFilter
        
        dm = DataManager(source='tushare')
        filter_obj = FinancialFilter(dm)
        
        # 创建测试股票列表
        test_stocks = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            '股票代码': ['000001.SZ', '600000.SH'],
            'name': ['平安银行', '浦发银行'],
            '股票名称': ['平安银行', '浦发银行'],
        })
        
        result = filter_obj.filter_stocks(
            test_stocks,
            revenue_threshold=3.0,
            profit_years=3
        )
        
        assert isinstance(result, pd.DataFrame)
        # 结果应该包含筛选后的股票（可能为空）

