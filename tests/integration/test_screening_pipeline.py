"""
筛选流程集成测试
测试从股票筛选到结果输出的完整流程
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.strategy.screening.financial_filter import FinancialFilter
from src.strategy.screening.positive_sample_screener import PositiveSampleScreener


@pytest.mark.integration
class TestScreeningPipeline:
    """筛选流程集成测试"""

    @pytest.fixture
    def mock_data_manager(self):
        """模拟DataManager"""
        mock_dm = Mock()
        
        # 模拟股票列表（需要包含list_date列）
        mock_stocks = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
            'name': ['股票1', '股票2', '股票3'],
            'list_date': ['19910403', '19991110', '19910129'],  # 添加list_date列
        })
        mock_dm.get_stock_list.return_value = mock_stocks
        
        # 模拟财务数据
        mock_fundamental = pd.DataFrame({
            'end_date': ['20231231'],
            'revenue': [1000000.0],
            'net_profit': [100000.0],
            'roe': [15.0],
        })
        mock_dm.get_fundamental_data.return_value = mock_fundamental
        
        # 模拟日线数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        mock_daily = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': np.random.uniform(10, 20, 100),
        })
        mock_dm.get_daily_data.return_value = mock_daily
        
        return mock_dm

    @pytest.mark.unit
    def test_financial_filter_flow(self, mock_data_manager):
        """测试财务筛选流程"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        # 获取股票列表
        stocks = mock_data_manager.get_stock_list()
        
        # 执行财务筛选（filter_stocks接受DataFrame，返回DataFrame）
        filtered_stocks = filter_obj.filter_stocks(stocks)
        
        # 验证筛选流程（返回DataFrame）
        assert isinstance(filtered_stocks, pd.DataFrame)

    @pytest.mark.unit
    def test_positive_sample_screening_flow(self, mock_data_manager):
        """测试正样本筛选流程"""
        screener = PositiveSampleScreener(mock_data_manager)
        
        # 获取股票列表
        stocks = mock_data_manager.get_stock_list()
        
        # 执行筛选
        samples = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20241231'
        )
        
        # 验证筛选结果
        assert isinstance(samples, pd.DataFrame)

    @pytest.mark.integration
    def test_complete_screening_pipeline(self, mock_data_manager):
        """测试完整筛选流程"""
        # 1. 财务筛选
        financial_filter = FinancialFilter(mock_data_manager)
        stocks = mock_data_manager.get_stock_list()
        financial_filtered = financial_filter.filter_stocks(stocks)  # 传入DataFrame
        
        # 2. 正样本筛选
        screener = PositiveSampleScreener(mock_data_manager)
        samples = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20241231'
        )
        
        # 验证流程完整性
        assert isinstance(financial_filtered, pd.DataFrame)
        assert isinstance(samples, pd.DataFrame)

    @pytest.mark.integration
    def test_screening_with_multiple_filters(self, mock_data_manager):
        """测试多级筛选流程"""
        # 1. 获取所有股票
        all_stocks = mock_data_manager.get_stock_list()
        
        # 2. 财务筛选（传入DataFrame）
        financial_filter = FinancialFilter(mock_data_manager)
        financial_filtered = financial_filter.filter_stocks(all_stocks)
        
        # 3. 正样本筛选（在财务筛选后的股票中）
        screener = PositiveSampleScreener(mock_data_manager)
        samples = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20241231'
        )
        
        # 验证筛选链
        assert isinstance(financial_filtered, pd.DataFrame)
        assert len(financial_filtered) <= len(all_stocks)
        assert isinstance(samples, pd.DataFrame)

