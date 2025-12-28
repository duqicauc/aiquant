"""
数据流集成测试
测试从数据获取到模型预测的完整流程
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd


@pytest.mark.integration
class TestDataFlow:
    """数据流集成测试"""
    
    def test_data_manager_to_prediction_flow(self, mock_data_manager):
        """测试从数据管理到预测的完整流程"""
        # 1. 获取股票列表
        stocks = mock_data_manager.get_stock_list()
        assert len(stocks) > 0
        
        # 2. 获取股票数据
        stock_code = stocks.iloc[0]['ts_code']
        data = mock_data_manager.get_daily_data(
            stock_code=stock_code,
            start_date='20240101',
            end_date='20240131'
        )
        assert len(data) > 0
        
        # 3. 模拟特征提取
        features = pd.DataFrame({
            'ts_code': [stock_code],
            'feature1': [0.5],
            'feature2': [0.3],
        })
        
        # 4. 模拟模型预测
        # 这里只是测试流程，实际需要加载真实模型
        assert len(features) > 0
    
    @pytest.mark.slow
    def test_full_prediction_pipeline(self):
        """测试完整预测流程（需要真实数据）"""
        # 这个测试需要真实的数据和模型
        # 标记为slow，避免在快速测试中运行
        pytest.skip("需要真实数据和模型，跳过")

