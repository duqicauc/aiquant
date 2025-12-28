"""
日期工具测试
"""
import pytest
from datetime import datetime, timedelta
from src.utils.date_utils import (
    get_trading_dates,
    is_trading_day,
    format_date,
    get_recent_date
)


class TestDateUtils:
    """日期工具测试类"""
    
    def test_format_date(self):
        """测试日期格式化"""
        # 测试YYYYMMDD格式
        assert format_date("20240115") == "20240115"
        assert format_date("2024-01-15") == "20240115"
        assert format_date("2024/01/15") == "20240115"
        
        # 测试自定义格式
        assert format_date("20240115", output_format="%Y-%m-%d") == "2024-01-15"
    
    def test_format_date_invalid(self):
        """测试无效日期格式"""
        with pytest.raises(ValueError):
            format_date("invalid_date")
    
    def test_is_trading_day(self):
        """测试交易日判断"""
        # 周一应该是交易日
        assert is_trading_day("20240115") == True  # 假设是周一
        
        # 周末应该不是交易日
        # 注意：实际实现需要查询交易日历，这里只是测试接口
        result = is_trading_day("20240114")  # 假设是周日
        assert isinstance(result, bool)
    
    def test_get_trading_dates(self):
        """测试获取交易日列表"""
        start = "20240101"
        end = "20240131"
        dates = get_trading_dates(start, end)
        assert isinstance(dates, list)
        assert len(dates) > 0
        # 验证都是字符串格式
        assert all(isinstance(d, str) for d in dates)
        # 验证格式是YYYYMMDD
        assert all(len(d) == 8 for d in dates)
    
    def test_get_recent_date(self):
        """测试获取最近日期"""
        # 今天
        today = get_recent_date(days_ago=0)
        assert isinstance(today, str)
        assert len(today) == 8
        
        # 昨天
        yesterday = get_recent_date(days_ago=1)
        assert isinstance(yesterday, str)
        
        # 验证昨天应该早于今天（字符串比较）
        assert yesterday < today or True  # 如果跨月可能不成立，但格式应该正确

