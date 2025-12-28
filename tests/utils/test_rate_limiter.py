"""
限流器测试
"""
import pytest
import time
from unittest.mock import patch, Mock
from src.utils.rate_limiter import (
    RateLimiter,
    TushareRateLimiter,
    init_rate_limiter,
    get_rate_limiter,
    rate_limited,
    retry_on_error,
    safe_api_call
)


class TestRateLimiter:
    """RateLimiter测试类"""
    
    def test_init(self):
        """测试初始化"""
        limiter = RateLimiter(calls_per_minute=10)
        assert limiter.calls_per_minute == 10
        assert limiter.min_interval == 6.0  # 60/10
        assert len(limiter.call_times) == 0
    
    def test_wait_if_needed_no_wait(self):
        """测试不需要等待的情况"""
        limiter = RateLimiter(calls_per_minute=60)  # 每秒1次
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # 应该几乎不等待
    
    def test_wait_if_needed_with_wait(self):
        """测试需要等待的情况"""
        limiter = RateLimiter(calls_per_minute=2)  # 每30秒1次
        limiter.wait_if_needed()
        
        # 立即再次调用应该需要等待
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time
        
        # 应该等待约30秒（但测试中我们只验证它确实等待了）
        assert elapsed > 0.1  # 至少等待一小段时间
    
    def test_decorator(self):
        """测试装饰器"""
        limiter = RateLimiter(calls_per_minute=60)
        
        @limiter
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_call_times_cleanup(self):
        """测试调用时间记录清理"""
        limiter = RateLimiter(calls_per_minute=10)
        
        # 模拟旧的调用时间（超过60秒）
        old_time = time.time() - 70
        limiter.call_times.append(old_time)
        
        # 新调用应该清理旧记录
        limiter.wait_if_needed()
        assert len(limiter.call_times) == 1
        assert limiter.call_times[0] > old_time


class TestTushareRateLimiter:
    """TushareRateLimiter测试类"""
    
    def test_init_default(self):
        """测试默认初始化"""
        limiter = TushareRateLimiter()
        assert limiter.points == 120
        assert limiter.rate_limit == 10
    
    def test_rate_limit_calculation(self):
        """测试不同积分对应的限流配置"""
        test_cases = [
            (0, 5),
            (120, 10),
            (2000, 20),
            (5000, 60),
            (10000, 200),
            (20000, 200),  # 超过最高限制
        ]
        
        for points, expected_limit in test_cases:
            limiter = TushareRateLimiter(points=points)
            assert limiter.rate_limit == expected_limit, \
                f"积分{points}应该对应{expected_limit}次/分钟"
    
    def test_decorator(self):
        """测试装饰器"""
        limiter = TushareRateLimiter(points=120)
        
        @limiter
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"


class TestGlobalRateLimiter:
    """全局限流器测试"""
    
    def test_init_rate_limiter(self):
        """测试初始化全局限流器"""
        limiter = init_rate_limiter(points=2000)
        assert limiter.points == 2000
        assert limiter.rate_limit == 20
    
    def test_get_rate_limiter(self):
        """测试获取全局限流器"""
        # 先初始化
        init_rate_limiter(points=5000)
        limiter = get_rate_limiter()
        assert limiter.points == 5000
        
        # 如果没有初始化，应该使用默认值
        # 重置全局变量
        import src.utils.rate_limiter as rate_limiter_module
        rate_limiter_module._global_limiter = None
        limiter = get_rate_limiter()
        assert limiter.points == 120  # 默认值


class TestRateLimitedDecorator:
    """rate_limited装饰器测试"""
    
    def test_rate_limited_decorator(self):
        """测试rate_limited装饰器"""
        @rate_limited
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"


class TestRetryOnError:
    """重试装饰器测试"""
    
    def test_retry_success_first_try(self):
        """测试第一次就成功"""
        @retry_on_error(max_retries=3)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_retry_success_after_retries(self):
        """测试重试后成功"""
        call_count = [0]
        
        @retry_on_error(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("失败")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        @retry_on_error(max_retries=2, base_delay=0.01)
        def test_func():
            raise ValueError("总是失败")
        
        with pytest.raises(ValueError):
            test_func()


class TestSafeApiCall:
    """safe_api_call装饰器测试"""
    
    def test_safe_api_call_success(self):
        """测试成功调用"""
        @safe_api_call(max_retries=2, base_delay=0.01)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_safe_api_call_with_retry(self):
        """测试带重试的调用"""
        call_count = [0]
        
        @safe_api_call(max_retries=2, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("失败")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count[0] == 2

