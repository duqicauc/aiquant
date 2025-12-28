"""
API限流控制器

根据Tushare Pro的限流规则：
- 120积分：每分钟10次
- 2000积分：每分钟20次
- 5000积分：每分钟60次
- 10000积分+：每分钟200次

参考：https://tushare.pro/document/1?doc_id=108
"""
import time
import functools
from collections import deque
from threading import Lock
from src.utils.logger import log


class RateLimiter:
    """API限流器"""
    
    def __init__(self, calls_per_minute: int = 10):
        """
        初始化限流器
        
        Args:
            calls_per_minute: 每分钟允许的调用次数
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # 最小间隔（秒）
        self.call_times = deque()  # 记录调用时间
        self.lock = Lock()
        
        log.info(f"限流器已初始化: {calls_per_minute}次/分钟 (最小间隔{self.min_interval:.2f}秒)")
    
    def wait_if_needed(self):
        """如果需要，等待直到可以进行下一次调用"""
        with self.lock:
            now = time.time()
            
            # 清理60秒之前的记录
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()
            
            # 如果已达到限制，等待
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0])
                if sleep_time > 0:
                    # 静默模式：不输出限流等待日志（会产生大量日志）
                    # log.debug(f"限流等待: {sleep_time:.2f}秒")
                    time.sleep(sleep_time)
                    now = time.time()
                    # 清理过期记录
                    while self.call_times and now - self.call_times[0] > 60:
                        self.call_times.popleft()
            
            # 确保最小间隔
            if self.call_times:
                time_since_last = now - self.call_times[-1]
                if time_since_last < self.min_interval:
                    sleep_time = self.min_interval - time_since_last
                    time.sleep(sleep_time)
                    now = time.time()
            
            # 记录本次调用
            self.call_times.append(now)
    
    def __call__(self, func):
        """装饰器：在函数调用前进行限流"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


class TushareRateLimiter:
    """
    Tushare专用限流器

    根据积分自动调整限流策略
    支持不同接口的不同频次限制
    """

    # 不同积分对应的全局限流配置
    GLOBAL_RATE_LIMITS = {
        0: 5,        # 未注册：每分钟5次
        120: 50,     # 基础：每分钟50次
        2000: 200,   # 进阶：每分钟200次
        5000: 500,   # 专业：每分钟500次
        10000: 1000, # 旗舰：每分钟1000次
    }

    # 特定接口的频次限制（优先级高于全局限制）
    API_SPECIFIC_LIMITS = {
        'stk_factor': 100,    # 技术指标接口：每分钟100次
        'daily_basic': 200,   # 每日指标接口：每分钟200次
        'daily': 500,         # 日线数据接口：每分钟500次
        'weekly': 200,        # 周线数据接口：每分钟200次
        'monthly': 200,       # 月线数据接口：每分钟200次
    }
    
    def __init__(self, points: int = 120):
        """
        初始化Tushare限流器

        Args:
            points: Tushare积分
        """
        self.points = points
        self.global_rate_limit = self._get_global_rate_limit(points)

        # 为不同接口创建限流器
        self.limiters = {}
        for api_name, limit in self.API_SPECIFIC_LIMITS.items():
            # 接口特定限制不能超过全局积分允许的最大值
            actual_limit = min(limit, self.global_rate_limit)
            self.limiters[api_name] = RateLimiter(actual_limit)

        # 默认限流器（用于未指定接口的调用）
        self.default_limiter = RateLimiter(self.global_rate_limit)

        log.info(f"Tushare限流器: {points}积分 → 全局{self.global_rate_limit}次/分钟")

    def _get_global_rate_limit(self, points: int) -> int:
        """根据积分获取全局限流次数"""
        for threshold, limit in sorted(self.GLOBAL_RATE_LIMITS.items(), reverse=True):
            if points >= threshold:
                return limit
        return 5  # 默认最低限制

    def get_limiter(self, api_name: str = None) -> RateLimiter:
        """
        获取指定接口的限流器

        Args:
            api_name: 接口名称，如果为None则返回默认限流器

        Returns:
            对应的限流器实例
        """
        if api_name and api_name in self.limiters:
            return self.limiters[api_name]
        return self.default_limiter
    
    def __call__(self, func):
        """装饰器"""
        return self.limiter(func)


# 全局限流器实例
_global_limiter = None


def init_rate_limiter(points: int = 120):
    """
    初始化全局限流器

    Args:
        points: Tushare积分
    """
    global _global_limiter
    # 强制重新初始化，确保使用正确的积分设置
    _global_limiter = TushareRateLimiter(points)
    return _global_limiter


def get_rate_limiter() -> TushareRateLimiter:
    """获取全局限流器"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = TushareRateLimiter(120)  # 默认120积分
    return _global_limiter


def get_api_limiter(api_name: str = None) -> RateLimiter:
    """
    获取指定API接口的限流器

    Args:
        api_name: 接口名称

    Returns:
        对应的限流器实例
    """
    limiter = get_rate_limiter()
    return limiter.get_limiter(api_name)


def rate_limited(api_name: str = None):
    """
    限流装饰器

    Args:
        api_name: 接口名称，用于选择对应的限流器

    使用示例：
    @rate_limited()  # 使用默认限流器
    def fetch_data():
        ...

    @rate_limited('stk_factor')  # 使用stk_factor专用限流器
    def fetch_stk_factor():
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_api_limiter(api_name)
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3, base_delay: float = 1.0):
    """
    重试装饰器（指数退避）
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
    
    使用示例：
    @retry_on_error(max_retries=5, base_delay=1.0)
    def fetch_data():
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # 指数退避
                        error_msg = str(e)[:200] if str(e) else type(e).__name__
                        log.warning(
                            f"{func.__name__} 调用失败 (第{attempt+1}次)，"
                            f"{delay:.1f}秒后重试: {error_msg}"
                        )
                        time.sleep(delay)
                    else:
                        error_msg = str(e)[:200] if str(e) else type(e).__name__
                        log.error(
                            f"{func.__name__} 调用失败，已达最大重试次数({max_retries}): {error_msg}"
                        )
            
            # 所有重试都失败，抛出最后一个异常
            raise last_exception
        
        return wrapper
    return decorator


# 组合装饰器：限流 + 重试
def safe_api_call(api_name: str = None, max_retries: int = 3, base_delay: float = 1.0):
    """
    安全的API调用装饰器（限流 + 重试）

    Args:
        api_name: 接口名称，用于选择对应的限流器
        max_retries: 最大重试次数
        base_delay: 基础延迟时间

    使用示例：
    @safe_api_call(max_retries=5)
    def fetch_data():
        ...

    @safe_api_call('stk_factor', max_retries=5)
    def fetch_stk_factor():
        ...
    """
    def decorator(func):
        # 先应用重试，再应用限流
        func = retry_on_error(max_retries, base_delay)(func)
        func = rate_limited(api_name)(func)
        return func
    return decorator

