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
    
    def wait_if_needed(self, disable: bool = False):
        """
        如果需要，等待直到可以进行下一次调用
        
        Args:
            disable: 是否禁用限流（不等待，直接记录调用时间）
        """
        with self.lock:
            now = time.time()
            
            # 如果禁用限流，只记录调用时间，不等待
            if disable:
                self.call_times.append(now)
                return
            
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
        return self.default_limiter(func)


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


def rate_limited(api_name: str = None, disable: bool = False):
    """
    限流装饰器

    Args:
        api_name: 接口名称，用于选择对应的限流器
        disable: 是否禁用限流（不主动限流，遇到限流错误时自动重试）

    使用示例：
    @rate_limited()  # 使用默认限流器
    def fetch_data():
        ...

    @rate_limited('stk_factor', disable=True)  # 禁用限流，遇到限流错误时自动重试
    def fetch_stk_factor():
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not disable:
                limiter = get_api_limiter(api_name)
                limiter.wait_if_needed(disable=False)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_rate_limit_error(exception: Exception) -> bool:
    """
    判断是否是限流错误
    
    Args:
        exception: 异常对象
        
    Returns:
        是否是限流错误
    """
    error_str = str(exception).lower()
    # 常见的限流错误关键词
    rate_limit_keywords = [
        'rate limit',
        'rate_limit',
        'too many requests',
        '429',
        '请求过于频繁',
        '访问频率超限',
        '频率限制',
        '限流',
        'throttle',
        'quota exceeded',
        'quota limit'
    ]
    return any(keyword in error_str for keyword in rate_limit_keywords)


def retry_on_error(max_retries: int = 3, base_delay: float = 1.0, retry_on_rate_limit: bool = True):
    """
    重试装饰器（指数退避）
    
    Args:
        max_retries: 最大重试次数（限流错误也遵循此限制）
        base_delay: 基础延迟时间（秒）
        retry_on_rate_limit: 遇到限流错误时是否重试（最多max_retries次，不是无限重试）
    
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
                    is_rate_limit = is_rate_limit_error(e)
                    
                    if attempt < max_retries:
                        # 限流错误：适当延迟后重试（最多max_retries次）
                        if is_rate_limit and retry_on_rate_limit:
                            # 限流错误使用稍长的延迟，但不超过60秒
                            delay = min(base_delay * (2 ** min(attempt, 5)), 60.0)
                            error_msg = str(e)[:200] if str(e) else type(e).__name__
                            log.warning(
                                f"{func.__name__} 遇到限流错误 (第{attempt+1}次)，"
                                f"{delay:.1f}秒后重试: {error_msg}"
                            )
                        else:
                            # 其他错误：正常指数退避
                            delay = base_delay * (2 ** attempt)
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
def safe_api_call(api_name: str = None, max_retries: int = 3, base_delay: float = 1.0, 
                  retry_on_rate_limit: bool = True, disable_rate_limit: bool = False):
    """
    安全的API调用装饰器（限流 + 重试）

    Args:
        api_name: 接口名称，用于选择对应的限流器
        max_retries: 最大重试次数（-1表示无限重试，仅用于限流错误）
        base_delay: 基础延迟时间
        retry_on_rate_limit: 遇到限流错误时是否无限重试
        disable_rate_limit: 是否禁用主动限流（遇到限流错误时自动重试）

    使用示例：
    @safe_api_call(max_retries=5)
    def fetch_data():
        ...

    @safe_api_call('stk_factor', max_retries=5, disable_rate_limit=True)
    def fetch_stk_factor():
        ...
    """
    def decorator(func):
        # 先应用重试，再应用限流（如果不禁用）
        func = retry_on_error(max_retries, base_delay, retry_on_rate_limit)(func)
        if not disable_rate_limit:
            func = rate_limited(api_name, disable=False)(func)
        else:
            # 禁用限流，但仍然应用装饰器（只是不等待）
            func = rate_limited(api_name, disable=True)(func)
        return func
    return decorator

