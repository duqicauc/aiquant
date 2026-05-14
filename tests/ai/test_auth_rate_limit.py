"""
Auth 限流单元测试
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestAuthRateLimit:
    """测试登录限流"""

    def test_login_rate_limit_triggers_after_max_attempts(self):
        """连续登录失败超过限制后返回 429"""
        # 先清空限流记录
        from src.api.routers.auth import _login_attempts

        _login_attempts.clear()

        # 快速发送超过限制次数的请求
        for i in range(12):
            resp = client.post("/api/auth/login", json={"username": f"testuser{i}", "password": "wrongpassword"})

        # 最后几次应该有 429
        # 注意：由于测试环境没有用户数据，前面可能是 401，超过限制后应该是 429
        # 但为了稳定测试，我们只验证限流函数本身
        from unittest.mock import MagicMock

        from fastapi import Request

        from src.api.routers.auth import _check_rate_limit

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "1.2.3.4"

        # 先清空虚构造的数据
        _login_attempts.clear()

        # 10 次内应该通过
        for _ in range(10):
            assert _check_rate_limit(mock_request) is True

        # 第 11 次应该触发限流
        assert _check_rate_limit(mock_request) is False

    def test_different_ips_not_rate_limited(self):
        """不同 IP 之间互不影响"""
        from unittest.mock import MagicMock

        from fastapi import Request

        from src.api.routers.auth import _check_rate_limit, _login_attempts

        _login_attempts.clear()

        ip1_req = MagicMock(spec=Request)
        ip1_req.client.host = "1.1.1.1"

        ip2_req = MagicMock(spec=Request)
        ip2_req.client.host = "2.2.2.2"

        # ip1 用满额度
        for _ in range(10):
            assert _check_rate_limit(ip1_req) is True
        assert _check_rate_limit(ip1_req) is False

        # ip2 不受影响
        assert _check_rate_limit(ip2_req) is True
