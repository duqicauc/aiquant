"""
AI Agent API 路由单元测试 — 测试 FastAPI 端点（无需 LLM）

使用 monkeypatch 删除环境变量，确保测试在任何环境下都稳定。
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _reset_agent_state(monkeypatch):
    """辅助函数：清除环境变量并重置 Agent 全局状态。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from src.ai import agents as agents_mod

    agents_mod._agents_initialized = False
    agents_mod._agent_registry.clear()


class TestAIAgentEndpoints:
    """测试 /api/ai 端点（模拟无 API Key 场景）"""

    def test_list_agents_without_key(self, monkeypatch):
        """无 API Key 时返回可用 agents 列表但标记 unavailable"""
        _reset_agent_state(monkeypatch)
        resp = client.get("/api/ai/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "available" in data
        assert data["available"] is False

    def test_chat_without_key_returns_503(self, monkeypatch):
        """无 API Key 时 chat 返回 503"""
        _reset_agent_state(monkeypatch)
        resp = client.post("/api/ai/chat", params={"message": "帮我选股"})
        assert resp.status_code == 503
        assert "不可用" in resp.json()["detail"]

    def test_selector_without_key_returns_503(self, monkeypatch):
        """无 API Key 时 selector 返回 503"""
        _reset_agent_state(monkeypatch)
        resp = client.post("/api/ai/selector", params={"query": "帮我找科技股"})
        assert resp.status_code == 503

    def test_diagnose_without_key_returns_503(self, monkeypatch):
        """无 API Key 时 diagnose 返回 503"""
        _reset_agent_state(monkeypatch)
        resp = client.post("/api/ai/diagnose", params={"ts_code": "000001.SZ"})
        assert resp.status_code == 503

    def test_report_without_key_returns_503(self, monkeypatch):
        """无 API Key 时 report 返回 503"""
        _reset_agent_state(monkeypatch)
        resp = client.post("/api/ai/report")
        assert resp.status_code == 503

    def test_code_without_key_returns_503(self, monkeypatch):
        """无 API Key 时 code 返回 503"""
        _reset_agent_state(monkeypatch)
        resp = client.post("/api/ai/code", params={"question": "怎么写 pandas"})
        assert resp.status_code == 503


class TestIntentRouting:
    """测试意图路由函数"""

    def test_route_code_intent(self):
        """代码相关问题路由到 code_assistant"""
        from src.api.routers.ai_agent import _route_intent

        assert _route_intent("帮我写一个 Python 函数") == "code_assistant"
        assert _route_intent("这个脚本有 bug") == "code_assistant"
        assert _route_intent("怎么优化策略回测") == "code_assistant"

    def test_route_report_intent(self):
        """日报/复盘问题路由到 market_reporter"""
        from src.api.routers.ai_agent import _route_intent

        assert _route_intent("今日市场怎么样") == "market_reporter"
        assert _route_intent("生成收盘复盘") == "market_reporter"
        assert _route_intent("今日盘面总结") == "market_reporter"

    def test_route_diagnose_intent(self):
        """诊断问题路由到 stock_diagnoser"""
        from src.api.routers.ai_agent import _route_intent

        assert _route_intent("分析一下 000001.SZ") == "stock_diagnoser"
        assert _route_intent("这只股怎么样") == "stock_diagnoser"
        assert _route_intent("能买吗") == "stock_diagnoser"

    def test_route_default_selector(self):
        """默认路由到 stock_selector"""
        from src.api.routers.ai_agent import _route_intent

        assert _route_intent("帮我选股") == "stock_selector"
        assert _route_intent("推荐一些好股票") == "stock_selector"
        assert _route_intent("有什么热点") == "stock_selector"
