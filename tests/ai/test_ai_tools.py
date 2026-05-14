"""
AI 工具层单元测试 — 测试 ToolResponse 包装和工具函数
"""

import json

from agentscope.tool import ToolResponse

# 测试工具无需 LLM，直接导入
from src.ai.tools.hotspot_tool import query_hotspot_breakout, query_leaderboard
from src.ai.tools.market_tool import query_zt_pool
from src.ai.tools.prediction_tool import query_stock_prediction, query_top_predictions
from src.ai.tools.stock_tool import query_stock_kline, query_stock_moneyflow, query_stock_technical


class TestToolResponseFormat:
    """测试所有工具函数返回格式正确"""

    def test_query_hotspot_breakout_returns_tool_response(self):
        """query_hotspot_breakout 返回 ToolResponse"""
        # 由于需要 await，这里用同步测试（内部包装了 asyncio.run）
        import asyncio

        resp = asyncio.run(query_hotspot_breakout(min_score=90, top_n=5))
        assert isinstance(resp, ToolResponse)
        assert len(resp.content) > 0
        # 验证 content 是 dict 列表（TypedDict）
        assert isinstance(resp.content[0], dict)
        text = resp.content[0]["text"]
        assert "stocks" in text or "查询失败" in text or "date" in text

    def test_query_leaderboard_returns_tool_response(self):
        """query_leaderboard 返回 ToolResponse"""
        import asyncio

        resp = asyncio.run(query_leaderboard(top_n=10))
        assert isinstance(resp, ToolResponse)
        text = resp.content[0]["text"]
        assert "groups" in text or "查询失败" in text or "date" in text

    def test_query_stock_prediction_returns_tool_response(self):
        """query_stock_prediction 返回 ToolResponse"""
        import asyncio

        resp = asyncio.run(query_stock_prediction("000001.SZ"))
        assert isinstance(resp, ToolResponse)
        # 可能成功或失败，但格式要对
        text = resp.content[0]["text"]
        assert isinstance(text, str)

    def test_query_top_predictions_returns_tool_response(self):
        """query_top_predictions 返回 ToolResponse"""
        import asyncio

        resp = asyncio.run(query_top_predictions(top_n=5, min_prob=0.6))
        assert isinstance(resp, ToolResponse)

    def test_query_zt_pool_returns_tool_response(self):
        """query_zt_pool 返回 ToolResponse"""
        resp = query_zt_pool()
        assert isinstance(resp, ToolResponse)
        data = json.loads(resp.content[0]["text"])
        assert "stocks" in data or "查询失败" in resp.content[0].text

    def test_query_stock_kline_returns_tool_response(self):
        """query_stock_kline 返回 ToolResponse"""
        resp = query_stock_kline("000001.SZ", days=5)
        assert isinstance(resp, ToolResponse)

    def test_query_stock_technical_returns_tool_response(self):
        """query_stock_technical 返回 ToolResponse"""
        resp = query_stock_technical("000001.SZ", days=5)
        assert isinstance(resp, ToolResponse)

    def test_query_stock_moneyflow_returns_tool_response(self):
        """query_stock_moneyflow 返回 ToolResponse"""
        resp = query_stock_moneyflow("000001.SZ", days=3)
        assert isinstance(resp, ToolResponse)


class TestToolErrorHandling:
    """测试工具错误处理"""

    def test_invalid_ts_code_returns_error_text(self):
        """无效股票代码应返回错误信息"""
        import asyncio

        resp = asyncio.run(query_stock_prediction("INVALID.CODE"))
        text = resp.content[0]["text"]
        assert "未找到" in text or "查询失败" in text


class TestJsonHelper:
    """测试 JSON 序列化辅助函数"""

    def test_json_to_text_with_chinese(self):
        """中文正确序列化"""
        from src.ai.tools.hotspot_tool import _json_to_text

        text = _json_to_text({"概念": "人工智能", "股票": [{"名称": "测试股"}]})
        assert "人工智能" in text
        assert "测试股" in text
