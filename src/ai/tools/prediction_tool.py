"""
预测查询工具 — AgentScope Tool 封装 (v1.0.19)
"""

import json

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from src.api.routers.prediction import get_latest_predictions


def _json_to_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


async def query_stock_prediction(ts_code: str) -> ToolResponse:
    """
    查询单只股票的中期模型预测概率。

    Args:
        ts_code: 股票代码，如 000001.SZ
    """
    try:
        result = await get_latest_predictions(top_n=5000)
        stocks = result.get("data", [])
        target = None
        for s in stocks:
            if s.get("ts_code") == ts_code:
                target = s
                break

        if not target:
            return ToolResponse(content=[TextBlock(text=f"未找到 {ts_code} 的预测数据")])

        payload = {
            "ts_code": target.get("ts_code"),
            "name": target.get("name"),
            "prob": target.get("prob"),
            "industry": target.get("industry"),
            "market_cap": target.get("market_cap"),
            "pe_ttm": target.get("pe_ttm"),
            "update_date": target.get("update_date"),
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


async def query_top_predictions(top_n: int = 20, min_prob: float = 0.5) -> ToolResponse:
    """
    查询中期模型概率最高的 Top N 股票。

    Args:
        top_n: 返回数量
        min_prob: 最低概率阈值
    """
    try:
        result = await get_latest_predictions(top_n=top_n, min_prob=min_prob)
        stocks = []
        for s in result.get("data", [])[:top_n]:
            stocks.append(
                {
                    "ts_code": s.get("ts_code"),
                    "name": s.get("name"),
                    "prob": s.get("prob"),
                    "industry": s.get("industry"),
                }
            )
        return ToolResponse(content=[TextBlock(text=_json_to_text({"count": len(stocks), "stocks": stocks}))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])
