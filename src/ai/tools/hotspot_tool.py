"""
热点突破池查询工具 — AgentScope Tool 封装 (v1.0.19)

工具函数返回 ToolResponse，content 为 TextBlock 列表。
"""

import json

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from src.api.routers.hotspot import get_hotspot_breakout


def _json_to_text(data: dict) -> str:
    """将数据转为简洁文本，控制 token 长度。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


async def query_hotspot_breakout(
    min_score: float = 60,
    top_n: int = 20,
    mode: str = "breakout",
) -> ToolResponse:
    """
    查询热点突破池数据，获取基于热点题材+技术突破+资金流向+涨停质量的短线选股列表。

    Args:
        min_score: 最低综合评分 (0-100)，默认60
        top_n: 返回数量上限，默认20
        mode: 模式，可选 breakout(热点突破) / leaderboard(龙头梯队)
    """
    try:
        result = await get_hotspot_breakout(
            date=None,
            min_score=min_score,
            require_zt=False,
            top_n=top_n,
            mode=mode,
        )
        stocks = []
        for item in result.get("data", [])[:top_n]:
            stocks.append(
                {
                    "ts_code": item.get("ts_code"),
                    "name": item.get("name"),
                    "score": item.get("score"),
                    "concept": item.get("concept"),
                    "consecutive_boards": item.get("consecutive_boards"),
                    "breakout_signals": item.get("breakout_signals", []),
                    "recommendation": item.get("recommendation"),
                }
            )

        payload = {
            "date": result.get("date"),
            "mode": mode,
            "count": len(stocks),
            "stocks": stocks,
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


async def query_leaderboard(
    top_n: int = 50,
) -> ToolResponse:
    """
    查询龙头梯队数据，按连板数分层（最高标/中位龙/低位先锋/首板池）。

    Args:
        top_n: 每梯队返回数量上限
    """
    try:
        result = await get_hotspot_breakout(
            date=None,
            min_score=0,
            require_zt=False,
            top_n=top_n,
            mode="leaderboard",
        )
        groups = []
        for g in result.get("groups", []):
            stocks = []
            for s in g.get("stocks", [])[:10]:
                stocks.append(
                    {
                        "ts_code": s.get("ts_code"),
                        "name": s.get("name"),
                        "score": s.get("score"),
                        "concept": s.get("concept"),
                    }
                )
            groups.append(
                {
                    "tier": g.get("tier"),
                    "count": g.get("count"),
                    "concepts": g.get("concepts", [])[:5],
                    "top_stocks": stocks,
                }
            )

        payload = {
            "date": result.get("date"),
            "groups": groups,
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])
