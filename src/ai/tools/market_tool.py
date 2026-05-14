"""
市场分析工具 — AgentScope Tool 封装 (v1.0.19)
"""

import json

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from src.api.routers.market import get_hot_concepts, get_market_overview, get_sector_fund_flow
from src.data.market_heat_provider import market_heat_provider


def _json_to_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


async def query_market_overview() -> ToolResponse:
    """查询市场整体概览：指数涨跌、涨跌家数、涨停跌停数、资金流向"""
    try:
        result = await get_market_overview()
        data = result.get("data", {})
        payload = {
            "indices": data.get("indices", []),
            "up_count": data.get("up_count"),
            "down_count": data.get("down_count"),
            "up_limit": data.get("up_limit"),
            "down_limit": data.get("down_limit"),
            "north_money": data.get("north_money"),
            "summary": data.get("summary"),
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


async def query_hot_concepts(top_n: int = 10) -> ToolResponse:
    """查询当前热点题材板块，按涨停家数排序"""
    try:
        result = await get_hot_concepts(date=None, top_n=top_n)
        concepts = []
        for c in result.get("data", [])[:top_n]:
            concepts.append(
                {
                    "rank": c.get("rank"),
                    "name": c.get("name"),
                    "up_nums": c.get("up_nums"),
                    "cons_nums": c.get("cons_nums"),
                    "pct_chg": c.get("pct_chg"),
                }
            )
        payload = {"date": result.get("date"), "concepts": concepts}
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


async def query_sector_fund_flow() -> ToolResponse:
    """查询行业资金流向，主力净流入排名"""
    try:
        result = await get_sector_fund_flow()
        sectors = []
        for s in result.get("data", [])[:15]:
            sectors.append(
                {
                    "rank": s.get("rank"),
                    "name": s.get("name"),
                    "pct_chg": s.get("pct_chg"),
                    "main_force_net": s.get("main_force_net"),
                    "top_stock": s.get("top_stock"),
                }
            )
        return ToolResponse(content=[TextBlock(text=_json_to_text({"sectors": sectors}))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


def query_zt_pool() -> ToolResponse:
    """查询当日涨停股池"""
    try:
        data = market_heat_provider.get_zt_pool()
        stocks = []
        for item in data[:30]:
            stocks.append(
                {
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "industry": item.get("industry"),
                    "pct_chg": item.get("pct_chg"),
                    "consecutive_boards": item.get("consecutive_boards"),
                    "board_money": item.get("board_money"),
                    "open_count": item.get("open_count"),
                }
            )
        return ToolResponse(content=[TextBlock(text=_json_to_text({"count": len(stocks), "stocks": stocks}))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])
