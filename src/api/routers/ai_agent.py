"""
AI Agent API — 智能投资助手接口

提供自然语言交互入口，根据用户意图路由到不同 Agent：
- 选股 → stock_selector
- 诊断 → stock_diagnoser
- 日报 → market_reporter
- 代码/策略 → code_assistant
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ai.agents import get_agent, init_agents, list_agents
from src.utils.logger import log

router = APIRouter()


def _agent_available() -> bool:
    """检查 Agent 是否可用"""
    try:
        return init_agents()
    except Exception:
        return False


@router.get("/agents")
async def list_available_agents():
    """列出所有可用的 AI Agent"""
    if not _agent_available():
        return {"agents": [], "available": False, "reason": "LLM API Key 未配置或 AgentScope 初始化失败"}
    return {"agents": list_agents(), "available": True}


@router.post("/chat")
async def ai_chat(
    message: str = Query(..., description="用户输入的消息"),
    agent_type: Optional[str] = Query(None, description="指定 Agent 类型，不指定则自动路由"),
    conversation_id: Optional[str] = Query(None, description="会话ID，用于上下文保持"),
):
    """
    AI 智能助手对话接口。

    根据用户输入内容自动判断意图，路由到对应的 Agent：
    - 选股/筛选/推荐 → stock_selector
    - 诊断/分析某只股票 → stock_diagnoser
    - 日报/复盘/市场总结 → market_reporter
    - 代码/策略/回测 → code_assistant
    """
    if not _agent_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "AI 助手当前不可用。请检查 LLM API Key 是否配置"
                "（DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY）"
            ),
        )

    # 意图路由
    if agent_type is None:
        agent_type = _route_intent(message)

    agent = get_agent(agent_type)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"未知的 Agent 类型: {agent_type}")

    try:
        from agentscope.message import Msg

        msg = Msg(name="user", role="user", content=message)
        response = await agent(msg)

        # response.content 是 ContentBlock 列表，提取文本
        text_parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        return {
            "agent": agent_type,
            "response": "\n".join(text_parts) if text_parts else str(response),
            "conversation_id": conversation_id,
        }
    except Exception as e:
        log.error(f"AI Agent 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"AI 处理失败: {str(e)}")


@router.post("/selector")
async def ai_selector(
    query: str = Query(..., description="选股需求描述，如'帮我找近期放量突破的科技股的涨停股'"),
):
    """选股助手快捷接口"""
    return await ai_chat(message=query, agent_type="stock_selector")


@router.post("/diagnose")
async def ai_diagnose(
    ts_code: str = Query(..., description="股票代码，如 000001.SZ"),
    question: Optional[str] = Query(None, description="额外问题，如'这只股的资金流向如何'"),
):
    """个股诊断快捷接口"""
    msg = f"请诊断股票 {ts_code}"
    if question:
        msg += f"，重点关注: {question}"
    return await ai_chat(message=msg, agent_type="stock_diagnoser")


@router.post("/report")
async def ai_daily_report():
    """生成市场日报"""
    return await ai_chat(
        message="请生成今日市场复盘报告，包括市场概况、热点题材、涨停分析和明日关注",
        agent_type="market_reporter",
    )


@router.post("/code")
async def ai_code_assistant(
    question: str = Query(..., description="编程或策略相关的问题"),
):
    """代码助手快捷接口"""
    return await ai_chat(message=question, agent_type="code_assistant")


def _route_intent(message: str) -> str:
    """
    简单的意图路由，根据关键词匹配 Agent。
    后续可升级为 LLM-based 路由。
    """
    msg = message.lower()

    # 代码/策略相关（最高优先级）
    code_keywords = ["代码", "python", "策略", "回测", "脚本", "函数", "bug", "报错", "优化"]
    if any(k in msg for k in code_keywords):
        return "code_assistant"

    # 日报/复盘相关（次高优先级）
    report_keywords = ["日报", "复盘", "总结", "市场概况", "收盘", "盘面", "今日市场", "今日行情", "今日收盘"]
    if any(k in msg for k in report_keywords):
        return "market_reporter"

    # 诊断相关（包含股票代码格式或明确诊断意图）
    diagnose_keywords = ["诊断", "分析", "好不好", "能买吗", "怎么看", "怎么样"]
    # "怎么样" 单独出现时容易歧义，需要结合其他股票相关词
    stock_context = ["股", "股票", "票", "这只", "个票", "标"]
    if any(k in msg for k in diagnose_keywords):
        if "怎么样" in msg and not any(s in msg for s in stock_context):
            pass  # "怎么样" 无股票上下文，继续往下判断
        else:
            return "stock_diagnoser"

    # 默认选股
    return "stock_selector"
