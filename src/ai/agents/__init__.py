"""
AI Agent 注册中心 — AgentScope 1.0.19 多智能体初始化

参考 Claude Code / 龙虾等工具理念：
- Agent 不只是回答问题，而是能自主执行代码、读取数据、完成任务
- 人机协作：AI 提出方案，用户确认后执行
- 多 Agent 协作：选股Agent → 诊断Agent → 风控Agent
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from src.utils.logger import log

# ─── 全局 Agent 缓存 ───
_agents_initialized = False
_agent_registry: Dict[str, Any] = {}


def _get_model() -> Optional[OpenAIChatModel]:
    """根据环境变量创建模型实例，优先 DeepSeek，次选 OpenAI。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        return OpenAIChatModel(
            model_name="deepseek-chat",
            api_key=api_key,
            client_kwargs={"base_url": "https://api.deepseek.com/v1"},
            stream=False,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAIChatModel(
            model_name="gpt-4o-mini",
            api_key=api_key,
            stream=False,
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        # Anthropic 需要用专门的模型类，OpenAIChatModel 不支持
        try:
            from agentscope.model import AnthropicChatModel

            return AnthropicChatModel(
                model_name="claude-3-5-sonnet-20241022",
                api_key=api_key,
                stream=False,
            )
        except Exception as e:
            log.warning(f"Anthropic 模型初始化失败: {e}")

    return None


def _get_formatter():
    """根据当前 provider 返回对应的 formatter。"""
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekChatFormatter()
    return OpenAIChatFormatter()


def _wrap_async_tool(func):
    """将异步工具包装为同步工具（AgentScope 工具需同步）。"""

    def wrapper(**kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中（如 FastAPI），使用 nest_asyncio 或新线程
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, func(**kwargs))
                    return future.result()
            return loop.run_until_complete(func(**kwargs))
        except Exception as e:
            from agentscope.message import TextBlock
            from agentscope.tool import ToolResponse

            return ToolResponse(content=[TextBlock(type="text", text=f"工具执行错误: {e}")])

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def _create_toolkit_for_selector() -> Toolkit:
    """创建选股 Agent 的工具包。"""
    from src.ai.tools import hotspot_tool, market_tool, prediction_tool

    tk = Toolkit()
    tk.register_tool_function(_wrap_async_tool(hotspot_tool.query_hotspot_breakout))
    tk.register_tool_function(_wrap_async_tool(hotspot_tool.query_leaderboard))
    tk.register_tool_function(_wrap_async_tool(prediction_tool.query_top_predictions))
    tk.register_tool_function(_wrap_async_tool(market_tool.query_hot_concepts))
    tk.register_tool_function(_wrap_async_tool(market_tool.query_zt_pool))
    return tk


def _create_toolkit_for_diagnoser() -> Toolkit:
    """创建诊断 Agent 的工具包。"""
    from src.ai.tools import hotspot_tool, prediction_tool, stock_tool

    tk = Toolkit()
    tk.register_tool_function(stock_tool.query_stock_technical)
    tk.register_tool_function(stock_tool.query_stock_moneyflow)
    tk.register_tool_function(_wrap_async_tool(prediction_tool.query_stock_prediction))
    tk.register_tool_function(_wrap_async_tool(hotspot_tool.query_hotspot_breakout))
    return tk


def _create_toolkit_for_code() -> Toolkit:
    """创建代码助手 Agent 的工具包。"""
    from agentscope.tool import (
        execute_python_code,
        execute_shell_command,
        view_text_file,
    )

    tk = Toolkit()
    tk.register_tool_function(execute_python_code)
    tk.register_tool_function(execute_shell_command)
    tk.register_tool_function(view_text_file)
    return tk


def init_agents() -> bool:
    """
    初始化 AgentScope 和所有 Agent。
    在 FastAPI 应用启动时调用一次。
    """
    global _agents_initialized, _agent_registry
    if _agents_initialized:
        return True

    model = _get_model()
    if model is None:
        log.warning("LLM API Key 未配置（DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY），AI Agent 不可用")
        return False

    formatter = _get_formatter()

    try:
        # ─── 选股Agent ───
        _agent_registry["stock_selector"] = ReActAgent(
            name="选股助手",
            sys_prompt="""你是一位专业的A股短线选股分析师。
你的任务是根据用户的需求，调用热点突破池、龙头梯队、预测接口等工具，筛选出最符合条件的股票。

规则：
1. 用户提出需求后，先分析需要调用哪些工具
2. 调用工具获取数据
3. 对结果进行分析，给出选股理由
4. 用中文回复，结构清晰

注意：你提供的是数据分析结果，不是投资建议。用户需自主决策。""",
            model=model,
            formatter=formatter,
            toolkit=_create_toolkit_for_selector(),
            max_iters=5,
        )

        # ─── 诊断Agent ───
        _agent_registry["stock_diagnoser"] = ReActAgent(
            name="个股诊断师",
            sys_prompt="""你是一位专业的A股个股诊断分析师。
你的任务是对单只股票进行深度分析，综合技术指标、资金流向、模型预测、题材热度等维度给出诊断报告。

分析维度：
1. 技术形态（均线、VWAP、CMF、MFI等）
2. 资金流向（主力净流入、特大单买卖）
3. 中期模型概率
4. 所属题材热度
5. 涨停质量（如果涨停）

输出格式：
- 总体判断：强势/中性/弱势
- 关键信号：列举3-5个最重要的数据点
- 风险提示：列举潜在风险

注意：你提供的是数据分析结果，不是投资建议。""",
            model=model,
            formatter=formatter,
            toolkit=_create_toolkit_for_diagnoser(),
            max_iters=5,
        )

        # ─── 日报Agent（DialogAgent 已移除，用 ReActAgent 无工具版）───
        _agent_registry["market_reporter"] = ReActAgent(
            name="市场日报员",
            sys_prompt="""你是一位专业的A股市场复盘分析师。
你的任务是根据市场数据，生成每日复盘报告。

报告结构：
1. 市场概况：指数涨跌、涨跌家数、北向资金
2. 热点题材：当日最强板块、持续性判断
3. 涨停分析：封板率、炸板率、最高标
4. 明日关注：基于热点突破池评分最高的标的
5. 风险提示：市场情绪是否过热/过冷

注意：你提供的是市场数据分析，不是投资建议。""",
            model=model,
            formatter=formatter,
            toolkit=None,
            max_iters=3,
        )

        # ─── 代码助手Agent ───
        _agent_registry["code_assistant"] = ReActAgent(
            name="代码助手",
            sys_prompt="""你是一位专业的量化开发助手，擅长 Python 数据分析、Pandas、量化策略编写。
你可以帮用户：
1. 编写/调试 Python 数据分析代码
2. 解释量化模型的逻辑
3. 优化策略回测参数
4. 读取项目中的代码文件并给出改进建议

你有以下工具可用：
- execute_python_code: 执行 Python 代码并返回结果
- execute_shell_command: 执行 shell 命令
- view_text_file: 读取文件内容

安全规则：
- 执行代码前，先向用户确认代码内容
- 不要执行 rm -rf 等危险命令
- 修改文件前，先展示 diff 让用户确认
""",
            model=model,
            formatter=formatter,
            toolkit=_create_toolkit_for_code(),
            max_iters=10,
        )

        _agents_initialized = True
        log.info(f"AgentScope Agent 初始化完成，共 {len(_agent_registry)} 个 Agent")
        return True

    except Exception as e:
        log.error(f"Agent 初始化失败: {e}")
        return False


def get_agent(name: str):
    """获取指定名称的 Agent"""
    if not _agents_initialized:
        init_agents()
    return _agent_registry.get(name)


def list_agents() -> List[Dict[str, str]]:
    """列出所有可用的 Agent 信息"""
    if not _agents_initialized:
        init_agents()
    return [{"name": name, "type": "ReActAgent"} for name in _agent_registry.keys()]
