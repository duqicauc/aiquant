"""
LLM Provider — 统一封装多模型后端

支持的 Provider:
- deepseek (DeepSeek-V3 / DeepSeek-Chat)
- openai (GPT-4o / GPT-4o-mini)
- anthropic (Claude 3.5 Sonnet)

配置方式（环境变量）:
    LLM_PROVIDER=deepseek
    DEEPSEEK_API_KEY=sk-xxx
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-xxx

    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-xxx
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.utils.logger import log


class LLMProvider:
    """统一的 LLM 调用接口"""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).lower()
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or os.getenv(f"{self.provider.upper()}_BASE_URL")
        self.model = model or self._default_model()
        self._client = None

    def _get_api_key(self) -> Optional[str]:
        env_var = f"{self.provider.upper()}_API_KEY"
        key = os.getenv(env_var)
        if not key:
            log.warning(f"未设置 {env_var}，LLM 功能将不可用")
        return key

    def _default_model(self) -> str:
        defaults = {
            "deepseek": "deepseek-chat",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
        }
        return defaults.get(self.provider, "deepseek-chat")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"{self.provider.upper()}_API_KEY 未设置，无法调用 LLM")

        if self.provider == "deepseek":
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url or "https://api.deepseek.com/v1",
                )
            except ImportError:
                raise RuntimeError("使用 deepseek provider 需要安装 openai: pip install openai")

        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("使用 openai provider 需要安装 openai: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("使用 anthropic provider 需要安装 anthropic: pip install anthropic")

        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

        return self._client

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        统一的 chat completion 接口。

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
            tools: OpenAI 格式的 function calling tools
            temperature: 0-2
            max_tokens: 最大输出长度
            stream: 是否流式输出

        Returns:
            {
                "content": str | None,
                "tool_calls": List[{"name": str, "arguments": dict}] | None,
                "model": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            }
        """
        client = self._get_client()

        if self.provider in ("deepseek", "openai"):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                log.error(f"LLM 调用失败: {e}")
                raise

            if stream:
                return {"stream": resp, "model": self.model}

            message = resp.choices[0].message
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({"name": tc.function.name, "arguments": args})

            return {
                "content": message.content,
                "tool_calls": tool_calls,
                "model": resp.model,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            }

        elif self.provider == "anthropic":
            # Anthropic 使用不同的 API 格式
            system_msg = ""
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_messages.append({"role": m["role"], "content": m["content"]})

            kwargs = {
                "model": self.model,
                "messages": user_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_msg:
                kwargs["system"] = system_msg

            try:
                resp = client.messages.create(**kwargs)
            except Exception as e:
                log.error(f"LLM 调用失败: {e}")
                raise

            content = None
            for block in resp.content:
                if block.type == "text":
                    content = block.text
                    break

            return {
                "content": content,
                "tool_calls": None,
                "model": resp.model,
                "usage": {
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                },
            }

        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

    def simple_chat(self, user_message: str, system_message: Optional[str] = None) -> str:
        """简化版：单轮对话，直接返回文本内容"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        resp = self.chat_completion(messages)
        return resp.get("content") or ""


# ─── 全局单例 ───
_llm_instance: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMProvider()
    return _llm_instance
