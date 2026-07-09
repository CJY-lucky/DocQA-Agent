"""LLM 客户端 —— 封装 LLM API 调用，支持 Function Calling。"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src import logger


class LLMClient:
    """LLM 调用封装，支持 OpenAI / Ollama 等兼容后端。"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """
        Args:
            model: 模型名称
            api_key: API Key（支持 ${ENV_VAR} 格式）
            base_url: 自定义 API 地址
            temperature: 生成温度
            max_tokens: 最大输出 token 数
        """
        # 解析环境变量占位符
        api_key = self._resolve_env(api_key)
        base_url = self._resolve_env(base_url)

        kwargs: dict[str, Any] = {"api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(**kwargs)

    @staticmethod
    def _resolve_env(value: str) -> str:
        """解析 ${ENV_VAR} 格式的环境变量占位符。"""
        if value.startswith("${") and value.endswith("}"):
            env_name = value[2:-1]
            return os.environ.get(env_name, "")
        return value

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发送聊天请求。

        Args:
            messages: 消息列表，格式 [{"role": "...", "content": "..."}]
            tools: 可选的 Function Calling 工具定义列表

        Returns:
            {
                "content": str | None,
                "tool_calls": list[dict] | None,
                "finish_reason": str,
                "usage": dict,
            }
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return {
                "content": f"[LLM 调用错误: {e}]",
                "tool_calls": None,
                "finish_reason": "error",
                "usage": {},
            }

        choice = response.choices[0]
        message = choice.message

        # 提取 tool_calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                    if tc.function.arguments else {},
                })

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            "LLM 响应: finish=%s, tokens=%s, tool_calls=%s",
            choice.finish_reason,
            usage.get("total_tokens", "?"),
            [tc["name"] for tc in (tool_calls or [])],
        )

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
            "usage": usage,
        }

    def generate_with_context(
        self,
        question: str,
        context_chunks: list[str],
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """基于检索到的上下文生成回答（不使用 Function Calling）。

        Args:
            question: 用户问题
            context_chunks: 检索到的上下文片段
            history: 历史对话
            system_prompt: 自定义系统提示词

        Returns:
            生成的回答文本
        """
        if system_prompt is None:
            system_prompt = (
                "你是一个专业的技术文档问答助手。请根据提供的文档片段回答用户的问题。\n"
                "规则：\n"
                "1. 如果文档片段足以回答问题，请准确、简洁地回答\n"
                "2. 如果文档片段不足以回答问题，请明确说明信息不足，不要编造\n"
                "3. 回答时引用具体的文件名和片段来源\n"
                "4. 如果用户用中文提问，请用中文回答"
            )

        # 构建上下文
        context_text = "\n\n---\n\n".join(
            f"[来源 {i+1}] {chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"## 文档上下文\n\n{context_text}\n\n## 用户问题\n\n{question}"},
        ]

        # 插入历史对话
        if history:
            messages = [messages[0]] + history + [messages[1]]

        result = self.chat(messages)
        return result.get("content", "[未能生成回答]")
