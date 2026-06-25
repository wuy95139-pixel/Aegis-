"""
LLM 交互工具
===========
从 Orchestrator 提取出的 LLM function calling 和 streaming 基础设施。

包含：
  - execute_tool: 本地工具执行器
  - run_with_tools: 带 function calling 的非流式对话循环
  - stream_chat_with_tools: 带 function calling 的流式对话循环
"""

import json
import logging
from typing import List, Callable

from src.core.agents.intent_classifier import AVAILABLE_TOOLS
from src.core.tools.time_tools import (
    get_time_context,
    parse_chinese_time_expression,
    expression_to_cron,
    get_future_date,
    is_overdue,
)

logger = logging.getLogger(__name__)


def execute_tool(name: str, arguments: dict) -> str:
    """执行本地工具并返回结果字符串"""
    if name == "get_current_time":
        return get_time_context()
    elif name == "parse_time":
        text = arguments.get("text", "")
        if not text:
            return "错误：缺少 text 参数"
        result = parse_chinese_time_expression(text)
        if result:
            return result.isoformat()
        else:
            return f"无法解析时间表达: '{text}'。请尝试更具体的表述，如'明天下午3点'。"
    elif name == "time_to_cron":
        text = arguments.get("text", "")
        if not text:
            return "错误：缺少 text 参数"
        return expression_to_cron(text)
    elif name == "get_future_date":
        text = arguments.get("text", "")
        if not text:
            return "错误：缺少 text 参数"
        return get_future_date(text)
    elif name == "check_overdue":
        time_str = arguments.get("time_str", "")
        if not time_str:
            return "错误：缺少 time_str 参数"
        return is_overdue(time_str)
    else:
        return f"未知工具: {name}"


class LLMInteraction:
    """LLM 对话交互（function calling + streaming）"""

    def __init__(self, llm):
        self.llm = llm

    def run_with_tools(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        max_rounds: int = 3,
    ) -> str:
        """带 function calling 的 LLM 对话循环。

        如果 LLM 返回 tool_calls，执行工具并将结果回传。
        最多执行 max_rounds 轮工具调用。
        """
        for round_num in range(max_rounds):
            response = self.llm.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
            )

            tool_calls = response.get("tool_calls")
            if not tool_calls:
                return response.get("content", "")

            logger.info(
                f"Tool calls (round {round_num + 1}): "
                f"{[tc['function'] for tc in tool_calls]}"
            )

            assistant_msg = {
                "role": "assistant",
                "content": response.get("content") or "",
            }
            if response.get("reasoning_content"):
                assistant_msg["reasoning_content"] = response["reasoning_content"]
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            for tc in tool_calls:
                try:
                    args = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool arguments for {tc['function']}: "
                        f"{tc['arguments'][:200]}"
                    )
                    args = {}
                result_text = execute_tool(tc["function"], args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
                logger.debug(f"Tool result ({tc['function']}): {result_text}")

        logger.warning(
            f"Exceeded max tool rounds ({max_rounds}), getting final response"
        )
        return (
            self.llm.chat(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            .get("content", "")
        )

    def stream_chat_with_tools(
        self,
        messages: List[dict],
        stream_callback: Callable[[str], None],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """真正的流式聊天（支持工具调用）"""
        max_rounds = 3
        full_text = ""

        for round_num in range(max_rounds):
            round_content = ""
            tool_calls = None
            round_reasoning = None

            try:
                for event in self.llm.stream_chat_events(
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if event["type"] == "token":
                        stream_callback(event["content"])
                        round_content += event["content"]
                    elif event["type"] == "tool_calls":
                        tool_calls = event["tool_calls"]
                        round_reasoning = event.get("reasoning_content")
                    elif event["type"] == "done":
                        round_reasoning = (
                            event.get("reasoning_content") or round_reasoning
                        )
            except Exception as e:
                logger.warning(
                    f"Streaming failed (round {round_num}): {e}, falling back"
                )
                result = self.run_with_tools(messages)
                for i in range(0, len(result), 8):
                    stream_callback(result[i : i + 8])
                return full_text + result

            full_text += round_content

            if not tool_calls:
                return full_text

            logger.info(
                f"Stream tool calls (round {round_num + 1}): "
                f"{[tc['function'] for tc in tool_calls]}"
            )

            assistant_msg: dict = {"role": "assistant", "content": round_content}
            if round_reasoning:
                assistant_msg["reasoning_content"] = round_reasoning
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            for tc in tool_calls:
                try:
                    args = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse stream tool arguments for "
                        f"{tc['function']}: {tc['arguments'][:200]}"
                    )
                    args = {}
                result_text = execute_tool(tc["function"], args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })

        logger.warning(
            f"Exceeded max tool rounds ({max_rounds}), getting final response"
        )
        final = self.llm.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        text = final.get("content", "")
        for i in range(0, len(text), 8):
            stream_callback(text[i : i + 8])
        return full_text + text
