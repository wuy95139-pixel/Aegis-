"""
LLM 统一封装层
==============
设计决策：所有智能体通过此接口访问大模型，而非直接调用具体 SDK。
这样可以在 OpenAI / Anthropic / Azure / 本地模型之间无缝切换，
只需修改 config.yaml 中的 provider 字段。

可扩展点：
  - 新增 provider: 在 SUPPORTED_PROVIDERS 中注册新的工厂方法
  - 流式输出: 使用 stream=True 参数
  - 多模态: 在 messages 中传递 image_url 即可自动支持
"""

import os
import time as _time_module
import random
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    统一 LLM 接口封装
    默认使用 OpenAI 兼容 API (也支持 Anthropic、Azure、本地 vLLM 等)

    使用示例:
        llm = LLMProvider(config)
        reply = llm.chat([{"role": "user", "content": "你好"}])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典，含 provider, model, api_key, api_base 等字段
                    如果不传，从环境变量读取
        """
        self.config = config or {}
        self.provider = self.config.get("provider", "openai")

        # 从环境变量或配置中获取 API 信息
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        api_base = self.config.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.default_model = self.config.get("model") or os.getenv("DEFAULT_MODEL", "deepseek-v4-pro")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 4096)
        self._timeout = self.config.get("timeout", 120)
        self._retry_config = self.config.get("retry", {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
        })

        # 初始化客户端（禁用内置重试，由 _retry_with_backoff 控制）
        self.client = OpenAI(
            api_key=api_key, base_url=api_base,
            timeout=self._timeout, max_retries=0,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url=api_base,
            timeout=self._timeout, max_retries=0,
        )

        logger.info(f"LLM Provider initialized: provider={self.provider}, model={self.default_model}, base_url={api_base}")

    # ==================== Retry helper ====================

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        带指数退避的重试逻辑

        处理可重试的异常: 网络错误、速率限制(429)、服务器错误(5xx)
        不可重试的异常: 认证错误(401)、权限错误(403)、参数错误(400)
        """
        max_retries = self._retry_config.get("max_retries", 3)
        base_delay = self._retry_config.get("base_delay", 1.0)
        max_delay = self._retry_config.get("max_delay", 60.0)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                # 判断是否可重试
                if not self._is_retryable(e) or attempt >= max_retries:
                    if attempt >= max_retries:
                        logger.error(
                            f"LLM call failed after {max_retries + 1} attempts: {type(e).__name__}: {e}"
                        )
                    raise

                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s"
                )
                _time_module.sleep(delay)

    @staticmethod
    def _is_retryable(exception: Exception) -> bool:
        """判断异常是否可重试"""
        exc_name = type(exception).__name__
        # OpenAI SDK 异常类型
        retryable = {
            "RateLimitError", "APIConnectionError", "APITimeoutError",
            "InternalServerError", "ServiceUnavailableError",
        }
        if exc_name in retryable:
            return True
        # HTTP 状态码检查
        status = getattr(exception, "status_code", None)
        if status is not None and (status == 429 or status >= 500):
            return True
        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        同步聊天接口

        Args:
            messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
            model: 模型名，默认使用配置中的模型
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            tools: 工具定义列表 (OpenAI function calling 格式)
            tool_choice: 工具选择策略 ("auto" / "none" / {"type": "function", "function": {"name": "xxx"}})

        Returns:
            {"content": "...", "tool_calls": [...], "usage": {...}}
        """
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"

        logger.debug(f"Chat request: model={params['model']}, messages_count={len(messages)}")

        import time as _time
        _start = _time.time()
        try:
            # 使用指数退避重试
            response = self._retry_with_backoff(
                self.client.chat.completions.create, **params
            )
            _latency = (_time.time() - _start) * 1000
            result = self._parse_response(response)
            # 记录指标
            try:
                from src.utils.metrics import MetricsCollector, LLMCallMetrics
                MetricsCollector().record_llm_call(LLMCallMetrics(
                    model=params["model"],
                    prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=result.get("usage", {}).get("total_tokens", 0),
                    latency_ms=_latency,
                    success=True,
                ))
            except Exception:
                pass  # 指标记录失败不影响主流程
            return result
        except Exception:
            _latency = (_time.time() - _start) * 1000
            try:
                from src.utils.metrics import MetricsCollector, LLMCallMetrics
                MetricsCollector().record_llm_call(LLMCallMetrics(
                    model=params.get("model", "unknown"),
                    latency_ms=_latency,
                    success=False,
                ))
                MetricsCollector().record_error()
            except Exception:
                pass
            raise

    async def achat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步聊天接口 — 用于需要高并发的场景

        与 chat() 参数相同，返回结构相同。支持指数退避重试。
        """
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        import time as _time
        _start = _time.time()
        try:
            response = await self._async_retry_with_backoff(
                self.async_client.chat.completions.create, **params
            )
            _latency = (_time.time() - _start) * 1000
            result = self._parse_response(response)
            try:
                from src.utils.metrics import MetricsCollector, LLMCallMetrics
                MetricsCollector().record_llm_call(LLMCallMetrics(
                    model=params["model"],
                    prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=result.get("usage", {}).get("total_tokens", 0),
                    latency_ms=_latency,
                    success=True,
                ))
            except Exception:
                pass
            return result
        except Exception:
            _latency = (_time.time() - _start) * 1000
            try:
                from src.utils.metrics import MetricsCollector, LLMCallMetrics
                MetricsCollector().record_llm_call(LLMCallMetrics(
                    model=params.get("model", "unknown"),
                    latency_ms=_latency,
                    success=False,
                ))
                MetricsCollector().record_error()
            except Exception:
                pass
            raise

    async def _async_retry_with_backoff(self, func, *args, **kwargs):
        """异步版本的指数退避重试"""
        max_retries = self._retry_config.get("max_retries", 3)
        base_delay = self._retry_config.get("base_delay", 1.0)
        max_delay = self._retry_config.get("max_delay", 60.0)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._is_retryable(e) or attempt >= max_retries:
                    if attempt >= max_retries:
                        logger.error(
                            f"Async LLM call failed after {max_retries + 1} attempts: {type(e).__name__}: {e}"
                        )
                    raise

                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                logger.warning(
                    f"Async LLM call failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s"
                )
                # Use asyncio.sleep if available, fall back to synchronous
                try:
                    import asyncio
                    await asyncio.sleep(delay)
                except (ImportError, RuntimeError):
                    _time_module.sleep(delay)

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        文本嵌入 — 用于向量数据库存储和检索

        Args:
            texts: 待嵌入的文本列表
            model: 嵌入模型名

        Returns:
            嵌入向量列表
        """
        embed_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = self.client.embeddings.create(model=embed_model, input=texts)
        return [item.embedding for item in response.data]

    def stream_chat(self, messages, model=None, **kwargs):
        """
        流式聊天 — 用于实时输出的 UI 场景

        Yields:
            增量文本片段
        """
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs,
        }
        try:
            # 创建流时支持重试（仅连接阶段，非内容阶段）
            stream = self._retry_with_backoff(
                self.client.chat.completions.create, **params
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.warning(f"Stream chat failed: {type(e).__name__}: {e}")
            yield f"\n[流式输出中断: {type(e).__name__}]"

    def stream_chat_events(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """
        流式聊天 + 工具调用支持 — 真正降低 TTFT

        与 stream_chat 不同，此方法支持 tools 参数并正确处理工具调用。
        当 LLM 决定调用工具时，text token 输出自然为空，
        工具调用信息通过 'tool_calls' 事件返回。

        Yields:
            {"type": "token", "content": "..."}  — 文本增量
            {"type": "tool_calls", "tool_calls": [...]} — 累积完成的工具调用列表

        Returns:
            完整响应 dict，同 chat() 返回格式
        """
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            stream = self._retry_with_backoff(
                self.client.chat.completions.create, **params
            )
        except Exception as e:
            logger.warning(f"Stream chat events connection failed: {type(e).__name__}: {e}")
            yield {"type": "token", "content": f"\n[流式连接失败: {type(e).__name__}]"}
            yield {"type": "done", "reasoning_content": None}
            return {"content": "", "tool_calls": None, "reasoning_content": None}

        accumulated_tool_calls: Dict[int, Dict[str, str]] = {}
        full_content = ""
        reasoning_content = ""
        final_tool_calls = None

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                # DeepSeek thinking mode: accumulate reasoning_content for round-trip
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content += rc

                # 文本增量 → 直接 yield 给 UI
                if delta.content:
                    full_content += delta.content
                    yield {"type": "token", "content": delta.content}

                # 工具调用增量 → 跨 chunk 累积（不 yield 给 UI）
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "function": "",
                                "arguments": "",
                            }
                        tc = accumulated_tool_calls[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tc["function"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tc["arguments"] += tc_delta.function.arguments

                # 工具调用完成 → yield 结构化工具调用
                if finish == "tool_calls" and accumulated_tool_calls:
                    final_tool_calls = [
                        {
                            "id": tc["id"],
                            "function": tc["function"],
                            "arguments": tc["arguments"],
                        }
                        for tc in accumulated_tool_calls.values()
                        if tc["id"] and tc["function"]
                    ]
                    yield {
                        "type": "tool_calls",
                        "tool_calls": final_tool_calls,
                        "reasoning_content": reasoning_content or None,
                    }

            # DeepSeek thinking mode: reasoning_content must be round-tripped.
            # Yield it so callers can attach it to assistant messages.
            yield {
                "type": "done",
                "reasoning_content": reasoning_content or None,
            }

        except Exception as e:
            logger.warning(f"Stream chat events interrupted: {type(e).__name__}: {e}")
            yield {"type": "token", "content": f"\n[流式传输中断: {type(e).__name__}]"}
            yield {"type": "done", "reasoning_content": reasoning_content or None}

        return {
            "content": full_content,
            "tool_calls": final_tool_calls,
            "reasoning_content": reasoning_content or None,
        }

    def _parse_response(self, response) -> Dict[str, Any]:
        """解析 API 响应为统一格式"""
        choice = response.choices[0]
        msg = choice.message
        result = {
            "content": msg.content or "",
            "tool_calls": None,
            "reasoning_content": getattr(msg, "reasoning_content", None),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "finish_reason": choice.finish_reason,
        }

        # 解析 tool_calls (如果存在)
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]

        return result


# --- 工厂方法：便于未来扩展新的 LLM 提供商 ---

def create_llm_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """
    LLM 工厂方法
    当前仅返回 OpenAI 兼容的 LLMProvider。
    未来可扩展为根据 config.provider 返回不同实现。

    可扩展点: 如需支持 Anthropic 原生 API，在此处路由到 AnthropicProvider。
    """
    return LLMProvider(config)
