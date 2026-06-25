"""
指标收集器
==========
轻量级指标收集模块，跟踪 LLM 调用统计、请求计数、错误率等。

使用示例:
    from src.utils.metrics import MetricsCollector, LLMCallMetrics

    collector = MetricsCollector()
    collector.record_llm_call(LLMCallMetrics(
        model="deepseek-v4-pro",
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        latency_ms=850.0, success=True,
    ))
    collector.record_request("research")
    stats = collector.get_stats()
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LLMCallMetrics:
    """单次 LLM 调用的指标"""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True


class MetricsCollector:
    """
    全局指标收集器（单例，线程安全）

    收集以下指标:
      - LLM 调用次数、Token 用量、延迟
      - 总请求次数、错误次数
      - 意图分布统计
      - 估算成本（基于 DeepSeek 定价）
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._reset()
                    cls._instance = instance
        return cls._instance

    def _reset(self):
        """重置所有计数器"""
        self._llm_calls: List[LLMCallMetrics] = []
        self._request_count: int = 0
        self._error_count: int = 0
        self._start_time: float = time.time()
        self._intent_counts: Dict[str, int] = defaultdict(int)

    # ---- 记录方法 ----

    def record_llm_call(self, metrics: LLMCallMetrics):
        """记录一次 LLM API 调用（线程安全）"""
        with self._lock:
            self._llm_calls.append(metrics)

    def record_request(self, intent: str = "unknown"):
        """记录一次用户请求（线程安全）"""
        with self._lock:
            self._request_count += 1
            self._intent_counts[intent] += 1

    def record_error(self):
        """记录一次错误（线程安全）"""
        with self._lock:
            self._error_count += 1

    # ---- 统计查询 ----

    def get_stats(self) -> dict:
        """获取完整统计摘要（线程安全读取）"""
        with self._lock:
            uptime = time.time() - self._start_time
            total_tokens = sum(c.total_tokens for c in self._llm_calls)
            total_llm_calls = len(self._llm_calls)
            avg_latency = (
                sum(c.latency_ms for c in self._llm_calls) / total_llm_calls
                if total_llm_calls > 0 else 0.0
            )
            success_rate = (
                sum(1 for c in self._llm_calls if c.success) / total_llm_calls
                if total_llm_calls > 0 else 1.0
            )
            recent_calls = [
                {
                    "model": c.model,
                    "tokens": c.total_tokens,
                    "latency_ms": round(c.latency_ms, 1),
                    "success": c.success,
                }
                for c in self._llm_calls[-10:]
            ]
            intent_dist = dict(self._intent_counts)
            request_count = self._request_count
            error_count = self._error_count
            # Copy llm_calls for cost estimation outside lock
            llm_calls_snapshot = list(self._llm_calls)

        # Cost estimation outside lock to avoid blocking
        cost = 0.0
        for c in llm_calls_snapshot:
            cost += c.prompt_tokens * 0.14 / 1_000_000
            cost += c.completion_tokens * 0.28 / 1_000_000

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": request_count,
            "total_errors": error_count,
            "error_rate": round(error_count / max(request_count, 1), 4),
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "success_rate": round(success_rate, 4),
            "estimated_cost_usd": round(cost, 6),
            "intent_distribution": intent_dist,
            "recent_llm_calls": recent_calls,
        }

    def get_token_usage(self) -> dict:
        """获取 Token 用量统计"""
        prompt_tokens = sum(c.prompt_tokens for c in self._llm_calls)
        completion_tokens = sum(c.completion_tokens for c in self._llm_calls)
        return {
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_calls": len(self._llm_calls),
        }

    # ---- 内部方法 ----

    def _estimate_cost(self) -> float:
        """
        估算 LLM 调用成本（基于 DeepSeek 定价）
        DeepSeek-V3: $0.14/1M input tokens, $0.28/1M output tokens
        """
        cost = 0.0
        for c in self._llm_calls:
            cost += c.prompt_tokens * 0.14 / 1_000_000
            cost += c.completion_tokens * 0.28 / 1_000_000
        return cost

    def reset(self):
        """重置所有指标（通常用于测试）"""
        with self._lock:
            self._reset()
