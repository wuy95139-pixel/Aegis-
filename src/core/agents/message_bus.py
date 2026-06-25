"""
MessageBus — Agent 间消息传递基础设施（发布-订阅模式）

从 Orchestrator 提取出的独立模块。
"""

import logging
from typing import Dict, List, Callable
from collections import deque

from src.models.schemas import AgentMessage

logger = logging.getLogger(__name__)


class MessageBus:
    """消息总线 — Agent 间松耦合通信"""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: deque[AgentMessage] = deque(maxlen=max_queue_size)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[AgentMessage] = []

    @property
    def history(self) -> List[AgentMessage]:
        return self._history

    def publish(self, message: AgentMessage) -> None:
        self._queue.append(message)
        self._history.append(message)
        targets = (
            self._subscribers.values()
            if message.receiver == "broadcast"
            else [self._subscribers.get(message.receiver, [])]
        )
        for handlers in targets:
            for handler in handlers:
                try:
                    handler(message)
                except Exception:
                    logger.exception(
                        f"MessageBus handler failed for message {message.id}"
                    )

    def subscribe(
        self, agent_name: str, handler: Callable[[AgentMessage], None]
    ) -> None:
        self._subscribers.setdefault(agent_name, []).append(handler)

    def get_history(self, limit: int = 50) -> List[AgentMessage]:
        return self._history[-limit:]
