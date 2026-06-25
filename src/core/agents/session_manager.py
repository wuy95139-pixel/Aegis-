"""
会话管理器 (SessionManager)
===========================
从 Orchestrator 提取出的会话隔离逻辑。

每个 session 拥有独立的 ShortTermMemory，避免不同用户/会话的对话历史互相污染。
共享资源（LongTermMemory, FileStore, ChromaDB）通过全局 MemoryRetriever 复用。
自动清理超过 TTL 未活动的会话，防止内存泄漏。
"""

import time
import logging
import threading
from typing import Any, Dict, Optional

from src.core.memory.short_term import ShortTermMemory
from src.core.memory.retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class SessionManager:
    """管理多会话的专属记忆"""

    def __init__(
        self,
        global_memory: Optional[MemoryRetriever] = None,
        short_term_max_tokens: int = 16000,
        short_term_window: int = 20,
        session_ttl_seconds: int = 7200,
        llm: Any = None,
    ):
        self._global_memory = global_memory
        self._llm = llm  # 用于会话级短记忆的 LLM 语义压缩
        self._config = {
            "short_term_max_tokens": short_term_max_tokens,
            "short_term_window": short_term_window,
            "session_ttl_seconds": session_ttl_seconds,
        }
        self._sessions: Dict[str, MemoryRetriever] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> MemoryRetriever:
        """获取或创建会话专属的 MemoryRetriever。

        Args:
            session_id: 会话标识符（由前端生成并持久化在 localStorage）

        Returns:
            会话专属的 MemoryRetriever 实例
        """
        now = time.time()

        with self._lock:
            # 自动清理过期会话
            self._cleanup_locked(now)

            # 创建或获取会话记忆
            if session_id not in self._sessions:
                session_stm = ShortTermMemory(
                    max_tokens=self._config["short_term_max_tokens"],
                    window_size=self._config["short_term_window"],
                    llm=self._llm,
                )
                global_ltm = (
                    self._global_memory.long_term
                    if self._global_memory
                    else None
                )
                global_fs = (
                    self._global_memory.file_store
                    if self._global_memory
                    else None
                )

                session_retriever = MemoryRetriever(
                    short_term=session_stm,
                    long_term=global_ltm,
                    file_store=global_fs,
                )
                self._sessions[session_id] = session_retriever
                logger.info(
                    f"Session created: {session_id[:8]}... "
                    f"(total: {len(self._sessions)})"
                )

            self._last_access[session_id] = now
            return self._sessions[session_id]

    def cleanup(self, session_id: str) -> bool:
        """清理会话记忆（用户主动清除或会话过期时调用）"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._last_access.pop(session_id, None)
                logger.info(f"Session cleaned up: {session_id[:8]}...")
                return True
        return False

    def cleanup_expired(self):
        """定期清理所有过期 session"""
        with self._lock:
            self._cleanup_locked(time.time())

    def get_existing(self, session_id: str) -> Optional[MemoryRetriever]:
        """线程安全地获取已存在的 session memory，不创建新的。"""
        with self._lock:
            return self._sessions.get(session_id)

    # ===================== Internal =====================

    def _cleanup_locked(self, now: float):
        """在持有 lock 的前提下清理过期会话"""
        ttl = self._config["session_ttl_seconds"]
        expired = [
            sid
            for sid, last_access in self._last_access.items()
            if now - last_access > ttl
        ]
        for sid in expired:
            if sid in self._sessions:
                del self._sessions[sid]
            del self._last_access[sid]
        if expired:
            logger.info(
                f"Cleaned up {len(expired)} expired sessions "
                f"(remaining: {len(self._sessions)})"
            )
