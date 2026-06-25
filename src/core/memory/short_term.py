"""
短期记忆模块
============
基于滑动窗口的对话上下文缓冲区。
存储最近 N 轮对话的完整内容，用于当前会话的上下文填充。

设计决策：
  - 使用 deque 实现高效滑动窗口 (O(1) 追加和弹出)
  - 按 token 数估算来决定是否触发压缩 (summarize)
  - 短期记忆仅存在内存中，进程重启即丢失 (会话级)
"""

import tiktoken
import logging
from collections import deque
from typing import Any, List, Optional

from src.models.schemas import ConversationTurn

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆 — 对话上下文滑动窗口

    使用示例:
        stm = ShortTermMemory(max_tokens=16000, window_size=20)
        stm.add_turn(ConversationTurn(role="user", content="..."))
        context = stm.get_context()  # 获取当前窗口内的对话
    """

    def __init__(
        self,
        max_tokens: int = 16000,
        window_size: int = 20,
        model: str = "gpt-4",
        llm: Any = None,
    ):
        """
        Args:
            max_tokens: 最大 token 数，超过后自动压缩
            window_size: 最大轮次 (user+assistant 对话轮数)
            model: 用于 token 编码的模型名
            llm: 可选的 LLMProvider，用于 LLM 语义压缩
        """
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.model = model
        self._llm = llm  # 可选的 LLM provider，用于语义压缩

        # 核心存储：deque 保存 ConversationTurn
        self._turns: deque[ConversationTurn] = deque(maxlen=window_size * 2)

        # 压缩后的摘要 (当对话超过 token 限制时使用)
        self._compressed_summary: Optional[str] = None

        try:
            self._tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def add_turn(self, turn: ConversationTurn) -> None:
        """添加一轮对话"""
        self._turns.append(turn)
        logger.debug(f"Added turn: role={turn.role}, tokens={self._count_tokens(turn.content)}")

        # 检查是否需要压缩
        if self._total_tokens() > self.max_tokens:
            self._compress()

    def get_context(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        获取当前窗口的对话上下文

        Args:
            n: 返回最近 n 轮，None 表示全部

        Returns:
            对话轮次列表 (按时间排序)
        """
        turns = list(self._turns)
        if n is not None:
            turns = turns[-n:]
        return turns

    def get_messages_for_llm(self, system_prompt: Optional[str] = None) -> List[dict]:
        """
        转为 LLM 可用的 messages 格式
        如果有压缩摘要，在最前面插入摘要作为 system 消息的补充

        Args:
            system_prompt: 系统提示词

        Returns:
            [{"role": "...", "content": "..."}, ...]
        """
        messages = []

        # 构建完整的 system 消息
        system_content = system_prompt or ""
        if self._compressed_summary:
            system_content += f"\n\n[历史对话摘要]\n{self._compressed_summary}"

        if system_content:
            messages.append({"role": "system", "content": system_content})

        for turn in self._turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    def clear(self) -> None:
        """清空短期记忆"""
        self._turns.clear()
        self._compressed_summary = None
        logger.info("Short-term memory cleared")

    def get_summary(self) -> Optional[str]:
        """获取压缩摘要"""
        return self._compressed_summary

    # --- 内部方法 ---

    def _total_tokens(self) -> int:
        """估算当前窗口的总 token 数"""
        total = 0
        for turn in self._turns:
            total += self._count_tokens(turn.content)
        return total

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        return len(self._tokenizer.encode(text))

    def _compress(self) -> None:
        """
        压缩策略：优先使用 LLM 语义压缩，回退到截断

        当 llm 参数可用时，调用 LLM 将前半部分对话总结为摘要；
        LLM 不可用或调用失败时，回退到 500 字符截断。
        """
        turns = list(self._turns)
        if len(turns) < 4:
            return  # 太少不压缩

        mid = len(turns) // 2
        old_turns = turns[:mid]
        recent_turns = turns[mid:]

        # 构建待压缩的对话文本
        conversation_text = "\n".join(
            f"[{t.role}]: {t.content[:200]}" for t in old_turns
        )

        # 优先使用 LLM 语义压缩
        summary = None
        if self._llm is not None:
            try:
                from src.core.memory._memory_llm_ops import summarize_text
                summary = summarize_text(self._llm, conversation_text)
                if summary:
                    logger.info(f"LLM compression: {len(old_turns)} turns → {len(summary)} chars summary")
            except Exception as e:
                logger.warning(f"LLM compression failed, falling back to truncation: {e}")

        # 回退: 500 字符截断
        if not summary:
            summary = conversation_text[:500]
            if len(conversation_text) > 500:
                summary += "..."

        self._compressed_summary = (
            f"之前对话涉及 {len(old_turns)} 轮，主要话题摘要：\n{summary}"
        )

        # 重新设置 deque
        self._turns = deque(recent_turns, maxlen=self.window_size * 2)

        logger.info(f"Compressed {len(old_turns)} old turns into summary, kept {len(recent_turns)} recent turns")
