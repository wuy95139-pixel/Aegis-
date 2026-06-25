"""
core/memory/short_term.py 测试
=============================
ShortTermMemory 滑动窗口、压缩、LLM 消息格式测试。
"""

import pytest
from src.core.memory.short_term import ShortTermMemory
from src.models.schemas import ConversationTurn


class TestAddTurn:
    def test_add_single_turn(self, short_term_memory):
        turn = ConversationTurn(role="user", content="Hello")
        short_term_memory.add_turn(turn)
        ctx = short_term_memory.get_context()
        assert len(ctx) == 1
        assert ctx[0].content == "Hello"

    def test_add_multiple_turns(self, short_term_memory):
        for i in range(5):
            short_term_memory.add_turn(ConversationTurn(role="user", content=f"Msg {i}"))
        ctx = short_term_memory.get_context()
        assert len(ctx) == 5

    def test_window_overflow_pops_oldest(self):
        """超出 window 容量（非常小的值）时最早的消息被弹出。"""
        stm = ShortTermMemory(max_tokens=100000, window_size=2)
        for i in range(6):
            stm.add_turn(ConversationTurn(role="user", content=f"Message {i}"))
        ctx = stm.get_context()
        # deque maxlen = window_size * 2 = 4
        assert len(ctx) <= 4
        # 最早的消息应该被弹出
        contents = [t.content for t in ctx]
        assert "Message 0" not in contents

    def test_tokens_tracked(self, short_term_memory):
        turn = ConversationTurn(role="user", content="Hello world")
        short_term_memory.add_turn(turn)
        # 验证 turn 存储不会被破坏
        ctx = short_term_memory.get_context()
        assert ctx[0].role == "user"


class TestGetContext:
    def test_get_all(self, short_term_memory):
        short_term_memory.add_turn(ConversationTurn(role="user", content="Q1"))
        short_term_memory.add_turn(ConversationTurn(role="assistant", content="A1"))
        ctx = short_term_memory.get_context()
        assert len(ctx) == 2

    def test_get_last_n(self, short_term_memory):
        for i in range(10):
            short_term_memory.add_turn(ConversationTurn(role="user", content=f"Msg {i}"))
        ctx = short_term_memory.get_context(n=3)
        assert len(ctx) == 3
        assert ctx[-1].content == "Msg 9"

    def test_empty_returns_empty_list(self, short_term_memory):
        assert short_term_memory.get_context() == []


class TestGetMessagesForLLM:
    def test_returns_llm_format(self, short_term_memory):
        short_term_memory.add_turn(ConversationTurn(role="user", content="Hello"))
        messages = short_term_memory.get_messages_for_llm()
        assert isinstance(messages, list)
        assert all("role" in m and "content" in m for m in messages)

    def test_includes_system_prompt(self, short_term_memory):
        messages = short_term_memory.get_messages_for_llm(system_prompt="You are helpful.")
        assert messages[0]["role"] == "system"
        assert "You are helpful." in messages[0]["content"]

    def test_no_system_prompt_omits_system(self, short_term_memory):
        short_term_memory.add_turn(ConversationTurn(role="user", content="Hello"))
        messages = short_term_memory.get_messages_for_llm()
        # 没有 system_prompt 和 compressed_summary 时不应该有 system 消息
        roles = [m["role"] for m in messages]
        assert "system" not in roles

    def test_includes_compressed_summary(self, short_term_memory):
        # 强制设置摘要
        short_term_memory._compressed_summary = "Previous conversation summary"
        messages = short_term_memory.get_messages_for_llm(system_prompt="Be helpful.")
        assert "system" == messages[0]["role"]
        assert "历史对话摘要" in messages[0]["content"]


class TestClear:
    def test_clears_all_turns(self, short_term_memory):
        short_term_memory.add_turn(ConversationTurn(role="user", content="Hello"))
        short_term_memory.clear()
        assert short_term_memory.get_context() == []

    def test_clears_summary(self, short_term_memory):
        short_term_memory._compressed_summary = "Some summary"
        short_term_memory.clear()
        assert short_term_memory.get_summary() is None


class TestCompress:
    def test_triggers_on_low_token_threshold(self):
        """设置很低的 max_tokens 确保压缩触发。"""
        stm = ShortTermMemory(max_tokens=10, window_size=20)
        for i in range(8):
            stm.add_turn(ConversationTurn(role="user", content=f"Long message number {i} with extra text"))
        # 应该触发了压缩：要么生成摘要，要么窗口被裁剪
        has_summary = stm.get_summary() is not None
        context_reduced = len(stm.get_context()) < 8
        assert has_summary or context_reduced, (
            f"Expected compression but got summary={has_summary}, context_len={len(stm.get_context())}"
        )

    def test_preserves_recent_turns(self):
        stm = ShortTermMemory(max_tokens=10, window_size=20)
        for i in range(10):
            stm.add_turn(ConversationTurn(role="user", content=f"Very long message number {i} with lots of text"))
        ctx = stm.get_context()
        # 压缩后后半部分应该保留
        last_content = ctx[-1].content
        assert "9" in last_content

    def test_too_few_turns_no_compress(self):
        """少于 4 轮不压缩。"""
        stm = ShortTermMemory(max_tokens=1, window_size=20)
        for i in range(3):
            stm.add_turn(ConversationTurn(role="user", content="A" * 100))
        # 因为只有 3 轮，_compress 直接 return
        # 所以即使超出 token 限制也不会压缩
        assert stm.get_summary() is None
