"""
core/memory/retriever.py 测试
============================
MemoryRetriever 的三路检索、降级、类型过滤测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.file_store import FileStore
from src.core.memory.types import MemoryType, MemoryFrontmatter
from src.models.schemas import MemoryEntry


class TestRetrieve:
    def test_three_way_retrieval(self, short_term_memory, mock_long_term):
        short_term_memory.add_turn(MagicMock(role="user", content="Hello"))
        mock_long_term.search.return_value = [
            MemoryEntry(id="m1", content="Memory content", source="test", similarity_score=0.9)
        ]
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term)

        result = retriever.retrieve("query", top_k=5)
        assert "relevant_memories" in result
        assert "recent_conversations" in result
        assert "file_memories" in result
        assert "combined_context" in result

    def test_chromadb_search_called(self, short_term_memory, mock_long_term):
        mock_long_term.count.return_value = 1  # 确保 ChromaDB 被视为可用
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term)
        retriever.retrieve("architecture", top_k=3)
        mock_long_term.search.assert_called()

    def test_file_store_fallback(self, short_term_memory, mock_long_term, temp_file_store):
        mock_long_term.search.return_value = []  # ChromaDB 为空
        temp_file_store.save(
            MemoryFrontmatter(name="arch", description="Architecture", type=MemoryType.PROJECT),
            "微服务架构设计文档。",
        )
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term, file_store=temp_file_store)
        result = retriever.retrieve("架构", top_k=5)
        assert len(result["file_memories"]) > 0

    def test_tag_filtering(self, short_term_memory, mock_long_term):
        mock_long_term.count.return_value = 1  # 确保 ChromaDB 可用
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term)
        retriever.retrieve("query", top_k=5, tags=["important"])
        mock_long_term.search.assert_called_once()
        call_kwargs = mock_long_term.search.call_args.kwargs
        assert call_kwargs.get("tags") == ["important"]

    def test_combined_context_built(self, short_term_memory, mock_long_term):
        mock_long_term.search.return_value = [
            MemoryEntry(id="m1", content="Important memory", source="test", similarity_score=0.9)
        ]
        short_term_memory.add_turn(MagicMock(role="user", content="Hello"))
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term)
        result = retriever.retrieve("important")
        assert len(result["combined_context"]) > 0


class TestRetrieveByType:
    def test_retrieve_user_profile(self, short_term_memory, mock_long_term, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="user_role", description="User role", type=MemoryType.USER, importance=0.8),
            "用户是资深后端工程师。",
        )
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term, file_store=temp_file_store)
        result = retriever.retrieve_user_profile()
        assert isinstance(result, dict)
        assert "user_profile" in result

    def test_retrieve_feedback_rules(self, short_term_memory, mock_long_term, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="rule1", description="Rule", type=MemoryType.FEEDBACK,
                            rule="Don't mock DB", severity="high"),
            "Rule content.",
        )
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term, file_store=temp_file_store)
        result = retriever.retrieve_feedback_rules()
        assert isinstance(result, dict)
        assert "feedback_rules" in result

    def test_retrieve_project_context(self, short_term_memory, mock_long_term, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="proj_context", description="Project", type=MemoryType.PROJECT,
                            fact="Deadline is Friday"),
            "Context content.",
        )
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term, file_store=temp_file_store)
        result = retriever.retrieve_project_context()
        assert isinstance(result, dict)
        assert "file_memories" in result


class TestExtractAndRemember:
    def test_stores_to_both_backends(self, short_term_memory, mock_long_term):
        retriever = MemoryRetriever(short_term=short_term_memory, long_term=mock_long_term)
        entry_id = retriever.extract_and_remember(
            "Important content about architecture.",
            source="test_chat",
            tags=["architecture"],
        )
        # 应该返回 ID
        assert entry_id is not None
        # ChromaDB store 被调用
        mock_long_term.store.assert_called_once()
