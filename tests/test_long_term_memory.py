"""
core/memory/long_term.py 测试
============================
LongTermMemory (ChromaDB) 的 store/search/delete/count 测试。
Mock ChromaDB 避免 SQLite 和 embedding API 调用。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid

from src.core.memory.long_term import LongTermMemory
from src.models.schemas import MemoryEntry


@pytest.fixture
def mock_chroma_collection():
    """创建完全 mock 的 ChromaDB collection。"""
    col = MagicMock()
    col.count.return_value = 3
    col.name = "test_collection"
    col.query.return_value = {
        "ids": [["mem_1", "mem_2"]],
        "documents": [["Content about architecture.", "User preferences."]],
        "metadatas": [[
            {"source": "chat_001", "tags": "architecture,microservices", "importance": 0.8, "created_at": "2025-01-01T00:00:00"},
            {"source": "chat_002", "tags": "preference,中文", "importance": 0.6, "created_at": "2025-01-02T00:00:00"},
        ]],
        "distances": [[0.1, 0.5]],
    }
    return col


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """创建完全 mock 的 ChromaDB PersistentClient。"""
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.heartbeat.return_value = None
    return client


@pytest.fixture
def mock_embedding_fn():
    """Mock embedding function。"""
    ef = Mock()
    return ef


@pytest.fixture
def ltm_with_mocks(mock_chroma_client, mock_chroma_collection, mock_embedding_fn):
    """创建带有 mock ChromaDB 的 LongTermMemory 实例。"""
    with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm.persist_dir = "/mock/path"
        ltm.collection_name = "test_memory"
        ltm.embedding_model = "test-model"
        ltm._is_persistent = True
        ltm.client = mock_chroma_client
        ltm.embedding_fn = mock_embedding_fn
        ltm.collection = mock_chroma_collection
        ltm._using_default_embedding = False
        return ltm


class TestStore:
    def test_store_returns_id(self, ltm_with_mocks, mock_chroma_collection):
        entry = MemoryEntry(
            id="test_001",
            content="This is a test memory about microservices.",
            source="test_chat",
            tags=["architecture", "test"],
            importance=0.8,
        )
        result = ltm_with_mocks.store(entry)
        assert result == "test_001"
        mock_chroma_collection.add.assert_called_once()

    def test_store_generates_id_if_empty(self, ltm_with_mocks):
        entry = MemoryEntry(
            id="",
            content="Memory without pre-assigned ID.",
            source="test",
            tags=["test"],
        )
        result = ltm_with_mocks.store(entry)
        assert result != ""
        assert len(result) > 0

    def test_store_with_embedding(self, ltm_with_mocks, mock_chroma_collection):
        entry = MemoryEntry(
            id="embed_001",
            content="Memory with custom embedding.",
            source="test",
            embedding=[0.1, 0.2, 0.3],
        )
        ltm_with_mocks.store(entry)
        # 验证调用时包含 embeddings 参数
        called_kwargs = mock_chroma_collection.add.call_args.kwargs
        assert "embeddings" in called_kwargs

    def test_store_without_embedding(self, ltm_with_mocks, mock_chroma_collection):
        entry = MemoryEntry(
            id="no_embed",
            content="Memory without embedding.",
            source="test",
        )
        ltm_with_mocks.store(entry)
        called_kwargs = mock_chroma_collection.add.call_args.kwargs
        assert "embeddings" not in called_kwargs

    def test_store_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        entry = MemoryEntry(id="nc", content="No collection", source="test")
        result = ltm_with_mocks.store(entry)
        assert result == "nc"

    def test_store_metadata_format(self, ltm_with_mocks, mock_chroma_collection):
        entry = MemoryEntry(
            id="meta_test",
            content="Testing metadata.",
            source="chat_abc",
            tags=["tag1", "tag2"],
            importance=0.7,
        )
        ltm_with_mocks.store(entry)
        called_kwargs = mock_chroma_collection.add.call_args.kwargs
        metadata = called_kwargs["metadatas"][0]
        assert metadata["source"] == "chat_abc"
        assert metadata["tags"] == "tag1,tag2"
        assert metadata["importance"] == 0.7


class TestSearch:
    def test_search_returns_entries(self, ltm_with_mocks):
        results = ltm_with_mocks.search("architecture")
        assert len(results) > 0
        assert results[0].id == "mem_1"

    def test_search_with_tag_filter(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.search("test", tags=["architecture"])
        query_kwargs = mock_chroma_collection.query.call_args.kwargs
        assert "where" in query_kwargs
        assert query_kwargs["where"] is not None

    def test_search_with_source_filter(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.search("test", source="chat_001")
        query_kwargs = mock_chroma_collection.query.call_args.kwargs
        where = query_kwargs.get("where")
        assert where is not None

    def test_top_k_limit(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.search("test", top_k=3)
        query_kwargs = mock_chroma_collection.query.call_args.kwargs
        assert query_kwargs["n_results"] == 3

    def test_top_k_capped_at_20(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.search("test", top_k=50)
        query_kwargs = mock_chroma_collection.query.call_args.kwargs
        assert query_kwargs["n_results"] == 20

    def test_search_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        results = ltm_with_mocks.search("test")
        assert results == []

    def test_similarity_score_clamped(self, ltm_with_mocks):
        """余弦距离可能 > 1，相似度应 clamp 到 >= 0。"""
        results = ltm_with_mocks.search("test")
        for entry in results:
            if entry.similarity_score is not None:
                assert entry.similarity_score >= 0.0

    def test_search_returns_memory_entries(self, ltm_with_mocks):
        results = ltm_with_mocks.search("query")
        for entry in results:
            assert isinstance(entry, MemoryEntry)


class TestCount:
    def test_count_delegates_to_collection(self, ltm_with_mocks):
        assert ltm_with_mocks.count() == 3

    def test_count_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        assert ltm_with_mocks.count() == 0


class TestDelete:
    def test_delete_calls_collection(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.delete("mem_1")
        mock_chroma_collection.delete.assert_called_once_with(ids=["mem_1"])

    def test_delete_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        ltm_with_mocks.delete("mem_1")  # 不应崩溃


class TestForgetSource:
    def test_forget_source_calls_collection(self, ltm_with_mocks, mock_chroma_collection):
        ltm_with_mocks.forget_source("chat_001")
        mock_chroma_collection.delete.assert_called_once()

    def test_forget_source_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        ltm_with_mocks.forget_source("chat_001")  # 不应崩溃


class TestListBySource:
    def test_returns_entries(self, ltm_with_mocks):
        mock_get_result = {
            "ids": ["mem_1"],
            "documents": ["Content"],
            "metadatas": [{"source": "chat_001", "tags": "test", "importance": 0.5, "created_at": "2025-01-01T00:00:00"}],
        }
        ltm_with_mocks.collection.get.return_value = mock_get_result
        results = ltm_with_mocks.list_by_source("chat_001")
        assert len(results) == 1
        assert results[0].id == "mem_1"

    def test_no_collection(self, ltm_with_mocks):
        ltm_with_mocks.collection = None
        assert ltm_with_mocks.list_by_source("test") == []


class TestSQLiteLockCleanup:
    def test_cleanup_method_exists(self):
        assert hasattr(LongTermMemory, '_cleanup_sqlite_locks')

    def test_cleanup_safe(self, tmp_path):
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm.persist_dir = str(tmp_path)
        lock_dir = tmp_path / "subdir"
        lock_dir.mkdir(exist_ok=True)
        (lock_dir / "test.lock").touch()
        ltm._cleanup_sqlite_locks()  # 不应崩溃
