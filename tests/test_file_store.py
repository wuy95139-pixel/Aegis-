"""
core/memory/file_store.py 测试
==============================
FileStore 的 CRUD、查询、全文搜索、索引管理测试。
使用 temp_file_store fixture（临时目录 + 真实 FileStore）。
"""

import pytest
from pathlib import Path

from src.core.memory.types import MemoryType, MemoryFrontmatter
from src.core.memory.file_store import FileStore


# ==================== Save / Get / Delete / Update ====================

class TestSave:
    def test_save_creates_file(self, temp_file_store):
        fm = MemoryFrontmatter(name="test_mem", description="A test", type=MemoryType.USER)
        path = temp_file_store.save(fm, "Hello world content")
        assert Path(path).exists()

    def test_save_contains_frontmatter(self, temp_file_store):
        fm = MemoryFrontmatter(
            name="test_fm",
            description="Testing frontmatter",
            type=MemoryType.PROJECT,
            tags=["test"],
        )
        temp_file_store.save(fm, "This is the body.")
        entry = temp_file_store.get("test_fm", MemoryType.PROJECT)
        assert entry["frontmatter"].name == "test_fm"
        assert entry["frontmatter"].description == "Testing frontmatter"

    def test_save_updates_index(self, temp_file_store):
        fm = MemoryFrontmatter(name="indexed", description="Indexed entry", type=MemoryType.USER)
        temp_file_store.save(fm, "Content", update_index=True)
        index = temp_file_store.get_index_entries()
        assert "user" in index
        assert any(e["name"] == "indexed" for e in index.get("user", []))

    def test_save_auto_sets_timestamps(self, temp_file_store):
        fm = MemoryFrontmatter(name="timestamps", description="Test", type=MemoryType.USER)
        temp_file_store.save(fm, "Content")
        entry = temp_file_store.get("timestamps", MemoryType.USER)
        assert entry["frontmatter"].created_at is not None
        assert entry["frontmatter"].updated_at is not None

    def test_save_chinese_content(self, temp_file_store):
        fm = MemoryFrontmatter(name="chinese", description="中文测试", type=MemoryType.USER)
        temp_file_store.save(fm, "用户喜欢使用中文交流，偏好简洁的回答风格。")
        entry = temp_file_store.get("chinese", MemoryType.USER)
        assert "中文" in entry["content"]


class TestGet:
    def test_get_existing_memory(self, temp_file_store):
        fm = MemoryFrontmatter(name="existing", description="Exists", type=MemoryType.USER)
        temp_file_store.save(fm, "Test content")
        entry = temp_file_store.get("existing", MemoryType.USER)
        assert entry is not None
        assert entry["content"] == "Test content"

    def test_get_missing_returns_none(self, temp_file_store):
        assert temp_file_store.get("nonexistent", MemoryType.USER) is None

    def test_get_across_types(self, temp_file_store):
        fm = MemoryFrontmatter(name="findme", description="Hidden", type=MemoryType.PROJECT)
        temp_file_store.save(fm, "Project content")
        # 不指定类型应能找到
        entry = temp_file_store.get("findme")
        assert entry is not None
        assert entry["content"] == "Project content"


class TestDelete:
    def test_delete_removes_file(self, temp_file_store):
        fm = MemoryFrontmatter(name="to_delete", description="Will be deleted", type=MemoryType.USER)
        temp_file_store.save(fm, "Content")
        assert temp_file_store.delete("to_delete", MemoryType.USER) is True
        assert temp_file_store.get("to_delete", MemoryType.USER) is None

    def test_delete_removes_index_entry(self, temp_file_store):
        fm = MemoryFrontmatter(name="indexed_del", description="Indexed to delete", type=MemoryType.USER)
        temp_file_store.save(fm, "Content")
        temp_file_store.delete("indexed_del", MemoryType.USER)
        index = temp_file_store.get_index_entries()
        user_entries = index.get("user", [])
        assert not any(e["name"] == "indexed_del" for e in user_entries)

    def test_delete_nonexistent_returns_false(self, temp_file_store):
        assert temp_file_store.delete("nonexistent", MemoryType.USER) is False


class TestUpdate:
    def test_update_content_preserves_frontmatter(self, temp_file_store):
        fm = MemoryFrontmatter(name="updatable", description="Original desc", type=MemoryType.USER, importance=0.5)
        temp_file_store.save(fm, "Original content")
        assert temp_file_store.update("updatable", MemoryType.USER, content="New content") is True
        entry = temp_file_store.get("updatable", MemoryType.USER)
        assert entry["content"] == "New content"
        assert entry["frontmatter"].description == "Original desc"

    def test_update_frontmatter_fields(self, temp_file_store):
        fm = MemoryFrontmatter(name="fm_updatable", description="Old", type=MemoryType.USER)
        temp_file_store.save(fm, "Content")
        temp_file_store.update("fm_updatable", MemoryType.USER, frontmatter_updates={"importance": 0.9})
        entry = temp_file_store.get("fm_updatable", MemoryType.USER)
        assert entry["frontmatter"].importance == 0.9

    def test_update_nonexistent_returns_false(self, temp_file_store):
        assert temp_file_store.update("nonexistent", MemoryType.USER, content="New") is False


# ==================== Query Operations ====================

class TestListByType:
    def test_returns_correct_type_only(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="user1", description="U", type=MemoryType.USER),
            "User content",
        )
        temp_file_store.save(
            MemoryFrontmatter(name="proj1", description="P", type=MemoryType.PROJECT),
            "Project content",
        )
        users = temp_file_store.list_by_type(MemoryType.USER)
        assert all(e["frontmatter"].type == MemoryType.USER for e in users)
        assert len(users) == 1

    def test_respects_limit(self, temp_file_store):
        for i in range(5):
            temp_file_store.save(
                MemoryFrontmatter(name=f"user_{i}", description=f"Desc {i}", type=MemoryType.USER),
                f"Content {i}",
            )
        results = temp_file_store.list_by_type(MemoryType.USER, limit=3)
        assert len(results) == 3

    def test_returns_empty_for_empty_dir(self, temp_file_store):
        assert temp_file_store.list_by_type(MemoryType.USER) == []


class TestSearchByTags:
    def test_exact_tag_match(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="tagged", description="T", type=MemoryType.USER, tags=["important"]),
            "Content",
        )
        results = temp_file_store.search_by_tags(["important"])
        assert len(results) == 1
        assert results[0]["frontmatter"].name == "tagged"

    def test_multi_tag_search(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="multi", description="M", type=MemoryType.USER, tags=["a", "b"]),
            "Content",
        )
        temp_file_store.save(
            MemoryFrontmatter(name="single", description="S", type=MemoryType.USER, tags=["c"]),
            "Content",
        )
        results = temp_file_store.search_by_tags(["a", "c"])
        assert len(results) >= 2  # Either tag matches

    def test_no_match_returns_empty(self, temp_file_store):
        assert temp_file_store.search_by_tags(["nonexistent_tag"]) == []


class TestSearchByImportance:
    def test_min_importance_filter(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="high", description="H", type=MemoryType.USER, importance=0.9),
            "High importance",
        )
        temp_file_store.save(
            MemoryFrontmatter(name="low", description="L", type=MemoryType.USER, importance=0.3),
            "Low importance",
        )
        results = temp_file_store.search_by_importance(min_importance=0.7)
        assert len(results) == 1
        assert results[0]["frontmatter"].name == "high"

    def test_sorted_by_importance(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="mid", description="M", type=MemoryType.USER, importance=0.5),
            "Mid",
        )
        temp_file_store.save(
            MemoryFrontmatter(name="top", description="T", type=MemoryType.USER, importance=0.9),
            "Top",
        )
        results = temp_file_store.search_by_importance(min_importance=0.0)
        assert results[0]["frontmatter"].importance >= results[-1]["frontmatter"].importance


class TestFullTextSearch:
    def test_exact_substring_match(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="arch", description="Architecture decision", type=MemoryType.PROJECT),
            "项目采用微服务架构设计。",
        )
        results = temp_file_store.full_text_search("架构", MemoryType.PROJECT)
        assert len(results) > 0

    def test_chinese_bigram_match(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="chinese_test", description="中文测试", type=MemoryType.PROJECT),
            "决定使用微服务加事件驱动架构。",
        )
        # "微服务"的 bigram 包括 "微服" 和 "服务"
        results = temp_file_store.full_text_search("之前提到的微服务方案", MemoryType.PROJECT)
        assert len(results) > 0, "Bigram matching should find '微服务'"

    def test_case_insensitive(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="case_test", description="Case", type=MemoryType.USER),
            "Using React for frontend.",
        )
        results = temp_file_store.full_text_search("REACT", MemoryType.USER)
        assert len(results) > 0

    def test_no_match_returns_empty(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="only", description="O", type=MemoryType.USER),
            "Some content",
        )
        results = temp_file_store.full_text_search("量子计算机", MemoryType.USER)
        assert len(results) == 0

    def test_searches_name_and_description(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="kafka_config", description="消息队列Kafka配置", type=MemoryType.PROJECT),
            "Some content about Kafka setup.",
        )
        results = temp_file_store.full_text_search("Kafka", MemoryType.PROJECT)
        assert len(results) > 0


class TestStats:
    def test_counts_accurate(self, temp_file_store):
        for i in range(3):
            temp_file_store.save(
                MemoryFrontmatter(name=f"u{i}", description=f"U{i}", type=MemoryType.USER),
                "content",
            )
        for i in range(2):
            temp_file_store.save(
                MemoryFrontmatter(name=f"p{i}", description=f"P{i}", type=MemoryType.PROJECT),
                "content",
            )
        counts = temp_file_store.count_by_type()
        assert counts["user"] == 3
        assert counts["project"] == 2

    def test_total_accurate(self, temp_file_store):
        for i in range(3):
            temp_file_store.save(
                MemoryFrontmatter(name=f"t{i}", description=f"T{i}", type=MemoryType.USER),
                "content",
            )
        assert temp_file_store.total_count() == 3

    def test_get_stats_structured(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="s", description="S", type=MemoryType.USER),
            "content",
        )
        stats = temp_file_store.get_stats()
        assert "total_memories" in stats
        assert "by_type" in stats
        assert stats["total_memories"] == 1


class TestIndexEntries:
    def test_parses_index_correctly(self, temp_file_store):
        temp_file_store.save(
            MemoryFrontmatter(name="parsed", description="Parsed entry for testing", type=MemoryType.USER),
            "Content here.",
        )
        index = temp_file_store.get_index_entries()
        assert "user" in index
        entries = index["user"]
        parsed_entry = next((e for e in entries if e["name"] == "parsed"), None)
        assert parsed_entry is not None
        assert "description" in parsed_entry

    def test_empty_index(self, temp_file_store):
        """新 FileStore 没有 MEMORY.md，返回空。"""
        index = temp_file_store.get_index_entries()
        assert index == {}
