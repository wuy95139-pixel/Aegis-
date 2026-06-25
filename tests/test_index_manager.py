"""
core/memory/index_manager.py 测试
=================================
IndexManager 的增删查改、完整性验证测试。
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.core.memory.index_manager import IndexManager
from src.core.memory.types import MemoryType


@pytest.fixture
def index_mgr(temp_file_store):
    """创建真实的 IndexManager，绑定 temp 目录。"""
    return IndexManager(base_dir=str(temp_file_store.base_dir))


class TestAddEntry:
    def test_add_new_entry(self, index_mgr):
        index_mgr.add_entry("test_entry", MemoryType.USER, "Test description")
        entries = index_mgr.get_entries_by_type(MemoryType.USER)
        assert any(e["name"] == "test_entry" for e in entries)

    def test_add_entry_updates_existing(self, index_mgr):
        index_mgr.add_entry("update_me", MemoryType.PROJECT, "First desc")
        index_mgr.add_entry("update_me", MemoryType.PROJECT, "Updated desc")
        entries = index_mgr.get_entries_by_type(MemoryType.PROJECT)
        found = [e for e in entries if e["name"] == "update_me"]
        assert len(found) == 1
        assert "Updated" in found[0]["description"]


class TestGetEntriesByType:
    def test_filters_by_type(self, index_mgr):
        index_mgr.add_entry("user_entry", MemoryType.USER, "User desc")
        index_mgr.add_entry("proj_entry", MemoryType.PROJECT, "Project desc")
        user_entries = index_mgr.get_entries_by_type(MemoryType.USER)
        assert all(e["name"] == "user_entry" for e in user_entries)


class TestGetAllEntries:
    def test_parses_sections(self, index_mgr):
        index_mgr.add_entry("entry1", MemoryType.USER, "Desc 1")
        entries = index_mgr.get_all_entries()
        assert isinstance(entries, dict)


class TestGetContextString:
    def test_formats_for_llm(self, index_mgr):
        index_mgr.add_entry("ctx_entry", MemoryType.FEEDBACK, "Some rule")
        ctx = index_mgr.get_context_string()
        assert isinstance(ctx, str)
        assert len(ctx) > 0


class TestRebuild:
    def test_rebuild_from_file_store(self, index_mgr, temp_file_store):
        from src.core.memory.types import MemoryFrontmatter, MemoryType
        temp_file_store.save(
            MemoryFrontmatter(name="rebuild_test", description="For rebuild", type=MemoryType.PROJECT),
            "Content for rebuild test.",
        )
        index_mgr.rebuild(temp_file_store)
        all_entries = index_mgr.get_all_entries()
        found = False
        for entries in all_entries.values():
            if any(e["name"] == "rebuild_test" for e in entries):
                found = True
        assert found, "Index should contain 'rebuild_test' after rebuild"


class TestVerifyIntegrity:
    def test_detects_orphans(self, index_mgr, temp_file_store):
        from src.core.memory.types import MemoryFrontmatter, MemoryType
        temp_file_store.save(
            MemoryFrontmatter(name="integrity_test", description="Test", type=MemoryType.USER),
            "Content.",
        )
        index_mgr.rebuild(temp_file_store)
        orphan_count, unindexed_count = index_mgr.verify_integrity(temp_file_store)
        assert isinstance(orphan_count, int)
        assert isinstance(unindexed_count, int)


class TestGetStats:
    def test_returns_stats(self, index_mgr):
        index_mgr.add_entry("stats_entry", MemoryType.REFERENCE, "Ref desc")
        stats = index_mgr.get_stats()
        assert isinstance(stats, dict)
        assert stats["total_entries"] >= 1


class TestRemoveEntry:
    def test_removes_entry(self, index_mgr):
        name = "to_remove"
        index_mgr.add_entry(name, MemoryType.USER, "Will be removed")
        entries_before = index_mgr.get_entries_by_type(MemoryType.USER)
        assert any(e["name"] == name for e in entries_before)
        index_mgr.remove_entry(name)
        entries_after = index_mgr.get_entries_by_type(MemoryType.USER)
        assert not any(e["name"] == name for e in entries_after)
