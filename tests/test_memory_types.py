"""
core/memory/types.py 测试
=========================
记忆类型定义、过滤函数测试。
"""

import pytest
from datetime import datetime

from src.core.memory.types import (
    MemoryType, MemoryFrontmatter,
    is_worth_remembering, get_type_rule,
    DO_NOT_SAVE, TYPE_RULES,
)


class TestMemoryFrontmatter:
    def test_creation_with_defaults(self):
        fm = MemoryFrontmatter(name="test", description="A test memory", type=MemoryType.USER)
        assert fm.name == "test"
        assert fm.importance == 0.5
        assert fm.tags == []
        assert fm.created_at is None

    def test_serialization_roundtrip(self):
        now = datetime.now()
        fm = MemoryFrontmatter(
            name="test_mem",
            description="Test description",
            type=MemoryType.PROJECT,
            tags=["important", "architecture"],
            importance=0.8,
            created_at=now,
            fact="Use microservices",
            why="Scalability",
            how_to_apply="Consider for new services",
            status="active",
        )
        # model_dump 再重新构造
        dumped = fm.model_dump()
        reloaded = MemoryFrontmatter(**dumped)
        assert reloaded.name == fm.name
        assert reloaded.type == fm.type
        assert reloaded.importance == fm.importance
        assert reloaded.fact == fm.fact

    def test_datetime_serialization(self):
        now = datetime.now()
        fm = MemoryFrontmatter(
            name="dated",
            description="With dates",
            type=MemoryType.USER,
            created_at=now,
            updated_at=now,
        )
        dumped = fm.model_dump()
        assert "created_at" in dumped
        assert "updated_at" in dumped

    def test_feedback_type_fields(self):
        fm = MemoryFrontmatter(
            name="feedback_1",
            description="User feedback",
            type=MemoryType.FEEDBACK,
            rule="Don't mock DB",
            why="Caused prod issue",
            how_to_apply="Tests must use real DB",
            severity="high",
        )
        assert fm.rule == "Don't mock DB"
        assert fm.severity == "high"

    def test_reference_type_fields(self):
        fm = MemoryFrontmatter(
            name="ref_1",
            description="External reference",
            type=MemoryType.REFERENCE,
            pointer="https://example.com",
            system="GitHub",
        )
        assert fm.pointer == "https://example.com"
        assert fm.system == "GitHub"

    def test_experience_type_fields(self):
        fm = MemoryFrontmatter(
            name="exp_1",
            description="Past experience",
            type=MemoryType.EXPERIENCE,
            situation="Migration failed",
            approach="Direct upgrade",
            outcome="failure",
            lesson="Test in staging first",
        )
        assert fm.outcome == "failure"
        assert fm.lesson == "Test in staging first"


class TestIsWorthRemembering:
    def test_short_content_rejected(self):
        assert is_worth_remembering("ab", MemoryType.USER) is False
        assert is_worth_remembering("", MemoryType.USER) is False

    def test_code_snippet_rejected(self):
        assert is_worth_remembering("def foo(): pass", MemoryType.USER) is False
        assert is_worth_remembering("import os", MemoryType.PROJECT) is False
        assert is_worth_remembering("class MyClass:", MemoryType.FEEDBACK) is False
        assert is_worth_remembering("const x = 1;", MemoryType.REFERENCE) is False
        assert is_worth_remembering("function() {}", MemoryType.USER) is False

    def test_git_command_rejected(self):
        assert is_worth_remembering("git status", MemoryType.PROJECT) is False
        assert is_worth_remembering("git log", MemoryType.PROJECT) is False
        assert is_worth_remembering("git diff HEAD", MemoryType.PROJECT) is False

    def test_valid_content_accepted(self):
        assert is_worth_remembering("用户是一名资深后端工程师", MemoryType.USER) is True
        assert is_worth_remembering("项目需要支持微服务架构", MemoryType.PROJECT) is True
        assert is_worth_remembering("不要 mock 数据库", MemoryType.FEEDBACK) is True

    def test_case_insensitive(self):
        assert is_worth_remembering("DEF foo(): pass", MemoryType.USER) is False


class TestGetTypeRule:
    def test_returns_rules_for_known_type(self):
        rule = get_type_rule(MemoryType.USER)
        assert "when_to_save" in rule
        assert "how_to_use" in rule
        assert "examples" in rule

    def test_unknown_type_returns_empty(self):
        # 测试不存在的枚举值的情况
        result = get_type_rule("nonexistent")
        assert result == {}


class TestConstants:
    def test_do_not_save_not_empty(self):
        assert len(DO_NOT_SAVE) > 0

    def test_all_types_have_rules(self):
        for mt in MemoryType:
            assert mt in TYPE_RULES, f"Missing rules for {mt}"
            assert "when_to_save" in TYPE_RULES[mt]
            assert "how_to_use" in TYPE_RULES[mt]
