"""
core/memory/memory_manager.py 测试
==================================
MemoryManager 的单例、写入、读取、经验、偏好、自适应、维护操作测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import threading

from src.core.memory.memory_manager import MemoryManager
from src.core.memory.types import MemoryType, MemoryFrontmatter
from src.core.memory.experience import Outcome
from src.core.memory.adaptive import AdaptiveGuidance
from src.core.memory.short_term import ShortTermMemory
from src.core.memory.retriever import MemoryRetriever
from src.models.schemas import MemoryEntry, ConversationTurn


# 每个测试前重置 MemoryManager 单例
@pytest.fixture(autouse=True)
def reset_memory_manager():
    MemoryManager._instance = None
    yield
    MemoryManager._instance = None


# ==================== Singleton & Init Tests ====================

class TestSingleton:
    def test_returns_same_instance(self, mock_llm):
        mm1 = MemoryManager(llm=mock_llm)
        mm2 = MemoryManager()
        assert mm1 is mm2

    def test_init_only_runs_once(self, mock_llm):
        mm1 = MemoryManager(llm=mock_llm, config={"debug": True})
        mm2 = MemoryManager(llm=Mock(), config={"debug": False})
        assert mm2.config == {"debug": True}
        assert mm2.llm is mock_llm

    def test_thread_safe_singleton(self):
        instances = []

        def create():
            mm = MemoryManager()
            instances.append(mm)

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        assert all(mm is first for mm in instances)

    def test_subsystems_none_before_initialize(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        assert mm.short_term is None
        assert mm.long_term is None
        assert mm.file_store is None
        assert mm.retriever is None
        assert mm.experience_lib is None
        assert mm.preference_learner is None
        assert mm.adaptive_engine is None


class TestRepr:
    def test_repr_format(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        r = repr(mm)
        assert "MemoryManager" in r


# ==================== Remember Tests ====================

class TestRemember:
    @pytest.fixture
    def mm(self, mock_llm, temp_file_store, mock_long_term):
        mm = MemoryManager(llm=mock_llm)
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        mock_long_term.count.return_value = 0
        mock_long_term.store.return_value = "mock_vector_id"
        return mm

    def test_dual_write_file_and_vector(self, mm, mock_long_term):
        result = mm.remember(
            content="用户喜欢用 Go 语言开发后端服务",
            memory_type=MemoryType.USER,
            source="test",
            extract_key_points=False,
        )
        assert result["status"] == "success"
        assert result["file_id"] is not None
        assert result["vector_id"] is not None
        mock_long_term.store.assert_called_once()

    def test_skips_not_worth_remembering(self, mm):
        result = mm.remember(
            content="git status",
            memory_type=MemoryType.PROJECT,
            extract_key_points=False,
        )
        assert result["status"] == "skipped"

    def test_not_initialized_error(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.remember("content", MemoryType.USER)
        assert result["status"] == "error"

    def test_extracts_key_points(self, mm, mock_llm):
        mock_llm.chat.return_value = {
            "content": "- 要点1\n- 要点2\n- 要点3",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {},
        }
        result = mm.remember(
            content="这是一段需要提取关键信息的较长文本，" * 10,
            memory_type=MemoryType.PROJECT,
            extract_key_points=True,
        )
        assert len(result["key_points"]) >= 1

    def test_file_store_error_graceful(self, mm, mock_long_term):
        """FileStore 写入失败时仍尝试 ChromaDB。"""
        broken_fs = Mock()
        broken_fs.save.side_effect = IOError("disk full")
        mm.file_store = broken_fs
        result = mm.remember(
            content="Important memory content",
            memory_type=MemoryType.USER,
            extract_key_points=False,
        )
        assert "file_error" in result

    def test_chromadb_error_non_fatal(self, mm, mock_long_term):
        """ChromaDB 写入失败时 FileStore 写入仍成功。"""
        mock_long_term.store.side_effect = Exception("ChromaDB down")
        result = mm.remember(
            content="Important content here",
            memory_type=MemoryType.PROJECT,
            extract_key_points=False,
        )
        assert result["file_id"] is not None


# ==================== Conversation Tests ====================

class TestRememberConversation:
    def test_no_llm_returns_error(self, temp_file_store, mock_long_term):
        mm = MemoryManager()
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        turns = [ConversationTurn(role="user", content="Hello")]
        result = mm.remember_conversation(turns, source="test")
        assert result["status"] == "error"


# ==================== Recall Tests ====================

class TestRecall:
    def test_no_retriever_returns_empty(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.recall("query")
        assert result["relevant_memories"] == []
        assert result["combined_context"] == ""


# ==================== Profile / Rules / Project / References Tests ====================

class TestGetUserProfile:
    def test_returns_dict(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_user_profile()
        assert isinstance(result, dict)


class TestGetFeedbackRules:
    def test_returns_list(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_feedback_rules()
        assert isinstance(result, list)


class TestGetProjectContext:
    def test_returns_dict(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_project_context()
        assert isinstance(result, dict)


class TestGetReferences:
    def test_returns_dict(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_references()
        assert isinstance(result, dict)

    def test_filters_by_system(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_references(system="github")
        assert isinstance(result, dict)


# ==================== Experience Operations Tests ====================

class TestRecordExperience:
    def test_not_initialized_returns_error(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.record_experience("situation", "approach")
        assert result["status"] == "error"

    def test_record_success_shortcut(self, temp_file_store, mock_long_term, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        from src.core.memory.experience import ExperienceLibrary
        mm.experience_lib = ExperienceLibrary(
            file_store=temp_file_store,
            long_term_memory=mock_long_term,
        )
        mock_long_term.store.return_value = "exp_id"
        result = mm.record_success("Built feature", "Used TDD", "Tests first works")
        assert result["status"] == "success"
        assert "case_id" in result

    def test_record_failure_shortcut(self, temp_file_store, mock_long_term, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        from src.core.memory.experience import ExperienceLibrary
        mm.experience_lib = ExperienceLibrary(
            file_store=temp_file_store,
            long_term_memory=mock_long_term,
        )
        mock_long_term.store.return_value = "exp_id"
        result = mm.record_failure("Failed deploy", "No rollback plan", "Always have rollback")
        assert result["status"] == "success"

    def test_get_relevant_experiences(self, temp_file_store, mock_long_term, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        from src.core.memory.experience import ExperienceLibrary
        mm.experience_lib = ExperienceLibrary(
            file_store=temp_file_store,
            long_term_memory=mock_long_term,
        )
        result = mm.get_relevant_experiences("test situation")
        assert "experiences" in result
        assert "guidance_text" in result


# ==================== Preference Operations Tests ====================

class TestLearnPreferences:
    def test_not_initialized_returns_error(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.learn_preferences("user message", "response", "reaction")
        assert result["status"] == "error"

    def test_learns_from_interaction(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        from src.core.memory.preference import PreferenceLearner
        mm.preference_learner = PreferenceLearner()
        result = mm.learn_preferences(
            user_message="我喜欢简洁的代码",
            assistant_response="好的",
            user_reaction="对",
        )
        assert result["status"] == "success"

    def test_detect_preferences_returns_list(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.detect_preferences("some message")
        assert isinstance(result, list)

    def test_get_active_preferences_returns_list(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_active_preferences()
        assert isinstance(result, list)

    def test_get_preference_prompt_returns_str(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_preference_prompt()
        assert isinstance(result, str)


# ==================== Adaptive Behavior Tests ====================

class TestBeforeAction:
    def test_not_initialized_returns_empty(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.before_action("test situation")
        assert result["is_empty"] is True

    def test_with_engine_returns_guidance(self, temp_file_store, mock_long_term, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        from src.core.memory.experience import ExperienceLibrary
        from src.core.memory.preference import PreferenceLearner
        from src.core.memory.adaptive import AdaptiveEngine
        exp_lib = ExperienceLibrary(file_store=temp_file_store, long_term_memory=mock_long_term)
        pref = PreferenceLearner()
        mm.adaptive_engine = AdaptiveEngine(
            experience_library=exp_lib,
            preference_learner=pref,
            memory_manager=mm,
            llm=mock_llm,
        )
        result = mm.before_action("实现一个新功能", "code_generate")
        assert "guidance_text" in result
        assert isinstance(result["is_empty"], bool)


class TestAfterAction:
    def test_not_initialized_returns_error(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.after_action("situation", "approach", "success")
        assert result["status"] == "error"


class TestLearnFromInteraction:
    def test_not_initialized_returns_error(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.learn_from_interaction("user msg", "response")
        assert result["status"] == "error"


class TestGetBehavioralPrompt:
    def test_not_initialized_returns_empty(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.get_behavioral_prompt()
        assert result == ""


# ==================== Forget Tests ====================

class TestForget:
    def test_forget_removes_from_all_stores(self, mock_llm, temp_file_store, mock_long_term):
        mm = MemoryManager(llm=mock_llm)
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        from src.core.memory.types import MemoryType as MT
        # 先保存一条记忆
        temp_file_store.save(
            MemoryFrontmatter(name="delete_me", description="To be deleted", type=MT.USER),
            "Content to delete.",
        )
        result = mm.forget("delete_me", MT.USER)
        assert result is True
        # 验证文件已删除
        assert temp_file_store.get("delete_me", MT.USER) is None

    def test_forget_nonexistent(self, mock_llm, temp_file_store, mock_long_term):
        mm = MemoryManager(llm=mock_llm)
        mm.file_store = temp_file_store
        mm.long_term = mock_long_term
        result = mm.forget("does_not_exist", MemoryType.PROJECT)
        assert result is False


# ==================== Update Memory Tests ====================

class TestUpdateMemory:
    def test_no_file_store_returns_false(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.update_memory("test", MemoryType.USER, content="new")
        assert result is False


# ==================== Rebuild Index Tests ====================

class TestRebuildIndex:
    def test_no_file_store_returns_false(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.rebuild_index()
        assert result is False


# ==================== Full Context Tests ====================

class TestGetFullContext:
    def test_returns_string(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        ctx = mm.get_full_context("test query")
        assert isinstance(ctx, str)

    def test_selective_includes(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        ctx = mm.get_full_context(
            "test query",
            include_profile=False,
            include_rules=False,
            include_project=False,
            include_experiences=False,
            include_preferences=False,
            include_behavioral=False,
        )
        assert isinstance(ctx, str)


class TestGetLLMSystemContext:
    def test_with_extra_prompt(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        ctx = mm.get_llm_system_context(base_system_prompt="You are a helpful assistant.")
        assert "You are a helpful assistant." in ctx

    def test_no_base_prompt(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        ctx = mm.get_llm_system_context()
        assert isinstance(ctx, str)


# ==================== Conversation Management Tests ====================

class TestAddConversationTurn:
    def test_adds_turn_to_short_term(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.short_term = ShortTermMemory(max_tokens=500, window_size=6)
        mm.add_conversation_turn("user", "Hello")
        turns = mm.get_conversation_context()
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello"

    def test_no_short_term_no_crash(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.add_conversation_turn("user", "Hello")  # 不应崩溃

    def test_get_conversation_messages_llm_format(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.short_term = ShortTermMemory(max_tokens=500, window_size=6)
        mm.add_conversation_turn("user", "Hi there")
        messages = mm.get_conversation_messages(system_prompt="Be helpful")
        assert isinstance(messages, list)
        # 第一个应该是 system prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"


class TestSummarizeConversation:
    def test_too_few_turns_returns_empty(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        mm.short_term = ShortTermMemory(max_tokens=500, window_size=6)
        result = mm.summarize_conversation()
        assert result == ""

    def test_no_short_term_returns_empty(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        result = mm.summarize_conversation()
        assert result == ""

    def test_no_llm_returns_empty(self):
        mm = MemoryManager()
        mm.short_term = ShortTermMemory(max_tokens=500, window_size=6)
        mm.add_conversation_turn("user", "Message 1")
        mm.add_conversation_turn("assistant", "Reply 1")
        mm.add_conversation_turn("user", "Message 2")
        mm.add_conversation_turn("assistant", "Reply 2")
        result = mm.summarize_conversation()
        assert result == ""


# ==================== Stats Tests ====================

class TestGetStats:
    def test_returns_all_section_keys(self, mock_llm):
        mm = MemoryManager(llm=mock_llm)
        stats = mm.get_stats()
        assert "short_term" in stats
        assert "long_term" in stats
        assert "file_store" in stats
        assert "index" in stats
        assert "experiences" in stats
        assert "preferences" in stats
        assert "adaptive" in stats
