"""
学习层测试
=========
preference.py (PreferenceLearner), experience.py (ExperienceLibrary), adaptive.py (AdaptiveEngine) 测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.memory.preference import (
    SignalType, PreferenceSignal, PreferenceLearner,
)
from src.core.memory.experience import (
    Outcome, ExperienceCase, ExperienceLibrary,
)
from src.core.memory.adaptive import (
    GuidanceLevel, AdaptiveGuidance, AdaptiveEngine,
)


# ==================== PreferenceSignal Tests ====================

class TestPreferenceSignal:
    def test_reinforce_increases_confidence(self):
        sig = PreferenceSignal(preference="用户喜欢简洁回答", category="communication")
        original = sig.confidence
        sig.reinforce(context="用户说好", evidence="用户说好")
        assert sig.confidence > original

    def test_confidence_capped_at_095(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.94)
        sig.reinforce(evidence="test")
        assert sig.confidence <= 0.95

    def test_weaken_halves_confidence(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.8)
        sig.weaken(reason="用户说不对")
        assert sig.confidence == pytest.approx(0.4)

    def test_weaken_minimum_005(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.01)
        sig.weaken(reason="test")
        assert sig.confidence >= 0.05

    def test_is_active_default_threshold(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.5)
        assert sig.is_active() is True
        sig.confidence = 0.3
        assert sig.is_active() is False

    def test_is_active_custom_threshold(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.3)
        assert sig.is_active(threshold=0.2) is True

    def test_is_strong(self):
        sig = PreferenceSignal(preference="test", category="test", confidence=0.8)
        # occurrences starts at 1, is_strong requires >= 2
        sig.reinforce(evidence="second observation")
        assert sig.occurrences >= 2
        assert sig.confidence >= 0.7
        assert sig.is_strong() is True


# ==================== PreferenceLearner Tests ====================

class TestPreferenceLearner:
    @pytest.fixture
    def learner(self):
        return PreferenceLearner()

    def test_detect_explicit_preference(self, learner):
        signals = learner._rule_based_detect(
            "我喜欢简洁的代码风格", context="test", source="test"
        )
        assert len(signals) > 0

    def test_detect_correction(self, learner):
        signals = learner._rule_based_detect(
            "不对，不要用那个框架", context="test", source="test"
        )
        assert len(signals) > 0

    def test_observe_interaction(self, learner):
        result = learner.observe_interaction(
            user_message="我喜欢简单的回答",
            assistant_response="好的",
            user_reaction="对，就这样",
        )
        assert isinstance(result, dict)
        assert "signals_detected" in result

    def test_get_active_preferences_min_confidence(self, learner):
        # With no LLM, _rule_based_detect from observe_interaction creates signals
        learner.observe_interaction(
            user_message="我喜欢简洁的代码风格",
            assistant_response="ok",
            user_reaction="对，就这样",
        )
        active = learner.get_active_preferences(min_confidence=0.3)
        assert isinstance(active, list)

    def test_get_preference_prompt(self, learner):
        prompt = learner.get_preference_prompt()
        assert isinstance(prompt, str)


# ==================== ExperienceCase Tests ====================

class TestExperienceCase:
    def test_derive_lesson_success(self):
        case = ExperienceCase(
            situation="测试",
            approach="Mock LLM",
            outcome=Outcome.SUCCESS,
            source="test",
        )
        lesson = case._derive_lesson()
        assert "有效" in lesson or "Mock LLM" in lesson

    def test_derive_lesson_failure(self):
        case = ExperienceCase(
            situation="测试失败",
            approach="直接调用",
            outcome=Outcome.FAILURE,
            source="test",
        )
        lesson = case._derive_lesson()
        assert "避免" in lesson or "直接调用" in lesson


# ==================== ExperienceLibrary Tests ====================

class TestExperienceLibrary:
    @pytest.fixture
    def library(self, temp_file_store, mock_long_term):
        return ExperienceLibrary(
            file_store=temp_file_store,
            long_term_memory=mock_long_term,
        )

    def test_record(self, library):
        case = library.record(
            situation="测试用例失败",
            approach="Mock了LLM并重试",
            outcome=Outcome.SUCCESS,
            source="test_session",
            auto_extract_lesson=False,
        )
        assert case is not None
        assert len(case.case_id) > 0

    def test_record_with_lesson_extraction(self, library, mock_llm):
        library.llm = mock_llm
        mock_llm.chat.return_value = {
            "content": "应该总是 mock 外部服务",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        case = library.record(
            situation="测试",
            approach="Mocked",
            outcome=Outcome.SUCCESS,
            source="test",
            auto_extract_lesson=True,
        )
        assert case is not None

    def test_get_successful_approaches(self, library):
        library.record(situation="S1", approach="A", outcome=Outcome.SUCCESS, source="test",
                       auto_extract_lesson=False)
        library.record(situation="S2", approach="B", outcome=Outcome.FAILURE, source="test",
                       auto_extract_lesson=False)
        successes = library.get_successful_approaches("S1", top_k=5)
        assert isinstance(successes, list)

    def test_get_failed_approaches(self, library):
        library.record(situation="S1", approach="A", outcome=Outcome.FAILURE, source="test",
                       auto_extract_lesson=False)
        failures = library.get_failed_approaches("S1", top_k=5)
        assert isinstance(failures, list)

    def test_get_statistics(self, library):
        library.record(situation="S1", approach="A", outcome=Outcome.SUCCESS, source="test",
                       auto_extract_lesson=False)
        library.record(situation="S2", approach="B", outcome=Outcome.FAILURE, source="test",
                       auto_extract_lesson=False)
        stats = library.get_statistics()
        assert stats["total"] == 2


# ==================== AdaptiveEngine Tests ====================

class TestAdaptiveEngine:
    @pytest.fixture
    def engine(self, temp_file_store, mock_long_term, mock_llm):
        exp_lib = ExperienceLibrary(
            file_store=temp_file_store,
            long_term_memory=mock_long_term,
        )
        pref_learner = PreferenceLearner()
        return AdaptiveEngine(
            experience_library=exp_lib,
            preference_learner=pref_learner,
            llm=mock_llm,
        )

    def test_before_action_returns_guidance(self, engine):
        guidance = engine.before_action("实现一个新功能", context={})
        assert isinstance(guidance, AdaptiveGuidance)

    def test_after_action_records(self, engine):
        result = engine.after_action(
            situation="实现了功能",
            approach="用接口抽象",
            outcome=Outcome.SUCCESS,
            user_feedback="很好，这样可以",
        )
        assert isinstance(result, dict)

    def test_learn_from_interaction(self, engine):
        result = engine.learn_from_interaction(
            user_message="实现功能",
            assistant_response="用了设计模式",
            user_reaction="好",
        )
        assert isinstance(result, dict)

    def test_positive_feedback_detection(self, engine):
        assert engine._is_positive_feedback("好，就这样") is True
        assert engine._is_positive_feedback("谢谢") is True
        assert engine._is_positive_feedback("不行，重做") is False

    def test_negative_feedback_detection(self, engine):
        assert engine._is_negative_feedback("不对，不能这样") is True
        assert engine._is_negative_feedback("太长了") is True
        assert engine._is_negative_feedback("好的") is False


# ==================== AdaptiveGuidance Tests ====================

class TestAdaptiveGuidance:
    def test_to_text_formats(self):
        guidance = AdaptiveGuidance()
        guidance.add_should("使用接口隔离", "便于测试", "test")
        guidance.add_avoid("避免直接依赖", "耦合太高", "test")
        guidance.add_may("用户喜欢简洁代码", "用户偏好", "test")
        text = guidance.to_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_dict_serializable(self):
        guidance = AdaptiveGuidance()
        guidance.add_should("R1", "reason", "test")
        guidance.add_avoid("W1", "reason", "test")
        d = guidance.to_dict()
        assert isinstance(d, dict)
        assert "items" in d

    def test_is_empty(self):
        empty = AdaptiveGuidance()
        assert empty.is_empty() is True
        full = AdaptiveGuidance()
        full.add_should("Something", "because", "test")
        assert full.is_empty() is False
