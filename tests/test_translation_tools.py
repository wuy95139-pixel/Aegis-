"""
core/tools/translation_tools.py 测试
====================================
TranslationTool 的翻译、润色功能测试。
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.tools.translation_tools import TranslationTool
from src.models.schemas import TranslationResult


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.chat.return_value = {
        "content": "你好世界",
        "tool_calls": None,
        "usage": {},
        "finish_reason": "stop",
    }
    return llm


class TestTranslate:
    def test_basic_translation(self, mock_llm):
        tool = TranslationTool(mock_llm)
        result = tool.translate("Hello world", target_lang="zh-CN")
        assert isinstance(result, TranslationResult)
        assert result.translated_text == "你好世界"
        mock_llm.chat.assert_called_once()

    def test_with_glossary(self, mock_llm):
        tool = TranslationTool(mock_llm)
        glossary = {"AI": "人工智能"}
        result = tool.translate("AI is smart", glossary=glossary)
        # glossary 应出现在 prompt 中
        call_args = mock_llm.chat.call_args[1]["messages"][0]["content"]
        assert "术语表" in call_args
        assert "AI" in call_args

    def test_auto_source_language(self, mock_llm):
        tool = TranslationTool(mock_llm)
        result = tool.translate("Bonjour le monde")
        assert result.source_lang_detected is not None

    def test_error_graceful(self, mock_llm):
        mock_llm.chat.side_effect = Exception("LLM down")
        tool = TranslationTool(mock_llm)
        result = tool.translate("Hello")
        assert "翻译失败" in result.translated_text
        assert result.confidence == 0.0


class TestPolishText:
    def test_professional_style(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "Polished professional text.",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("raw text", style="professional")
        assert isinstance(result, str)
        assert "Polished" in result

    def test_casual_style(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "Hey, check this out!",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("Please examine the following", style="casual")
        assert "check this out" in result.lower() or "Hey" in result

    def test_academic_style(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "It is hereby demonstrated that...",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("This shows that...", style="academic")
        assert len(result) > 0

    def test_concise_style(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "Key point.",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("This is a very long text that goes on and on", style="concise")
        assert len(result) > 0

    def test_with_target_length(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "Short summary.",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("Long text here...", target_length="缩减到50字")
        assert isinstance(result, str)

    def test_long_input_truncation(self, mock_llm):
        mock_llm.chat.return_value = {
            "content": "Polished.",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        tool = TranslationTool(mock_llm)
        long_text = "x" * 10000  # 超过 8000 字符限制
        result = tool.polish_text(long_text)
        assert "原文共 10000 字符" in result or "Polished" in result

    def test_error_graceful(self, mock_llm):
        mock_llm.chat.side_effect = Exception("LLM down")
        tool = TranslationTool(mock_llm)
        result = tool.polish_text("text to polish")
        assert "润色失败" in result
