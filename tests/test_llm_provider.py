"""
core/llm/provider.py 测试
=========================
LLMProvider 的初始化、chat、响应解析、embed、stream 测试。
Mock OpenAI client 避免真实 API 调用。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.llm.provider import LLMProvider, create_llm_provider


@pytest.fixture
def mock_openai_response():
    """标准的 mock OpenAI chat completion 响应。"""
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = "Hello, how can I help?"
    message.tool_calls = None
    choice.message = message
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    return response


@pytest.fixture
def mock_client(mock_openai_response):
    """Mock OpenAI client。"""
    client = MagicMock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


@pytest.fixture
def llm_with_mocks(mock_client):
    """创建 LLMProvider，mock OpenAI/AsyncOpenAI。"""
    with patch("src.core.llm.provider.OpenAI", return_value=mock_client):
        with patch("src.core.llm.provider.AsyncOpenAI", return_value=MagicMock()):
            provider = LLMProvider(config={
                "api_key": "sk-test",
                "api_base": "https://test.api.com/v1",
                "model": "test-model",
            })
            return provider


class TestInit:
    def test_config_from_dict(self):
        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = LLMProvider(config={
                "api_key": "sk-args",
                "api_base": "https://args.api.com/v1",
                "model": "args-model",
            })
            assert provider.config["api_key"] == "sk-args"
            assert provider.config["api_base"] == "https://args.api.com/v1"
            assert provider.default_model == "args-model"

    def test_config_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("DEFAULT_MODEL", "env-model")
        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = LLMProvider()
            assert provider.default_model == "env-model"


class TestChat:
    def test_basic_chat(self, llm_with_mocks):
        result = llm_with_mocks.chat([
            {"role": "user", "content": "Hello"},
        ])
        assert result["content"] == "Hello, how can I help?"
        assert result["tool_calls"] is None
        assert "usage" in result

    def test_with_tools(self, llm_with_mocks):
        tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
        result = llm_with_mocks.chat(
            [{"role": "user", "content": "test"}],
            tools=tools,
        )
        assert result is not None

    def test_with_max_tokens(self, llm_with_mocks):
        result = llm_with_mocks.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        assert result is not None


class TestParseResponse:
    def test_normal_response(self):
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = "Normal response"
        message.tool_calls = None
        # 删除潜在的 reasoning_content
        del message.reasoning_content
        choice.message = message
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 3
        response.usage.total_tokens = 8

        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = LLMProvider(config={"api_key": "sk-test"})
            result = provider._parse_response(response)
            assert result["content"] == "Normal response"
            assert result["tool_calls"] is None
            assert result["finish_reason"] == "stop"

    def test_parse_tool_calls(self):
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = None
        del message.reasoning_content

        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function = MagicMock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"city": "Beijing"}'
        message.tool_calls = [tool_call]
        choice.message = message
        choice.finish_reason = "tool_calls"
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15

        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = LLMProvider(config={"api_key": "sk-test"})
            result = provider._parse_response(response)
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"] == "get_weather"

    def test_parse_deepseek_reasoning_content(self):
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = "Answer"
        message.tool_calls = None
        # 设置 reasoning_content
        reasoning_content = "Let me think step by step..."
        message.reasoning_content = reasoning_content
        choice.message = message
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10
        response.usage.total_tokens = 30

        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = LLMProvider(config={"api_key": "sk-test"})
            result = provider._parse_response(response)
            assert result["reasoning_content"] == reasoning_content


class TestEmbed:
    def test_embed_returns_vectors(self):
        mock_embed_response = MagicMock()
        mock_embed_data = MagicMock()
        mock_embed_data.embedding = [0.1, 0.2, 0.3]
        mock_embed_response.data = [mock_embed_data]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed_response

        with patch("src.core.llm.provider.OpenAI", return_value=mock_client):
            with patch("src.core.llm.provider.AsyncOpenAI"):
                provider = LLMProvider(config={"api_key": "sk-test"})
                result = provider.embed(["hello world"])
                assert len(result) == 1
                assert len(result[0]) == 3


class TestStreamChat:
    def test_yields_chunks(self):
        mock_client = MagicMock()
        chunk1 = MagicMock()
        c1 = MagicMock()
        c1.delta = MagicMock()
        c1.delta.content = "Hello"
        c1.finish_reason = None
        chunk1.choices = [c1]

        chunk2 = MagicMock()
        c2 = MagicMock()
        c2.delta = MagicMock()
        c2.delta.content = " World"
        c2.finish_reason = "stop"
        chunk2.choices = [c2]

        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        with patch("src.core.llm.provider.OpenAI", return_value=mock_client):
            with patch("src.core.llm.provider.AsyncOpenAI"):
                provider = LLMProvider(config={"api_key": "sk-test"})
                chunks = list(provider.stream_chat([{"role": "user", "content": "Hi"}]))
                assert "Hello" in chunks[0]
                assert "World" in chunks[1]


class TestCreateLLMProvider:
    def test_factory_returns_llm_provider(self):
        with patch("src.core.llm.provider.OpenAI"), patch("src.core.llm.provider.AsyncOpenAI"):
            provider = create_llm_provider(config={"api_key": "sk-factory"})
            assert isinstance(provider, LLMProvider)
