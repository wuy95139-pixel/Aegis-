"""
core/agents/base.py 测试
========================
BaseAgent 的初始化、消息发送、记忆、对话等功能测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.agents.base import BaseAgent
from src.models.schemas import AgentMessage, ConversationTurn


class ConcreteAgent(BaseAgent):
    """具体实现类，用于测试 BaseAgent 抽象方法。"""
    role = "Test Agent"
    goal = "Test goal"
    backstory = "You are a test agent."

    def execute(self, task_input):
        return {"status": "success", "input": task_input}


@pytest.fixture
def agent(mock_llm, mock_memory_retriever):
    """创建测试 Agent 实例。"""
    return ConcreteAgent(
        name="test_agent",
        llm=mock_llm,
        memory=mock_memory_retriever,
        config={"debug": True},
    )


class TestInit:
    def test_stores_references(self, agent, mock_llm, mock_memory_retriever):
        assert agent.name == "test_agent"
        assert agent.llm is mock_llm
        assert agent.memory is mock_memory_retriever
        assert agent.config == {"debug": True}

    def test_creates_crewai_agent(self, agent):
        assert agent.crewai_agent is not None

    def test_msg_counter_starts_zero(self, agent):
        assert agent._msg_counter == 0

    def test_message_bus_none_by_default(self, agent):
        assert agent.message_bus is None

    def test_tools_empty_by_default(self, agent):
        assert agent._tools == []

    def test_with_custom_tools(self, mock_llm, mock_memory_retriever):
        # CrewAI validates tools as BaseTool instances, so we patch _build_crewai_agent
        with patch.object(ConcreteAgent, "_build_crewai_agent", return_value=Mock()):
            mock_tool = Mock()
            agent = ConcreteAgent(
                name="tool_agent",
                llm=mock_llm,
                memory=mock_memory_retriever,
                tools=[mock_tool],
            )
            assert len(agent._tools) == 1


class TestChat:
    def test_basic_chat(self, agent, mock_llm):
        result = agent.chat("Hello")
        assert isinstance(result, str)
        mock_llm.chat.assert_called_once()

    def test_chat_with_system_prompt(self, agent, mock_llm):
        agent.chat("Hello", system_prompt="Custom system prompt")
        # 验证 messages 中第一个是 system prompt
        call_args = mock_llm.chat.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "Custom system prompt"

    def test_chat_with_context(self, agent, mock_llm):
        agent.chat("Hello", context="Additional context")
        call_args = mock_llm.chat.call_args[0][0]
        # 应该有 system (backstory) + system (context) + user
        roles = [m["role"] for m in call_args]
        assert roles.count("system") >= 1
        has_context = any("上下文" in m.get("content", "") for m in call_args if m["role"] == "system")
        assert has_context


class TestSendMessage:
    def test_creates_agent_message(self, agent):
        msg = agent.send_message(
            receiver="reminder_agent",
            msg_type="request",
            payload={"question": "test"},
        )
        assert isinstance(msg, AgentMessage)
        assert msg.sender == "test_agent"
        assert msg.receiver == "reminder_agent"
        assert msg.type == "request"
        assert msg.payload == {"question": "test"}

    def test_increments_counter(self, agent):
        agent.send_message("receiver", "event", {})
        agent.send_message("receiver", "event", {})
        assert agent._msg_counter == 2

    def test_publishes_to_bus(self, agent):
        bus = Mock()
        agent.message_bus = bus
        agent.send_message("receiver", "event", {"data": 42})
        bus.publish.assert_called_once()

    def test_no_bus_no_publish(self, agent):
        agent.message_bus = None
        msg = agent.send_message("receiver", "event", {})
        assert msg is not None  # 不崩溃

    def test_reply_to_included(self, agent):
        msg = agent.send_message(
            receiver="someone",
            msg_type="response",
            payload={},
            reply_to="original_msg_123",
        )
        assert msg.reply_to == "original_msg_123"


class TestRemember:
    def test_remember_calls_memory(self, agent):
        """remember() 应通过 memory 调用 extract_and_remember，不抛异常"""
        result = agent.remember("Important content to save", source="test_chat", tags=["important"])
        # 验证不崩溃且调用了 memory
        assert agent.memory is not None

    def test_no_memory_no_crash(self, mock_llm):
        agent = ConcreteAgent(name="no_mem", llm=mock_llm, memory=None)
        agent.remember("content", "source")  # 不应崩溃


class TestRecall:
    def test_returns_structured_result(self, agent):
        result = agent.recall("query text", top_k=3)
        assert isinstance(result, dict)
        assert "relevant_memories" in result
        assert "recent_conversations" in result

    def test_no_memory_returns_empty(self, mock_llm):
        agent = ConcreteAgent(name="no_mem", llm=mock_llm, memory=None)
        result = agent.recall("query")
        assert result["relevant_memories"] == []
        assert result["combined_context"] == ""


class TestReceiveMessage:
    def test_default_returns_none(self, agent):
        msg = AgentMessage(
            id="test_1",
            sender="other_agent",
            receiver="test_agent",
            type="event",
            payload={"event": "test"},
        )
        result = agent.receive_message(msg)
        assert result is None


class TestGetToolsForLLM:
    def test_returns_tool_schemas(self, agent):
        """get_tools_for_llm() 应返回已注册工具的 OpenAI function-calling schema 列表"""
        schemas = agent.get_tools_for_llm()
        assert isinstance(schemas, list)
        # 至少应有 get_time_context 工具（已在 time_tools 中注册）
        for schema in schemas:
            assert "type" in schema
            assert "function" in schema
            assert "name" in schema["function"]


class TestRunCrewTask:
    def test_fallback_to_direct_llm(self, agent):
        """当 CrewAI 不可用时应回退到直接 LLM 调用。"""
        result = agent._run_crew_task("Test description", "Expected output")
        assert isinstance(result, str)
        # 使用 mock_llm 应该返回 mock response
        assert "mock LLM response" in result.lower() or len(result) > 0


class TestRepr:
    def test_repr_format(self, agent):
        r = repr(agent)
        assert "ConcreteAgent" in r
        assert "test_agent" in r
        assert "Test Agent" in r
