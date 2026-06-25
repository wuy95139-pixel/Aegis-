"""
Agent 单元测试
=============
测试各 Agent 的核心功能。

运行:
    cd d:/Aegis
    python -m pytest tests/test_agents.py -v

注意: 运行测试需要在 .env 中配置 OPENAI_API_KEY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch

from src.core.llm.provider import LLMProvider
from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.retriever import MemoryRetriever
from src.core.agents.file_processor import FileProcessorAgent
from src.core.agents.task_dispatcher import TaskDispatcherAgent
from src.core.agents.memory_agent import MemoryAgent
from src.core.agents.reminder_agent import ReminderAgent
from src.core.agents.research_agent import ResearchAgent
from src.core.agents.orchestrator import Orchestrator, MessageBus
from src.models.schemas import *


def create_mock_llm():
    """创建 mock LLM Provider (无需真实 API Key)"""
    mock = Mock(spec=LLMProvider)
    mock.chat.return_value = {
        "content": "这是一个模拟的 LLM 回复。",
        "tool_calls": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "finish_reason": "stop",
    }
    return mock


def create_mock_memory():
    """创建 mock MemoryRetriever (无需真实 ChromaDB)"""
    st = ShortTermMemory(max_tokens=1000, window_size=10)
    lt = Mock(spec=LongTermMemory)
    lt.count.return_value = 0
    lt.search.return_value = []
    lt.store.return_value = "mock_id"
    return MemoryRetriever(short_term=st, long_term=lt)


# ===================== MessageBus 测试 =====================

class TestMessageBus:
    def test_publish_and_subscribe(self):
        bus = MessageBus()
        received = []

        bus.subscribe("agent_a", lambda msg: received.append(msg))

        msg = AgentMessage(
            id="test_1",
            sender="agent_b",
            receiver="agent_a",
            type="request",
            payload={"data": "hello"},
        )
        bus.publish(msg)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

    def test_broadcast(self):
        bus = MessageBus()
        received_a = []
        received_b = []

        bus.subscribe("agent_a", lambda msg: received_a.append(msg))
        bus.subscribe("agent_b", lambda msg: received_b.append(msg))

        msg = AgentMessage(
            id="test_2",
            sender="orchestrator",
            receiver="broadcast",
            type="event",
            payload={"event": "shutdown"},
        )
        bus.publish(msg)

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_get_history(self):
        bus = MessageBus()
        for i in range(5):
            bus.publish(AgentMessage(
                id=f"msg_{i}",
                sender="test",
                receiver="test",
                type="request",
                payload={"index": i},
            ))
        assert len(bus.get_history(3)) == 3


# ===================== Agent 测试 =====================

class TestFileProcessorAgent:
    """文件处理代理测试"""

    def test_agent_initialization(self):
        llm = create_mock_llm()
        agent = FileProcessorAgent(llm)
        assert agent.name == "file_processor"
        assert agent.role == "文件处理专家"
        assert agent.translation_tool is not None

    def test_parse_txt_file(self, tmp_path):
        """测试文本文件解析"""
        # 创建临时文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\n这是测试内容。", encoding="utf-8")

        llm = create_mock_llm()
        agent = FileProcessorAgent(llm)
        result = agent.execute({
            "filepath": str(test_file),
            "action": "parse",
        })

        assert result["status"] == "success"
        assert "Hello World" in result["result_text"]
        assert result["parsed_file"].file_type == FileType.TXT

    def test_extract_todos(self):
        """测试待办事项提取"""
        llm = create_mock_llm()
        llm.chat.return_value["content"] = '[{"title": "完成报告", "description": "Q2总结", "assignee": "张三", "deadline": "2024-06-15"}]'

        agent = FileProcessorAgent(llm)
        todos = agent._extract_todos("张三需要在6月15日前完成Q2总结报告")
        assert len(todos) == 1
        assert todos[0]["title"] == "完成报告"

    def test_unsupported_file(self, tmp_path):
        """测试不支持的文件格式"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")

        llm = create_mock_llm()
        agent = FileProcessorAgent(llm)
        result = agent.execute({"filepath": str(test_file)})
        assert result["status"] == "error"


class TestTaskDispatcherAgent:
    """任务分派代理测试"""

    def test_agent_initialization(self):
        llm = create_mock_llm()
        agent = TaskDispatcherAgent(llm)
        assert agent.name == "task_dispatcher"
        assert len(agent._person_pool) > 0

    def test_dispatch_by_keyword(self):
        llm = create_mock_llm()
        agent = TaskDispatcherAgent(llm)

        todos = [
            {"title": "前端页面优化", "description": "优化首页加载速度", "deadline": "2024-06-10"},
            {"title": "后端API开发", "description": "开发用户管理接口", "deadline": "2024-06-15"},
        ]

        result = agent.execute({"todos": todos, "source": "test", "auto_assign": True})

        assert result["status"] == "success"
        assert len(result["assigned_todos"]) == 2

        # 前端任务应分派给前端工程师
        frontend_todo = result["assigned_todos"][0]
        assert frontend_todo.assignee is not None
        assert "前端" in frontend_todo.assignee.role

    def test_dispatch_no_match(self):
        llm = create_mock_llm()
        agent = TaskDispatcherAgent(llm)

        todos = [{"title": "未知领域的任务", "description": "没有匹配的人员"}]

        result = agent.execute({"todos": todos, "source": "test", "auto_assign": True})

        # 无法匹配的任务应放入 unassigned
        assert len(result["assigned_todos"]) == 0
        assert len(result["unassigned"]) == 1


class TestMemoryAgent:
    """记忆代理测试"""

    def test_agent_initialization(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        agent = MemoryAgent(llm, memory)
        assert agent.name == "memory_agent"

    def test_store_operation(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        agent = MemoryAgent(llm, memory)

        result = agent.execute({
            "operation": "store",
            "content": "用户偏好使用中文交流",
            "source": "test_chat",
            "tags": ["preference"],
        })

        assert result["status"] == "success"

    def test_retrieve_operation(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        agent = MemoryAgent(llm, memory)

        result = agent.execute({
            "operation": "retrieve",
            "query": "用户偏好",
            "top_k": 3,
        })

        assert result["status"] == "success"
        assert "context" in result


class TestReminderAgent:
    """提醒代理测试"""

    def test_agent_initialization(self):
        llm = create_mock_llm()
        agent = ReminderAgent(llm)
        assert agent.name == "reminder_agent"
        assert agent.calendar is not None

    def test_set_and_check_reminder(self):
        llm = create_mock_llm()
        agent = ReminderAgent(llm)

        # 设置提醒 (过去的时间，确保立即触发)
        from datetime import datetime, timedelta
        past_time = datetime.now() - timedelta(minutes=5)

        set_result = agent.execute({
            "operation": "set",
            "title": "测试提醒",
            "trigger_time": past_time.isoformat(),
            "description": "这是一个测试",
        })
        assert set_result["status"] == "success"

        # 检查提醒
        check_result = agent.execute({"operation": "check"})
        assert check_result["status"] == "success"

    def test_list_reminders(self):
        llm = create_mock_llm()
        agent = ReminderAgent(llm)

        result = agent.execute({"operation": "list"})
        assert result["status"] == "success"
        assert "reminders" in result


class TestResearchAgent:
    """研究代理测试"""

    def test_agent_initialization(self):
        llm = create_mock_llm()
        agent = ResearchAgent(llm)
        assert agent.name == "research_agent"
        assert agent.search_engine is not None

    def test_research_with_mock_search(self):
        llm = create_mock_llm()
        llm.chat.return_value["content"] = """
## 执行摘要
AI技术的发展迅速，多智能体系统成为新趋势。

## 背景介绍
随着大语言模型的发展，AI智能体成为研究热点。

## 关键发现
- 发现1：多智能体协作提升效率
- 发现2：AutoGen和CrewAI是主流框架

## SWOT分析
- 优势：提升开发效率
- 劣势：学习成本高
- 机会：市场需求大
- 威胁：技术迭代快

## 建议
- 建议1：优先学习CrewAI
- 建议2：建立评测体系

## 来源
- https://example.com/1
"""

        agent = ResearchAgent(llm)
        result = agent.execute({
            "topic": "AI智能体发展趋势",
            "sources": ["web"],
            "max_results": 3,
            "include_swot": True,
        })

        assert result["status"] == "success"
        assert result["report"] is not None
        assert "AI" in result["report"].title
        assert len(result["report"].findings) > 0


# ===================== 集成测试 =====================

class TestOrchestratorIntegration:
    """Orchestrator 集成测试"""

    def test_orchestrator_initialization(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        orchestrator = Orchestrator(llm, memory)

        assert "file_processor" in orchestrator.agents
        assert "task_dispatcher" in orchestrator.agents
        assert "memory_agent" in orchestrator.agents
        assert "reminder_agent" in orchestrator.agents
        assert "research_agent" in orchestrator.agents

    def test_intent_classification(self):
        """测试关键词回退分类（mock LLM 不返回 JSON 时触发回退）"""
        llm = create_mock_llm()
        memory = create_mock_memory()
        orchestrator = Orchestrator(llm, memory)
        classifier = orchestrator.intent_classifier

        # 无文件：关键词回退
        assert classifier.classify("你好", None, "")["intent"] == "general_chat"
        assert classifier.classify("搜索AI最新新闻", None, "")["intent"] == "research"
        assert classifier.classify("提醒我明天开会", None, "")["intent"] == "reminder_set"
        assert classifier.classify("有哪些待办", None, "")["intent"] == "task_inquiry"

        # 有文件：关键词回退可区分文件操作
        assert classifier.classify("翻译这个文件", "/tmp/test.docx", "")["intent"] == "file_translate"
        assert classifier.classify("润色一下", "/tmp/test.docx", "")["intent"] == "file_polish"
        assert classifier.classify("生成PPT", "/tmp/test.pptx", "")["intent"] == "file_generate_ppt"
        assert classifier.classify("提取待办事项", "/tmp/test.docx", "")["intent"] == "file_extract_todos"
        assert classifier.classify("随便说点啥", "/tmp/test.docx", "")["intent"] == "file_parse"

    def test_file_processing_intent_with_attachment(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        orchestrator = Orchestrator(llm, memory)

        result = orchestrator.intent_classifier.classify("随便说点啥", "/tmp/test.docx", "")
        assert result["intent"] == "file_parse"

    def test_general_chat(self):
        llm = create_mock_llm()
        memory = create_mock_memory()
        orchestrator = Orchestrator(llm, memory)

        result = orchestrator._execute_general_chat("你好", "")
        assert result["status"] == "success"
        assert "response" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
