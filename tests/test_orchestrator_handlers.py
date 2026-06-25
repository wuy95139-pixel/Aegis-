"""
core/agents/orchestrator.py 测试
================================
Orchestrator 的关键方法和 MessageBus 测试。
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.core.agents.orchestrator import MessageBus
from src.models.schemas import AgentMessage


# ==================== MessageBus Tests ====================

class TestMessageBus:
    def test_publish_adds_to_queue(self):
        bus = MessageBus()
        msg = AgentMessage(id="m1", sender="a", receiver="b", type="event", payload={})
        bus.publish(msg)
        assert len(bus._history) == 1
        assert bus._history[0].id == "m1"

    def test_publish_to_specific_receiver(self):
        bus = MessageBus()
        handler = Mock()
        bus._subscribers["agent_b"] = [handler]
        msg = AgentMessage(id="m2", sender="a", receiver="agent_b", type="event", payload={})
        bus.publish(msg)
        handler.assert_called_once_with(msg)

    def test_broadcast_to_all(self):
        bus = MessageBus()
        h1, h2 = Mock(), Mock()
        bus._subscribers["agent_a"] = [h1]
        bus._subscribers["agent_b"] = [h2]
        msg = AgentMessage(id="m3", sender="x", receiver="broadcast", type="event", payload={})
        bus.publish(msg)
        h1.assert_called_once()
        h2.assert_called_once()

    def test_handler_exception_does_not_propagate(self):
        bus = MessageBus()
        bad_handler = Mock(side_effect=RuntimeError("ouch"))
        good_handler = Mock()
        bus._subscribers["agent_a"] = [bad_handler, good_handler]
        msg = AgentMessage(id="m4", sender="x", receiver="agent_a", type="event", payload={})
        bus.publish(msg)  # 不应崩溃
        assert good_handler.called

    def test_queue_max_size(self):
        bus = MessageBus(max_queue_size=3)
        for i in range(5):
            msg = AgentMessage(id=f"m{i}", sender="a", receiver="b", type="event", payload={})
            bus.publish(msg)
        # 队列仅保留最后 3 条
        assert len(bus._queue) == 3

    def test_nonexistent_receiver_no_crash(self):
        bus = MessageBus()
        msg = AgentMessage(id="m5", sender="a", receiver="no_one", type="event", payload={})
        bus.publish(msg)  # 不应崩溃
        assert len(bus._history) == 1

    def test_history_tracks_all(self):
        bus = MessageBus()
        for i in range(5):
            msg = AgentMessage(id=f"h{i}", sender="a", receiver="b", type="event", payload={})
            bus.publish(msg)
        assert len(bus._history) == 5


# ==================== Validate File Path Tests ====================

class TestValidateFilePath:
    def test_path_traversal_blocked(self):
        from src.core.agents.orchestrator import Orchestrator
        with pytest.raises(ValueError, match="超出允许范围"):
            Orchestrator._validate_file_path("/etc/passwd")

    def test_path_traversal_blocked_windows(self):
        from src.core.agents.orchestrator import Orchestrator
        with pytest.raises(ValueError, match="超出允许范围"):
            Orchestrator._validate_file_path("C:\\Windows\\System32\\config\\SAM")


# ==================== Extract Completed Task Name Tests ====================

class TestExtractCompletedTaskName:
    """测试 _extract_completed_task_name 的正则提取逻辑。

    由于该方法依赖 self.task_store 和 self.agents，通过 mock 隔离。"""

    @pytest.fixture
    def orchestrator(self, create_test_orchestrator):
        """创建带 mock Agent 的 orchestrator 用于测试任务名提取"""
        mock_agents = {
            "reminder_agent": Mock()
        }
        mock_agents["reminder_agent"].execute.return_value = {"reminders": []}
        return create_test_orchestrator(agents=mock_agents)

    def test_extracts_before_keyword(self, orchestrator):
        result = orchestrator._extract_completed_task_name("语言识别/分离的开发做完了")
        assert result is not None
        assert "语言识别" in result

    def test_extracts_after_keyword(self, orchestrator):
        result = orchestrator._extract_completed_task_name("完成了单元测试编写")
        assert result is not None
        assert "单元测试" in result

    def test_strips_prefixes(self, orchestrator):
        result = orchestrator._extract_completed_task_name("我已经完成了代码审查")
        assert result is not None
        # 前缀 "我已经" 应被去除
        assert not result.startswith("我已")

    def test_no_completion_keyword_returns_none(self, orchestrator):
        result = orchestrator._extract_completed_task_name("今天天气怎么样")
        assert result is None

    def test_task_store_match_priority(self, orchestrator):
        orchestrator.task_store.add_task_group(
            tasks=[{"title": "编写核心模块"}],
            group_name="Test",
            context="测试",
        )
        result = orchestrator._extract_completed_task_name("我已经编写核心模块完成了")
        assert result == "编写核心模块"
