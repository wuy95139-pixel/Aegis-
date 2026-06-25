"""
共享测试 Fixtures
=================
所有测试文件通过此 conftest.py 获取统一的 mock 和 helper fixtures。

设计原则:
  - Mock ChromaDB、LLM API、外部网络调用
  - 使用真实 ShortTermMemory、FileStore、TaskStore（纯 Python，快速）
  - autouse fixtures 确保测试隔离（Config 单例重置等）
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm.provider import LLMProvider
from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.file_store import FileStore
from src.core.memory.types import MemoryType, MemoryFrontmatter
from src.models.schemas import MemoryEntry, ConversationTurn, AgentMessage


# ===================== Mock LLM =====================

@pytest.fixture
def mock_llm():
    """创建 mock LLMProvider，chat 返回标准响应。"""
    llm = Mock(spec=LLMProvider)
    llm.chat.return_value = {
        "content": "This is a mock LLM response.",
        "tool_calls": None,
        "reasoning_content": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "finish_reason": "stop",
    }
    llm.stream_chat.return_value = iter(["chunk1", "chunk2"])
    llm.embed.return_value = [[0.1] * 1536]
    llm.default_model = "test-model"
    llm.config = {}
    llm.temperature = 0.7
    return llm


def _make_mock_llm_with_response(json_str: str, tool_calls=None):
    """创建返回特定 JSON 的 mock LLMProvider。"""
    llm = Mock(spec=LLMProvider)
    llm.chat.return_value = {
        "content": json_str,
        "tool_calls": tool_calls,
        "reasoning_content": None,
        "usage": {},
        "finish_reason": "stop",
    }
    llm.stream_chat.return_value = iter([json_str])
    llm.embed.return_value = [[0.1] * 1536]
    llm.default_model = "test-model"
    llm.config = {}
    llm.temperature = 0.7
    return llm


# ===================== Memory Subsystems =====================

@pytest.fixture
def short_term_memory():
    """真实 ShortTermMemory，小配置用于快速测试。"""
    return ShortTermMemory(max_tokens=500, window_size=6)


@pytest.fixture
def mock_long_term():
    """Mock LongTermMemory（ChromaDB），避免 SQLite 和 embedding 调用。"""
    lt = Mock(spec=LongTermMemory)
    lt.count.return_value = 0
    lt.search.return_value = []
    lt.store.return_value = "mock_vector_id"
    lt.delete.return_value = None
    lt.forget_source.return_value = None
    lt.list_by_source.return_value = []
    return lt


@pytest.fixture
def mock_memory_retriever(short_term_memory, mock_long_term):
    """MemoryRetriever，真实 ShortTermMemory + Mock LongTermMemory。"""
    return MemoryRetriever(
        short_term=short_term_memory,
        long_term=mock_long_term,
        file_store=None,
    )


@pytest.fixture
def temp_file_store():
    """临时目录中的真实 FileStore。每个测试自动清理。"""
    base_dir = tempfile.mkdtemp(prefix="aegis_test_fs_")
    fs = FileStore(base_dir=base_dir)
    yield fs
    shutil.rmtree(base_dir, ignore_errors=True)


# ===================== Temp Files =====================

@pytest.fixture
def temp_txt_file(tmp_path):
    """临时 .txt 文件。"""
    f = tmp_path / "test.txt"
    f.write_text("Hello World\nThis is test content.\n", encoding="utf-8")
    return str(f)


@pytest.fixture
def temp_csv_file(tmp_path):
    """临时 .csv 文件。"""
    f = tmp_path / "test.csv"
    f.write_text("name,value,category\nalpha,10,A\nbeta,20,B\ngamma,30,A\n", encoding="utf-8")
    return str(f)


@pytest.fixture
def temp_xlsx_file(tmp_path):
    """临时 .xlsx 文件（通过 openpyxl 创建）。"""
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["name", "value", "category"])
        ws.append(["alpha", 10, "A"])
        ws.append(["beta", 20, "B"])
        ws.append(["gamma", 30, "A"])
        f = tmp_path / "test.xlsx"
        wb.save(str(f))
        return str(f)
    except ImportError:
        pytest.skip("openpyxl not installed")


# ===================== Config =====================

@pytest.fixture(autouse=True)
def reset_config_singleton():
    """每个测试后重置 Config 单例，防止状态泄漏。"""
    from src.utils.config import Config
    Config._instance = None
    Config._config = {}
    yield
    Config._instance = None
    Config._config = {}


# ===================== Task Store =====================

@pytest.fixture
def empty_task_store():
    """内存模式 TaskStore（无持久化）。"""
    from src.core.tools.task_store import TaskStore
    return TaskStore(storage_path=None)


@pytest.fixture
def populated_task_store(empty_task_store):
    """预填充 3 个任务的 TaskStore。"""
    empty_task_store.add_task_group(
        tasks=[
            {"title": "Write tests", "priority": "high"},
            {"title": "Review PR", "priority": "medium"},
            {"title": "Update docs", "priority": "low"},
        ],
        group_name="Sprint Tasks",
        context="Testing sprint",
        set_active=True,
    )
    return empty_task_store


# ===================== Agents =====================

@pytest.fixture
def mock_agents_dict(mock_llm, mock_memory_retriever):
    """创建所有 5 个 Agent 的字典（用于 Orchestrator 测试）。"""
    from src.core.agents.file_processor import FileProcessorAgent
    from src.core.agents.task_dispatcher import TaskDispatcherAgent
    from src.core.agents.memory_agent import MemoryAgent
    from src.core.agents.reminder_agent import ReminderAgent
    from src.core.agents.research_agent import ResearchAgent

    return {
        "file_processor": FileProcessorAgent(mock_llm, mock_memory_retriever),
        "task_dispatcher": TaskDispatcherAgent(mock_llm, mock_memory_retriever),
        "memory_agent": MemoryAgent(mock_llm, mock_memory_retriever),
        "reminder_agent": ReminderAgent(mock_llm, mock_memory_retriever),
        "research_agent": ResearchAgent(mock_llm, mock_memory_retriever),
    }


# ===================== Orchestrator Factory =====================

@pytest.fixture
def create_test_orchestrator(mock_llm, mock_memory_retriever, tmp_path):
    """创建完整初始化的 Orchestrator 用于测试。

    使用 Orchestrator.__init__（而非 __new__）正常构造，
    返回工厂函数以支持按需覆盖属性。

    用法:
        def test_something(create_test_orchestrator):
            orch = create_test_orchestrator()
            # 或者带覆盖:
            orch = create_test_orchestrator(
                task_store=TaskStore(storage_path=str(tmp_path / "tasks.json")),
                agents=mock_agents,  # 替换为 mock Agent
            )
    """
    from src.core.agents.orchestrator import Orchestrator
    from src.core.tools.task_store import TaskStore

    created = []  # 跟踪创建的实例用于清理

    def _create(**overrides):
        task_store = overrides.pop('task_store', None)
        config = overrides.pop('config', None)
        memory_manager = overrides.pop('memory_manager', None)
        agents_override = overrides.pop('agents', None)

        if config is None:
            config = {}

        orch = Orchestrator(
            llm=mock_llm,
            memory=mock_memory_retriever,
            config=config,
            memory_manager=memory_manager,
        )

        # 替换为内存模式 TaskStore（避免 ./data/tasks.json 文件泄漏）
        if task_store is not None:
            orch.task_store = task_store
        else:
            orch.task_store = TaskStore(storage_path=None)
        # 同步更新 handlers 中的 task_store 引用
        if hasattr(orch, 'handlers') and orch.handlers is not None:
            orch.handlers.task.task_store = orch.task_store
            orch.handlers.file.task_store = orch.task_store

        # 替换 Agent（如果提供了 mock agents）
        if agents_override is not None:
            orch.agents = agents_override

        # 应用其他覆盖（包括 memory_manager → 同步到 background）
        for key, value in overrides.items():
            setattr(orch, key, value)
        if 'memory_manager' in overrides and hasattr(orch, 'background') and orch.background is not None:
            orch.background._memory_manager = overrides['memory_manager']

        created.append(orch)
        return orch

    yield _create

    # 清理：停止后台调度器线程
    for orch in created:
        try:
            for agent in orch.agents.values():
                if hasattr(agent, 'calendar') and hasattr(agent.calendar, 'stop_scheduler'):
                    agent.calendar.stop_scheduler()
        except Exception:
            pass


# ===================== Structured Data Helpers =====================

@pytest.fixture
def sample_structured_data():
    """创建一份样本 StructuredData 用于 MCP 测试。"""
    from src.models.schemas import StructuredData, ColumnStats

    return StructuredData(
        filename="test.csv",
        columns=["name", "value", "category"],
        column_stats={
            "name": ColumnStats(name="name", dtype="string", count=3, null_count=0, unique_count=3),
            "value": ColumnStats(
                name="value", dtype="numeric", count=3, null_count=0, unique_count=3,
                min=10.0, max=30.0, mean=20.0, median=20.0, std=10.0,
                q1=15.0, q3=25.0, sample_values=[10, 20, 30],
            ),
            "category": ColumnStats(name="category", dtype="string", count=3, null_count=0, unique_count=2),
        },
        rows=[
            {"name": "alpha", "value": "10", "category": "A"},
            {"name": "beta", "value": "20", "category": "B"},
            {"name": "gamma", "value": "30", "category": "A"},
        ],
        row_count=3,
        col_count=3,
    )
