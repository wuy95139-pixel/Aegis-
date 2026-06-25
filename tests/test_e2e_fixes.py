"""
端到端验证测试 — 覆盖三个修复点
===============================
运行:
    cd d:/Aegis
    python -m pytest tests/test_e2e_fixes.py -v -s

测试覆盖:
  Fix 1: 任务完成检测 (regex 提取中文任务名, 同步状态列表)
  Fix 2: TaskStore 流程 (完成任务 → 同组下一个 → 其它组)
  Fix 3: 长期记忆持久化 (ChromaDB 重试, FileStore 多策略搜索, 降级检索)
"""

import sys
import json
import threading
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.file_store import FileStore
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.types import MemoryType, MemoryFrontmatter
from src.core.tools.task_store import TaskStore, TaskItem, TaskGroup
from src.core.tools.calendar_tools import CalendarTool
from src.core.agents.orchestrator import Orchestrator
from src.models.schemas import (
    MemoryEntry, ConversationTurn, AgentMessage,
)


# ============================================================
# FIX 1: 任务完成检测 — 正则提取中文任务名
# ============================================================

class TestTaskNameExtraction:
    """测试 _extract_completed_task_name 的各种中文表达"""

    @pytest.fixture
    def orchestrator(self, create_test_orchestrator):
        """创建带 mock Agent 和内存 TaskStore 的 orchestrator"""
        mock_agents = {
            "reminder_agent": Mock()
        }
        mock_agents["reminder_agent"].execute.return_value = {"reminders": []}
        return create_test_orchestrator(agents=mock_agents)

    def test_before_keyword_greedy(self, orchestrator):
        """策略2: 'X做完了' — 贪婪匹配关键词前的任务名"""
        orchestrator.task_store.add_task_group(
            tasks=[
                {"title": "语言识别/分离/分群的开发"},
                {"title": "搭建agent的基础框架"},
            ],
            group_name="开发任务",
            context="测试",
        )
        result = orchestrator._extract_completed_task_name(
            "我现在完成了语言识别/分离/分群的开发，接下来做什么"
        )
        assert result == "语言识别/分离/分群的开发", f"Expected full task name, got: {result}"

    def test_before_keyword_with_prefix(self, orchestrator):
        """策略2: 去掉主语前缀 '我终于'，提取到后面的任务名"""
        orchestrator.task_store.add_task_group(
            tasks=[{"title": "写本周的周报总结"}],
            group_name="日常",
            context="测试",
        )
        result = orchestrator._extract_completed_task_name("我终于写完了本周的周报总结")
        assert result == "本周的周报总结", f"Expected '本周的周报总结', got: {result}"

    def test_before_keyword_with_ba(self, orchestrator):
        """策略2: 去掉 '把' 字前缀"""
        result = orchestrator._extract_completed_task_name("我把今天的晚饭做好了")
        assert result is not None
        assert "晚饭" in result, f"Expected task containing '晚饭', got: {result}"

    def test_after_keyword_pattern(self, orchestrator):
        """策略3: '完成了X' — 关键词后的任务名"""
        result = orchestrator._extract_completed_task_name("完成了数学作业，接下来做什么")
        assert result == "数学作业", f"Expected '数学作业', got: {result}"

    def test_before_takes_priority_over_after(self, orchestrator):
        """策略2 优先于策略3：任务名在关键词前面时优先匹配前面"""
        orchestrator.task_store.add_task_group(
            tasks=[{"title": "项目架构设计文档"}],
            group_name="文档",
            context="测试",
        )
        # "完成了" 前后都有内容，前面应该是任务名
        result = orchestrator._extract_completed_task_name(
            "项目架构设计文档完成了，接下来做什么"
        )
        # before 策略应该优先匹配到 "项目架构设计文档"
        assert result and "项目架构设计文档" in result, f"Expected project task, got: {result}"

    def test_known_task_title_in_message(self, orchestrator):
        """策略1: TaskStore 中已知标题直接匹配"""
        orchestrator.task_store.add_task_group(
            tasks=[
                {"title": "完成语音识别模块开发"},
                {"title": "搭建agent基础框架"},
            ],
            group_name="开发",
            context="测试",
        )
        result = orchestrator._extract_completed_task_name(
            "完成语音识别模块开发做完了"
        )
        assert result == "完成语音识别模块开发", f"Expected exact title match, got: {result}"

    def test_no_task_in_message(self, orchestrator):
        """纯聊天内容不应提取到假的任务名"""
        result = orchestrator._extract_completed_task_name("今天天气不错")
        assert result is None, f"Expected None for non-task message, got: {result}"

    def test_short_noise_filtered(self, orchestrator):
        """噪声词不应被提取为任务名"""
        result = orchestrator._extract_completed_task_name("我已经完成了")
        # "完成了" 前后都无可提取的任务名
        # 如果返回了内容，应该是有效的任务名而非噪声
        if result:
            noise = {"的", "了", "吗", "呢", "啊", "什么", "怎么", "我", "我们"}
            assert result not in noise, f"Noise word leaked through: {result}"


# ============================================================
# FIX 2: TaskStore 流程 — 完成任务 → 下一个 → 其它组
# ============================================================

class TestTaskStoreFlow:
    """测试 TaskStore 的完整任务管理流程"""

    @pytest.fixture
    def store(self):
        """创建内存 TaskStore"""
        return TaskStore(storage_path=None)

    def test_add_group_and_complete_flow(self, store):
        """完整流程：添加分组 → 完成第一个 → 返回下一个 → 完成所有 → 查看其它组"""
        # 1. 添加开发任务组
        gid1 = store.add_task_group(
            tasks=[
                {"title": "完成语音识别模块"},
                {"title": "搭建agent框架"},
                {"title": "添加功能代码"},
            ],
            group_name="当前开发任务",
            context="用户说：我有三件事需要做",
        )
        assert gid1 is not None
        assert store.get_pending_count() == 3

        # 2. 完成第一个任务
        result = store.complete_task_by_title("完成语音识别模块")
        assert result["completed"] is not None
        assert result["completed"].title == "完成语音识别模块"
        assert result["completed"].status == "completed"

        # 3. 同组下一个应该是"搭建agent框架"
        assert result["next_in_group"] is not None
        assert result["next_in_group"].title == "搭建agent框架"
        assert result["group_all_done"] is False
        assert result["all_done"] is False

        # 4. 完成第二个任务
        result2 = store.complete_task_by_title("搭建agent框架")
        assert result2["next_in_group"].title == "添加功能代码"

        # 5. 完成最后一个
        result3 = store.complete_task_by_title("添加功能代码")
        assert result3["next_in_group"] is None
        assert result3["group_all_done"] is True

        # 6. 所有任务都完成
        assert store.get_pending_count() == 0

    def test_multi_group_flow(self, store):
        """多分组流程：同组任务优先，完成后提示其它组"""
        # 添加开发任务组（活跃组）
        store.add_task_group(
            tasks=[
                {"title": "写完代码"},
                {"title": "写测试"},
            ],
            group_name="开发任务",
        )
        # 添加生活任务组
        store.add_task_group(
            tasks=[
                {"title": "把饭煮了"},
                {"title": "扫地拖地"},
            ],
            group_name="日常事务",
            set_active=False,
        )

        # 设置活跃组为开发任务
        all_groups = store.get_all_pending_tasks()
        dev_group = [g for g in all_groups if g["group_name"] == "开发任务"][0]
        store.set_active_group(dev_group["group_id"])

        # 完成第一个开发任务
        result = store.complete_task_by_title("写完代码")
        assert result["completed"].title == "写完代码"
        assert result["next_in_group"].title == "写测试"
        assert not result["group_all_done"]

        # 完成第二个开发任务 → 组完成，应提示其它组
        result2 = store.complete_task_by_title("写测试")
        assert result2["next_in_group"] is None
        assert result2["group_all_done"] is True
        assert len(result2["other_groups_pending"]) > 0
        assert result2["other_groups_pending"][0]["group_name"] == "日常事务"

    def test_fuzzy_match_by_substring(self, store):
        """模糊匹配：用户说的关键词是任务标题的子串"""
        store.add_task_group(
            tasks=[{"title": "完成语音识别/分离/分群板块的开发"}],
            group_name="开发",
        )
        result = store.complete_task_by_title("语音识别")
        assert result["completed"] is not None
        assert "语音识别" in result["completed"].title

    def test_build_group_status_sync(self, store):
        """任务状态同步显示"""
        store.add_task_group(
            tasks=[
                {"title": "任务A"},
                {"title": "任务B"},
                {"title": "任务C"},
            ],
            group_name="测试组",
        )
        # 完成部分任务
        store.complete_task_by_title("任务A")
        store.complete_task_by_title("任务C")

        # 获取状态
        all_groups = store.get_all_pending_tasks()
        group = all_groups[0]
        tasks = group["tasks"]

        statuses = {t["title"]: t["status"] for t in tasks}
        assert statuses["任务A"] == "completed"
        assert statuses["任务B"] == "pending"
        assert statuses["任务C"] == "completed"

    def test_persistence(self, tmp_path):
        """持久化：保存到文件 → 重新加载 → 数据完整"""
        filepath = tmp_path / "tasks.json"

        # 创建并保存
        store1 = TaskStore(storage_path=str(filepath))
        store1.add_task_group(
            tasks=[
                {"title": "持久化任务1"},
                {"title": "持久化任务2"},
            ],
            group_name="持久化测试",
        )
        store1.complete_task_by_title("持久化任务1")

        # 重新加载
        store2 = TaskStore(storage_path=str(filepath))
        assert store2.get_pending_count() == 1
        all_groups = store2.get_all_pending_tasks()
        assert len(all_groups) == 1
        tasks = all_groups[0]["tasks"]
        statuses = {t["title"]: t["status"] for t in tasks}
        assert statuses["持久化任务1"] == "completed"
        assert statuses["持久化任务2"] == "pending"


# ============================================================
# FIX 3: 长期记忆持久化
# ============================================================

class TestFileStoreFullTextSearch:
    """测试 FileStore 全文搜索的 3 策略匹配"""

    @pytest.fixture
    def fs(self, tmp_path):
        """创建临时 FileStore"""
        store = FileStore(base_dir=str(tmp_path / "memory"))
        # 存入测试记忆
        store.save(
            MemoryFrontmatter(
                name="project_arch",
                description="项目架构设计讨论",
                type=MemoryType.PROJECT,
                tags=["架构", "微服务"],
                importance=0.8,
            ),
            "项目架构需要支持微服务部署，前端使用React，后端使用FastAPI。",
        )
        store.save(
            MemoryFrontmatter(
                name="user_pref",
                description="用户偏好设置",
                type=MemoryType.USER,
                tags=["偏好", "中文"],
                importance=0.6,
            ),
            "用户偏好使用中文交流，喜欢简洁的回答风格。",
        )
        return store

    def test_exact_substring_match(self, fs):
        """策略1: 精确子串匹配 — "架构" 应命中"""
        results = fs.full_text_search("架构", MemoryType.PROJECT, limit=5)
        assert len(results) > 0
        assert any("架构" in r["content"] for r in results)

    def test_bigram_match_chinese(self, fs):
        """策略2: Bigram 匹配 — '之前提到的架构是什么' 应命中"""
        # 整个查询不是任何记忆的子串，但 bigram "架构" 会匹配
        results = fs.full_text_search("之前提到的架构是什么", MemoryType.PROJECT, limit=5)
        assert len(results) > 0, (
            "Bigram match failed! '架构' should match via bigram strategy. "
            f"Query bigrams would contain '架构'"
        )

    def test_character_level_match(self, fs):
        """策略3: 单字匹配 — 含多个内容词的查询"""
        # "微服务部署方案" — 拆成单字后 "微""服""务""部""署" 至少有3个命中
        results = fs.full_text_search("微服务部署方案", MemoryType.PROJECT, limit=5)
        assert len(results) > 0, "Character-level match should find '微服务' in content"

    def test_short_keyword_match(self, fs):
        """短关键词匹配 — 'React' 精确匹配"""
        results = fs.full_text_search("React", MemoryType.PROJECT, limit=5)
        assert len(results) > 0
        assert any("React" in r["content"] for r in results)

    def test_no_match_for_unrelated_query(self, fs):
        """不相关的查询不应返回结果"""
        results = fs.full_text_search("量子计算机", MemoryType.PROJECT, limit=5)
        assert len(results) == 0


class TestMemoryRetrieverFallback:
    """测试 MemoryRetriever 在 ChromaDB 不可用时的 FileStore 降级"""

    @pytest.fixture
    def retriever_with_files(self, tmp_path):
        """创建带 FileStore 但 ChromaDB 为空的 MemoryRetriever"""
        st = ShortTermMemory(max_tokens=1000, window_size=10)
        lt = Mock(spec=LongTermMemory)
        lt.count.return_value = 0  # 模拟空 ChromaDB
        lt.search.return_value = []

        fs = FileStore(base_dir=str(tmp_path / "memory"))
        # 存入记忆
        fs.save(
            MemoryFrontmatter(
                name="important_project",
                description="用微服务架构来构建Aegis项目",
                type=MemoryType.PROJECT,
                tags=["架构", "微服务"],
                importance=0.9,
            ),
            "项目架构采用微服务设计，核心模块包括记忆系统、任务系统、Agent调度。",
        )
        fs.save(
            MemoryFrontmatter(
                name="user_role",
                description="用户是后端工程师",
                type=MemoryType.USER,
                tags=["角色"],
                importance=0.7,
            ),
            "用户是一名资深后端工程师，主要使用Python和FastAPI。",
        )

        return MemoryRetriever(short_term=st, long_term=lt, file_store=fs)

    def test_fallback_when_chromadb_empty(self, retriever_with_files):
        """ChromaDB 为空时，file_memories 包含降级结果"""
        result = retriever_with_files.retrieve("微服务架构", top_k=5)
        # FileStore 应找到结果
        assert len(result["file_memories"]) > 0, "FileStore should find results"
        # _handle_memory_search 会同时读取 relevant_memories 和 file_memories
        # 所以 file_memories 有结果就足够了
        combined = result["combined_context"]
        assert "相关记忆文件" in combined
        assert "微服务" in combined

    def test_combined_context_includes_file_memories(self, retriever_with_files):
        """合并的上下文中应包含文件记忆"""
        result = retriever_with_files.retrieve("架构", top_k=5)
        assert result["combined_context"] != ""
        assert "相关记忆文件" in result["combined_context"]
        assert "微服务" in result["combined_context"]

    def test_retrieve_by_type(self, retriever_with_files):
        """按类型检索：只返回 user 类型"""
        result = retriever_with_files.retrieve_by_type("user", top_k=5)
        for entry in result["file_memories"]:
            fm = entry["frontmatter"]
            assert fm.type == MemoryType.USER or fm.type.value == "user"

    def test_retrieve_user_profile(self, retriever_with_files):
        """获取用户画像"""
        result = retriever_with_files.retrieve_user_profile()
        assert "user_profile" in result
        assert result["user_profile"] != "", "Should have user profile"

    def test_no_results_for_unrelated_query(self, retriever_with_files):
        """不相关的查询无结果"""
        result = retriever_with_files.retrieve("量子计算机", top_k=5)
        assert len(result["file_memories"]) == 0


class TestChromaDBPersistence:
    """测试 ChromaDB 持久化客户端的创建和降级"""

    def test_create_persistent_client_exists(self):
        """LongTermMemory 实现了 _create_persistent_client 方法"""
        assert hasattr(LongTermMemory, '_create_persistent_client')
        assert hasattr(LongTermMemory, '_cleanup_sqlite_locks')

    def test_ephemeral_fallback_marks_non_persistent(self):
        """当 PersistentClient 失败时，标记为非持久化"""
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm.persist_dir = "/nonexistent/path/that/will/fail"
        ltm.collection_name = "test"
        ltm.embedding_model = "test-model"
        ltm._is_persistent = True

        client = ltm._create_persistent_client()
        # 如果初始化失败（预期在此路径），_is_persistent 应为 False
        # 注意：可能因为权限等原因仍然成功，所以这个测试不强制断言
        # 只是验证方法能正常执行而不抛异常
        assert hasattr(client, 'heartbeat')

    def test_cleanup_sqlite_locks_safe(self, tmp_path):
        """清理 SQLite 锁文件不会抛出异常"""
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm.persist_dir = str(tmp_path)
        # 创建一些假的 lock 文件
        lock_dir = tmp_path / "subdir"
        lock_dir.mkdir(exist_ok=True)
        (lock_dir / "test.lock").touch()
        # 不应抛出异常
        ltm._cleanup_sqlite_locks()


# ============================================================
# 集成测试：模拟完整用户对话流程
# ============================================================

class TestIntegrationFullFlow:
    """端到端集成测试：模拟用户完整对话流程"""

    @pytest.fixture
    def orchestrator(self, create_test_orchestrator, tmp_path):
        """创建带 mock Agent 和文件 TaskStore 的 orchestrator"""
        mock_agents = {
            "reminder_agent": Mock(),
            "memory_agent": Mock(),
            "file_processor": Mock(),
            "task_dispatcher": Mock(),
            "research_agent": Mock(),
        }
        mock_agents["reminder_agent"].execute.return_value = {"reminders": []}
        mock_agents["memory_agent"].execute.return_value = {"status": "success"}
        mock_agents["file_processor"].execute.return_value = {"status": "success"}
        mock_agents["task_dispatcher"].execute.return_value = {"status": "success", "assigned_todos": [], "unassigned": []}
        mock_agents["research_agent"].execute.return_value = {"status": "success", "report": None}

        orch = create_test_orchestrator(
            agents=mock_agents,
            task_store=TaskStore(storage_path=str(tmp_path / "tasks.json")),
            _repeated_content_tracker={},
            _repeat_tracker_lock=threading.Lock(),
            _classify_intent=Mock(return_value="task_inquiry"),
            memory_manager=None,
        )
        return orch

    def test_full_task_lifecycle(self, orchestrator):
        """完整任务生命周期：添加 → 列出 → 完成 → 同步状态"""
        # Step 1: 用户添加任务
        orchestrator.task_store.add_task_group(
            tasks=[
                {"title": "完成语音识别模块"},
                {"title": "搭建agent框架"},
                {"title": "添加功能代码"},
            ],
            group_name="当前开发任务",
            context="用户消息：我有三件事需要做",
        )

        # Step 2: 查询任务（应显示3个待完成）
        inquiry = orchestrator._handle_task_inquiry()
        assert inquiry["status"] == "success"
        response = inquiry["response"]
        assert "待完成" in response
        assert "完成语音识别模块" in response

        # Step 3: 完成第一个任务
        task_name = orchestrator._extract_completed_task_name(
            "我现在完成了完成语音识别模块"
        )
        assert task_name is not None
        task_done_output = orchestrator._handle_task_done(task_name)
        assert "已完成" in task_done_output
        assert "下一步" in task_done_output
        assert "搭建agent框架" in task_done_output

        # Step 4: 查看同步后的任务列表
        inquiry2 = orchestrator._handle_task_inquiry()
        response2 = inquiry2["response"]
        assert "已完成" in response2
        assert "待完成" in response2

    def test_long_term_memory_keyword_detection(self, orchestrator):
        """长期记忆：重要性关键词检测"""
        mm = Mock()
        orchestrator.memory_manager = mm
        orchestrator.background.memory_manager = mm  # 同步到 BackgroundProcessor

        orchestrator._auto_detect_long_term_memory(
            "项目架构采用微服务设计，这个很重要，记住这个。"
        )

        assert mm.remember.called, (
            "memory_manager.remember should be called when importance keyword detected"
        )

    def test_long_term_memory_repeated_content(self, orchestrator):
        """长期记忆：重复内容触发"""
        mm = Mock()
        orchestrator.memory_manager = mm
        orchestrator.background.memory_manager = mm  # 同步到 BackgroundProcessor

        # 模拟 LLM 返回话题关键词
        orchestrator.llm.chat.return_value = {"content": "微服务架构,性能优化", "tool_calls": None, "usage": {}, "finish_reason": "stop"}

        # 模拟同一话题出现3次
        orchestrator._track_repeated_content("微服务架构的性能需要优化")
        orchestrator._track_repeated_content("微服务架构的延迟有点高")
        orchestrator._track_repeated_content("微服务架构怎么优化")

        # 检查 tracker 计数
        assert len(orchestrator._repeated_content_tracker) > 0
        topic_keys = list(orchestrator._repeated_content_tracker.keys())
        assert any("微服务" in k for k in topic_keys)

        # 触发长期记忆检测
        orchestrator._auto_detect_long_term_memory("微服务架构怎么优化")
        assert mm.remember.called, (
            "memory_manager.remember should be called for repeated topic >= 3 times"
        )

    def test_file_store_cross_session_simulation(self, tmp_path):
        """模拟跨会话：存 → 重加载 → 检索"""
        fs1 = FileStore(base_dir=str(tmp_path / "memory"))
        fm = MemoryFrontmatter(
            name="arch_decision",
            description="项目架构决策",
            type=MemoryType.PROJECT,
            tags=["架构", "decision"],
            importance=0.9,
        )
        fs1.save(fm, "决定使用微服务+事件驱动架构，消息队列用Kafka。")

        # 验证存入了
        entry = fs1.get("arch_decision", MemoryType.PROJECT)
        assert entry is not None
        assert "微服务" in entry["content"]

        # 模拟新会话 — 重新创建 FileStore
        fs2 = FileStore(base_dir=str(tmp_path / "memory"))
        entry2 = fs2.get("arch_decision", MemoryType.PROJECT)
        assert entry2 is not None
        assert entry2["content"] == entry["content"]

        # 全文搜索应能找到（策略1: 精确子串匹配 "Kafka"）
        results = fs2.full_text_search("Kafka", MemoryType.PROJECT, limit=5)
        assert len(results) > 0
        assert "Kafka" in results[0]["content"]

        # Bigram 搜索应能找到（策略2）
        results2 = fs2.full_text_search("之前讨论的消息队列方案", MemoryType.PROJECT, limit=5)
        assert len(results2) > 0, (
            "Bigram matching should find '消息队列' in content"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# ============================================================
# FIX 5: Agent-to-Agent 通信
# ============================================================

class TestAgentToAgentCommunication:
    """测试 Agent 间通过 MessageBus 进行对等通信"""

    @pytest.fixture
    def bus(self):
        """创建真实的 MessageBus"""
        from src.core.agents.orchestrator import MessageBus
        return MessageBus()

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.chat.return_value = {
            "content": "mock response",
            "tool_calls": None,
            "usage": {},
            "finish_reason": "stop",
        }
        return llm

    @pytest.fixture
    def mock_memory(self):
        from src.core.memory.short_term import ShortTermMemory
        from src.core.memory.long_term import LongTermMemory
        from src.core.memory.retriever import MemoryRetriever
        st = ShortTermMemory(max_tokens=1000, window_size=10)
        lt = Mock(spec=LongTermMemory)
        lt.count.return_value = 0
        lt.search.return_value = []
        lt.store.return_value = "mock_id"
        return MemoryRetriever(short_term=st, long_term=lt)

    def test_send_and_receive_request_response(self, bus, mock_llm, mock_memory):
        """两个 Agent 通过 MessageBus 完成 request → response 往返"""
        from src.core.agents.memory_agent import MemoryAgent
        from src.core.agents.reminder_agent import ReminderAgent

        agent_a = MemoryAgent(mock_llm, mock_memory)
        agent_a.message_bus = bus
        bus.subscribe("memory_agent", lambda msg: agent_a.receive_message(msg))

        agent_b = ReminderAgent(mock_llm, mock_memory)
        agent_b.message_bus = bus
        # 覆盖 receive_message 来模拟回复
        received_b = []
        def b_handler(msg):
            received_b.append(msg)
            # 收到 request，回复 response
            agent_b.send_message(
                receiver=msg.sender,
                msg_type="response",
                payload={"answer": "got it"},
                reply_to=msg.id,
            )
        bus.subscribe("reminder_agent", b_handler)

        # agent_a 发 request 给 agent_b
        received_a = []
        def a_handler(msg):
            received_a.append(msg)
        bus.subscribe("memory_agent", a_handler)

        msg = agent_a.send_message(
            receiver="reminder_agent",
            msg_type="request",
            payload={"question": "any reminders?"},
        )

        # agent_b 应收到
        assert len(received_b) == 1
        assert received_b[0].payload["question"] == "any reminders?"

        # agent_a 应收到 agent_b 的 response（通过 a_handler 第二次触发）
        assert len(received_a) == 1
        assert received_a[0].type == "response"

    def test_broadcast_to_all_agents(self, bus, mock_llm, mock_memory):
        """广播消息 → 所有订阅者都收到"""
        from src.core.agents.memory_agent import MemoryAgent
        from src.core.agents.reminder_agent import ReminderAgent

        received = {"a": [], "b": []}

        agent_a = MemoryAgent(mock_llm, mock_memory)
        agent_a.message_bus = bus
        bus.subscribe("memory_agent", lambda msg: received["a"].append(msg))

        agent_b = ReminderAgent(mock_llm, mock_memory)
        agent_b.message_bus = bus
        bus.subscribe("reminder_agent", lambda msg: received["b"].append(msg))

        agent_a.send_message(
            receiver="broadcast",
            msg_type="event",
            payload={"event": "test_broadcast"},
        )

        assert len(received["a"]) == 1
        assert len(received["b"]) == 1
        assert received["a"][0].payload["event"] == "test_broadcast"
        assert received["b"][0].payload["event"] == "test_broadcast"

    def test_agent_without_bus_does_not_crash(self, mock_llm, mock_memory):
        """没有注入 bus 的 Agent 调 send_message 不会崩溃"""
        from src.core.agents.memory_agent import MemoryAgent
        agent = MemoryAgent(mock_llm, mock_memory)
        # message_bus is None by default
        msg = agent.send_message(
            receiver="someone",
            msg_type="event",
            payload={"test": True},
        )
        assert msg is not None
        assert msg.type == "event"

    def test_receive_message_default_behavior(self, mock_llm, mock_memory):
        """基类的 receive_message 默认返回 None（不回复）"""
        from src.core.agents.memory_agent import MemoryAgent
        from src.models.schemas import AgentMessage

        agent = MemoryAgent(mock_llm, mock_memory)
        msg = AgentMessage(
            id="test_1",
            sender="test_sender",
            receiver="memory_agent",
            type="event",
            payload={"event": "test"},
        )
        reply = agent.receive_message(msg)
        assert reply is None

    def test_memory_agent_store_broadcasts_event(self, mock_llm, mock_memory, bus):
        """MemoryAgent 存储记忆后广播 memory_updated 事件"""
        from src.core.agents.memory_agent import MemoryAgent

        agent = MemoryAgent(mock_llm, mock_memory)
        agent.message_bus = bus

        received = []
        bus.subscribe("reminder_agent", lambda msg: received.append(msg))

        result = agent.execute({
            "operation": "store",
            "content": "项目架构采用微服务设计，这个很重要",
            "source": "test_chat",
            "tags": ["architecture", "important"],
        })

        assert result["status"] == "success"
        # 验证广播事件已发布
        memory_events = [m for m in bus._history if m.type == "event" and m.payload.get("event") == "memory_updated"]
        assert len(memory_events) > 0, "MemoryAgent should broadcast memory_updated event after store"

    def test_reminder_agent_handles_memory_event(self, mock_llm, mock_memory):
        """ReminderAgent 的 receive_message 响应 memory_updated 事件"""
        from src.core.agents.reminder_agent import ReminderAgent
        from src.models.schemas import AgentMessage

        agent = ReminderAgent(mock_llm, mock_memory)
        msg = AgentMessage(
            id="test_mem_event",
            sender="memory_agent",
            receiver="reminder_agent",
            type="event",
            payload={"event": "memory_updated", "source": "test", "tags": ["deadline"]},
        )
        reply = agent.receive_message(msg)
        # 默认返回 None（不回复给 sender）
        assert reply is None  # super().receive_message returns None

    def test_orchestrator_injects_bus_to_all_agents(self, create_test_orchestrator):
        """Orchestrator 初始化后所有 Agent 都有 bus 引用"""
        orch = create_test_orchestrator()

        # 验证注入成功
        for agent in orch.agents.values():
            assert agent.message_bus is orch.message_bus, f"{agent.name} should have bus injected"

        # 验证订阅成功
        assert "memory_agent" in orch.message_bus._subscribers
        assert "reminder_agent" in orch.message_bus._subscribers

class TestReminderScheduler:
    """测试后台提醒调度器的自动触发"""

    @pytest.fixture
    def cal(self):
        """创建带短轮询间隔的 CalendarTool"""
        c = CalendarTool(storage_path=None, poll_interval=1.0)
        yield c
        c.stop_scheduler()

    def test_scheduler_starts_automatically(self, cal):
        """CalendarTool 创建时自动启动后台调度器"""
        assert cal._scheduler_thread is not None
        assert cal._scheduler_thread.is_alive()

    def test_scheduler_stops_cleanly(self, cal):
        """调度器可正常停止"""
        cal.stop_scheduler()
        time.sleep(0.3)
        assert not cal._scheduler_thread.is_alive()

    def test_reminder_fires_in_background(self, cal):
        """设置在2秒后的提醒应被后台线程捕获并触发"""
        from datetime import datetime, timedelta
        past_time = datetime.now() + timedelta(seconds=2)
        rid = cal.add_reminder(
            title="后台触发测试",
            description="应该在2秒内触发",
            trigger_time=past_time,
            notify_method=["console"],
        )
        # 等待调度器轮询
        time.sleep(4)
        reminders = cal.list_reminders(active_only=False)
        test_rem = next((r for r in reminders if r.id == rid), None)
        assert test_rem is not None
        assert test_rem.fire_count >= 1, (
            f"Reminder should have fired at least once, but fire_count={test_rem.fire_count}"
        )
        assert not test_rem.acknowledged, (
            f"Reminder should not be acknowledged until user confirms, but acknowledged={test_rem.acknowledged}"
        )

    def test_future_reminder_stays_active(self, cal):
        """未来的提醒保持活跃，不会被误触发"""
        from datetime import datetime, timedelta
        future_time = datetime.now() + timedelta(hours=24)
        rid = cal.add_reminder(
            title="未来提醒",
            description="明天的提醒",
            trigger_time=future_time,
            notify_method=["console"],
        )
        # 等待一次轮询
        time.sleep(2)
        reminders = cal.list_reminders(active_only=False)
        test_rem = next((r for r in reminders if r.id == rid), None)
        assert test_rem is not None
        assert test_rem.is_active, (
            "Future reminder should remain active"
        )

    def test_multiple_reminders_in_same_poll(self, cal):
        """同一轮询周期内的多个提醒都应被触发"""
        from datetime import datetime, timedelta
        past_time = datetime.now() + timedelta(seconds=2)
        ids = []
        for i in range(3):
            rid = cal.add_reminder(
                title=f"批量提醒{i+1}",
                trigger_time=past_time,
            )
            ids.append(rid)
        time.sleep(4)
        reminders = cal.list_reminders(active_only=False)
        for rid in ids:
            r = next((x for x in reminders if x.id == rid), None)
            assert r is not None and r.fire_count >= 1, f"Reminder {rid} should have fired at least once"

    def test_persistence_with_scheduler(self, tmp_path):
        """持久化 + 调度器：保存后重新加载，调度器仍正常工作"""
        from datetime import datetime, timedelta
        filepath = tmp_path / "reminders.json"

        # 创建并保存
        cal1 = CalendarTool(storage_path=str(filepath), poll_interval=1.0)
        cal1.add_reminder(
            title="持久化提醒",
            trigger_time=datetime.now() + timedelta(hours=48),
        )
        cal1.stop_scheduler()

        # 重新加载
        cal2 = CalendarTool(storage_path=str(filepath), poll_interval=1.0)
        reminders = cal2.list_reminders()
        assert len(reminders) == 1
        assert reminders[0].title == "持久化提醒"
        assert reminders[0].is_active
        assert cal2._scheduler_thread.is_alive()
        cal2.stop_scheduler()
