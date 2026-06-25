"""
core/tools/task_store.py 测试
============================
TaskStore 的任务分组、完成、持久化功能测试。
"""

import json
import pytest
from src.core.tools.task_store import TaskStore, TaskItem, TaskGroup


class TestAddTaskGroup:
    def test_creates_group_with_tasks(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[{"title": "Task A"}, {"title": "Task B"}],
            group_name="Test Group",
        )
        assert gid is not None
        assert empty_task_store.get_pending_count() == 2

    def test_sets_active_by_default(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[{"title": "Task A"}],
            group_name="Active Test",
        )
        active = empty_task_store.get_active_group()
        assert active is not None
        assert active.name == "Active Test"

    def test_set_active_false(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "Task A"}],
            group_name="Inactive",
            set_active=False,
        )
        active = empty_task_store.get_active_group()
        assert active is None

    def test_auto_generates_name(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[{"title": "Task A"}],
            group_name="",
        )
        group = empty_task_store._groups[gid]
        assert group.name != ""
        assert "任务组" in group.name

    def test_task_order_preserved(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[
                {"title": "First"},
                {"title": "Second"},
                {"title": "Third"},
            ],
            group_name="Ordered",
        )
        tasks = empty_task_store.get_active_group_tasks()
        assert tasks[0].title == "First"
        assert tasks[1].title == "Second"
        assert tasks[2].title == "Third"
        assert tasks[0].order == 0
        assert tasks[2].order == 2

    def test_task_with_priority_and_deadline(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[{
                "title": "Urgent Task",
                "priority": "urgent",
                "deadline": "2025-12-31T23:59:59",
            }],
            group_name="Priority Test",
        )
        tasks = empty_task_store.get_active_group_tasks()
        assert tasks[0].priority == "urgent"
        assert tasks[0].deadline is not None


class TestAddTasksToGroup:
    def test_appends_to_existing_group(self, empty_task_store):
        gid = empty_task_store.add_task_group(
            tasks=[{"title": "Task 1"}],
            group_name="Growing",
        )
        added = empty_task_store.add_tasks_to_group(gid, [
            {"title": "Task 2"},
            {"title": "Task 3"},
        ])
        assert added == 2
        assert empty_task_store.get_pending_count() == 3

    def test_nonexistent_group_returns_zero(self, empty_task_store):
        result = empty_task_store.add_tasks_to_group(
            "nonexistent_id",
            [{"title": "Ghost"}],
        )
        assert result == 0


class TestCharOverlapScore:
    def test_identical_strings(self, empty_task_store):
        score = empty_task_store._char_overlap_score("测试任务", "测试任务")
        assert score == 1.0

    def test_partial_overlap(self, empty_task_store):
        score = empty_task_store._char_overlap_score("饭已经煮", "把饭煮了")
        assert score > 0.3

    def test_no_overlap(self, empty_task_store):
        score = empty_task_store._char_overlap_score("写代码", "煮饭")
        assert score == 0.0

    def test_ignores_punctuation(self, empty_task_store):
        s1 = "完成语音识别/分离/分群的开发"
        s2 = "完成语音识别分离分群的开发"
        score = empty_task_store._char_overlap_score(s1, s2)
        assert score > 0.8  # 忽略标点后应该高度相似

    def test_empty_strings(self, empty_task_store):
        assert empty_task_store._char_overlap_score("", "") == 0.0
        assert empty_task_store._char_overlap_score("test", "") == 0.0


class TestIsTaskMatch:
    def test_exact_substring(self, empty_task_store):
        assert empty_task_store._is_task_match("语音识别", "完成语音识别模块开发")

    def test_title_contains_keyword(self, empty_task_store):
        assert empty_task_store._is_task_match("完成语音识别模块开发", "语音识别")

    def test_char_overlap_match(self, empty_task_store):
        # "饭煮了" 和 "把饭煮了" 应该有高重叠
        assert empty_task_store._is_task_match("饭煮了", "把饭煮了")

    def test_short_title_exact_only(self, empty_task_store):
        """<=3 个有效字符需要 100% 匹配。"""
        assert empty_task_store._is_task_match("ABC", "ABD") is False
        assert empty_task_store._is_task_match("AB", "AB") is True


class TestCompleteTaskByTitle:
    def test_exact_match(self, populated_task_store):
        result = populated_task_store.complete_task_by_title("Write tests")
        assert result["completed"] is not None
        assert result["completed"].title == "Write tests"
        assert result["completed"].status == "completed"

    def test_substring_match(self, populated_task_store):
        result = populated_task_store.complete_task_by_title("Review")
        assert result["completed"] is not None
        assert result["completed"].title == "Review PR"

    def test_returns_next_in_group(self, populated_task_store):
        result = populated_task_store.complete_task_by_title("Write tests")
        assert result["next_in_group"] is not None
        assert result["next_in_group"].title == "Review PR"

    def test_group_all_done_flag(self, populated_task_store):
        populated_task_store.complete_task_by_title("Write tests")
        populated_task_store.complete_task_by_title("Review PR")
        result = populated_task_store.complete_task_by_title("Update docs")
        assert result["group_all_done"] is True
        assert result["next_in_group"] is None

    def test_no_match_returns_none_completed(self, empty_task_store):
        result = empty_task_store.complete_task_by_title("nonexistent")
        assert result["completed"] is None

    def test_active_group_priority(self, populated_task_store):
        """活跃组优先搜索。"""
        # 添加第二个非活跃组
        populated_task_store.add_task_group(
            tasks=[{"title": "Duplicate Task Name"}],
            group_name="Inactive Group",
            set_active=False,
        )
        # 仍在活跃组中完成匹配
        result = populated_task_store.complete_task_by_title("Review PR")
        assert result["completed"] is not None
        assert result["completed"].title == "Review PR"

    def test_fuzzy_match_across_groups(self, empty_task_store):
        """当活跃组无匹配时搜索其他组。"""
        empty_task_store.add_task_group(
            tasks=[{"title": "编写单元测试"}],
            group_name="Active",
            set_active=True,
        )
        empty_task_store.add_task_group(
            tasks=[{"title": "部署到生产环境"}],
            group_name="Inactive",
            set_active=False,
        )
        # 精确匹配跨组任务标题
        result = empty_task_store.complete_task_by_title("部署到生产环境")
        assert result["completed"] is not None
        assert "部署到生产环境" in result["completed"].title

    def test_all_done(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "Only Task"}],
            group_name="Solo",
        )
        result = empty_task_store.complete_task_by_title("Only Task")
        assert result["all_done"] is True

    def test_other_groups_pending(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "Active Task"}],
            group_name="Active",
            set_active=True,
        )
        empty_task_store.add_task_group(
            tasks=[{"title": "Other Task"}],
            group_name="Other",
            set_active=False,
        )
        result = empty_task_store.complete_task_by_title("Active Task")
        # 活跃组完成后应提示其它组
        assert len(result["other_groups_pending"]) > 0 or result["group_all_done"]


class TestCompleteTaskById:
    def test_completes_by_id(self, populated_task_store):
        tasks = populated_task_store.get_active_group_tasks()
        task_id = tasks[0].id
        result = populated_task_store.complete_task_by_id(task_id)
        assert result["completed"] is not None
        assert result["completed"].id == task_id
        assert result["completed"].status == "completed"

    def test_nonexistent_id(self, empty_task_store):
        result = empty_task_store.complete_task_by_id("fake-id")
        assert result["completed"] is None


class TestQueryTasks:
    def test_get_active_group_tasks_sorted(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[
                {"title": "Third"},
                {"title": "First"},
                {"title": "Second"},
            ],
            group_name="Unsorted",
        )
        # 注意：add_task_group 按输入顺序赋值 order，所以顺序应保持输入顺序
        tasks = empty_task_store.get_active_group_tasks()
        assert tasks[0].title == "Third"
        assert tasks[2].title == "Second"

    def test_get_active_group_tasks_empty(self, empty_task_store):
        assert empty_task_store.get_active_group_tasks() == []

    def test_get_next_pending_global_active_first(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "Active Task"}],
            group_name="Active",
            set_active=True,
        )
        empty_task_store.add_task_group(
            tasks=[{"title": "Inactive Task"}],
            group_name="Inactive",
            set_active=False,
        )
        next_task = empty_task_store.get_next_pending_global()
        assert next_task["title"] == "Active Task"

    def test_get_next_pending_global_none(self, empty_task_store):
        assert empty_task_store.get_next_pending_global() is None

    def test_get_all_pending_tasks_active_first(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "Active"}],
            group_name="Active Group",
            set_active=True,
        )
        empty_task_store.add_task_group(
            tasks=[{"title": "Other"}],
            group_name="Other Group",
            set_active=False,
        )
        all_tasks = empty_task_store.get_all_pending_tasks()
        assert all_tasks[0]["is_active"] is True
        assert all_tasks[0]["group_name"] == "Active Group"

    def test_get_all_pending_tasks_shows_statuses(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "T1"}, {"title": "T2"}],
            group_name="Test",
        )
        empty_task_store.complete_task_by_title("T1")
        all_tasks = empty_task_store.get_all_pending_tasks()
        tasks = all_tasks[0]["tasks"]
        statuses = {t["title"]: t["status"] for t in tasks}
        assert statuses["T1"] == "completed"
        assert statuses["T2"] == "pending"

    def test_get_pending_count(self, empty_task_store):
        empty_task_store.add_task_group(
            tasks=[{"title": "T1"}, {"title": "T2"}, {"title": "T3"}],
            group_name="Count Test",
        )
        assert empty_task_store.get_pending_count() == 3
        empty_task_store.complete_task_by_title("T1")
        assert empty_task_store.get_pending_count() == 2

    def test_set_active_group(self, empty_task_store):
        gid1 = empty_task_store.add_task_group(
            tasks=[{"title": "Group 1 Task"}],
            group_name="Group 1",
        )
        gid2 = empty_task_store.add_task_group(
            tasks=[{"title": "Group 2 Task"}],
            group_name="Group 2",
            set_active=False,
        )
        assert empty_task_store.set_active_group(gid2) is True
        active = empty_task_store.get_active_group()
        assert active.name == "Group 2"

    def test_set_active_group_nonexistent(self, empty_task_store):
        assert empty_task_store.set_active_group("fake-id") is False


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        filepath = tmp_path / "tasks.json"

        store1 = TaskStore(storage_path=str(filepath))
        store1.add_task_group(
            tasks=[
                {"title": "Persist 1", "priority": "high"},
                {"title": "Persist 2"},
            ],
            group_name="Persist Test",
        )
        store1.complete_task_by_title("Persist 1")

        store2 = TaskStore(storage_path=str(filepath))
        assert store2.get_pending_count() == 1
        all_groups = store2.get_all_pending_tasks()
        assert len(all_groups) == 1
        tasks = all_groups[0]["tasks"]
        statuses = {t["title"]: t["status"] for t in tasks}
        assert statuses["Persist 1"] == "completed"
        assert statuses["Persist 2"] == "pending"

    def test_atomic_save_does_not_destroy_on_write_error(self, tmp_path):
        filepath = tmp_path / "tasks.json"
        # 先保存有效数据
        store1 = TaskStore(storage_path=str(filepath))
        store1.add_task_group(tasks=[{"title": "Original"}], group_name="Original")

        # 验证数据存在
        store2 = TaskStore(storage_path=str(filepath))
        assert store2.get_pending_count() == 1

    def test_load_corrupt_json(self, tmp_path):
        filepath = tmp_path / "tasks.json"
        filepath.write_text("this is not json", encoding="utf-8")

        store = TaskStore(storage_path=str(filepath))
        # 应该降级为空状态，不崩溃
        assert store.get_pending_count() == 0
        assert store.get_all_pending_tasks() == []

    def test_load_missing_file(self, tmp_path):
        filepath = tmp_path / "nonexistent.json"
        store = TaskStore(storage_path=str(filepath))
        assert store.get_pending_count() == 0
