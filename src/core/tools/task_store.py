"""
任务存储模块 (TaskStore)
========================
独立的任务管理系统，与记忆系统和提醒系统解耦。

功能：
  - 支持任务分组（同一批次添加的任务自动归为一组）
  - 组内任务保持顺序
  - 完成任务后自动返回同组下一个任务
  - 支持持久化到 JSON 文件

设计原则：
  - 待办事件模块独立于记忆系统，只管理待办任务
  - 相同时间段、相同类型的任务放在一起，不混合
  - 查询"接下来做什么"时，优先返回同组未完成任务
"""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """单个任务项"""
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending / completed / cancelled
    order: int = 0
    priority: str = "medium"  # low / medium / high / urgent
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "order": self.order,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskItem":
        return cls(
            id=d["id"],
            title=d["title"],
            description=d.get("description", ""),
            status=d.get("status", "pending"),
            order=d.get("order", 0),
            priority=d.get("priority", "medium"),
            deadline=datetime.fromisoformat(d["deadline"]) if d.get("deadline") else None,
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            tags=d.get("tags", []),
        )


@dataclass
class TaskGroup:
    """任务分组 — 同一批次/同一上下文的任务集合"""
    id: str
    name: str  # 分组名称，如"当前项目开发任务"、"日常事务"
    tasks: List[TaskItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    context: str = ""  # 上下文描述，如用户说"帮我记下来"时的原始对话摘要

    def get_pending_tasks(self) -> List[TaskItem]:
        """获取未完成的任务（按 order 排序）"""
        return sorted(
            [t for t in self.tasks if t.status == "pending"],
            key=lambda t: t.order,
        )

    def get_completed_tasks(self) -> List[TaskItem]:
        """获取已完成的任务"""
        return [t for t in self.tasks if t.status == "completed"]

    def get_next_pending(self, after_order: int = -1) -> Optional[TaskItem]:
        """获取下一个待完成任务（在指定 order 之后）"""
        pending = self.get_pending_tasks()
        for task in pending:
            if task.order > after_order:
                return task
        return None

    def is_all_done(self) -> bool:
        """该组所有任务是否都已完成"""
        return all(t.status in ("completed", "cancelled") for t in self.tasks)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskGroup":
        return cls(
            id=d["id"],
            name=d["name"],
            tasks=[TaskItem.from_dict(t) for t in d.get("tasks", [])],
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(),
            context=d.get("context", ""),
        )


class TaskStore:
    """
    任务存储 — 管理所有任务分组

    使用示例:
        store = TaskStore(storage_path="./data/tasks.json")

        # 批量添加任务（同一上下文自动成组）
        group_id = store.add_task_group(
            tasks=[
                {"title": "完成语言识别/分离/分群板块的开发"},
                {"title": "搭建agent的基础框架"},
                {"title": "添加agent各个板块的功能代码内容"},
            ],
            group_name="当前项目开发任务",
            context="用户说：我现在有几件事需要做——第一，完成语言识别..."
        )

        # 完成一个任务，获取下一个
        next_task = store.complete_task_by_title("完成语言识别/分离/分群板块的开发")
        # → 返回同组的下一个任务："搭建agent的基础框架"

        # 获取当前活跃组的任务
        active_tasks = store.get_active_group_tasks()

        # 获取其它组的任务
        other_tasks = store.get_other_groups_tasks()
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 持久化文件路径 (JSON)，None 则仅内存
        """
        self._groups: Dict[str, TaskGroup] = {}
        self._active_group_id: Optional[str] = None
        self._lock = threading.RLock()  # 可重入锁，防止 get_all_pending_tasks → get_active_group 死锁
        self.storage_path = storage_path

        if storage_path and Path(storage_path).exists():
            self._load()

    # ==================== 添加任务 ====================

    def add_task_group(
        self,
        tasks: List[Dict[str, Any]],
        group_name: str = "",
        context: str = "",
        set_active: bool = True,
    ) -> str:
        """
        批量添加任务并创建分组

        Args:
            tasks: 任务列表 [{"title": "...", "description": "...", "priority": "...", "deadline": "..."}, ...]
            group_name: 分组名称，为空则自动生成
            context: 上下文描述（用户的原始输入摘要）
            set_active: 是否设为当前活跃组

        Returns:
            分组 ID
        """
        group_id = str(uuid.uuid4())[:8]
        if not group_name:
            group_name = f"任务组 {group_id}"

        task_items = []
        for i, t in enumerate(tasks):
            task = TaskItem(
                id=str(uuid.uuid4())[:8],
                title=t.get("title", ""),
                description=t.get("description", ""),
                order=i,
                priority=t.get("priority", "medium"),
                deadline=self._parse_deadline(t.get("deadline")),
                tags=t.get("tags", []),
            )
            task_items.append(task)

        group = TaskGroup(
            id=group_id,
            name=group_name,
            tasks=task_items,
            context=context,
        )

        with self._lock:
            self._groups[group_id] = group
            if set_active:
                self._active_group_id = group_id
            self._save()

        logger.info(f"Created task group '{group_name}' with {len(task_items)} tasks (id={group_id})")
        return group_id

    def add_tasks_to_group(
        self,
        group_id: str,
        tasks: List[Dict[str, Any]],
    ) -> int:
        """
        向已有分组追加任务

        Returns:
            添加的任务数量
        """
        group = self._groups.get(group_id)
        if not group:
            logger.warning(f"Task group not found: {group_id}")
            return 0

        with self._lock:
            max_order = max((t.order for t in group.tasks), default=-1)
            for i, t in enumerate(tasks):
                task = TaskItem(
                    id=str(uuid.uuid4())[:8],
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    order=max_order + 1 + i,
                    priority=t.get("priority", "medium"),
                    deadline=self._parse_deadline(t.get("deadline")),
                    tags=t.get("tags", []),
                )
                group.tasks.append(task)
            self._save()

        return len(tasks)

    # ==================== 完成任务 ====================

    def _char_overlap_score(self, a: str, b: str) -> float:
        """
        计算两个中文字符串的字符重叠度。

        用于模糊匹配用户表述（"饭已经煮"）和存储的任务标题（"把饭煮了"）。
        只考虑中文字符（Unicode CJK 范围），忽略标点和空白。

        Returns:
            0.0~1.0 的重叠分数（相对较短字符串的字符集合）
        """
        def extract_chars(s: str) -> set:
            # CJK + 字母 + 数字（排除标点空白）
            result = set()
            for c in s:
                if c.isspace() or c in '，,。.！!？?、：:；;（）()【】[]《》〈〉""''…—\-/\\|·':
                    continue
                if '一' <= c <= '鿿' or c.isalnum():
                    result.add(c)
            return result

        chars_a = extract_chars(a)
        chars_b = extract_chars(b)

        if not chars_a or not chars_b:
            return 0.0

        overlap = chars_a & chars_b
        min_len = min(len(chars_a), len(chars_b))
        if min_len == 0:
            return 0.0
        return len(overlap) / min_len

    def _is_task_match(self, keyword: str, title: str) -> bool:
        """判断关键词是否匹配任务标题（子串 + 字符重叠双策略）"""
        if keyword in title or title in keyword:
            return True
        score = self._char_overlap_score(keyword, title)
        # 用 _char_overlap_score 内部的字符提取逻辑计算字符数
        min_chars = min(self._count_significant_chars(keyword),
                        self._count_significant_chars(title))
        if min_chars <= 3:
            threshold = 1.0   # 短标题必须全部字符匹配（避免"任务A"误匹配"任务B"）
        elif min_chars <= 5:
            threshold = 0.6
        else:
            threshold = 0.5
        return score >= threshold

    def _count_significant_chars(self, s: str) -> int:
        """统计有意义的字符数（CJK + 字母数字，排除标点）"""
        count = 0
        for c in s:
            if c.isspace() or c in '，,。.！!？?、：:；;（）()【】[]《》〈〉""''…—\-/\\|·':
                continue
            if '一' <= c <= '鿿' or c.isalnum():
                count += 1
        return count

    def complete_task_by_title(self, title_keyword: str) -> Dict[str, Any]:
        """
        通过标题关键词完成一个任务（模糊匹配）

        优先在当前活跃组中查找，找不到则搜索所有组。

        Returns:
            {
                "completed": TaskItem | None,
                "group": TaskGroup | None,
                "next_in_group": TaskItem | None,     # 同组下一个待办
                "group_all_done": bool,                # 同组是否全部完成
                "other_groups_pending": [...],          # 其它组待办摘要
                "all_done": bool,                       # 所有任务是否全部完成
            }
        """
        result = {
            "completed": None,
            "group": None,
            "next_in_group": None,
            "group_all_done": False,
            "other_groups_pending": [],
            "all_done": False,
        }

        with self._lock:
            matched_task = None
            matched_group = None

            # 1. 先搜索活跃组
            if self._active_group_id and self._active_group_id in self._groups:
                group = self._groups[self._active_group_id]
                for task in group.tasks:
                    if task.status == "pending" and self._is_task_match(title_keyword, task.title):
                        matched_task = task
                        matched_group = group
                        break

            # 2. 活跃组没找到，搜索所有组
            if not matched_task:
                for group in self._groups.values():
                    if group.id == self._active_group_id:
                        continue
                    for task in group.tasks:
                        if task.status == "pending" and self._is_task_match(title_keyword, task.title):
                            matched_task = task
                            matched_group = group
                            break
                    if matched_task:
                        break

            if not matched_task:
                logger.debug(f"No pending task matched: '{title_keyword}'")
                return result

            # 标记完成
            matched_task.status = "completed"
            matched_task.completed_at = datetime.now()
            result["completed"] = matched_task
            result["group"] = matched_group

            # 查找同组下一个待办
            next_task = matched_group.get_next_pending(matched_task.order)
            result["next_in_group"] = next_task
            result["group_all_done"] = matched_group.is_all_done()

            # 收集其它组的待办
            for gid, group in self._groups.items():
                if gid == matched_group.id:
                    continue
                pending = group.get_pending_tasks()
                if pending:
                    result["other_groups_pending"].append({
                        "group_id": gid,
                        "group_name": group.name,
                        "pending_count": len(pending),
                        "tasks": [{"title": t.title, "order": t.order} for t in pending],
                    })

            # 检查是否所有任务都完成
            result["all_done"] = all(g.is_all_done() for g in self._groups.values())

            logger.info(f"Task completed: '{matched_task.title}' in group '{matched_group.name}'")
            self._save()
        return result

    def complete_task_by_id(self, task_id: str) -> Dict[str, Any]:
        """通过 ID 完成指定任务"""
        with self._lock:
            for group in self._groups.values():
                for task in group.tasks:
                    if task.id == task_id and task.status == "pending":
                        task.status = "completed"
                        task.completed_at = datetime.now()
                        self._save()
                        return {
                            "completed": task,
                            "group": group,
                            "next_in_group": group.get_next_pending(task.order),
                            "group_all_done": group.is_all_done(),
                        }
        return {"completed": None}

    # ==================== 查询任务 ====================

    def get_active_group(self) -> Optional[TaskGroup]:
        """获取当前活跃的任务组"""
        with self._lock:
            if self._active_group_id and self._active_group_id in self._groups:
                return self._groups[self._active_group_id]
        return None

    def get_next_pending_global(self) -> Optional[Dict[str, Any]]:
        """获取全局第一个待办任务（活跃组优先，按 order 排序）"""
        with self._lock:
            # 活跃组优先
            if self._active_group_id and self._active_group_id in self._groups:
                group = self._groups[self._active_group_id]
                for task in sorted(group.tasks, key=lambda t: t.order):
                    if task.status == "pending":
                        return {"id": task.id, "title": task.title, "group_name": group.name}

            # 搜索所有组
            for group in self._groups.values():
                if group.id == self._active_group_id:
                    continue
                for task in sorted(group.tasks, key=lambda t: t.order):
                    if task.status == "pending":
                        return {"id": task.id, "title": task.title, "group_name": group.name}
        return None

    def get_active_group_tasks(self) -> List[TaskItem]:
        """获取当前活跃组的所有任务（保持顺序）"""
        group = self.get_active_group()
        if not group:
            return []
        return sorted(group.tasks, key=lambda t: t.order)

    def get_other_groups_tasks(self) -> List[Dict[str, Any]]:
        """获取其它组（非活跃组）的待办摘要"""
        with self._lock:
            result = []
            for gid, group in self._groups.items():
                if gid == self._active_group_id:
                    continue
                pending = group.get_pending_tasks()
                if pending:
                    result.append({
                        "group_id": gid,
                        "group_name": group.name,
                        "pending_count": len(pending),
                        "tasks": [{"id": t.id, "title": t.title, "status": t.status, "order": t.order} for t in group.tasks],
                    })
        return result

    def get_all_pending_tasks(self) -> List[Dict[str, Any]]:
        """获取所有待办任务（分组）"""
        with self._lock:
            result = []
            active = self.get_active_group()
            if active:
                result.append({
                    "group_id": active.id,
                    "group_name": active.name,
                    "is_active": True,
                    "tasks": [{"id": t.id, "title": t.title, "status": t.status, "order": t.order} for t in sorted(active.tasks, key=lambda x: x.order)],
                })
            for gid, group in self._groups.items():
                if gid == self._active_group_id:
                    continue
                result.append({
                    "group_id": gid,
                    "group_name": group.name,
                    "is_active": False,
                    "tasks": [{"id": t.id, "title": t.title, "status": t.status, "order": t.order} for t in sorted(group.tasks, key=lambda x: x.order)],
                })
        return result

    def set_active_group(self, group_id: str) -> bool:
        """切换活跃组"""
        with self._lock:
            if group_id in self._groups:
                self._active_group_id = group_id
                logger.info(f"Switched active group to: {self._groups[group_id].name}")
                return True
        return False

    def get_pending_count(self) -> int:
        """获取所有待办任务总数"""
        with self._lock:
            count = 0
            for group in self._groups.values():
                count += len(group.get_pending_tasks())
        return count

    # ==================== 持久化 ====================

    def _save(self):
        """原子保存到 JSON 文件（先写临时文件，再 rename）"""
        if not self.storage_path:
            return
        data = {
            "active_group_id": self._active_group_id,
            "groups": [g.to_dict() for g in self._groups.values()],
            "updated_at": datetime.now().isoformat(),
        }
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="tasks_", dir=Path(self.storage_path).parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.storage_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load(self):
        """从 JSON 文件恢复（损坏时降级为空状态，防止启动崩溃）"""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._active_group_id = data.get("active_group_id")
            for gd in data.get("groups", []):
                group = TaskGroup.from_dict(gd)
                self._groups[group.id] = group
            logger.info(f"Loaded {len(self._groups)} task groups from {self.storage_path}")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(
                f"Failed to load tasks from {self.storage_path}: {e}. "
                "Starting with empty task store."
            )
            self._active_group_id = None
            self._groups = {}

    def _parse_deadline(self, deadline_str: Optional[str]) -> Optional[datetime]:
        """解析截止时间字符串"""
        if not deadline_str:
            return None
        try:
            return datetime.fromisoformat(deadline_str)
        except (ValueError, TypeError):
            return None
