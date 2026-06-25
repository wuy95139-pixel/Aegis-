"""
日历与提醒工具
=============
管理提醒的创建、检查、触发和通知。

设计决策：
  - 提醒存储在内存 + 可选的持久化 (JSON 文件)
  - 每 N 分钟检查一次到期提醒 (通过 check_interval 配置)
  - 通知方式可扩展 (console / email / webhook / 短信)

可扩展点：
  - 接入真实日历系统: Google Calendar / Outlook API
  - 高级触发条件: 支持 cron 表达式解析
  - 邮件监测: 通过 IMAP 检查未回复邮件
"""

import logging
import os
import uuid
import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Callable

from croniter import croniter
from src.models.schemas import Reminder, ReminderType
from src.core.tools._notification_manager import NotificationManager
from src.core.tools._scheduled_task_manager import ScheduledTaskManager
from src.core.tools._workload_analyzer import (
    analyze_workload as _analyze_workload_impl,
    PRIORITY_HOURS,
    OVERLOAD_THRESHOLD,
    WARNING_THRESHOLD,
    BUSY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class CalendarTool:
    """
    日历提醒工具

    使用示例:
        cal = CalendarTool()
        rid = cal.add_reminder(
            title="周会",
            trigger_time=datetime(2024, 6, 1, 14, 0),
            notify_method=["console", "email"],
        )
        due = cal.check_due_reminders()  # 检查到期提醒
    """

    def __init__(self, storage_path: Optional[str] = None, poll_interval: float = 30.0):
        """
        Args:
            storage_path: 提醒持久化文件路径 (JSON)，None 则仅内存
            poll_interval: 后台轮询间隔（秒），默认30秒
        """
        self._reminders: dict[str, Reminder] = {}
        self._lock = threading.Lock()
        self.storage_path = storage_path

        # 通知渠道（委托给 NotificationManager）
        self._notifications = NotificationManager()

        # Windows 计划任务（委托给 ScheduledTaskManager）
        self._scheduled_tasks = ScheduledTaskManager(
            register_app_id=self._register_windows_app_id,
        )

        # 后台调度器
        self._poll_interval = poll_interval
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._start_scheduler()

        # 从文件恢复提醒
        if storage_path and Path(storage_path).exists():
            self._load()

        # 清理不再存在的提醒对应的孤立计划任务
        self._scheduled_tasks.cleanup_stale(set(self._reminders.keys()))

    def add_reminder(
        self,
        title: str,
        description: Optional[str] = None,
        trigger_time: Optional[datetime] = None,
        trigger_event: Optional[str] = None,
        cron_expression: Optional[str] = None,
        notify_method: Optional[List[str]] = None,
    ) -> str:
        """
        添加新提醒

        Args:
            title: 提醒标题
            description: 详细描述
            trigger_time: 时间触发时间点
            trigger_event: 事件触发描述
            cron_expression: 循环表达式
            notify_method: 通知方式列表

        Returns:
            提醒 ID
        """
        # 确定提醒类型
        if cron_expression:
            rtype = ReminderType.RECURRING
        elif trigger_event:
            rtype = ReminderType.EVENT_TRIGGERED
        else:
            rtype = ReminderType.TIME_TRIGGERED

        reminder = Reminder(
            id=str(uuid.uuid4())[:8],
            type=rtype,
            title=title,
            description=description,
            trigger_time=trigger_time,
            trigger_event=trigger_event,
            cron_expression=cron_expression,
            notify_method=notify_method or ["sound", "toast", "console"],
        )

        with self._lock:
            self._reminders[reminder.id] = reminder
            if self.storage_path:
                self._save()

        # 创建 Windows 计划任务以支持离线提醒
        if trigger_time and rtype == ReminderType.TIME_TRIGGERED:
            self._scheduled_tasks.create(reminder)

        logger.info(f"Added reminder: {reminder.id} - {title}")
        return reminder.id

    def check_due_reminders(self) -> List[Reminder]:
        """
        检查所有到期的提醒

        行为：
          - 时间触发：到期后触发通知，不立即停用。用户确认后才停用。
          - 未确认时：每隔 snooze_minutes 再次提醒，直到 fire_count >= max_fires
          - 循环提醒：按 cron 表达式触发，每次触发都需要确认

        Returns:
            到期提醒列表
        """
        now = datetime.now()
        due: List[Reminder] = []
        save_needed = False

        with self._lock:
            for reminder in self._reminders.values():
                if not reminder.is_active:
                    continue

                if reminder.type == ReminderType.TIME_TRIGGERED and reminder.trigger_time:
                    if not self._is_past(reminder.trigger_time, now):
                        continue

                    if not reminder.acknowledged:
                        if reminder.fire_count == 0:
                            due.append(reminder)
                            reminder.last_triggered = now
                            reminder.fire_count = 1
                            save_needed = True
                            logger.info(f"Reminder fired (1st): {reminder.title}")
                        elif reminder.last_triggered and self._is_past(
                            reminder.last_triggered + timedelta(minutes=reminder.snooze_minutes), now
                        ):
                            reminder.fire_count += 1
                            due.append(reminder)
                            reminder.last_triggered = now
                            save_needed = True
                            logger.info(
                                f"Reminder re-fired ({reminder.fire_count}/{reminder.max_fires}): {reminder.title}"
                            )

                        if reminder.fire_count >= reminder.max_fires:
                            reminder.is_active = False
                            logger.info(f"Reminder max fires reached, deactivating: {reminder.title}")

                elif reminder.type == ReminderType.EVENT_TRIGGERED:
                    pass

                elif reminder.type == ReminderType.RECURRING and reminder.cron_expression:
                    try:
                        base = reminder.last_triggered or reminder.created_at
                        cron = croniter(reminder.cron_expression, base)
                        next_fire = cron.get_next(datetime)
                        if self._is_past(next_fire, now):
                            due.append(reminder)
                            reminder.last_triggered = now
                            reminder.fire_count += 1
                            reminder.acknowledged = False
                            save_needed = True
                            logger.info(
                                f"Recurring reminder fired: {reminder.title} "
                                f"(cron={reminder.cron_expression}, next={cron.get_next(datetime)})"
                            )
                    except (ValueError, KeyError) as e:
                        logger.error(f"Invalid cron expression for reminder {reminder.id}: {e}")

            if save_needed and self.storage_path:
                self._save()

        # 在锁外发送通知（避免通知阻塞其他线程）
        for reminder in due:
            self._trigger_notification(reminder)

        if due:
            logger.info(f"Found {len(due)} due reminders")

        return due

    def acknowledge_reminder(self, reminder_id: str) -> bool:
        """确认提醒 — 用户点击"确认"，删除该提醒"""
        with self._lock:
            if reminder_id in self._reminders:
                r = self._reminders.pop(reminder_id)
                logger.info(f"Reminder acknowledged and deleted: {r.title}")
                if self.storage_path:
                    self._save()
                self._scheduled_tasks.delete(reminder_id)
                return True
        return False

    def snooze_reminder(self, reminder_id: str, minutes: int = 5) -> bool:
        """延迟提醒 — 用户点击"稍后提醒"，过几分钟再响"""
        with self._lock:
            if reminder_id in self._reminders:
                r = self._reminders[reminder_id]
                r.snooze_minutes = minutes
                r.acknowledged = False
                r.last_triggered = datetime.now()
                r.fire_count = max(0, r.fire_count - 1)
                # 更新计划任务到新的延迟时间
                r.trigger_time = datetime.now() + timedelta(minutes=minutes)
                logger.info(f"Reminder snoozed for {minutes}min: {r.title}")
                if self.storage_path:
                    self._save()
                self._scheduled_tasks.create(r)
                return True
        return False

    def list_reminders(self, active_only: bool = False) -> List[Reminder]:
        """列出所有提醒"""
        with self._lock:
            if active_only:
                return [r for r in self._reminders.values() if r.is_active]
            return list(self._reminders.values())

    def cancel_reminder(self, reminder_id: str) -> bool:
        """取消提醒"""
        with self._lock:
            if reminder_id in self._reminders:
                self._reminders[reminder_id].is_active = False
                logger.info(f"Cancelled reminder: {reminder_id}")
                if self.storage_path:
                    self._save()
                self._scheduled_tasks.delete(reminder_id)
                return True
        return False

    def delete_scheduled_task(self, reminder_id: str) -> None:
        """删除关联的 Windows 计划任务（公开接口）"""
        self._scheduled_tasks.delete(reminder_id)

    def clear_all_reminders(self) -> int:
        """清除所有已过期/已确认/不活跃的提醒，返回清除数量"""
        removed = 0
        with self._lock:
            to_remove = []
            now = datetime.now()
            for rid, r in list(self._reminders.items()):
                if r.acknowledged or not r.is_active:
                    to_remove.append(rid)
                elif r.type == ReminderType.TIME_TRIGGERED and r.trigger_time and r.trigger_time < now - timedelta(hours=1):
                    to_remove.append(rid)
                elif r.type == ReminderType.TIME_TRIGGERED and not r.trigger_time:
                    to_remove.append(rid)
            for rid in to_remove:
                self._reminders.pop(rid, None)
                self._scheduled_tasks.delete(rid)
                removed += 1
            if removed > 0 and self.storage_path:
                self._save()
        return removed

    # ===================== 负荷分析（委托给 _workload_analyzer） =====================

    PRIORITY_HOURS = PRIORITY_HOURS
    OVERLOAD_THRESHOLD = OVERLOAD_THRESHOLD
    WARNING_THRESHOLD = WARNING_THRESHOLD
    BUSY_THRESHOLD = BUSY_THRESHOLD

    def analyze_workload(self, todos: list, target_date: Optional[datetime] = None) -> dict:
        """Analyze task load for a date, with rescheduling suggestions."""
        return _analyze_workload_impl(todos, target_date)

    def _todos_from_reminders(self) -> list:
        """从活跃提醒反向构建简易 TodoItem 字典列表（用于无外部 todos 时的后备）"""
        todos = []
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        for r in self.list_reminders(active_only=True):
            if not r.trigger_time:
                continue
            # 只取今明两天的提醒
            if today <= r.trigger_time < tomorrow + timedelta(days=1):
                # 推断优先级：标题含"紧急"/"重要"为 high，否则 medium
                title_lower = r.title.lower()
                if any(kw in title_lower for kw in ["紧急", "urgent", "重要", "critical", "important", "p0", "p1"]):
                    priority = "high"
                elif any(kw in title_lower for kw in ["普通", "normal", "低优", "p2", "p3"]):
                    priority = "low"
                else:
                    priority = "medium"

                todos.append({
                    "title": r.title,
                    "priority": priority,
                    "deadline": r.trigger_time,
                })

        # 按优先级排序
        order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        todos.sort(key=lambda t: order.get(t.get("priority", "medium"), 2))
        return todos

    def check_daily_workload(self) -> dict:
        """便捷方法：基于已有提醒分析今日负荷"""
        todos = self._todos_from_reminders()
        return self.analyze_workload(todos)

    def register_notify_handler(self, method: str, handler: Callable):
        """Register a custom notification channel (e.g., email, webhook, SMS)."""
        self._notifications.register_handler(method, handler)

    # ==================== 后台调度器 ====================

    def _start_scheduler(self):
        """启动后台轮询线程，定期检查到期提醒"""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return

        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="reminder-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"Reminder scheduler started (poll every {self._poll_interval}s)")

    def stop_scheduler(self):
        """停止后台调度器"""
        self._scheduler_stop.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            logger.info("Reminder scheduler stopped")

    def _scheduler_loop(self):
        """后台轮询循环，带指数退避"""
        backoff = self._poll_interval
        max_backoff = min(self._poll_interval * 16, 600)  # 最多 10 分钟
        consecutive_failures = 0

        while not self._scheduler_stop.is_set():
            try:
                self.check_due_reminders()
                self._process_signal_files()
                # 成功则重置退避
                if consecutive_failures > 0:
                    logger.info(f"Scheduler recovered after {consecutive_failures} failures")
                consecutive_failures = 0
                backoff = self._poll_interval
            except Exception as e:
                consecutive_failures += 1
                backoff = min(self._poll_interval * (2 ** min(consecutive_failures - 1, 4)), max_backoff)
                logger.error(
                    f"Scheduler check failed (#{consecutive_failures}): {e}. "
                    f"Next retry in {backoff:.0f}s"
                )
            self._scheduler_stop.wait(backoff)

    @staticmethod
    def _is_past(trigger_time: datetime, now: datetime) -> bool:
        """安全比较时间是否已过，自动处理 native/aware datetime 混用"""
        try:
            return trigger_time <= now
        except TypeError:
            # 如果一个 naive 一个 aware，降级比较
            ts_trigger = trigger_time.timestamp()
            ts_now = now.timestamp()
            return ts_trigger <= ts_now

    # --- 内部方法 ---

    def _trigger_notification(self, reminder: Reminder):
        self._notifications.trigger(reminder)

    def send_simple_toast(self, title: str, body: str = ""):
        self._notifications.send_simple_toast(title, body)

    # Backward-compat: notification channel methods delegate to NotificationManager
    def _notify_sound(self, reminder: Reminder):
        self._notifications._notify_sound(reminder)

    def _notify_toast(self, reminder: Reminder):
        self._notifications._notify_toast(reminder)

    def _notify_console(self, reminder: Reminder):
        self._notifications._notify_console(reminder)

    @staticmethod
    def _register_windows_app_id():
        NotificationManager._register_windows_app_id()

    def _process_signal_files(self):
        """
        检查并处理信号文件（由 toast_handler.py 在按钮点击时写入）。

        信号文件格式: data/signals/{reminder_id}.json
          {"action": "confirm|snooze", "reminder_id": "...", "timestamp": "..."}
        """
        signal_dir = Path("./data/signals")
        if not signal_dir.exists():
            return

        processed = 0
        for signal_file in list(signal_dir.glob("*.json")):
            try:
                with open(signal_file, "r", encoding="utf-8") as f:
                    signal = json.load(f)

                action = signal.get("action")
                reminder_id = signal.get("reminder_id")

                if action == "confirm":
                    self.acknowledge_reminder(reminder_id)
                    logger.info(f"Toast confirm: reminder {reminder_id} acknowledged and removed")
                elif action == "snooze":
                    self.snooze_reminder(reminder_id, minutes=5)
                    logger.info(f"Toast snooze: reminder {reminder_id} delayed by 5min")

                # 删除已处理的信号文件
                signal_file.unlink()
                processed += 1
            except Exception as e:
                logger.debug(f"Signal file processing failed: {e}")

        if processed:
            logger.debug(f"Processed {processed} toast signal(s)")

    # ==================== 测试用的静态入口 ====================

    @staticmethod
    def _sanitize_for_powershell_xml(value: str) -> str:
        return ScheduledTaskManager.sanitize_for_powershell_xml(value)

    @staticmethod
    def _sanitize_task_name(name: str) -> str:
        return ScheduledTaskManager.sanitize_task_name(name)

    def _save(self):
        """持久化到 JSON 文件"""
        if not self.storage_path:
            return
        data = []
        for r in self._reminders.values():
            data.append({
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "trigger_time": r.trigger_time.isoformat() if r.trigger_time else None,
                "trigger_event": r.trigger_event,
                "cron_expression": r.cron_expression,
                "notify_method": r.notify_method,
                "is_active": r.is_active,
                "last_triggered": r.last_triggered.isoformat() if r.last_triggered else None,
                "type": r.type.value,
                "acknowledged": r.acknowledged,
                "snooze_minutes": r.snooze_minutes,
                "fire_count": r.fire_count,
                "max_fires": r.max_fires,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="reminders_", dir=Path(self.storage_path).parent
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
        """从 JSON 文件恢复提醒（损坏时降级为空状态，防止启动崩溃）"""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                f"Failed to load reminders from {self.storage_path}: {e}. "
                "Starting with empty reminder store."
            )
            return

        loaded = 0
        for item in data:
            try:
                rtype_str = item.get("type", "time_triggered")
                try:
                    rtype = ReminderType(rtype_str)
                except ValueError:
                    rtype = ReminderType.TIME_TRIGGERED
                reminder = Reminder(
                    id=item["id"],
                    type=rtype,
                    title=item.get("title", "未命名提醒"),
                    description=item.get("description"),
                    trigger_time=datetime.fromisoformat(item["trigger_time"]) if item.get("trigger_time") else None,
                    trigger_event=item.get("trigger_event"),
                    cron_expression=item.get("cron_expression"),
                    notify_method=item.get("notify_method", ["console"]),
                    is_active=item.get("is_active", True),
                    last_triggered=datetime.fromisoformat(item["last_triggered"]) if item.get("last_triggered") else None,
                    acknowledged=item.get("acknowledged", False),
                    snooze_minutes=item.get("snooze_minutes", 5),
                    fire_count=item.get("fire_count", 0),
                    max_fires=item.get("max_fires", 5),
                    created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else None,
                )
                self._reminders[reminder.id] = reminder
                loaded += 1
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Skipping corrupted reminder entry {item.get('id', 'unknown')}: {e}"
                )
        logger.info(f"Loaded {loaded} reminders from {self.storage_path}")
