"""
智能提醒代理 (ReminderAgent)
===========================
职责：
  1. 按用户要求设置时间触发或事件触发的提醒
  2. 自动检查到期提醒并通知用户
  3. 监测未回复的重要邮件/消息，主动建议跟进

协作关系：
  输入: 提醒设置请求 or 待跟进事项 (来自 TaskDispatcherAgent)
  输出: 提醒通知 → 用户

可扩展点：
  - 邮件集成：通过 IMAP 监测未回复邮件
  - 消息集成：读取 Slack/微信/钉钉 的未读消息
  - 智能跟进建议：基于对话历史判断哪些消息需要跟进
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.core.agents.base import BaseAgent
from src.core.tools.calendar_tools import CalendarTool
from src.models.schemas import Reminder, ReminderType, TodoItem

logger = logging.getLogger(__name__)


class ReminderAgent(BaseAgent):
    """智能提醒代理 — 设置和管理提醒"""

    role = "智能提醒专家"
    goal = "精准设置提醒，主动监测待跟进事项，确保用户不错过任何重要事务"
    backstory = """
你是一位细心的提醒专家，像一位可靠的私人秘书。
你需要：
- 理解用户的提醒需求，设置合适的触发条件
- 定时检查提醒，及时通知用户
- 主动监测未回复的重要邮件和消息
- 对于长期未处理的任务，自动升级提醒频率
- 区分紧急和常规提醒，避免过度打扰

你遵循的原则：
1. 重要事项多次提醒，但不过度骚扰
2. 智能降噪：不重要的提醒可以合并
3. 尊重用户的工作时间，避免非工作时间打扰
"""

    def __init__(self, llm, memory=None, config=None):
        super().__init__(
            name="reminder_agent",
            llm=llm,
            memory=memory,
            tools=[],
            config=config,
        )
        import os as _os
        data_dir = _os.environ.get("AEGIS_DATA_DIR", "./data")
        self.calendar = CalendarTool(storage_path=str(Path(data_dir) / "reminders.json"))

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行提醒相关操作

        task_input 结构:
          {
            "operation": "set" | "check" | "followup" | "list" | "cancel",
            "title": "提醒标题",
            "trigger_time": "2024-06-01T14:00:00",  # ISO 格式
            "trigger_event": "收到张三的回复",
            "cron_expression": "0 9 * * 1-5",
            "notify_method": ["console"],
          }

        Returns:
          操作结果
        """
        operation = task_input.get("operation", "check")

        if operation == "set":
            return self._handle_set_reminder(task_input)
        elif operation == "check":
            return self._handle_check_reminders(task_input)
        elif operation == "followup":
            return self._handle_followup_check(task_input)
        elif operation == "list":
            return self._handle_list_reminders(task_input)
        elif operation == "cancel":
            return self._handle_cancel_reminder(task_input)
        elif operation == "workload":
            return self._handle_workload_check(task_input)
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}

    def receive_message(self, message):
        """响应其他 Agent 发来的事件"""
        from src.models.schemas import AgentMessage
        event = message.payload.get("event", "")
        if event == "memory_updated":
            logger.info(
                f"[reminder_agent] 收到 memory_updated 事件: "
                f"source={message.payload.get('source')}, "
                f"tags={message.payload.get('tags')}"
            )
            # 可扩展：如果记忆包含截止日期，自动创建提醒
        return super().receive_message(message)

    def _handle_set_reminder(self, task_input: Dict) -> Dict:
        """设置新提醒"""
        title = task_input.get("title", "未命名提醒")
        description = task_input.get("description", "")
        notify_method = task_input.get("notify_method", ["console"])

        # 解析触发时间
        trigger_time = None
        if task_input.get("trigger_time"):
            try:
                trigger_time = datetime.fromisoformat(task_input["trigger_time"])
            except (ValueError, TypeError):
                pass

        # 创建提醒
        reminder_id = self.calendar.add_reminder(
            title=title,
            description=description,
            trigger_time=trigger_time,
            trigger_event=task_input.get("trigger_event"),
            cron_expression=task_input.get("cron_expression"),
            notify_method=notify_method,
        )

        return {
            "status": "success",
            "reminder_id": reminder_id,
            "message": f"已设置提醒: {title}",
        }

    def _handle_check_reminders(self, task_input: Dict) -> Dict:
        """检查到期提醒"""
        due_reminders = self.calendar.check_due_reminders()

        return {
            "status": "success",
            "due_reminders": due_reminders,
            "count": len(due_reminders),
            "message": f"有 {len(due_reminders)} 个到期提醒" if due_reminders else "暂无到期提醒",
        }

    def _handle_followup_check(self, task_input: Dict) -> Dict:
        """
        检查需要跟进的事项

        监测逻辑：
          - 检查超过 N 天未回复的重要邮件
          - 检查过期的待办事项
          - 智能判断哪些沟通需要主动跟进

        可扩展点：接入真实邮件/消息系统
        """
        # TODO: 实现邮件/消息监测
        # 1. IMAP 连接检查未回复邮件
        # 2. 使用 LLM 判断邮件重要性
        # 3. 对重要未回复邮件建议跟进

        suggestions = []

        # 检查过期的提醒任务
        all_reminders = self.calendar.list_reminders(active_only=True)
        now = datetime.now()
        for r in all_reminders:
            if r.trigger_time and r.trigger_time < now - timedelta(days=1):
                suggestions.append({
                    "type": "overdue_reminder",
                    "id": r.id,
                    "title": r.title,
                    "overdue_since": r.trigger_time.isoformat(),
                })

        return {
            "status": "success",
            "followup_suggestions": suggestions,
            "count": len(suggestions),
            "message": f"发现 {len(suggestions)} 个需要跟进的事项",
        }

    def _handle_workload_check(self, task_input: Dict) -> Dict:
        """
        负荷感知检查 — 分析当日任务总量，超载时给出排期建议。

        task_input 可选字段:
          - todos: 外部传入的待办列表（若未提供则从已有提醒推断）
          - target_date: 要检查的日期 (ISO 字符串)，默认今天
        """
        target_date = None
        if task_input.get("target_date"):
            try:
                target_date = datetime.fromisoformat(task_input["target_date"])
            except (ValueError, TypeError):
                pass

        # 优先使用外部传入的 todos，否则从提醒推断
        todos = task_input.get("todos")
        if todos:
            result = self.calendar.analyze_workload(todos, target_date)
        else:
            result = self.calendar.check_daily_workload()

        status = result.get("status", "clear")
        status_emoji = {
            "overload": "🔴", "warning": "🟡", "busy": "🟢", "clear": "✅"
        }.get(status, "⚪")

        return {
            "status": "success",
            "workload": result,
            "message": (
                f"{status_emoji} 今日负荷: {result['total_estimated_hours']}h / {result['task_count']} 项"
                f" — {status}"
            ),
        }

    def _handle_cancel_reminder(self, task_input: Dict) -> Dict:
        """取消/完成提醒"""
        reminder_id = task_input.get("reminder_id", "")
        if not reminder_id:
            return {"status": "error", "message": "缺少 reminder_id"}

        ok = self.calendar.cancel_reminder(reminder_id)
        return {
            "status": "success" if ok else "error",
            "reminder_id": reminder_id,
            "message": f"已取消提醒: {reminder_id}" if ok else f"未找到提醒: {reminder_id}",
        }

    def _handle_list_reminders(self, task_input: Dict) -> Dict:
        """列出所有提醒"""
        active_only = task_input.get("active_only", False)
        reminders = self.calendar.list_reminders(active_only=active_only)

        return {
            "status": "success",
            "reminders": reminders,
            "count": len(reminders),
        }

    def create_reminders_from_todos(self, todos: List[TodoItem]) -> List[str]:
        """
        从待办事项自动创建提醒

        Args:
            todos: 待办事项列表

        Returns:
            创建的提醒 ID 列表
        """
        reminder_ids = []
        for todo in todos:
            if todo.deadline:
                rid = self.calendar.add_reminder(
                    title=f"[待办] {todo.title}",
                    description=todo.description,
                    trigger_time=todo.deadline - timedelta(hours=1),  # 提前1小时提醒
                    notify_method=["console"],
                )
                reminder_ids.append(rid)

        logger.info(f"Created {len(reminder_ids)} reminders from todos")
        return reminder_ids
