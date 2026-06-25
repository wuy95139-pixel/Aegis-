"""
任务与提醒处理器
================
从 Orchestrator 提取出的任务查询、提醒管理、工作负荷处理器。
"""

import re
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.core.agents.orchestrator_utils import extract_json
from src.core.tools.time_tools import get_time_context

logger = logging.getLogger(__name__)


class TaskHandlers:
    """任务和提醒相关所有意图处理器"""

    def __init__(self, llm: "LLMProvider", agents: dict, task_store: "TaskStore", run_with_tools_fn: "Callable | None", memory_manager: "MemoryManager | None" = None):
        self.llm = llm
        self.agents = agents
        self.task_store = task_store
        self._run_with_tools = run_with_tools_fn
        self.memory_manager = memory_manager

    # ===================== 提醒设置 =====================

    def reminder_set(self, user_message: str, params: dict, stream_callback=None) -> dict:
        """设置提醒 — 使用 function calling 精准解析时间"""
        title = params.get("title", "提醒")
        description = params.get("description", "")
        trigger_time = None
        cron_expression = None

        time_prompt = f"""从用户消息中提取提醒的时间信息。

规则:
- 如果用户说了具体时间点（如"明天下午3点"、"下周一上午10点"），调用 parse_time
- 如果用户说了重复规则（如"每天早上8点"、"每周五下午5点"），调用 time_to_cron
- 如果没有明确时间，不要调用任何工具
- 必须在 parse_time 和 time_to_cron 中选择最合适的一个，不要同时调用
- 工具调用结果中包含解析后的时间。请在最后一行以 "RESULT: <ISO时间或cron表达式>" 的格式输出解析结果
- 如果无时间信息，最后一行输出 "RESULT: NONE"

用户消息: {user_message}

请提取时间信息:"""

        try:
            time_result = self._run_with_tools(
                messages=[{"role": "user", "content": time_prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            for line in time_result.strip().split("\n"):
                line = line.strip()
                if line.startswith("RESULT:"):
                    value = line[7:].strip()
                    if value and value != "NONE":
                        is_iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}", value))
                        if is_iso:
                            trigger_time = value
                        else:
                            cron_expression = value
                    break
        except Exception as e:
            logger.warning(f"Time extraction via tools failed: {e}, falling back to original message")
            from src.core.tools.time_tools import parse_chinese_time_expression, expression_to_cron

            def _try_extract_time(msg: str):
                cron = expression_to_cron(msg)
                if cron and not cron.startswith("无法解析"):
                    return None, cron
                parsed = parse_chinese_time_expression(msg)
                if parsed:
                    return parsed.isoformat(), None
                now = datetime.now()
                simple_patterns = [
                    (r"(\d+)\s*分[钟]?\s*[之以]?后", "minutes"),
                    (r"(\d+)\s*(?:个?\s*)?(?:小?时|h(?:our)?)\s*[之以]?后", "hours"),
                    (r"(\d+)\s*天\s*[之以]?后", "days"),
                    (r"(\d+)\s*秒[钟]?\s*[之以]?后", "seconds"),
                ]
                for pattern, unit in simple_patterns:
                    m = re.search(pattern, msg)
                    if m:
                        n = int(m.group(1))
                        delta = timedelta(**{unit: n})
                        return (now + delta).isoformat(), None
                return None, None

            trigger_time, cron_expression = _try_extract_time(user_message)

        result = self.agents["reminder_agent"].execute({
            "operation": "set",
            "title": title,
            "description": description,
            "trigger_time": trigger_time,
            "cron_expression": cron_expression,
            "notify_method": ["sound", "toast", "console"],
        })

        if cron_expression:
            time_display = f"循环: {cron_expression}"
        elif trigger_time:
            time_display = trigger_time
        else:
            time_display = "事件触发"
        return {"status": "success", "response": f"⏰ 提醒已设置\n**{title}**\n时间: {time_display}"}

    # ===================== 提醒检查 =====================

    def reminder_check(self) -> dict:
        followup = self.agents["reminder_agent"].execute({"operation": "followup"})
        reminders_list = self.agents["reminder_agent"].execute({"operation": "list", "active_only": True})

        parts = ["## ⏰ 提醒与跟进\n"]

        if reminders_list.get("reminders"):
            parts.append(f"### 活跃提醒 ({reminders_list['count']})")
            for r in reminders_list["reminders"]:
                t = r.trigger_time.strftime("%m-%d %H:%M") if r.trigger_time else "事件触发"
                parts.append(f"- **{r.title}** — {t}")

        if followup.get("followup_suggestions"):
            parts.append(f"\n### ⚠️ 需要跟进 ({followup['count']})")
            for s in followup["followup_suggestions"]:
                parts.append(f"- {s.get('title', s.get('id', ''))}")

        if not reminders_list.get("reminders") and not followup.get("followup_suggestions"):
            parts.append("暂无提醒或需要跟进的事项。")

        return {"status": "success", "response": "\n".join(parts)}

    # ===================== 提醒取消 =====================

    def reminder_cancel(self, user_msg: str = "", params: dict = None) -> dict:
        """取消/删除提醒 — 支持全部清除或按名称/关键词删除单条。"""
        params = params or {}
        reminder_agent = self.agents.get("reminder_agent")
        if not reminder_agent:
            return {"status": "error", "response": "提醒代理未初始化。"}

        # 判断是否全部删除
        all_keywords = ["全部", "所有", "一切", "统统", "都删", "都取消", "清除掉", "清掉", "全删", "全都"]
        is_delete_all = any(kw in user_msg for kw in all_keywords)

        if is_delete_all:
            result = reminder_agent.execute({"operation": "list", "active_only": True})
            reminders = result.get("reminders", [])
            if not reminders:
                return {"status": "success", "response": "当前没有活跃的提醒需要删除。"}

            count = 0
            for r in reminders:
                ok = reminder_agent.calendar.cancel_reminder(r.id)
                if ok:
                    count += 1
            return {"status": "success", "response": f"🗑️ 已删除 {count} 条提醒。"}

        # 按关键词删除单条
        title_keyword = params.get("title", "").strip()
        if not title_keyword:
            extract_prompt = f"""从用户消息中提取要删除的提醒名称/关键词。只返回关键词，不要其他内容。如果无法确定具体名称，返回空字符串。

用户消息: {user_msg}

关键词:"""
            try:
                resp = self.llm.chat(
                    messages=[{"role": "user", "content": extract_prompt}],
                    temperature=0.1,
                    max_tokens=50,
                )
                title_keyword = resp.get("content", "").strip().strip('"').strip("'")
            except Exception as e:
                logger.debug("LLM title extraction failed in reminder_cancel: %s", e)

        if not title_keyword:
            return {"status": "success", "response": "请指定要删除哪条提醒，例如\"删除'开会'这条提醒\"。\n\n💡 提示：说\"删除所有提醒\"可一键清除全部。"}

        result = reminder_agent.execute({"operation": "list", "active_only": True})
        reminders = result.get("reminders", [])
        matched = None
        for r in reminders:
            if title_keyword in r.title or r.title in title_keyword:
                matched = r
                break

        if not matched:
            return {"status": "success", "response": f"未找到包含「{title_keyword}」的提醒。"}

        ok = reminder_agent.calendar.cancel_reminder(matched.id)
        if ok:
            return {"status": "success", "response": f"🗑️ 已删除提醒: **{matched.title}**"}
        return {"status": "error", "response": f"删除提醒失败: {matched.title}"}

    # ===================== 工作负荷 =====================

    def workload_check(self, user_message: str = "", params: dict = None) -> dict:
        """负荷感知提醒"""
        params = params or {}
        agent = self.agents.get("reminder_agent")
        if not agent:
            return {"status": "error", "response": "提醒代理未初始化。"}

        task_input = {"operation": "workload"}
        if params.get("date"):
            task_input["target_date"] = params["date"]

        result = agent.execute(task_input)
        if result.get("status") != "success":
            return {"status": "error", "response": f"负荷检查失败: {result.get('message', '')}"}

        wl = result["workload"]
        parts = [
            f"## {result['message']}\n",
            f"**日期**: {wl['date']} | **任务数**: {wl['task_count']} 项 | **预估耗时**: {wl['total_estimated_hours']}h\n",
        ]

        if wl.get("tasks"):
            parts.append("### 📋 今日任务\n")
            priority_emoji = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            for t in wl["tasks"]:
                emoji = priority_emoji.get(t["priority"], "⚪")
                parts.append(f"- {emoji} **{t['title']}** ({t['priority']}, ~{t['estimated_hours']}h)")
            parts.append("")

        if wl.get("suggestions"):
            parts.append("### 💡 排期建议\n")
            for s in wl["suggestions"]:
                parts.append(f"- {s}\n")
            parts.append("")

        if wl["status"] == "clear":
            parts.append("✅ 当日任务负荷正常，精力充足。\n")

        return {"status": "success", "response": "\n".join(parts)}

    # ===================== 任务添加 =====================

    def task_add(self, user_msg: str, params: dict = None) -> dict:
        """从自然语言中提取任务列表并添加到 TaskStore。

        与 reminder_set 的区分：
          - reminder_set: 有时间或"提醒"关键词 → 创建定时提醒
          - task_add: 无时间、无"提醒"关键词 → 创建待办任务组
        """
        params = params or {}

        # 用 LLM 从用户消息中提取任务列表
        extract_prompt = f"""从用户消息中提取所有待办任务项，返回 JSON 数组。

规则:
- 每个任务是一个对象: {{"title": "任务标题", "priority": "medium"}}
- priority 可选值: "high"（紧急/重要）、"medium"（普通）、"low"（低优）
- 如果任务标题中含"紧急"/"urgent"/"重要"，priority 设为 "high"
- 只返回 JSON 数组，不要其他内容
- 如果无法提取到任务，返回空数组 []

用户消息: {user_msg}

JSON 数组:"""

        tasks = []
        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": extract_prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = resp.get("content", "[]")
            # 容忍 markdown 代码块包裹
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0]
            tasks = json.loads(raw)
            if not isinstance(tasks, list):
                tasks = []
        except Exception as e:
            logger.warning(f"task_add LLM extraction failed: {e}")

        if not tasks:
            return {
                "status": "success",
                "response": "未能从消息中识别到明确的待办事项。请用\"第一xxx、第二xxx\"或\"1.xxx 2.xxx\"的格式列举。",
            }

        # 生成分组名
        group_name = params.get("group_name", "")
        if not group_name:
            try:
                name_prompt = f"""根据以下任务列表，生成一个简洁的分组名称（10字以内，不含引号）。

任务: {json.dumps([t['title'] for t in tasks[:5]], ensure_ascii=False)}

分组名称:"""
                name_resp = self.llm.chat(
                    messages=[{"role": "user", "content": name_prompt}],
                    temperature=0.3,
                    max_tokens=30,
                )
                group_name = name_resp.get("content", "").strip().strip('"').strip("'").strip()
                if not group_name or len(group_name) > 20:
                    group_name = f"待办任务组"
            except Exception:
                group_name = f"待办任务组"

        # 写入 TaskStore
        group_id = self.task_store.add_task_group(
            tasks=tasks,
            group_name=group_name,
            context=user_msg,
            set_active=True,
        )

        # 构建响应
        lines = [f"## ✅ 已添加 {len(tasks)} 项任务到「{group_name}」\n"]
        for i, t in enumerate(tasks, 1):
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", "medium"), "🟡")
            lines.append(f"{i}. {emoji} **{t['title']}**")
        lines.append(f"\n共 {len(tasks)} 项")

        return {"status": "success", "response": "\n".join(lines)}

    # ===================== 任务查询 =====================

    def task_inquiry(self, user_msg: str = "", stream_callback=None) -> dict:
        """查询所有待办任务（按分组展示）"""
        next_hint = ""
        if user_msg and any(kw in user_msg for kw in ["接下来", "下一步", "然后", "还要", "其它", "其他"]):
            pending = self.task_store.get_next_pending_global()
            if pending:
                next_hint = f"📌 下一步: **{pending['title']}**\n\n"
            else:
                next_hint = "🎉 所有任务已完成！\n\n"

        all_groups = self.task_store.get_all_pending_tasks()

        if not all_groups:
            if next_hint:
                return {"status": "success", "response": next_hint.strip()}
            return {"status": "success", "response": "暂无待办或提醒。"}

        sections = [next_hint.strip()] if next_hint else []

        for group in all_groups:
            if group["is_active"]:
                lines = [f"## 📋 待办与任务\n"]
                lines.append(f"### 📌 {group['group_name']}")
                for idx, task in enumerate(group["tasks"], 1):
                    if task["status"] == "completed":
                        lines.append(f"{idx}. ~~**{task['title']}**~~ — 已完成")
                    elif task["status"] == "cancelled":
                        lines.append(f"{idx}. ~~{task['title']}~~ — 已取消")
                    else:
                        lines.append(f"{idx}. **{task['title']}** — 待完成")
                sections.append("\n".join(lines))

        for group in all_groups:
            if group["is_active"]:
                continue
            lines = [f"### 📋 {group['group_name']}"]
            for idx, task in enumerate(group["tasks"], 1):
                if task["status"] == "completed":
                    lines.append(f"{idx}. ~~**{task['title']}**~~ — 已完成")
                elif task["status"] == "cancelled":
                    lines.append(f"{idx}. ~~{task['title']}~~ — 已取消")
                else:
                    lines.append(f"{idx}. **{task['title']}** — 待完成")
            sections.append("\n".join(lines))

        return {"status": "success", "response": "\n\n".join(sections)}

    # ===================== 任务完成检测 =====================

    def extract_completed_task_name(self, user_msg: str) -> Optional[str]:
        """从用户消息中提取被完成的任务名称。"""
        all_groups = self.task_store.get_all_pending_tasks()
        for group in all_groups:
            for task in group["tasks"]:
                if task["status"] == "pending" and len(task["title"]) >= 3:
                    if task["title"] in user_msg:
                        logger.info(f"Task matched by title in message: '{task['title']}'")
                        return task["title"]

        reminders_result = self.agents["reminder_agent"].execute({
            "operation": "list", "active_only": True,
        })
        for r in reminders_result.get("reminders", []):
            if len(r.title) >= 3 and r.title in user_msg:
                logger.info(f"Task matched by reminder title in message: '{r.title}'")
                return r.title

        before_kw_pattern = re.search(
            r"([^，,。！!？?]{3,60})"
            r"(?:做完了|做好了|写完了|搞完了|弄完了|干完了|也完成了|已经完成了|已完成|好了)"
            r"(?:[。！!，,、]|$|了[吗呢吧啊]|$)",
            user_msg
        )
        if not before_kw_pattern:
            before_kw_pattern = re.search(
                r"([^，,。！!？?]{3,60})"
                r"(?:完成了|做好了|做完了|写完了|搞完了|弄完了|干完了|也完成了|已经完成了|已完成|好了)",
                user_msg
            )

        task_name = None
        if before_kw_pattern:
            raw = before_kw_pattern.group(1).strip()
            prefixes = ["我现在", "我终于", "我已经", "我刚刚", "我刚", "我今天",
                       "我", "我们", "大家", "终于", "已经", "刚刚", "刚"]
            for prefix in prefixes:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):].strip()
            for prefix in ["把", "将", "那个", "这个"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):].strip()
            for filler in ["已经", "已", "终于", "就", "都", "还", "也"]:
                raw = raw.replace(filler, "")
            for suffix in ["的", "了", "吗", "呢", "吧"]:
                if raw.endswith(suffix):
                    raw = raw[:-1].strip()
            noise = {"的", "了", "吗", "呢", "啊", "吧", "什么", "怎么",
                    "现在", "终于", "已经", "这个", "那个", "我", "我们",
                    "接下来", "然后", "应该", "还有", "其它", "其他"}
            if len(raw) >= 2 and raw not in noise:
                task_name = raw
                logger.info(f"Task extracted (before keyword): '{task_name}'")

        if not task_name:
            after_kw_pattern = re.search(
                r"(?:做完了|完成了|做好了|写完了|搞完了|弄完了|干完了|也完成了|已经完成了|好了)[，,\s]*"
                r"(.{3,60}?)"
                r"(?:[。！!，,、]|$|接下来|然后|应该|还有|其它|其他|帮我|你看)",
                user_msg
            )
            if after_kw_pattern:
                raw = after_kw_pattern.group(1).strip()
                for prefix in ["把", "将", "那个", "这个"]:
                    if raw.startswith(prefix):
                        raw = raw[len(prefix):].strip()
                for suffix in ["的", "了", "吗", "呢", "吧"]:
                    if raw.endswith(suffix):
                        raw = raw[:-1].strip()
                noise = {"的", "了", "吗", "呢", "啊", "吧", "什么", "怎么", "哪个",
                       "现在", "终于", "已经", "接下来", "然后", "应该", "还有"}
                if len(raw) >= 2 and raw not in noise:
                    task_name = raw
                    logger.info(f"Task extracted (after keyword): '{task_name}'")

        return task_name

    def detect_task_completion(self, user_msg: str) -> str:
        """检测用户消息中的任务完成声明"""
        has_done_keyword = bool(re.search(
            r"(?:做完了|做好了|写完了|搞完了|弄完了|干完了|也完成了|已经完成了|已完成|好了|完成了)",
            user_msg
        ))
        if not has_done_keyword:
            return ""

        task_name = self.extract_completed_task_name(user_msg)
        if not task_name:
            return ""

        return self.task_done(task_name)

    def task_done(self, task_keyword: str) -> str:
        """处理任务完成声明"""
        result = self.task_store.complete_task_by_title(task_keyword)
        completed = result.get("completed")

        if not completed:
            reminders_result = self.agents["reminder_agent"].execute({
                "operation": "list", "active_only": True,
            })
            reminders = reminders_result.get("reminders", [])
            for r in reminders:
                if task_keyword in r.title or r.title in task_keyword:
                    self.agents["reminder_agent"].calendar.acknowledge_reminder(r.id)
                    logger.info(f"Task completed (from reminders): {r.title} (id={r.id})")

                    if self.memory_manager:
                        try:
                            self.memory_manager.record_success(
                                situation=f"完成任务: {r.title}",
                                approach="用户声明完成 → ReminderAgent 匹配 → acknowledge_reminder",
                                lesson=f"用户按时完成了任务「{r.title}」",
                                context_tags=["task_completion", "reminder", "auto"],
                            )
                        except Exception as e:
                            logger.debug(f"Experience recording failed (non-fatal): {e}")

                    next_reminders = [x for x in reminders if x.id != r.id and x.is_active]
                    lines = [f"✅ 已完成: {r.title}"]
                    if next_reminders:
                        nt = next_reminders[0]
                        lines.append(f"👉 下一步: {nt.title}")
                    else:
                        lines.append("🎉 所有提醒都完成了！")
                    return "\n".join(lines)
            logger.debug(f"No matching task found for: {task_keyword}")
            return ""

        group = result.get("group")
        next_in_group = result.get("next_in_group")
        group_all_done = result.get("group_all_done", False)
        other_groups = result.get("other_groups_pending", [])
        all_done = result.get("all_done", False)

        lines = [f"✅ 已完成: {completed.title}"]

        if next_in_group:
            lines.append(f"👉 下一步: {next_in_group.title}")
        elif group_all_done and other_groups:
            lines.append(f"🎯 「{group.name}」全部完成！")
            lines.append("")
            lines.append("📋 其它待办与任务：")
            for og in other_groups:
                for t in og["tasks"]:
                    lines.append(f"  - **{t['title']}**")
        elif all_done:
            lines.append("🎉 所有任务都完成了！")
        else:
            lines.append("🎉 所有任务都完成了！")

        lines.append("")
        lines.append(self.build_group_status(group))

        return "\n".join(lines)

    @staticmethod
    def build_group_status(group) -> str:
        """构建单个任务组的状态列表"""
        lines = [f"## 📋 {group.name}"]
        tasks = sorted(group.tasks, key=lambda t: t.order)
        for idx, task in enumerate(tasks, 1):
            if task.status == "completed":
                lines.append(f"{idx}. ~~**{task.title}**~~ — 已完成")
            elif task.status == "cancelled":
                lines.append(f"{idx}. ~~{task.title}~~ — 已取消")
            else:
                lines.append(f"{idx}. **{task.title}** — 待完成")
        return "\n".join(lines)
