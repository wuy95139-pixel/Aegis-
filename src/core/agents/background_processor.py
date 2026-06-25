"""
后台自动处理器
==============
从 Orchestrator 提取出的后台自动任务。

处理内容：
  - 个人信息自动记忆
  - 待办自动提取
  - 长期记忆自动检测
  - 重复内容追踪

包含节流控制（每 N 次请求执行一次）和重复内容追踪器状态。
"""

import re
import json
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.agents.orchestrator_utils import extract_json
from src.core.memory.types import MemoryType

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """后台自动任务处理器（节流控制 + 重复内容追踪）"""

    def __init__(
        self,
        llm,
        agents,
        task_store,
        memory_manager=None,
        auto_process_interval: int = 3,
    ):
        self.llm = llm
        self.agents = agents
        self.task_store = task_store
        self.memory_manager = memory_manager
        self._auto_process_interval = auto_process_interval
        self._request_count = 0
        self._repeated_content_tracker: Dict[str, Dict[str, Any]] = {}
        self._repeat_tracker_lock = threading.Lock()

    @property
    def repeated_content_tracker(self):
        return self._repeated_content_tracker

    @property
    def repeat_tracker_lock(self):
        return self._repeat_tracker_lock

    @property
    def request_count(self):
        return self._request_count

    def should_run_auto_tasks(self) -> bool:
        """检查是否应该运行自动任务（基于节流间隔）"""
        self._request_count += 1
        return self._request_count % self._auto_process_interval == 0

    # ===================== 个人信息记忆 =====================

    def auto_remember_personal_info(self, user_msg: str):
        """自动识别用户消息中的个人信息并持久化到记忆系统。"""
        if not self.memory_manager:
            return

        mm = self.memory_manager
        stored = False

        name_patterns = [
            r"我叫\s*([一-鿿\w]+)",
            r"我是\s*([一-鿿\w]+)",
            r"我的名字[是叫]?\s*([一-鿿\w]+)",
            r"称呼我[为]?\s*([一-鿿\w]+)",
        ]
        for pattern in name_patterns:
            m = re.search(pattern, user_msg)
            if m:
                name = m.group(1)
                stop_words = {"一个", "谁", "什么", "哪个", "怎么", "这个", "那个", "你的", "我的"}
                if name not in stop_words and len(name) <= 10:
                    try:
                        mm.remember(
                            content=f"用户的名字是{name}",
                            memory_type=MemoryType.USER,
                            source="auto_detect",
                            tags=["user_name", "identity"],
                            importance=0.8,
                            extract_key_points=False,
                        )
                        logger.info(f"Auto-remembered user name: {name}")
                        stored = True
                    except Exception as e:
                        logger.debug(f"Auto-remember name failed: {e}")
                break

        pref_patterns = [
            r"(?:我喜欢|我希望|我偏好|我想要|我希望你)\s*(.{2,50}?)(?:的|。|$)",
            r"(?:不要|别|少)\s*(.{2,50}?)(?:的|。|$)",
        ]
        for pattern in pref_patterns:
            m = re.search(pattern, user_msg)
            if m:
                pref = m.group(1).strip()
                if len(pref) >= 2 and len(pref) <= 50:
                    try:
                        if any(neg in user_msg[:m.start()] for neg in ["不要", "别", "少"]):
                            rule = f"避免{pref}"
                            why = f"用户明确表示不喜欢{pref}"
                        else:
                            rule = f"遵循{pref}"
                            why = f"用户明确表示喜欢{pref}"

                        mm.remember(
                            content=rule,
                            memory_type=MemoryType.FEEDBACK,
                            source="auto_detect",
                            tags=["user_preference", "auto"],
                            importance=0.7,
                            extract_key_points=False,
                            additional_metadata={"rule": rule, "why": why, "how_to_apply": f"在相关场景中{rule}"},
                        )
                        logger.info(f"Auto-remembered preference: {rule}")
                        stored = True
                    except Exception as e:
                        logger.debug(f"Auto-remember preference failed: {e}")
                break

        if stored:
            logger.info("Personal info auto-remembered from conversation")

    # ===================== 待办自动提取 =====================

    def auto_extract_todos(self, user_msg: str) -> str:
        """自动检测用户消息中的待办事项并存入 TaskStore。"""
        has_list_indicators = bool(re.search(
            r"第[一二三四五六七八九十\d]+|[12]\.|[12]．|首先|然后|接着|最后|帮我记|记住|别忘了|记下来",
            user_msg
        ))
        if not has_list_indicators:
            return ""

        from src.core.tools.time_tools import get_time_context as _get_time_context
        time_context = _get_time_context()
        prompt = f"""{time_context}

从用户消息中提取所有待办事项。要求：
1. 如果用户有明确的顺序要求（第一/第二/然后/接着），保留顺序
2. 如果用户给出了时间要求，提取为 ISO 格式；没有时间要求的，trigger_time 设为 null
3. 每个待办提取：title（简短标题）、trigger_time（ISO或null）、description（可选说明）
4. 尝试判断这些待办是否属于同一上下文的批量任务（如用户说"我有几件事需要做"）

用户消息: "{user_msg[:800]}"

返回 JSON 数组（不要其他内容）:
[{{"title": "任务标题", "trigger_time": "2026-05-20T15:00:00", "description": "补充说明"}}]

如果用户消息中没有明确待办事项，返回空数组: []"""

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = extract_json(resp["content"])
            todos = json.loads(raw)
        except Exception as e:
            logger.debug(f"Auto-extract todos failed: {e}")
            return ""

        if not todos or not isinstance(todos, list):
            return ""

        timed_tasks = []
        untimed_tasks = []

        for todo in todos[:10]:
            title = todo.get("title", "").strip()
            if not title:
                continue
            if todo.get("trigger_time"):
                timed_tasks.append(todo)
            else:
                untimed_tasks.append(todo)

        for todo in timed_tasks:
            self.agents["reminder_agent"].execute({
                "operation": "set",
                "title": todo["title"],
                "description": todo.get("description", ""),
                "trigger_time": todo.get("trigger_time"),
                "cron_expression": None,
                "notify_method": ["console"],
            })

        output_parts = []
        if untimed_tasks:
            first_title = untimed_tasks[0]["title"]
            if len(untimed_tasks) == 1:
                group_name = first_title
            else:
                group_name = f"{first_title}等{len(untimed_tasks)}项任务"

            context_summary = user_msg[:100].replace("\n", " ")
            self.task_store.add_task_group(
                tasks=untimed_tasks,
                group_name=group_name,
                context=context_summary,
                set_active=True,
            )
            logger.info(f"Auto-created task group '{group_name}' with {len(untimed_tasks)} tasks")

            task_list = "\n".join(f"  {i+1}. {t['title']}" for i, t in enumerate(untimed_tasks))
            output_parts.append(f"📋 已记录待办任务组「{group_name}」：\n{task_list}")

        if timed_tasks:
            logger.info(f"Auto-created {len(timed_tasks)} timed reminders from conversation")
            time_list = "\n".join(f"  ⏰ {t['title']} — {t.get('trigger_time', '待定')}" for t in timed_tasks)
            output_parts.append(f"⏰ 已创建 {len(timed_tasks)} 个定时提醒：\n{time_list}")

        return "\n".join(output_parts) if output_parts else ""

    # ===================== 长期记忆自动检测 =====================

    def auto_detect_long_term_memory(self, user_msg: str):
        """自动检测需要存入长期记忆的内容。"""
        if not self.memory_manager:
            return

        mm = self.memory_manager

        importance_keywords = [
            "很重要", "十分重要", "非常重要", "特别重要", "极其重要",
            "这个重要", "至关重要", "关键的是", "最关键", "核心",
            "记住这个", "别忘了这个", "务必记住", "千万记住", "牢记",
            "这个是重点", "重点内容", "重中之重", "很关键", "非常关键",
            "特别关键", "极其关键", "这个关键", "最关键", "核心内容",
            "核心信息", "核心要点", "重要信息", "重要内容", "重要的事", "必须记住",
        ]

        has_importance_kw = any(kw in user_msg for kw in importance_keywords)
        if has_importance_kw:
            context_parts = []
            for kw in importance_keywords:
                if kw in user_msg:
                    idx = user_msg.find(kw)
                    start = max(0, idx - 60)
                    end = min(len(user_msg), idx + len(kw) + 60)
                    snippet = user_msg[start:end].strip()
                    context_parts.append(snippet)
                    break

            if context_parts:
                content_to_remember = context_parts[0]
                try:
                    mm.remember(
                        content=content_to_remember,
                        memory_type=MemoryType.PROJECT,
                        source="auto_importance_keyword",
                        tags=["important", "user_marked", "auto"],
                        importance=0.9,
                        extract_key_points=True,
                        additional_metadata={
                            "fact": content_to_remember,
                            "why": "用户标记为重要内容",
                            "how_to_apply": "在后续对话中优先参考此内容",
                            "status": "active",
                        },
                    )
                    logger.info(f"Long-term memory auto-saved (importance keyword): {content_to_remember[:80]}...")
                except Exception as e:
                    logger.debug(f"Auto-save to long-term memory failed: {e}")

        with self._repeat_tracker_lock:
            items_snapshot = list(self._repeated_content_tracker.items())
        for topic, tracker in items_snapshot:
            count = tracker.get("count", 0)
            if count >= 3 and not tracker.get("saved_to_memory"):
                messages = tracker.get("messages", [])
                combined = " | ".join(messages[-3:])
                try:
                    mm.remember(
                        content=combined,
                        memory_type=MemoryType.PROJECT,
                        source="auto_repeated_topic",
                        tags=["repeated", "auto", topic[:20]],
                        importance=0.75,
                        extract_key_points=True,
                        additional_metadata={
                            "fact": f"用户反复提及的话题: {topic}",
                            "why": f"该话题在对话中被提及{count}次",
                            "how_to_apply": "该话题对用户很重要，注意关联",
                            "status": "active",
                        },
                    )
                    with self._repeat_tracker_lock:
                        tracker["saved_to_memory"] = True
                    logger.info(f"Long-term memory auto-saved (repeated topic): {topic} ({count} times)")
                except Exception as e:
                    logger.debug(f"Auto-save repeated topic failed: {e}")

    # ===================== 重复内容追踪 =====================

    def track_repeated_content(self, user_msg: str):
        """追踪对话中反复出现的话题关键词。"""
        cleaned = re.sub(r"[，,。.！!？?、\s]+", "", user_msg)
        if len(cleaned) < 5:
            return

        try:
            prompt = f"""从以下用户消息中提取1-3个核心话题关键词（每个1-5个字），用逗号分隔。
只返回关键词，不要其它内容。

用户消息: {user_msg[:200]}

关键词:"""

            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )
            topics_text = resp["content"].strip()
        except Exception:
            logger.debug("LLM topic classification failed, falling back to raw text", exc_info=True)
            topics_text = cleaned[:10]

        topics = [t.strip() for t in topics_text.replace("、", ",").replace("，", ",").split(",") if t.strip()]
        now = datetime.now()

        with self._repeat_tracker_lock:
            for topic in topics:
                if len(topic) < 2 or len(topic) > 15:
                    continue

                matched_key = None
                for existing_topic in self._repeated_content_tracker:
                    if topic in existing_topic or existing_topic in topic:
                        matched_key = existing_topic
                        break

                if matched_key:
                    tracker = self._repeated_content_tracker[matched_key]
                    tracker["count"] += 1
                    tracker["messages"].append(user_msg[:100])
                    tracker["last_updated"] = now.isoformat()
                    logger.debug(f"Repeated topic '{matched_key}' count: {tracker['count']}")
                else:
                    self._repeated_content_tracker[topic] = {
                        "count": 1,
                        "messages": [user_msg[:100]],
                        "first_seen": now.isoformat(),
                        "last_updated": now.isoformat(),
                        "saved_to_memory": False,
                    }

            stale_keys = []
            for key, tracker in self._repeated_content_tracker.items():
                last_updated = tracker.get("last_updated", tracker.get("first_seen", ""))
                if last_updated:
                    try:
                        updated_time = datetime.fromisoformat(last_updated)
                        if (now - updated_time).total_seconds() > 3600:
                            stale_keys.append(key)
                    except (ValueError, TypeError):
                        pass

            for key in stale_keys:
                del self._repeated_content_tracker[key]
