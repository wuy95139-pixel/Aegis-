"""
调度器 (Orchestrator) — Aegis 核心调度器
========================================
使用 LLM 进行意图识别和参数提取，所有功能通过自然语言触发。

支持的意图:
  - file_parse      仅解析文件
  - file_translate  翻译文件/文本
  - file_polish     润色文件/文本
  - file_generate_ppt   从文件/文本生成 PPT
  - file_extract_todos  提取待办 → 分派 → 提醒
  - research        研究分析
  - reminder_set    设置提醒
  - reminder_check  检查提醒/跟进
  - task_inquiry    查询待办/任务
  - memory_search   搜索对话记忆
  - memory_summarize 总结对话
  - briefing        生成简报
  - general_chat    通用对话

架构：薄外观模式 — Orchestrator 保留所有公开 API 不变，
内部实现委托给子模块（message_bus, session_manager, handler_registry 等）。
"""

import json
import re
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta

from src.core.agents.message_bus import MessageBus  # noqa: F401 — re-export
from src.core.agents.base import BaseAgent
from src.core.agents.file_processor import FileProcessorAgent
from src.core.agents.task_dispatcher import TaskDispatcherAgent
from src.core.agents.memory_agent import MemoryAgent
from src.core.agents.reminder_agent import ReminderAgent
from src.core.agents.research_agent import ResearchAgent
from src.core.agents.intent_classifier import IntentClassifier, AVAILABLE_TOOLS
from src.core.agents.router import build_handlers
from src.core.agents.session_manager import SessionManager
from src.core.agents.handler_registry import HandlerRegistry
from src.core.agents.background_processor import BackgroundProcessor
from src.core.agents.conversation_logger import ConversationLogger
from src.core.agents.llm_interaction import LLMInteraction, execute_tool
from src.core.agents.orchestrator_utils import (
    extract_json,
    validate_file_path,
    get_content,
    build_smart_kpis,
    aggregate_chart_data,
    build_column_overview,
)
from src.core.llm.provider import LLMProvider
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.short_term import ShortTermMemory
from src.core.tools.translation_tools import TranslationTool
from src.core.tools.time_tools import (
    get_time_context,
    parse_chinese_time_expression,
    expression_to_cron,
    get_future_date,
    is_overdue,
)
from src.core.tools.task_store import TaskStore
from src.models.schemas import AgentMessage

logger = logging.getLogger(__name__)

# 支持流式输出的意图列表
_STREAMABLE_INTENTS = frozenset({
    "general_chat", "file_translate", "file_polish", "file_qa",
    "file_generate_ppt", "research", "memory_search", "memory_summarize",
    "briefing", "task_inquiry", "reminder_set",
})


class Orchestrator:
    """Aegis 主调度器 — 自然语言驱动（薄外观）"""

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryRetriever,
        config: Optional[Dict] = None,
        memory_manager=None,
    ):
        self.llm = llm
        self.memory = memory
        self.config = config or {}
        self.memory_manager = memory_manager

        # 基础设施
        self.message_bus = MessageBus()
        self.translation_tool = TranslationTool(llm)
        self.intent_classifier = IntentClassifier(llm)
        self.llm_interaction = LLMInteraction(llm)

        # Agent 注册表（保持向后兼容）
        self.agents: Dict[str, BaseAgent] = {
            "file_processor": FileProcessorAgent(llm, memory, self.config),
            "task_dispatcher": TaskDispatcherAgent(llm, memory, self.config),
            "memory_agent": MemoryAgent(llm, memory, self.config),
            "reminder_agent": ReminderAgent(llm, memory, self.config),
            "research_agent": ResearchAgent(llm, memory, self.config),
        }

        # 独立任务存储
        task_path = self.config.get("system", {}).get("data_dir", "./data")
        self.task_store = TaskStore(storage_path=str(Path(task_path) / "tasks.json"))

        # 会话管理器（传入 LLM 以支持会话级短记忆语义压缩）
        st_config = self.config.get("memory", {})
        self.session_manager = SessionManager(
            global_memory=memory,
            short_term_max_tokens=st_config.get("short_term", {}).get("max_tokens", 16000),
            short_term_window=st_config.get("short_term", {}).get("window_size", 20),
            session_ttl_seconds=7200,
            llm=llm,
        )

        # 后台处理器（节流控制 + 重复内容追踪）
        self.background = BackgroundProcessor(
            llm=llm,
            agents=self.agents,
            task_store=self.task_store,
            memory_manager=memory_manager,
        )

        # 对话日志器
        self.conversation_logger = ConversationLogger(
            memory=memory,
            session_manager=self.session_manager,
        )

        # 处理器注册中心（聚合所有领域处理器）
        self.handlers = HandlerRegistry(
            llm=llm,
            agents=self.agents,
            memory=memory,
            translation_tool=self.translation_tool,
            task_store=self.task_store,
            session_manager=self.session_manager,
            memory_manager=memory_manager,
            llm_interaction=self.llm_interaction,
        )

        # 注入消息总线到所有 Agent
        for agent in self.agents.values():
            agent.message_bus = self.message_bus

        for name, agent in self.agents.items():
            self.message_bus.subscribe(name, self._create_message_handler(name))

        logger.info(f"Orchestrator ready: {list(self.agents.keys())}")

    # ==================== 会话隔离（委托给 SessionManager） ====================

    def get_session_memory(self, session_id: str) -> MemoryRetriever:
        return self.session_manager.get(session_id)

    def cleanup_session(self, session_id: str) -> bool:
        return self.session_manager.cleanup(session_id)

    def _cleanup_expired_sessions(self):
        self.session_manager.cleanup_expired()

    def get_existing_session_memory(self, session_id: str):
        return self.session_manager.get_existing(session_id)

    # backward compat
    def _get_existing_session_memory(self, session_id: str):
        return self.get_existing_session_memory(session_id)

    # ==================== 主入口 ====================

    def process_user_request(
        self,
        user_message: str,
        attached_file: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理用户请求 — 使用 LLM 识别意图并路由

        核心流程:
          1. 检索相关记忆 + 用户画像 + 自适应行为指导
          2. LLM 分析意图 + 提取参数
          3. 自适应引擎 before_action → 路由执行 → after_action
        """
        logger.info(f"Processing: '{user_message[:100]}...'" + (f" [session={session_id[:8]}]" if session_id else ""))

        # Step 1: 检索记忆 + 增强上下文
        if session_id:
            session_mem = self.get_session_memory(session_id)
            mem_result = self.agents["memory_agent"].execute({
                "operation": "retrieve", "query": user_message, "top_k": 3,
                "session_retriever": session_mem,
            })
        else:
            mem_result = self.agents["memory_agent"].execute({
                "operation": "retrieve", "query": user_message, "top_k": 3,
            })

        # Step 2: 自适应引擎 — 行动前获取行为指导
        guidance_text = ""
        if self.memory_manager:
            try:
                guidance = self.memory_manager.before_action(user_message, "general")
                guidance_text = guidance.get("guidance_text", "")
                if guidance_text:
                    logger.debug(f"Adaptive guidance: {guidance_text[:200]}")
            except Exception as e:
                logger.warning(f"Adaptive guidance failed (non-fatal): {e}")

        # Step 3: 检测任务完成声明
        completion_output = self._detect_task_completion(user_message)

        # Step 4: 意图识别
        intent_info = self.intent_classifier.classify(
            user_message, attached_file, mem_result.get("context", "")
        )

        intent = intent_info.get("intent", "general_chat")
        params = intent_info.get("params", {})
        logger.info(f"Intent: {intent}, params: {params}")

        # Step 5: 路由执行
        handlers = build_handlers(
            orchestrator=self,
            user_message=user_message,
            attached_file=attached_file,
            params=params,
            session_id=session_id,
            stream_callback=stream_callback,
        )

        handler = handlers.get(intent)
        if handler:
            try:
                if stream_callback and intent in _STREAMABLE_INTENTS:
                    # 流式意图: handler 内部负责推送 token
                    result = handler()
                    if result and "response" not in result:
                        result["response"] = ""
                else:
                    result = handler()
                    if stream_callback:
                        stream_callback(result.get("response", ""))
            except Exception as e:
                logger.error(f"Handler '{intent}' failed: {type(e).__name__}: {e}")
                result = {
                    "status": "error",
                    "response": f"[系统错误] 处理请求时发生异常: {type(e).__name__}",
                    "intent": intent,
                }
        else:
            result = self._handle_general_chat(
                user_message,
                mem_result.get("context", ""),
                guidance_text=guidance_text,
                stream_callback=stream_callback,
                session_id=session_id,
                attached_file=attached_file,
            )

        # Step 6: 后处理 — 注入完成检测结果
        if completion_output and result.get("status") == "success":
            current = result.get("response", "")
            result["response"] = completion_output + ("\n\n" + current if current else "")

        # Step 7: 自适应引擎 — 记录行动结果
        if self.memory_manager:
            try:
                self.memory_manager.after_action(
                    situation=user_message,
                    approach=f"intent={intent}, params={json.dumps(params, ensure_ascii=False)[:200]}",
                    outcome=result.get("status", "unknown"),
                    context_tags=[intent, "auto"],
                )
            except Exception as e:
                logger.warning(f"after_action failed (non-fatal): {e}")

        # Step 8: 自动处理（节流）
        if self.background.should_run_auto_tasks():
            try:
                self._auto_remember_personal_info(user_message)
            except Exception as e:
                logger.warning(f"Auto-remember personal info failed: {e}")
            try:
                extraction = self._auto_extract_todos(user_message)
                if extraction and result.get("status") == "success":
                    current = result.get("response", "")
                    result["response"] = (current + "\n\n" + extraction).strip()
            except Exception as e:
                logger.warning(f"Auto-extract todos failed: {e}")
            try:
                self._auto_detect_long_term_memory(user_message)
            except Exception as e:
                logger.warning(f"Auto-detect long-term memory failed: {e}")
            try:
                self._track_repeated_content(user_message)
            except Exception as e:
                logger.warning(f"Track repeated content failed: {e}")

        # Step 9: 学习闭环
        if self.memory_manager:
            try:
                self.memory_manager.learn_from_interaction(
                    user_message=user_message,
                    assistant_response=result.get("response", "")[:500],
                    context={"intent": intent, "params": params},
                )
            except Exception as e:
                logger.warning(f"learn_from_interaction failed (non-fatal): {e}")

        # Step 10: 记录对话到短期记忆 + 持久化
        self._record_conversation(
            user_message, result.get("response", "")[:1000], session_id,
        )

        return result

    # ==================== Function Calling（委托给 LLMInteraction） ====================

    def _execute_tool(self, name: str, arguments: dict) -> str:
        return execute_tool(name, arguments)

    def _run_with_tools(
        self, messages: List[dict],
        temperature: float = 0.3, max_tokens: int = 1000, max_rounds: int = 3,
    ) -> str:
        return self.llm_interaction.run_with_tools(
            messages, temperature=temperature, max_tokens=max_tokens, max_rounds=max_rounds,
        )

    def _stream_chat_with_tools(
        self, messages: List[dict], stream_callback: Callable[[str], None],
        temperature: float = 0.7, max_tokens: int = 1500,
    ) -> str:
        return self.llm_interaction.stream_chat_with_tools(
            messages, stream_callback, temperature=temperature, max_tokens=max_tokens,
        )

    # ==================== Handler: 文件处理 ====================

    @staticmethod
    def _validate_file_path(filepath: str) -> str:
        return validate_file_path(filepath)

    def _handle_file_parse(self, filepath: Optional[str]) -> dict:
        return self.handlers.file.parse(filepath)

    def _handle_file_qa(self, question: str, filepath: Optional[str], stream_callback=None) -> dict:
        return self.handlers.file.qa(question, filepath, stream_callback=stream_callback)

    def _handle_audio_transcribe(self, filepath: Optional[str]) -> dict:
        return self.handlers.file.audio_transcribe(filepath)

    def _handle_translate(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        return self.handlers.file.translate(msg, filepath, params, stream_callback=stream_callback)

    def _handle_polish(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        return self.handlers.file.polish(msg, filepath, params, stream_callback=stream_callback)

    def _handle_generate_ppt(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        return self.handlers.file.generate_ppt(msg, filepath, params, stream_callback=stream_callback)

    def _handle_file_extract_todos(self, filepath: Optional[str], params: dict) -> dict:
        return self.handlers.file.extract_todos(filepath, params)

    # ==================== Handler: 研究 ====================

    def _handle_research(self, params: dict, stream_callback=None) -> dict:
        return self.handlers.research.research(params, stream_callback=stream_callback)

    # ==================== JSON 提取工具 ====================

    def _extract_json(self, raw: str) -> str:
        return extract_json(raw)

    # ==================== Handler: 图表生成 ====================

    def _handle_chart_generate(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        return self.handlers.data.chart_generate(user_message, attached_file, params)

    def _handle_chart_generate_fallback(
        self, user_message: str, attached_file: str, params: dict
    ) -> dict:
        return self.handlers.data.chart_generate_fallback(user_message, attached_file, params)

    @staticmethod
    def _build_smart_kpis(data, numeric_cols: list) -> list:
        return build_smart_kpis(data, numeric_cols)

    @staticmethod
    def _aggregate_chart_data(data, x_col: str, y_col: str, chart_type: str):
        return aggregate_chart_data(data, x_col, y_col, chart_type)

    @staticmethod
    def _build_column_overview(data) -> str:
        return build_column_overview(data)

    # ==================== Handler: 数据看板 ====================

    def _handle_dashboard_create(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        return self.handlers.data.dashboard_create(user_message, attached_file, params)

    # ==================== Handler: 综合可视化分析 ====================

    def _handle_visual_analysis(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        return self.handlers.data.visual_analysis(user_message, attached_file, params)

    # ==================== Handler: 提醒 ====================

    def _handle_reminder_set(self, user_message: str, params: dict, stream_callback=None) -> dict:
        return self.handlers.task.reminder_set(user_message, params, stream_callback=stream_callback)

    def _handle_reminder_check(self) -> dict:
        return self.handlers.task.reminder_check()

    def _handle_reminder_cancel(self, user_msg: str = "", params: dict = None) -> dict:
        return self.handlers.task.reminder_cancel(user_msg, params)

    # ==================== Handler: 负荷感知 ====================

    def _handle_workload_check(self, user_message: str = "", params: dict = None) -> dict:
        return self.handlers.task.workload_check(user_message, params)

    # ==================== Handler: 任务 ====================

    def _handle_task_inquiry(self, user_msg: str = "", stream_callback=None) -> dict:
        return self.handlers.task.task_inquiry(user_msg, stream_callback=stream_callback)

    def _handle_task_add(self, user_msg: str = "", params: dict = None) -> dict:
        return self.handlers.task.task_add(user_msg, params)

    # ==================== Handler: 记忆 ====================

    def _handle_memory_search(self, params: dict, session_id: Optional[str] = None, stream_callback=None) -> dict:
        return self.handlers.memory_handlers.memory_search(params, session_id, stream_callback=stream_callback)

    def _handle_memory_summarize(self, session_id: Optional[str] = None, stream_callback=None) -> dict:
        return self.handlers.memory_handlers.memory_summarize(session_id, stream_callback=stream_callback)

    # ==================== Handler: 简报 ====================

    def _handle_briefing(self, stream_callback=None) -> dict:
        return self.handlers.memory_handlers.briefing(orchestrator=self, stream_callback=stream_callback)

    # ==================== Handler: 通用对话 ====================

    def _handle_general_chat(
        self,
        msg: str,
        context: str,
        guidance_text: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
        attached_file: Optional[str] = None,
    ) -> dict:
        return self.handlers.chat.general_chat(
            msg, context, guidance_text, stream_callback, session_id, attached_file,
        )

    # ==================== 辅助方法 ====================

    def _get_content(self, filepath: Optional[str], msg: str, extracted_text: Optional[str] = None) -> str:
        return get_content(filepath, msg, extracted_text)

    def _auto_remember_personal_info(self, user_msg: str):
        self.background.auto_remember_personal_info(user_msg)

    def _extract_completed_task_name(self, user_msg: str) -> Optional[str]:
        return self.handlers.task.extract_completed_task_name(user_msg)

    def _detect_task_completion(self, user_msg: str) -> str:
        return self.handlers.task.detect_task_completion(user_msg)

    def _auto_extract_todos(self, user_msg: str) -> str:
        return self.background.auto_extract_todos(user_msg)

    def _handle_task_done(self, task_keyword: str) -> str:
        return self.handlers.task.task_done(task_keyword)

    def _build_group_status(self, group) -> str:
        return self.handlers.task.build_group_status(group)

    def _auto_detect_long_term_memory(self, user_msg: str):
        self.background.auto_detect_long_term_memory(user_msg)

    def _track_repeated_content(self, user_msg: str):
        self.background.track_repeated_content(user_msg)

    def _record_conversation(self, user_msg: str, assistant_msg: str, session_id: Optional[str] = None):
        self.conversation_logger.record(user_msg, assistant_msg, session_id)

    def _persist_conversation(self, user_msg: str, assistant_msg: str, session_id: Optional[str] = None):
        self.conversation_logger._persist(user_msg, assistant_msg, session_id)

    def _create_message_handler(self, agent_name: str) -> Callable:
        def handler(message: AgentMessage):
            logger.debug(f"[{agent_name}] received {message.type} from {message.sender}")
            agent = self.agents.get(agent_name)
            if not agent:
                return
            try:
                reply = agent.receive_message(message)
                if reply and isinstance(reply, dict):
                    agent.send_message(
                        receiver=message.sender,
                        msg_type="response",
                        payload=reply,
                        reply_to=message.id,
                    )
            except Exception as e:
                logger.warning(f"Agent {agent_name} message handler error: {e}")
        return handler

    def shutdown(self):
        logger.info("Shutting down Orchestrator...")
        reminder = self.agents.get("reminder_agent")
        if reminder and hasattr(reminder, 'calendar') and reminder.calendar:
            try:
                reminder.calendar.stop_scheduler()
            except Exception as e:
                logger.warning(f"Failed to stop reminder scheduler: {e}")
        for name, agent in self.agents.items():
            try:
                agent.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown agent {name}: {e}")
        logger.info("Orchestrator shutdown complete.")

    # ==================== 向后兼容：测试直接访问的内部属性 ====================

    @property
    def _repeated_content_tracker(self):
        if hasattr(self, 'background') and self.background is not None:
            return self.background.repeated_content_tracker
        return self.__dict__.get('_repeated_content_tracker', {})

    @_repeated_content_tracker.setter
    def _repeated_content_tracker(self, value):
        if hasattr(self, 'background') and self.background is not None:
            self.background._repeated_content_tracker = value
        else:
            self.__dict__['_repeated_content_tracker'] = value

    @property
    def _repeat_tracker_lock(self):
        if hasattr(self, 'background') and self.background is not None:
            return self.background.repeat_tracker_lock
        return self.__dict__.get('_repeat_tracker_lock', None)

    @_repeat_tracker_lock.setter
    def _repeat_tracker_lock(self, value):
        if hasattr(self, 'background') and self.background is not None:
            self.background._repeat_tracker_lock = value
        else:
            self.__dict__['_repeat_tracker_lock'] = value

    @property
    def _request_count(self):
        if hasattr(self, 'background') and self.background is not None:
            return self.background.request_count
        return self.__dict__.get('_request_count', 0)

    # ==================== 兼容旧 API ====================

    def execute_task_inquiry(self, msg: str = "") -> dict:
        return self._handle_task_inquiry(msg)

    # backward compat
    def _execute_task_inquiry(self, _msg: str = "") -> dict:
        """[deprecated] 兼容旧 API"""
        return self.execute_task_inquiry(_msg)

    def _execute_general_chat(self, msg: str, context: str) -> dict:
        """[deprecated] 兼容旧 API，请使用 _handle_general_chat"""
        return self._handle_general_chat(msg, context)
