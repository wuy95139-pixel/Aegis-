"""
通用对话处理器
==============
从 Orchestrator 提取出的通用对话处理器，含 streaming 和 tool-calling 支持。
"""

import logging
from typing import Dict, Any, Optional, Callable

from src.core.tools.time_tools import get_time_context

logger = logging.getLogger(__name__)


class ChatHandler:
    """通用对话处理器"""

    def __init__(self, llm: "LLMProvider", memory: "MemoryRetriever", memory_manager: "MemoryManager | None" = None, session_manager: "SessionManager | None" = None, llm_interaction: "LLMInteraction | None" = None):
        self.llm = llm
        self.memory = memory
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self.llm_interaction = llm_interaction

    def general_chat(
        self,
        msg: str,
        context: str,
        guidance_text: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
        attached_file: Optional[str] = None,
    ) -> dict:
        time_context = get_time_context()

        enhanced_context = ""
        if self.memory_manager:
            try:
                enhanced_context = self.memory_manager.get_full_context(
                    query=msg,
                    include_profile=True,
                    include_rules=True,
                    include_project=True,
                    include_experiences=True,
                    include_preferences=True,
                    include_behavioral=True,
                )
            except Exception as e:
                logger.debug(f"Enhanced context fetch failed (non-fatal): {e}")

        system_prompt = f"""{time_context}

你是 Aegis，一个多智能体个人工作助理。你可以帮用户:
- 📄 处理文件（上传后让我翻译、润色、生成PPT、提取待办、语音转录）
- 🔍 研究分析（说"研究xxx"，我会搜索并生成SWOT报告）
- ⏰ 设置提醒（说"提醒我xxx"）
- 📋 管理待办（说"我有哪些待办"）
- 🧠 搜索记忆（说"之前聊过的xxx"）
- 📊 生成简报（说"生成简报"）

请用友好、简洁的中文回复。"""

        messages = [{"role": "system", "content": system_prompt}]

        if enhanced_context:
            messages.append({"role": "system", "content": f"[持久记忆与用户上下文]\n{enhanced_context}"})
        elif context:
            messages.append({"role": "system", "content": f"[相关历史上下文]\n{context}"})
        else:
            recent_context = ""
            stm = None
            session_mem = (
                self.session_manager.get_existing(session_id)
                if session_id and self.session_manager
                else None
            )
            if session_mem:
                stm = session_mem.short_term
            elif self.memory and hasattr(self.memory, 'short_term'):
                stm = self.memory.short_term

            if stm:
                turns = stm.get_context(n=6)
                if turns:
                    recent_context = "\n".join(
                        f"[{t.role}]: {t.content[:300]}" for t in turns
                    )
            if recent_context:
                messages.append({"role": "system", "content": f"[最近对话]\n{recent_context}"})

        if guidance_text:
            messages.append({"role": "system", "content": guidance_text})

        if attached_file:
            try:
                from src.core.tools.file_tools import parse_file
                from src.utils.common import sanitize_for_prompt
                parsed = parse_file(attached_file)
                if parsed.raw_text and not parsed.raw_text.startswith("[不支持"):
                    file_ctx = (
                        f"[用户已上传文件: {parsed.filename} ({parsed.file_type.value})]\n\n"
                        f"{sanitize_for_prompt(parsed.raw_text, max_len=8000)}"
                    )
                    messages.append({"role": "user", "content": file_ctx})
            except Exception as e:
                logger.warning(f"Failed to parse attached file for general_chat: {e}")

        messages.append({"role": "user", "content": msg})

        if stream_callback and self.llm_interaction:
            try:
                full_response = self.llm_interaction.stream_chat_with_tools(messages, stream_callback)
                return {"status": "success", "response": full_response, "streamed": True}
            except Exception as e:
                logger.warning(f"Streaming failed, falling back to non-streaming: {e}")

        if self.llm_interaction:
            reply = self.llm_interaction.run_with_tools(messages)
        else:
            reply = self.llm.chat(messages=messages).get("content", "")
        return {"status": "success", "response": reply}
