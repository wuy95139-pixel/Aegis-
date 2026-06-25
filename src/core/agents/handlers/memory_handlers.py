"""
记忆与简报处理器
================
从 Orchestrator 提取出的记忆搜索、记忆总结、简报生成处理器。
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryHandlers:
    """记忆和简报相关所有意图处理器"""

    def __init__(self, llm: "LLMProvider", agents: dict, memory: "MemoryRetriever", session_manager: "SessionManager | None" = None):
        self.llm = llm
        self.agents = agents
        self.memory = memory
        self.session_manager = session_manager

    def memory_search(self, params: dict, session_id: Optional[str] = None, stream_callback=None) -> dict:
        query = params.get("query", "")
        if not query:
            return {"status": "error", "response": "请输入要搜索的关键词。"}

        session_mem = (
            self.session_manager.get_existing(session_id)
            if session_id and self.session_manager
            else None
        )
        if session_mem:
            mem_result = session_mem.retrieve(query, top_k=5)
        elif self.memory:
            mem_result = self.memory.retrieve(query, top_k=5)
        else:
            mem_result = {"relevant_memories": [], "file_memories": [], "recent_conversations": []}

        all_context = []

        for m in mem_result.get("relevant_memories", []):
            content = getattr(m, 'content', str(m))
            all_context.append(f"[长期记忆] {content}")

        for entry in mem_result.get("file_memories", []):
            fm = entry.get("frontmatter")
            content = entry.get("content", "")
            type_label = fm.type.value if fm and hasattr(fm.type, 'value') else "memory"
            all_context.append(f"[{type_label}] {content}")

        recent = mem_result.get("recent_conversations", [])
        for turn in recent[-6:]:
            all_context.append(f"[{turn.role}]: {turn.content}")

        if not all_context:
            return {"status": "success", "response": f"没有找到关于「{query}」的相关记忆。"}

        context_text = "\n---\n".join(c[:500] for c in all_context[-10:])
        prompt = f"""用户问: "{query}"

以下是从记忆系统中检索到的所有相关信息（包含长期记忆、文件记忆和最近对话）：

---
{context_text[:4000]}
---

请基于以上信息回答用户的问题。如果相关信息中有用户的名字、偏好、之前说过的话，请直接引用。
如果没有找到用户问的具体信息，请如实说明。用简洁的中文回答。"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        return {"status": "success", "response": response["content"].strip()}

    def memory_summarize(self, session_id: Optional[str] = None, stream_callback=None) -> dict:
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
            turns = stm.get_context()
            if turns:
                text = "\n".join(f"[{t.role}]: {t.content}" for t in turns)
                result = self.agents["memory_agent"].execute({
                    "operation": "summarize", "content": text,
                })
                return {
                    "status": "success",
                    "response": f"## 📝 对话总结 ({len(turns)} 轮)\n\n{result.get('summary', '')}"
                }

        return {"status": "success", "response": "暂无对话历史可供总结。"}

    def briefing(self, orchestrator=None, stream_callback=None) -> dict:
        from src.workflows.reminder_followup import run_morning_briefing
        result = run_morning_briefing(orchestrator=orchestrator)
        return {"status": "success", "response": f"## 📊 今日简报\n\n{result.get('briefing', '')}"}
