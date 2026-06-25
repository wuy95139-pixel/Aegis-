"""
对话记忆代理 (MemoryAgent)
==========================
职责：
  1. 管理短期对话上下文 (滑动窗口)
  2. 从对话中提取关键信息存入长期记忆
  3. 在回答问题时检索相关历史信息，使回答具有连续性和个性化

协作关系：
  输入: 用户消息 + 当前对话上下文
  输出: 增强后的上下文 (含历史记忆) → 供其他 Agent 使用

可扩展点：
  - 用户画像：从长期对话中构建用户偏好画像
  - 情感记忆：记住用户对某些话题的情感倾向
  - 知识图谱：构建实体关系图
"""

import logging
from typing import Dict, Any, List, Optional

from src.core.agents.base import BaseAgent
from src.models.schemas import ConversationTurn, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryAgent(BaseAgent):
    """对话记忆代理 — 管理和检索对话记忆"""

    role = "记忆管理专家"
    goal = "精确管理用户对话历史，在需要时检索最相关的上下文，确保每次回答都有据可依且体现个性化"
    backstory = """
你是一位细心的记忆管理专家，负责维护用户的对话记忆系统。
你需要：
- 记录每次对话的关键信息 (偏好、决策、重要事实)
- 在用户提问时，检索相关的历史信息作为回答的背景
- 识别需要长期记住的信息 (用户偏好、项目背景、常用联系人等)
- 对过时或矛盾的信息进行更新和修正

你的回答应该体现出对用户历史对话的了解，让用户感到被理解。
"""

    def __init__(self, llm, memory=None, config=None):
        super().__init__(
            name="memory_agent",
            llm=llm,
            memory=memory,
            tools=[],
            config=config,
        )
        # 用户画像缓存
        self._user_profile: Dict[str, Any] = {}

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行记忆相关操作

        task_input 结构:
          {
            "operation": "store" | "retrieve" | "summarize" | "profile",
            "content": "...",                      # 要存储的内容
            "query": "...",                        # 检索查询
            "source": "chat_001",                  # 来源
            "tags": ["preference", "project"],
          }

        Returns:
          根据操作返回不同结构
        """
        operation = task_input.get("operation", "retrieve")
        logger.info(f"MemoryAgent: operation={operation}")

        if operation == "store":
            return self._handle_store(task_input)
        elif operation == "retrieve":
            return self._handle_retrieve(task_input)
        elif operation == "summarize":
            return self._handle_summarize(task_input)
        elif operation == "profile":
            return self._handle_profile(task_input)
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}

    def _handle_store(self, task_input: Dict) -> Dict:
        """存储信息到长期记忆"""
        content = task_input.get("content", "")
        source = task_input.get("source", "manual")
        tags = task_input.get("tags", [])

        # 使用 LLM 提取关键信息点再存储
        key_points = self._extract_key_points(content)
        stored_ids = []

        if not self.memory:
            logger.warning("MemoryAgent._handle_store: self.memory is None, cannot store. "
                           "Check that MemoryRetriever was properly initialized.")
            return {"status": "error", "stored_count": 0, "error": "Memory system not available"}

        for point in key_points:
            entry_id = self.memory.extract_and_remember(point, source, tags)
            stored_ids.append(entry_id)

        # 广播记忆更新事件，通知其他 Agent（示范 Agent-to-Agent 通信）
        if stored_ids:
            self.send_message(
                receiver="broadcast",
                msg_type="event",
                payload={
                    "event": "memory_updated",
                    "source": source,
                    "tags": tags,
                    "count": len(stored_ids),
                },
            )

        return {
            "status": "success",
            "stored_count": len(stored_ids),
            "stored_ids": stored_ids,
            "key_points": key_points,
        }

    def _handle_retrieve(self, task_input: Dict) -> Dict:
        """检索相关记忆"""
        query = task_input.get("query", "")
        top_k = task_input.get("top_k", 5)
        tags = task_input.get("tags")

        # 会话隔离：优先使用 task_input 传入的 session_retriever
        session_retriever = task_input.get("session_retriever")
        if session_retriever:
            memory_result = session_retriever.retrieve(query, top_k=top_k)
        else:
            memory_result = self.recall(query, top_k=top_k)

        # 用 LLM 将检索到的记忆整合成连贯的上下文
        context = self._synthesize_context(query, memory_result)

        return {
            "status": "success",
            "query": query,
            "context": context,
            "relevant_memories": memory_result.get("relevant_memories", []),
            "recent_conversations": memory_result.get("recent_conversations", []),
        }

    def _handle_summarize(self, task_input: Dict) -> Dict:
        """总结对话历史"""
        content = task_input.get("content", "")
        summary = self._summarize_conversation(content)

        return {
            "status": "success",
            "summary": summary,
        }

    def _handle_profile(self, task_input: Dict) -> Dict:
        """获取或更新用户画像"""
        # TODO: 从长期记忆中聚合用户画像
        return {
            "status": "success",
            "profile": self._user_profile,
        }

    def _extract_key_points(self, text: str) -> List[str]:
        """
        使用 LLM 从文本中提取关键信息点

        避免存储冗余内容，只存储可复用的关键事实。
        """
        if not text or len(text) < 20:
            return [text] if text else []

        # TODO: LLM 提取关键点
        prompt = f"""从以下文本中提取 3-5 个关键信息点，每个信息点一句话概括。
聚焦于：用户偏好、决策事项、项目信息、联系人、重要日期等。

文本：
---
{text[:2000]}
---

每个信息点一行，不要编号。"""

        try:
            result_text = self._run_crew_task(
                description=prompt,
                expected_output="3-5个关键信息点，每行一个，不要编号",
            )
            points = [line.strip("-• ").strip() for line in result_text.strip().split("\n") if line.strip()]
            return points if points else [text[:200]]
        except Exception as e:
            logger.warning(f"Key point extraction failed: {e}")
            return [text[:200]]

    def _synthesize_context(self, query: str, memory_result: dict) -> str:
        """
        将检索到的记忆整合为自然语言上下文

        包含向量搜索结果 + 最近对话，确保对话连贯性。
        """
        parts = []

        # 1. 向量检索结果
        if memory_result.get("relevant_memories"):
            memories_text = "\n".join(
                f"- {m.content[:300]}" for m in memory_result["relevant_memories"][:5]
            )
            parts.append(f"历史记忆:\n{memories_text}")

        # 2. 最近对话（确保会话连贯性）
        if memory_result.get("recent_conversations"):
            conv_text = "\n".join(
                f"[{t.role}]: {t.content[:200]}"
                for t in memory_result["recent_conversations"][-10:]
            )
            parts.append(f"最近对话:\n{conv_text}")

        if not parts:
            return ""

        combined = "\n\n".join(parts)

        # 用 LLM 整合
        prompt = f"""用户正在问: "{query}"

根据以下相关的历史记忆和最近对话，生成一段简洁的上下文背景 (2-3 句话)：

{combined}

上下文背景："""

        try:
            result_text = self._run_crew_task(
                description=prompt,
                expected_output="2-3句简洁的上下文背景描述",
            )
            return result_text.strip()
        except Exception as e:
            logger.warning(f"Context synthesis failed: {e}")
            return combined

    def _summarize_conversation(self, text: str) -> str:
        """总结对话"""
        prompt = f"""请用 3-5 句话总结以下对话的关键内容：

---
{text[:4000]}
---

总结："""

        try:
            result_text = self._run_crew_task(
                description=prompt,
                expected_output="3-5句话的对话摘要",
            )
            return result_text.strip()
        except Exception:
            return ""
