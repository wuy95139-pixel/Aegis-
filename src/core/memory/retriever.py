"""
记忆检索器（增强版）
=================
混合检索器：同时查询短期记忆 (对话上下文)、长期记忆 (向量数据库)、
文件记忆存储 (Markdown 文件)，智能合并结果并提供给 Agent 使用。

增强点（结合 Claude Code 记忆系统优势）：
  - 三路检索：短期 + 向量 + 文件
  - 按记忆类型过滤（user / feedback / project / reference）
  - 重要性加权排序
  - LLM 合成上下文（来自 Aegis）
  - 文件存储降级方案（向量数据库不可用时）

设计决策：
  - 短期记忆提供会话连贯性 (最近说了什么)
  - 长期记忆提供跨会话语义知识 (以前讨论过什么)
  - 文件记忆提供结构化类型检索 (用户画像、反馈规则、项目背景、外部参考)
  - 检索时短记忆优先 (时效性)，长记忆补充 (持久性)，文件记忆补充 (类型化)
"""

import logging
from typing import List, Optional, Dict, Any

from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.file_store import FileStore
from src.core.memory.types import MemoryType
from src.models.schemas import MemoryEntry, ConversationTurn

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    混合记忆检索器（增强版）

    使用示例:
        retriever = MemoryRetriever(short_term_mem, long_term_mem, file_store)
        context = retriever.retrieve("用户偏好是什么?", top_k=5)
        # 带类型过滤
        context = retriever.retrieve("项目约束", top_k=5, memory_types=["project"])
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        file_store: Optional[FileStore] = None,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.file_store = file_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_recent_conversations: bool = True,
        tags: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> dict:
        """
        混合检索：三路并行查询

        Args:
            query: 查询文本
            top_k: 长期记忆返回条数
            include_recent_conversations: 是否包含最近对话
            tags: 按标签过滤
            memory_types: 按记忆类型过滤 ["user", "feedback", "project", "reference"]
            min_importance: 最低重要性阈值

        Returns:
            {
                "relevant_memories": [...],     # 长期记忆条目 (ChromaDB)
                "file_memories": [...],          # 文件记忆条目 (Markdown)
                "recent_conversations": [...],   # 最近对话
                "combined_context": "..."        # 合并后的上下文字符串
            }
        """
        result = {
            "relevant_memories": [],
            "file_memories": [],
            "recent_conversations": [],
            "combined_context": "",
        }

        # 解析 memory_types 为枚举
        type_enums = None
        if memory_types:
            try:
                type_enums = [MemoryType(t) for t in memory_types]
            except ValueError:
                logger.warning(f"Invalid memory_types: {memory_types}")

        # === 1. 向量搜索（ChromaDB） ===
        chromadb_available = self.long_term and self.long_term.count() > 0
        if chromadb_available:
            try:
                long_term_results = self.long_term.search(
                    query,
                    top_k=top_k,
                    tags=tags,
                )
                # 按重要性过滤
                if min_importance > 0:
                    long_term_results = [
                        m for m in long_term_results
                        if m.importance >= min_importance
                    ]
                # 按记忆类型过滤（如果向量数据库存储了类型信息）
                if type_enums:
                    long_term_results = [
                        m for m in long_term_results
                        if any(t.value in m.tags for t in type_enums)
                    ]
                result["relevant_memories"] = long_term_results
            except Exception as e:
                logger.warning(f"Long-term memory search failed: {e}")

        # === 2. 文件存储搜索（Markdown 文件）— 始终执行 ===
        # FileStore 是 SOURCE OF TRUTH，即使 ChromaDB 可用也要搜索
        if self.file_store:
            try:
                file_results = []
                search_types = type_enums if type_enums else list(MemoryType)

                for mt in search_types:
                    # 全文搜索（使用改进的多策略匹配）
                    text_matches = self.file_store.full_text_search(
                        query, mt, limit=top_k * 2  # 多取一些供排序
                    )
                    for fm_entry in text_matches:
                        if fm_entry not in file_results:
                            file_results.append(fm_entry)

                    # 标签搜索作为补充
                    if tags:
                        tagged = self.file_store.search_by_tags(tags, mt, limit=top_k)
                        for fm_entry in tagged:
                            if fm_entry not in file_results:
                                file_results.append(fm_entry)

                # 按重要性 + 时间排序
                file_results.sort(
                    key=lambda e: (
                        e["frontmatter"].importance,
                        str(e["frontmatter"].created_at or ""),
                    ),
                    reverse=True,
                )
                result["file_memories"] = file_results[:top_k]

                # ChromaDB 不可用时，file_memories 已包含所有结果
                # _handle_memory_search 会同时读取 relevant_memories 和 file_memories
                # 所以无需复制 — file_memories 就足够了
                if not chromadb_available and file_results:
                    logger.info(
                        f"ChromaDB unavailable, using FileStore fallback: "
                        f"{len(file_results[:top_k])} file memories found"
                    )
            except Exception as e:
                logger.warning(f"File store search failed: {e}")

        # === 3. 短期记忆 ===
        if include_recent_conversations and self.short_term:
            result["recent_conversations"] = self.short_term.get_context(n=10)

        # === 4. 合并为统一上下文 ===
        result["combined_context"] = self._build_combined_context(result)

        return result

    def retrieve_by_type(
        self,
        memory_type: str,
        query: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """
        按类型检索记忆（用于特定场景）

        Args:
            memory_type: 记忆类型 (user/feedback/project/reference)
            query: 可选查询文本
            top_k: 返回条数

        Returns:
            同 retrieve() 格式
        """
        return self.retrieve(
            query=query or "",
            top_k=top_k,
            memory_types=[memory_type],
        )

    def retrieve_user_profile(self) -> dict:
        """获取用户画像相关记忆"""
        result = self.retrieve("用户偏好 角色 技术栈 知识水平", top_k=10, memory_types=["user"])

        # 合成用户画像
        if result["file_memories"]:
            profile_parts = []
            for entry in result["file_memories"]:
                fm = entry["frontmatter"]
                profile_parts.append(f"- {fm.name}: {entry['content'][:200]}")

            result["user_profile"] = "## 用户画像\n" + "\n".join(profile_parts)
        else:
            result["user_profile"] = ""

        return result

    def retrieve_feedback_rules(self) -> dict:
        """获取用户反馈规则"""
        result = self.retrieve("", top_k=20, memory_types=["feedback"])

        rules = []
        if result["file_memories"]:
            for entry in result["file_memories"]:
                fm = entry["frontmatter"]
                severity = fm.severity or "medium"
                rule = fm.rule or entry["content"][:200]
                why = fm.why or ""
                how = fm.how_to_apply or ""
                rules.append({
                    "name": fm.name,
                    "rule": rule,
                    "why": why,
                    "how_to_apply": how,
                    "severity": severity,
                })

        result["feedback_rules"] = rules
        return result

    def retrieve_project_context(self) -> dict:
        """获取项目上下文"""
        return self.retrieve("项目目标 约束 截止日期 决策", top_k=10, memory_types=["project"])

    def retrieve_references(self, system: Optional[str] = None) -> dict:
        """获取外部参考"""
        result = self.retrieve("", top_k=10, memory_types=["reference"])

        if system and result["file_memories"]:
            result["file_memories"] = [
                e for e in result["file_memories"]
                if e["frontmatter"].system == system
            ]

        return result

    def extract_and_remember(
        self,
        content: str,
        source: str,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        importance: float = 0.5,
    ) -> Optional[str]:
        """
        从内容中提取关键信息并存入长期记忆和文件存储

        增强点：同时写入 ChromaDB 和 FileStore

        Args:
            content: 内容文本
            source: 来源标识
            tags: 标签
            memory_type: 记忆类型（user/feedback/project/reference）
            importance: 重要性评分

        Returns:
            存储的记忆 ID，失败返回 None
        """
        key_info = content[:1000]

        # 1. 存入 ChromaDB（向量检索用）
        entry = MemoryEntry(
            id="",
            content=key_info,
            source=source,
            tags=tags or [],
            importance=importance,
        )

        entry_id = None
        try:
            entry_id = self.long_term.store(entry)
            logger.info(f"Remembered (vector): {entry_id} from {source}")
        except Exception as e:
            logger.error(f"Failed to remember in vector DB: {e}")

        # 2. 存入 FileStore（文件持久化用）
        if self.file_store and memory_type:
            try:
                from src.core.memory.types import MemoryFrontmatter

                fm = MemoryFrontmatter(
                    name=f"mem_{entry_id or 'auto'}",
                    description=content[:100],
                    type=MemoryType(memory_type),
                    tags=tags or [],
                    importance=importance,
                )
                self.file_store.save(fm, content)
                logger.info(f"Remembered (file): {fm.name}")
            except Exception as e:
                logger.error(f"Failed to remember in file store: {e}")

        return entry_id

    # ==================== 内部方法 ====================

    def _build_combined_context(self, result: dict) -> str:
        """构建合并的上下文字符串"""
        parts = []

        # 文件记忆（优先级最高：结构化的关键信息）
        if result["file_memories"]:
            parts.append("## 相关记忆文件")
            for entry in result["file_memories"]:
                fm = entry["frontmatter"]
                type_label = fm.type.value if isinstance(fm.type, MemoryType) else (fm.type or "unknown")
                # 经验类型显示特殊标记
                if type_label == "experience":
                    outcome = getattr(fm, 'outcome', '') or ''
                    icon = {"success": "OK", "failure": "FAIL", "partial": "~"}.get(outcome, '')
                    type_label = f"experience:{icon}"
                parts.append(
                    f"- [{type_label.upper()}] {fm.name}: {entry['content'][:300]}"
                )

        # 向量检索结果
        if result["relevant_memories"]:
            if not result["file_memories"]:
                parts.append("## 相关历史记忆")
            for mem in result["relevant_memories"]:
                parts.append(f"- [{mem.source}] {mem.content[:300]}")

        # 最近对话
        if result["recent_conversations"]:
            parts.append("\n## 最近对话")
            for turn in result["recent_conversations"]:
                parts.append(f"[{turn.role}]: {turn.content[:200]}")

        return "\n".join(parts)
