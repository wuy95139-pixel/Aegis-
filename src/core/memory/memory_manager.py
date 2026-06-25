"""
统一记忆管理器（增强版）
========================
结合 Claude Code 记忆系统（文件类型化）与 Aegis 记忆系统（向量检索 + LLM 提取）
的优势，提供统一的记忆操作入口。

新增能力（v2）：
  - 经验学习：记录每次操作的方法与结果，从成功/失败中学习
  - 偏好检测：识别显式和隐式用户偏好，累积置信度
  - 自适应行为：行动前综合经验+偏好+规则生成指导，行动后记录结果

架构：
  ┌──────────────────────────────────────────────────────┐
  │                  MemoryManager                         │
  │             (统一入口，单例模式)                         │
  ├──────────────────────────────────────────────────────┤
  │  基础层                                                │
  │  ├─ ShortTermMemory   │ 对话上下文缓冲区 (deque)        │
  │  ├─ LongTermMemory    │ 向量语义检索 (ChromaDB)        │
  │  ├─ FileStore         │ 文件持久化 (Markdown)          │
  │  ├─ IndexManager      │ 中央索引 (MEMORY.md)           │
  │  └─ MemoryRetriever   │ 混合检索器                      │
  ├──────────────────────────────────────────────────────┤
  │  学习层（NEW）                                          │
  │  ├─ ExperienceLibrary │ 经验库（成功/失败案例）          │
  │  ├─ PreferenceLearner │ 偏好学习器（显式+隐式）          │
  │  └─ AdaptiveEngine    │ 自适应引擎（学习闭环）           │
  ├──────────────────────────────────────────────────────┤
  │  LLM Provider         │ 关键点提取 + 上下文合成          │
  └──────────────────────────────────────────────────────┘

学习闭环：
  行动前: before_action() → 查经验+偏好+规则 → 生成指导 → 注入 LLM 提示
  行动后: after_action()  → 记录结果 → 更新经验库 → 强化/弱化偏好
"""

import uuid
import logging
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.file_store import FileStore
from src.core.memory.index_manager import IndexManager
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.experience import ExperienceLibrary, Outcome
from src.core.memory.preference import PreferenceLearner
from src.core.memory.adaptive import AdaptiveEngine, AdaptiveGuidance
from src.core.memory.types import (
    MemoryType, MemoryFrontmatter,
    is_worth_remembering,
)
from src.core.memory._memory_llm_ops import (
    extract_key_points as _extract_key_points_impl,
    classify_and_extract as _classify_and_extract_impl,
    synthesize_context as _synthesize_context_impl,
    summarize_text as _summarize_text_impl,
)
from src.models.schemas import MemoryEntry, ConversationTurn

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    统一记忆管理器 — 所有记忆操作的单入口

    使用示例:
        mm = MemoryManager(llm=llm_provider)
        mm.initialize("./data/memory", "./data/chroma_db")

        # 写入
        mm.remember(
            content="用户是 Go 后端工程师，偏好简洁回答",
            memory_type=MemoryType.USER,
            source="chat_001",
        )

        # 读取
        result = mm.recall("用户偏好什么?", memory_types=["user"])
        context = mm.recall("项目约束", memory_types=["project"])

        # 特定场景
        profile = mm.get_user_profile()
        rules = mm.get_feedback_rules()
    """

    _instance: Optional["MemoryManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式：全局只有一个 MemoryManager（线程安全）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            llm: LLMProvider 实例（用于提取和合成）
            config: 全局配置字典
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.llm = llm
            self.config = config or {}
            self._initialized = True

            # 子系统（延迟初始化）
            self.short_term: Optional[ShortTermMemory] = None
            self.long_term: Optional[LongTermMemory] = None
            self.file_store: Optional[FileStore] = None
            self.index_manager: Optional[IndexManager] = None
            self.retriever: Optional[MemoryRetriever] = None

            # 学习层
            self.experience_lib: Optional[ExperienceLibrary] = None
            self.preference_learner: Optional[PreferenceLearner] = None
            self.adaptive_engine: Optional[AdaptiveEngine] = None

    def initialize(
        self,
        file_store_dir: str = "./data/memory",
        chroma_dir: str = "./data/chroma_db",
        collection_name: str = "aegis_long_term_memory",
        embedding_model: str = "text-embedding-3-small",
        short_term_max_tokens: int = 16000,
        short_term_window: int = 20,
    ):
        """
        初始化所有记忆子系统

        Args:
            file_store_dir: 文件存储根目录
            chroma_dir: ChromaDB 持久化目录
            collection_name: ChromaDB 集合名
            embedding_model: 嵌入模型
            short_term_max_tokens: 短期记忆最大 token 数
            short_term_window: 短期记忆窗口大小
        """
        logger.info("Initializing MemoryManager...")

        # 1. 短期记忆（传入 LLM 以支持语义压缩）
        self.short_term = ShortTermMemory(
            max_tokens=short_term_max_tokens,
            window_size=short_term_window,
            llm=self.llm,
        )

        # 2. 长期记忆（向量数据库）
        self.long_term = LongTermMemory(
            persist_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        # 3. 文件存储
        self.file_store = FileStore(base_dir=file_store_dir)

        # 4. 索引管理器
        self.index_manager = IndexManager(base_dir=file_store_dir)

        # 5. 混合检索器
        self.retriever = MemoryRetriever(
            short_term=self.short_term,
            long_term=self.long_term,
            file_store=self.file_store,
        )

        # 6. 经验库
        self.experience_lib = ExperienceLibrary(
            file_store=self.file_store,
            long_term_memory=self.long_term,
            llm=self.llm,
        )

        # 7. 偏好学习器
        self.preference_learner = PreferenceLearner(
            file_store=self.file_store,
            llm=self.llm,
        )

        # 8. 自适应引擎（学习闭环）
        self.adaptive_engine = AdaptiveEngine(
            experience_library=self.experience_lib,
            preference_learner=self.preference_learner,
            memory_manager=self,
            llm=self.llm,
        )

        # 确保索引存在
        if self.file_store.total_count() > 0 and not self.index_manager.index_path.exists():
            self.index_manager.rebuild(self.file_store)

        # 校验索引完整性
        orphan, unindexed = self.index_manager.verify_integrity(self.file_store)
        if orphan > 0 or unindexed > 0:
            logger.info(f"Index integrity: {orphan} orphan entries, {unindexed} unindexed files")
            if unindexed > 0:
                self.index_manager.rebuild(self.file_store)

        logger.info(
            f"MemoryManager initialized: "
            f"file_store={self.file_store.total_count()} memories, "
            f"vector_store={self.long_term.count()} vectors, "
            f"short_term={len(self.short_term.get_context())} turns, "
            f"experiences={self.experience_lib.get_statistics().get('total', 0)}, "
            f"preferences={self.preference_learner.get_statistics().get('total_signals', 0)}"
        )

    # ==================== 写入操作 ====================

    def remember(
        self,
        content: str,
        memory_type: MemoryType,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        extract_key_points: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        记住一条信息（两阶段写入：文件 + 向量）

        Args:
            content: 要记忆的内容
            memory_type: 记忆类型 (user/feedback/project/reference)
            source: 来源标识
            tags: 分类标签
            importance: 重要性评分 (0-1)
            extract_key_points: 是否用 LLM 先提取关键点
            additional_metadata: 类型特定的额外元数据

        Returns:
            {"status": "success"/"error", "file_id": ..., "vector_id": ..., ...}
        """
        # 检查子系统是否已初始化
        if not self.file_store or not self.long_term:
            return {"status": "error", "reason": "MemoryManager not initialized — call initialize() first"}

        # 检查是否值得记住
        if not is_worth_remembering(content, memory_type):
            return {"status": "skipped", "reason": "content not worth remembering"}

        result = {
            "status": "success",
            "file_id": None,
            "vector_id": None,
            "key_points": [],
        }

        # 使用 LLM 提取关键点（如果可用）
        if extract_key_points and self.llm:
            key_points = self._extract_key_points(content, memory_type)
            result["key_points"] = key_points
            store_content = "\n\n".join(key_points) if key_points else content
        else:
            store_content = content

        # 生成 ID
        memory_id = str(uuid.uuid4())[:8]
        name = additional_metadata.get("name") if additional_metadata else None
        if not name:
            name = f"{memory_type.value}_{memory_id}"

        # 1. 写入 FileStore（SOURCE OF TRUTH）
        file_ok = False
        try:
            fm = self._build_frontmatter(
                name=name,
                memory_type=memory_type,
                content=store_content,
                tags=tags,
                importance=importance,
                additional_metadata=additional_metadata,
            )
            file_path = self.file_store.save(fm, store_content, update_index=True)
            result["file_id"] = str(file_path)
            result["name"] = name
            file_ok = True
        except Exception as e:
            logger.error(f"FileStore write failed: {e}")
            result["file_error"] = str(e)

        # 2. 写入 ChromaDB（SEARCH INDEX） — 仅在 FileStore 成功时
        if file_ok:
            try:
                entry = MemoryEntry(
                    id=memory_id,
                    content=store_content,
                    source=source,
                    tags=(tags or []) + [memory_type.value],
                    importance=importance,
                )
                vector_id = self.long_term.store(entry)
                result["vector_id"] = vector_id
            except Exception as e:
                logger.warning(f"ChromaDB write failed (non-fatal): {e}")
                result["vector_error"] = str(e)
                result["status"] = "partial"

        logger.info(f"Remembered [{memory_type.value}]: {name}")
        return result

    def remember_conversation(
        self,
        turns: List[ConversationTurn],
        source: str,
    ) -> Dict[str, Any]:
        """
        从对话中提取并存储多个类型的记忆

        自动区分：
          - 用户说出的个人信息 → MemoryType.USER
          - 用户给出的反馈/纠正 → MemoryType.FEEDBACK
          - 用户提到的项目信息 → MemoryType.PROJECT
          - 用户提到的外部资源 → MemoryType.REFERENCE

        Args:
            turns: 对话轮次列表
            source: 对话来源 ID

        Returns:
            存储结果汇总
        """
        if not self.llm:
            return {"status": "error", "message": "LLM required for conversation memory extraction"}

        # 构建对话文本
        conv_text = "\n".join(
            f"[{t.role}]: {t.content}" for t in turns
        )

        # 使用 LLM 分类提取
        extracted = self._classify_and_extract(conv_text)

        results = {"user": [], "feedback": [], "project": [], "reference": []}

        for category, items in extracted.items():
            mt = MemoryType(category)
            for item in items:
                r = self.remember(
                    content=item.get("content", ""),
                    memory_type=mt,
                    source=source,
                    tags=item.get("tags", []),
                    importance=item.get("importance", 0.5),
                    extract_key_points=False,  # 已经提取过了
                    additional_metadata=item.get("metadata", {}),
                )
                results[category].append(r)

        total = sum(len(v) for v in results.values())
        logger.info(f"Conversation memory extracted: {total} entries from {len(turns)} turns")
        return {"status": "success", "total": total, "results": results}

    # ==================== 读取操作 ====================

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        synthesize: bool = True,
    ) -> Dict[str, Any]:
        """
        检索相关记忆

        Args:
            query: 查询文本
            top_k: 返回条数
            memory_types: 按类型过滤
            tags: 按标签过滤
            min_importance: 最低重要性
            synthesize: 是否用 LLM 合成上下文

        Returns:
            检索结果字典，包含 combined_context
        """
        if not self.retriever:
            return {"relevant_memories": [], "file_memories": [], "combined_context": ""}

        result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            tags=tags,
            memory_types=memory_types,
            min_importance=min_importance,
        )

        # LLM 合成上下文
        if synthesize and self.llm and (result.get("file_memories") or result.get("relevant_memories")):
            result["synthesized_context"] = self._synthesize_context(query, result)

        return result

    def get_user_profile(self) -> Dict[str, Any]:
        """获取用户画像"""
        if not self.retriever:
            return {}
        return self.retriever.retrieve_user_profile()

    def get_feedback_rules(self) -> List[Dict[str, str]]:
        """获取所有用户反馈规则（用于行为指导）"""
        if not self.retriever:
            return []
        result = self.retriever.retrieve_feedback_rules()
        return result.get("feedback_rules", [])

    def get_project_context(self) -> Dict[str, Any]:
        """获取项目上下文"""
        if not self.retriever:
            return {}
        return self.retriever.retrieve_project_context()

    def get_references(self, system: Optional[str] = None) -> Dict[str, Any]:
        """获取外部参考"""
        if not self.retriever:
            return {}
        return self.retriever.retrieve_references(system=system)

    # ==================== 经验学习操作 ====================

    def record_experience(
        self,
        situation: str,
        approach: str,
        outcome: str = "success",
        lesson: str = "",
        context_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        记录一次操作经验

        Args:
            situation: 当时的情况
            approach: 采取的方法
            outcome: 结果 (success/failure/partial/unknown)
            lesson: 提炼的教训
            context_tags: 上下文标签

        Returns:
            记录结果
        """
        if not self.experience_lib:
            return {"status": "error", "message": "Experience library not initialized"}

        case = self.experience_lib.record(
            situation=situation,
            approach=approach,
            outcome=Outcome(outcome),
            lesson=lesson,
            context_tags=context_tags or [],
        )
        return {"status": "success", "case_id": case.case_id, "lesson": case.lesson}

    def record_success(
        self, situation: str, approach: str, lesson: str = "", **kwargs
    ) -> Dict[str, Any]:
        """快捷方法：记录成功经验"""
        return self.record_experience(situation, approach, "success", lesson, **kwargs)

    def record_failure(
        self, situation: str, approach: str, lesson: str = "", **kwargs
    ) -> Dict[str, Any]:
        """快捷方法：记录失败经验"""
        return self.record_experience(situation, approach, "failure", lesson, **kwargs)

    def get_relevant_experiences(
        self, situation: str, top_k: int = 5,
    ) -> Dict[str, Any]:
        """获取与当前情况相关的历史经验"""
        if not self.experience_lib:
            return {"experiences": [], "guidance_text": ""}

        return {
            "experiences": self.experience_lib.get_similar_situations(situation, top_k),
            "guidance_text": self.experience_lib.get_guidance_text(situation),
            "successful": [
                c.to_summary()
                for c in self.experience_lib.get_successful_approaches(situation, top_k=3)
            ],
            "failed": [
                c.to_summary()
                for c in self.experience_lib.get_failed_approaches(situation, top_k=3)
            ],
        }

    # ==================== 偏好学习操作 ====================

    def learn_preferences(
        self,
        user_message: str,
        assistant_response: str = "",
        user_reaction: str = "",
        context: str = "",
    ) -> Dict[str, Any]:
        """
        从交互中学习用户偏好

        Args:
            user_message: 用户消息
            assistant_response: 助手回复
            user_reaction: 用户对回复的反应
            context: 上下文

        Returns:
            学习到的偏好信号
        """
        if not self.preference_learner:
            return {"status": "error", "message": "Preference learner not initialized"}

        result = self.preference_learner.observe_interaction(
            user_message=user_message,
            assistant_response=assistant_response,
            user_reaction=user_reaction,
            context=context,
        )
        return {"status": "success", **result}

    def detect_preferences(
        self, message: str, context: str = "",
    ) -> List[Dict[str, Any]]:
        """从消息中检测偏好信号"""
        if not self.preference_learner:
            return []

        signals = self.preference_learner.detect_signals(message, context=context)
        return [s.to_dict() for s in signals]

    def get_active_preferences(self, min_confidence: float = 0.4) -> List[Dict[str, Any]]:
        """获取当前活跃的用户偏好"""
        if not self.preference_learner:
            return []

        return [
            s.to_dict()
            for s in self.preference_learner.get_active_preferences(min_confidence)
        ]

    def get_preference_prompt(self) -> str:
        """获取偏好提示文本（用于注入 LLM 上下文）"""
        if not self.preference_learner:
            return ""
        return self.preference_learner.get_preference_prompt()

    # ==================== 自适应行为操作 ====================

    def before_action(
        self, situation: str, action_type: str = "general",
    ) -> Dict[str, Any]:
        """
        行动前获取自适应指导

        综合经验库 + 偏好学习器 + 反馈规则，生成行动指导。

        Args:
            situation: 当前情况描述
            action_type: 行动类型 (code_refactor / code_generate / explain / search / file_ops / general)

        Returns:
            指导字典（含 text 和 items）
        """
        if not self.adaptive_engine:
            return {"guidance_text": "", "items": [], "is_empty": True}

        guidance = self.adaptive_engine.before_action(situation, action_type)
        return {
            "guidance_text": guidance.to_text(),
            "items": guidance.to_dict()["items"],
            "summary": guidance.summary,
            "confidence": guidance.confidence,
            "is_empty": guidance.is_empty(),
        }

    def after_action(
        self,
        situation: str,
        approach: str,
        outcome: str,
        user_feedback: str = "",
        context_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        行动后记录结果，完成学习闭环

        Args:
            situation: 行动时的情况
            approach: 采取的方法
            outcome: 结果 (success/failure/partial/unknown)
            user_feedback: 用户反馈
            context_tags: 上下文标签

        Returns:
            学习结果汇总
        """
        if not self.adaptive_engine:
            return {"status": "error", "message": "Adaptive engine not initialized"}

        return self.adaptive_engine.after_action(
            situation=situation,
            approach=approach,
            outcome=Outcome(outcome),
            user_feedback=user_feedback,
            context_tags=context_tags,
        )

    def learn_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
        user_reaction: str = "",
        context: str = "",
    ) -> Dict[str, Any]:
        """从一次完整交互中学习（便捷方法）"""
        if not self.adaptive_engine:
            return {"status": "error", "message": "Adaptive engine not initialized"}

        return self.adaptive_engine.learn_from_interaction(
            user_message=user_message,
            assistant_response=assistant_response,
            user_reaction=user_reaction,
            context=context,
        )

    def get_behavioral_prompt(
        self, situation: str = "", action_type: str = "general",
    ) -> str:
        """获取行为指导提示文本（用于注入 LLM 系统提示）"""
        if not self.adaptive_engine:
            return ""
        return self.adaptive_engine.get_behavioral_prompt(situation, action_type)

    # ==================== 维护操作 ====================

    def forget(self, name: str, memory_type: MemoryType) -> bool:
        """删除一条记忆（文件 + 向量 + 索引）"""
        success = True

        # 从 ChromaDB 删除（name 格式: "{type}_{id}"，提取 id）
        if self.long_term:
            prefix = f"{memory_type.value}_"
            if name.startswith(prefix):
                try:
                    self.long_term.delete(name[len(prefix):])
                except Exception as e:
                    logger.warning(f"ChromaDB delete failed for '{name}': {e}")

        # 从 FileStore 删除
        if self.file_store:
            if not self.file_store.delete(name, memory_type, update_index=False):
                success = False

        # 从索引删除
        if self.index_manager:
            self.index_manager.remove_entry(name)

        logger.info(f"Forgot: [{memory_type.value}] {name}")
        return success

    def forget_all_of_type(self, memory_type: MemoryType) -> int:
        """删除某类型的所有记忆"""
        if not self.file_store:
            return 0

        entries = self.file_store.list_by_type(memory_type)
        count = 0
        for entry in entries:
            name = entry["frontmatter"].name
            if self.forget(name, memory_type):
                count += 1
        return count

    def update_memory(
        self,
        name: str,
        memory_type: MemoryType,
        content: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """更新一条记忆（文件 + 向量 + 索引）"""
        if not self.file_store:
            return False

        success = self.file_store.update(name, memory_type, content, kwargs)

        # 同步 ChromaDB：删除旧向量后重新存储
        if success and content is not None and self.long_term:
            prefix = f"{memory_type.value}_"
            if name.startswith(prefix):
                memory_id = name[len(prefix):]
                try:
                    self.long_term.delete(memory_id)
                    entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        source=kwargs.get("source", ""),
                        tags=kwargs.get("tags", []) + [memory_type.value],
                        importance=kwargs.get("importance", 0.5),
                    )
                    self.long_term.store(entry)
                except Exception as e:
                    logger.warning(f"ChromaDB update failed for '{name}': {e}")

        if success and self.index_manager and kwargs.get("description"):
            self.index_manager.update_description(name, kwargs["description"])
        return success

    def rebuild_index(self) -> bool:
        """完全重建索引"""
        if not self.file_store or not self.index_manager:
            return False
        self.index_manager.rebuild(self.file_store)
        return True

    # ==================== 上下文获取 ====================

    def get_full_context(
        self,
        query: str,
        include_profile: bool = True,
        include_rules: bool = True,
        include_project: bool = True,
        include_experiences: bool = True,
        include_preferences: bool = True,
        include_behavioral: bool = True,
    ) -> str:
        """
        获取完整的增强上下文（用于注入 LLM 系统提示）

        合并：用户画像 + 反馈规则 + 项目上下文 + 相关记忆 + 最近对话
             + 历史经验 + 偏好学习 + 行为指导

        Args:
            query: 当前查询
            include_profile: 是否包含用户画像
            include_rules: 是否包含反馈规则
            include_project: 是否包含项目上下文
            include_experiences: 是否包含历史经验
            include_preferences: 是否包含学习到的偏好
            include_behavioral: 是否包含自适应行为指导

        Returns:
            格式化的上下文字符串
        """
        parts = []

        # 1. 行为指导（优先级最高：直接影响当前行动）
        if include_behavioral and self.adaptive_engine:
            behavioral = self.adaptive_engine.get_behavioral_prompt(query, "general")
            if behavioral:
                parts.append(behavioral)

        # 2. 用户偏好（学习到的）
        if include_preferences and self.preference_learner:
            pref_prompt = self.preference_learner.get_preference_prompt()
            if pref_prompt:
                parts.append(pref_prompt)

        # 3. 用户画像
        if include_profile:
            profile = self.get_user_profile()
            if profile.get("user_profile"):
                parts.append(profile["user_profile"])

        # 4. 反馈规则
        if include_rules:
            rules = self.get_feedback_rules()
            if rules:
                parts.append("## 用户偏好规则")
                for rule in rules:
                    parts.append(f"- [{rule.get('severity', 'medium').upper()}] {rule['rule']}")
                    if rule.get("why"):
                        parts.append(f"  原因: {rule['why']}")
                parts.append("")

        # 5. 项目上下文
        if include_project:
            proj = self.get_project_context()
            if proj.get("combined_context"):
                parts.append("## 项目背景")
                parts.append(proj["combined_context"])

        # 6. 历史经验
        if include_experiences and self.experience_lib:
            exp_text = self.experience_lib.get_guidance_text(query)
            if exp_text:
                parts.append(exp_text)

        # 7. 语义检索结果
        if query:
            result = self.recall(query, synthesize=True)
            if result.get("synthesized_context"):
                parts.append("## 相关历史上下文")
                parts.append(result["synthesized_context"])
            elif result.get("combined_context"):
                parts.append(result["combined_context"])

        # 8. MEMORY.md 索引摘要
        if self.index_manager:
            index_context = self.index_manager.get_context_string(max_entries_per_type=5)
            if index_context:
                parts.append(index_context)

        return "\n\n".join(parts)

    def get_llm_system_context(self, base_system_prompt: str = "") -> str:
        """
        构建完整的 LLM 系统提示（包含所有记忆上下文）

        设计：基础提示 + 记忆索引 + 反馈规则 + 用户画像
        """
        memory_context = self.get_full_context(
            query="",  # 不针对特定查询
            include_profile=True,
            include_rules=True,
            include_project=True,
        )

        if memory_context:
            return f"{base_system_prompt}\n\n---\n## 持久记忆\n{memory_context}"
        return base_system_prompt

    # ==================== 对话管理 ====================

    def add_conversation_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加对话轮次到短期记忆"""
        if self.short_term:
            turn = ConversationTurn(
                role=role,
                content=content,
                metadata=metadata or {},
            )
            self.short_term.add_turn(turn)

    def get_conversation_context(self, n: int = 10) -> List[ConversationTurn]:
        """获取最近对话"""
        if self.short_term:
            return self.short_term.get_context(n=n)
        return []

    def get_conversation_messages(self, system_prompt: str = "") -> List[Dict]:
        """获取 LLM 格式的对话消息"""
        if self.short_term:
            return self.short_term.get_messages_for_llm(system_prompt)
        return []

    def summarize_conversation(self) -> str:
        """LLM 总结当前对话"""
        if not self.short_term or not self.llm:
            return ""

        turns = self.short_term.get_context()
        if len(turns) < 4:
            return ""

        conv_text = "\n".join(f"[{t.role}]: {t.content}" for t in turns)
        return self._summarize_text(conv_text)

    # ==================== 统计信息 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        stats = {
            "short_term": {
                "turns": len(self.short_term.get_context()) if self.short_term else 0,
                "has_summary": bool(self.short_term.get_summary()) if self.short_term else False,
            },
            "long_term": {
                "total_vectors": self.long_term.count() if self.long_term else 0,
            },
            "file_store": self.file_store.get_stats() if self.file_store else {},
            "index": self.index_manager.get_stats() if self.index_manager else {},
            "experiences": self.experience_lib.get_statistics() if self.experience_lib else {},
            "preferences": self.preference_learner.get_statistics() if self.preference_learner else {},
            "adaptive": self.adaptive_engine.get_learning_summary() if self.adaptive_engine else {},
        }
        return stats

    # ==================== 内部方法 ====================

    def _build_frontmatter(
        self,
        name: str,
        memory_type: MemoryType,
        content: str,
        tags: Optional[List[str]],
        importance: float,
        additional_metadata: Optional[Dict[str, Any]],
    ) -> MemoryFrontmatter:
        """构建 MemoryFrontmatter 对象"""
        kwargs = {
            "name": name,
            "description": content[:100].replace("\n", " "),
            "type": memory_type,
            "tags": tags or [],
            "importance": importance,
        }

        if additional_metadata:
            # 合并类型特定字段
            for key in ["role", "expertise", "preferences", "tools_and_stack",
                        "rule", "why", "how_to_apply", "severity",
                        "fact", "status", "deadline",
                        "pointer", "system",
                        "situation", "approach", "outcome", "lesson", "context_tags"]:
                if key in additional_metadata:
                    kwargs[key] = additional_metadata[key]

        return MemoryFrontmatter(**kwargs)

    def _extract_key_points(self, text: str, memory_type: MemoryType) -> List[str]:
        return _extract_key_points_impl(self.llm, text, memory_type)

    def _classify_and_extract(self, conversation_text: str) -> Dict[str, List[Dict]]:
        return _classify_and_extract_impl(self.llm, conversation_text)

    def _synthesize_context(self, query: str, retrieval_result: dict) -> str:
        return _synthesize_context_impl(self.llm, query, retrieval_result)

    def _summarize_text(self, text: str) -> str:
        return _summarize_text_impl(self.llm, text)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<MemoryManager("
            f"files={stats['file_store'].get('total_memories', 0)}, "
            f"vectors={stats['long_term']['total_vectors']}, "
            f"turns={stats['short_term']['turns']}, "
            f"experiences={stats.get('experiences', {}).get('total', 0)}, "
            f"preferences={stats.get('preferences', {}).get('total_signals', 0)})>"
        )
