"""
经验库
======
记录每次尝试的方法和结果，使系统能从成功和失败中学习。

核心理念：
  - 每次重要操作后记录：做了什么、结果如何、学到什么
  - 下次遇到类似情况时，检索相关经验作为决策参考
  - 成功经验被复用，失败经验被避开

数据结构：
  ExperienceCase:
    - situation: 当时的情况/上下文
    - approach: 采取的方法/策略
    - outcome: success / failure / partial / unknown
    - lesson: 提炼的教训（关键！这是可复用的知识）
    - context_tags: 便于相似性匹配的标签
    - confidence: 对这个教训的确信度

检索方式：
  1. 语义相似度匹配（通过 ChromaDB）
  2. 标签匹配
  3. 结果类型过滤（只看成功案例 / 只看失败教训）
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict

from src.models.schemas import MemoryEntry

logger = logging.getLogger(__name__)


class Outcome(str, Enum):
    """操作结果"""
    SUCCESS = "success"         # 完全成功
    FAILURE = "failure"         # 失败
    PARTIAL = "partial"         # 部分成功
    UNKNOWN = "unknown"         # 结果未知


class ExperienceCase:
    """
    一条经验记录

    设计：轻量级数据类，可直接序列化存储到 FileStore 和 ChromaDB
    """

    def __init__(
        self,
        situation: str,
        approach: str,
        outcome: Outcome,
        lesson: str = "",
        context_tags: Optional[List[str]] = None,
        source: str = "",
        confidence: float = 0.5,
        case_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.case_id = case_id or str(uuid.uuid4())[:8]
        self.situation = situation
        self.approach = approach
        self.outcome = outcome if isinstance(outcome, Outcome) else Outcome(outcome)
        self.lesson = lesson or self._derive_lesson()
        self.context_tags = context_tags or []
        self.source = source
        self.confidence = confidence
        self.created_at = created_at or datetime.now()

    def _derive_lesson(self) -> str:
        """当没有明确教训时，从结果推导"""
        if self.outcome == Outcome.SUCCESS:
            return f"方法有效: {self.approach[:200]}"
        elif self.outcome == Outcome.FAILURE:
            return f"应避免: {self.approach[:200]}"
        else:
            return f"需改进: {self.approach[:200]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "situation": self.situation,
            "approach": self.approach,
            "outcome": self.outcome.value,
            "lesson": self.lesson,
            "context_tags": self.context_tags,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_memory_content(self) -> str:
        """转为适合存储到记忆系统的文本"""
        return (
            f"情况: {self.situation}\n"
            f"方法: {self.approach}\n"
            f"结果: {self.outcome.value}\n"
            f"教训: {self.lesson}"
        )

    def to_summary(self) -> str:
        """一行摘要"""
        outcome_icon = {"success": "✓", "failure": "✗", "partial": "△", "unknown": "?"}
        icon = outcome_icon.get(self.outcome.value, "?")
        return f"[{icon}] {self.lesson[:120]}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceCase":
        return cls(
            case_id=data.get("case_id"),
            situation=data.get("situation", ""),
            approach=data.get("approach", ""),
            outcome=data.get("outcome", "unknown"),
            lesson=data.get("lesson", ""),
            context_tags=data.get("context_tags", []),
            source=data.get("source", ""),
            confidence=data.get("confidence", 0.5),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at") else None,
        )

    def __repr__(self):
        return f"<ExperienceCase({self.case_id}) [{self.outcome.value}] {self.lesson[:60]}>"


class ExperienceLibrary:
    """
    经验库 — 管理和检索经验案例

    使用示例:
        lib = ExperienceLibrary(file_store, long_term_memory)

        # 记录一次操作
        lib.record(
            situation="用户要求重构认证模块",
            approach="先用接口抽象隔离旧代码，再逐步替换",
            outcome=Outcome.SUCCESS,
            lesson="渐进式重构比一次性重写更安全",
            context_tags=["refactoring", "auth", "incremental"],
        )

        # 检索相关经验
        cases = lib.find_relevant("需要升级数据库schema")
        for case in cases:
            print(case.to_summary())

        # 获取针对当前情况的建议
        guidance = lib.get_guidance("需要修改核心API")
        # → {"recommend": [...], "avoid": [...], "because": [...]}
    """

    def __init__(self, file_store=None, long_term_memory=None, llm=None):
        """
        Args:
            file_store: FileStore 实例（用于文件持久化）
            long_term_memory: LongTermMemory 实例（用于向量检索）
            llm: LLMProvider 实例（用于智能提取教训）
        """
        self.file_store = file_store
        self.long_term = long_term_memory
        self.llm = llm

        # 经验缓存
        self._cases: Dict[str, ExperienceCase] = {}
        self._tag_index: Dict[str, List[str]] = defaultdict(list)

    # ==================== 记录操作 ====================

    def record(
        self,
        situation: str,
        approach: str,
        outcome: Outcome,
        lesson: str = "",
        context_tags: Optional[List[str]] = None,
        source: str = "",
        confidence: float = 0.5,
        auto_extract_lesson: bool = True,
    ) -> ExperienceCase:
        """
        记录一次经验

        Args:
            situation: 当时的情况
            approach: 采取的方法
            outcome: 结果
            lesson: 提炼的教训（留空则自动生成）
            context_tags: 上下文标签
            source: 来源
            confidence: 确信度
            auto_extract_lesson: 是否用 LLM 自动提炼教训

        Returns:
            ExperienceCase
        """
        # LLM 自动提炼教训
        if auto_extract_lesson and not lesson and self.llm:
            lesson = self._extract_lesson(situation, approach, outcome)

        case = ExperienceCase(
            situation=situation,
            approach=approach,
            outcome=outcome,
            lesson=lesson,
            context_tags=context_tags or [],
            source=source,
            confidence=confidence,
        )

        # 存入缓存
        self._cases[case.case_id] = case
        for tag in case.context_tags:
            self._tag_index[tag].append(case.case_id)

        # 持久化到 FileStore
        if self.file_store:
            try:
                from src.core.memory.types import MemoryFrontmatter, MemoryType
                fm = MemoryFrontmatter(
                    name=f"exp_{case.case_id}",
                    description=f"[{outcome.value}] {case.lesson[:80]}",
                    type=MemoryType.EXPERIENCE,
                    tags=["experience", outcome.value] + case.context_tags,
                    importance=case.confidence,
                    situation=case.situation,
                    approach=case.approach,
                    outcome=outcome.value,
                    lesson=case.lesson,
                )
                self.file_store.save(fm, case.to_memory_content())
            except Exception as e:
                logger.warning(f"Failed to persist experience to file: {e}")

        # 索引到 ChromaDB
        if self.long_term:
            try:
                entry = MemoryEntry(
                    id=f"exp_{case.case_id}",
                    content=case.to_memory_content(),
                    source=source or "experience_library",
                    tags=["experience", outcome.value] + case.context_tags,
                    importance=case.confidence,
                )
                self.long_term.store(entry)
            except Exception as e:
                logger.warning(f"Failed to index experience to vector DB: {e}")

        logger.info(
            f"Experience recorded: [{outcome.value}] {case.lesson[:80]}"
        )
        return case

    def record_success(
        self,
        situation: str,
        approach: str,
        lesson: str = "",
        **kwargs,
    ) -> ExperienceCase:
        """快捷方法：记录成功经验"""
        return self.record(situation, approach, Outcome.SUCCESS, lesson, **kwargs)

    def record_failure(
        self,
        situation: str,
        approach: str,
        lesson: str = "",
        **kwargs,
    ) -> ExperienceCase:
        """快捷方法：记录失败经验"""
        return self.record(situation, approach, Outcome.FAILURE, lesson, **kwargs)

    # ==================== 检索操作 ====================

    def find_relevant(
        self,
        situation: str,
        top_k: int = 5,
        outcome_filter: Optional[Outcome] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[ExperienceCase]:
        """
        查找与当前情况相关的经验

        检索策略：
          1. 向量语义搜索（ChromaDB）
          2. 标签匹配
          3. 文件全文搜索（降级方案）

        Args:
            situation: 当前情况描述
            top_k: 返回条数
            outcome_filter: 只看某种结果
            tags: 按标签过滤
            min_confidence: 最低确信度

        Returns:
            相关经验列表（按相关度排序）
        """
        results = []

        # 1. 向量搜索
        if self.long_term and self.long_term.count() > 0:
            try:
                search_tags = ["experience"]
                if outcome_filter:
                    search_tags.append(outcome_filter.value)
                if tags:
                    search_tags.extend(tags)

                entries = self.long_term.search(
                    query=situation,
                    top_k=top_k,
                    tags=search_tags,
                )

                for entry in entries:
                    case = self._entry_to_case(entry)
                    if case and case.confidence >= min_confidence:
                        results.append(case)
            except Exception as e:
                logger.warning(f"Vector search for experiences failed: {e}")

        # 2. 标签匹配（补充）
        if tags and self.file_store:
            try:
                from src.core.memory.types import MemoryType
                file_entries = self.file_store.search_by_tags(
                    tags=["experience"] + tags,
                    memory_type=MemoryType.EXPERIENCE,
                    limit=top_k,
                )
                for entry in file_entries:
                    case = ExperienceCase(
                        situation=entry["frontmatter"].why or "",
                        approach=entry["frontmatter"].how_to_apply or "",
                        outcome=Outcome.FAILURE,  # default
                        lesson=entry["content"][:200],
                        context_tags=entry["frontmatter"].tags or [],
                    )
                    if case.case_id not in {r.case_id for r in results}:
                        results.append(case)
            except Exception as e:
                logger.warning(f"File search for experiences failed: {e}")

        # 去重并按确信度排序
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.confidence, reverse=True):
            if r.case_id not in seen:
                seen.add(r.case_id)
                unique.append(r)

        return unique[:top_k]

    def get_successful_approaches(self, situation: str, top_k: int = 3) -> List[ExperienceCase]:
        """只获取成功的经验"""
        return self.find_relevant(situation, top_k=top_k, outcome_filter=Outcome.SUCCESS)

    def get_failed_approaches(self, situation: str, top_k: int = 3) -> List[ExperienceCase]:
        """只获取失败的经验（避免踩坑）"""
        return self.find_relevant(situation, top_k=top_k, outcome_filter=Outcome.FAILURE)

    def get_guidance(self, situation: str) -> Dict[str, Any]:
        """
        获取针对当前情况的行动指导

        综合成功经验和失败教训，生成建议。

        Returns:
            {
                "situation": str,
                "recommend": [...],   # 推荐做法
                "avoid": [...],       # 应避免的做法
                "because": [...],     # 原因（引用具体经验）
                "confidence": float,  # 综合确信度
                "has_relevant_experience": bool,
            }
        """
        successes = self.get_successful_approaches(situation, top_k=3)
        failures = self.get_failed_approaches(situation, top_k=3)

        guidance = {
            "situation": situation,
            "recommend": [],
            "avoid": [],
            "because": [],
            "confidence": 0.0,
            "has_relevant_experience": bool(successes or failures),
        }

        if not successes and not failures:
            guidance["because"].append("没有找到相关经验，这是新的情况。")
            return guidance

        for case in successes:
            guidance["recommend"].append(case.approach)
            guidance["because"].append(f"[成功经验] {case.lesson}")

        for case in failures:
            guidance["avoid"].append(case.approach)
            guidance["because"].append(f"[失败教训] {case.lesson}")

        # 综合确信度 = 平均
        all_cases = successes + failures
        if all_cases:
            guidance["confidence"] = sum(c.confidence for c in all_cases) / len(all_cases)

        return guidance

    def get_guidance_text(self, situation: str) -> str:
        """获取自然语言格式的行动指导（用于注入 LLM 上下文）"""
        g = self.get_guidance(situation)

        if not g["has_relevant_experience"]:
            return ""

        parts = ["## 历史经验参考"]

        if g["recommend"]:
            parts.append("\n### 推荐做法")
            for i, rec in enumerate(g["recommend"]):
                because = g["because"][i] if i < len(g["because"]) else ""
                parts.append(f"- {rec}")
                if because:
                    parts.append(f"  *{because}*")

        if g["avoid"]:
            parts.append("\n### 应避免")
            for i, av in enumerate(g["avoid"]):
                because = g["because"][len(g["recommend"]) + i] if len(g["recommend"]) + i < len(g["because"]) else ""
                parts.append(f"- ❌ {av}")
                if because:
                    parts.append(f"  *{because}*")

        return "\n".join(parts)

    def get_similar_situations(self, situation: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        找到最相似的历史情况（无论成败）

        Returns:
            [{"situation": ..., "outcome": ..., "similarity": ...}, ...]
        """
        cases = self.find_relevant(situation, top_k=top_k)
        return [
            {
                "situation": c.situation,
                "approach": c.approach,
                "outcome": c.outcome.value,
                "lesson": c.lesson,
                "case_id": c.case_id,
            }
            for c in cases
        ]

    # ==================== 统计 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取经验统计"""
        all_cases = list(self._cases.values())

        # 也尝试从文件存储加载
        if self.file_store and not all_cases:
            all_cases = self._load_cases_from_files()

        if not all_cases:
            return {"total": 0, "by_outcome": {}, "top_tags": []}

        by_outcome = defaultdict(int)
        tag_counts = defaultdict(int)

        for case in all_cases:
            by_outcome[case.outcome.value] += 1
            for tag in case.context_tags:
                tag_counts[tag] += 1

        total = len(all_cases)
        success_rate = by_outcome.get("success", 0) / total if total > 0 else 0

        return {
            "total": total,
            "by_outcome": dict(by_outcome),
            "success_rate": round(success_rate, 2),
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "recent_lessons": [
                c.to_summary()
                for c in sorted(all_cases, key=lambda x: x.created_at or datetime.min, reverse=True)[:5]
            ],
        }

    # ==================== 内部方法 ====================

    def _extract_lesson(self, situation: str, approach: str, outcome: Outcome) -> str:
        """使用 LLM 从情况+方法+结果中提炼教训"""
        if not self.llm:
            return ""

        prompt = f"""从以下经验中提炼一条简洁的教训（一句话）：

情况：{situation[:300]}
方法：{approach[:300]}
结果：{outcome.value}

教训应该：
- 如果成功：说明为什么这个方法有效，什么情况下可以复用
- 如果失败：说明失败的根本原因，如何避免
- 聚焦于可迁移的原则，而非具体细节

教训："""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response["content"].strip()
        except Exception as e:
            logger.warning(f"Lesson extraction failed: {e}")
            return ""

    def _entry_to_case(self, entry: MemoryEntry) -> Optional[ExperienceCase]:
        """将 MemoryEntry 转为 ExperienceCase"""
        try:
            content = entry.content
            outcome = Outcome.UNKNOWN

            if "结果: success" in content:
                outcome = Outcome.SUCCESS
            elif "结果: failure" in content:
                outcome = Outcome.FAILURE
            elif "结果: partial" in content:
                outcome = Outcome.PARTIAL

            # 解析结构化内容
            situation = ""
            approach = ""
            lesson = ""

            for line in content.split("\n"):
                if line.startswith("情况:"):
                    situation = line[3:].strip()
                elif line.startswith("方法:"):
                    approach = line[3:].strip()
                elif line.startswith("教训:"):
                    lesson = line[3:].strip()

            return ExperienceCase(
                case_id=entry.id.replace("exp_", ""),
                situation=situation or entry.content[:200],
                approach=approach or "",
                outcome=outcome,
                lesson=lesson or entry.content[:200],
                context_tags=[t for t in entry.tags if t not in ("experience", outcome.value)],
                source=entry.source,
                confidence=entry.importance,
            )
        except Exception as e:
            logger.debug(f"Failed to convert entry to case: {e}")
            return None

    def _load_cases_from_files(self) -> List[ExperienceCase]:
        """从 FileStore 加载所有经验案例"""
        if not self.file_store:
            return []

        cases = []
        try:
            from src.core.memory.types import MemoryType
            entries = self.file_store.search_by_tags(
                ["experience"], memory_type=MemoryType.EXPERIENCE, limit=200
            )
            for entry in entries:
                fm = entry["frontmatter"]
                outcome_str = fm.outcome or "unknown"
                if not outcome_str or outcome_str == "unknown":
                    for tag in fm.tags:
                        if tag in ("success", "failure", "partial"):
                            outcome_str = tag
                            break

                case = ExperienceCase(
                    case_id=fm.name.replace("exp_", ""),
                    situation=fm.situation or "",
                    approach=fm.approach or "",
                    outcome=Outcome(outcome_str),
                    lesson=fm.lesson or entry["content"][:200],
                    context_tags=fm.tags or [],
                    confidence=fm.importance,
                )
                cases.append(case)
                self._cases[case.case_id] = case
        except Exception as e:
            logger.warning(f"Failed to load cases from files: {e}")

        return cases

    def __repr__(self):
        stats = self.get_statistics()
        return f"<ExperienceLibrary(total={stats['total']}, success_rate={stats['success_rate']})>"
