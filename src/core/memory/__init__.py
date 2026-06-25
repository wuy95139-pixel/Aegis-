"""
Aegis 记忆系统（增强版 v2）
===========================
结合 Claude Code 记忆系统（文件类型化、Markdown 持久化、中央索引）
与 Aegis 记忆系统（向量检索、LLM 提取合成、双层级架构）的优势。

新增 v2 能力：
  - 经验学习 (ExperienceLibrary): 记录每次操作的成功/失败，相似情况检索
  - 偏好学习 (PreferenceLearner): 检测显式+隐式偏好，置信度累积，矛盾解决
  - 自适应引擎 (AdaptiveEngine): 行动前指导 + 行动后记录 = 持续改进闭环

模块结构：
  types.py          — 记忆类型定义（user / feedback / project / reference）
  short_term.py     — 短期对话缓冲区（滑动窗口 + 自动压缩）
  long_term.py      — 长期向量存储（ChromaDB + 语义检索）
  file_store.py     — 文件持久化存储（Markdown + YAML frontmatter）
  index_manager.py  — MEMORY.md 中央索引管理
  retriever.py      — 三路混合检索器（短期 + 向量 + 文件）
  experience.py     — 经验库（成功/失败案例学习）              [NEW]
  preference.py     — 偏好学习器（显式+隐式偏好检测）         [NEW]
  adaptive.py       — 自适应行为引擎（学习闭环）              [NEW]
  memory_manager.py — 统一入口（单例，编排所有子系统）

快速使用：
  from src.core.memory import MemoryManager, MemoryType

  mm = MemoryManager(llm=llm_provider)
  mm.initialize()

  # 基础：记住和回忆
  mm.remember("用户是 Go 工程师", memory_type=MemoryType.USER)
  result = mm.recall("用户偏好什么?")

  # 进阶：经验学习
  mm.record_success("重构认证模块", "先用接口隔离再逐步替换",
                     lesson="渐进式重构比一次性重写更安全")
  experiences = mm.get_relevant_experiences("要升级数据库schema")

  # 进阶：偏好检测
  mm.learn_preferences(
      user_message="能不能不要每次都在末尾加总结？",
      assistant_response="好的，我会注意。",
      user_reaction="对，就这样",
  )
  active_prefs = mm.get_active_preferences()

  # 进阶：自适应闭环
  guidance = mm.before_action("用户要求修改核心API", action_type="code_refactor")
  # ... 执行操作 ...
  mm.after_action("用户要求修改核心API", "用适配器模式隔离",
                  outcome="success", user_feedback="这次改得很好")

  # 获取完整 LLM 上下文
  full_context = mm.get_full_context(query="当前问题")
"""

from src.core.memory.types import (
    MemoryType,
    MemoryFrontmatter,
    UserMemoryMetadata,
    FeedbackMemoryMetadata,
    ProjectMemoryMetadata,
    ReferenceMemoryMetadata,
    ExperienceMemoryMetadata,
    TYPE_RULES,
    DO_NOT_SAVE,
    get_type_rule,
    is_worth_remembering,
)
from src.core.memory.short_term import ShortTermMemory
from src.core.memory.long_term import LongTermMemory
from src.core.memory.file_store import FileStore
from src.core.memory.index_manager import IndexManager
from src.core.memory.retriever import MemoryRetriever
from src.core.memory.experience import (
    ExperienceLibrary,
    ExperienceCase,
    Outcome,
)
from src.core.memory.preference import (
    PreferenceLearner,
    PreferenceSignal,
    SignalType,
)
from src.core.memory.adaptive import (
    AdaptiveEngine,
    AdaptiveGuidance,
    GuidanceLevel,
)
from src.core.memory.memory_manager import MemoryManager

__all__ = [
    # 类型
    "MemoryType",
    "MemoryFrontmatter",
    "UserMemoryMetadata",
    "FeedbackMemoryMetadata",
    "ProjectMemoryMetadata",
    "ReferenceMemoryMetadata",
    "ExperienceMemoryMetadata",
    "TYPE_RULES",
    "DO_NOT_SAVE",
    "get_type_rule",
    "is_worth_remembering",
    # 存储层
    "ShortTermMemory",
    "LongTermMemory",
    "FileStore",
    "IndexManager",
    # 检索
    "MemoryRetriever",
    # 经验学习 [NEW]
    "ExperienceLibrary",
    "ExperienceCase",
    "Outcome",
    # 偏好学习 [NEW]
    "PreferenceLearner",
    "PreferenceSignal",
    "SignalType",
    # 自适应引擎 [NEW]
    "AdaptiveEngine",
    "AdaptiveGuidance",
    "GuidanceLevel",
    # 统一管理
    "MemoryManager",
]
