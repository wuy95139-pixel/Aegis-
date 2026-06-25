"""
记忆类型定义
============
结合 Claude Code 记忆系统的类型分类与 Aegis 的数据模型优势。

四种记忆类型：
  - user     : 用户角色、偏好、知识水平
  - feedback : 用户对工作方式的反馈（正面和负面）
  - project  : 项目背景、目标、约束、进度
  - reference: 外部系统指针（文档、API、看板等）

每种类型有明确的：
  - 何时保存 (when_to_save)
  - 如何使用 (how_to_use)
  - 前端格式 (frontmatter schema)
"""

from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型枚举 — 对应 Claude Code 的四类记忆 + 经验类型"""
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"
    EXPERIENCE = "experience"  # 经验案例：记录成功/失败的操作经验


# ===================== 类型特定的 Frontmatter Schema =====================

class UserMemoryMetadata(BaseModel):
    """用户记忆的元数据 — 关于用户是谁、知道什么、偏好什么"""
    name: str = ""                              # 记忆名称
    description: str = ""                       # 一行描述，用于相关性判断
    type: MemoryType = MemoryType.USER
    role: Optional[str] = None                  # 用户角色
    expertise: List[str] = Field(default_factory=list)    # 专长领域
    preferences: List[str] = Field(default_factory=list)  # 偏好
    tools_and_stack: List[str] = Field(default_factory=list)  # 常用工具/技术栈


class FeedbackMemoryMetadata(BaseModel):
    """反馈记忆的元数据 — 用户如何希望被辅助"""
    name: str = ""
    description: str = ""
    type: MemoryType = MemoryType.FEEDBACK
    rule: str = ""                              # 反馈规则本身
    why: str = ""                               # 原因（过去的教训或偏好）
    how_to_apply: str = ""                      # 何时/何处应用此规则
    severity: str = "medium"                    # low / medium / high


class ProjectMemoryMetadata(BaseModel):
    """项目记忆的元数据 — 进行中的工作背景"""
    name: str = ""
    description: str = ""
    type: MemoryType = MemoryType.PROJECT
    fact: str = ""                              # 事实或决策
    why: str = ""                               # 动机（约束、截止日期、干系人要求）
    how_to_apply: str = ""                      # 如何影响建议和决策
    status: str = "active"                      # active / completed / superseded
    deadline: Optional[str] = None              # 相关截止日期


class ReferenceMemoryMetadata(BaseModel):
    """参考记忆的元数据 — 外部系统信息指针"""
    name: str = ""
    description: str = ""
    type: MemoryType = MemoryType.REFERENCE
    pointer: str = ""                           # URL、路径、频道名等
    system: str = ""                            # 外部系统名（GitHub, Slack, Linear 等）


class ExperienceMemoryMetadata(BaseModel):
    """经验记忆的元数据 — 记录操作的成功/失败经验"""
    name: str = ""
    description: str = ""
    type: MemoryType = MemoryType.EXPERIENCE
    situation: str = ""                         # 当时的情况
    approach: str = ""                          # 采取的方法
    outcome: str = ""                           # success / failure / partial
    lesson: str = ""                            # 提炼的教训
    context_tags: List[str] = Field(default_factory=list)


# ===================== 统一的记忆 Frontmatter =====================

class MemoryFrontmatter(BaseModel):
    """统一的记忆 Frontmatter — 所有类型共用，类型特定字段可选"""
    name: str
    description: str
    type: MemoryType

    # 用户类型字段
    role: Optional[str] = None
    expertise: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    tools_and_stack: List[str] = Field(default_factory=list)

    # 反馈类型字段
    rule: Optional[str] = None
    why: Optional[str] = None
    how_to_apply: Optional[str] = None
    severity: Optional[str] = None

    # 项目类型字段
    fact: Optional[str] = None
    status: Optional[str] = None
    deadline: Optional[str] = None

    # 参考类型字段
    pointer: Optional[str] = None
    system: Optional[str] = None

    # 经验类型字段
    situation: Optional[str] = None
    approach: Optional[str] = None
    outcome: Optional[str] = None
    lesson: Optional[str] = None
    context_tags: List[str] = Field(default_factory=list)

    # 通用字段
    tags: List[str] = Field(default_factory=list)
    importance: float = 0.5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ===================== 类型保存规则 =====================

# 每种类型的保存时机规则
TYPE_RULES: dict[MemoryType, dict] = {
    MemoryType.USER: {
        "when_to_save": [
            "了解用户的角色、职责、知识水平",
            "发现用户的偏好或工作习惯",
            "用户明确表示自己的背景或技术栈",
        ],
        "how_to_use": [
            "根据用户角色调整解释的深度和角度",
            "根据用户技术栈选择相关的类比和示例",
            "用用户熟悉的工具/概念来类比新知识",
        ],
        "examples": [
            "用户说：我做了十年 Go 开发，但这是第一次接触 React",
            "用户说：我是数据科学家，正在调研日志系统",
        ],
    },
    MemoryType.FEEDBACK: {
        "when_to_save": [
            "用户纠正了你的做法（'不要这样'、'停下来'）",
            "用户确认了某个非显而易见的做法（'对，就是这样'）",
            "用户对代码风格、提交信息、工作流程提出偏好",
        ],
        "how_to_use": [
            "在后续协作中遵循已确立的规则",
            "遇到类似场景时主动应用已学到的偏好",
            "当规则之间有冲突时，优先遵循更新近的反馈",
        ],
        "examples": [
            "用户说：不要 mock 数据库，上次出过生产事故",
            "用户说：这类重构我更喜欢一个大的 PR 而不是多个小的",
        ],
    },
    MemoryType.PROJECT: {
        "when_to_save": [
            "了解谁在做什么、为什么做、何时完成",
            "发现项目约束（截止日期、合规要求、技术限制）",
            "用户分享了计划的动机或背景",
        ],
        "how_to_use": [
            "根据项目约束过滤建议",
            "结合项目截止日期评估方案优先级",
            "所有建议需与项目目标对齐",
        ],
        "examples": [
            "用户说：周四之后所有非紧急 PR 冻结，移动端要发版",
            "用户说：重写认证中间件是因为法务要求合规",
        ],
    },
    MemoryType.REFERENCE: {
        "when_to_save": [
            "了解到外部系统的位置和用途",
            "发现文档、面板、看板的 URL",
            "知道某个资源在哪个平台管理",
        ],
        "how_to_use": [
            "当用户提到外部系统时，知道去哪里查找",
            "主动使用已知的外部资源获取最新信息",
        ],
        "examples": [
            "用户说：流水线 bug 都记在 Linear 的 INGEST 项目里",
            "用户说：oncall 关注 grafana.internal/d/api-latency 这个面板",
        ],
    },
    MemoryType.EXPERIENCE: {
        "when_to_save": [
            "完成了一次有明确结果的操作（成功或失败）",
            "用户给出了对操作结果的评价",
            "尝试了新方法并有了明确结论",
        ],
        "how_to_use": [
            "遇到类似情况时参考历史经验",
            "避免重复已经失败的方案",
            "优先复用验证过的成功方案",
        ],
        "examples": [
            "用户说：上次用接口抽象隔离后逐步替换，效果很好",
            "用户说：直接升级数据库那次出错就是因为没先在 staging 预演",
        ],
    },
}

# 不应保存为记忆的内容（来自 Claude Code）
DO_NOT_SAVE = [
    "代码模式、约定、架构 — 可以通过当前项目状态推导",
    "Git 历史、最近改动 — git log / git blame 是权威来源",
    "调试方案或修复方案 — 修正在代码里，提交信息有上下文",
    "CLAUDE.md 已记录的内容 — 避免重复",
    "临时的任务细节 — 进行中的工作、临时状态、当前对话上下文",
]


def get_type_rule(memory_type: MemoryType) -> dict:
    """获取指定类型的保存/使用规则"""
    return TYPE_RULES.get(memory_type, {})


def is_worth_remembering(content: str, memory_type: MemoryType) -> bool:
    """
    检查内容是否值得记住

    过滤掉临时性的、可从代码推导的、已记录的内容。
    """
    content_lower = content.lower()

    # 太短的内容不值得记（中文信息密度高，5字以上即可）
    if len(content) < 5:
        return False

    # 排除代码片段（启发式）
    code_indicators = [
        "def ", "class ", "import ", "function(", "=>", "const ",
        "git status", "git log", "git diff",
    ]
    for indicator in code_indicators:
        if indicator in content_lower:
            return False

    return True
