"""
数据模型定义（增强版）
====================
使用 Pydantic 定义所有核心数据结构，确保类型安全和自动校验。

设计决策：
  - 所有跨 Agent 传递的数据都使用这些模型，避免 dict 满天飞
  - 模型尽量保持 flat，减少嵌套
  - 可扩展点：新增字段通过 Optional 字段实现向后兼容

增强（结合 Claude Code 记忆系统）：
  - TypedMemoryEntry: 支持 user/feedback/project/reference 四种记忆类型
  - FeedbackRule: 结构化的用户反馈规则
  - UserProfile: 聚合的用户画像
  - ProjectContext: 项目上下文快照
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


# ===================== 枚举 =====================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    PNG = "png"
    JPG = "jpg"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    GIF = "gif"
    TXT = "txt"
    MD = "md"
    UNKNOWN = "unknown"


class ReminderType(str, Enum):
    TIME_TRIGGERED = "time_triggered"
    EVENT_TRIGGERED = "event_triggered"
    RECURRING = "recurring"


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


# ===================== 用户/人员 =====================

class Person(BaseModel):
    """人员信息 — 用于任务指派"""
    name: str
    email: Optional[str] = None
    role: Optional[str] = None            # 角色/职位
    department: Optional[str] = None      # 部门
    tags: List[str] = Field(default_factory=list)  # 标签，用于规则匹配


# ===================== 文件 =====================

class ParsedFile(BaseModel):
    """解析后的文件内容"""
    filename: str
    file_type: FileType
    raw_text: str = ""                    # 提取的纯文本
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据 (作者、日期等)
    images: List[bytes] = Field(default_factory=list)       # 内嵌图片 (二进制)
    tables: List[List[List[str]]] = Field(default_factory=list)  # 提取的表格
    page_count: int = 0
    parsed_at: datetime = Field(default_factory=datetime.now)


class ColumnStats(BaseModel):
    """数值列的统计摘要"""
    name: str
    dtype: str = "string"  # "numeric" / "string"
    count: int = 0
    null_count: int = 0
    unique_count: int = 0
    # 数值列专用
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    # 前5个样本值
    sample_values: List[Any] = Field(default_factory=list)


class StructuredData(BaseModel):
    """结构化数据（CSV/Excel 解析结果）"""
    filename: str
    sheet_name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    column_stats: Dict[str, ColumnStats] = Field(default_factory=dict)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    truncated: bool = False
    file_encoding: Optional[str] = None


class TranslationRequest(BaseModel):
    """翻译请求"""
    source_text: str
    source_lang: str = "auto"             # 源语言 (auto=自动检测)
    target_lang: str = "zh-CN"
    glossary: Optional[Dict[str, str]] = None  # 术语表


class TranslationResult(BaseModel):
    """翻译结果"""
    translated_text: str
    source_lang_detected: str
    target_lang: str
    confidence: float = 1.0


# ===================== 任务 =====================

class TodoItem(BaseModel):
    """待办事项"""
    title: str
    description: Optional[str] = None
    assignee: Optional[Person] = None     # 指派给谁
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None
    source_context: Optional[str] = None  # 来源 (从哪段内容提取的)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)


# ===================== 对话记忆 =====================

class ConversationTurn(BaseModel):
    """单轮对话记录"""
    role: str                             # "user" | "assistant" | "agent"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(BaseModel):
    """长期记忆条目 — 存入向量数据库"""
    id: str
    content: str                          # 摘要后的内容
    embedding: Optional[List[float]] = None
    source: str                           # 来源 (对话ID / 文件名等)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    importance: float = 0.5               # 重要性评分 (0-1)
    similarity_score: Optional[float] = None  # 搜索时的余弦相似度 (供排序使用)


# ===================== 提醒 =====================

class Reminder(BaseModel):
    """提醒定义"""
    id: str
    type: ReminderType
    title: str
    description: Optional[str] = None
    trigger_time: Optional[datetime] = None     # 时间触发
    trigger_event: Optional[str] = None         # 事件触发 (如 "收到张三的回复")
    cron_expression: Optional[str] = None       # 循环表达式 (如 "0 9 * * 1-5")
    notify_method: List[str] = Field(default_factory=lambda: ["console"])
    created_at: datetime = Field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None   # 上次触发时间
    is_active: bool = True
    acknowledged: bool = False                  # 用户是否已确认
    snooze_minutes: int = 5                     # 未确认时几分钟后再次提醒
    fire_count: int = 0                         # 已触发次数（未确认的）
    max_fires: int = 5                          # 最大触发次数（超过后停止）


# ===================== 研究 =====================

class ResearchQuery(BaseModel):
    """研究查询"""
    topic: str
    keywords: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=lambda: ["web"])
    max_results: int = 10
    language: str = "zh-CN"


class ResearchSource(BaseModel):
    """研究来源条目"""
    title: str
    url: str
    snippet: str
    source_type: str                      # web / news / scholar
    published_at: Optional[datetime] = None
    relevance_score: float = 0.0


class SWOTAnalysis(BaseModel):
    """SWOT 分析"""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """结构化研究报告"""
    title: str
    executive_summary: str                # 摘要
    introduction: str                     # 背景介绍
    findings: List[str] = Field(default_factory=list)  # 分析发现
    swot: Optional[SWOTAnalysis] = None
    sources: List[ResearchSource] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)  # 建议
    generated_at: datetime = Field(default_factory=datetime.now)
    format: ReportFormat = ReportFormat.MARKDOWN


# ===================== 类型化记忆（来自 Claude Code 记忆系统） =====================

class MemoryTypeEnum(str, Enum):
    """记忆类型枚举"""
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class TypedMemoryEntry(BaseModel):
    """类型化的长期记忆条目 — 结合 Claude Code 类型 + Aegis 向量存储"""
    id: str = ""
    content: str                              # 记忆内容
    memory_type: MemoryTypeEnum = MemoryTypeEnum.USER
    source: str = ""                          # 来源标识
    tags: List[str] = Field(default_factory=list)
    importance: float = 0.5                   # 重要性 0-1
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    # 类型特定字段
    # user
    role: Optional[str] = None
    expertise: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)

    # feedback
    rule: Optional[str] = None
    why: Optional[str] = None
    how_to_apply: Optional[str] = None
    severity: Optional[str] = None            # low / medium / high

    # project
    fact: Optional[str] = None
    project_status: Optional[str] = None       # active / completed / superseded
    deadline: Optional[str] = None

    # reference
    pointer: Optional[str] = None              # URL、路径等
    external_system: Optional[str] = None      # 外部系统名


class FeedbackRule(BaseModel):
    """结构化的用户反馈规则"""
    name: str
    rule: str                                 # 规则内容
    why: str = ""                             # 原因
    how_to_apply: str = ""                    # 应用场景
    severity: str = "medium"                  # low / medium / high
    source: str = ""                          # 来源
    created_at: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """聚合的用户画像"""
    role: Optional[str] = None
    expertise: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    tools_and_stack: List[str] = Field(default_factory=list)
    communication_style: Optional[str] = None  # 偏好的沟通风格
    last_updated: datetime = Field(default_factory=datetime.now)
    sources: List[str] = Field(default_factory=list)  # 来源记忆 ID 列表


class ProjectContext(BaseModel):
    """项目上下文快照"""
    name: str = ""
    description: str = ""
    constraints: List[str] = Field(default_factory=list)
    deadlines: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    status: str = "active"
    last_updated: datetime = Field(default_factory=datetime.now)


# ===================== Agent 消息 =====================

# ===================== 任务分组管理（与记忆系统解耦） =====================

class TaskContextType(str, Enum):
    """任务上下文类型"""
    CURRENT_PROJECT = "current_project"
    OTHER = "other"
    ARCHIVED = "archived"


class TaskGroupSummary(BaseModel):
    """任务组的摘要信息"""
    group_id: str
    group_name: str
    context_type: TaskContextType = TaskContextType.CURRENT_PROJECT
    task_count: int = 0
    pending_count: int = 0
    completed_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = False


class AgentMessage(BaseModel):
    """智能体间消息 — Agent 通信的标准协议"""
    id: str
    sender: str                           # 发送者 Agent 名称
    receiver: str                         # 接收者 Agent 名称 (or "broadcast")
    type: str                             # 消息类型: "request" | "response" | "event"
    payload: Dict[str, Any]               # 消息体
    reply_to: Optional[str] = None        # 回复某条消息的 ID
    timestamp: datetime = Field(default_factory=datetime.now)
