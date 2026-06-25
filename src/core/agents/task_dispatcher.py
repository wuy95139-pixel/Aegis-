"""
任务指派代理 (TaskDispatcherAgent)
================================
职责：
  1. 接收从文件/对话中提取的待办事项
  2. 按预设规则自动指派给相关人员
  3. 跟踪任务状态，发送提醒

协作关系：
  输入: 待办事项列表 (来自 FileProcessorAgent 或 MemoryAgent)
  输出: TodoItem 列表 (已指派) → 传递给 ReminderAgent 设置提醒

可扩展点：
  - 复杂指派规则引擎 (基于技能、负载、优先级)
  - 与企业 IM 集成 (钉钉/飞书/企业微信自动通知)
  - 任务看板同步 (Jira / Trello / Notion)
"""

import logging
from typing import Dict, Any, List, Optional

from src.core.agents.base import BaseAgent
from src.models.schemas import TodoItem, Person, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class TaskDispatcherAgent(BaseAgent):
    """任务指派代理 — 智能提取和分派待办事项"""

    role = "任务调度专家"
    goal = "精准从内容中提取待办事项，按照预设规则自动指派给最合适的人员，确保每项任务都有明确的责任人"
    backstory = """
你是一位高效的任务调度专家，擅长从复杂的信息中识别行动点并合理分配。
你需要：
- 从文件、对话、会议纪要中识别待办事项
- 根据人员角色、技能、负载和历史数据智能分配任务
- 确保每项任务都有明确的负责人、优先级和截止日期
- 对于无法自动指派的特殊任务，标记为需要人工审核

你的指派原则：
1. 优先匹配角色和专业领域
2. 考虑负载均衡，避免某个人承担过多
3. 紧急任务优先指派给当前可用人员
4. 标注不确定的指派以供人工确认
"""

    def __init__(self, llm, memory=None, config=None):
        super().__init__(
            name="task_dispatcher",
            llm=llm,
            memory=memory,
            tools=[],
            config=config,
        )
        # 人员库 — TODO: 从配置文件或数据库加载
        self._person_pool: List[Person] = self._load_person_pool()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务分派

        task_input 结构:
          {
            "todos": [
              {"title": "...", "description": "...", "assignee": "...", "deadline": "..."},
              ...
            ],
            "source": "meeting_notes_2024-06-01.docx",
            "auto_assign": true,
          }

        Returns:
          {
            "status": "success" | "error",
            "assigned_todos": [TodoItem, ...],
            "unassigned": [...],     # 无法自动分派的任务
          }
        """
        raw_todos = task_input.get("todos", [])
        source = task_input.get("source", "unknown")
        auto_assign = task_input.get("auto_assign", True)

        logger.info(f"TaskDispatcher received {len(raw_todos)} todos from {source}")

        assigned: List[TodoItem] = []
        unassigned: List[Dict] = []

        for raw_todo in raw_todos:
            todo = self._create_todo_item(raw_todo, source)

            if auto_assign:
                # 尝试自动分派
                assignee = self._find_assignee(todo)

                if assignee:
                    todo.assignee = assignee
                    todo.status = TaskStatus.PENDING
                    assigned.append(todo)
                    logger.info(f"Assigned: '{todo.title}' → {assignee.name}")
                else:
                    # 无法自动分派
                    unassigned.append(raw_todo)
                    logger.info(f"Unassigned: '{todo.title}' (no matching person)")
            else:
                # 不自动分派，仅创建 TodoItem
                assigned.append(todo)

        # 记忆：记录分派结果
        self.remember(
            content=f"任务分派: 从'{source}'指派了{len(assigned)}个任务，{len(unassigned)}个未分配",
            source=f"dispatch:{source}",
            tags=["task_dispatch", "todos"],
        )

        return {
            "status": "success",
            "assigned_todos": assigned,
            "unassigned": unassigned,
            "source": source,
            "summary": f"成功分派 {len(assigned)} 个任务，{len(unassigned)} 个待人工确认",
        }

    def _create_todo_item(self, raw: Dict[str, str], source: str) -> TodoItem:
        """将原始待办转为 TodoItem 模型"""
        # 优先级推断
        priority = TaskPriority.MEDIUM
        title_lower = raw.get("title", "").lower()
        if any(kw in title_lower for kw in ["紧急", "urgent", "asap", "立即"]):
            priority = TaskPriority.URGENT
        elif any(kw in title_lower for kw in ["重要", "important", "重点"]):
            priority = TaskPriority.HIGH

        # 截止日期解析 — 三阶段回退策略
        from datetime import datetime
        deadline = None
        deadline_str = raw.get("deadline", "")
        if deadline_str:
            # 第一阶段：中文自然语言时间解析 (如"明天下午3点"、"下周一")
            from src.core.tools.time_tools import parse_chinese_time_expression
            deadline = parse_chinese_time_expression(deadline_str)
            if deadline is None:
                # 第二阶段：ISO 格式日期
                try:
                    deadline = datetime.fromisoformat(deadline_str)
                except (ValueError, TypeError):
                    pass
            if deadline is None:
                # 第三阶段：dateparser 通用自然语言解析 (可选依赖)
                try:
                    import dateparser
                    deadline = dateparser.parse(deadline_str)
                    if deadline is not None:
                        logger.debug(f"dateparser parsed '{deadline_str}' -> {deadline}")
                except ImportError:
                    pass
            if deadline is None:
                logger.debug(f"Could not parse deadline: '{deadline_str}'")

        return TodoItem(
            title=raw.get("title", "未命名任务"),
            description=raw.get("description", ""),
            priority=priority,
            deadline=deadline,
            source_context=source,
        )

    def _find_assignee(self, todo: TodoItem) -> Optional[Person]:
        """
        为待办事项寻找合适的负责人

        匹配策略：
          1. 精确匹配：如果原始数据中指定了负责人名称
          2. 关键词匹配：根据任务标题中的关键词匹配角色
          3. 默认指派：返回默认人员

        可扩展点：
          - 基于 LLM 的语义匹配
          - 基于负载的智能分配
          - 基于历史数据的推荐
        """
        # TODO: 实现完整的匹配逻辑

        # 策略1: 直接匹配
        title_lower = todo.title.lower()

        # 策略2: 关键词 → 角色映射
        keyword_role_map = {
            "设计": ["设计师", "UI"],
            "开发": ["开发", "工程师"],
            "前端": ["前端工程师"],
            "后端": ["后端工程师"],
            "测试": ["测试工程师"],
            "运维": ["运维工程师"],
            "产品": ["产品经理"],
            "市场": ["市场专员"],
            "销售": ["销售经理"],
            "财务": ["财务专员"],
            "人事": ["HR"],
            "行政": ["行政专员"],
        }

        matched_roles = []
        for keyword, roles in keyword_role_map.items():
            if keyword in title_lower:
                matched_roles.extend(roles)

        # 在人员库中查找匹配的人员
        if matched_roles:
            for person in self._person_pool:
                if person.role in matched_roles:
                    return person
                if any(tag in matched_roles for tag in person.tags):
                    return person

        # 策略3: 默认指派
        default_name = self.config.get("agents", {}).get("task_dispatcher", {}).get("default_assignee", "")
        if default_name and default_name != "未分配":
            for person in self._person_pool:
                if person.name == default_name:
                    return person

        return None  # 无法自动分派

    def _load_person_pool(self) -> List[Person]:
        """
        加载人员库

        TODO: 从配置文件或数据库加载
        当前返回示例数据
        """
        return [
            Person(name="张三", email="zhangsan@example.com", role="产品经理", department="产品部", tags=["产品", "管理"]),
            Person(name="李四", email="lisi@example.com", role="前端工程师", department="技术部", tags=["前端", "React"]),
            Person(name="王五", email="wangwu@example.com", role="后端工程师", department="技术部", tags=["后端", "Python"]),
            Person(name="赵六", email="zhaoliu@example.com", role="设计师", department="设计部", tags=["UI", "设计"]),
            Person(name="陈七", email="chenqi@example.com", role="测试工程师", department="质量部", tags=["测试", "QA"]),
        ]

    def add_person(self, person: Person):
        """添加人员 (可扩展点)"""
        self._person_pool.append(person)

    def remove_person(self, name: str) -> bool:
        """移除人员"""
        self._person_pool = [p for p in self._person_pool if p.name != name]
        return True
