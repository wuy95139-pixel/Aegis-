"""
智能体基类
========
所有 Aegis 智能体的抽象基类，封装 CrewAI Agent 的创建和工具注册。

设计决策：
  - 使用组合模式：AegisAgent 包含 CrewAI Agent，而非继承
    这样可以在不修改 CrewAI 的情况下添加 Aegis 特有的能力
  - 每个 Agent 都持有对 LLM、Memory、Tools 的引用
  - 支持同步和异步执行模式

可扩展点：
  - 新增 Agent 类型：继承 BaseAgent，设置 role/goal/backstory 和 tools 即可
  - Agent 间通信：通过 AgentMessage 模型和 MessageBus
  - 流式输出：添加 stream=True 参数，使用 LLM 的流式接口
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from crewai import Agent as CrewAIAgent, Task as CrewAITask

from src.core.llm.provider import LLMProvider
from src.core.memory.retriever import MemoryRetriever
from src.models.schemas import AgentMessage, ConversationTurn

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Aegis 智能体基类

    每个子类需要定义:
      - role: str        — 角色名称
      - goal: str        — 目标
      - backstory: str   — 背景设定 (用于 LLM 的角色扮演)
      - tools: List      — 使用的工具列表
    """

    # 子类必须覆盖的属性
    role: str = "Base Agent"
    goal: str = "Base goal"
    backstory: str = "You are a helpful assistant."

    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        memory: Optional[MemoryRetriever] = None,
        tools: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            name: Agent 唯一名称 (如 "file_processor")
            llm: LLM 提供商
            memory: 记忆检索器
            tools: 工具列表
            config: 全局配置
        """
        self.name = name
        self.llm = llm
        self.memory = memory
        self.config = config or {}
        self._tools = tools or []

        # 创建 CrewAI Agent (底层执行引擎)
        self.crewai_agent = self._build_crewai_agent()

        # 消息 ID 计数器
        self._msg_counter = 0

        # 消息总线引用（由 Orchestrator 在构造后注入）
        self.message_bus: Optional[Any] = None

        logger.info(f"Agent initialized: {self.name} ({self.role})")

    def _build_crewai_agent(self) -> CrewAIAgent:
        """
        构建 CrewAI Agent 实例

        CrewAI Agent 定义角色、目标、工具等元数据，
        通过 _run_crew_task() 方法使用 CrewAI 的任务编排、委托和反思能力。
        """
        crewai_llm = self._create_crewai_llm()
        return CrewAIAgent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools=self._tools,
            llm=crewai_llm,
            verbose=False,
            allow_delegation=True,
            max_iter=10,
        )

    def _create_crewai_llm(self):
        """从 Aegis LLMProvider 创建 CrewAI 兼容的 LLM 对象"""
        try:
            from crewai import LLM as CrewLLM
            return CrewLLM(
                model=self.llm.default_model,
                api_key=self.llm.config.get("api_key", ""),
                base_url=self.llm.config.get("api_base", ""),
                temperature=self.llm.temperature,
            )
        except Exception as e:
            logger.warning(f"Failed to create CrewAI LLM from Aegis config: {e}. CrewAI will use its default.")
            return None

    def _run_crew_task(self, description: str, expected_output: str = "") -> str:
        """
        通过 CrewAI 运行任务 — 利用 CrewAI 的任务编排、自主反思和委托能力 (Issue 3)

        与直接调用 self.llm.chat() 不同，此方法经过完整的 CrewAI 流程:
          - 任务分解（Task decomposition）
          - 工具使用（Tool use）
          - 自主反思（Self-reflection）
          - 跨 Agent 委托（allow_delegation）

        当 CrewAI 不可用时（如测试环境无 API key），自动回退到直接 LLM 调用。

        Args:
            description: 任务描述
            expected_output: 期望的输出格式描述

        Returns:
            CrewAI 执行结果文本
        """
        from crewai import Task, Crew, Process

        try:
            task = Task(
                description=description,
                expected_output=expected_output or "用中文输出结果",
                agent=self.crewai_agent,
            )

            crew = Crew(
                agents=[self.crewai_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )

            result = crew.kickoff()
            return str(result) if result else ""
        except Exception as e:
            logger.warning(f"CrewAI execution failed, falling back to direct LLM (tools unavailable): {e}")
            response = self.llm.chat(
                messages=[{"role": "user", "content": description}],
                temperature=0.5,
                max_tokens=3000,
            )
            return response.get("content", "")

    @abstractmethod
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Agent 的核心逻辑

        每个子类必须实现此方法，定义该 Agent 的具体行为。

        Args:
            task_input: 任务输入数据，包含具体的指令和上下文

        Returns:
            任务执行结果
        """
        pass

    async def aexecute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步执行 (默认调用同步版本，子类可覆盖)

        Args:
            task_input: 任务输入数据

        Returns:
            任务执行结果
        """
        return self.execute(task_input)

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        与 Agent 对话的便捷方法

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词 (覆盖 Agent 的 backstory)
            context: 额外的上下文信息

        Returns:
            Agent 的回复文本
        """
        messages = [{"role": "system", "content": system_prompt or self.backstory}]

        if context:
            messages.append({"role": "system", "content": f"[上下文]\n{context}"})

        messages.append({"role": "user", "content": user_message})

        response = self.llm.chat(messages)
        return response["content"]

    def send_message(
        self,
        receiver: str,
        msg_type: str,
        payload: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> AgentMessage:
        """
        发送消息给其他 Agent

        设计决策：消息传递通过 Orchestrator 的 MessageBus 进行，
        这里生成消息对象，由 Orchestrator 负责路由。

        Args:
            receiver: 接收者 Agent 名称
            msg_type: 消息类型
            payload: 消息体
            reply_to: 回复的消息 ID

        Returns:
            AgentMessage
        """
        self._msg_counter += 1
        msg = AgentMessage(
            id=f"{self.name}_{self._msg_counter}_{uuid.uuid4().hex[:6]}",
            sender=self.name,
            receiver=receiver,
            type=msg_type,
            payload=payload,
            reply_to=reply_to,
        )
        # 发布到消息总线（如果已注入）
        if self.message_bus:
            self.message_bus.publish(msg)
        logger.debug(f"Agent message: {self.name} → {receiver} [{msg_type}]")
        return msg

    def remember(self, content: str, source: str, tags: Optional[List[str]] = None):
        """
        将重要信息存入长期记忆

        Args:
            content: 要记忆的内容
            source: 来源标识
            tags: 标签
        """
        if self.memory:
            entry_id = self.memory.extract_and_remember(content, source, tags)
            if entry_id:
                logger.info(f"Agent {self.name} remembered: {entry_id[:8]}...")

    def recall(self, query: str, top_k: int = 5) -> dict:
        """
        从记忆中检索相关信息

        Args:
            query: 查询
            top_k: 返回条数

        Returns:
            检索结果字典
        """
        if self.memory:
            return self.memory.retrieve(query, top_k=top_k)
        return {"relevant_memories": [], "recent_conversations": [], "combined_context": ""}

    def receive_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """
        接收来自其他 Agent 的消息（子类可覆盖以响应特定事件）

        默认行为：记录日志，不做任何处理。
        子类覆盖此方法以实现对特定事件的响应。

        Args:
            message: 收到的消息

        Returns:
            可选的回复数据字典。如果返回 dict，Orchestrator 会
            自动包装为 response 类型消息发回给发送者。
        """
        logger.debug(f"[{self.name}] received {message.type} from {message.sender}: {message.payload}")
        return None

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        将工具列表转为 LLM function calling 格式

        使用 _tool_registry 自动从已注册的 @tool 函数生成 OpenAI function schema。
        CrewAI Tool 也可通过此方法获取 schema。
        """
        from src.core.tools._tool_registry import get_tool_registry
        return get_tool_registry().get_all_schemas()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, role={self.role})>"
