"""
处理器注册中心
==============
聚合所有意图处理器子模块，统一管理依赖注入。

从 Orchestrator 提取的薄胶水层 — 构造所有 handler 实例，
Orchestrator 的 _handle_* 方法通过此注册中心找到对应的处理器。
"""

import logging

from src.core.agents.handlers.file_handlers import FileHandlers
from src.core.agents.handlers.data_handlers import DataHandlers
from src.core.agents.handlers.task_handlers import TaskHandlers
from src.core.agents.handlers.memory_handlers import MemoryHandlers
from src.core.agents.handlers.chat_handler import ChatHandler
from src.core.agents.handlers.research_handler import ResearchHandlers

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """聚合所有意图处理器，提供统一访问入口"""

    def __init__(
        self,
        llm: "LLMProvider",
        agents: dict,
        memory: "MemoryRetriever",
        translation_tool: "TranslationTool | None" = None,
        task_store: "TaskStore | None" = None,
        session_manager: "SessionManager | None" = None,
        memory_manager: "MemoryManager | None" = None,
        llm_interaction: "LLMInteraction | None" = None,
    ):
        self.file = FileHandlers(
            llm=llm,
            agents=agents,
            translation_tool=translation_tool,
            task_store=task_store,
            memory_manager=memory_manager,
        )
        self.data = DataHandlers(llm=llm)
        self.task = TaskHandlers(
            llm=llm,
            agents=agents,
            task_store=task_store,
            run_with_tools_fn=(
                llm_interaction.run_with_tools if llm_interaction else None
            ),
            memory_manager=memory_manager,
        )
        self.memory_handlers = MemoryHandlers(
            llm=llm,
            agents=agents,
            memory=memory,
            session_manager=session_manager,
        )
        self.chat = ChatHandler(
            llm=llm,
            memory=memory,
            memory_manager=memory_manager,
            session_manager=session_manager,
            llm_interaction=llm_interaction,
        )
        self.research = ResearchHandlers(llm=llm, agents=agents)

        # 保存引用供 orchestrator 透传
        self.agents = agents
        self.session_manager = session_manager
