"""
处理器子模块
============
按领域划分的意图处理器集合，从 Orchestrator 提取。

每个模块定义处理器类，构造函数接收共享依赖（llm, agents, task_store 等），
方法签名与原来 Orchestrator 的 _handle_* 方法兼容。

通过 HandlerRegistry 统一管理和路由。
"""

from src.core.agents.handlers.file_handlers import FileHandlers
from src.core.agents.handlers.data_handlers import DataHandlers
from src.core.agents.handlers.task_handlers import TaskHandlers
from src.core.agents.handlers.memory_handlers import MemoryHandlers
from src.core.agents.handlers.chat_handler import ChatHandler

__all__ = [
    "FileHandlers",
    "DataHandlers",
    "TaskHandlers",
    "MemoryHandlers",
    "ChatHandler",
]
