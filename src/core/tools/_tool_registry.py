"""
工具注册表
==========
装饰器 + 注册表模式，支持从 Python 函数签名自动生成 OpenAI function-calling schema。

使用示例:
    from src.core.tools._tool_registry import tool, get_tool_registry

    @tool(description="获取当前日期和时间")
    def get_current_time() -> str:
        ...

    # 获取所有已注册工具的 OpenAI function-calling schema
    schemas = get_tool_registry().get_all_schemas()

设计决策:
  - 使用装饰器注册，零侵入现有代码
  - 从 Python 类型注解自动推导 JSON Schema 类型
  - 使用函数文档字符串作为 description
  - 线程安全的全局注册表单例
"""

import inspect
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, get_type_hints, get_origin, get_args

logger = logging.getLogger(__name__)

# Python type → JSON Schema type mapping
_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """将 Python 类型转为 JSON Schema 片段"""
    # Handle None / NoneType
    if py_type is type(None):
        return {"type": "null"}

    # Handle Optional[X] = Union[X, None]
    origin = get_origin(py_type)
    args = get_args(py_type)
    if origin is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _python_type_to_json_schema(non_none[0])
            schema["nullable"] = True
            return schema

    # Handle list types
    if origin is list or py_type is list:
        item_type = args[0] if args else str
        item_schema = _python_type_to_json_schema(item_type)
        return {"type": "array", "items": item_schema}

    # Handle dict types
    if origin is dict or py_type is dict:
        return {"type": "object"}

    # Handle basic types
    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    # Handle Enum types
    if inspect.isclass(py_type) and hasattr(py_type, "__members__"):
        return {"type": "string", "enum": list(py_type.__members__.keys())}

    # Default to string for unknown types
    return {"type": "string"}


class ToolRegistry:
    """全局工具注册表 — 线程安全单例"""

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tools: Dict[str, Dict[str, Any]] = {}
                    cls._instance._funcs: Dict[str, Callable] = {}
        return cls._instance

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        注册一个工具函数

        Args:
            func: 被注册的函数
            name: 工具名（默认使用函数名）
            description: 工具描述（默认使用 docstring 第一行）

        Returns:
            注册的工具名
        """
        tool_name = name or func.__name__

        # 提取描述
        if description:
            desc = description
        elif func.__doc__:
            desc = func.__doc__.strip().split("\n")[0].strip()
        else:
            desc = f"Call the {tool_name} function"

        # 从类型注解生成参数 schema
        try:
            hints = get_type_hints(func)
        except Exception as e:
            logger.debug("get_type_hints failed for %s: %s", tool_name, e)
            hints = {}

        # 过滤掉 return 注解
        params = {}
        required = []
        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_schema = {}
            if param_name in hints:
                param_schema = _python_type_to_json_schema(hints[param_name])
            else:
                param_schema = {"type": "string"}

            # 从 docstring 提取参数描述（:param name: description）
            if func.__doc__ and f":param {param_name}:" in func.__doc__:
                param_desc_match = func.__doc__.split(f":param {param_name}:", 1)
                if len(param_desc_match) > 1:
                    param_schema["description"] = param_desc_match[1].split("\n")[0].strip()

            params[param_name] = param_schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }

        with self._lock:
            self._tools[tool_name] = schema
            self._funcs[tool_name] = func

        logger.debug(f"Registered tool: {tool_name} ({len(params)} params)")
        return tool_name

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有已注册工具的 OpenAI function-calling schema 列表"""
        with self._lock:
            return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取已注册的工具函数"""
        with self._lock:
            return self._funcs.get(name)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        执行已注册的工具

        Args:
            name: 工具名
            arguments: 参数字典

        Returns:
            函数执行结果的字符串表示
        """
        func = self.get_tool(name)
        if func is None:
            return f"[错误] 未知工具: {name}"

        try:
            result = func(**arguments)
            return str(result) if result is not None else ""
        except Exception as e:
            logger.warning(f"Tool '{name}' execution failed: {e}")
            return f"[错误] 工具 '{name}' 执行失败: {e}"

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名"""
        with self._lock:
            return list(self._tools.keys())


# ---- 装饰器 ----

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    标记函数为 LLM 可调用工具的装饰器

    Args:
        name: 工具名（默认函数名）
        description: 工具描述（默认 docstring）
    """
    def decorator(func: Callable) -> Callable:
        get_tool_registry().register(func, name=name, description=description)
        return func
    return decorator


# ---- 便捷函数 ----

def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表单例"""
    return ToolRegistry()
