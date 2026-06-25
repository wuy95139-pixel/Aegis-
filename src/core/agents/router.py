"""
路由表 (Router)
===============
从 Orchestrator 水平拆分出的意图路由模块。

职责:
  - 维护意图 → 处理器方法名的映射表
  - 构建处理器 lambda 字典供 Orchestrator 使用

使用:
    from src.core.agents.router import build_handlers

    handlers = build_handlers(
        orchestrator=self,
        user_message=user_message,
        attached_file=attached_file,
        params=params,
        session_id=session_id,
    )
    handler = handlers.get(intent)
    if handler:
        result = handler()
"""

from typing import Any, Callable, Dict, Optional

# 意图 → Orchestrator 处理器方法名
# 注意：值是 Orchestrator 实例上的方法名，由 build_handlers() 通过 getattr 调用
ROUTE_TABLE: Dict[str, str] = {
    "file_parse":          "_handle_file_parse",
    "file_translate":      "_handle_translate",
    "file_polish":         "_handle_polish",
    "file_generate_ppt":   "_handle_generate_ppt",
    "file_extract_todos":  "_handle_file_extract_todos",
    "file_qa":             "_handle_file_qa",
    "audio_transcribe":    "_handle_audio_transcribe",
    "research":            "_handle_research",
    "reminder_set":        "_handle_reminder_set",
    "reminder_check":      "_handle_reminder_check",
    "reminder_cancel":     "_handle_reminder_cancel",
    "task_add":            "_handle_task_add",
    "task_inquiry":        "_handle_task_inquiry",
    "memory_search":       "_handle_memory_search",
    "memory_summarize":    "_handle_memory_summarize",
    "briefing":            "_handle_briefing",
    "chart_generate":      "_handle_chart_generate",
    "dashboard_create":    "_handle_dashboard_create",
    "visual_analysis":     "_handle_visual_analysis",
    "workload_check":      "_handle_workload_check",
    "general_chat":        "_handle_general_chat",
}

# 需要 user_message 参数的方法
_REQUIRES_USER_MESSAGE = {
    "_handle_translate", "_handle_polish", "_handle_generate_ppt",
    "_handle_reminder_set", "_handle_reminder_cancel", "_handle_chart_generate",
    "_handle_dashboard_create", "_handle_visual_analysis",
    "_handle_workload_check", "_handle_general_chat",
    "_handle_file_qa", "_handle_task_add", "_handle_task_inquiry",
}

# 需要 attached_file 参数的方法
_REQUIRES_ATTACHED_FILE = {
    "_handle_file_parse", "_handle_translate", "_handle_polish",
    "_handle_generate_ppt", "_handle_file_extract_todos",
    "_handle_file_qa", "_handle_audio_transcribe",
    "_handle_chart_generate", "_handle_dashboard_create",
    "_handle_visual_analysis",
}

# 需要 params 参数的方法
_REQUIRES_PARAMS = {
    "_handle_translate", "_handle_polish", "_handle_generate_ppt",
    "_handle_file_extract_todos", "_handle_research",
    "_handle_reminder_set", "_handle_reminder_cancel", "_handle_task_add", "_handle_memory_search",
    "_handle_chart_generate", "_handle_dashboard_create",
    "_handle_visual_analysis", "_handle_workload_check",
}

# 需要 session_id 参数的方法
_REQUIRES_SESSION_ID = {
    "_handle_memory_search", "_handle_memory_summarize",
    "_handle_general_chat",
}

# 支持流式输出的方法（这些方法接受 stream_callback 关键字参数）
_STREAMABLE_METHODS = {
    "_handle_translate", "_handle_polish", "_handle_generate_ppt",
    "_handle_file_qa", "_handle_research", "_handle_memory_search",
    "_handle_memory_summarize", "_handle_briefing",
    "_handle_reminder_set", "_handle_task_inquiry",
}


def build_handlers(
    orchestrator: Any,
    user_message: str,
    attached_file: Optional[str],
    params: dict,
    session_id: Optional[str] = None,
    stream_callback: Optional[Callable] = None,
) -> Dict[str, Callable[[], dict]]:
    """
    根据路由表构建处理器 lambda 字典。

    每个 lambda 捕获其所需的参数，在调用时执行对应的 Orchestrator 方法。

    Args:
        orchestrator: Orchestrator 实例
        user_message: 用户消息
        attached_file: 上传文件路径
        params: LLM 提取的参数
        session_id: 会话 ID
        stream_callback: 可选的流式回调

    Returns:
        {intent_name: lambda -> result_dict}
    """
    handlers: Dict[str, Callable[[], dict]] = {}

    for intent, method_name in ROUTE_TABLE.items():
        handler_method = getattr(orchestrator, method_name, None)
        if handler_method is None:
            continue

        # 根据方法签名需要的参数构建 lambda
        needs_msg = method_name in _REQUIRES_USER_MESSAGE
        needs_file = method_name in _REQUIRES_ATTACHED_FILE
        needs_params = method_name in _REQUIRES_PARAMS
        needs_sid = method_name in _REQUIRES_SESSION_ID
        supports_streaming = method_name in _STREAMABLE_METHODS and stream_callback is not None

        if method_name == "_handle_general_chat":
            # general_chat 有特殊签名（需要 context, guidance_text, stream_callback）
            # 在 orchestrator 的 process_user_request 中特殊处理
            continue

        args = []
        if needs_msg:
            args.append(user_message)
        if needs_file:
            args.append(attached_file)
        if needs_params:
            args.append(params)
        if needs_sid:
            args.append(session_id)

        # 流式回调作为最后一个关键字参数
        if supports_streaming:
            handlers[intent] = _make_handler_with_stream(handler_method, stream_callback, *args)
        else:
            handlers[intent] = _make_handler(handler_method, *args)

    return handlers


def _make_handler(method: Callable, *args: Any) -> Callable[[], dict]:
    """创建零参数 lambda，捕获参数值。"""
    return lambda: method(*args)


def _make_handler_with_stream(method: Callable, stream_cb: Callable, *args: Any) -> Callable[[], dict]:
    """创建带流式回调的 lambda。"""
    return lambda: method(*args, stream_callback=stream_cb)


def get_handler_method_name(intent: str) -> Optional[str]:
    """获取指定意图的处理器方法名。"""
    return ROUTE_TABLE.get(intent)


def get_all_intents() -> list:
    """获取所有已注册的意图类型。"""
    return list(ROUTE_TABLE.keys())
