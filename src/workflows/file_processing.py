"""
文件处理工作流
=============
预定义的多 Agent 协作流程：上传文件 → 解析 → 翻译/润色 → 提取待办 → 分派 → 设置提醒

可扩展点:
  - 添加审批步骤：在分派前由用户确认
  - 添加通知步骤：分派后通知相关人员
  - 支持批量文件处理
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_file_processing_workflow(
    orchestrator,
    filepath: str,
    action: str = "translate",
    target_lang: str = "zh-CN",
    style: str = "professional",
    auto_assign: bool = True,
) -> dict:
    """
    运行文件处理完整工作流

    流程:
      1. FileProcessorAgent: 解析文件 + 翻译/润色/生成PPT
      2. TaskDispatcherAgent: 提取待办事项 + 自动分派
      3. ReminderAgent: 为待办事项创建提醒
      4. MemoryAgent: 存储关键信息到长期记忆

    Args:
        orchestrator: Orchestrator 实例
        filepath: 文件路径
        action: 处理操作 (translate / polish / generate_ppt)
        target_lang: 目标语言
        style: 润色风格
        auto_assign: 是否自动分派任务

    Returns:
        工作流执行结果
    """
    logger.info(f"Starting file processing workflow: {filepath}, action={action}")

    # Step 1: 解析文件
    file_result = orchestrator.agents["file_processor"].execute({
        "filepath": filepath,
        "action": action,
        "target_lang": target_lang,
        "style": style,
    })

    if file_result["status"] != "success":
        return {"status": "error", "step": "file_processing", "error": file_result}

    # Step 2: 分派待办事项
    todos = file_result.get("extracted_todos", [])
    dispatch_result = None

    if todos:
        dispatch_result = orchestrator.agents["task_dispatcher"].execute({
            "todos": todos,
            "source": Path(filepath).name,
            "auto_assign": auto_assign,
        })

    # Step 3: 为已分派的任务创建提醒
    reminder_ids = []
    if dispatch_result and dispatch_result.get("assigned_todos"):
        reminder_ids = orchestrator.agents["reminder_agent"].create_reminders_from_todos(
            dispatch_result["assigned_todos"]
        )

    # Step 4: 存储处理结果到长期记忆
    orchestrator.agents["memory_agent"].execute({
        "operation": "store",
        "content": f"文件处理: {Path(filepath).name}, 操作: {action}, 提取了 {len(todos)} 个待办事项",
        "source": f"workflow:file_processing:{Path(filepath).name}",
        "tags": ["workflow", "file_processing", action],
    })

    # 构建最终结果
    result_text = file_result.get("result_text", "")
    result = {
        "status": "success",
        "filename": Path(filepath).name,
        "action": action,
        "file_result": {
            "file_type": file_result.get("parsed_file", {}).file_type.value if hasattr(file_result.get("parsed_file", {}), "file_type") else "unknown",
            "content_preview": result_text,  # 保留完整文本，由 _format_file_result 按场景处理
        },
        "todos_count": len(todos),
        "assigned_count": len(dispatch_result.get("assigned_todos", [])) if dispatch_result else 0,
        "unassigned_count": len(dispatch_result.get("unassigned", [])) if dispatch_result else 0,
        "reminders_created": len(reminder_ids),
    }

    logger.info(f"Workflow completed: {result}")
    return result
