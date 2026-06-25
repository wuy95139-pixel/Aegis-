"""
提醒跟进工作流
=============
预定义的提醒和跟进检查流程。

可扩展点:
  - 智能降噪：合并相似提醒
  - 优先级排序：根据重要性和紧急程度排列
  - 自动化跟进：自动发送提醒邮件/消息
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def run_reminder_followup_workflow(
    orchestrator,
    include_email_check: bool = False,
) -> dict:
    """
    运行提醒跟进检查工作流

    流程:
      1. ReminderAgent: 检查到期提醒
      2. ReminderAgent: 检查需要跟进的事项
      3. MemoryAgent: 检索被遗忘的事项

    Args:
        orchestrator: Orchestrator 实例
        include_email_check: 是否检查邮件跟进

    Returns:
        工作流执行结果
    """
    logger.info("Starting reminder followup workflow")

    # Step 1: 检查到期提醒
    due_check = orchestrator.agents["reminder_agent"].execute({
        "operation": "check",
    })

    # Step 2: 检查跟进事项
    followup = orchestrator.agents["reminder_agent"].execute({
        "operation": "followup",
    })

    # Step 3: 从长期记忆中查找可能的遗漏
    memory_result = orchestrator.agents["memory_agent"].execute({
        "operation": "retrieve",
        "query": "待办 任务 未完成 需要处理",
        "top_k": 5,
        "tags": ["todos", "task_dispatch"],
    })

    # 构建汇总
    summary_parts = ["## 📊 提醒与跟进报告\n"]
    summary_parts.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    summary_parts.append(f"### ⏰ 到期提醒 ({due_check.get('count', 0)})")
    for r in due_check.get("due_reminders", []):
        time_str = r.trigger_time.strftime("%m-%d %H:%M") if r.trigger_time else "N/A"
        summary_parts.append(f"- {r.title} (触发: {time_str})")

    summary_parts.append(f"\n### ⚠️ 需要跟进 ({followup.get('count', 0)})")
    for s in followup.get("followup_suggestions", []):
        summary_parts.append(f"- {s.get('title', s.get('id', ''))}")

    if memory_result.get("relevant_memories"):
        summary_parts.append(f"\n### 🔍 历史待办提醒")
        for mem in memory_result["relevant_memories"][:3]:
            summary_parts.append(f"- {mem.content[:150]}...")

    return {
        "status": "success",
        "summary": "\n".join(summary_parts),
        "due_reminders_count": due_check.get("count", 0),
        "followup_count": followup.get("count", 0),
        "due_reminders": due_check.get("due_reminders", []),
        "followup_suggestions": followup.get("followup_suggestions", []),
    }


def run_morning_briefing(
    orchestrator,
) -> dict:
    """
    早晨简报工作流 — 每日自动运行

    生成包含以下内容的简报:
      - 今日待办提醒
      - 需要跟进的事项
      - 今日日程概览
    """
    logger.info("Generating morning briefing")

    # 获取提醒和跟进
    reminder_result = orchestrator.agents["reminder_agent"].execute({
        "operation": "list",
        "active_only": True,
    })

    followup = orchestrator.agents["reminder_agent"].execute({
        "operation": "followup",
    })

    # 用 LLM 生成简报
    reminders_text = "\n".join(
        f"- {r.title}" + (f" ({r.trigger_time.strftime('%H:%M')})" if r.trigger_time else "")
        for r in reminder_result.get("reminders", [])[:10]
    )

    followup_text = "\n".join(
        f"- {s.get('title', s.get('id', ''))}"
        for s in followup.get("followup_suggestions", [])[:5]
    )

    prompt = f"""你是一位私人助理。请根据以下信息生成一份简洁的早晨简报。

今日提醒:
{reminders_text or "(无)"}

需要跟进:
{followup_text or "(无)"}

用友好的语气，2-3 句话总结今天需要注意的事项。"""

    response = orchestrator.llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )

    return {
        "status": "success",
        "briefing": response["content"],
        "active_reminders": len(reminder_result.get("reminders", [])),
        "followup_count": followup.get("count", 0),
    }
