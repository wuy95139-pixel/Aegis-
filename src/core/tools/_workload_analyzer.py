"""
Workload analysis for CalendarTool
===================================
Estimates daily task load and generates rescheduling suggestions.

Extracted from calendar_tools.py.
"""

from datetime import datetime, timedelta
from typing import Optional


# Priority → estimated hours default mapping
PRIORITY_HOURS = {
    "urgent": 3.0,
    "high": 2.0,
    "medium": 1.0,
    "low": 0.5,
}
OVERLOAD_THRESHOLD = 8.0   # > 8h → overload
WARNING_THRESHOLD = 6.0    # > 6h → heavy
BUSY_THRESHOLD = 4.0       # > 4h → busy


def analyze_workload(todos: list, target_date: Optional[datetime] = None) -> dict:
    """Analyze task load for a given date, with rescheduling suggestions.

    Args:
        todos: List of TodoItem-like objects (with priority/deadline attrs)
               or dicts with 'title', 'priority', 'deadline' keys.
        target_date: Date to analyze (default: today).

    Returns:
        dict with date, total_estimated_hours, task_count, status,
        tasks list, and suggestions.
    """
    target = target_date or datetime.now()
    target_day = target.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day = target_day + timedelta(days=1)

    day_tasks = []
    for t in todos:
        dl = None
        priority_str = "medium"
        title = ""
        if hasattr(t, "deadline"):
            dl = t.deadline
            priority_str = getattr(t, "priority", "medium")
            if hasattr(priority_str, "value"):
                priority_str = priority_str.value
            title = getattr(t, "title", "")
        elif isinstance(t, dict):
            dl_raw = t.get("deadline")
            if isinstance(dl_raw, str):
                try:
                    dl = datetime.fromisoformat(dl_raw)
                except (ValueError, TypeError):
                    pass
            elif isinstance(dl_raw, datetime):
                dl = dl_raw
            priority_str = t.get("priority", "medium")
            title = t.get("title", "")
        else:
            continue

        if dl and target_day <= dl < next_day:
            priority_str = str(priority_str).lower()
            hours = PRIORITY_HOURS.get(priority_str, 1.0)
            day_tasks.append({
                "title": title,
                "priority": priority_str,
                "deadline": dl.isoformat() if dl else None,
                "estimated_hours": hours,
            })

    total_hours = sum(t["estimated_hours"] for t in day_tasks)
    count = len(day_tasks)

    if total_hours >= OVERLOAD_THRESHOLD:
        status = "overload"
    elif total_hours >= WARNING_THRESHOLD:
        status = "warning"
    elif total_hours >= BUSY_THRESHOLD:
        status = "busy"
    else:
        status = "clear"

    suggestions = []
    if status in ("overload", "warning"):
        low_tasks = [t for t in day_tasks if t["priority"] in ("low", "medium")]
        postpone = low_tasks[: max(1, len(low_tasks) // 2)]
        for t in postpone:
            suggestions.append(
                f"Suggest postponing '{t['title']}' "
                f"(priority={t['priority']}, est.{t['estimated_hours']}h)"
            )
        if status == "overload":
            suggestions.insert(
                0,
                f"Warning: daily load {total_hours:.1f}h / {count} items — overload. "
                "Prioritize urgent/high tasks, reschedule the rest.",
            )
        else:
            suggestions.insert(
                0,
                f"Note: daily load {total_hours:.1f}h / {count} items — heavy. "
                "Consider postponing some medium/low tasks.",
            )

    if status == "overload" and not suggestions:
        suggestions.append(
            f"{count} items estimated at {total_hours:.1f}h. "
            "Consider reducing tasks or extending deadlines."
        )

    return {
        "date": target_day.strftime("%Y-%m-%d"),
        "total_estimated_hours": round(total_hours, 1),
        "task_count": count,
        "status": status,
        "tasks": day_tasks,
        "suggestions": suggestions,
    }
