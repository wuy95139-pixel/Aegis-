"""
Toast 按钮点击处理器
===================
当用户点击 Windows Toast 通知上的按钮（"确认"/"稍后"）时，
Windows 通过 aegis: 协议启动此脚本，将操作写入信号文件。

CalendarTool 的后台调度器会定期检查信号文件并处理。
"""
import sys
import json
import os
from datetime import datetime
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        return

    arg = sys.argv[1]  # "aegis:confirm/reminder_id" 或 "aegis:snooze/reminder_id"

    try:
        # 解析协议 URL
        payload = arg.split(":", 1)[1] if ":" in arg else arg
        action, reminder_id = payload.split("/", 1)
    except (ValueError, IndexError):
        return

    if action not in ("confirm", "snooze"):
        return

    # 写入信号文件
    signal_dir = Path("./data/signals")
    signal_dir.mkdir(parents=True, exist_ok=True)

    signal = {
        "action": action,
        "reminder_id": reminder_id,
        "timestamp": datetime.now().isoformat(),
    }

    signal_file = signal_dir / f"{reminder_id}.json"
    with open(signal_file, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
