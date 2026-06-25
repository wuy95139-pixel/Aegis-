"""
对话日志器
==========
从 Orchestrator 提取出的对话记录和持久化逻辑。

将每次对话写入短期记忆和每日 Markdown 文件。
支持会话隔离和历史查看。
"""

import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from src.models.schemas import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationLogger:
    """对话记录器：短期记忆 + 文件持久化"""

    def __init__(self, memory, session_manager=None):
        self.memory = memory
        self.session_manager = session_manager

    def record(
        self,
        user_msg: str,
        assistant_msg: str,
        session_id: Optional[str] = None,
    ):
        """记录对话到短期记忆，并持久化到文件"""
        session_mem = (
            self.session_manager.get_existing(session_id)
            if session_id and self.session_manager
            else None
        )
        if session_mem:
            stm = session_mem.short_term
            if stm:
                stm.add_turn(ConversationTurn(role="user", content=user_msg))
                if assistant_msg:
                    stm.add_turn(ConversationTurn(role="assistant", content=assistant_msg))
        elif hasattr(self.memory, 'short_term') and self.memory.short_term:
            self.memory.short_term.add_turn(ConversationTurn(role="user", content=user_msg))
            if assistant_msg:
                self.memory.short_term.add_turn(ConversationTurn(role="assistant", content=assistant_msg))

        self._persist(user_msg, assistant_msg, session_id)

    def _persist(
        self,
        user_msg: str,
        assistant_msg: str,
        session_id: Optional[str] = None,
    ):
        """将对话追加到每日 Markdown 文件"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            # Resolve conversations directory relative to project or use env override
            import os as _os
            data_dir = _os.environ.get("AEGIS_DATA_DIR", "./data")
            conv_dir = Path(data_dir) / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            filepath = conv_dir / f"{date_str}.md"

            if not filepath.exists():
                filepath.write_text(f"# Aegis 会话记录 — {date_str}\n\n", encoding="utf-8")

            sid = f" ({session_id[:8]})" if session_id else ""
            entry = f"## {time_str}{sid}\n\n"
            entry += f"**用户:** {user_msg}\n\n"
            if assistant_msg:
                truncated = assistant_msg[:2000] + ("..." if len(assistant_msg) > 2000 else "")
                entry += f"**Aegis:** {truncated}\n\n"
            entry += "---\n\n"

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            logger.debug("Conversation persistence failed", exc_info=True)
