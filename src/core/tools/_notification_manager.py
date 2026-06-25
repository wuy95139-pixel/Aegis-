"""
Notification channels for CalendarTool
======================================
Windows Toast, sound beep, and console notification implementations.

Extracted from calendar_tools.py to keep the CalendarTool class focused
on reminder CRUD and scheduling.
"""

import logging
import platform
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

from src.core.tools._win_toast import (
    show_toast_interactive,
    show_simple_toast,
    register_windows_app_id,
    _IS_WINDOWS,
)
from src.models.schemas import Reminder, ReminderType

logger = logging.getLogger(__name__)


class NotificationManager:
    """System notification channels: Toast (Windows), sound, console."""

    def __init__(self):
        self._notify_handlers: dict[str, callable] = {
            "sound": self._notify_sound,
            "toast": self._notify_toast,
            "console": self._notify_console,
        }
        self._register_windows_app_id()

    def register_handler(self, method: str, handler: callable):
        """Register a custom notification channel (e.g., email, webhook, SMS)."""
        self._notify_handlers[method] = handler
        logger.info("Registered notify handler: %s", method)

    def trigger(self, reminder: Reminder):
        """Fire all configured notification channels for a reminder."""
        for method in reminder.notify_method:
            handler = self._notify_handlers.get(method)
            if handler:
                try:
                    handler(reminder)
                except Exception as e:
                    logger.error("Notify handler '%s' failed: %s", method, e)

    def send_simple_toast(self, title: str, body: str = ""):
        """Send a simple toast without interaction buttons (e.g., 'next step' hints)."""
        if not _IS_WINDOWS:
            return
        show_simple_toast(title, body)

    # ==================== channel implementations ====================

    def _notify_sound(self, reminder: Reminder):
        """System beep (winsound on Windows, ASCII bell cross-platform)."""
        try:
            if platform.system() == "Windows":
                import winsound

                winsound.MessageBeep(0x00000030)
                time.sleep(0.1)
                winsound.MessageBeep(0x00000030)
            else:
                print("\a", end="", flush=True)
        except Exception as e:
            logger.debug("Windows audible beep failed, falling back to BEL: %s", e)
            print("\a", end="", flush=True)

    def _notify_toast(self, reminder: Reminder):
        """Windows 10/11 toast with confirm/snooze action buttons."""
        if not _IS_WINDOWS:
            return

        title = reminder.title
        body = reminder.description or ""
        if reminder.trigger_time:
            time_str = reminder.trigger_time.strftime("%H:%M")
            body = f"⏰ {time_str}\n{body}" if body else f"⏰ Time: {time_str}"

        show_toast_interactive(
            reminder_id=reminder.id,
            title=title,
            body=body,
        )

    def _notify_console(self, reminder: Reminder):
        """Console notification with ASCII bell and colored separator."""
        lines = [
            f"\n\a{'='*52}",
            f"  ⏰  Reminder: {reminder.title}",
        ]
        if reminder.description:
            lines.append(f"       Detail: {reminder.description}")
        if reminder.trigger_time:
            time_str = reminder.trigger_time.strftime("%Y-%m-%d %H:%M")
            lines.append(f"       Time: {time_str}")
        if reminder.type == ReminderType.RECURRING:
            lines.append(f"       Recurring: {reminder.cron_expression}")
        lines.append(f"{'='*52}\n")
        print("\n".join(lines), flush=True)

    # ==================== app identity ====================

    @staticmethod
    def _register_windows_app_id():
        """Register Aegis AppUserModelId so toasts appear in Action Center."""
        register_windows_app_id()
