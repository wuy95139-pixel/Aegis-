"""
Windows scheduled task manager for CalendarTool
===============================================
Creates/deletes Windows Task Scheduler entries so reminders fire even
when Aegis is not running.

Extracted from calendar_tools.py.
"""

import logging
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.models.schemas import Reminder

logger = logging.getLogger(__name__)

class ScheduledTaskManager:
    """Manages Windows Task Scheduler entries for offline reminder notifications."""

    def __init__(self, register_app_id: Optional[Callable[[], None]] = None):
        self._register_app_id = register_app_id

    @staticmethod
    def sanitize_for_powershell_xml(value: str) -> str:
        """Sanitize user input for safe embedding in XML CDATA sections.

        Only the CDATA terminator is dangerous — split it so it is never
        interpreted as a real end-of-CDATA marker.
        """
        if not value:
            return ""
        return value.replace("]]>", "]]]]><![CDATA[>")

    @staticmethod
    def sanitize_task_name(name: str) -> str:
        """Sanitize a Windows scheduled task name for schtasks.

        Only allows: alphanumeric, underscore, hyphen, space, dot.
        """
        if not name:
            return "Aegis_Reminder_unknown"
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-. ")
        sanitized = "".join(c if c in allowed else "_" for c in name)
        return sanitized[:128]

    # ----------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------

    def create(self, reminder: Reminder):
        """Create a Windows scheduled task that shows a Toast at trigger_time.

        The task launches pythonw.exe (no console, no flash) which runs a
        self-contained Python script that shows an interactive Toast with
        Confirm / Snooze buttons.
        """
        if platform.system() != "Windows":
            return
        if not reminder.trigger_time:
            return

        if self._register_app_id:
            self._register_app_id()

        task_name = self.sanitize_task_name(f"Aegis_Reminder_{reminder.id}")
        trigger_time = reminder.trigger_time

        now = datetime.now()
        if trigger_time <= now:
            logger.debug("Skipping past-due scheduled task for: %s", reminder.title)
            return

        script_path = self._build_reminder_script(reminder)
        pythonw_path = str(Path(sys.executable).parent / "pythonw.exe")

        schedule_time = trigger_time.strftime("%H:%M")
        schedule_date = trigger_time.strftime("%Y/%m/%d")

        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", task_name, "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )

            result = subprocess.run(
                [
                    "schtasks", "/Create",
                    "/TN", task_name,
                    "/SC", "ONCE",
                    "/SD", schedule_date,
                    "/ST", schedule_time,
                    # pythonw.exe is a GUI-subsystem binary — zero console flash.
                    "/TR", f'"{pythonw_path}" "{script_path}"',
                    "/F",
                    "/RL", "LIMITED",
                ],
                capture_output=True, timeout=15,
                encoding="gbk", errors="replace",
            )
            if result.returncode == 0:
                logger.info(
                    "Scheduled task created: %s at %s %s",
                    task_name, schedule_date, schedule_time,
                )
            else:
                logger.warning(
                    "Failed to create scheduled task %s: %s",
                    task_name,
                    result.stderr.strip() if result.stderr else result.stdout.strip(),
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning("Scheduled task creation failed for %s: %s", task_name, e)

    def delete(self, reminder_id: str):
        """Delete a Windows scheduled task and its associated .py script."""
        if platform.system() != "Windows":
            return
        task_name = self.sanitize_task_name(f"Aegis_Reminder_{reminder_id}")
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", task_name, "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
            logger.debug("Scheduled task deleted: %s", task_name)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("Scheduled task deletion failed for %s: %s", task_name, e)

        script_path = Path("./data/tasks") / f"reminder_{reminder_id}.py"
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    def cleanup_stale(self, active_reminder_ids: set):
        """Remove orphaned scheduled tasks whose reminders no longer exist."""
        if platform.system() != "Windows":
            return
        try:
            result = subprocess.run(
                ["schtasks", "/Query", "/FO", "CSV", "/NH"],
                capture_output=True, timeout=10,
                encoding="gbk", errors="replace",
            )
            for line in (result.stdout or "").splitlines():
                if "Aegis_Reminder_" in line:
                    for rid in list(active_reminder_ids):
                        task_name = f"Aegis_Reminder_{rid}"
                        if task_name in line and rid not in active_reminder_ids:
                            self.delete(rid)
                            logger.info("Cleaned up stale scheduled task for: %s", rid)
        except Exception as e:
            logger.debug("Stale task cleanup skipped: %s", e)

    # ----------------------------------------------------------------
    # script generation
    # ----------------------------------------------------------------

    def _build_reminder_script(self, reminder: Reminder) -> Path:
        """Generate the self-contained .py reminder script and return its path.

        All values are embedded at generation time so the generated script
        has zero f-strings — no double-brace escaping maze, no NameErrors.
        """
        script_dir = Path("./data/tasks")
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"reminder_{reminder.id}.py"

        title = self.sanitize_for_powershell_xml(reminder.title or "")
        body = self.sanitize_for_powershell_xml(reminder.description or "")
        if reminder.trigger_time:
            time_str = reminder.trigger_time.strftime("%H:%M")
            body = f"⏰ {time_str}\\n{body}" if body else f"⏰ Time: {time_str}"

        signal_dir = str(Path("./data/signals").absolute())

        # Build the toast XML and PS script HERE so the generated .py script
        # is just a dumb executor — no f-strings, no variable interpolation.
        toast_xml = (
            '<toast duration="long">\n'
            "    <visual>\n"
            '        <binding template="ToastImageAndText02">\n'
            f'            <text id="1"><![CDATA[{title}]]></text>\n'
            f'            <text id="2"><![CDATA[{body}]]></text>\n'
            "        </binding>\n"
            "    </visual>\n"
            "    <actions>\n"
            f'        <action activationType="protocol" content="确认" arguments="aegis:confirm/{reminder.id}" />\n'
            f'        <action activationType="protocol" content="稍后(5分钟)" arguments="aegis:snooze/{reminder.id}" />\n'
            "    </actions>\n"
            '    <audio silent="true" />\n'
            "</toast>"
        )

        # PowerShell snippet — single f-string, no nested substitution.
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null\n"
            "[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null\n"
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null\n"
            "\n"
            '$Template = @"\n'
            f"{toast_xml}\n"
            '"@\n'
            "\n"
            "$Xml = New-Object Windows.Data.Xml.Dom.XmlDocument\n"
            "$Xml.LoadXml($Template)\n"
            "$Toast = [Windows.UI.Notifications.ToastNotification]::new($Xml)\n"
            f'$Toast.Tag = "{reminder.id}"\n'
            '$Toast.Group = "Aegis"\n'
            '$Notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Aegis")\n'
            "$Notifier.Show($Toast)\n"
        )

        content = f'''"""Aegis reminder toast — auto-generated, do not edit."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_REMINDER_ID = "{reminder.id}"
_SIGNAL_DIR = r"{signal_dir}"

# Pre-built at generation time — the script just executes these.
_TOAST_PS = r"""{ps_script}"""

if __name__ == "__main__":
    try:
        subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", _TOAST_PS],
            creationflags=_CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        Path(_SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
        signal_path = Path(_SIGNAL_DIR) / f"{{_REMINDER_ID}}.json"
        signal_path.write_text(
            json.dumps({{
                "action": "show",
                "reminder_id": _REMINDER_ID,
                "timestamp": datetime.now().isoformat(),
            }}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
'''
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".py", prefix=f"reminder_{reminder.id}_", dir=str(script_dir)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(script_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return script_path.absolute()
