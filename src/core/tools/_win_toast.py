"""
Windows Toast notification helpers
==================================
Shows Windows 10/11 Toast notifications with zero console flash.

Uses PowerShell WinRT interop internally (the only reliable way to access
ToastNotificationManager from Python without extra packages), but wraps it
with CREATE_NO_WINDOW so no terminal appears — not even briefly.

Usage:
    from src.core.tools._win_toast import show_toast_interactive

    show_toast_interactive(
        reminder_id="abc123",
        title="Meeting",
        body="Starts in 10 minutes",
    )
"""

import logging
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

if _IS_WINDOWS:
    _PROTOCOL_REGISTERED = False


def _xml_escape(s: str) -> str:
    """Escape a string for XML text content (not attribute)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cdata_safe(s: str) -> str:
    """Make a string safe for CDATA by splitting the terminator."""
    return s.replace("]]>", "]]]]><![CDATA[>")


def show_toast_interactive(reminder_id: str, title: str, body: str) -> None:
    """Show a Windows Toast with Confirm + Snooze action buttons.

    Button clicks are handled by the aegis:// URL protocol (if registered)
    and write signal files to data/signals/ for Aegis to process.
    """
    if not _IS_WINDOWS:
        return

    _ensure_protocol_registered()

    escaped_title = _cdata_safe(_xml_escape(title))
    escaped_body = _cdata_safe(_xml_escape(body))

    toast_xml = f"""<toast duration="long">
    <visual>
        <binding template="ToastImageAndText02">
            <text id="1"><![CDATA[{escaped_title}]]></text>
            <text id="2"><![CDATA[{escaped_body}]]></text>
        </binding>
    </visual>
    <actions>
        <action activationType="protocol" content="确认" arguments="aegis:confirm/{reminder_id}" />
        <action activationType="protocol" content="稍后(5分钟)" arguments="aegis:snooze/{reminder_id}" />
    </actions>
    <audio silent="true" />
</toast>"""

    _run_ps_toast(toast_xml, tag=reminder_id)


def show_simple_toast(title: str, body: str) -> None:
    """Show a simple Windows Toast with no action buttons."""
    if not _IS_WINDOWS:
        return

    escaped_title = _cdata_safe(_xml_escape(title))
    escaped_body = _cdata_safe(_xml_escape(body))

    toast_xml = f"""<toast duration="short">
    <visual>
        <binding template="ToastImageAndText02">
            <text id="1"><![CDATA[{escaped_title}]]></text>
            <text id="2"><![CDATA[{escaped_body}]]></text>
        </binding>
    </visual>
    <audio silent="true" />
</toast>"""

    _run_ps_toast(toast_xml)


def _run_ps_toast(toast_xml: str, tag: str = "") -> None:
    """Execute the toast XML via a hidden PowerShell subprocess.

    Uses CREATE_NO_WINDOW so no console ever appears.  Launched in a
    daemon thread so we don't block the caller.
    """
    tag_line = f'$Toast.Tag = "{tag}"' if tag else ""
    group_line = '$Toast.Group = "Aegis"'

    ps_script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$Template = @"
{toast_xml}
"@

$Xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$Xml.LoadXml($Template)
$Toast = [Windows.UI.Notifications.ToastNotification]::new($Xml)
{tag_line}
{group_line}
$Notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Aegis")
$Notifier.Show($Toast)
"""
    try:
        proc = subprocess.Popen(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        threading.Thread(target=proc.wait, daemon=True).start()
    except Exception as e:
        logger.debug("PowerShell toast invocation failed (non-critical): %s", e)


def _ensure_protocol_registered() -> None:
    """Register aegis:// URL protocol in Windows registry for interactive toasts."""
    if not _IS_WINDOWS:
        return
    global _PROTOCOL_REGISTERED
    if _PROTOCOL_REGISTERED:
        return

    try:
        import winreg

        handler_path = str(Path(__file__).resolve().parent / "toast_handler.py")
        pythonw = str(Path(sys.executable).parent / "pythonw.exe")

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Classes\aegis") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:Aegis Protocol")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")

        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, r"SOFTWARE\Classes\aegis\shell\open\command"
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f'"{pythonw}" "{handler_path}" %1')

        _PROTOCOL_REGISTERED = True
    except Exception as e:
        logger.debug("Windows URL protocol registration failed (non-critical): %s", e)


def register_windows_app_id() -> None:
    """Register Aegis AppUserModelId so toasts show under 'Aegis' in Action Center."""
    if not _IS_WINDOWS:
        return
    try:
        import winreg

        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, r"Software\Classes\AppUserModelId\Aegis"
        ) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Aegis Personal Assistant")
    except Exception as e:
        logger.debug("Windows AppUserModelId registration failed (non-critical): %s", e)
