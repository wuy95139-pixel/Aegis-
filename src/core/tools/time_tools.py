"""
时间工具
========
为 Aegis 个人助理提供时间上下文注入和自然语言时间解析。

设计决策：
  - 不依赖外部 MCP 进程，纯 Python 实现
  - get_time_context() 用于注入 LLM system prompt
  - parse_chinese_time_expression() 用于解析用户输入中的时间表达
  - 所有时间基于系统本地时钟（datetime.now()）
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

# 中文星期映射
CHINESE_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 中文数字映射
_CHINESE_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# 时间单位模式
_TIME_UNIT_PATTERNS = [
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*天[之以]?后"), "days"),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*个?\s*小?时[之以]?后"), "hours"),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*分[钟][之以]?后"), "minutes"),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*秒[之以]?后"), "seconds"),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*周[之以]?后"), "weeks"),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*个?月[之以]?后"), "months"),
]

# 相对日期模式
_RELATIVE_DAY_PATTERNS = [
    (re.compile(r"今天|今日"), 0),
    (re.compile(r"明天|明日"), 1),
    (re.compile(r"后天|后日"), 2),
    (re.compile(r"大后天"), 3),
    (re.compile(r"昨天|昨日"), -1),
    (re.compile(r"前天|前日"), -2),
]

# 星期模式
_WEEKDAY_PATTERNS = {
    "周一": 0, "星期一": 0, "礼拜一": 0,
    "周二": 1, "星期二": 1, "礼拜二": 1,
    "周三": 2, "星期三": 2, "礼拜三": 2,
    "周四": 3, "星期四": 3, "礼拜四": 3,
    "周五": 4, "星期五": 4, "礼拜五": 4,
    "周六": 5, "星期六": 5, "礼拜六": 5,
    "周日": 6, "星期天": 6, "星期日": 6, "礼拜天": 6, "礼拜日": 6,
}

# 时段模式 (24h)
_TIME_OF_DAY = [
    (re.compile(r"早上|早晨|早"), 8),
    (re.compile(r"上午"), 10),
    (re.compile(r"中午|正午"), 12),
    (re.compile(r"下午"), 14),
    (re.compile(r"傍晚|黄昏"), 18),
    (re.compile(r"晚上|今晚"), 20),
    (re.compile(r"夜里|深夜|凌晨"), 2),
]


from src.core.tools._tool_registry import tool  # noqa: E402


@tool(description="获取当前日期、星期几和精确时间")
def get_time_context() -> str:
    """生成当前时间上下文字符串，用于注入 LLM system prompt"""
    now = datetime.now()
    weekday = CHINESE_WEEKDAYS[now.weekday()]
    return f"今天是 {now.strftime('%Y-%m-%d')} ({weekday})，当前时间 {now.strftime('%H:%M')} (UTC+8)"


def _parse_chinese_number(text: str) -> int:
    """将中文数字转为整数（支持百、千）"""
    text = text.strip()
    if text in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[text]
    # "一百二十" → 120, "一千三百" → 1300
    result = 0
    for unit_char, unit_val in [("千", 1000), ("百", 100)]:
        if unit_char in text:
            parts = text.split(unit_char)
            prefix = parts[0] if parts[0] else "一"
            result += _parse_chinese_number(prefix) * unit_val
            text = parts[1] if len(parts) > 1 else ""
            break
    # "十二" → 12, "十" → 10
    if "十" in text:
        parts = text.split("十")
        tens = _CHINESE_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CHINESE_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        result += tens * 10 + ones
    elif text:
        result += _CHINESE_DIGITS.get(text, 0)
    if result > 0:
        return result
    try:
        return int(text)
    except ValueError:
        return 0


def _parse_relative_day(text: str, ref: datetime) -> Optional[datetime]:
    """解析相对日期表达：今天/明天/后天/昨天/下周一"""
    # 检查"下周一/下周三/下星期X"
    next_weekday_match = re.match(r"下\s*(周|星期|礼拜)\s*([一二三四五六日天])", text)
    if next_weekday_match:
        target_wd = _WEEKDAY_PATTERNS.get(f"周{next_weekday_match.group(2)}", 0)
        days_ahead = (target_wd - ref.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # "下周一" → 下周，不是今天
        days_ahead += 7  # "下周" = 本周 + 7
        return ref + timedelta(days=days_ahead)

    # 检查"这周一/本周一"
    this_weekday_match = re.match(r"这\s*(周|星期|礼拜)\s*([一二三四五六日天])", text)
    if this_weekday_match:
        target_wd = _WEEKDAY_PATTERNS.get(f"周{this_weekday_match.group(2)}", 0)
        days_ahead = (target_wd - ref.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 0  # "这周一" → 今天就是周一
        return ref + timedelta(days=days_ahead)

    # 检查纯星期几（默认最近的未来日期）
    for wd_name, wd_num in _WEEKDAY_PATTERNS.items():
        if text.startswith(wd_name):
            days_ahead = (wd_num - ref.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 0  # 今天就是，返回今天
            return ref + timedelta(days=days_ahead)

    # 检查今天/明天/后天/昨天
    for pattern, offset in _RELATIVE_DAY_PATTERNS:
        if pattern.search(text):
            return ref + timedelta(days=offset)

    return None


def _parse_time_of_day(text: str) -> Optional[int]:
    """解析时段，返回默认小时"""
    for pattern, hour in _TIME_OF_DAY:
        if pattern.search(text):
            return hour
    return None


def parse_chinese_time_expression(text: str, ref: Optional[datetime] = None) -> Optional[datetime]:
    """
    解析中文时间表达式为 datetime。

    支持格式：
      - 相对日期: "明天下午3点", "后天早上9点", "下周一上午10点", "周五晚上8点"
      - 时间段: "3天后", "2小时后", "30分钟后"
      - 绝对时间: "2025-06-01 14:00"

    Args:
        text: 中文时间表达文本
        ref: 参考时间，默认为 datetime.now()

    Returns:
        解析后的 datetime，失败返回 None
    """
    if ref is None:
        ref = datetime.now()

    # 去掉多余空白
    text = text.strip()

    # 尝试解析 ISO 格式
    iso_patterns = [
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?)",
        r"(\d{4}/\d{2}/\d{2}[T ]\d{2}:\d{2}(:\d{2})?)",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in iso_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                iso_str = m.group(1).replace("/", "-").replace("T", " ")
                if ":" in iso_str:
                    return datetime.strptime(iso_str, "%Y-%m-%d %H:%M")
                else:
                    return datetime.strptime(iso_str, "%Y-%m-%d")
            except ValueError:
                pass

    # 检测时段（上午/下午/晚上）
    is_afternoon = False
    is_evening = False
    period_match = re.search(r"下午|晚上|夜里|深夜|傍晚|黄昏", text)
    if period_match:
        is_afternoon = True
    evening_match = re.search(r"晚上|夜里|深夜|傍晚|黄昏", text)
    if evening_match:
        is_evening = True

    # 解析具体时间点 (HH:MM)
    hour = 0
    minute = 0
    has_explicit_time = False
    time_match = re.search(r"(\d{1,2})[点:：](\d{1,2})?\s*(分)?", text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        has_explicit_time = True
        # 12小时制转换：下午3点 → 15:00（但下午12点 → 12:00，不转换）
        if is_afternoon and 1 <= hour <= 11:
            hour += 12
    else:
        # 没有具体时间，使用时段默认小时
        parsed_hour = _parse_time_of_day(text)
        if parsed_hour is not None:
            hour = parsed_hour

    # 解析日期部分
    result_date = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 检查"X天/小时/分钟后"
    for pattern, unit in _TIME_UNIT_PATTERNS:
        m = pattern.search(text)
        if m:
            num = _parse_chinese_number(m.group(1))
            if unit == "days":
                return ref + timedelta(days=num)
            elif unit == "hours":
                return ref + timedelta(hours=num)
            elif unit == "minutes":
                return ref + timedelta(minutes=num)
            elif unit == "seconds":
                return ref + timedelta(seconds=num)
            elif unit == "weeks":
                return ref + timedelta(weeks=num)
            # months 不做精确计算，按 30 天估算
            elif unit == "months":
                return ref + timedelta(days=num * 30)

    # 解析相对日期
    date_result = _parse_relative_day(text, result_date)
    if date_result is not None:
        return date_result

    # 如果只有时间没有日期，返回今天的这个时间
    if hour > 0 or minute > 0:
        return result_date

    return None


def format_datetime_chinese(dt: datetime) -> str:
    """将 datetime 格式化为中文可读字符串"""
    weekday = CHINESE_WEEKDAYS[dt.weekday()]
    return f"{dt.strftime('%Y-%m-%d')} ({weekday}) {dt.strftime('%H:%M')}"


def expression_to_cron(text: str, ref: Optional[datetime] = None) -> str:
    """
    将中文重复时间表达转为 cron 表达式。

    支持格式：
      - "每天早上8点" → "0 8 * * *"
      - "每周一早上9点" → "0 9 * * 1"
      - "每周五下午5点" → "0 17 * * 5"
      - "每月1号上午9点" → "0 9 1 * *"
      - "每(周几)(时间)" 的各种变体

    Args:
        text: 中文重复时间表达
        ref: 参考时间，默认 now()

    Returns:
        cron 表达式字符串，解析失败返回错误描述
    """
    if ref is None:
        ref = datetime.now()

    # 解析时间点
    hour = 9  # 默认上午9点
    minute = 0
    time_match = re.search(r"(\d{1,2})[点:：](\d{1,2})?\s*(分)?", text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        # 12小时制转换
        if re.search(r"下午|晚上|夜里|傍晚", text) and 1 <= hour <= 11:
            hour += 12
    else:
        # 时段推断
        tod = _parse_time_of_day(text)
        if tod is not None:
            hour = tod

    # 每天
    if re.search(r"每[天一]|每天|每日", text):
        return f"{minute} {hour} * * *"

    # 每周
    week_match = re.search(r"每[周週]([一二三四五六日天])", text)
    if week_match:
        weekday_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 0, "天": 0}
        wd = weekday_map.get(week_match.group(1), 0)
        return f"{minute} {hour} * * {wd}"

    # 额外匹配模式："每个周一"、"每个工作日"
    if not week_match:
        week_match2 = re.search(r"每[个\s]*(?:星期|礼拜)?\s*([一二三四五六日天])", text)
        if week_match2:
            weekday_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 0, "天": 0}
            wd = weekday_map.get(week_match2.group(1), 0)
            return f"{minute} {hour} * * {wd}"

    # 每月
    month_match = re.search(r"每月\s*(\d{1,2})\s*[号日]", text)
    if month_match:
        day = int(month_match.group(1))
        return f"{minute} {hour} {min(day, 28)} * *"

    # 如果已经由 week_match 处理过，检查是否还有其他模式
    # 工作日
    if re.search(r"工作日|周一到周五|周一至周五", text):
        return f"{minute} {hour} * * 1-5"

    # 周末
    if re.search(r"周末|周六日|周六周日", text):
        return f"{minute} {hour} * * 6,0"

    return f"无法解析为 cron 表达式: '{text}'。请用更明确的表达，如'每天早上8点'、'每周一上午10点'、'每月1号上午9点'。"


def get_future_date(text: str, ref: Optional[datetime] = None) -> str:
    """
    计算未来日期，返回中文可读格式 + ISO datetime。

    支持：
      - "下周五" → "2026-05-23 (周五)"
      - "3天后" → "2026-05-21 (周四)"
      - "下周一是几月几号" → 精确日期

    Args:
        text: 中文日期表达
        ref: 参考时间

    Returns:
        包含日期和星期几的字符串
    """
    dt = parse_chinese_time_expression(text, ref)
    if dt is None:
        return f"无法计算日期: '{text}'。请尝试更明确的表达。"

    weekday = CHINESE_WEEKDAYS[dt.weekday()]
    return f"{dt.strftime('%Y-%m-%d')} ({weekday})"


def is_overdue(time_str: str, ref: Optional[datetime] = None) -> str:
    """
    判断一个时间是否已过期。

    Args:
        time_str: ISO datetime 字符串或中文时间表达
        ref: 参考时间，默认 now()

    Returns:
        包含判断结果和相对时间差的描述
    """
    if ref is None:
        ref = datetime.now()

    # 先尝试解析 ISO 格式
    dt = None
    try:
        if "T" in time_str or "-" in time_str:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00").split("+")[0].split(".")[0])
    except (ValueError, TypeError):
        pass

    # 再尝试中文表达
    if dt is None:
        dt = parse_chinese_time_expression(time_str, ref)

    if dt is None:
        return f"无法解析时间: '{time_str}'"

    diff = dt - ref
    total_seconds = diff.total_seconds()

    if total_seconds < 0:
        # 已过期
        total_seconds = abs(total_seconds)
        if total_seconds < 60:
            return f"已过期（{int(total_seconds)} 秒前）"
        elif total_seconds < 3600:
            return f"已过期（{int(total_seconds / 60)} 分钟前）"
        elif total_seconds < 86400:
            return f"已过期（{int(total_seconds / 3600)} 小时前）"
        else:
            return f"已过期（{int(total_seconds / 86400)} 天前，即 {dt.strftime('%Y-%m-%d %H:%M')}）"
    elif total_seconds == 0:
        return "就是现在"
    else:
        # 未过期
        if total_seconds < 60:
            return f"即将到来（{int(total_seconds)} 秒后）"
        elif total_seconds < 3600:
            return f"{int(total_seconds / 60)} 分钟后"
        elif total_seconds < 86400:
            return f"{int(total_seconds / 3600)} 小时后（{dt.strftime('%H:%M')}）"
        else:
            weekday = CHINESE_WEEKDAYS[dt.weekday()]
            return f"{int(total_seconds / 86400)} 天后（{dt.strftime('%Y-%m-%d')} {weekday} {dt.strftime('%H:%M')}）"
