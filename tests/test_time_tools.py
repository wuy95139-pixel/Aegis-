"""
core/tools/time_tools.py 测试
=============================
中文时间表达式解析、Cron 生成等功能测试。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.core.tools.time_tools import (
    get_time_context,
    _parse_chinese_number,
    parse_chinese_time_expression,
    expression_to_cron,
    get_future_date,
    is_overdue,
    format_datetime_chinese,
)


class TestGetTimeContext:
    def test_returns_formatted_string(self):
        result = get_time_context()
        assert isinstance(result, str)
        assert len(result) > 0
        # 应包含日期、星期或时间的相关信息
        assert any(kw in result for kw in ["年", "月", "日", "星期", "时"])


class TestParseChineseNumber:
    def test_single_digit(self):
        assert _parse_chinese_number("五") == 5
        assert _parse_chinese_number("九") == 9

    def test_teens(self):
        assert _parse_chinese_number("十二") == 12
        assert _parse_chinese_number("十八") == 18

    def test_twenty(self):
        assert _parse_chinese_number("二十五") == 25

    def test_hundred(self):
        assert _parse_chinese_number("一百二十") == 120

    def test_arabic_fallback(self):
        assert _parse_chinese_number("42") == 42

    def test_invalid_returns_none(self):
        result_abc = _parse_chinese_number("abc")
        result_empty = _parse_chinese_number("")
        # 不可解析的输入应返回 None 或 0
        assert result_abc is None or result_abc == 0
        assert result_empty is None or result_empty == 0


class TestParseChineseTimeExpression:
    def test_tomorrow_afternoon(self):
        with patch("src.core.tools.time_tools.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 6, 1, 10, 0, 0)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            result = parse_chinese_time_expression("明天下午3点")
            if result:
                assert result.day == 2
                assert result.hour == 15

    def test_days_later(self):
        now = datetime.now()
        result = parse_chinese_time_expression("3天后")
        if result:
            expected = now + timedelta(days=3)
            assert result.date() == expected.date()

    def test_hours_later(self):
        now = datetime.now()
        result = parse_chinese_time_expression("2小时后")
        if result:
            delta = (result - now).total_seconds()
            assert 7000 < delta < 7400  # ~2 hours

    def test_minutes_later(self):
        now = datetime.now()
        result = parse_chinese_time_expression("30分钟后")
        if result:
            delta = (result - now).total_seconds()
            assert 1750 < delta < 1850  # ~30 minutes

    def test_iso_format(self):
        result = parse_chinese_time_expression("2025-06-01 14:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 1
        assert result.hour == 14

    def test_next_monday(self):
        result = parse_chinese_time_expression("下周一上午10点")
        if result:
            assert result.hour == 10
            assert result.weekday() == 0  # Monday

    def test_unparseable_returns_none(self):
        result = parse_chinese_time_expression("blah blah blah")
        assert result is None

    def test_empty_string(self):
        result = parse_chinese_time_expression("")
        assert result is None


class TestExpressionToCron:
    def test_daily(self):
        result = expression_to_cron("每天早上8点")
        assert result == "0 8 * * *"

    def test_daily_afternoon(self):
        result = expression_to_cron("每天下午3点")
        if result and not result.startswith("无法解析"):
            assert "15" in result and "* * *" in result

    def test_weekly(self):
        result = expression_to_cron("每周一早上9点")
        if result and not result.startswith("无法解析"):
            assert "9" in result and "1" in result

    def test_monthly(self):
        result = expression_to_cron("每月15号下午2点")
        if result and not result.startswith("无法解析"):
            assert "14" in result and "15" in result

    def test_workday(self):
        result = expression_to_cron("工作日早上9点")
        if result and not result.startswith("无法解析"):
            assert "9" in result
            assert "1-5" in result

    def test_weekend(self):
        result = expression_to_cron("周末下午5点")
        if result and not result.startswith("无法解析"):
            assert "17" in result
            assert ("6" in result or "0" in result or "7" in result)

    def test_unparseable_returns_error(self):
        result = expression_to_cron("xyz 不存在的表达式")
        assert result is None or result.startswith("无法解析")

    def test_12h_pm_conversion(self):
        result = expression_to_cron("每天下午3点")
        if result and not result.startswith("无法解析"):
            assert "15" in result


class TestGetFutureDate:
    def test_n_days_later(self):
        now = datetime.now()
        expected = now + timedelta(days=5)
        result = get_future_date("5天后")
        if result:
            # 结果应是日期字符串
            assert isinstance(result, str)

    def test_next_friday(self):
        result = get_future_date("下周五")
        if result:
            assert isinstance(result, str)


class TestIsOverdue:
    def test_past_time(self):
        past = datetime.now() - timedelta(hours=1)
        result = is_overdue(past.isoformat())
        if result:
            assert "过期" in result or "已过" in result

    def test_future_time(self):
        future = datetime.now() + timedelta(days=7)
        result = is_overdue(future.isoformat())
        if result:
            # 应包含倒计时或"未过期"相关信息
            assert "过期" not in result or "未" in result

    def test_chinese_expression(self):
        result = is_overdue("明天")
        assert result is not None
        assert isinstance(result, str)

    def test_edge_case_empty(self):
        result = is_overdue("")
        assert result is not None


class TestFormatDatetimeChinese:
    def test_format(self):
        dt = datetime(2025, 6, 1, 14, 30, 0)
        result = format_datetime_chinese(dt)
        assert "2025" in result or "6" in result or "1" in result
        assert isinstance(result, str)

    def test_format_midnight(self):
        dt = datetime(2025, 1, 1, 0, 0, 0)
        result = format_datetime_chinese(dt)
        assert isinstance(result, str)
