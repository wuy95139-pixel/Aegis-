"""
api/rate_limiter.py 测试
========================
RateLimitMiddleware 的滑动窗口、路径限流、清理、公共路径测试。
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch

from src.api.rate_limiter import RateLimitMiddleware


def _make_request(client_ip="127.0.0.1", path="/api/chat"):
    """创建 mock Request。"""
    req = MagicMock()
    req.url.path = path
    req.client.host = client_ip
    return req


async def _call_next(request):
    return {"status": "ok"}


def _dispatch(middleware, request):
    """同步运行 dispatch 协程。"""
    return asyncio.run(middleware.dispatch(request, _call_next))


# ==================== Public Path Tests ====================

class TestPublicPath:
    def test_exact_public_path_bypasses(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "10.0.0.1")
        middleware = RateLimitMiddleware(app=None, public_paths={"/health"})
        resp = _dispatch(middleware, _make_request(path="/health"))
        assert resp == {"status": "ok"}

    def test_public_prefix_bypasses(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "10.0.0.1")
        middleware = RateLimitMiddleware(app=None, public_prefixes=("/static/",))
        resp = _dispatch(middleware, _make_request(path="/static/app.js"))
        assert resp == {"status": "ok"}

    def test_no_ip_bypasses(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: None)
        middleware = RateLimitMiddleware(app=None)
        resp = _dispatch(middleware, _make_request())
        assert resp == {"status": "ok"}


# ==================== Rate Limit Tests ====================

class TestRateLimit:
    def test_within_limit_allows(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "10.0.0.1")
        middleware = RateLimitMiddleware(app=None, default_limit=10)
        resp = _dispatch(middleware, _make_request())
        assert resp == {"status": "ok"}

    def test_over_limit_returns_429(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "10.0.0.1")
        middleware = RateLimitMiddleware(app=None, default_limit=3)
        req = _make_request()
        for _ in range(3):
            resp = _dispatch(middleware, req)
            assert resp == {"status": "ok"}
        resp = _dispatch(middleware, req)
        assert resp.status_code == 429

    def test_path_specific_limit(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "10.0.0.1")
        middleware = RateLimitMiddleware(
            app=None, default_limit=100, path_limits={"/api/expensive": 2}
        )
        req = _make_request(path="/api/expensive")
        for _ in range(2):
            resp = _dispatch(middleware, req)
            assert resp == {"status": "ok"}
        resp = _dispatch(middleware, req)
        assert resp.status_code == 429


# ==================== Is Rate Limited Unit Tests ====================

class TestIsRateLimited:
    def test_first_request_not_limited(self):
        middleware = RateLimitMiddleware(app=None, default_limit=10)
        assert middleware._is_rate_limited("10.0.0.1", 10) is False

    def test_increments_count(self):
        middleware = RateLimitMiddleware(app=None, default_limit=10)
        for _ in range(5):
            assert middleware._is_rate_limited("10.0.0.2", 10) is False
        assert middleware._is_rate_limited("10.0.0.2", 5) is True

    def test_different_ips_independent(self):
        middleware = RateLimitMiddleware(app=None, default_limit=3)
        for _ in range(3):
            assert middleware._is_rate_limited("10.0.0.1", 3) is False
        assert middleware._is_rate_limited("10.0.0.2", 3) is False

    def test_same_second_merge(self):
        """同一秒内的请求合并计数条目。"""
        middleware = RateLimitMiddleware(app=None, default_limit=100)
        with patch.object(time, 'time', return_value=1000.5):
            middleware._is_rate_limited("10.0.0.1", 100)
            middleware._is_rate_limited("10.0.0.1", 100)
            middleware._is_rate_limited("10.0.0.1", 100)
        assert len(middleware._windows.get("10.0.0.1", [])) == 1
        assert middleware._windows["10.0.0.1"][0][1] == 3


# ==================== Cleanup Tests ====================

class TestCleanup:
    def test_removes_expired_entries(self):
        middleware = RateLimitMiddleware(app=None, window_seconds=60.0)
        with middleware._lock:
            middleware._windows["old_ip"] = [(time.time() - 120, 5)]
        middleware._cleanup_expired(time.time())
        assert "old_ip" not in middleware._windows

    def test_preserves_fresh_entries(self):
        middleware = RateLimitMiddleware(app=None, window_seconds=60.0)
        now = time.time()
        with middleware._lock:
            middleware._windows["fresh_ip"] = [(now - 30, 3)]
        middleware._cleanup_expired(now)
        assert "fresh_ip" in middleware._windows


# ==================== Get Client IP Tests ====================

class TestGetClientIP:
    def test_returns_ip(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limiter.get_client_ip", lambda r: "192.168.1.1")
        req = Mock()
        ip = RateLimitMiddleware._get_client_ip(req)
        assert ip == "192.168.1.1"
