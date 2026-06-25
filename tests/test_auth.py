"""
api/auth.py 测试
================
AuthMiddleware 的 IP 白名单、API Key 检查、模式验证、环境变量工厂测试。
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
import ipaddress

from src.api.auth import AuthMiddleware, create_auth_middleware_from_env


def _make_request(client_ip="127.0.0.1", path="/api/chat", api_key=None):
    """创建 mock Request。"""
    req = MagicMock()
    req.url.path = path
    req.client.host = client_ip
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    req.headers.get = lambda k, default="": headers.get(k.lower(), default)
    return req


async def _call_next(request):
    return {"status": "ok"}


def _dispatch(middleware, request):
    """同步运行 dispatch 协程。"""
    return asyncio.run(middleware.dispatch(request, _call_next))


# ==================== Mode Tests ====================

class TestAuthMode:
    def test_mode_off_bypasses(self):
        middleware = AuthMiddleware(app=None, mode="off")
        resp = _dispatch(middleware, _make_request())
        assert resp == {"status": "ok"}

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid auth mode"):
            AuthMiddleware(app=None, mode="invalid")

    def test_mode_or_allows_ip_only(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "10.0.0.1")
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], mode="or",
        )
        resp = _dispatch(middleware, _make_request())
        assert resp == {"status": "ok"}

    def test_mode_or_allows_key_only(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "1.2.3.4")
        middleware = AuthMiddleware(
            app=None, api_keys=["secret"], mode="or",
        )
        resp = _dispatch(middleware, _make_request(api_key="secret"))
        assert resp == {"status": "ok"}

    def test_mode_or_denies_both_fail(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "1.2.3.4")
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"], mode="or",
        )
        resp = _dispatch(middleware, _make_request())
        assert resp.status_code == 403

    def test_mode_and_requires_both(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "10.0.0.1")
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"], mode="and",
        )
        resp = _dispatch(middleware, _make_request(api_key="secret"))
        assert resp == {"status": "ok"}

    def test_mode_and_denies_ip_fail(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "1.2.3.4")
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"], mode="and",
        )
        resp = _dispatch(middleware, _make_request(api_key="secret"))
        assert resp.status_code == 403

    def test_mode_and_denies_key_fail(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "10.0.0.1")
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"], mode="and",
        )
        resp = _dispatch(middleware, _make_request(api_key="wrong"))
        assert resp.status_code == 401


# ==================== Public Path Tests ====================

class TestPublicPath:
    def test_exact_public_path_bypasses(self):
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"],
            mode="and", public_paths={"/health"},
        )
        resp = _dispatch(middleware, _make_request(path="/health"))
        assert resp == {"status": "ok"}

    def test_public_prefix_bypasses(self):
        middleware = AuthMiddleware(
            app=None, whitelist=["10.0.0.0/8"], api_keys=["secret"],
            mode="and", public_prefixes={"/static/"},
        )
        resp = _dispatch(middleware, _make_request(path="/static/app.js"))
        assert resp == {"status": "ok"}


# ==================== No Config Bypass ====================

class TestNoConfig:
    def test_no_whitelist_no_keys_bypasses(self):
        middleware = AuthMiddleware(app=None, mode="or")
        resp = _dispatch(middleware, _make_request())
        assert resp == {"status": "ok"}


# ==================== IP Check Tests ====================

class TestIPCheck:
    def test_single_ip_match(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "192.168.1.5")
        middleware = AuthMiddleware(app=None, whitelist=["192.168.1.5"])
        assert middleware._check_ip(Mock()) is True

    def test_cidr_match(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "10.0.0.55")
        middleware = AuthMiddleware(app=None, whitelist=["10.0.0.0/8"])
        assert middleware._check_ip(Mock()) is True

    def test_ip_not_in_whitelist(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "1.2.3.4")
        middleware = AuthMiddleware(app=None, whitelist=["10.0.0.0/8"])
        assert middleware._check_ip(Mock()) is False

    def test_no_ip_available(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: None)
        middleware = AuthMiddleware(app=None, whitelist=["10.0.0.0/8"])
        assert middleware._check_ip(Mock()) is False

    def test_invalid_ip_address(self, monkeypatch):
        monkeypatch.setattr("src.api.auth.get_client_ip", lambda r: "not-an-ip")
        middleware = AuthMiddleware(app=None, whitelist=["10.0.0.0/8"])
        assert middleware._check_ip(Mock()) is False

    def test_no_whitelist_or_mode_returns_false(self):
        middleware = AuthMiddleware(app=None, mode="or")
        assert middleware._check_ip(Mock()) is False

    def test_no_whitelist_and_mode_returns_true(self):
        middleware = AuthMiddleware(app=None, mode="and")
        assert middleware._check_ip(Mock()) is True


# ==================== API Key Check Tests ====================

class TestAPIKeyCheck:
    def test_valid_key(self):
        middleware = AuthMiddleware(app=None, api_keys=["secret-key"])
        req = Mock()
        req.headers.get = lambda k, d="": "secret-key" if k.lower() == "x-api-key" else d
        assert middleware._check_api_key(req) is True

    def test_invalid_key(self):
        middleware = AuthMiddleware(app=None, api_keys=["secret-key"])
        req = Mock()
        req.headers.get = lambda k, d="": "wrong-key" if k.lower() == "x-api-key" else d
        assert middleware._check_api_key(req) is False

    def test_missing_key_header(self):
        middleware = AuthMiddleware(app=None, api_keys=["secret-key"])
        req = Mock()
        req.headers.get = lambda k, d="": d
        assert middleware._check_api_key(req) is False

    def test_no_api_keys_configured(self):
        middleware = AuthMiddleware(app=None)
        assert middleware._check_api_key(Mock()) is False


# ==================== Whitelist Parse Tests ====================

class TestParseWhitelist:
    def test_parses_cidr_and_ip(self):
        nets = AuthMiddleware._parse_whitelist(["192.168.1.1", "10.0.0.0/8"])
        assert len(nets) == 2

    def test_skips_invalid_entries(self):
        nets = AuthMiddleware._parse_whitelist(["invalid", "192.168.1.1", "", "not-ip"])
        assert len(nets) == 1

    def test_empty_entries_returns_empty(self):
        nets = AuthMiddleware._parse_whitelist([])
        assert nets == []


# ==================== Factory Tests ====================

class TestCreateAuthMiddlewareFromEnv:
    def test_returns_auth_middleware(self, monkeypatch):
        monkeypatch.setenv("AEGIS_AUTH_MODE", "or")
        middleware = create_auth_middleware_from_env()
        assert isinstance(middleware, AuthMiddleware)

    def test_defaults_to_or_mode(self, monkeypatch):
        monkeypatch.delenv("AEGIS_AUTH_MODE", raising=False)
        middleware = create_auth_middleware_from_env()
        assert middleware._mode == "or"

    def test_parses_env_whitelist(self, monkeypatch):
        monkeypatch.setenv("AEGIS_IP_WHITELIST", "10.0.0.1,192.168.0.0/16")
        middleware = create_auth_middleware_from_env()
        assert len(middleware._ip_nets) == 2

    def test_parses_env_api_keys(self, monkeypatch):
        monkeypatch.setenv("AEGIS_API_KEYS", "key1,key2")
        middleware = create_auth_middleware_from_env()
        assert len(middleware._api_keys) == 2

    def test_off_mode(self, monkeypatch):
        monkeypatch.setenv("AEGIS_AUTH_MODE", "off")
        middleware = create_auth_middleware_from_env()
        assert middleware._mode == "off"
