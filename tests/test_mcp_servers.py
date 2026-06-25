"""
core/tools/mcp_tools.py 测试
============================
MCPClient 和 MCPManager 的连接、调用、异常处理测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.tools.mcp_tools import MCPClient, MCPManager
import httpx


# ==================== MCPClient Tests ====================

class TestMCPClientInit:
    def test_stores_endpoint(self):
        client = MCPClient("http://localhost:8001", "search")
        assert client.endpoint == "http://localhost:8001"
        assert client.service_type == "search"

    def test_strips_trailing_slash(self):
        client = MCPClient("http://localhost:8001/", "translation")
        assert client.endpoint == "http://localhost:8001"

    def test_default_timeout(self):
        client = MCPClient("http://localhost:8001", "search")
        assert client.timeout == 30

    def test_custom_timeout(self):
        client = MCPClient("http://localhost:8001", "search", timeout=10)
        assert client.timeout == 10

    def test_with_api_key(self):
        client = MCPClient("http://localhost:8001", "search", api_key="secret")
        assert client.api_key == "secret"


class TestMCPClientConnect:
    def test_connect_success(self):
        with patch.object(httpx.Client, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            client = MCPClient("http://localhost:8001", "search")
            result = client.connect()
            assert result is True
            assert client._connected is True

    def test_connect_failure(self):
        with patch.object(httpx.Client, 'get', side_effect=Exception("Connection refused")):
            client = MCPClient("http://localhost:8001", "search")
            result = client.connect()
            assert result is False
            assert client._connected is False

    def test_connect_non_200(self):
        with patch.object(httpx.Client, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 503
            mock_get.return_value = mock_resp

            client = MCPClient("http://localhost:8001", "search")
            result = client.connect()
            assert result is False


class TestMCPClientCall:
    def test_call_connected(self):
        with patch.object(httpx.Client, 'get') as mock_get, \
             patch.object(httpx.Client, 'post') as mock_post:
            mock_get.return_value = Mock(status_code=200)
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"result": "success"}
            mock_post.return_value = mock_resp

            client = MCPClient("http://localhost:8001", "search", api_key="key1")
            client.connect()
            result = client.call("search", {"query": "test"})
            assert result["result"] == "success"

    def test_call_not_connected_cant_reconnect(self):
        """未连接且重连失败时返回错误。"""
        with patch.object(httpx.Client, 'get', side_effect=Exception("Connection refused")):
            client = MCPClient("http://localhost:8001", "search")
            result = client.call("search", {"query": "test"})
            assert "error" in result


class TestMCPClientClose:
    def test_close_disconnects(self):
        client = MCPClient("http://localhost:8001", "search")
        client._connected = True
        client.close()
        assert client._connected is False


# ==================== MCPManager Tests ====================

class TestMCPManager:
    @pytest.fixture
    def manager(self):
        return MCPManager()

    def test_register_service(self, manager):
        client = manager.register(
            service_type="search",
            endpoint="http://localhost:8001/mcp",
            name="my_search",
            api_key="key1",
        )
        assert isinstance(client, MCPClient)
        assert "my_search" in manager._clients
        assert manager._clients["my_search"].endpoint == "http://localhost:8001/mcp"

    def test_register_default_name(self, manager):
        client = manager.register(
            service_type="search",
            endpoint="http://localhost:8001",
        )
        assert "search" in manager._clients

    def test_call_registered_service(self, manager):
        with patch.object(httpx.Client, 'get') as mock_get, \
             patch.object(httpx.Client, 'post') as mock_post:
            mock_get.return_value = Mock(status_code=200)
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"result": "found"}
            mock_post.return_value = mock_resp

            manager.register("search", "http://localhost:8001", name="search1")
            result = manager.call("search1", "search", {"q": "AI"})
            assert "result" in result

    def test_call_unregistered_service(self, manager):
        result = manager.call("nonexistent", "search", {})
        assert "error" in result
        assert "not registered" in result["error"].lower()

    def test_is_available_connected(self, manager):
        with patch.object(httpx.Client, 'get') as mock_get:
            mock_get.return_value = Mock(status_code=200)
            client = manager.register("search", "http://localhost:8001", name="s1")
            client.connect()
        assert manager.is_available("s1") is True

    def test_is_available_not_connected(self, manager):
        manager.register("search", "http://localhost:8001", name="s2")
        assert manager.is_available("s2") is False

    def test_is_available_not_registered(self, manager):
        assert manager.is_available("ghost") is False

    def test_list_services(self, manager):
        manager.register("search", "http://localhost:8001", name="a")
        manager.register("translation", "http://localhost:8002", name="b")
        services = manager.list_services()
        assert "a" in services
        assert "b" in services
        assert len(services) == 2

    def test_close_all(self, manager):
        manager.register("search", "http://localhost:8001", name="a")
        manager.register("translation", "http://localhost:8002", name="b")
        manager.close_all()
        for client in manager._clients.values():
            assert client._connected is False
