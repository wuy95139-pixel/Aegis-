"""
MCP (Model Context Protocol) 连接工具
====================================
为外部工具提供标准化的连接接口。

设计决策：
  - MCP 是 Anthropic 提出的开放协议，用于 AI 与外部工具的标准通信
  - 每个 MCP 服务对应一个客户端实例
  - 支持自动重连和健康检查
  - 当 MCP 不可用时，自动降级到本地 fallback

可扩展点：
  - 新增 MCP 服务：在 MCP_SERVICE_TYPES 注册类型，实现对应的 Client
  - 服务发现: 通过注册中心动态发现 MCP 服务
  - 鉴权: 支持 token / API Key / OAuth 等多种鉴权方式
"""

import logging
import os
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)

# 支持的 MCP 服务类型
MCP_SERVICE_TYPES = ["search", "translation", "calendar", "email", "storage"]


class MCPClient:
    """
    MCP 客户端基类 — 负责与外部工具服务的连接和通信

    使用示例:
        client = MCPClient("http://localhost:8001", "search")
        result = await client.call("search", {"query": "AI trends"})
    """

    def __init__(
        self,
        endpoint: str,
        service_type: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Args:
            endpoint: MCP 服务端点 URL
            service_type: 服务类型 (search / translation / calendar 等)
            api_key: API Key (如果需要)
            timeout: 请求超时秒数
        """
        self.endpoint = endpoint.rstrip("/")
        self.service_type = service_type
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._connected = False
        self.server_capabilities: Dict[str, Any] = {}
        self.server_info: Dict[str, Any] = {}
        self._transport: str = "http"  # http | stdio (未来扩展)

    def _mcp_send(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送 JSON-RPC 请求到 MCP 端点

        Args:
            payload: JSON-RPC 请求体

        Returns:
            解析后的完整响应数据或 None
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = self._client.post(
                f"{self.endpoint}/",
                json=payload,
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, dict):
                    logger.warning(f"MCP unexpected response type: {type(data).__name__}")
                    return None
                if "error" in data:
                    logger.warning(f"MCP JSON-RPC error: {data['error']}")
                    return None
                return data
            logger.warning(f"MCP HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as e:
            logger.warning(f"MCP send failed: {e}")
            return None

    def connect(self) -> bool:
        """
        连接到 MCP 服务 — 执行标准 JSON-RPC initialize 握手

        MCP 协议握手流程:
          1. Client → Server: initialize request
          2. Server → Client: initialize response (含 capabilities + serverInfo)
          3. Client → Server: notifications/initialized (确认就绪)

        Returns:
            是否连接成功
        """
        # Step 1-2: JSON-RPC initialize 握手
        init_data = self._mcp_send({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "aegis-mcp-client",
                    "version": "0.1.0",
                },
            },
            "id": 1,
        })

        if init_data:
            init_result = init_data.get("result", {})
            if isinstance(init_result, dict):
                self.server_capabilities = init_result.get("capabilities", {})
                self.server_info = init_result.get("serverInfo", {})
                logger.info(
                    f"MCP initialized: {self.service_type} @ {self.endpoint}, "
                    f"server={self.server_info.get('name', 'unknown')} "
                    f"v{self.server_info.get('version', '?')}"
                )

                # Step 3: 发送 initialized 通知
                self._mcp_send({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                })
            else:
                # 非标准 MCP 响应（如普通 HTTP API），标记已连接
                self.server_info = {"name": self.service_type, "version": "legacy"}
                logger.info(f"MCP connected (non-standard response): {self.service_type} @ {self.endpoint}")

            self._connected = True
            return True

        # 兼容性回退: 尝试旧的 /health 端点
        logger.info(f"MCP initialize handshake failed for {self.service_type}, trying /health fallback")
        try:
            resp = self._client.get(f"{self.endpoint}/health")
            if resp.status_code == 200:
                self._connected = True
                logger.info(f"MCP connected (legacy /health): {self.service_type} @ {self.endpoint}")
                return True
        except Exception as e:
            logger.warning(f"MCP connect failed to {self.service_type} @ {self.endpoint}: {e}")

        self._connected = False
        return False

    def call(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用 MCP 服务

        Args:
            method: 方法名
            params: 参数

        Returns:
            服务返回结果
        """
        if not self._connected:
            logger.warning(f"MCP {self.service_type} not connected, calling connect()")
            if not self.connect():
                return {"error": f"MCP service {self.service_type} is not available"}

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": 1,
            }

            resp = self._client.post(
                f"{self.endpoint}/rpc",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"MCP call failed: {resp.status_code} - {resp.text}")
                return {"error": f"HTTP {resp.status_code}"}

        except Exception as e:
            logger.error(f"MCP call exception: {e}")
            return {"error": str(e)}

    async def acall(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """异步调用 MCP 服务"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                    "id": 1,
                }
                resp = await client.post(
                    f"{self.endpoint}/rpc",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
                return resp.json()
            except Exception as e:
                logger.error(f"MCP async call exception: {e}")
                return {"error": str(e)}

    def close(self):
        """关闭连接"""
        self._client.close()
        self._connected = False


class MCPManager:
    """
    MCP 服务管理器 — 统一管理所有 MCP 连接

    使用示例:
        mgr = MCPManager(config)
        mgr.register("search", "http://localhost:8001")
        result = mgr.call("search", "web_search", {"query": "AI"})
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Aegis 全局配置
        """
        self._clients: Dict[str, MCPClient] = {}
        self.config = config or {}

        # 从配置中加载 MCP 服务
        mcp_servers = self.config.get("mcp_servers", {})
        for svc_name, svc_config in mcp_servers.items():
            if svc_config.get("enabled") and svc_config.get("endpoint"):
                self.register(
                    service_type=svc_config.get("type", svc_name),
                    endpoint=svc_config["endpoint"],
                    name=svc_name,
                )

    def register(
        self,
        service_type: str,
        endpoint: str,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> MCPClient:
        """
        注册一个 MCP 服务

        Args:
            service_type: 服务类型
            endpoint: 服务端点
            name: 逻辑名称 (用于后续引用)
            api_key: API Key

        Returns:
            MCPClient 实例
        """
        client = MCPClient(endpoint, service_type, api_key)
        client_name = name or service_type
        self._clients[client_name] = client
        logger.info(f"MCP registered: {client_name} ({service_type}) @ {endpoint}")
        return client

    def call(self, service_name: str, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用指定 MCP 服务

        Args:
            service_name: 注册时的逻辑名称
            method: 方法名
            params: 参数

        Returns:
            服务返回结果
        """
        client = self._clients.get(service_name)
        if not client:
            return {"error": f"MCP service '{service_name}' not registered"}

        return client.call(method, params)

    def is_available(self, service_name: str) -> bool:
        """检查 MCP 服务是否可用"""
        client = self._clients.get(service_name)
        return client is not None and client._connected

    def list_services(self) -> List[str]:
        """列出所有已注册的 MCP 服务"""
        return list(self._clients.keys())

    def close_all(self):
        """关闭所有连接"""
        for client in self._clients.values():
            client.close()
        logger.info("All MCP connections closed")
