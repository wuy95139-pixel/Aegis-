"""
认证中间件 (AuthMiddleware)
===========================
IP 白名单 + API Key 双重认证。

支持两种模式:
  - "or" (默认): IP 或 API Key 任一通过即放行
  - "and": IP 和 API Key 必须同时通过

配置方式（优先级: 环境变量 > config.yaml > 默认值）:
  - AEGIS_AUTH_MODE="or" | "and" | "off"
  - AEGIS_IP_WHITELIST="127.0.0.1,10.0.0.0/8,192.168.1.5"
  - AEGIS_API_KEYS="key1,key2,key3"

使用:
    from src.api.auth import AuthMiddleware

    app.add_middleware(
        AuthMiddleware,
        whitelist=["127.0.0.1", "10.0.0.0/8"],
        api_keys={"my-secret-key"},
        public_paths={"/health", "/api/health"},
        mode="or",
    )
"""

import os
import ipaddress
import logging
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.common import get_client_ip

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """IP 白名单 + API Key 双重认证中间件"""

    def __init__(
        self,
        app,
        whitelist: Optional[Iterable[str]] = None,
        api_keys: Optional[Iterable[str]] = None,
        public_paths: Optional[Iterable[str]] = None,
        public_prefixes: Optional[Iterable[str]] = None,
        mode: str = "or",
    ):
        super().__init__(app)
        self._ip_nets: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        if whitelist:
            self._ip_nets = self._parse_whitelist(whitelist)
        self._api_keys: Set[str] = set(api_keys) if api_keys else set()
        self._public_paths: Set[str] = set(public_paths) if public_paths else set()
        self._public_prefixes: tuple = tuple(public_prefixes) if public_prefixes else ()
        self._mode = mode.lower()

        if self._mode not in ("or", "and", "off"):
            raise ValueError(f"Invalid auth mode '{mode}': must be 'or', 'and', or 'off'")

        if self._mode != "off":
            logger.info(
                f"AuthMiddleware enabled: mode={self._mode}, "
                f"whitelist={len(self._ip_nets)} nets, "
                f"api_keys={len(self._api_keys)} keys"
            )
        else:
            logger.info("AuthMiddleware: authentication disabled")

    async def dispatch(self, request: Request, call_next):
        # 认证关闭
        if self._mode == "off":
            return await call_next(request)

        # 公开路径跳过认证
        path = request.url.path
        if path in self._public_paths or path.startswith(self._public_prefixes):
            return await call_next(request)

        # 无认证配置时放行（开发模式）
        if not self._ip_nets and not self._api_keys:
            return await call_next(request)

        ip_ok = self._check_ip(request)
        key_ok = self._check_api_key(request)

        if self._mode == "or":
            if not ip_ok and not key_ok:
                client_ip = self._get_client_ip(request)
                logger.warning(f"AUTH_DENIED: mode=or, ip={client_ip}, path={path}")
                return JSONResponse(
                    {"detail": "Access denied"},
                    status_code=403,
                )
        elif self._mode == "and":
            if not ip_ok:
                client_ip = self._get_client_ip(request)
                logger.warning(f"AUTH_DENIED: mode=and, reason=ip, ip={client_ip}, path={path}")
                return JSONResponse(
                    {"detail": "Access denied"},
                    status_code=403,
                )
            if not key_ok:
                client_ip = self._get_client_ip(request)
                logger.warning(f"AUTH_DENIED: mode=and, reason=api_key, ip={client_ip}, path={path}")
                return JSONResponse(
                    {"detail": "Access denied"},
                    status_code=401,
                )

        return await call_next(request)

    def _check_ip(self, request: Request) -> bool:
        """检查客户端 IP 是否在白名单中。无白名单时：
        - or 模式返回 False（不通过 IP 检查，但密钥可能通过）
        - and 模式返回 True（放行：and 意味着"必须同时通过"，但白名单为空说明不想限制 IP）"""
        if not self._ip_nets:
            return self._mode != "or"

        client_ip = self._get_client_ip(request)
        if not client_ip:
            return False

        try:
            addr = ipaddress.ip_address(client_ip)
        except ValueError:
            logger.debug(f"Invalid client IP: {client_ip}")
            return False

        for net in self._ip_nets:
            if addr in net:
                return True
        return False

    def _check_api_key(self, request: Request) -> bool:
        """检查 API Key 是否有效"""
        if not self._api_keys:
            return False

        api_key = request.headers.get("X-API-Key", "")
        return api_key in self._api_keys

    @staticmethod
    def _get_client_ip(request: Request) -> Optional[str]:
        """获取真实客户端 IP（委托给共享实现）"""
        return get_client_ip(request)

    @staticmethod
    def _parse_whitelist(entries: Iterable[str]) -> List:
        """解析 IP 白名单条目（支持 CIDR 和单个 IP）"""
        nets = []
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            try:
                nets.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                logger.warning(f"Invalid IP/network in whitelist: {entry}")
        return nets


def create_auth_middleware_from_env() -> AuthMiddleware:
    """
    从环境变量创建已配置的 AuthMiddleware 实例。

    环境变量:
      - AEGIS_AUTH_MODE: "or" | "and" | "off" (默认 "or")
      - AEGIS_IP_WHITELIST: 逗号分隔的 IP/CIDR 列表
      - AEGIS_API_KEYS: 逗号分隔的 API Key 列表

    Returns:
        配置好的 AuthMiddleware 实例
    """
    mode = os.getenv("AEGIS_AUTH_MODE", "or").strip().lower()

    whitelist_str = os.getenv("AEGIS_IP_WHITELIST", "")
    whitelist = [e.strip() for e in whitelist_str.split(",") if e.strip()] if whitelist_str else []

    keys_str = os.getenv("AEGIS_API_KEYS", "")
    api_keys = set(k.strip() for k in keys_str.split(",") if k.strip()) if keys_str else set()

    # 尝试从 Config 补充（环境变量优先）
    try:
        from src.utils.config import Config
        cfg = Config()
        if not whitelist:
            whitelist = cfg.get("auth.ip_whitelist", [])
        if not api_keys:
            keys_from_cfg = cfg.get("auth.api_keys", [])
            api_keys = set(keys_from_cfg)
        if mode == "or":
            mode = cfg.get("auth.mode", mode)
    except Exception as e:
        logger.debug("Failed to load auth config from Config, using env/cli values: %s", e)

    return AuthMiddleware(
        app=None,
        whitelist=whitelist,
        api_keys=api_keys,
        public_paths={"/health", "/api/health", "/"},
        public_prefixes=("/static/", "/dashboards/", "/reports/"),
        mode=mode,
    )
