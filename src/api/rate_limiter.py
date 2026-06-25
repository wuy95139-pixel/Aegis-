"""
速率限制中间件 (RateLimiter)
============================
基于滑动窗口的每 IP 速率限制。

策略:
  - 滑动窗口计数：在配置的时间窗口内限制每个 IP 的请求数
  - 可配置不同端点的不同限制
  - 内存存储（无需 Redis），适合单机部署
  - 自动清理过期条目防止内存泄漏

配置:
  - AEGIS_RATE_LIMIT_ENABLED="true" | "false"
  - AEGIS_RATE_LIMIT_REQUESTS=60       # 默认每窗口最大请求数
  - AEGIS_RATE_LIMIT_WINDOW=60         # 默认窗口秒数

使用:
    from src.api.rate_limiter import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        default_limit=60,       # 默认每窗口 60 次
        window_seconds=60,      # 60 秒窗口
        public_paths={"/health", "/api/health"},
    )
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.common import get_client_ip

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """滑动窗口速率限制中间件"""

    def __init__(
        self,
        app,
        default_limit: int = 60,
        window_seconds: float = 60.0,
        public_paths: Optional[Set[str]] = None,
        public_prefixes: Optional[Tuple[str, ...]] = None,
        path_limits: Optional[Dict[str, int]] = None,
    ):
        super().__init__(app)
        self._default_limit = default_limit
        self._window_seconds = window_seconds
        self._public_paths = public_paths or set()
        self._public_prefixes = public_prefixes or ()
        self._path_limits = path_limits or {}

        # 存储: {client_ip: [(timestamp, count), ...]}
        self._windows: Dict[str, List[Tuple[float, int]]] = {}
        self._lock = threading.Lock()

        # 定期清理（每 5 分钟清理一次过期条目）
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

        logger.info(
            f"RateLimitMiddleware: {default_limit} req/{window_seconds}s window"
            + (f", {len(path_limits)} path-specific limits" if path_limits else "")
        )

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 公开路径不限流
        if path in self._public_paths or path.startswith(self._public_prefixes):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        if not client_ip:
            return await call_next(request)

        limit = self._path_limits.get(path, self._default_limit)

        if self._is_rate_limited(client_ip, limit):
            logger.debug(f"Rate limited: {client_ip} on {path}")
            return JSONResponse(
                {"detail": "Too many requests. Please try again later."},
                status_code=429,
            )

        return await call_next(request)

    def _is_rate_limited(self, client_ip: str, limit: int) -> bool:
        """滑动窗口检查：当前窗口内请求数是否超过限制"""
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # 定期清理
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired(now)
                self._last_cleanup = now

            if client_ip not in self._windows:
                self._windows[client_ip] = [(now, 1)]
                return False

            entries = self._windows[client_ip]

            # 移除窗口外的条目并计数窗口内请求
            window_count = 0
            new_entries = []
            for ts, count in entries:
                if ts >= window_start:
                    window_count += count
                    new_entries.append((ts, count))

            if window_count >= limit:
                self._windows[client_ip] = new_entries
                return True

            # 合并同一秒内的计数（减少内存占用）
            if new_entries and now - new_entries[-1][0] < 1.0:
                last_ts, last_count = new_entries[-1]
                new_entries[-1] = (last_ts, last_count + 1)
            else:
                new_entries.append((now, 1))

            self._windows[client_ip] = new_entries
            return False

    def _cleanup_expired(self, now: float):
        """清理所有过期条目"""
        window_start = now - self._window_seconds
        expired_ips = []
        for ip, entries in self._windows.items():
            fresh = [(ts, c) for ts, c in entries if ts >= window_start]
            if fresh:
                self._windows[ip] = fresh
            else:
                expired_ips.append(ip)
        for ip in expired_ips:
            del self._windows[ip]

    @staticmethod
    def _get_client_ip(request: Request) -> Optional[str]:
        return get_client_ip(request)
