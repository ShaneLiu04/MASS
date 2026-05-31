"""
MASS 爬虫 Session 连接池

按域名共享 requests.Session，实现 TCP 连接复用，减少握手开销。

设计:
- 同一域名组的爬虫共享同一个 Session（连接池）
- Session 自动复用 HTTP keep-alive 连接
- 提供适配器配置（连接池大小、重试策略）
- 域名级频率控制：同一 domain_key 的所有爬虫实例共享限流
"""
import random
import socket
import threading
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.util.retry import Retry
from loguru import logger

# 启用 TCP Keep-Alive，防止长连接因 NAT/防火墙超时断开
# 全局设置，影响所有 urllib3 HTTP 连接
HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
]


class SessionPool:
    """
    全局 Session 连接池（单例）
    
    用法:
        pool = SessionPool.get_instance()
        session = pool.get_session("eastmoney")
        # 所有 eastmoney 域名下的爬虫共享此 session
    """

    _instance: Optional["SessionPool"] = None
    _lock = threading.Lock()

    # 域名组映射：将相关域名归为一组共享 Session
    DEFAULT_DOMAIN_GROUPS = {
        "eastmoney": ["eastmoney.com", "eastmoney.cn"],
        "sina": ["sina.com.cn", "sinajs.cn", "sinaimg.cn"],
        "ths": ["10jqka.com.cn"],
        "tx": ["gtimg.cn", "qq.com"],
    }

    def __init__(
        self,
        pool_connections: int = 20,
        pool_maxsize: int = 20,
        max_retries: int = 3,
    ):
        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._max_retries = max_retries
        self._sessions: Dict[str, requests.Session] = {}
        self._lock = threading.Lock()

        # 域名级频率控制：同一 domain_key 的所有爬虫实例共享
        self._domain_last_request: Dict[str, float] = {}
        self._domain_locks: Dict[str, threading.Lock] = {}

    @classmethod
    def get_instance(cls) -> "SessionPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _create_session(self) -> requests.Session:
        """创建配置好的 Session"""
        session = requests.Session()

        # 配置连接池适配器
        # 加入 429（Too Many Requests）到重试列表，配合 raise_on_status=False
        # 避免重试库直接抛异常覆盖原始响应，让上层有机会读取 Retry-After 头
        adapter = HTTPAdapter(
            pool_connections=self._pool_connections,
            pool_maxsize=self._pool_maxsize,
            max_retries=Retry(
                total=self._max_retries,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504, 429],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                raise_on_status=False,
            ),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 统一基础请求头
        session.headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        })

        return session

    def get_session(self, domain_key: str) -> requests.Session:
        """
        获取指定域名组的 Session
        
        Args:
            domain_key: 域名组标识，如 "eastmoney", "sina", "ths", "tx"
        """
        with self._lock:
            if domain_key not in self._sessions:
                self._sessions[domain_key] = self._create_session()
                logger.debug(f"[SessionPool] 创建新 Session: {domain_key}")
            return self._sessions[domain_key]

    def resolve_domain_key(self, url: str) -> str:
        """
        根据 URL 解析域名组 key
        
        Args:
            url: 完整 URL，如 "https://push2.eastmoney.com/api/..."
        
        Returns:
            域名组 key，如 "eastmoney"
        """
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        hostname_lower = hostname.lower()

        for key, domains in self.DEFAULT_DOMAIN_GROUPS.items():
            for domain in domains:
                if domain in hostname_lower:
                    return key

        # 未匹配到已知组，返回 hostname 作为独立 key
        return hostname_lower.replace(".", "_")

    def close_all(self) -> None:
        """关闭所有 Session，释放连接池资源"""
        with self._lock:
            for key, session in self._sessions.items():
                session.close()
                logger.debug(f"[SessionPool] 关闭 Session: {key}")
            self._sessions.clear()

    def rate_limit(self, domain_key: str, interval: float) -> None:
        """
        域名级频率控制。

        同一 domain_key（如 'eastmoney', 'sina'）的所有爬虫实例共享
        同一个最后请求时间，确保对同一域名组的实际请求频率不会超标。

        Args:
            domain_key: 域名组标识，如 "eastmoney", "sina"
            interval: 两次请求之间的最小间隔（秒）
        """
        lock = self._domain_locks.setdefault(domain_key, threading.Lock())
        with lock:
            last = self._domain_last_request.get(domain_key, 0.0)
            elapsed = time.time() - last
            if elapsed < interval:
                sleep_time = interval - elapsed + random.uniform(0, 0.1)
                time.sleep(sleep_time)
            self._domain_last_request[domain_key] = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "session_keys": list(self._sessions.keys()),
                "rate_limited_domains": list(self._domain_last_request.keys()),
            }
