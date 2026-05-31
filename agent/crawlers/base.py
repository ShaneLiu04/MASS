"""
MASS 爬虫基类
提供统一的请求管理、重试、请求间隔、UA轮换、断路器
"""
import json
import time
import random
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import requests
from loguru import logger

from agent.crawlers.utils import UserAgentRotator, validate_data
from agent.crawlers.session_pool import SessionPool
from agent.crawlers.circuit_breaker import CircuitBreaker


class DataQualityError(Exception):
    """数据质量异常"""
    pass


class BaseCrawler(ABC):
    """
    爬虫抽象基类

    特性:
    - 通过 SessionPool 按域名复用 TCP 连接
    - 集成 CircuitBreaker，连续失败自动熔断

    子类必须实现:
    - name: 爬虫标识名
    - priority: 优先级（越大越优先）
    - fetch(): 统一数据获取接口
    - domain_key: 域名组标识，用于连接复用（如 "eastmoney", "sina"）
    """

    name: str = "base"
    priority: int = 0
    timeout: int = 30
    retries: int = 3
    retry_delay: float = 1.0
    request_interval: float = 0.5
    domain_key: str = "default"

    def __init__(self):
        self._session_pool = SessionPool.get_instance()
        self._session: Optional[requests.Session] = None
        self._circuit_breaker = CircuitBreaker(
            name=self.name,
            failure_threshold=5,
            recovery_timeout=60.0,
        )
        self._ua_rotator = UserAgentRotator()

    def _get_session(self, url: str) -> requests.Session:
        """获取（或复用）对应域名的 Session"""
        if self._session is None:
            # 优先使用子类定义的 domain_key，否则从 URL 解析
            key = self.domain_key if self.domain_key != "default" else self._session_pool.resolve_domain_key(url)
            self._session = self._session_pool.get_session(key)
        return self._session

    def _rate_limit(self) -> None:
        """
        请求频率控制：委托给 SessionPool 按 domain_key 共享。

        同一域名组（如 eastmoney）的所有爬虫实例共享频率控制，
        避免多个实例同时对同一域名发起请求导致频率翻倍。
        """
        self._session_pool.rate_limit(self.domain_key, self.request_interval)

    def _execute_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[requests.Response]:
        """
        统一执行 HTTP 请求的核心方法。

        负责：
        - 断路器保护（快速失败）
        - Session 连接复用
        - 自动频率控制
        - UA 轮换
        - 指数退避重试
        - 异常捕获与日志

        Returns:
            成功时返回 requests.Response 对象，失败返回 None
        """
        if not self._circuit_breaker.can_execute():
            logger.warning(f"[{self.name}] 断路器 OPEN，跳过请求: {url}")
            return None

        self._rate_limit()

        # 轮换 UA
        req_headers = {"User-Agent": self._ua_rotator.get()}
        if headers:
            req_headers.update(headers)

        session = self._get_session(url)
        last_exception = None
        effective_timeout = timeout if timeout is not None else self.timeout

        for attempt in range(1, self.retries + 1):
            try:
                logger.debug(f"[{self.name}] 请求 {url} (尝试 {attempt}/{self.retries})")
                response = session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=req_headers,
                    timeout=effective_timeout,
                    **kwargs
                )
                response.raise_for_status()
                self._circuit_breaker.record_success()
                return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"[{self.name}] 请求超时 ({attempt}/{self.retries}): {url}")
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status = e.response.status_code if e.response else "unknown"
                logger.warning(f"[{self.name}] HTTP错误 {status} ({attempt}/{self.retries}): {url}")
                # 403/429 可能被封，增加更长等待
                if e.response and e.response.status_code in (403, 429):
                    time.sleep(self.retry_delay * (2 ** attempt) + random.uniform(1, 3))
                    continue
            except Exception as e:
                last_exception = e
                logger.warning(f"[{self.name}] 请求异常 ({attempt}/{self.retries}): {e}")

            # 指数退避
            if attempt < self.retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(sleep_time)

        # 所有重试耗尽，记录断路器失败
        self._circuit_breaker.record_failure()
        logger.warning(f"[{self.name}] 请求最终失败: {url}，错误: {last_exception}")
        return None

    def _request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        统一请求方法 — JSON 解析版。

        在 _execute_request 基础上解析 JSON 响应，支持 JSONP 自动提取。
        """
        response = self._execute_request(
            url, method=method, params=params, headers=headers, **kwargs
        )
        if response is None:
            return None

        try:
            return response.json()
        except ValueError:
            # 可能是 JSONP，尝试提取 JSON 部分
            text = response.text
            if text.startswith("(") and text.endswith(")"):
                text = text[1:-1]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"[{self.name}] JSON 解析失败: {url}")
                return None

    def _request_text(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        统一请求方法 — 原始文本版。

        在 _execute_request 基础上返回原始响应文本，
        适用于 Sina/Tx 等非 JSON 格式的接口。

        Args:
            encoding: 覆盖响应的字符编码（如 'gbk'），None 则自动检测
        """
        response = self._execute_request(
            url, method=method, params=params, headers=headers, **kwargs
        )
        if response is None:
            return None

        if encoding:
            response.encoding = encoding
        return response.text.strip()

    @abstractmethod
    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        统一数据获取接口

        Args:
            stock_code: 股票代码（6位数字，如 '600000'）
            data_type: 数据类型，如 "fundamentals", "fund_flow", "market_context", "macro", "sentiment"
            **kwargs: 额外参数

        Returns:
            标准化后的数据字典，获取失败返回 None
        """
        pass

    def health_check(self) -> bool:
        """健康检查：快速验证爬虫是否可用"""
        return True
