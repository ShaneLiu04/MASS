"""
MASS Flask 中间件
请求日志、错误处理、限流、CORS
"""
import os
import time
import traceback
from typing import Dict, Any, List
from functools import wraps

from flask import request, jsonify, current_app
from loguru import logger

# ── CORS 白名单 ──
# 支持逗号分隔多个来源，生产环境严禁使用通配符 '*'
_ALLOWED_ORIGINS: List[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",")
    if origin.strip()
]


class RequestLogger:
    """请求日志中间件"""
    
    @staticmethod
    def init_app(app):
        @app.before_request
        def before_request():
            request.start_time = time.time()
            request.request_id = f"{time.time():.6f}"
        
        @app.after_request
        def after_request(response):
            duration = time.time() - getattr(request, 'start_time', time.time())
            status = response.status_code
            method = request.method
            path = request.path
            
            level = "INFO" if status < 400 else "WARNING"
            logger.log(
                level,
                f"[{request.request_id}] {method} {path} - {status} - {duration:.3f}s"
            )
            
            # 添加响应头
            response.headers['X-Request-ID'] = request.request_id
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
            return response


class ErrorHandler:
    """统一错误处理"""
    
    @staticmethod
    def init_app(app):
        @app.errorhandler(400)
        def bad_request(e):
            return jsonify({
                "error": "Bad Request",
                "message": str(e.description if hasattr(e, 'description') else e),
                "request_id": getattr(request, 'request_id', None),
            }), 400
        
        @app.errorhandler(404)
        def not_found(e):
            return jsonify({
                "error": "Not Found",
                "message": f"接口 {request.path} 不存在",
                "request_id": getattr(request, 'request_id', None),
            }), 404
        
        @app.errorhandler(405)
        def method_not_allowed(e):
            return jsonify({
                "error": "Method Not Allowed",
                "message": f"{request.method} 方法不允许",
                "request_id": getattr(request, 'request_id', None),
            }), 405
        
        @app.errorhandler(429)
        def rate_limit_exceeded(e):
            return jsonify({
                "error": "Rate Limit Exceeded",
                "message": "请求过于频繁，请稍后重试",
                "request_id": getattr(request, 'request_id', None),
            }), 429
        
        @app.errorhandler(500)
        def internal_error(e):
            logger.exception("服务器内部错误")
            return jsonify({
                "error": "Internal Server Error",
                "message": "服务器内部错误，请稍后重试" if not app.debug else str(e),
                "request_id": getattr(request, 'request_id', None),
            }), 500
        
        @app.errorhandler(Exception)
        def unhandled_exception(e):
            logger.exception("未处理的异常")
            return jsonify({
                "error": "Internal Server Error",
                "message": "服务器发生未知错误" if not app.debug else str(e),
                "request_id": getattr(request, 'request_id', None),
            }), 500


import threading
from collections import deque

class RateLimiter:
    """
    线程安全内存限流器 — O(1) 滑动窗口 + 定期过期 key 清理
    使用 deque 实现高效的时间戳管理
    """

    def __init__(self, default_limit: int = 60, window: int = 60):
        self.default_limit = default_limit
        self.window = window
        self._requests: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._access_count = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 1000  # 每 1000 次请求或每 window 秒清理一次过期 key

    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求 — 线程安全 O(1)，即时 + 定期双重清理过期 key"""
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            # ── 全局清理：计数触发 OR 时间触发（低流量兜底）──
            self._access_count += 1
            if (self._access_count % self._cleanup_interval == 0
                    or now - self._last_cleanup > self.window):
                stale = [
                    k for k, q in self._requests.items()
                    if not q or q[-1] < cutoff
                ]
                for k in stale:
                    del self._requests[k]
                self._last_cleanup = now

            if key not in self._requests:
                self._requests[key] = deque()

            q = self._requests[key]
            while q and q[0] < cutoff:
                q.popleft()

            # ── 即时清理：当前 key 的记录全部过期，替换为新记录 ──
            if not q:
                self._requests[key] = deque([now])
                return True

            if len(q) >= self.default_limit:
                return False

            q.append(now)
            return True
    
    def limit(self, max_requests: int = 60, window: int = 60):
        """装饰器：限流 — 支持 X-Forwarded-For 穿透反向代理"""
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                client_ip = _get_client_ip()
                client_id = request.headers.get('X-API-Key', client_ip)
                key = f"{f.__name__}:{client_id}"
                
                limiter = getattr(current_app, '_rate_limiter', None)
                if limiter is None:
                    limiter = RateLimiter(max_requests, window)
                    current_app._rate_limiter = limiter
                
                if not limiter.is_allowed(key):
                    return jsonify({
                        "error": "Rate Limit Exceeded",
                        "message": f"该接口限制 {max_requests} 次/{window}秒",
                    }), 429
                
                return f(*args, **kwargs)
            return wrapped
        return decorator


def _get_client_ip() -> str:
    """
    获取真实客户端 IP，优先读取 X-Forwarded-For（支持反向代理穿透）。
    若存在多个代理地址，取最左侧（最接近客户端）的 IP。
    """
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        # X-Forwarded-For: client, proxy1, proxy2
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


class CORSHeaders:
    """CORS 跨域支持 — 基于白名单的 Origin 校验，禁止通配符 '*'"""

    @staticmethod
    def init_app(app):
        @app.after_request
        def add_cors_headers(response):
            origin = request.headers.get('Origin', '')
            # 仅对白名单内的 Origin 放行；不在白名单的不设置该头，浏览器会阻止跨域
            if origin in _ALLOWED_ORIGINS:
                response.headers['Access-Control-Allow-Origin'] = origin
                # 告知缓存机制：响应内容随 Origin 变化，避免缓存污染
                response.headers['Vary'] = 'Origin'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key'
            response.headers['Access-Control-Max-Age'] = '86400'
            return response

        @app.before_request
        def handle_options():
            if request.method == 'OPTIONS':
                return '', 200


def register_middleware(app):
    """注册所有中间件"""
    RequestLogger.init_app(app)
    ErrorHandler.init_app(app)
    CORSHeaders.init_app(app)
    logger.info("中间件注册完成")
