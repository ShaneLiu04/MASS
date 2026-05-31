"""
MASS 缓存层
支持内存缓存 + 可选Redis扩展

设计:
- 使用 OrderedDict 实现真 LRU（O(1) 访问、插入、淘汰）
- 支持 TTL 过期清理
- 线程安全（RLock）
- 全局单例 + 直接实例化双模式
"""
import time
import threading
from collections import OrderedDict
from typing import Any, Optional, Callable
from dataclasses import dataclass
from functools import wraps

from loguru import logger

from config import CACHE_TTL_SECONDS


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    expire_at: float
    created_at: float


class CacheManager:
    """
    线程安全的内存缓存管理器 — 真 LRU 实现。

    特性:
    - OrderedDict 实现 O(1) 访问、插入、淘汰
    - 访问时自动将条目移至末尾（LRU）
    - 写入时自动淘汰最旧条目（超出 max_size）
    - 惰性清理过期数据，避免集中式全量扫描

    支持直接实例化（测试友好），也可通过 get_instance() 获取全局单例。
    生产环境可替换为 Redis 后端。
    """

    _instance: Optional["CacheManager"] = None
    _lock = threading.Lock()

    def __init__(self, default_ttl: int = CACHE_TTL_SECONDS, max_size: int = 1000):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._mutex = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0  # 淘汰计数
        self._expired_cleaned = 0  # 过期清理计数

    @classmethod
    def get_instance(cls) -> "CacheManager":
        """获取全局单例（向后兼容）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值。命中时将该条目移至末尾（LRU）。"""
        with self._mutex:
            entry = self._store.get(key)

            if entry is None:
                self._misses += 1
                return None

            now = time.time()
            if now > entry.expire_at:
                # 惰性删除过期条目
                del self._store[key]
                self._misses += 1
                self._expired_cleaned += 1
                return None

            # LRU: 命中后移至末尾（最新）
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值。

        - 已存在则更新并移至末尾
        - 超出 max_size 时 O(1) 淘汰最旧条目
        - 写入时顺带清理少量过期条目（惰性清理）
        """
        ttl = ttl or self._default_ttl
        with self._mutex:
            now = time.time()

            # 如果 key 已存在，更新并移至末尾
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = CacheEntry(
                    value=value,
                    expire_at=now + ttl,
                    created_at=now,
                )
                return

            # 超出容量限制：淘汰最旧的条目
            while len(self._store) >= self._max_size:
                # popitem(last=False) 从头部弹出最旧的（LRU 或已过期）
                evicted_key, evicted_entry = self._store.popitem(last=False)
                self._evictions += 1
                if evicted_entry.expire_at < now:
                    self._expired_cleaned += 1
                logger.debug(f"缓存淘汰: {evicted_key} (LRU)")

            self._store[key] = CacheEntry(
                value=value,
                expire_at=now + ttl,
                created_at=now,
            )

    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._mutex:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._mutex:
            self._store.clear()
            logger.info("缓存已清空")

    def exists(self, key: str) -> bool:
        """检查键是否存在（不更新 LRU）"""
        with self._mutex:
            entry = self._store.get(key)
            if entry is None:
                return False
            if time.time() > entry.expire_at:
                del self._store[key]
                return False
            return True

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._mutex:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate * 100, 2),
                "total_requests": total,
                "evictions": self._evictions,
                "expired_cleaned": self._expired_cleaned,
            }

    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        装饰器：自动缓存函数结果

        Usage:
            @cache.cached(ttl=300, key_prefix="stock")
            def get_stock_data(code):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

                # 尝试读取缓存
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # 执行函数
                result = func(*args, **kwargs)

                # 写入缓存
                self.set(cache_key, result, ttl)
                return result

            # 附加清除缓存方法
            wrapper.cache_clear = lambda: self.delete(
                f"{key_prefix}:{func.__name__}"
            )
            return wrapper
        return decorator


# 全局缓存实例
cache = CacheManager()


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """快捷装饰器"""
    return cache.cached(ttl=ttl, key_prefix=key_prefix)
