"""
MASS 自适应并发控制器 (Adaptive Concurrency Controller)
v2.3 新增：
- 基于响应时间与错误率的动态并发限制调整
- 滑动窗口统计，避免瞬时抖动
- 队列深度感知，预防性降级
- 渐进式恢复（快速降级、缓慢升级）
"""
import threading
import time
from collections import deque
from typing import Dict, Any, Optional

from loguru import logger


class AdaptiveConcurrencyController:
    """
    自适应并发控制器

    核心机制：
    1. 滑动窗口记录最近 N 次请求的响应时间和错误状态
    2. 定期（每次记录后）评估系统健康度
    3. 根据健康度动态调整并发限制

    调整策略：
    - 快速降级：响应时间 > 慢阈值 或 错误率 > 高阈值 → 立即减少并发
    - 缓慢升级：响应时间 < 快阈值 且 错误率 < 低阈值 → 逐步增加并发
    - 冷却期：每次调整后 5 秒内不再调整，避免抖动

    线程安全：所有状态变更受 _lock 保护
    """

    def __init__(
        self,
        min_concurrent: int = 2,
        max_concurrent: int = 10,
        slow_threshold: float = 30.0,
        fast_threshold: float = 10.0,
        error_high: float = 0.20,
        error_low: float = 0.05,
        window_size: int = 50,
        cooldown_seconds: float = 5.0,
    ):
        """
        Args:
            min_concurrent: 最小并发限制
            max_concurrent: 最大并发限制
            slow_threshold: 平均响应时间超过此值触发降级（秒）
            fast_threshold: 平均响应时间低于此值才允许升级（秒）
            error_high: 错误率超过此值触发大幅降级
            error_low: 错误率低于此值才允许升级
            window_size: 滑动窗口大小
            cooldown_seconds: 调整冷却期（秒）
        """
        self.min_concurrent = max(1, min_concurrent)
        self.max_concurrent = max(self.min_concurrent, max_concurrent)
        self.slow_threshold = slow_threshold
        self.fast_threshold = fast_threshold
        self.error_high = error_high
        self.error_low = error_low
        self.cooldown_seconds = cooldown_seconds

        # 滑动窗口
        self._response_times: deque = deque(maxlen=window_size)
        self._error_flags: deque = deque(maxlen=window_size)
        self._wait_times: deque = deque(maxlen=window_size)  # 信号量等待时间

        # 当前限制
        self._current_limit = self.min_concurrent

        # 冷却期
        self._last_adjustment_time = 0.0

        # 统计指标
        self._total_requests = 0
        self._total_success = 0
        self._total_errors = 0

        self._lock = threading.Lock()

        logger.info(
            f"AdaptiveConcurrencyController 初始化: "
            f"min={self.min_concurrent}, max={self.max_concurrent}, "
            f"slow_threshold={slow_threshold}s, fast_threshold={fast_threshold}s"
        )

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        尝试获取并发许可（信号量语义）

        使用内部计数器实现信号量，支持动态调整限制。

        Args:
            timeout: 最大等待时间（秒）

        Returns:
            bool: 是否成功获取
        """
        deadline = time.time() + timeout
        wait_start = time.time()

        while True:
            with self._lock:
                # 检查是否有可用许可
                if self._current_limit > 0:
                    self._current_limit -= 1
                    return True

            # 计算剩余等待时间
            remaining = deadline - time.time()
            if remaining <= 0:
                # 记录等待时间（即使失败）
                self._record_wait_time(time.time() - wait_start)
                return False

            # 短暂休眠后重试
            time.sleep(min(0.1, remaining))

    def release(self) -> None:
        """释放并发许可"""
        with self._lock:
            # 释放后不超过 max_concurrent
            self._current_limit = min(self._current_limit + 1, self.max_concurrent)

    def record_result(
        self,
        duration: float,
        success: bool,
        wait_time: float = 0.0,
    ) -> None:
        """
        记录请求结果，并触发自适应调整

        Args:
            duration: 请求处理时间（秒）
            success: 是否成功
            wait_time: 信号量等待时间（秒）
        """
        with self._lock:
            self._response_times.append(duration)
            self._error_flags.append(0 if success else 1)
            if wait_time > 0:
                self._wait_times.append(wait_time)

            self._total_requests += 1
            if success:
                self._total_success += 1
            else:
                self._total_errors += 1

        # 触发调整（在锁外执行，避免阻塞）
        self._adjust_limit()

    def _record_wait_time(self, wait_time: float) -> None:
        """记录等待时间（用于超时场景）"""
        with self._lock:
            self._wait_times.append(wait_time)

    def _adjust_limit(self) -> None:
        """
        根据历史表现调整并发限制

        调整规则：
        1. 窗口数据不足（<10条）→ 不调整
        2. 冷却期内 → 不调整
        3. 错误率高（>error_high）→ 大幅降级（-2）
        4. 响应慢（>slow_threshold）→ 降级（-1）
        5. 响应快（<fast_threshold）且错误率低（<error_low）→ 升级（+1）
        6. 队列等待时间长（>5s）→ 降级（-1）
        """
        with self._lock:
            # 冷却期检查
            now = time.time()
            if now - self._last_adjustment_time < self.cooldown_seconds:
                return

            # 窗口数据不足
            if len(self._response_times) < 10:
                return

            avg_time = sum(self._response_times) / len(self._response_times)
            error_rate = sum(self._error_flags) / len(self._error_flags)
            avg_wait = sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0

            old_limit = self._current_limit
            new_limit = old_limit

            # 优先级1: 错误率高 → 大幅降级
            if error_rate > self.error_high:
                new_limit = max(self.min_concurrent, old_limit - 2)
                logger.warning(
                    f"[并发控制] 错误率过高({error_rate:.1%})，并发限制 {old_limit} → {new_limit}"
                )

            # 优先级2: 响应慢 → 降级
            elif avg_time > self.slow_threshold:
                new_limit = max(self.min_concurrent, old_limit - 1)
                logger.info(
                    f"[并发控制] 响应缓慢({avg_time:.1f}s)，并发限制 {old_limit} → {new_limit}"
                )

            # 优先级3: 队列等待长 → 降级
            elif avg_wait > 5.0:
                new_limit = max(self.min_concurrent, old_limit - 1)
                logger.info(
                    f"[并发控制] 队列等待长({avg_wait:.1f}s)，并发限制 {old_limit} → {new_limit}"
                )

            # 优先级4: 响应快且错误率低 → 升级
            elif avg_time < self.fast_threshold and error_rate < self.error_low:
                new_limit = min(self.max_concurrent, old_limit + 1)
                if new_limit > old_limit:
                    logger.info(
                        f"[并发控制] 系统健康({avg_time:.1f}s, 错误率{error_rate:.1%})，"
                        f"并发限制 {old_limit} → {new_limit}"
                    )

            if new_limit != old_limit:
                self._current_limit = new_limit
                self._last_adjustment_time = now

    def get_limit(self) -> int:
        """获取当前并发限制"""
        with self._lock:
            return self._current_limit

    def get_stats(self) -> Dict[str, Any]:
        """
        获取控制器统计信息（用于监控/调试）

        Returns:
            Dict with keys: current_limit, min, max, avg_response_time,
                           error_rate, avg_wait_time, total_requests,
                           total_success, total_errors, window_size
        """
        with self._lock:
            avg_time = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times else 0.0
            )
            error_rate = (
                sum(self._error_flags) / len(self._error_flags)
                if self._error_flags else 0.0
            )
            avg_wait = (
                sum(self._wait_times) / len(self._wait_times)
                if self._wait_times else 0.0
            )
            return {
                "current_limit": self._current_limit,
                "min_concurrent": self.min_concurrent,
                "max_concurrent": self.max_concurrent,
                "avg_response_time": round(avg_time, 2),
                "error_rate": round(error_rate, 4),
                "avg_wait_time": round(avg_wait, 2),
                "total_requests": self._total_requests,
                "total_success": self._total_success,
                "total_errors": self._total_errors,
                "window_size": len(self._response_times),
                "cooldown_seconds": self.cooldown_seconds,
                "last_adjustment_ago": round(time.time() - self._last_adjustment_time, 1),
            }

    def reset(self) -> None:
        """重置控制器状态（用于测试或手动恢复）"""
        with self._lock:
            self._response_times.clear()
            self._error_flags.clear()
            self._wait_times.clear()
            self._current_limit = self.min_concurrent
            self._last_adjustment_time = 0.0
            self._total_requests = 0
            self._total_success = 0
            self._total_errors = 0
            logger.info("[并发控制] 状态已重置")
