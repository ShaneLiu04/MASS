"""
MASS 爬虫断路器 (Circuit Breaker)

防止故障爬虫源持续占用线程池资源：
- CLOSED: 正常状态，请求通过
- OPEN: 连续失败达到阈值，请求直接拒绝（快速失败）
- HALF_OPEN: 经过恢复期后，允许一次探测请求
"""
import time
import threading
from typing import Optional
from enum import Enum, auto

from loguru import logger


class CircuitState(Enum):
    CLOSED = auto()      # 正常
    OPEN = auto()        # 熔断中，快速失败
    HALF_OPEN = auto()   # 半开，允许探测


class CircuitBreaker:
    """
    爬虫源断路器
    
    用法:
        cb = CircuitBreaker(name="eastmoney", failure_threshold=5, recovery_timeout=60)
        if not cb.can_execute():
            return None  # 快速失败
        try:
            result = fetch()
            cb.record_success()
            return result
        except Exception:
            cb.record_failure()
            raise
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def can_execute(self) -> bool:
        """检查当前是否允许执行请求"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # 检查是否已过恢复期
                if self._last_failure_time and (
                    time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"[CB:{self.name}] 进入 HALF_OPEN 状态，允许探测请求")
                    return True
                logger.debug(f"[CB:{self.name}] OPEN 状态，快速失败")
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                logger.debug(f"[CB:{self.name}] HALF_OPEN 探测次数已满")
                return False

            return True

    def record_success(self) -> None:
        """记录成功"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._successes += 1
                # 连续成功恢复 CLOSED
                if self._successes >= 1:
                    self._transition_to(CircuitState.CLOSED)
            else:
                self._failures = max(0, self._failures - 1)
                self._successes += 1

    def record_failure(self) -> None:
        """记录失败"""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            self._successes = 0

            if self._state == CircuitState.HALF_OPEN:
                # 探测失败，重新 OPEN
                self._transition_to(CircuitState.OPEN)
            elif self._failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换"""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            if new_state == CircuitState.OPEN:
                logger.warning(
                    f"[CB:{self.name}] {old_state.name} → OPEN "
                    f"(连续失败 {self._failures} 次，暂停 {self.recovery_timeout}s)"
                )
            elif new_state == CircuitState.CLOSED:
                logger.info(f"[CB:{self.name}] {old_state.name} → CLOSED (恢复正常)")
                self._failures = 0
                self._half_open_calls = 0

    def get_stats(self) -> dict:
        """获取断路器统计"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failures": self._failures,
                "successes": self._successes,
                "last_failure": self._last_failure_time,
            }
