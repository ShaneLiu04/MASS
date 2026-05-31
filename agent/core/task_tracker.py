"""
MASS TaskTracker — 跨页面流式任务持久化
客户端断连后重连可恢复事件流

位置: agent/core/task_tracker.py (核心层)
说明: 本模块位于 agent/core/，可被 api/ 层调用，但绝不反向依赖 api/。
"""
import time
import threading
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque

from loguru import logger


@dataclass
class _TrackedTask:
    task_id: str
    stock_code: str
    task_type: str          # "diagnosis" | "prediction"
    status: str             # "running" | "completed" | "error"
    started_at: float
    finished_at: float = 0.0
    buffer: deque = field(default_factory=lambda: deque(maxlen=200))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, event: Dict[str, Any]) -> None:
        with self.lock:
            self.buffer.append(event)

    def get_buffered(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.buffer)

    def finish(self) -> None:
        self.status = "completed"
        self.finished_at = time.time()

    def error(self) -> None:
        self.status = "error"
        self.finished_at = time.time()

    def expired(self, ttl: int = 600) -> bool:
        if self.status == "running":
            return False
        return time.time() - self.finished_at > ttl


class TaskTracker:
    """
    跟踪所有进行中的诊断/预测任务。
    支持直接实例化（测试友好），也可通过 get_instance() 获取全局单例。
    线程安全，支持多客户端并发。
    """

    _instance: Optional["TaskTracker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._tasks: Dict[str, _TrackedTask] = {}
        self._by_stock: Dict[str, str] = {}  # stock_code → task_id (同一股票去重)
        self._mutex = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "TaskTracker":
        """获取全局单例（向后兼容）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ──

    def start_task(self, stock_code: str, task_type: str = "diagnosis") -> str:
        """注册新任务，返回 task_id。同股票已有活跃任务则返回已有 ID。"""
        with self._mutex:
            self._cleanup_expired()
            # 同一股票去重：返回已有任务
            existing = self._by_stock.get(stock_code)
            if existing and existing in self._tasks:
                task = self._tasks[existing]
                if task.status == "running":
                    logger.info(f"复用已有任务: {existing} (stock={stock_code})")
                    return existing

            task_id = f"{stock_code}_{int(time.time() * 1000)}"
            self._tasks[task_id] = _TrackedTask(
                task_id=task_id,
                stock_code=stock_code,
                task_type=task_type,
                status="running",
                started_at=time.time(),
            )
            self._by_stock[stock_code] = task_id
            logger.info(f"任务已注册: {task_id} type={task_type}")
            return task_id

    def get_task(self, task_id: str) -> Optional[_TrackedTask]:
        """获取任务状态，不存在或已过期返回 None"""
        with self._mutex:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.expired():
                self._remove_task(task_id)
                return None
            return task

    def push_event(self, task_id: str, event: Dict[str, Any]) -> None:
        task = self.get_task(task_id)
        if task:
            task.push(event)

    def finish_task(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.finish()
            logger.info(f"任务完成: {task_id}, buffer={len(task.buffer)} events")

    def error_task(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.error()
            logger.warning(f"任务出错: {task_id}")

    def cancel_task(self, task_id: str) -> bool:
        with self._mutex:
            task = self._tasks.pop(task_id, None)
            if task:
                self._by_stock.pop(task.stock_code, None)
                logger.info(f"任务已取消: {task_id}")
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        with self._mutex:
            running = sum(1 for t in self._tasks.values() if t.status == "running")
            return {
                "total_tasks": len(self._tasks),
                "running": running,
                "completed": len(self._tasks) - running,
            }

    # ── Internal ──

    def _remove_task(self, task_id: str) -> None:
        task = self._tasks.pop(task_id, None)
        if task:
            self._by_stock.pop(task.stock_code, None)

    def _cleanup_expired(self) -> None:
        expired = [tid for tid, t in self._tasks.items() if t.expired()]
        for tid in expired:
            self._remove_task(tid)


# 全局单例
task_tracker = TaskTracker()
