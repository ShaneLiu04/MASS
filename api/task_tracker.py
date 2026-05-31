"""
MASS TaskTracker — API 层代理导出

核心实现已下放到 agent/core/task_tracker.py，本文件保留以维持向后兼容。
api/ 层代码可直接从 api.task_tracker 导入，也可从 agent.core.task_tracker 导入。
"""
from agent.core.task_tracker import TaskTracker, task_tracker

__all__ = ["TaskTracker", "task_tracker"]
