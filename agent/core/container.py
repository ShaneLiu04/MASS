"""
MASS 依赖注入容器 (DI Container)

解决单例模式泛滥导致的测试困难问题：
- 生产环境：通过容器获取全局单例
- 测试环境：可注入 Mock 替换任意组件

用法：
    # 生产环境
    container = get_container()
    orchestrator = AgentOrchestrator(container=container)

    # 测试环境
    container = Container()
    container.register("blackboard", MockBlackboard())
    orchestrator = AgentOrchestrator(container=container)
"""
from typing import Dict, Any, Optional

from loguru import logger


class Container:
    """
    MASS 依赖注入容器
    
    每个组件延迟初始化，但可通过 register() 在测试时注入 Mock。
    """

    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {
            "blackboard": self._make_blackboard,
            "cache": self._make_cache,
            "crawler_registry": self._make_crawler_registry,
            "task_tracker": self._make_task_tracker,
            "stock_data_tool": self._make_stock_data_tool,
        }

    # ── 核心组件工厂 ──

    def _make_blackboard(self):
        from agent.core.blackboard import get_blackboard
        return get_blackboard()

    def _make_cache(self):
        from agent.core.cache import CacheManager
        return CacheManager.get_instance()

    def _make_crawler_registry(self):
        from agent.crawlers.registry import CrawlerRegistry
        return CrawlerRegistry.get_instance()

    def _make_task_tracker(self):
        from agent.core.task_tracker import TaskTracker
        return TaskTracker.get_instance()

    def _make_stock_data_tool(self):
        from agent.tools.stock_data_tool import StockDataTool
        return StockDataTool.get_instance()

    # ── Public API ──

    def get(self, name: str) -> Any:
        """获取组件实例。优先返回已注册的，否则延迟初始化。"""
        if name in self._registry:
            return self._registry[name]
        if name in self._factories:
            instance = self._factories[name]()
            self._registry[name] = instance
            return instance
        raise KeyError(f"未注册的组件: {name}")

    def register(self, name: str, instance: Any) -> "Container":
        """注册/覆盖组件实例（用于测试注入 Mock）。返回 self 支持链式调用。"""
        self._registry[name] = instance
        logger.debug(f"Container: 组件 '{name}' 已被注册/覆盖")
        return self

    def has(self, name: str) -> bool:
        """检查组件是否已注册或可被创建"""
        return name in self._registry or name in self._factories

    def reset(self, name: Optional[str] = None) -> None:
        """重置组件注册。name=None 时重置全部。"""
        if name is None:
            self._registry.clear()
            logger.debug("Container: 所有组件注册已重置")
        else:
            self._registry.pop(name, None)
            logger.debug(f"Container: 组件 '{name}' 注册已重置")

    # ── 快捷属性 ──

    @property
    def blackboard(self):
        return self.get("blackboard")

    @property
    def cache(self):
        return self.get("cache")

    @property
    def crawler_registry(self):
        return self.get("crawler_registry")

    @property
    def task_tracker(self):
        return self.get("task_tracker")

    @property
    def stock_data_tool(self):
        return self.get("stock_data_tool")


# 全局默认容器（向后兼容）
_default_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局默认容器"""
    global _default_container
    if _default_container is None:
        _default_container = Container()
    return _default_container


def set_container(container: Container) -> None:
    """设置全局默认容器（用于测试）"""
    global _default_container
    _default_container = container
    logger.debug("全局默认 Container 已被替换")
