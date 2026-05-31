"""
MASS 共享黑板系统 (Shared Blackboard)
支持内存后端（默认）与 Redis 后端（分布式部署）

架构：
- AbstractBlackboard: 定义统一接口
- Blackboard: 内存实现，线程安全，按股票分片锁
- RedisBlackboard: Redis 实现，支持分布式、TTL、无容量上限
- get_blackboard(): 工厂函数，根据配置自动选择后端
"""
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol

import orjson
import pandas as pd
from loguru import logger


# ── 序列化辅助 ──

def _snapshot_to_dict(snapshot: "StockSnapshot") -> Dict[str, Any]:
    """将 StockSnapshot 转为可 JSON 序列化的 dict（清理 numpy 类型）"""
    import numpy as np

    def _sanitize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    kline_records = None
    if snapshot.kline_df is not None and not snapshot.kline_df.empty:
        kline_records = _sanitize(snapshot.kline_df.to_dict(orient="records"))

    return {
        "stock_code": snapshot.stock_code,
        "stock_name": snapshot.stock_name,
        "current_price": float(snapshot.current_price),
        "kline_records": kline_records,
        "indicators": _sanitize(snapshot.indicators),
        "fundamentals": _sanitize(snapshot.fundamentals),
        "fund_flow": _sanitize(snapshot.fund_flow),
        "market_context": _sanitize(snapshot.market_context),
        "sentiment_data": _sanitize(snapshot.sentiment_data),
        "macro_data": _sanitize(snapshot.macro_data),
        "risk_metrics": _sanitize(snapshot.risk_metrics),
        "data_quality": _sanitize(snapshot.data_quality),
        "timestamp": snapshot.timestamp.isoformat(),
    }


def _snapshot_from_dict(data: Dict[str, Any]) -> "StockSnapshot":
    """从 dict 恢复 StockSnapshot"""
    kline_records = data.pop("kline_records", None)
    kline_df = pd.DataFrame(kline_records) if kline_records else None
    ts_str = data.pop("timestamp", None)
    timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
    return StockSnapshot(kline_df=kline_df, timestamp=timestamp, **data)


def _opinion_to_dict(opinion: "AgentOpinion") -> Dict[str, Any]:
    """AgentOpinion → dict（复用已有的 to_dict）"""
    return opinion.to_dict()


def _opinion_from_dict(data: Dict[str, Any]) -> "AgentOpinion":
    """dict → AgentOpinion"""
    ts = data.pop("timestamp", None)
    timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.now()
    return AgentOpinion(timestamp=timestamp, **data)


# ── 数据模型 ──

@dataclass
class StockSnapshot:
    """共享数据快照 - 所有Agent分析的基础数据"""
    stock_code: str
    stock_name: str
    current_price: float = 0.0
    kline_df: Optional[Any] = None          # pandas DataFrame (OHLCV)
    indicators: Dict[str, Any] = field(default_factory=dict)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    fund_flow: Dict[str, Any] = field(default_factory=dict)
    market_context: Dict[str, Any] = field(default_factory=dict)
    sentiment_data: Dict[str, Any] = field(default_factory=dict)
    macro_data: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    _prompt_context_cache: Optional[str] = field(default=None, repr=False, compare=False)

    def to_summary(self) -> Dict[str, Any]:
        """生成数据摘要（用于返回给前端）"""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "current_price": self.current_price,
            "industry": self.fundamentals.get("industry", ""),
            "indicator_names": list(self.indicators.keys()),
            "fundamental_keys": list(self.fundamentals.keys()),
            "fund_flow_keys": list(self.fund_flow.keys()),
            "market_context_keys": list(self.market_context.keys()),
            "sentiment_keys": list(self.sentiment_data.keys()),
            "macro_keys": list(self.macro_data.keys()),
            "risk_keys": list(self.risk_metrics.keys()),
            "timestamp": self.timestamp.isoformat(),
        }

    def to_prompt_context(self) -> str:
        """将快照转为LLM Prompt可用的文本描述（惰性缓存）"""
        if self._prompt_context_cache is not None:
            return self._prompt_context_cache

        def _sanitize(obj):
            """递归清理 numpy/pandas 类型为 Python 原生类型"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        _dump = lambda data: orjson.dumps(_sanitize(data), option=orjson.OPT_INDENT_2).decode()

        lines = [
            f"股票代码: {self.stock_code}",
            f"股票名称: {self.stock_name}",
            f"当前价格: {self.current_price}",
            "",
            "=== 技术指标 ===",
            _dump(self.indicators),
            "",
            "=== 基本面数据 ===",
            _dump(self.fundamentals),
            "",
            "=== 资金流向 ===",
            _dump(self.fund_flow),
            "",
            "=== 市场情绪 ===",
            _dump(self.sentiment_data),
            "",
            "=== 宏观环境 ===",
            _dump(self.macro_data),
            "",
            "=== 风险指标 ===",
            _dump(self.risk_metrics),
        ]
        self._prompt_context_cache = "\n".join(lines)
        return self._prompt_context_cache


@dataclass
class AgentOpinion:
    """单个Agent的观点"""
    agent_id: str
    signal: int                     # -1(卖出), 0(观望), 1(买入)
    confidence: float               # 0.0 ~ 1.0
    reasoning: str                  # 自然语言推理
    key_factors: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    is_revision: bool = False
    original_signal: Optional[int] = None
    revision_round: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_flags": self.risk_flags,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp.isoformat(),
            "is_revision": self.is_revision,
            "original_signal": self.original_signal,
            "revision_round": self.revision_round,
        }


# ── 抽象接口 ──

class AbstractBlackboard(ABC):
    """黑板抽象接口 — 支持内存与 Redis 两种后端"""

    @abstractmethod
    def publish_snapshot(self, snapshot: StockSnapshot) -> None:
        """发布新的数据快照（会清空之前的观点）"""
        ...

    @abstractmethod
    def submit_opinion(self, stock_code: str, opinion: AgentOpinion) -> None:
        """提交Agent观点"""
        ...

    @abstractmethod
    def get_snapshot(self, stock_code: str) -> Optional[StockSnapshot]:
        """获取数据快照"""
        ...

    @abstractmethod
    def get_opinions(self, stock_code: str) -> List[AgentOpinion]:
        """获取某股票的所有Agent观点"""
        ...

    @abstractmethod
    def get_opinion_by_agent(self, stock_code: str, agent_id: str) -> Optional[AgentOpinion]:
        """获取特定Agent的观点"""
        ...

    @abstractmethod
    def clear_stock(self, stock_code: str) -> None:
        """清理某股票的所有数据"""
        ...

    @abstractmethod
    def get_all_stock_codes(self) -> List[str]:
        """获取所有有快照的股票代码"""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取黑板统计信息"""
        ...


# ── 内存实现 ──

class Blackboard(AbstractBlackboard):
    """
    内存黑板 — 线程安全，按股票分片锁
    适合单节点部署；200 只股票上限防止内存无限增长。
    """
    _instance: Optional["Blackboard"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._snapshots: Dict[str, StockSnapshot] = {}
        self._opinions: Dict[str, List[AgentOpinion]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_stocks = 200

    @classmethod
    def get_instance(cls) -> "Blackboard":
        """获取全局单例（向后兼容）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_lock(self, stock_code: str) -> threading.Lock:
        with self._global_lock:
            if stock_code not in self._locks:
                self._locks[stock_code] = threading.Lock()
            return self._locks[stock_code]

    def _maybe_evict(self) -> None:
        with self._global_lock:
            if len(self._snapshots) <= self._max_stocks:
                return
            sorted_codes = sorted(
                self._snapshots.keys(),
                key=lambda c: self._snapshots[c].timestamp
            )
            evict_count = len(sorted_codes) // 5
            for code in sorted_codes[:evict_count]:
                self._snapshots.pop(code, None)
                self._opinions.pop(code, None)
                self._locks.pop(code, None)

    # ── AbstractBlackboard 接口实现 ──

    def publish_snapshot(self, snapshot: StockSnapshot) -> None:
        code = snapshot.stock_code
        lock = self._get_lock(code)
        with lock:
            with self._global_lock:
                self._snapshots[code] = snapshot
                self._opinions[code] = []
        self._maybe_evict()

    def submit_opinion(self, stock_code: str, opinion: AgentOpinion) -> None:
        lock = self._get_lock(stock_code)
        with lock:
            with self._global_lock:
                if stock_code not in self._opinions:
                    self._opinions[stock_code] = []
                self._opinions[stock_code].append(opinion)

    def get_snapshot(self, stock_code: str) -> Optional[StockSnapshot]:
        lock = self._get_lock(stock_code)
        with lock:
            with self._global_lock:
                return self._snapshots.get(stock_code)

    def get_opinions(self, stock_code: str) -> List[AgentOpinion]:
        lock = self._get_lock(stock_code)
        with lock:
            with self._global_lock:
                return list(self._opinions.get(stock_code, []))

    def get_opinion_by_agent(self, stock_code: str, agent_id: str) -> Optional[AgentOpinion]:
        lock = self._get_lock(stock_code)
        with lock:
            with self._global_lock:
                opinions = self._opinions.get(stock_code, [])
                for op in opinions:
                    if op.agent_id == agent_id:
                        return op
                return None

    def clear_stock(self, stock_code: str) -> None:
        lock = self._get_lock(stock_code)
        with lock:
            with self._global_lock:
                self._snapshots.pop(stock_code, None)
                self._opinions.pop(stock_code, None)
                self._locks.pop(stock_code, None)

    def get_all_stock_codes(self) -> List[str]:
        with self._global_lock:
            return list(self._snapshots.keys())

    def get_stats(self) -> Dict[str, Any]:
        with self._global_lock:
            return {
                "backend": "memory",
                "total_snapshots": len(self._snapshots),
                "total_opinions": sum(len(ops) for ops in self._opinions.values()),
                "stocks": list(self._snapshots.keys()),
            }


# ── Redis 实现 ──

class RedisBlackboard(AbstractBlackboard):
    """
    Redis 分布式黑板

    数据结构：
    - mass:snapshot:{stock_code}  → String (orjson bytes), TTL=3600s
    - mass:opinions:{stock_code}  → List  (JSON strings), TTL=3600s
    - mass:stocks                 → Set   (所有有快照的股票代码)

    特点：
    - 无 200 只上限限制
    - 支持多实例共享状态（分布式部署）
    - 自动 TTL 过期，防止僵尸数据
    - Redis 不可用时降级到内存黑板
    """

    _SNAPSHOT_KEY = "mass:snapshot:{}"
    _OPINIONS_KEY = "mass:opinions:{}"
    _STOCKS_KEY = "mass:stocks"
    _DEFAULT_TTL = 3600

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Any = None
        self._fallback: Optional[Blackboard] = None
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis
            self._redis = redis.from_url(self._redis_url, decode_responses=False)
            # 健康检查
            self._redis.ping()
            logger.info(f"Redis 黑板初始化成功: {self._redis_url}")
        except Exception as e:
            logger.warning(f"Redis 连接失败: {e}，回退到内存黑板")
            self._redis = None
            self._fallback = Blackboard()

    def _using_redis(self) -> bool:
        return self._redis is not None

    # ── AbstractBlackboard 接口实现 ──

    def publish_snapshot(self, snapshot: StockSnapshot) -> None:
        if not self._using_redis():
            self._fallback.publish_snapshot(snapshot)
            return

        code = snapshot.stock_code
        key = self._SNAPSHOT_KEY.format(code)
        data = orjson.dumps(_snapshot_to_dict(snapshot))
        pipe = self._redis.pipeline()
        pipe.setex(key, self._DEFAULT_TTL, data)
        pipe.delete(self._OPINIONS_KEY.format(code))
        pipe.sadd(self._STOCKS_KEY, code)
        pipe.execute()

    def submit_opinion(self, stock_code: str, opinion: AgentOpinion) -> None:
        if not self._using_redis():
            self._fallback.submit_opinion(stock_code, opinion)
            return

        key = self._OPINIONS_KEY.format(stock_code)
        data = orjson.dumps(_opinion_to_dict(opinion))
        pipe = self._redis.pipeline()
        pipe.lpush(key, data)
        pipe.expire(key, self._DEFAULT_TTL)
        pipe.execute()

    def get_snapshot(self, stock_code: str) -> Optional[StockSnapshot]:
        if not self._using_redis():
            return self._fallback.get_snapshot(stock_code)

        key = self._SNAPSHOT_KEY.format(stock_code)
        raw = self._redis.get(key)
        if raw is None:
            return None
        return _snapshot_from_dict(orjson.loads(raw))

    def get_opinions(self, stock_code: str) -> List[AgentOpinion]:
        if not self._using_redis():
            return self._fallback.get_opinions(stock_code)

        key = self._OPINIONS_KEY.format(stock_code)
        raw_list = self._redis.lrange(key, 0, -1)
        if not raw_list:
            return []
        return [_opinion_from_dict(orjson.loads(item)) for item in raw_list]

    def get_opinion_by_agent(self, stock_code: str, agent_id: str) -> Optional[AgentOpinion]:
        if not self._using_redis():
            return self._fallback.get_opinion_by_agent(stock_code, agent_id)

        opinions = self.get_opinions(stock_code)
        for op in opinions:
            if op.agent_id == agent_id:
                return op
        return None

    def clear_stock(self, stock_code: str) -> None:
        if not self._using_redis():
            self._fallback.clear_stock(stock_code)
            return

        pipe = self._redis.pipeline()
        pipe.delete(self._SNAPSHOT_KEY.format(stock_code))
        pipe.delete(self._OPINIONS_KEY.format(stock_code))
        pipe.srem(self._STOCKS_KEY, stock_code)
        pipe.execute()

    def get_all_stock_codes(self) -> List[str]:
        if not self._using_redis():
            return self._fallback.get_all_stock_codes()

        codes = self._redis.smembers(self._STOCKS_KEY)
        return [c.decode() if isinstance(c, bytes) else c for c in codes]

    def get_stats(self) -> Dict[str, Any]:
        if not self._using_redis():
            return self._fallback.get_stats()

        stocks = self.get_all_stock_codes()
        pipe = self._redis.pipeline()
        for code in stocks:
            pipe.exists(self._SNAPSHOT_KEY.format(code))
            pipe.llen(self._OPINIONS_KEY.format(code))
        results = pipe.execute()

        total_snapshots = 0
        total_opinions = 0
        for i in range(0, len(results), 2):
            if results[i]:
                total_snapshots += 1
            total_opinions += results[i + 1]

        return {
            "backend": "redis",
            "redis_url": self._redis_url,
            "total_snapshots": total_snapshots,
            "total_opinions": total_opinions,
            "stocks": stocks,
        }


# ── 工厂函数 ──

_BLACKBOARD_INSTANCE: Optional[AbstractBlackboard] = None
_BLACKBOARD_LOCK = threading.Lock()


def get_blackboard() -> AbstractBlackboard:
    """
    获取黑板实例（工厂函数）

    优先根据环境变量选择后端：
    - USE_REDIS_BLACKBOARD=True + REDIS_URL 可用 → RedisBlackboard
    - 否则 → Blackboard（内存）

    结果会被缓存，进程内始终返回同一实例。
    """
    global _BLACKBOARD_INSTANCE
    if _BLACKBOARD_INSTANCE is not None:
        return _BLACKBOARD_INSTANCE

    with _BLACKBOARD_LOCK:
        if _BLACKBOARD_INSTANCE is not None:
            return _BLACKBOARD_INSTANCE

        use_redis = os.getenv("USE_REDIS_BLACKBOARD", "False").lower() == "true"
        if use_redis:
            try:
                _BLACKBOARD_INSTANCE = RedisBlackboard()
                if _BLACKBOARD_INSTANCE._using_redis():
                    logger.info("黑板后端: Redis")
                    return _BLACKBOARD_INSTANCE
                # Redis 初始化失败但已创建 fallback，仍然可用
                logger.info("黑板后端: Redis(降级到内存)")
                return _BLACKBOARD_INSTANCE
            except Exception as e:
                logger.error(f"Redis 黑板创建异常: {e}，使用内存黑板")
                _BLACKBOARD_INSTANCE = Blackboard()
        else:
            _BLACKBOARD_INSTANCE = Blackboard()
            logger.info("黑板后端: 内存")

        return _BLACKBOARD_INSTANCE


def set_blackboard(bb: AbstractBlackboard) -> None:
    """手动设置黑板实例（用于测试注入 Mock）"""
    global _BLACKBOARD_INSTANCE
    with _BLACKBOARD_LOCK:
        _BLACKBOARD_INSTANCE = bb
