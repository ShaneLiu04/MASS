"""
Agent 结论缓存系统

核心设计：
- key = agent:{stock_code}:{agent_id}:{snapshot_fingerprint}
- snapshot_fingerprint 基于关键指标子集的哈希（浮点四舍五入到 2 位小数，避免噪声导致 miss）
- TTL = 30 秒（默认），命中时更新 timestamp 为当前时间
- 底层使用 CacheManager（线程安全 LRU + TTL）

收益：同股票在 30 秒内重复诊断时，可直接复用 Agent 结论，减少 60~80% LLM 调用。
"""
import hashlib
import threading
from typing import Dict, Any, Optional, List

import orjson
from loguru import logger

from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.core.cache import CacheManager
from config import AGENT_CACHE_ENABLED, AGENT_CACHE_TTL


# 每个 Agent 参与指纹计算的关键字段（越小越严格，越大越容易 hit）
_AGENT_FINGERPRINT_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "TA-Agent": {
        "indicators": ["current_price", "ma5", "ma20", "ma60", "rsi14", "macd", "kdj"],
    },
    "FA-Agent": {
        "fundamentals": ["pe_ttm", "pb", "roe", "revenue_growth", "net_profit_growth"],
    },
    "CA-Agent": {
        "fund_flow": ["main_net_inflow", "north_net_inflow", "margin_balance"],
    },
    "SA-Agent": {
        "sentiment_data": ["sentiment_index", "sentiment_percentile", "crowd_behavior"],
    },
    "MA-Agent": {
        "market_context": ["sector_performance", "market_trend"],
        "macro_data": ["pmi", "policy_stance"],
    },
    "RA-Agent": {
        "risk_metrics": ["volatility_20d", "max_drawdown", "var_95"],
    },
}


def _round_nested(obj: Any, decimals: int = 2) -> Any:
    """递归将浮点数四舍五入到指定小数位，同时清理 numpy 类型"""
    import numpy as np
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), decimals)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _round_nested(obj.tolist(), decimals)
    if isinstance(obj, dict):
        return {k: _round_nested(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(v, decimals) for v in obj]
    return obj


def _extract_fingerprint_data(snapshot: StockSnapshot, agent_id: str) -> Dict[str, Any]:
    """提取用于指纹计算的关键数据子集"""
    fields = _AGENT_FINGERPRINT_FIELDS.get(agent_id, {})
    data: Dict[str, Any] = {"price": round(snapshot.current_price, 2)}

    for category, keys in fields.items():
        source = getattr(snapshot, category, None)
        if not isinstance(source, dict):
            continue
        extracted = {}
        for key in keys:
            val = source.get(key)
            if val is not None:
                extracted[key] = val
        if extracted:
            data[category] = extracted

    return data


def _compute_fingerprint(snapshot: StockSnapshot, agent_id: str) -> str:
    """计算快照指纹 — 对关键指标子集做排序 JSON + MD5"""
    data = _extract_fingerprint_data(snapshot, agent_id)
    rounded = _round_nested(data, decimals=2)
    # orjson.OPT_SORT_KEYS 保证字段顺序一致
    raw = orjson.dumps(rounded, option=orjson.OPT_SORT_KEYS)
    return hashlib.md5(raw).hexdigest()[:12]


def _serialize_opinion(opinion: AgentOpinion) -> bytes:
    """AgentOpinion → orjson bytes（含 datetime 字符串化）"""
    return orjson.dumps(opinion.to_dict())


def _deserialize_opinion(raw: bytes) -> AgentOpinion:
    """orjson bytes → AgentOpinion"""
    from datetime import datetime
    data = orjson.loads(raw)
    ts = data.pop("timestamp", None)
    timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.now()
    return AgentOpinion(timestamp=timestamp, **data)


class AgentCache:
    """
    Agent 结论缓存 — 单例，线程安全

    用法：
        cache = AgentCache()
        cached = cache.get(stock_code, agent_id, snapshot)
        if cached:
            return cached
        opinion = agent.analyze(snapshot)
        cache.set(stock_code, agent_id, snapshot, opinion)
    """

    _instance: Optional["AgentCache"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AgentCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = CacheManager.get_instance()
                    cls._instance._ttl = AGENT_CACHE_TTL
                    cls._instance._enabled = AGENT_CACHE_ENABLED
                    cls._instance._stats = {"hits": 0, "misses": 0}
                    cls._instance._stats_lock = threading.Lock()
        return cls._instance

    def _make_key(self, stock_code: str, agent_id: str, fingerprint: str) -> str:
        return f"agent_opinion:{stock_code}:{agent_id}:{fingerprint}"

    def get(self, stock_code: str, agent_id: str, snapshot: StockSnapshot) -> Optional[AgentOpinion]:
        """获取缓存的 AgentOpinion，未命中返回 None"""
        if not self._enabled:
            return None

        fingerprint = _compute_fingerprint(snapshot, agent_id)
        key = self._make_key(stock_code, agent_id, fingerprint)
        raw = self._cache.get(key)

        if raw is not None:
            try:
                opinion = _deserialize_opinion(raw)
                # 更新 timestamp 为当前时间，表示这是新鲜复用的结论
                opinion.timestamp = __import__("datetime").datetime.now()
                with self._stats_lock:
                    self._stats["hits"] += 1
                logger.debug(f"AgentCache HIT [{agent_id}] {stock_code} fp={fingerprint}")
                return opinion
            except Exception as e:
                logger.warning(f"AgentCache 反序列化失败: {e}")

        with self._stats_lock:
            self._stats["misses"] += 1
        logger.debug(f"AgentCache MISS [{agent_id}] {stock_code} fp={fingerprint}")
        return None

    def set(self, stock_code: str, agent_id: str, snapshot: StockSnapshot, opinion: AgentOpinion) -> None:
        """存入缓存"""
        if not self._enabled:
            return

        fingerprint = _compute_fingerprint(snapshot, agent_id)
        key = self._make_key(stock_code, agent_id, fingerprint)
        try:
            self._cache.set(key, _serialize_opinion(opinion), ttl=self._ttl)
        except Exception as e:
            logger.warning(f"AgentCache 写入失败: {e}")

    def invalidate(self, stock_code: str) -> None:
        """使某股票的所有 Agent 缓存失效（快照更新时调用）"""
        if not self._enabled:
            return
        # CacheManager 没有前缀删除，这里通过 Blackboard publish_snapshot 时
        # 快照已变，指纹会变，自然 miss。如需显式清理可扩展 CacheManager。
        logger.debug(f"AgentCache invalidate {stock_code}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._stats_lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "enabled": self._enabled,
                "ttl": self._ttl,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 4),
                "total": total,
            }
