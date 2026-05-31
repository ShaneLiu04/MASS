"""
Prompt 压缩器 — 关键指标提取 + 摘要

核心设计：
- 每个 Agent 只保留其分析维度最关心的核心字段
- 数值精度截断（4 位 → 2 位小数）
- 删除空值、默认值、冗余嵌套
- 输出结构化的紧凑文本，Token 预计减少 40~50%

用法：
    from agent.core.prompt_compressor import PromptCompressor
    pc = PromptCompressor()
    compact_text = pc.compress_for_agent(snapshot, "TA-Agent")
"""
from typing import Dict, Any, List, Optional

import orjson
from loguru import logger

from agent.core.blackboard import StockSnapshot
from config import PROMPT_COMPRESSION_ENABLED


# ── 每个 Agent 关注的核心字段白名单 ──
# 字段路径用 "." 分隔，如 "indicators.ma5"
_AGENT_FIELD_WHITELIST: Dict[str, Dict[str, List[str]]] = {
    "TA-Agent": {
        "indicators": [
            "current_price", "change_pct", "volume", "turnover",
            "ma5", "ma20", "ma60", "ma120",
            "rsi14", "macd", "macd_signal", "macd_hist",
            "kdj_k", "kdj_d", "kdj_j",
            "boll_upper", "boll_mid", "boll_lower",
            "volume_ma5", "volume_ma20",
            "multi_timeframe", "support_resistance", "chart_patterns", "ta_score",
        ],
        "market_context": ["sector_performance", "market_trend"],
    },
    "FA-Agent": {
        "fundamentals": [
            "pe_ttm", "pb", "ps", "roe", "roic",
            "gross_margin", "net_margin", "operating_margin",
            "debt_to_equity", "current_ratio", "quick_ratio",
            "revenue_growth", "net_profit_growth",
            "operating_cash_flow", "free_cash_flow",
            "industry", "company_name",
            "forward_pe", "peg", "implied_roe",
        ],
        "market_context": ["sector_pe", "sector_pb"],
    },
    "CA-Agent": {
        "fund_flow": [
            "main_net_inflow", "main_inflow_pct",
            "north_net_inflow", "north_cumulative",
            "margin_balance", "margin_buy", "margin_repay",
            "block_trade_net", "block_trade_count",
            "chip_cr90", "vwap_main_cost", "profit_ratio", "lock_ratio",
        ],
        "indicators": ["current_price", "volume", "volume_ma5", "volume_ma20"],
    },
    "SA-Agent": {
        "sentiment_data": [
            "sentiment_index", "sentiment_percentile", "sentiment_momentum",
            "crowd_behavior", "extreme_signal",
            "news", "news_analysis",
            "sector_heat",
            "kline_sentiment", "fund_flow_sentiment",
        ],
        "fundamentals": ["industry"],
    },
    "MA-Agent": {
        "market_context": [
            "index_performance", "sector_flows", "market_style",
            "liquidity_score", "policy_direction",
        ],
        "macro_data": [
            "pmi", "pmi_trend", "interest_rate", "fx_rate",
            "policy_stance", "global_economy",
        ],
        "fundamentals": ["industry", "quarterly_revenue_growth"],
    },
    "RA-Agent": {
        "risk_metrics": [
            "volatility_20d", "volatility_60d",
            "max_drawdown", "var_95", "cvar_95",
            "beta", "sharpe_ratio",
            "gap_risk", "liquidity_risk",
            "stress_test", "tail_risk",
        ],
        "indicators": ["current_price", "ma60"],
        "fundamentals": ["debt_to_equity", "current_ratio"],
    },
}


# ── 默认值集合（等于这些值时视为无信息，可删除） ──
# 注意：list/dict 不可哈希，不在集合中，由 _is_empty 单独判断
_DEFAULT_SENTINELS = {
    None, "", "无", "未知", "N/A", "n/a", "NA",
    0, 0.0, "0", "0.0", "0%",
    "null", "NULL",
}


def _is_empty(val: Any) -> bool:
    """判断值是否为空/无信息"""
    # 先检查不可哈希类型，避免 TypeError
    if isinstance(val, dict) and len(val) == 0:
        return True
    if isinstance(val, (list, tuple)) and len(val) == 0:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    # 对可哈希类型检查哨兵集合
    try:
        if val in _DEFAULT_SENTINELS:
            return True
    except TypeError:
        pass
    return False


def _round_nested(obj: Any, decimals: int = 2) -> Any:
    """递归截断浮点数精度"""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_nested(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_nested(v, decimals) for v in obj]
    return obj


def _prune_empty(obj: Any) -> Any:
    """递归删除空值字段"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            pv = _prune_empty(v)
            if not _is_empty(pv):
                cleaned[k] = pv
        return cleaned
    if isinstance(obj, list):
        cleaned = [_prune_empty(v) for v in obj]
        return [v for v in cleaned if not _is_empty(v)]
    return obj


def _extract_fields(source: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """从 source dict 中提取指定字段，支持嵌套路径如 'a.b'"""
    result: Dict[str, Any] = {}
    for key in keys:
        if "." in key:
            parts = key.split(".")
            val = source
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    val = None
                    break
            if val is not None:
                # 将嵌套路径扁平化存储，如 "a.b" → "a_b"
                flat_key = key.replace(".", "_")
                result[flat_key] = val
        elif key in source:
            result[key] = source[key]
    return result


def _dict_to_compact_text(data: Any, title: str = "") -> str:
    """将 dict/list 转为紧凑的键值文本（比 JSON 更省 token）"""
    lines = [f"=== {title} ==="] if title else []

    if isinstance(data, list):
        for idx, item in enumerate(data, 1):
            if isinstance(item, dict):
                lines.append(f"第{idx}条:")
                for sk, sv in item.items():
                    lines.append(f"  {sk}: {sv}")
            else:
                lines.append(f"  {item}")
        return "\n".join(lines)

    if not isinstance(data, dict):
        lines.append(str(data))
        return "\n".join(lines)

    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for sk, sv in v.items():
                lines.append(f"  {sk}: {sv}")
        elif isinstance(v, list):
            if len(v) == 0:
                continue
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


class PromptCompressor:
    """
    Prompt 压缩器 — 为每个 Agent 提取核心指标，删除冗余数据

    开启后，每个 Agent 的 prompt 中只包含其分析维度最关心的字段，
    而非完整的 StockSnapshot JSON dump。
    """

    def __init__(self, enabled: Optional[bool] = None):
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = PROMPT_COMPRESSION_ENABLED
        self._whitelist = _AGENT_FIELD_WHITELIST

    def compress_for_agent(self, snapshot: StockSnapshot, agent_id: str) -> str:
        """
        为指定 Agent 生成压缩后的 prompt 文本

        Args:
            snapshot: 股票数据快照
            agent_id: Agent 标识，如 "TA-Agent"

        Returns:
            压缩后的 prompt 文本字符串
        """
        if not self._enabled:
            # 未开启时回退到原始全量 prompt
            return snapshot.to_prompt_context()

        fields = self._whitelist.get(agent_id)
        if not fields:
            logger.warning(f"PromptCompressor: {agent_id} 无白名单，回退到全量 prompt")
            return snapshot.to_prompt_context()

        compressed: Dict[str, Any] = {
            "stock_code": snapshot.stock_code,
            "stock_name": snapshot.stock_name,
            "current_price": round(snapshot.current_price, 2),
        }

        for category, keys in fields.items():
            source = getattr(snapshot, category, None)
            if not isinstance(source, dict):
                continue
            extracted = _extract_fields(source, keys)
            if extracted:
                compressed[category] = extracted

        # K 线数据仅 TA-Agent 保留最近 5 条（原 10 条）
        if agent_id == "TA-Agent" and snapshot.kline_df is not None and not snapshot.kline_df.empty:
            df = snapshot.kline_df.tail(5)
            klines = []
            for row in df.itertuples():
                klines.append({
                    "date": str(getattr(row, "date", getattr(row, "Index", ""))),
                    "open": round(getattr(row, "open", 0), 2),
                    "high": round(getattr(row, "high", 0), 2),
                    "low": round(getattr(row, "low", 0), 2),
                    "close": round(getattr(row, "close", 0), 2),
                    "volume": int(getattr(row, "volume", 0)),
                })
            compressed["kline_recent_5"] = klines

        # 数值截断 + 删除空值
        rounded = _round_nested(compressed, decimals=2)
        pruned = _prune_empty(rounded)

        # 转为紧凑文本
        lines = [
            f"股票代码: {pruned.pop('stock_code', '')}",
            f"股票名称: {pruned.pop('stock_name', '')}",
            f"当前价格: {pruned.pop('current_price', 0)}",
        ]
        for category, data in pruned.items():
            lines.append("")
            lines.append(_dict_to_compact_text(data, title=category))

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取压缩器状态"""
        return {
            "enabled": self._enabled,
            "agent_whitelist_count": len(self._whitelist),
        }
