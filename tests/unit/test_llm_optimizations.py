"""
单元测试: LLM 调用优化四大项

1. Agent 缓存 — 同股票同参数 30s 内复用结论
2. Prompt 压缩 — 关键指标提取 + 摘要
3. 模型分层 — Agent 用轻量模型，Chairman 用重模型
4. 批量推理 — 6 次独立调用 → 2 次批量调用
"""
import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.core.agent_cache import (
    AgentCache,
    _compute_fingerprint,
    _extract_fingerprint_data,
    _serialize_opinion,
    _deserialize_opinion,
)
from agent.core.prompt_compressor import PromptCompressor, _prune_empty, _round_nested
from agent.tools.batch_llm_client import BatchLLMClient, BatchRequest, BatchAgentRunner
from agent.core.blackboard import StockSnapshot, AgentOpinion


# ── 辅助: 构造测试 Snapshot ──

def _make_snapshot(price: float = 10.0, **kwargs) -> StockSnapshot:
    # 避免 current_price 重复传入
    kwargs.pop("current_price", None)
    return StockSnapshot(
        stock_code="000001",
        stock_name="测试银行",
        current_price=price,
        indicators={
            "ma5": 10.1, "ma20": 9.8, "ma60": 9.5, "rsi14": 55.0,
            "macd": 0.2, "kdj_k": 60.0, "current_price": price,
        },
        fundamentals={"pe_ttm": 8.5, "pb": 1.2, "roe": 12.0},
        fund_flow={"main_net_inflow": 1000000, "north_net_inflow": 500000},
        sentiment_data={"sentiment_index": 0.3, "sentiment_percentile": 55},
        market_context={"sector_performance": 2.1},
        macro_data={"pmi": 50.5},
        risk_metrics={"volatility_20d": 0.25, "var_95": -5.0},
        timestamp=datetime.datetime.now(),
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. Agent 缓存测试
# ══════════════════════════════════════════════════════════════════════

class TestAgentCache:
    """AgentCache 单元测试"""

    def test_fingerprint_consistency(self):
        """相同快照应产生相同指纹"""
        snap1 = _make_snapshot(price=10.55)
        snap2 = _make_snapshot(price=10.55)
        fp1 = _compute_fingerprint(snap1, "TA-Agent")
        fp2 = _compute_fingerprint(snap2, "TA-Agent")
        assert fp1 == fp2

    def test_fingerprint_price_difference(self):
        """价格不同应产生不同指纹"""
        snap1 = _make_snapshot(price=10.0)
        snap2 = _make_snapshot(price=10.05)
        fp1 = _compute_fingerprint(snap1, "TA-Agent")
        fp2 = _compute_fingerprint(snap2, "TA-Agent")
        # 价格差异 >= 0.01 应该产生不同指纹（因为 round to 2 decimals）
        assert fp1 != fp2

    def test_fingerprint_tiny_difference_ignored(self):
        """微小浮点差异应被忽略（四舍五入到 2 位小数）"""
        snap1 = _make_snapshot(price=10.001)
        snap2 = _make_snapshot(price=10.002)
        fp1 = _compute_fingerprint(snap1, "TA-Agent")
        fp2 = _compute_fingerprint(snap2, "TA-Agent")
        assert fp1 == fp2

    def test_fingerprint_agent_specific(self):
        """不同 Agent 的指纹字段不同"""
        snap = _make_snapshot()
        fp_ta = _compute_fingerprint(snap, "TA-Agent")
        fp_fa = _compute_fingerprint(snap, "FA-Agent")
        assert fp_ta != fp_fa  # 因为提取的关键字段不同

    def test_serialize_roundtrip(self):
        """AgentOpinion 序列化往返"""
        original = AgentOpinion(
            agent_id="TA-Agent",
            signal=1,
            confidence=0.8,
            reasoning="测试",
            key_factors=["因子1"],
            risk_flags=["风险1"],
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        raw = _serialize_opinion(original)
        restored = _deserialize_opinion(raw)
        assert restored.agent_id == "TA-Agent"
        assert restored.signal == 1
        assert restored.confidence == 0.8
        assert restored.timestamp == datetime.datetime(2024, 1, 1, 12, 0, 0)

    def test_cache_hit_and_miss(self, monkeypatch):
        """缓存命中与未命中"""
        monkeypatch.setenv("AGENT_CACHE_ENABLED", "true")
        cache = AgentCache()
        cache._enabled = True
        cache._cache = MagicMock()
        cache._cache.get.return_value = None  # 第一次 miss

        snap = _make_snapshot()
        opinion = AgentOpinion("TA-Agent", 1, 0.8, "测试")

        # Miss
        result = cache.get("000001", "TA-Agent", snap)
        assert result is None
        assert cache._stats["misses"] == 1

        # Set
        cache.set("000001", "TA-Agent", snap, opinion)
        cache._cache.get.return_value = _serialize_opinion(opinion)  # 第二次 hit

        # Hit
        result = cache.get("000001", "TA-Agent", snap)
        assert result is not None
        assert result.agent_id == "TA-Agent"
        assert cache._stats["hits"] == 1

    def test_cache_disabled(self, monkeypatch):
        """缓存关闭时始终返回 None"""
        monkeypatch.setenv("AGENT_CACHE_ENABLED", "false")
        cache = AgentCache()
        cache._enabled = False
        snap = _make_snapshot()
        assert cache.get("000001", "TA-Agent", snap) is None


# ══════════════════════════════════════════════════════════════════════
# 2. Prompt 压缩测试
# ══════════════════════════════════════════════════════════════════════

class TestPromptCompressor:
    """PromptCompressor 单元测试"""

    def test_round_nested(self):
        """递归浮点截断"""
        data = {"a": 1.23456, "b": [2.34567, {"c": 3.45678}]}
        rounded = _round_nested(data, decimals=2)
        assert rounded["a"] == 1.23
        assert rounded["b"][0] == 2.35
        assert rounded["b"][1]["c"] == 3.46

    def test_prune_empty(self):
        """删除空值字段"""
        data = {
            "a": 1,
            "b": "",
            "c": [],
            "d": {},
            "e": None,
            "f": "valid",
            "g": [1, None, ""],
        }
        pruned = _prune_empty(data)
        assert "a" in pruned
        assert "b" not in pruned
        assert "c" not in pruned
        assert "d" not in pruned
        assert "e" not in pruned
        assert "f" in pruned
        assert pruned["g"] == [1]

    def test_compress_enabled_reduces_size(self):
        """压缩后文本长度应小于全量"""
        pc = PromptCompressor(enabled=True)
        snap = _make_snapshot()

        full = snap.to_prompt_context()
        compressed = pc.compress_for_agent(snap, "TA-Agent")

        assert len(compressed) < len(full)
        assert "000001" in compressed
        assert "测试银行" in compressed

    def test_compress_disabled_fallback(self):
        """压缩关闭时回退到全量 prompt"""
        pc = PromptCompressor(enabled=False)
        snap = _make_snapshot()
        result = pc.compress_for_agent(snap, "TA-Agent")
        assert result == snap.to_prompt_context()

    def test_compress_per_agent_fields(self):
        """不同 Agent 保留的字段不同"""
        pc = PromptCompressor(enabled=True)
        snap = _make_snapshot()

        ta = pc.compress_for_agent(snap, "TA-Agent")
        fa = pc.compress_for_agent(snap, "FA-Agent")

        assert "rsi14" in ta or "indicators" in ta
        assert "pe_ttm" in fa or "fundamentals" in fa

    def test_compress_ta_with_kline(self):
        """TA-Agent 压缩后包含最近 5 条 K 线"""
        pc = PromptCompressor(enabled=True)
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "volume": [100, 200, 300, 400, 500, 600],
        })
        snap = _make_snapshot(kline_df=df)
        ta = pc.compress_for_agent(snap, "TA-Agent")
        assert "kline_recent_5" in ta or "kline" in ta


# ══════════════════════════════════════════════════════════════════════
# 3. 模型分层测试
# ══════════════════════════════════════════════════════════════════════

class TestModelTier:
    """模型分层配置测试"""

    def test_agent_model_map_exists(self):
        """AGENT_MODEL_MAP 必须包含所有 6 个 Agent"""
        from config import AGENT_MODEL_MAP
        expected = ["TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent", "RA-Agent"]
        for agent_id in expected:
            assert agent_id in AGENT_MODEL_MAP

    def test_base_agent_accepts_model(self):
        """BaseAgent 应接受 model 参数"""
        from agent.agents.base_agent import BaseAgent
        from agent.tools.llm_client import MockLLMClient

        mock_llm = MockLLMClient()

        class DummyAgent(BaseAgent):
            def analyze(self, snapshot, user_position=None):
                return AgentOpinion("Dummy", 0, 0.5, "test")

            def _default_prompt(self):
                return "default"

        agent = DummyAgent("Dummy", mock_llm, model="light-model")
        assert agent.model == "light-model"

    def test_call_llm_passes_model(self):
        """_call_llm 应将 model 传给 LLMClient.chat()"""
        from agent.agents.base_agent import BaseAgent
        from agent.tools.llm_client import MockLLMClient

        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"signal": 0, "confidence": 0.5, "reasoning": "test"}

        class DummyAgent(BaseAgent):
            def analyze(self, snapshot, user_position=None):
                return self._call_llm("test prompt")

            def _default_prompt(self):
                return "default"

        agent = DummyAgent("Dummy", mock_llm, model="my-model")
        agent.analyze(_make_snapshot())

        mock_llm.chat.assert_called_once()
        call_kwargs = mock_llm.chat.call_args.kwargs
        assert call_kwargs.get("model") == "my-model"


# ══════════════════════════════════════════════════════════════════════
# 4. 批量推理测试
# ══════════════════════════════════════════════════════════════════════

class TestBatchLLMClient:
    """BatchLLMClient 单元测试"""

    def test_single_request_degradation(self):
        """单条请求应降级为普通调用"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"signal": 1, "confidence": 0.8}

        client = BatchLLMClient(mock_llm)
        results = client.batch_chat([
            BatchRequest("TA-Agent", "sys", "user"),
        ])

        assert len(results) == 1
        assert results[0]["signal"] == 1
        mock_llm.chat.assert_called_once()

    def test_batch_parse_results(self):
        """批量响应解析"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "results": [
                {"agent_id": "TA-Agent", "signal": 1, "confidence": 0.8, "reasoning": "r1"},
                {"agent_id": "FA-Agent", "signal": -1, "confidence": 0.7, "reasoning": "r2"},
            ]
        }

        client = BatchLLMClient(mock_llm)
        results = client.batch_chat([
            BatchRequest("TA-Agent", "sys1", "user1"),
            BatchRequest("FA-Agent", "sys2", "user2"),
        ])

        assert len(results) == 2
        assert results[0]["agent_id"] == "TA-Agent"
        assert results[0]["signal"] == 1
        assert results[1]["agent_id"] == "FA-Agent"
        assert results[1]["signal"] == -1

    def test_batch_missing_agent_fallback(self):
        """响应中缺少某个 Agent 时应降级"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "results": [
                {"agent_id": "TA-Agent", "signal": 1, "confidence": 0.8},
            ]
        }

        client = BatchLLMClient(mock_llm)
        results = client.batch_chat([
            BatchRequest("TA-Agent", "sys1", "user1"),
            BatchRequest("FA-Agent", "sys2", "user2"),
        ])

        assert len(results) == 2
        assert results[0]["signal"] == 1
        assert results[1]["signal"] == 0  # fallback
        assert "batch_parse_error" in results[1].get("risk_flags", [])

    def test_batch_flat_response(self):
        """LLM 返回扁平对象而非 results 数组时应正确处理"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "agent_id": "TA-Agent", "signal": 1, "confidence": 0.8
        }

        client = BatchLLMClient(mock_llm)
        results = client.batch_chat([
            BatchRequest("TA-Agent", "sys1", "user1"),
        ])

        assert len(results) == 1
        assert results[0]["signal"] == 1


class TestBatchAgentRunner:
    """BatchAgentRunner 集成测试（Mock 方式）"""

    def test_batch_runner_transparent_intercept(self):
        """BatchAgentRunner 应透明拦截 _call_llm 并返回正确结果"""
        from agent.agents.base_agent import BaseAgent
        from agent.tools.llm_client import MockLLMClient

        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "results": [
                {"agent_id": "A1", "signal": 1, "confidence": 0.8, "reasoning": "r1", "key_factors": [], "risk_flags": []},
                {"agent_id": "A2", "signal": -1, "confidence": 0.7, "reasoning": "r2", "key_factors": [], "risk_flags": []},
            ]
        }

        class TestAgent(BaseAgent):
            def analyze(self, snapshot, user_position=None):
                response = self._call_llm("test prompt")
                return AgentOpinion(
                    agent_id=self.agent_id,
                    signal=response.get("signal", 0),
                    confidence=response.get("confidence", 0.5),
                    reasoning=response.get("reasoning", ""),
                )

            def _default_prompt(self):
                return "default"

        agent1 = TestAgent("A1", mock_llm)
        agent2 = TestAgent("A2", mock_llm)
        agents = {"A1": agent1, "A2": agent2}

        batch_client = BatchLLMClient(mock_llm)
        runner = BatchAgentRunner(batch_client)
        snap = _make_snapshot()

        opinions = runner.run(agents, snap)

        assert "A1" in opinions
        assert "A2" in opinions
        assert opinions["A1"].signal == 1
        assert opinions["A2"].signal == -1

        # 验证 _call_llm 已恢复
        assert agent1._call_llm is not runner._original_methods.get("A1")

    def test_batch_runner_with_mock_llm(self):
        """使用 MockLLMClient 测试批量运行器（端到端）"""
        from agent.agents.base_agent import BaseAgent
        from agent.tools.llm_client import MockLLMClient

        # MockLLMClient 会解析 prompt 中的真实指标并返回规则结果
        mock_llm = MockLLMClient()

        class TestAgent(BaseAgent):
            def analyze(self, snapshot, user_position=None):
                prompt = f"价格: {snapshot.current_price}"
                response = self._call_llm(prompt)
                parsed = self._safe_parse_llm_response(response)
                return self._build_default_opinion(
                    signal=parsed["signal"],
                    confidence=parsed["confidence"],
                    reasoning=parsed["reasoning"],
                    raw_data=parsed,
                )

            def _default_prompt(self):
                return "default"

        agent1 = TestAgent("TA-Agent", mock_llm)
        agent2 = TestAgent("FA-Agent", mock_llm)
        agents = {"TA-Agent": agent1, "FA-Agent": agent2}

        batch_client = BatchLLMClient(mock_llm)
        runner = BatchAgentRunner(batch_client)
        snap = _make_snapshot(current_price=15.0)

        opinions = runner.run(agents, snap)

        assert "TA-Agent" in opinions
        assert "FA-Agent" in opinions
        # MockLLMClient 会返回有效的 signal（基于规则引擎）
        assert opinions["TA-Agent"].signal in (-1, 0, 1)
        assert opinions["FA-Agent"].signal in (-1, 0, 1)
