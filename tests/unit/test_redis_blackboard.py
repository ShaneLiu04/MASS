"""
单元测试: RedisBlackboard 分布式黑板

由于测试环境没有 Redis 服务，全部使用 Mock 验证逻辑正确性。
覆盖：正常 Redis 操作、降级回退、DataFrame 序列化。
"""
import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from agent.core.blackboard import (
    _snapshot_to_dict,
    _snapshot_from_dict,
    _opinion_to_dict,
    _opinion_from_dict,
    RedisBlackboard,
    Blackboard,
    StockSnapshot,
    AgentOpinion,
    get_blackboard,
    set_blackboard,
    AbstractBlackboard,
)


# ── 辅助: 构建 Mock Redis ──

def _make_mock_redis():
    """返回一个支持常用命令的 Mock Redis 实例"""
    mock = MagicMock()
    mock.ping.return_value = True

    # pipeline 内建状态
    pipe = MagicMock()
    pipe.execute.return_value = []
    mock.pipeline.return_value = pipe

    return mock, pipe


class TestSerializationHelpers:
    """序列化/反序列化辅助函数测试"""

    def test_snapshot_to_dict_with_dataframe(self):
        df = pd.DataFrame({"open": [1.0, 2.0], "close": [1.5, 2.5]})
        snap = StockSnapshot(
            stock_code="000001",
            stock_name="测试",
            current_price=10.0,
            kline_df=df,
            indicators={"ma5": 10},
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        d = _snapshot_to_dict(snap)
        assert d["stock_code"] == "000001"
        assert d["kline_records"] == [{"open": 1.0, "close": 1.5}, {"open": 2.0, "close": 2.5}]
        assert d["timestamp"] == "2024-01-01T12:00:00"

    def test_snapshot_to_dict_without_dataframe(self):
        snap = StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=10.0, kline_df=None
        )
        d = _snapshot_to_dict(snap)
        assert d["kline_records"] is None

    def test_snapshot_roundtrip(self):
        df = pd.DataFrame({"open": [1.0], "close": [1.5]})
        original = StockSnapshot(
            stock_code="000001",
            stock_name="测试",
            current_price=10.0,
            kline_df=df,
            indicators={"ma5": 10},
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        restored = _snapshot_from_dict(_snapshot_to_dict(original))
        assert restored.stock_code == "000001"
        assert restored.current_price == 10.0
        assert restored.kline_df is not None
        assert list(restored.kline_df.columns) == ["open", "close"]
        assert restored.timestamp == datetime.datetime(2024, 1, 1, 12, 0, 0)

    def test_opinion_roundtrip(self):
        original = AgentOpinion(
            agent_id="TA-Agent",
            signal=1,
            confidence=0.8,
            reasoning="测试",
            key_factors=["因子1"],
            risk_flags=["风险1"],
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        restored = _opinion_from_dict(_opinion_to_dict(original))
        assert restored.agent_id == "TA-Agent"
        assert restored.signal == 1
        assert restored.confidence == 0.8
        assert restored.reasoning == "测试"
        assert restored.key_factors == ["因子1"]
        assert restored.timestamp == datetime.datetime(2024, 1, 1, 12, 0, 0)


class TestRedisBlackboardMocked:
    """RedisBlackboard — 使用 Mock Redis 测试全部接口"""

    @pytest.fixture(autouse=True)
    def _reset_factory(self):
        """每个测试前重置黑板工厂，防止单例污染"""
        set_blackboard(None)
        yield
        set_blackboard(None)

    def test_redis_init_success(self):
        mock_redis, _ = _make_mock_redis()
        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard(redis_url="redis://test:6379/0")
            assert rb._using_redis() is True
            mock_redis.ping.assert_called_once()

    def test_redis_init_failure_fallback(self):
        """Redis ping 失败时自动回退到内存黑板"""
        with patch("redis.from_url", side_effect=ImportError("no redis")):
            rb = RedisBlackboard(redis_url="redis://test:6379/0")
            assert rb._using_redis() is False
            assert rb._fallback is not None

    def test_publish_snapshot(self):
        mock_redis, pipe = _make_mock_redis()
        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            snap = StockSnapshot(
                stock_code="000001", stock_name="测试", current_price=10.0
            )
            rb.publish_snapshot(snap)

        pipe.setex.assert_called_once()
        # key 应该包含 stock_code
        assert "000001" in pipe.setex.call_args[0][0]
        pipe.delete.assert_called_once()
        pipe.sadd.assert_called_once_with("mass:stocks", "000001")

    def test_submit_opinion(self):
        mock_redis, pipe = _make_mock_redis()
        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            op = AgentOpinion("TA-Agent", 1, 0.8, "测试")
            rb.submit_opinion("000001", op)

        pipe.lpush.assert_called_once()
        pipe.expire.assert_called_once()

    def test_get_snapshot(self):
        mock_redis, _ = _make_mock_redis()
        snap = StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=10.0
        )
        mock_redis.get.return_value = __import__("orjson").dumps(
            _snapshot_to_dict(snap)
        )

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            result = rb.get_snapshot("000001")

        assert result is not None
        assert result.stock_code == "000001"
        assert result.current_price == 10.0

    def test_get_snapshot_miss(self):
        mock_redis, _ = _make_mock_redis()
        mock_redis.get.return_value = None

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            assert rb.get_snapshot("999999") is None

    def test_get_opinions(self):
        mock_redis, _ = _make_mock_redis()
        op = AgentOpinion("TA-Agent", 1, 0.8, "测试")
        raw = __import__("orjson").dumps(_opinion_to_dict(op))
        mock_redis.lrange.return_value = [raw]

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            ops = rb.get_opinions("000001")

        assert len(ops) == 1
        assert ops[0].agent_id == "TA-Agent"
        assert ops[0].signal == 1

    def test_get_opinion_by_agent(self):
        mock_redis, _ = _make_mock_redis()
        op1 = AgentOpinion("TA-Agent", 1, 0.8, "测试")
        op2 = AgentOpinion("FA-Agent", 0, 0.6, "测试")
        raw1 = __import__("orjson").dumps(_opinion_to_dict(op1))
        raw2 = __import__("orjson").dumps(_opinion_to_dict(op2))
        mock_redis.lrange.return_value = [raw1, raw2]

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            result = rb.get_opinion_by_agent("000001", "FA-Agent")

        assert result is not None
        assert result.agent_id == "FA-Agent"

    def test_clear_stock(self):
        mock_redis, pipe = _make_mock_redis()
        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            rb.clear_stock("000001")

        pipe.delete.assert_any_call("mass:snapshot:000001")
        pipe.delete.assert_any_call("mass:opinions:000001")
        pipe.srem.assert_called_once_with("mass:stocks", "000001")

    def test_get_all_stock_codes(self):
        mock_redis, _ = _make_mock_redis()
        mock_redis.smembers.return_value = {b"000001", "000002"}

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            codes = rb.get_all_stock_codes()

        assert set(codes) == {"000001", "000002"}

    def test_get_stats(self):
        mock_redis, pipe = _make_mock_redis()
        mock_redis.smembers.return_value = ["000001", "000002"]
        # pipeline execute: exists=1, llen=3 for 000001; exists=0, llen=0 for 000002
        pipe.execute.return_value = [1, 3, 0, 0]

        with patch("redis.from_url", return_value=mock_redis):
            rb = RedisBlackboard()
            stats = rb.get_stats()

        assert stats["backend"] == "redis"
        assert stats["total_snapshots"] == 1
        assert stats["total_opinions"] == 3
        assert set(stats["stocks"]) == {"000001", "000002"}


class TestBlackboardFallback:
    """Redis 不可用时自动回退到内存黑板的端到端测试"""

    def test_fallback_to_memory(self):
        """Redis 连接失败后所有操作自动回退到内存黑板"""
        with patch("redis.from_url", side_effect=Exception("Connection refused")):
            rb = RedisBlackboard(redis_url="redis://bad:6379/0")
            assert rb._using_redis() is False

            snap = StockSnapshot(
                stock_code="000001", stock_name="测试", current_price=10.0
            )
            rb.publish_snapshot(snap)
            assert rb.get_snapshot("000001").current_price == 10.0

            op = AgentOpinion("TA-Agent", 1, 0.8, "测试")
            rb.submit_opinion("000001", op)
            assert len(rb.get_opinions("000001")) == 1

            rb.clear_stock("000001")
            assert rb.get_snapshot("000001") is None


class TestFactory:
    """get_blackboard() 工厂函数测试"""

    def test_factory_returns_memory_by_default(self, monkeypatch):
        monkeypatch.delenv("USE_REDIS_BLACKBOARD", raising=False)
        set_blackboard(None)
        bb = get_blackboard()
        assert isinstance(bb, Blackboard)

    def test_factory_returns_redis_when_enabled(self, monkeypatch):
        monkeypatch.setenv("USE_REDIS_BLACKBOARD", "true")
        set_blackboard(None)

        mock_redis, _ = _make_mock_redis()
        with patch("redis.from_url", return_value=mock_redis):
            bb = get_blackboard()

        assert isinstance(bb, RedisBlackboard)

    def test_factory_caches_instance(self, monkeypatch):
        monkeypatch.delenv("USE_REDIS_BLACKBOARD", raising=False)
        set_blackboard(None)
        bb1 = get_blackboard()
        bb2 = get_blackboard()
        assert bb1 is bb2

    def test_set_blackboard_injection(self):
        """set_blackboard 支持测试注入 Mock"""
        mock_bb = MagicMock(spec=AbstractBlackboard)
        set_blackboard(mock_bb)
        assert get_blackboard() is mock_bb
        set_blackboard(None)

    def test_factory_falls_back_on_redis_init_failure(self, monkeypatch):
        monkeypatch.setenv("USE_REDIS_BLACKBOARD", "true")
        set_blackboard(None)

        with patch("redis.from_url", side_effect=Exception("no redis")):
            bb = get_blackboard()

        assert isinstance(bb, RedisBlackboard)
        assert bb._using_redis() is False
        assert bb._fallback is not None
