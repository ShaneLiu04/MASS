"""
BacktestEngine multi_agent 策略单元测试
"""
import numpy as np
import pandas as pd
import pytest

from agent.core.backtest_engine import BacktestEngine, BacktestConfig
from agent.core.decision_engine import DecisionEngine


class TestBacktestMultiAgent:
    """测试 multi_agent 策略回测与数据闭环"""

    def _make_df(self, n: int = 120, seed: int = 42) -> pd.DataFrame:
        """构造模拟 K 线数据"""
        np.random.seed(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # 构造带趋势的价格序列
        ret = np.random.randn(n) * 0.008
        ret[:40] += 0.004   # 前期上涨
        ret[80:] -= 0.004   # 后期下跌
        price = 100 * np.exp(np.cumsum(ret))
        return pd.DataFrame({
            "date": dates,
            "open": price * (1 + np.random.randn(n) * 0.005),
            "high": price * (1 + abs(np.random.randn(n)) * 0.012),
            "low": price * (1 - abs(np.random.randn(n)) * 0.012),
            "close": price,
            "volume": np.random.randint(100_000, 1_000_000, n),
        })

    def test_multi_agent_requires_decision_engine(self):
        """multi_agent 策略必须传入 decision_engine"""
        df = self._make_df()
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="multi_agent"):
            engine.run(df, "000001", strategy="multi_agent")

    def test_multi_agent_runs_and_records_outcomes(self):
        """multi_agent 策略应执行回测并记录 outcome"""
        df = self._make_df(n=150)
        de = DecisionEngine()
        de.reset_accuracy_history()
        config = BacktestConfig(multi_agent_lookback=10)
        engine = BacktestEngine(config)

        result = engine.run(df, "000001", strategy="multi_agent", decision_engine=de)

        # 回测应产生结果
        assert result.stock_code == "000001"
        assert result.strategy == "multi_agent"
        assert isinstance(result.total_return_pct, float)

        # 准确率统计应有记录
        stats = de.get_accuracy_stats()
        total_samples = sum(s["samples"] for s in stats.values())
        assert total_samples > 0, "应有 outcome 被记录"

    def test_multi_agent_with_lookback(self):
        """不同 lookback 应产生不同记录数"""
        df = self._make_df(n=150)
        de1 = DecisionEngine()
        de1.reset_accuracy_history()
        de2 = DecisionEngine()
        de2.reset_accuracy_history()

        engine1 = BacktestEngine(BacktestConfig(multi_agent_lookback=5))
        engine2 = BacktestEngine(BacktestConfig(multi_agent_lookback=20))

        engine1.run(df, "000001", strategy="multi_agent", decision_engine=de1)
        engine2.run(df, "000001", strategy="multi_agent", decision_engine=de2)

        total1 = sum(s["samples"] for s in de1.get_accuracy_stats().values())
        total2 = sum(s["samples"] for s in de2.get_accuracy_stats().values())
        # lookback 越小，记录越多（因为更多信号能活到验证日）
        assert total1 >= total2

    def test_multi_agent_strategy_desc(self):
        """策略描述应正确注册"""
        engine = BacktestEngine()
        fn, desc = engine._get_strategy("multi_agent")
        assert fn == engine._strategy_multi_agent
        assert "Agent" in desc

    def test_simulate_agent_opinions_structure(self):
        """模拟 Agent 信号应包含所有必要 Agent"""
        df = self._make_df()
        engine = BacktestEngine()
        opinions = engine._simulate_agent_opinions(df, 60)
        expected = {"TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent", "RA-Agent"}
        assert set(opinions.keys()) == expected
        for op in opinions.values():
            assert op.signal in (-1, 0, 1)
            assert 0.0 <= op.confidence <= 1.0
