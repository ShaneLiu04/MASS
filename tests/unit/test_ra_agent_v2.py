"""
RA-Agent v2.0 单元测试
覆盖压力测试引擎、尾部风险指标、降级逻辑、Prompt构建
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from agent.agents.ra_agent import RA_Agent
from agent.core.blackboard import StockSnapshot
from agent.models.agent_response import StressTestResult, StressScenario


class TestRA_AgentV2(unittest.TestCase):
    """RA-Agent v2.0 测试类"""

    def _make_kline_df(self, n=120, trend="random", volatility=0.02, seed=42):
        """生成测试K线数据"""
        np.random.seed(seed)
        if trend == "up":
            close = 100 + np.cumsum(np.random.randn(n) * volatility + 0.001)
        elif trend == "down":
            close = 100 + np.cumsum(np.random.randn(n) * volatility - 0.001)
        else:
            close = 100 + np.cumsum(np.random.randn(n) * volatility)

        df = pd.DataFrame({
            "open": close - np.random.rand(n) * volatility * 100,
            "high": close + np.random.rand(n) * volatility * 150,
            "low": close - np.random.rand(n) * volatility * 150,
            "close": close,
            "volume": np.random.randint(10000, 100000, n),
            "amount": np.random.randint(1e6, 1e8, n),
        })
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        return df

    def _make_agent(self):
        """构造带 MockLLM 的 RA_Agent"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "signal": 0,
            "confidence": 0.75,
            "risk_level": 3,
            "max_position_pct": 0.15,
            "risk_reward_ratio": 2.0,
            "reasoning": "测试推理",
            "key_factors": ["波动率正常"],
            "risk_flags": [],
            "black_scenarios": ["测试黑天鹅"],
            "position_sizing_formula": "测试公式",
        }
        return RA_Agent("RA-Agent", mock_llm)

    def _make_snapshot(self, kline=None, risk_metrics=None, indicators=None):
        """构造 StockSnapshot"""
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            kline_df=kline,
            risk_metrics=risk_metrics or {
                "annual_volatility": 25.0,
                "max_drawdown": -12.0,
                "sharpe_ratio": 1.2,
                "beta": 1.1,
                "avg_amount_5d": 5.5,
                "annual_return": 8.0,
            },
            indicators=indicators or {
                "atr14": 0.65,
                "atr_pct": 4.3,
                "position_20d": 0.55,
                "position_60d": 0.48,
                "high_20d": 16.2,
                "low_20d": 14.1,
                "high_60d": 17.0,
                "low_60d": 13.5,
            },
        )

    # ── 压力测试引擎 ──
    def test_run_stress_tests_structure(self):
        """压力测试结果结构完整性"""
        agent = self._make_agent()
        kline = self._make_kline_df(n=120)
        snapshot = self._make_snapshot(kline=kline)

        stress = agent._run_stress_tests(snapshot)
        self.assertIsInstance(stress, StressTestResult)

        # 四情景
        self.assertIsInstance(stress.base, StressScenario)
        self.assertIsInstance(stress.bull, StressScenario)
        self.assertIsInstance(stress.bear, StressScenario)
        self.assertIsInstance(stress.black_swan, StressScenario)

        # 概率和为 1
        total_prob = stress.base.probability + stress.bull.probability + stress.bear.probability + stress.black_swan.probability
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # 悲观与黑天鹅回撤应更深
        self.assertLess(stress.bear.max_drawdown, stress.base.max_drawdown)
        self.assertLess(stress.black_swan.max_drawdown, stress.bear.max_drawdown)

        # 尾部风险指标存在
        self.assertIsNotNone(stress.var_95)
        self.assertIsNotNone(stress.cvar_95)
        self.assertIn(stress.var_method, ("历史模拟", "正态近似"))

    def test_var_cvar_historical(self):
        """历史模拟法 VaR/CVaR 计算正确性"""
        agent = self._make_agent()
        kline = self._make_kline_df(n=120, volatility=0.02)

        var_95, cvar_95, method = agent._calculate_var_cvar(kline, annual_vol=25.0)
        self.assertEqual(method, "历史模拟")
        # VaR 应为负数（损失）
        self.assertLess(var_95, 0)
        # CVaR 应比 VaR 更悲观（更负）
        self.assertLessEqual(cvar_95, var_95)

    def test_var_cvar_normal_fallback(self):
        """数据不足时回退正态法"""
        agent = self._make_agent()
        var_95, cvar_95, method = agent._calculate_var_cvar(None, annual_vol=32.0)
        self.assertEqual(method, "正态近似")
        self.assertLess(var_95, 0)
        self.assertLessEqual(cvar_95, var_95)

    def test_max_consecutive_losses(self):
        """最大连续亏损天数统计"""
        agent = self._make_agent()

        # 构造明确连续下跌序列
        close = np.array([100, 99, 98, 97, 96, 97, 95, 94, 93, 92, 100])
        df = pd.DataFrame({"close": close})
        streak = agent._count_max_consecutive_losses(df)
        # 最大连续下跌：100->99->98->97->96 (4天) 和 97->95->94->93->92 (4天)
        self.assertEqual(streak, 4)

        # 空数据回退
        self.assertEqual(agent._count_max_consecutive_losses(None), 0)

    def test_blowup_probability_with_data(self):
        """有 K 线数据时的爆仓概率"""
        agent = self._make_agent()
        kline = self._make_kline_df(n=120, volatility=0.02)
        prob = agent._estimate_blowup_probability(kline, annual_vol=25.0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 100.0)

    def test_blowup_probability_without_data(self):
        """无 K 线数据时的爆仓概率（正态近似）"""
        agent = self._make_agent()
        prob = agent._estimate_blowup_probability(None, annual_vol=60.0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 100.0)

    def test_gap_risk_analysis(self):
        """跳空风险统计"""
        agent = self._make_agent()
        # 构造含跳空数据
        df = pd.DataFrame({
            "open": [100, 102.5, 99.0, 101.0, 95.0, 100.0],
            "close": [101, 102, 100, 96, 99, 101],
        })
        gap = agent._analyze_gap_risk(df)
        self.assertIn("up_gaps", gap)
        self.assertIn("down_gaps", gap)
        self.assertIn("max_down_gap_pct", gap)

    def test_liquidity_risk(self):
        """流动性风险指标"""
        agent = self._make_agent()
        kline = self._make_kline_df(n=120)
        risk_metrics = {"avg_amount_5d": 3.2}
        liq = agent._analyze_liquidity_risk(kline, risk_metrics)
        self.assertIn("amihud_illiquidity", liq)
        self.assertIn("amount_volatility_5d", liq)
        self.assertIn("amount_trend_20d", liq)
        self.assertGreaterEqual(liq["amihud_illiquidity"], 0)

    def test_volatility_term_structure(self):
        """波动率期限结构"""
        agent = self._make_agent()
        kline = self._make_kline_df(n=120, volatility=0.02)
        vts = agent._calculate_volatility_term_structure(kline)
        self.assertIn("vol_5d", vts)
        self.assertIn("vol_20d", vts)
        self.assertIn("vol_60d", vts)
        self.assertIn("vol_trend_ratio", vts)
        self.assertIn("vol_trend_desc", vts)
        # 期限结构应有值
        self.assertGreater(vts["vol_20d"], 0)

    # ── 主流程与降级 ──
    def test_analyze_returns_opinion(self):
        """analyze 主流程返回合法 AgentOpinion"""
        agent = self._make_agent()
        snapshot = self._make_snapshot(kline=self._make_kline_df(n=120))
        opinion = agent.analyze(snapshot)

        self.assertEqual(opinion.agent_id, "RA-Agent")
        self.assertEqual(opinion.signal, 0)
        self.assertGreaterEqual(opinion.confidence, 0.0)
        self.assertLessEqual(opinion.confidence, 1.0)
        self.assertTrue(len(opinion.reasoning) > 0)

        raw = opinion.raw_data
        self.assertIn("risk_level", raw)
        self.assertIn("stress_test", raw)
        self.assertTrue(1 <= raw["risk_level"] <= 5)
        self.assertTrue(0.05 <= raw["max_position_pct"] <= 0.50)

    def test_analyze_event_risk_override(self):
        """事件风险强制提升 risk_level"""
        agent = self._make_agent()
        # Mock 返回低 risk_level 但含财报风险
        agent.llm.chat.return_value = {
            "signal": 0,
            "confidence": 0.7,
            "risk_level": 2,
            "max_position_pct": 0.20,
            "risk_reward_ratio": 1.5,
            "reasoning": "测试",
            "key_factors": [],
            "risk_flags": ["即将披露财报"],
            "black_scenarios": [],
            "position_sizing_formula": "",
        }
        snapshot = self._make_snapshot()
        opinion = agent.analyze(snapshot)
        self.assertGreaterEqual(opinion.raw_data["risk_level"], 3)

    def test_fallback_opinion(self):
        """降级逻辑返回合法 Opinion 且含压力测试"""
        agent = self._make_agent()
        # 让 LLM 抛异常触发降级
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")
        snapshot = self._make_snapshot(kline=self._make_kline_df(n=120))

        opinion = agent.analyze(snapshot)
        self.assertEqual(opinion.agent_id, "RA-Agent")
        self.assertEqual(opinion.signal, 0)
        self.assertIn("stress_test", opinion.raw_data)
        self.assertIn("LLM调用异常", str(opinion.risk_flags))

    def test_fallback_risk_level_bounds(self):
        """降级风险等级始终在 1~5"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")

        # 高波动场景
        snapshot_high = self._make_snapshot(
            kline=self._make_kline_df(n=120, volatility=0.05),
            risk_metrics={"annual_volatility": 55, "max_drawdown": -45, "beta": 1.5},
        )
        op_high = agent.analyze(snapshot_high)
        self.assertEqual(op_high.raw_data["risk_level"], 5)

        # 低波动场景
        snapshot_low = self._make_snapshot(
            kline=self._make_kline_df(n=120, volatility=0.005),
            risk_metrics={"annual_volatility": 8, "max_drawdown": -3, "beta": 0.8},
        )
        op_low = agent.analyze(snapshot_low)
        # 低波动基础等级为1，但压力测试校正（如连续亏损天数）可能合理提升至2
        self.assertLessEqual(op_low.raw_data["risk_level"], 2)

    def test_fallback_position_sizing(self):
        """降级仓位计算逻辑合理性"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")

        snapshot = self._make_snapshot(
            kline=self._make_kline_df(n=120, volatility=0.03),
            risk_metrics={"annual_volatility": 35, "max_drawdown": -25, "beta": 1.2},
        )
        opinion = agent.analyze(snapshot)
        max_pos = opinion.raw_data["max_position_pct"]
        self.assertTrue(0.05 <= max_pos <= 0.50)
        # 高波动 + 深回撤，仓位应偏低
        self.assertLess(max_pos, 0.25)

    # ── Prompt 构建 ──
    def test_prompt_contains_stress_test(self):
        """Prompt 包含压力测试章节"""
        agent = self._make_agent()
        snapshot = self._make_snapshot(kline=self._make_kline_df(n=120))
        prompt = agent._build_ra_prompt(snapshot)
        self.assertIn("压力测试情景", prompt)
        self.assertIn("尾部风险指标", prompt)
        self.assertIn("跳空风险统计", prompt)
        self.assertIn("流动性风险指标", prompt)
        self.assertIn("波动率期限结构", prompt)

    def test_prompt_with_user_position(self):
        """含用户持仓时 Prompt 包含组合风险分析"""
        agent = self._make_agent()
        snapshot = self._make_snapshot()
        user_pos = {
            "positions": [
                {"code": "600519", "name": "贵州茅台", "value": 100000, "beta": 0.9, "volatility": 22, "industry": "白酒", "max_drawdown": -15},
            ],
            "planned_investment": 50000,
        }
        prompt = agent._build_ra_prompt(snapshot, user_position=user_pos)
        self.assertIn("组合风险分析", prompt)
        self.assertIn("组合Beta", prompt)


if __name__ == "__main__":
    unittest.main()
