"""
RA-Agent v2.1 组合风险分析单元测试
覆盖组合Beta、行业集中度、组合波动率、边际风险贡献、仓位约束
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from agent.agents.ra_agent import RA_Agent
from agent.core.blackboard import StockSnapshot
from agent.models.agent_response import PortfolioRiskResult


class TestRA_AgentPortfolioRisk(unittest.TestCase):
    """RA-Agent 组合风险分析测试类"""

    def _make_kline_df(self, n=120):
        """生成测试K线数据"""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.02)
        df = pd.DataFrame({
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1.0,
            "low": close - np.random.rand(n) * 1.0,
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
            "key_factors": [],
            "risk_flags": [],
            "black_scenarios": [],
            "position_sizing_formula": "",
        }
        return RA_Agent("RA-Agent", mock_llm)

    def _make_snapshot(self, beta=1.1, vol=25.0, industry="银行", dd=-12.0):
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            kline_df=self._make_kline_df(),
            risk_metrics={
                "annual_volatility": vol,
                "max_drawdown": dd,
                "sharpe_ratio": 1.2,
                "beta": beta,
                "avg_amount_5d": 5.5,
                "annual_return": 8.0,
            },
            fundamentals={"industry": industry},
            indicators={
                "atr14": 0.65, "atr_pct": 4.3,
                "position_20d": 0.55, "position_60d": 0.48,
                "high_20d": 16.2, "low_20d": 14.1,
                "high_60d": 17.0, "low_60d": 13.5,
            },
        )

    # ── 基础结构 ──
    def test_analyze_returns_portfolio_risk(self):
        """analyze 返回的 raw_data 包含 portfolio_risk"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "600519", "name": "贵州茅台", "value": 200000, "beta": 0.9, "volatility": 22, "industry": "白酒", "max_drawdown": -15},
            ],
            "planned_investment": 50000,
        }
        opinion = agent.analyze(self._make_snapshot(), user_position=user_pos)
        self.assertIn("portfolio_risk", opinion.raw_data)
        pr = opinion.raw_data["portfolio_risk"]
        self.assertIn("new_beta", pr)
        self.assertIn("industry_hhi", pr)

    def test_portfolio_risk_no_user_position(self):
        """无 user_position 时返回默认空结果"""
        agent = self._make_agent()
        pr = agent._analyze_portfolio_risk(self._make_snapshot(), None)
        self.assertIsInstance(pr, PortfolioRiskResult)
        self.assertEqual(pr.existing_position_count, 0)

    def test_portfolio_risk_empty_positions(self):
        """空持仓（首仓场景）"""
        agent = self._make_agent()
        user_pos = {"positions": [], "planned_investment": 100000, "total_portfolio_value": 100000}
        pr = agent._analyze_portfolio_risk(self._make_snapshot(), user_pos)
        self.assertEqual(pr.existing_position_count, 0)
        self.assertEqual(pr.position_constraint, "首仓建议")
        self.assertEqual(pr.new_weight_pct, 100.0)

    # ── 组合 Beta ──
    def test_portfolio_beta_calculation(self):
        """组合 Beta 计算正确性"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 100000, "beta": 1.0, "volatility": 20, "industry": "A", "max_drawdown": -10},
                {"code": "B", "name": "B", "value": 100000, "beta": 1.5, "volatility": 25, "industry": "B", "max_drawdown": -15},
            ],
            "planned_investment": 100000,
        }
        # 新增股票 beta=1.1
        pr = agent._analyze_portfolio_risk(self._make_snapshot(beta=1.1), user_pos)
        # 当前组合 beta = (100k*1.0 + 100k*1.5) / 200k = 1.25
        self.assertAlmostEqual(pr.current_beta, 1.25, places=1)
        # 新增后 = (100k*1.0 + 100k*1.5 + 100k*1.1) / 300k = 1.20
        self.assertAlmostEqual(pr.new_beta, 1.20, places=1)

    def test_beta_status_thresholds(self):
        """Beta 状态阈值判断"""
        agent = self._make_agent()
        # Beta 超标场景
        pr_high = agent._analyze_portfolio_risk(
            self._make_snapshot(beta=1.8),
            {"positions": [{"code": "A", "value": 100000, "beta": 1.6, "volatility": 25, "industry": "A", "max_drawdown": -15}], "planned_investment": 100000}
        )
        self.assertEqual(pr_high.beta_status, "超标")

        # Beta 正常场景
        pr_low = agent._analyze_portfolio_risk(
            self._make_snapshot(beta=0.8),
            {"positions": [{"code": "A", "value": 100000, "beta": 0.8, "volatility": 20, "industry": "A", "max_drawdown": -10}], "planned_investment": 100000}
        )
        self.assertEqual(pr_low.beta_status, "正常")

    # ── 行业集中度 ──
    def test_industry_concentration_high(self):
        """高行业集中度场景"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 200000, "beta": 1.0, "volatility": 20, "industry": "银行", "max_drawdown": -10},
                {"code": "B", "value": 100000, "beta": 1.2, "volatility": 22, "industry": "银行", "max_drawdown": -12},
            ],
            "planned_investment": 100000,
        }
        # 新增股票也是银行
        pr = agent._analyze_portfolio_risk(self._make_snapshot(industry="银行"), user_pos)
        # 银行总占比 = 400k/400k = 100%
        self.assertEqual(pr.max_industry_pct, 100.0)
        self.assertEqual(pr.industry_overlap, "高度重叠")
        self.assertEqual(pr.concentration_risk, "高")
        self.assertGreater(pr.industry_hhi, 2500)

    def test_industry_concentration_diversified(self):
        """分散持仓场景"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 50000, "beta": 1.0, "volatility": 20, "industry": "银行", "max_drawdown": -10},
                {"code": "B", "value": 50000, "beta": 1.2, "volatility": 22, "industry": "白酒", "max_drawdown": -12},
                {"code": "C", "value": 50000, "beta": 0.9, "volatility": 18, "industry": "医药", "max_drawdown": -8},
                {"code": "D", "value": 50000, "beta": 1.1, "volatility": 21, "industry": "电子", "max_drawdown": -11},
            ],
            "planned_investment": 50000,
        }
        pr = agent._analyze_portfolio_risk(self._make_snapshot(industry="新能源"), user_pos)
        self.assertEqual(pr.industry_overlap, "无")
        self.assertEqual(pr.concentration_risk, "低")
        self.assertLess(pr.industry_hhi, 2200)

    # ── 组合波动率与边际风险贡献 ──
    def test_portfolio_volatility_increases(self):
        """新增高波动股票后组合波动率上升"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 200000, "beta": 0.8, "volatility": 15, "industry": "A", "max_drawdown": -8},
            ],
            "planned_investment": 50000,
        }
        # 新增股票 vol=40
        pr = agent._analyze_portfolio_risk(self._make_snapshot(vol=40.0), user_pos)
        self.assertGreater(pr.new_volatility, pr.current_volatility)
        self.assertGreater(pr.marginal_risk_contribution, 0)

    # ── 仓位约束 ──
    def test_position_constraint_beta_exceeded(self):
        """组合Beta超标时仓位受限"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 200000, "beta": 1.6, "volatility": 25, "industry": "A", "max_drawdown": -15},
            ],
            "planned_investment": 100000,
        }
        pr = agent._analyze_portfolio_risk(self._make_snapshot(beta=1.6), user_pos)
        self.assertEqual(pr.beta_status, "超标")
        self.assertIn("组合Beta超标", pr.position_constraint)
        self.assertLessEqual(pr.recommended_max_position, 0.10)

    def test_position_constraint_concentration(self):
        """行业过度集中时仓位受限"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 300000, "beta": 1.0, "volatility": 20, "industry": "银行", "max_drawdown": -10},
            ],
            "planned_investment": 100000,
        }
        pr = agent._analyze_portfolio_risk(self._make_snapshot(industry="银行"), user_pos)
        self.assertEqual(pr.concentration_risk, "高")
        self.assertIn("行业过度集中", pr.position_constraint)
        self.assertLessEqual(pr.recommended_max_position, 0.10)

    def test_position_constraint_normal(self):
        """正常场景下仓位约束宽松"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 40000, "beta": 0.9, "volatility": 18, "industry": "白酒", "max_drawdown": -10},
                {"code": "B", "value": 40000, "beta": 1.0, "volatility": 20, "industry": "医药", "max_drawdown": -12},
                {"code": "C", "value": 40000, "beta": 1.0, "volatility": 19, "industry": "银行", "max_drawdown": -9},
                {"code": "D", "value": 40000, "beta": 1.1, "volatility": 21, "industry": "电子", "max_drawdown": -11},
            ],
            "planned_investment": 40000,
        }
        pr = agent._analyze_portfolio_risk(self._make_snapshot(beta=1.0, industry="新能源"), user_pos)
        self.assertEqual(pr.beta_status, "正常")
        self.assertEqual(pr.concentration_risk, "低")
        self.assertEqual(pr.position_constraint, "可配置")
        self.assertGreaterEqual(pr.recommended_max_position, 0.15)

    # ── 降级逻辑中的组合风险 ──
    def test_fallback_with_portfolio_constraint(self):
        """降级逻辑中组合风险约束生效"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")
        user_pos = {
            "positions": [
                {"code": "A", "value": 300000, "beta": 1.6, "volatility": 28, "industry": "银行", "max_drawdown": -15},
            ],
            "planned_investment": 100000,
        }
        opinion = agent.analyze(self._make_snapshot(beta=1.6, industry="银行"), user_position=user_pos)
        self.assertIn("portfolio_risk", opinion.raw_data)
        self.assertIn("组合Beta超标", str(opinion.risk_flags))
        # 仓位应被压缩
        self.assertLessEqual(opinion.raw_data["max_position_pct"], 0.10)

    def test_fallback_without_portfolio(self):
        """无持仓时降级逻辑正常"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")
        opinion = agent.analyze(self._make_snapshot())
        self.assertIn("portfolio_risk", opinion.raw_data)
        self.assertEqual(opinion.raw_data["portfolio_risk"]["existing_position_count"], 0)

    # ── Prompt 构建 ──
    def test_prompt_contains_portfolio_risk(self):
        """Prompt 包含组合风险分析章节"""
        agent = self._make_agent()
        user_pos = {
            "positions": [
                {"code": "A", "value": 100000, "beta": 1.0, "volatility": 20, "industry": "白酒", "max_drawdown": -10},
            ],
            "planned_investment": 50000,
        }
        prompt = agent._build_ra_prompt(self._make_snapshot(), user_position=user_pos)
        self.assertIn("组合风险分析", prompt)
        self.assertIn("组合Beta", prompt)
        self.assertIn("行业集中度", prompt)
        self.assertIn("边际风险贡献", prompt)

    def test_prompt_first_position(self):
        """首仓场景 Prompt 正确"""
        agent = self._make_agent()
        user_pos = {"positions": [], "planned_investment": 100000}
        prompt = agent._build_ra_prompt(self._make_snapshot(), user_position=user_pos)
        self.assertIn("首只持仓", prompt)


if __name__ == "__main__":
    unittest.main()
