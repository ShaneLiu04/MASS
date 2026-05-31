"""
RA-Agent v2.2 动态止损策略单元测试
覆盖多种止损策略计算、智能推荐逻辑、降级逻辑整合
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from agent.agents.ra_agent import RA_Agent
from agent.core.blackboard import StockSnapshot
from agent.models.agent_response import DynamicStopLossResult, StopLossStrategy


class TestRA_AgentDynamicStopLoss(unittest.TestCase):
    """RA-Agent 动态止损策略测试类"""

    def _make_agent(self):
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

    def _make_snapshot(
        self,
        price=15.0,
        atr=0.65,
        atr_pct=4.3,
        vol=25.0,
        beta=1.1,
        win_rate=50.0,
        low_20d=14.1,
        low_60d=13.5,
        boll_lower=13.8,
        ma_alignment="",
        trend_direction="",
    ):
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=price,
            kline_df=None,
            risk_metrics={
                "annual_volatility": vol,
                "max_drawdown": -12.0,
                "sharpe_ratio": 1.2,
                "beta": beta,
                "avg_amount_5d": 5.5,
                "annual_return": 8.0,
                "win_rate": win_rate,
            },
            fundamentals={"industry": "银行"},
            indicators={
                "atr14": atr,
                "atr_pct": atr_pct,
                "low_20d": low_20d,
                "low_60d": low_60d,
                "boll_lower": boll_lower,
                "ma_alignment": ma_alignment,
                "trend_direction": trend_direction,
                "position_20d": 0.55,
                "position_60d": 0.48,
                "high_20d": 16.2,
                "high_60d": 17.0,
            },
        )

    # ── 基础结构 ──
    def test_analyze_returns_dynamic_stop_loss(self):
        """analyze 返回的 raw_data 包含 dynamic_stop_loss"""
        agent = self._make_agent()
        opinion = agent.analyze(self._make_snapshot())
        self.assertIn("dynamic_stop_loss", opinion.raw_data)
        dsl = opinion.raw_data["dynamic_stop_loss"]
        self.assertIn("recommended_strategy", dsl)
        self.assertIn("recommended_stop_loss", dsl)

    def test_dynamic_stop_loss_structure(self):
        """动态止损结果包含 8 种策略"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot())
        self.assertIsInstance(dsl, DynamicStopLossResult)
        self.assertIsInstance(dsl.volatility_adaptive, StopLossStrategy)
        self.assertIsInstance(dsl.atr_based_1x, StopLossStrategy)
        self.assertIsInstance(dsl.atr_based_2x, StopLossStrategy)
        self.assertIsInstance(dsl.atr_based_3x, StopLossStrategy)
        self.assertIsInstance(dsl.trailing, StopLossStrategy)
        self.assertIsInstance(dsl.time_based, StopLossStrategy)
        self.assertIsInstance(dsl.technical_support, StopLossStrategy)
        self.assertIsInstance(dsl.technical_bollinger, StopLossStrategy)
        self.assertIsInstance(dsl.technical_low, StopLossStrategy)

    # ── ATR 止损计算 ──
    def test_atr_based_calculation(self):
        """ATR 止损价格计算正确"""
        agent = self._make_agent()
        price, atr = 15.0, 0.65
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(price=price, atr=atr))
        self.assertAlmostEqual(dsl.atr_based_1x.stop_price, price - atr, places=2)
        self.assertAlmostEqual(dsl.atr_based_2x.stop_price, price - 2 * atr, places=2)
        self.assertAlmostEqual(dsl.atr_based_3x.stop_price, price - 3 * atr, places=2)

    def test_atr_based_percentages(self):
        """ATR 止损幅度绝对值单调递增"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot())
        # 1x ATR 止损幅度更接近 0（更小亏损），3x ATR 更远离 0（更大亏损）
        self.assertGreater(dsl.atr_based_1x.stop_pct, dsl.atr_based_2x.stop_pct)
        self.assertGreater(dsl.atr_based_2x.stop_pct, dsl.atr_based_3x.stop_pct)
        self.assertGreater(dsl.atr_based_1x.stop_price, dsl.atr_based_2x.stop_price)

    # ── 波动率自适应止损 ──
    def test_volatility_adaptive_low_vol(self):
        """低波动环境止损比例收紧"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=10.0))
        self.assertAlmostEqual(dsl.volatility_adaptive.stop_pct, -5.0, places=1)

    def test_volatility_adaptive_high_vol(self):
        """高波动环境止损比例放宽"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=50.0))
        self.assertAlmostEqual(dsl.volatility_adaptive.stop_pct, -12.0, places=1)

    def test_volatility_adaptive_mid_vol(self):
        """中等波动环境止损比例适中"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=25.0))
        self.assertAlmostEqual(dsl.volatility_adaptive.stop_pct, -7.0, places=1)

    # ── 技术位止损 ──
    def test_technical_low_stop(self):
        """前期低点止损在技术位下方"""
        agent = self._make_agent()
        low_20d = 14.1
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(low_20d=low_20d))
        self.assertAlmostEqual(dsl.technical_low.stop_price, low_20d * 0.98, places=2)

    def test_technical_support_stop(self):
        """支撑位止损使用 60 日低点和布林带下轨的较高者"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(low_60d=13.5, boll_lower=13.8))
        # support_level = max(13.5, 13.8) = 13.8
        self.assertAlmostEqual(dsl.technical_support.stop_price, 13.8 * 0.97, places=2)

    # ── 时间止损 ──
    def test_time_stop_high_win_rate(self):
        """高胜率时间窗口更长"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(win_rate=60.0))
        # 高胜率 -> 15天, -5%
        self.assertEqual(dsl.time_based.stop_pct, -5.0)

    def test_time_stop_low_win_rate(self):
        """低胜率时间窗口更短、止损更紧"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(win_rate=40.0))
        # 低胜率 -> 7天, -10%
        self.assertEqual(dsl.time_based.stop_pct, -10.0)

    # ── 智能推荐 ──
    def test_recommend_high_volatility(self):
        """高波动推荐 3x ATR"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=40.0))
        self.assertEqual(dsl.recommended_strategy, "atr_based_3x")
        self.assertIn("高波动", dsl.recommendation_reason)

    def test_recommend_low_vol_high_win(self):
        """低波动高胜率推荐 1x ATR"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=12.0, win_rate=60.0))
        self.assertEqual(dsl.recommended_strategy, "atr_based_1x")
        self.assertIn("低波动", dsl.recommendation_reason)

    def test_recommend_bull_trend(self):
        """多头排列推荐移动止损"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(ma_alignment="多头排列"))
        self.assertEqual(dsl.recommended_strategy, "trailing")
        self.assertIn("多头排列", dsl.recommendation_reason)

    def test_recommend_up_trend(self):
        """上升趋势推荐移动止损"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(trend_direction="上升趋势"))
        self.assertEqual(dsl.recommended_strategy, "trailing")

    def test_recommend_high_beta(self):
        """高Beta推荐波动率自适应"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(beta=1.5))
        self.assertEqual(dsl.recommended_strategy, "volatility_adaptive")
        self.assertIn("高Beta", dsl.recommendation_reason)

    def test_recommend_default(self):
        """默认推荐 2x ATR"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(vol=25.0, beta=1.0))
        self.assertEqual(dsl.recommended_strategy, "atr_based_2x")

    # ── 移动止损规则 ──
    def test_trailing_rule_format(self):
        """移动止损规则包含上移逻辑"""
        agent = self._make_agent()
        dsl = agent._calculate_dynamic_stop_loss(self._make_snapshot(atr=0.5))
        self.assertIn("初始止损", dsl.trailing_rule)
        self.assertIn("止损上移", dsl.trailing_rule)
        self.assertIn("永不下移", dsl.trailing_rule)

    # ── 降级逻辑 ──
    def test_fallback_uses_dynamic_stop_loss(self):
        """降级逻辑使用动态止损"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")
        opinion = agent.analyze(self._make_snapshot())
        self.assertIn("dynamic_stop_loss", opinion.raw_data)
        dsl = opinion.raw_data["dynamic_stop_loss"]
        self.assertGreater(dsl["recommended_stop_loss"], 0)
        self.assertIn("推荐止损策略", str(opinion.reasoning))

    def test_fallback_stop_loss_not_zero(self):
        """降级止损价不为零"""
        agent = self._make_agent()
        agent.llm.chat.side_effect = RuntimeError("模拟 LLM 失败")
        opinion = agent.analyze(self._make_snapshot(price=20.0, atr=0.8))
        self.assertLess(opinion.raw_data["recommended_stop_loss"], 20.0)
        self.assertGreater(opinion.raw_data["recommended_stop_loss"], 15.0)

    # ── Prompt 构建 ──
    def test_prompt_contains_dynamic_stop_loss(self):
        """Prompt 包含动态止损策略章节"""
        agent = self._make_agent()
        prompt = agent._build_ra_prompt(self._make_snapshot())
        self.assertIn("动态止损策略", prompt)
        self.assertIn("ATR 止损", prompt)
        self.assertIn("移动止损", prompt)
        self.assertIn("时间止损", prompt)
        self.assertIn("技术位止损", prompt)

    def test_prompt_contains_recommended_strategy(self):
        """Prompt 包含推荐策略"""
        agent = self._make_agent()
        prompt = agent._build_ra_prompt(self._make_snapshot())
        self.assertIn("【推荐】", prompt)
        self.assertIn("推荐理由", prompt)


if __name__ == "__main__":
    unittest.main()
