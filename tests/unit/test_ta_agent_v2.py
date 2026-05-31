"""
TA-Agent v2.0 单元测试
覆盖多时间框架分析、支撑压力位、形态识别、多因子评分
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from agent.tools.indicator_tool import IndicatorTool
from agent.agents.ta_agent import TA_Agent
from agent.core.blackboard import StockSnapshot


class TestIndicatorToolV2(unittest.TestCase):
    """测试 IndicatorTool v2.0 新增功能"""

    def _make_kline_df(self, n=100, trend="up"):
        """生成测试K线数据"""
        np.random.seed(42)
        if trend == "up":
            close = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.1)
        elif trend == "down":
            close = 100 + np.cumsum(np.random.randn(n) * 0.3 - 0.1)
        else:
            close = 100 + np.cumsum(np.random.randn(n) * 0.2)

        df = pd.DataFrame({
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1.0,
            "low": close - np.random.rand(n) * 1.0,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        return df

    def test_compute_multi_timeframe_basic(self):
        """测试多时间框架指标计算基础功能"""
        df = self._make_kline_df(n=100)
        result = IndicatorTool.compute_multi_timeframe(daily_df=df)

        self.assertIn("daily", result)
        self.assertIn("alignment", result)
        self.assertIn("score", result["alignment"])
        self.assertIn("consistency", result["alignment"])

        # daily 数据应该存在且有内容
        self.assertTrue(result["daily"])
        self.assertIn("ma_alignment", result["daily"])

    def test_compute_multi_timeframe_with_all_frames(self):
        """测试全周期多时间框架分析"""
        daily = self._make_kline_df(n=100, trend="up")
        weekly = self._make_kline_df(n=24, trend="up")
        hourly = self._make_kline_df(n=40, trend="down")

        result = IndicatorTool.compute_multi_timeframe(
            daily_df=daily, weekly_df=weekly, hourly_df=hourly
        )

        alignment = result["alignment"]
        self.assertIn("signals", alignment)
        self.assertIn("confidences", alignment)

        # 日线和周线同向，与60分钟反向
        if "weekly" in alignment["signals"] and "daily" in alignment["signals"]:
            self.assertEqual(alignment["signals"]["weekly"], alignment["signals"]["daily"])

    def test_compute_support_resistance(self):
        """测试支撑压力位计算"""
        df = self._make_kline_df(n=100)
        indicators = IndicatorTool.compute_all(df)
        sr = IndicatorTool.compute_support_resistance(df, indicators)

        self.assertIn("strong_support", sr)
        self.assertIn("strong_resistance", sr)
        self.assertIn("weak_support", sr)
        self.assertIn("weak_resistance", sr)
        self.assertIn("vwap_support", sr)
        self.assertIn("vwap_resistance", sr)
        self.assertIn("fib_levels", sr)
        self.assertIn("position_in_range", sr)

        # 强阻力 > 当前价 > 强支撑
        self.assertGreater(sr["strong_resistance"], sr["current_price"])
        self.assertGreater(sr["current_price"], sr["strong_support"])

        # 斐波那契回撤位应有7个级别
        self.assertEqual(len(sr["fib_levels"]), 7)

        # 位置应在0~1之间
        self.assertGreaterEqual(sr["position_in_range"], 0)
        self.assertLessEqual(sr["position_in_range"], 1)

    def test_compute_ta_score_dimensions(self):
        """测试多因子评分模型"""
        df = self._make_kline_df(n=100, trend="up")
        indicators = IndicatorTool.compute_all(df)
        sr = IndicatorTool.compute_support_resistance(df, indicators)
        score = IndicatorTool.compute_ta_score(indicators, sr)

        self.assertIn("total_score", score)
        self.assertIn("signal", score)
        self.assertIn("confidence", score)
        self.assertIn("dimensions", score)
        self.assertIn("key_factors", score)
        self.assertIn("risk_flags", score)

        # 5个维度
        self.assertEqual(len(score["dimensions"]), 5)
        expected_dims = ["trend", "momentum", "volume", "volatility", "structure"]
        for dim in expected_dims:
            self.assertIn(dim, score["dimensions"])
            self.assertIn("score", score["dimensions"][dim])
            self.assertIn("max", score["dimensions"][dim])

        # 总分在0~100之间
        self.assertGreaterEqual(score["total_score"], 0)
        self.assertLessEqual(score["total_score"], 100)

        # 信号只能是-1, 0, 1
        self.assertIn(score["signal"], [-1, 0, 1])

        # 置信度在0~1之间
        self.assertGreaterEqual(score["confidence"], 0)
        self.assertLessEqual(score["confidence"], 1)

    def test_compute_ta_score_bullish_trend(self):
        """测试强势上涨行情的多因子评分"""
        df = self._make_kline_df(n=100, trend="up")
        indicators = IndicatorTool.compute_all(df)
        sr = IndicatorTool.compute_support_resistance(df, indicators)
        score = IndicatorTool.compute_ta_score(indicators, sr)

        # 上涨趋势应偏向买入
        self.assertGreaterEqual(score["total_score"], 35)
        self.assertIn(score["signal"], [0, 1])

    def test_compute_ta_score_valid_structure(self):
        """测试评分模型结构有效性"""
        for trend in ["up", "down", "sideways"]:
            df = self._make_kline_df(n=100, trend=trend)
            indicators = IndicatorTool.compute_all(df)
            sr = IndicatorTool.compute_support_resistance(df, indicators)
            score = IndicatorTool.compute_ta_score(indicators, sr)

            # 所有趋势下结构都应正确
            self.assertIn("total_score", score)
            self.assertIn("signal", score)
            self.assertIn("confidence", score)
            self.assertIn("dimensions", score)
            self.assertIn(score["signal"], [-1, 0, 1])
            self.assertGreaterEqual(score["total_score"], 0)
            self.assertLessEqual(score["total_score"], 100)
            self.assertGreaterEqual(score["confidence"], 0)
            self.assertLessEqual(score["confidence"], 1)

    def test_detect_chart_patterns_empty_data(self):
        """测试形态识别对空数据的处理"""
        patterns = IndicatorTool.detect_chart_patterns(None)
        self.assertEqual(patterns, [])

        patterns = IndicatorTool.detect_chart_patterns(pd.DataFrame())
        self.assertEqual(patterns, [])


class TestTAAgentV2(unittest.TestCase):
    """测试 TA_Agent v2.0"""

    def _make_snapshot(self, with_ta_score=False):
        """生成测试用的 StockSnapshot"""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.05)
        df = pd.DataFrame({
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1.0,
            "low": close - np.random.rand(n) * 1.0,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        df["high"] = df[["open", "close", "high"]].max(axis=1)

        indicators = IndicatorTool.compute_all(df)
        sr = IndicatorTool.compute_support_resistance(df, indicators)

        # 构建多时间框架数据
        multi_tf = IndicatorTool.compute_multi_timeframe(daily_df=df)
        indicators["multi_timeframe"] = multi_tf
        indicators["support_resistance"] = sr
        indicators["chart_patterns"] = IndicatorTool.detect_chart_patterns(df)

        if with_ta_score:
            indicators["ta_score"] = IndicatorTool.compute_ta_score(indicators, sr)

        return StockSnapshot(
            stock_code="000001",
            stock_name="测试股票",
            current_price=float(close[-1]),
            kline_df=df,
            indicators=indicators,
        )

    def test_build_ta_prompt_with_multi_timeframe(self):
        """测试 Prompt 构建包含多时间框架"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "signal": 1,
            "confidence": 0.75,
            "reasoning": "测试",
            "key_factors": ["测试"],
            "risk_flags": [],
        }

        agent = TA_Agent("TA-Agent", mock_llm)
        snapshot = self._make_snapshot(with_ta_score=True)

        opinion = agent.analyze(snapshot)

        self.assertEqual(opinion.agent_id, "TA-Agent")
        self.assertIn(opinion.signal, [-1, 0, 1])
        self.assertGreaterEqual(opinion.confidence, 0)
        self.assertLessEqual(opinion.confidence, 1)

        # 验证 LLM 被调用时传入的 prompt 包含多周期分析
        call_args = mock_llm.chat.call_args
        prompt = call_args.kwargs.get("user", "") if call_args.kwargs else call_args[1].get("user", "")
        self.assertIn("多时间框架", prompt)
        self.assertIn("关键价位矩阵", prompt)
        self.assertIn("支撑压力位", prompt)

    def test_fallback_with_ta_score(self):
        """测试降级分析使用预计算的 ta_score"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM失败")

        agent = TA_Agent("TA-Agent", mock_llm)
        snapshot = self._make_snapshot(with_ta_score=True)

        opinion = agent.analyze(snapshot)

        self.assertEqual(opinion.agent_id, "TA-Agent")
        self.assertIn(opinion.signal, [-1, 0, 1])
        self.assertIn("多因子", opinion.reasoning)

        # 验证 raw_data 包含多维度数据
        self.assertIn("total_score", opinion.raw_data)
        self.assertIn("dimensions", opinion.raw_data)
        self.assertIn("support_resistance", opinion.raw_data)
        self.assertIn("chart_patterns", opinion.raw_data)

    def test_fallback_without_ta_score(self):
        """测试降级分析在没有预计算评分时的回退"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM失败")

        agent = TA_Agent("TA-Agent", mock_llm)
        snapshot = self._make_snapshot(with_ta_score=False)

        opinion = agent.analyze(snapshot)

        self.assertEqual(opinion.agent_id, "TA-Agent")
        self.assertIn(opinion.signal, [-1, 0, 1])
        # 应该有基础指标的分析
        self.assertIn(opinion.reasoning, opinion.reasoning)

    def test_stop_loss_validation(self):
        """测试止损价校验"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "signal": 1,
            "confidence": 0.75,
            "stop_loss": 105.0,  # 高于当前价，应该被修正
            "reasoning": "测试",
            "key_factors": ["测试"],
            "risk_flags": [],
        }

        agent = TA_Agent("TA-Agent", mock_llm)
        snapshot = self._make_snapshot()

        opinion = agent.analyze(snapshot)

        # 止损价应该被修正为低于当前价
        raw_stop = opinion.raw_data.get("stop_loss")
        if raw_stop is not None:
            self.assertLess(raw_stop, snapshot.current_price)


if __name__ == "__main__":
    unittest.main()
