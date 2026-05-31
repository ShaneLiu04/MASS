"""
DecisionEngine 动态权重学习 + 非线性置信度加权单元测试
"""
import os
import json
import tempfile
from pathlib import Path

import pytest

from agent.core.decision_engine import DecisionEngine
from agent.core.blackboard import AgentOpinion


class TestDecisionEngineDynamicWeights:
    """测试动态权重学习核心功能"""

    def test_record_outcome_buy_correct(self):
        """买入预测 + 大涨 = 正确记录"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        assert de.record_outcome("TA-Agent", 1, 8.0) is True
        stats = de.get_accuracy_stats()
        assert stats["TA"]["samples"] == 1
        assert stats["TA"]["correct"] == 1

    def test_record_outcome_buy_incorrect(self):
        """买入预测 + 微涨 = 错误记录"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        assert de.record_outcome("TA-Agent", 1, 3.0) is True
        stats = de.get_accuracy_stats()
        assert stats["TA"]["samples"] == 1
        assert stats["TA"]["correct"] == 0

    def test_record_outcome_hold_correct(self):
        """观望预测 + 横盘 = 正确记录"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        assert de.record_outcome("FA-Agent", 0, 2.0) is True
        stats = de.get_accuracy_stats()
        assert stats["FA"]["correct"] == 1

    def test_record_outcome_sell_correct(self):
        """卖出预测 + 大跌 = 正确记录"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        assert de.record_outcome("SA-Agent", -1, -8.0) is True
        stats = de.get_accuracy_stats()
        assert stats["SA"]["correct"] == 1

    def test_record_outcome_invalid_agent(self):
        """非投票 Agent 应被跳过"""
        de = DecisionEngine()
        assert de.record_outcome("XX-Agent", 1, 10.0) is False

    def test_compute_dynamic_weights_with_history(self):
        """有足够历史后应产生权重调整"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        # 制造 TA 高准确率，FA 低准确率
        for _ in range(25):
            de.record_outcome("TA", 1, 8.0)
        for _ in range(25):
            de.record_outcome("FA", 1, -8.0)

        weights = de.compute_dynamic_weights(market_cycle="")
        assert weights["TA"] > weights["FA"]
        # 调整范围在 ±10% 内
        assert 0.05 <= weights["TA"] <= 0.50
        assert 0.05 <= weights["FA"] <= 0.50

    def test_compute_dynamic_weights_not_ready(self):
        """样本不足时不应调整"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        for _ in range(5):
            de.record_outcome("TA", 1, 8.0)

        weights = de.compute_dynamic_weights(market_cycle="")
        # 样本不足 20，应与基础权重一致（oscillation 方案 TA=0.25）
        base = de._get_base_weights("")
        assert abs(weights["TA"] - base["TA"]) < 0.01

    def test_accuracy_persistence(self):
        """准确率状态应持久化到 JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "test_accuracy.json"
            # 临时替换全局路径
            from agent.core import decision_engine as de_module
            orig_path = de_module.ACCURACY_STATE_PATH
            try:
                de_module.ACCURACY_STATE_PATH = state_path
                de = DecisionEngine()
                de.reset_accuracy_history()
                de.record_outcome("TA", 1, 8.0)
                del de

                # 重新实例化应恢复状态
                de2 = DecisionEngine()
                stats = de2.get_accuracy_stats()
                assert stats["TA"]["samples"] == 1
            finally:
                de_module.ACCURACY_STATE_PATH = orig_path

    def test_compute_weighted_decision_with_dynamic(self):
        """加权决策应集成动态权重"""
        de = DecisionEngine()
        de.reset_accuracy_history()
        opinions = {
            "TA-Agent": AgentOpinion(agent_id="TA-Agent", signal=1, confidence=0.8, reasoning="test"),
            "FA-Agent": AgentOpinion(agent_id="FA-Agent", signal=1, confidence=0.8, reasoning="test"),
            "CA-Agent": AgentOpinion(agent_id="CA-Agent", signal=-1, confidence=0.8, reasoning="test"),
            "SA-Agent": AgentOpinion(agent_id="SA-Agent", signal=0, confidence=0.6, reasoning="test"),
            "MA-Agent": AgentOpinion(agent_id="MA-Agent", signal=0, confidence=0.6, reasoning="test"),
            "RA-Agent": AgentOpinion(agent_id="RA-Agent", signal=0, confidence=0.7, reasoning="test",
                                      raw_data={"risk_level": 2, "max_position_pct": 0.2}),
        }
        result = de.compute_weighted_decision(opinions, use_dynamic_weights=True)
        assert "weights_used" in result
        assert "weights_source" in result
        assert result["weights_source"] == "dynamic"

    def test_reset_accuracy_history(self):
        """重置功能"""
        de = DecisionEngine()
        de.record_outcome("TA", 1, 8.0)
        de.reset_accuracy_history("TA")
        assert de.get_accuracy_stats()["TA"]["samples"] == 0
        de.record_outcome("FA", 1, 8.0)
        de.reset_accuracy_history()
        for agent in de.VOTING_AGENTS:
            assert de.get_accuracy_stats()[agent]["samples"] == 0


class TestConfidenceTransform:
    """测试非线性置信度变换"""

    def test_high_confidence_amplified(self):
        """高置信度应被放大"""
        de = DecisionEngine()
        # conf=0.81 → sqrt(0.81)=0.9，明显放大
        assert de.confidence_transform(0.81) == pytest.approx(0.9, abs=0.01)
        # conf=1.0 → 1.0
        assert de.confidence_transform(1.0) == 1.0

    def test_low_confidence_compressed(self):
        """低置信度应被压缩"""
        de = DecisionEngine()
        # conf=0.49 → 0.49^2=0.2401，大幅压缩
        assert de.confidence_transform(0.49) == pytest.approx(0.2401, abs=0.001)
        # conf=0.0 → 0.0
        assert de.confidence_transform(0.0) == 0.0

    def test_mid_confidence_unchanged(self):
        """中间置信度保持不变"""
        de = DecisionEngine()
        assert de.confidence_transform(0.6) == 0.6
        assert de.confidence_transform(0.75) == 0.75
        assert de.confidence_transform(0.5) == 0.25  # 边界：≤0.5 触发压缩

    def test_boundary_thresholds(self):
        """边界阈值测试"""
        de = DecisionEngine()
        from agent.core.decision_engine import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD
        # 刚好等于阈值
        assert de.confidence_transform(CONFIDENCE_HIGH_THRESHOLD) == pytest.approx(
            CONFIDENCE_HIGH_THRESHOLD ** 0.5, abs=0.001
        )
        assert de.confidence_transform(CONFIDENCE_LOW_THRESHOLD) == pytest.approx(
            CONFIDENCE_LOW_THRESHOLD ** 2.0, abs=0.001
        )

    def test_weighted_decision_with_nonlinear(self):
        """启用非线性置信度后，高置信度 Agent 影响力应提升"""
        de = DecisionEngine()
        de.reset_accuracy_history()

        opinions = {
            "TA-Agent": AgentOpinion(agent_id="TA-Agent", signal=1, confidence=0.85, reasoning="test"),
            "FA-Agent": AgentOpinion(agent_id="FA-Agent", signal=-1, confidence=0.45, reasoning="test"),
            "RA-Agent": AgentOpinion(agent_id="RA-Agent", signal=0, confidence=0.7, reasoning="test",
                                      raw_data={"risk_level": 2, "max_position_pct": 0.2}),
        }

        # 启用非线性
        result_on = de.compute_weighted_decision(opinions, use_nonlinear_confidence=True)
        # 关闭非线性
        result_off = de.compute_weighted_decision(opinions, use_nonlinear_confidence=False)

        # 高置信度 TA (0.85→~0.922) 应比低置信度 FA (0.45→0.2025) 影响力大得多
        # 所以启用非线性后，weighted_score 应该更偏向 TA 的方向（正值更大）
        assert result_on["weighted_score"] > result_off["weighted_score"]
        assert result_on["confidence_transform"]["enabled"] is True
        assert result_off["confidence_transform"]["enabled"] is False

    def test_weighted_decision_nonlinear_disabled_by_config(self):
        """全局关闭非线性置信度后，参数无法强制开启"""
        from agent.core import decision_engine as de_module
        orig = de_module.NONLINEAR_CONFIDENCE_ENABLED
        try:
            de_module.NONLINEAR_CONFIDENCE_ENABLED = False
            de = DecisionEngine()
            opinions = {
                "TA-Agent": AgentOpinion(agent_id="TA-Agent", signal=1, confidence=0.85, reasoning="test"),
                "RA-Agent": AgentOpinion(agent_id="RA-Agent", signal=0, confidence=0.7, reasoning="test",
                                          raw_data={"risk_level": 2}),
            }
            result = de.compute_weighted_decision(opinions, use_nonlinear_confidence=True)
            assert result["confidence_transform"]["enabled"] is False
        finally:
            de_module.NONLINEAR_CONFIDENCE_ENABLED = orig

    def test_votes_contain_transformed_weighted(self):
        """votes 列表应包含变换后的加权信号值"""
        de = DecisionEngine()
        opinions = {
            "TA-Agent": AgentOpinion(agent_id="TA-Agent", signal=1, confidence=0.81, reasoning="test"),
            "RA-Agent": AgentOpinion(agent_id="RA-Agent", signal=0, confidence=0.7, reasoning="test",
                                      raw_data={"risk_level": 2}),
        }
        result = de.compute_weighted_decision(opinions, use_nonlinear_confidence=True)
        ta_vote = [v for v in result["votes"] if v["agent"] == "TA-Agent"][0]
        # confidence 0.81 变换后 ≈ 0.9，再乘以 weight 和 signal
        # weight 约为 0.25，weighted ≈ 0.9 * 0.25 = 0.225
        assert ta_vote["weighted"] > 0.20  # 比线性时 (0.81*0.25=0.2025) 更大
