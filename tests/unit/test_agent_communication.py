"""
Agent 间通信引擎单元测试
覆盖两轮分析、冲突检测、修正触发、通信摘要构建
"""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from agent.core.communication import AgentCommunicationEngine
from agent.core.blackboard import AgentOpinion, StockSnapshot


class TestAgentCommunicationEngine(unittest.TestCase):
    """Agent 间通信引擎测试类"""

    def _make_opinion(
        self,
        agent_id="TA-Agent",
        signal=0,
        confidence=0.7,
        reasoning="测试理由",
        key_factors=None,
        risk_flags=None,
        raw_data=None,
    ):
        return AgentOpinion(
            agent_id=agent_id,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors or [],
            risk_flags=risk_flags or [],
            raw_data=raw_data or {},
        )

    def _make_engine(self):
        """构造带 Mock Orchestrator 的通信引擎"""
        mock_orch = MagicMock()
        mock_orch.blackboard = MagicMock()
        mock_orch.agents = {}
        return AgentCommunicationEngine(mock_orch)

    # ── 通信摘要构建 ──
    def test_build_communication_summary_basic(self):
        """通信摘要包含所有Agent的关键信息"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8, "多头排列", ["MACD金叉"], ["波动率高"]),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75, "估值合理", ["PE低"], []),
            "RA-Agent": self._make_opinion("RA-Agent", 0, 0.6, "风险可控", [], ["波动率偏高"]),
        }
        summary = engine._build_communication_summary(opinions)
        self.assertIn("TA-Agent", summary)
        self.assertIn("买入", summary)
        self.assertIn("MACD金叉", summary)
        self.assertIn("整体信号统计", summary)

    def test_build_communication_summary_with_metrics(self):
        """通信摘要包含量化指标"""
        engine = self._make_engine()
        opinions = {
            "FA-Agent": self._make_opinion(
                "FA-Agent", 1, 0.8, "基本面好",
                raw_data={"fundamental_score": 85, "valuation_gap": "低估15%"}
            ),
        }
        summary = engine._build_communication_summary(opinions)
        self.assertIn("基本面评分: 85", summary)

    # ── 冲突检测 ──
    def test_detect_conflicts_no_conflict(self):
        """观点一致时不应检测到冲突"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", 1, 0.7),
            "RA-Agent": self._make_opinion("RA-Agent", 0, 0.6),
        }
        conflicts = engine._detect_conflicts(opinions)
        self.assertEqual(len(conflicts), 0)

    def test_detect_conflicts_majority_sell_one_buy(self):
        """多数卖出，单个买入 → 买入Agent应被标记"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", -1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", 1, 0.7),   # 显著分歧
            "RA-Agent": self._make_opinion("RA-Agent", 0, 0.6),
        }
        conflicts = engine._detect_conflicts(opinions)
        self.assertIn("CA-Agent", conflicts)

    def test_detect_conflicts_low_confidence(self):
        """置信度低于阈值应触发修正"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
            "SA-Agent": self._make_opinion("SA-Agent", 1, 0.35),  # 置信度低
        }
        conflicts = engine._detect_conflicts(opinions)
        self.assertIn("SA-Agent", conflicts)

    def test_detect_conflicts_key_agent_isolated(self):
        """关键Agent（TA/FA/CA）被2个以上反对时应触发修正"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),   # 关键Agent，孤立
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", -1, 0.7),
            "SA-Agent": self._make_opinion("SA-Agent", -1, 0.6),
        }
        conflicts = engine._detect_conflicts(opinions)
        self.assertIn("TA-Agent", conflicts)

    def test_detect_conflicts_max_revisions(self):
        """冲突Agent超过MAX_REVISIONS时只修正置信度最低的"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),   # 与多数相反
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", -1, 0.7),
            "SA-Agent": self._make_opinion("SA-Agent", -1, 0.6),
            "MA-Agent": self._make_opinion("MA-Agent", -1, 0.5),  # 也与多数相反，但置信度最低
        }
        # 多数 = -1 (4票), TA=1 是少数
        # 只有 TA-Agent 被标记，因为只有一个Agent与多数相反
        conflicts = engine._detect_conflicts(opinions)
        self.assertLessEqual(len(conflicts), engine.MAX_REVISIONS)

    def test_detect_conflicts_excludes_ra_agent(self):
        """RA-Agent 不应被标记为冲突"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
            "RA-Agent": self._make_opinion("RA-Agent", 0, 0.3),  # 置信度低但不应触发
        }
        conflicts = engine._detect_conflicts(opinions)
        self.assertNotIn("RA-Agent", conflicts)

    # ── 修正触发判断 ──
    def test_needs_revision_signal_divergence(self):
        """信号差值>=2时触发修正"""
        engine = self._make_engine()
        op = self._make_opinion("TA-Agent", 1, 0.8)
        self.assertTrue(engine._needs_revision("TA-Agent", op, -1, {}))

    def test_needs_revision_no_divergence(self):
        """信号差值<2时不触发"""
        engine = self._make_engine()
        op = self._make_opinion("TA-Agent", 1, 0.8)
        self.assertFalse(engine._needs_revision("TA-Agent", op, 0, {}))

    def test_needs_revision_low_confidence(self):
        """置信度低触发修正"""
        engine = self._make_engine()
        op = self._make_opinion("TA-Agent", 1, 0.4)
        self.assertTrue(engine._needs_revision("TA-Agent", op, 1, {}))

    def test_needs_revision_key_agent_isolated(self):
        """关键Agent被孤立触发修正"""
        engine = self._make_engine()
        op = self._make_opinion("TA-Agent", 1, 0.8)
        all_ops = {
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.8),
            "CA-Agent": self._make_opinion("CA-Agent", -1, 0.8),
        }
        self.assertTrue(engine._needs_revision("TA-Agent", op, -1, all_ops))

    # ── 分歧报告 ──
    def test_conflict_report_no_conflict(self):
        """无冲突时报告正确"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
        }
        report = engine.get_conflict_report(opinions)
        self.assertFalse(report["has_conflict"])
        self.assertEqual(report["majority_signal"], 1)

    def test_conflict_report_with_conflict(self):
        """有冲突时报告正确"""
        engine = self._make_engine()
        opinions = {
            "TA-Agent": self._make_opinion("TA-Agent", -1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", 1, 0.7),
        }
        report = engine.get_conflict_report(opinions)
        self.assertTrue(report["has_conflict"])
        self.assertEqual(report["majority_signal"], -1)
        self.assertIn("CA-Agent", report["conflict_agents"])

    # ── 两轮分析流程 ──
    def test_run_two_round_no_conflict(self):
        """无冲突时直接返回Round1结果"""
        engine = self._make_engine()
        round1 = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
        }
        engine.orchestrator._run_agents_parallel.return_value = round1

        result = engine.run_two_round_analysis("000001")
        self.assertEqual(result["TA-Agent"].signal, 1)
        self.assertEqual(result["FA-Agent"].signal, 1)
        # 不应有修正
        self.assertFalse(result["TA-Agent"].is_revision)

    def test_run_two_round_with_revision(self):
        """有冲突时执行修正分析"""
        engine = self._make_engine()
        round1 = {
            "TA-Agent": self._make_opinion("TA-Agent", -1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", -1, 0.75),
            "CA-Agent": self._make_opinion("CA-Agent", 1, 0.7),  # 与多数相反
        }
        engine.orchestrator._run_agents_parallel.return_value = round1

        # Mock Agent revise 方法
        mock_agent = MagicMock()
        revised_op = self._make_opinion("CA-Agent", -1, 0.65, "修正后")
        revised_op.is_revision = True
        revised_op.original_signal = 1
        mock_agent.revise.return_value = revised_op
        engine.orchestrator.agents["CA-Agent"] = mock_agent

        result = engine.run_two_round_analysis("000001")
        self.assertEqual(result["CA-Agent"].signal, -1)
        self.assertTrue(result["CA-Agent"].is_revision)
        self.assertEqual(result["CA-Agent"].original_signal, 1)
        mock_agent.revise.assert_called_once()

    def test_run_two_round_revision_failure_fallback(self):
        """修正失败时回退到原始结论"""
        engine = self._make_engine()
        original = self._make_opinion("CA-Agent", 1, 0.7)
        round1 = {
            "TA-Agent": self._make_opinion("TA-Agent", -1, 0.8),
            "CA-Agent": original,
        }
        engine.orchestrator._run_agents_parallel.return_value = round1

        mock_agent = MagicMock()
        mock_agent.revise.side_effect = RuntimeError("LLM失败")
        engine.orchestrator.agents["CA-Agent"] = mock_agent

        result = engine.run_two_round_analysis("000001")
        self.assertEqual(result["CA-Agent"].signal, 1)  # 原始信号
        self.assertFalse(result["CA-Agent"].is_revision)

    def test_run_two_round_progress_callback(self):
        """进度回调被正确调用"""
        engine = self._make_engine()
        round1 = {
            "TA-Agent": self._make_opinion("TA-Agent", 1, 0.8),
            "FA-Agent": self._make_opinion("FA-Agent", 1, 0.75),
        }
        engine.orchestrator._run_agents_parallel.return_value = round1

        progress_calls = []
        def progress_cb(stage, msg, progress, **kwargs):
            progress_calls.append((stage, msg, progress))

        engine.run_two_round_analysis("000001", progress_cb=progress_cb)
        self.assertTrue(any("Round 1" in msg for _, msg, _ in progress_calls))


if __name__ == "__main__":
    unittest.main()
