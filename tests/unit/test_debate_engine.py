"""
DebateEngine 单元测试
覆盖：LLM 评估、规则降级、辩论全流程
"""
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from agent.core.debate import DebateEngine, DebateRound, DebateResult
from agent.core.blackboard import AgentOpinion


class MockLLM:
    """模拟 LLM 客户端"""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0

    def chat(self, system: str = "", user: str = "", json_mode: bool = False, **kwargs) -> Any:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


class TestDebateEngineRuleBased:
    """测试规则引擎降级方案"""

    def test_rule_based_challenger_wins(self):
        """质疑方获胜：回应中出现大量让步词"""
        engine = DebateEngine(llm_client=None)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="数据有问题",
                response="确实有道理，我承认忽略了风险",  # 多个让步词
            ),
        ]
        consensus, winner, delta = engine._rule_based_evaluate(rounds)
        assert winner == "TA-Agent"
        assert delta == pytest.approx(0.10)
        assert consensus is False

    def test_rule_based_responder_wins(self):
        """回应方获胜：两轮防守充分且无让步"""
        engine = DebateEngine(llm_client=None)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="一般性质疑",
                response="评估范围内，整体判断不变",  # 防守词，无让步
            ),
            DebateRound(
                round_number=2,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge=" another question",
                response="现有数据支持充分，判断清晰",  # 防守词
            ),
        ]
        consensus, winner, delta = engine._rule_based_evaluate(rounds)
        assert winner == "FA-Agent"
        assert delta == pytest.approx(0.08)
        assert consensus is True

    def test_rule_based_tie(self):
        """势均力敌"""
        engine = DebateEngine(llm_client=None)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="一般性质疑",
                response="一般性回应",
            ),
        ]
        consensus, winner, delta = engine._rule_based_evaluate(rounds)
        assert winner is None
        assert delta == pytest.approx(0.0)
        assert consensus is False

    def test_empty_rounds(self):
        """空轮次应返回平局"""
        engine = DebateEngine(llm_client=None)
        consensus, winner, delta = engine._evaluate_debate([])
        assert winner is None
        assert delta == 0.0
        assert consensus is False


class TestDebateEngineLLMEvaluate:
    """测试 LLM 逻辑评估"""

    def test_llm_challenger_wins(self):
        """LLM 判定质疑方获胜"""
        llm = MockLLM(responses=[{
            "challenger_score": 4,
            "responder_score": 2,
            "consensus_reached": False,
            "winner": "challenger",
            "reason": "质疑方击中了核心数据矛盾",
        }])
        engine = DebateEngine(llm_client=llm)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="CA-Agent",
                responder="TA-Agent",
                challenge="成交量与价格背离",
                response="短期波动正常",
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner == "CA-Agent"
        assert delta == pytest.approx(0.10)
        assert consensus is False
        assert llm.call_count == 1

    def test_llm_responder_wins(self):
        """LLM 判定回应方获胜"""
        llm = MockLLM(responses=[{
            "challenger_score": 2,
            "responder_score": 4,
            "consensus_reached": True,
            "winner": "responder",
            "reason": "回应方提供了详实的数据支撑",
        }])
        engine = DebateEngine(llm_client=llm)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="SA-Agent",
                responder="FA-Agent",
                challenge="情绪过于乐观",
                response="PE/PB 均处于历史 30% 分位，有数据支撑",
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner == "FA-Agent"
        assert delta == pytest.approx(0.08)
        assert consensus is True

    def test_llm_tie(self):
        """LLM 判定势均力敌"""
        llm = MockLLM(responses=[{
            "challenger_score": 3,
            "responder_score": 3,
            "consensus_reached": False,
            "winner": "",
            "reason": "双方论证各有优劣",
        }])
        engine = DebateEngine(llm_client=llm)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="MA-Agent",
                challenge="趋势不明",
                response="宏观确实复杂",
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner is None
        assert delta == pytest.approx(0.0)
        assert consensus is False

    def test_llm_response_is_string(self):
        """LLM 返回字符串时应自动解析 JSON"""
        import json
        llm = MockLLM(responses=[json.dumps({
            "challenger_score": 5,
            "responder_score": 1,
            "consensus_reached": False,
            "winner": "challenger",
            "reason": "压倒性优势",
        })])
        engine = DebateEngine(llm_client=llm)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="硬伤",
                response="无法反驳",
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner == "TA-Agent"

    def test_llm_failure_falls_back_to_rule(self):
        """LLM 调用异常时应降级到规则引擎"""
        class BadLLM:
            def chat(self, **kwargs):
                raise RuntimeError("LLM 服务不可用")

        engine = DebateEngine(llm_client=BadLLM())
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="数据有问题",
                response="确实有道理，我承认",  # 让步词 → challenger 赢
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner == "TA-Agent"
        assert delta == pytest.approx(0.10)

    def test_llm_none_falls_back_to_rule(self):
        """llm_client=None 时应直接使用规则引擎"""
        engine = DebateEngine(llm_client=None)
        rounds = [
            DebateRound(
                round_number=1,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="一般性质疑",
                response="评估范围内，整体判断不变",
            ),
            DebateRound(
                round_number=2,
                challenger="TA-Agent",
                responder="FA-Agent",
                challenge="又一次质疑",
                response="现有数据支持充分",
            ),
        ]
        consensus, winner, delta = engine._evaluate_debate(rounds)
        assert winner == "FA-Agent"


class TestDebateEngineIntegration:
    """集成测试：端到端辩论流程"""

    def test_detect_conflicts(self):
        """冲突检测"""
        engine = DebateEngine(llm_client=None)
        opinions = {
            "TA-Agent": AgentOpinion(
                agent_id="TA-Agent", signal=1, confidence=0.8,
                reasoning="看涨", key_factors=["金叉"], risk_flags=["波动大"],
            ),
            "FA-Agent": AgentOpinion(
                agent_id="FA-Agent", signal=-1, confidence=0.7,
                reasoning="估值高", key_factors=["PE高"], risk_flags=["业绩下滑"],
            ),
            "SA-Agent": AgentOpinion(
                agent_id="SA-Agent", signal=0, confidence=0.5,
                reasoning="观望", key_factors=[], risk_flags=["波动大"],
            ),
        }
        conflicts = engine.detect_conflicts(opinions)
        assert len(conflicts) >= 1
        # TA vs FA: signal_diff=2, conf_gap=0.1, risk_flags 差异 >=2
        ta_fa = [c for c in conflicts if {c["agent_a"], c["agent_b"]} == {"TA-Agent", "FA-Agent"}]
        assert len(ta_fa) == 1
        assert ta_fa[0]["severity"] >= 20  # signal_diff=2 * 10

    def test_run_debate_with_mock_llm(self):
        """完整辩论流程"""
        llm = MockLLM(responses=[
            "你的成交量数据有问题，未考虑大宗交易影响。",  # challenge
            "已考虑大宗交易，该因素在评估范围内。",        # response
            {  # evaluation
                "challenger_score": 2,
                "responder_score": 4,
                "consensus_reached": True,
                "winner": "responder",
                "reason": "回应方数据充分",
            },
        ])
        engine = DebateEngine(llm_client=llm)
        result = engine.run_debate(
            topic="TA vs FA",
            challenger_op=AgentOpinion(
                agent_id="TA-Agent", signal=1, confidence=0.6,
                reasoning="趋势向上", key_factors=["MA金叉"],
            ),
            responder_op=AgentOpinion(
                agent_id="FA-Agent", signal=-1, confidence=0.8,
                reasoning="估值偏高", key_factors=["PE过高"],
            ),
            context="000001 平安银行",
            max_rounds=1,
        )
        assert isinstance(result, DebateResult)
        assert result.topic == "TA vs FA"
        assert len(result.rounds) == 1
        assert result.winner == "FA-Agent"
        assert result.consensus_reached is True
        assert result.confidence_delta == pytest.approx(0.08)
        assert "FA-Agent" in result.summary

    def test_to_dict(self):
        """序列化"""
        engine = DebateEngine(llm_client=None)
        result = DebateResult(
            topic="test",
            rounds=[DebateRound(1, "A", "B", "c", "r")],
            summary="s",
            consensus_reached=True,
            winner="A",
            confidence_delta=0.1,
        )
        d = engine.to_dict(result)
        assert d["topic"] == "test"
        assert d["consensus_reached"] is True
        assert d["confidence_delta"] == 0.1
        assert "winner" not in d  # to_dict 故意省略 winner
        assert len(d["rounds"]) == 1
