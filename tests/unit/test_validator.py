"""
单元测试: DecisionValidator — 回测闭环决策验证器

覆盖：决策记录、准确率分析、系统性偏差检测、Prompt 建议、回测验证
"""
import json
import pytest
from datetime import datetime

from agent.core.validator import (
    DecisionValidator,
    AgentPerformance,
    BiasReport,
    ValidationSummary,
    _BIAS_MIN_SAMPLES,
)


# ── 辅助：构造模拟决策包 ──

def _make_decision_package(stock_code="000001", opinions=None, final_signal=1, market_cycle=""):
    return {
        "stock_code": stock_code,
        "stock_name": "测试银行",
        "decision_date": "2024-01-15",
        "decision_time": "10:30:00",
        "market_cycle": market_cycle,
        "opinions": opinions or {
            "TA-Agent": {"signal": 1, "confidence": 0.8, "reasoning": "MA金叉"},
            "FA-Agent": {"signal": 1, "confidence": 0.7, "reasoning": "估值偏低"},
            "CA-Agent": {"signal": 0, "confidence": 0.5, "reasoning": "资金平稳"},
            "SA-Agent": {"signal": -1, "confidence": 0.6, "reasoning": "情绪过热"},
            "MA-Agent": {"signal": 1, "confidence": 0.65, "reasoning": "宏观偏多"},
            "RA-Agent": {"signal": 0, "confidence": 0.9, "reasoning": "风险可控"},
        },
        "final_decision": {"decision": final_signal, "confidence": 0.75},
        "data_summary": {},
        "risk_metrics": {},
        "indicators": {},
        "data_quality": {},
    }


def _make_records(agent_id, signals, actuals, confidences):
    """构造单 Agent 的决策记录列表"""
    return [
        {"signal": s, "confidence": c, "actual_return_pct": a}
        for s, a, c in zip(signals, actuals, confidences)
    ]


class TestDecisionValidatorBasics:
    """基础功能测试"""

    def test_init(self):
        v = DecisionValidator()
        assert v._history_path is not None
        assert v._bias_path is not None

    def test_record_and_load_decision(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "test_history.jsonl"
        v._bias_path = tmp_path / "test_bias.json"

        pkg = _make_decision_package()
        v._record_decision(pkg, actual_return_pct=12.5)

        decisions = v._load_decisions_for_stock("000001")
        assert len(decisions) == 1
        assert decisions[0]["stock_code"] == "000001"
        assert decisions[0]["actual_return_pct"] == 12.5
        assert "TA-Agent" in decisions[0]["opinions"]

    def test_load_recent_decisions(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "test_history.jsonl"
        v._bias_path = tmp_path / "test_bias.json"

        for i in range(5):
            pkg = _make_decision_package(stock_code=f"{i:06d}")
            v._record_decision(pkg, actual_return_pct=float(i))

        recent = v._load_recent_decisions(3)
        assert len(recent) == 3

    def test_reset_history(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "test_history.jsonl"
        v._bias_path = tmp_path / "test_bias.json"

        v._record_decision(_make_decision_package(), 5.0)
        v.reset_history()

        assert not v._history_path.exists()
        assert not v._bias_path.exists()


class TestAgentPerformance:
    """Agent 绩效计算测试"""

    def test_compute_single_agent_all_correct_buy(self):
        records = _make_records(
            "TA-Agent",
            signals=[1, 1, 1, 1, 1],
            actuals=[10.0, 8.0, 12.0, 6.0, 15.0],
            confidences=[0.8, 0.7, 0.9, 0.6, 0.85],
        )
        perf = DecisionValidator._compute_single_agent_performance("TA-Agent", records)
        assert perf.total_predictions == 5
        assert perf.correct_predictions == 5
        assert perf.accuracy == 1.0
        assert perf.buy_accuracy == 1.0
        assert perf.signal_distribution["buy"] == 5

    def test_compute_single_agent_mixed(self):
        records = _make_records(
            "FA-Agent",
            signals=[1, -1, 0, 1, -1, 0],
            actuals=[10.0, -8.0, 2.0, -3.0, -12.0, 1.0],
            confidences=[0.8, 0.7, 0.5, 0.6, 0.9, 0.5],
        )
        perf = DecisionValidator._compute_single_agent_performance("FA-Agent", records)
        # 买入>5%: 第1条 correct; 第4条 actual=-3% wrong
        # 卖出<-5%: 第2条 correct; 第5条 correct
        # 观望|收益|<5%: 第3条 |2%|<5 correct; 第6条 |1%|<5 correct
        assert perf.correct_predictions == 5
        assert perf.accuracy == pytest.approx(5 / 6, rel=1e-4)
        assert perf.buy_accuracy == 0.5
        assert perf.sell_accuracy == 1.0
        assert perf.hold_accuracy == 1.0

    def test_compute_single_agent_overconfidence(self):
        """错误时置信度更高 → 过度自信"""
        records = _make_records(
            "SA-Agent",
            signals=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            actuals=[-10.0, -8.0, -6.0, 12.0, -5.0, 15.0, -7.0, 8.0, -9.0, 10.0],
            confidences=[0.9, 0.85, 0.9, 0.5, 0.85, 0.6, 0.9, 0.5, 0.85, 0.6],
        )
        perf = DecisionValidator._compute_single_agent_performance("SA-Agent", records)
        # correct: 12, 15, 8, 10 → confidences [0.5, 0.6, 0.5, 0.6] avg=0.55
        # wrong: -10, -8, -6, -5, -7, -9 → confidences [0.9, 0.85, 0.9, 0.85, 0.9, 0.85] avg≈0.875
        assert perf.avg_confidence_when_wrong > perf.avg_confidence_when_correct

    def test_recent_trend_improving(self):
        records = _make_records(
            "TA-Agent",
            signals=[1]*10,
            actuals=[-10.0, -8.0, -6.0, -5.0, -7.0,  # 前5条全错
                     12.0, 15.0, 8.0, 10.0, 20.0],    # 后5条全对
            confidences=[0.7]*10,
        )
        perf = DecisionValidator._compute_single_agent_performance("TA-Agent", records)
        assert perf.recent_trend == "improving"

    def test_recent_trend_degrading(self):
        records = _make_records(
            "TA-Agent",
            signals=[1]*10,
            actuals=[12.0, 15.0, 8.0, 10.0, 20.0,  # 前5条全对
                     -10.0, -8.0, -6.0, -5.0, -7.0],  # 后5条全错
            confidences=[0.7]*10,
        )
        perf = DecisionValidator._compute_single_agent_performance("TA-Agent", records)
        assert perf.recent_trend == "degrading"


class TestSystematicBiasDetection:
    """系统性偏差检测测试"""

    def test_detect_low_accuracy_bias(self):
        """整体准确率过低 → high severity"""
        perfs = [
            AgentPerformance(
                agent_id="FA-Agent",
                total_predictions=20,
                correct_predictions=3,
                accuracy=0.15,
                signal_distribution={"buy": 10, "sell": 5, "hold": 5},
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        assert len(biases) >= 1
        assert biases[0].bias_type == "low_overall_accuracy"
        assert biases[0].severity == "high"
        assert "FA-Agent" in biases[0].description
        assert biases[0].weight_adjustment < 0

    def test_detect_overconfidence(self):
        """错误时置信度更高"""
        perfs = [
            AgentPerformance(
                agent_id="SA-Agent",
                total_predictions=20,
                correct_predictions=8,
                accuracy=0.40,
                avg_confidence_when_correct=0.55,
                avg_confidence_when_wrong=0.85,
                signal_distribution={"buy": 7, "sell": 7, "hold": 6},
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        overconf = [b for b in biases if b.bias_type == "overconfidence"]
        assert len(overconf) == 1
        assert overconf[0].severity == "medium"

    def test_detect_directional_bias(self):
        """买入和卖出准确率差异大"""
        perfs = [
            AgentPerformance(
                agent_id="TA-Agent",
                total_predictions=20,
                correct_predictions=10,
                accuracy=0.50,
                buy_accuracy=0.10,
                sell_accuracy=0.80,
                hold_accuracy=0.50,
                signal_distribution={"buy": 10, "sell": 5, "hold": 5},
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        directional = [b for b in biases if b.bias_type == "directional_bias"]
        assert len(directional) == 1
        assert "买入" in directional[0].description  # 买入准确率更低

    def test_detect_signal_imbalance(self):
        """信号分布严重偏斜"""
        perfs = [
            AgentPerformance(
                agent_id="MA-Agent",
                total_predictions=20,
                correct_predictions=10,
                accuracy=0.50,
                signal_distribution={"buy": 18, "sell": 1, "hold": 1},
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        imbalance = [b for b in biases if b.bias_type == "signal_imbalance"]
        assert len(imbalance) == 1
        assert "buy" in imbalance[0].description

    def test_no_bias_when_sufficient_and_good(self):
        """表现良好时不应报告偏差"""
        perfs = [
            AgentPerformance(
                agent_id="TA-Agent",
                total_predictions=20,
                correct_predictions=16,
                accuracy=0.80,
                avg_confidence_when_correct=0.80,
                avg_confidence_when_wrong=0.60,
                buy_accuracy=0.75,
                sell_accuracy=0.85,
                signal_distribution={"buy": 8, "sell": 6, "hold": 6},
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        assert len(biases) == 0

    def test_no_bias_when_insufficient_samples(self):
        """样本不足时不应报告偏差"""
        perfs = [
            AgentPerformance(
                agent_id="TA-Agent",
                total_predictions=3,
                correct_predictions=0,
                accuracy=0.0,
            )
        ]
        biases = DecisionValidator()._detect_systematic_bias(perfs)
        assert len(biases) == 0


class TestValidateAndLearn:
    """validate_and_learn 端到端测试"""

    def test_validate_and_learn_records_and_returns_summary(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "history.jsonl"
        v._bias_path = tmp_path / "bias.json"

        pkg = _make_decision_package()
        summary = v.validate_and_learn(pkg, actual_return_pct=12.5)

        assert isinstance(summary, ValidationSummary)
        assert summary.stock_code == "000001"
        assert summary.actual_return_pct == 12.5

        # 验证已写入文件
        decisions = v._load_decisions_for_stock("000001")
        assert len(decisions) == 1

    def test_validate_and_learn_updates_decision_engine(self, tmp_path):
        from agent.core.decision_engine import DecisionEngine

        v = DecisionValidator()
        v._history_path = tmp_path / "history.jsonl"
        v._bias_path = tmp_path / "bias.json"

        engine = DecisionEngine()
        before_stats = engine.get_accuracy_stats()

        pkg = _make_decision_package()
        summary = v.validate_and_learn(pkg, actual_return_pct=12.5, decision_engine=engine)

        after_stats = engine.get_accuracy_stats()
        # TA-Agent predicted signal=1, actual=12.5% > 5% → correct recorded
        assert after_stats["TA"]["samples"] == before_stats["TA"]["samples"] + 1

    def test_validate_and_learn_detects_bias(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "history.jsonl"
        v._bias_path = tmp_path / "bias.json"

        # 构造 20 条 FA-Agent 持续错误的记录
        for i in range(20):
            pkg = _make_decision_package(
                stock_code="000002",
                opinions={
                    "FA-Agent": {"signal": 1, "confidence": 0.8, "reasoning": "买入"},
                },
            )
            v._record_decision(pkg, actual_return_pct=-8.0)  # 实际大跌，买入错误

        # 再验证一次，触发检测
        summary = v.validate_and_learn(
            _make_decision_package(stock_code="000002"),
            actual_return_pct=-8.0,
        )

        # 此时应有系统性偏差报告
        fa_perfs = [p for p in summary.agent_performances if p.agent_id == "FA-Agent"]
        if fa_perfs and fa_perfs[0].total_predictions >= _BIAS_MIN_SAMPLES:
            assert len(summary.bias_reports) > 0


class TestValidateBacktest:
    """validate_backtest 测试"""

    def test_validate_backtest_insufficient_data(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "history.jsonl"
        v._bias_path = tmp_path / "bias.json"

        # 构造一个模拟的 BacktestResult
        from agent.core.backtest_engine import BacktestResult
        result = BacktestResult(
            stock_code="000001",
            strategy="multi_agent",
            strategy_desc="测试",
            start_date="2024-01-01",
            end_date="2024-01-30",
            initial_capital=100000.0,
            final_capital=105000.0,
            total_return_pct=5.0,
            buy_hold_return_pct=3.0,
            excess_return_pct=2.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=-2.0,
            win_rate=60.0,
            volatility_annual=15.0,
            trade_count=5,
            trades=[],
            equity_curve=[100000, 101000, 102000, 103000, 104000, 105000],
            dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        )

        report = v.validate_backtest(result)
        assert report["status"] == "insufficient_data"

    def test_validate_backtest_with_data(self, tmp_path):
        v = DecisionValidator()
        v._history_path = tmp_path / "history.jsonl"
        v._bias_path = tmp_path / "bias.json"

        # 预写入足够的历史决策
        for i in range(25):
            pkg = _make_decision_package(
                stock_code="000003",
                opinions={
                    "TA-Agent": {"signal": 1, "confidence": 0.8, "reasoning": "买入"},
                    "FA-Agent": {"signal": 1 if i < 20 else 0, "confidence": 0.7, "reasoning": "基本面"},
                },
            )
            # FA-Agent 前 20 次买入都错了（实际下跌），后 5 次观望
            actual = -8.0 if i < 20 else 2.0
            v._record_decision(pkg, actual_return_pct=actual)

        from agent.core.backtest_engine import BacktestResult
        result = BacktestResult(
            stock_code="000003",
            strategy="multi_agent",
            strategy_desc="测试",
            start_date="2024-01-01",
            end_date="2024-02-01",
            initial_capital=100000.0,
            final_capital=95000.0,
            total_return_pct=-5.0,
            buy_hold_return_pct=0.0,
            excess_return_pct=-5.0,
            sharpe_ratio=-0.5,
            max_drawdown_pct=-8.0,
            win_rate=30.0,
            volatility_annual=20.0,
            trade_count=10,
            trades=[],
            equity_curve=[100000, 99000, 98000, 97000, 96000, 95000],
            dates=["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-01"],
        )

        report = v.validate_backtest(result)
        assert report["status"] == "completed"
        assert report["total_decisions"] == 25
        assert len(report["agent_performances"]) > 0
        assert len(report["top_suggestions"]) > 0


class TestIsPredictionCorrect:
    """预测正确性判定规则测试"""

    def test_buy_correct(self):
        assert DecisionValidator._is_prediction_correct(1, 8.0) is True
        assert DecisionValidator._is_prediction_correct(1, 5.1) is True

    def test_buy_incorrect(self):
        assert DecisionValidator._is_prediction_correct(1, 3.0) is False
        assert DecisionValidator._is_prediction_correct(1, -5.0) is False

    def test_sell_correct(self):
        assert DecisionValidator._is_prediction_correct(-1, -8.0) is True
        assert DecisionValidator._is_prediction_correct(-1, -5.1) is True

    def test_sell_incorrect(self):
        assert DecisionValidator._is_prediction_correct(-1, -3.0) is False
        assert DecisionValidator._is_prediction_correct(-1, 5.0) is False

    def test_hold_correct(self):
        assert DecisionValidator._is_prediction_correct(0, 3.0) is True
        assert DecisionValidator._is_prediction_correct(0, -4.0) is True
        assert DecisionValidator._is_prediction_correct(0, 0.0) is True

    def test_hold_incorrect(self):
        assert DecisionValidator._is_prediction_correct(0, 6.0) is False
        assert DecisionValidator._is_prediction_correct(0, -6.0) is False
