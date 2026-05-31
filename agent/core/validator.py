"""
MASS 决策验证器 — 回测/实盘闭环优化 (DecisionValidator v1.0)

核心职责：
1. 记录完整决策历史（DecisionPackage + 实际收益）
2. 按 Agent / 市场周期 / 信号类型分析准确率
3. 检测系统性偏差（如 FA 在熊市持续高估、SA 过度逆向等）
4. 生成 Prompt 微调建议与权重调整建议
5. 自动回填 DecisionEngine 动态权重

数据持久化：
- decision_history.jsonl: 每条决策的完整记录（追加写，便于长期追踪）
- bias_reports.json: 系统性偏差报告（覆盖写，保留最新分析）

用法：
    validator = DecisionValidator()
    validator.validate_and_learn(decision_package, actual_return_pct=12.5)

    # 回测结束后批量验证
    validator.validate_backtest(backtest_result, decision_engine)
"""
import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger

from config import DATA_DIR


# ── 持久化路径 ──
DECISION_HISTORY_PATH = Path(DATA_DIR) / "decision_history.jsonl"
BIAS_REPORT_PATH = Path(DATA_DIR) / "bias_reports.json"

# ── 默认配置 ──
_BIAS_MIN_SAMPLES = int(os.getenv("VALIDATOR_BIAS_MIN_SAMPLES", "10"))
_BIAS_ACCURACY_THRESHOLD = float(os.getenv("VALIDATOR_BIAS_ACCURACY_THRESHOLD", "0.30"))
_BIAS_CONSECUTIVE_ERRORS = int(os.getenv("VALIDATOR_BIAS_CONSECUTIVE_ERRORS", "5"))


@dataclass
class AgentPerformance:
    """单个 Agent 在一段时期内的绩效统计"""
    agent_id: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    buy_accuracy: float = 0.0          # signal=1 时的准确率
    sell_accuracy: float = 0.0         # signal=-1 时的准确率
    hold_accuracy: float = 0.0         # signal=0 时的准确率
    signal_distribution: Dict[str, int] = field(default_factory=lambda: {"buy": 0, "sell": 0, "hold": 0})
    recent_trend: str = "insufficient"  # improving / stable / degrading / insufficient


@dataclass
class BiasReport:
    """系统性偏差报告"""
    agent_id: str
    bias_type: str                      # systematic_overvaluation / systematic_undervaluation / reverse_indicator / overconfidence / underconfidence
    severity: str                       # high / medium / low
    description: str
    affected_samples: int
    accuracy: float
    suggestion: str
    prompt_adjustment: str              # 具体的 Prompt 调整建议文本
    weight_adjustment: float            # 建议的权重调整值（如 -0.05）
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationSummary:
    """一次验证的完整摘要"""
    stock_code: str
    decision_date: str
    actual_return_pct: float
    agent_performances: List[AgentPerformance] = field(default_factory=list)
    bias_reports: List[BiasReport] = field(default_factory=list)
    weight_adjustments_applied: Dict[str, float] = field(default_factory=dict)
    prompt_suggestions: List[str] = field(default_factory=list)


class DecisionValidator:
    """
    决策验证器 — 连接回测/实盘结果与 Agent 优化的闭环系统
    """

    # 参与投票的 Agent（RA 不参与方向投票）
    VOTING_AGENTS = ["TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent"]

    def __init__(self):
        self._lock = threading.Lock()
        self._history_path = DECISION_HISTORY_PATH
        self._bias_path = BIAS_REPORT_PATH
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # 公共 API
    # ══════════════════════════════════════════════════════════════════════

    def validate_and_learn(
        self,
        decision_package: Dict[str, Any],
        actual_return_pct: float,
        decision_engine: Optional[Any] = None,
    ) -> ValidationSummary:
        """
        验证单次决策并学习

        Args:
            decision_package: Orchestrator 返回的 DecisionPackage dict
            actual_return_pct: 实际收益率（百分比，如 8.5 表示 +8.5%）
            decision_engine: 可选，用于直接更新动态权重

        Returns:
            ValidationSummary
        """
        stock_code = decision_package.get("stock_code", "")
        decision_date = decision_package.get("decision_date", "")
        opinions = decision_package.get("opinions", {})
        final_decision = decision_package.get("final_decision", {})
        market_cycle = decision_package.get("market_cycle", "")

        # 1. 记录决策
        self._record_decision(decision_package, actual_return_pct)

        # 2. 分析各 Agent 准确率
        performances = self._analyze_agent_accuracy(stock_code, limit=_BIAS_MIN_SAMPLES * 2)

        # 3. 检测系统性偏差
        biases = self._detect_systematic_bias(performances)

        # 4. 生成 Prompt 微调建议
        suggestions = []
        for bias in biases:
            sug = self._suggest_prompt_adjustment(bias)
            if sug:
                suggestions.append(sug)

        # 5. 更新决策引擎权重（如果提供了 engine）
        weight_adjustments = {}
        if decision_engine is not None:
            for agent_id, opinion_data in opinions.items():
                if agent_id == "RA-Agent":
                    continue
                signal = opinion_data.get("signal", 0)
                success = decision_engine.record_outcome(agent_id, signal, actual_return_pct)
                if success:
                    weight_adjustments[agent_id] = actual_return_pct

        summary = ValidationSummary(
            stock_code=stock_code,
            decision_date=decision_date,
            actual_return_pct=actual_return_pct,
            agent_performances=performances,
            bias_reports=biases,
            weight_adjustments_applied=weight_adjustments,
            prompt_suggestions=suggestions,
        )

        if biases:
            logger.warning(
                f"决策验证完成: {stock_code} 实际收益 {actual_return_pct:.2f}%, "
                f"检测到 {len(biases)} 项系统性偏差"
            )
        else:
            logger.info(f"决策验证完成: {stock_code} 实际收益 {actual_return_pct:.2f}%, 无系统性偏差")

        return summary

    def validate_backtest(
        self,
        backtest_result: Any,
        decision_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        回测结束后批量验证 — 分析整个回测期间的 Agent 表现

        Args:
            backtest_result: BacktestResult 对象
            decision_engine: 可选，用于更新动态权重

        Returns:
            验证报告 dict
        """
        stock_code = backtest_result.stock_code
        strategy = backtest_result.strategy

        # 读取该股票在回测区间内的所有历史决策
        decisions = self._load_decisions_for_stock(stock_code)
        if len(decisions) < _BIAS_MIN_SAMPLES:
            logger.info(f"回测验证: {stock_code} 历史决策不足 ({len(decisions)} < {_BIAS_MIN_SAMPLES})，跳过深度分析")
            return {"stock_code": stock_code, "status": "insufficient_data", "samples": len(decisions)}

        # 按 Agent 聚合绩效
        performances = self._compute_backtest_performances(decisions)

        # 检测偏差
        biases = self._detect_systematic_bias(performances)

        # 生成综合报告
        report = {
            "stock_code": stock_code,
            "strategy": strategy,
            "status": "completed",
            "total_decisions": len(decisions),
            "backtest_return_pct": backtest_result.total_return_pct,
            "agent_performances": [self._perf_to_dict(p) for p in performances],
            "bias_reports": [self._bias_to_dict(b) for b in biases],
            "top_suggestions": self._generate_top_suggestions(performances, biases),
            "generated_at": datetime.now().isoformat(),
        }

        # 持久化偏差报告
        self._save_bias_report(report)

        logger.info(
            f"回测验证完成: {stock_code} 策略={strategy} 收益={backtest_result.total_return_pct:.2f}% "
            f"样本={len(decisions)} 偏差={len(biases)}"
        )
        return report

    def get_agent_accuracy_report(self, agent_id: Optional[str] = None, n: int = 100) -> Dict[str, Any]:
        """
        获取 Agent 准确率报告（用于诊断 API）

        Args:
            agent_id: 指定 Agent，None 则返回全部
            n: 最近 N 条决策
        """
        decisions = self._load_recent_decisions(n)
        performances = self._compute_backtest_performances(decisions)

        if agent_id:
            performances = [p for p in performances if p.agent_id == agent_id]

        return {
            "sample_size": len(decisions),
            "performances": [self._perf_to_dict(p) for p in performances],
            "generated_at": datetime.now().isoformat(),
        }

    def reset_history(self) -> None:
        """清空决策历史（谨慎使用）"""
        with self._lock:
            if self._history_path.exists():
                self._history_path.unlink()
            if self._bias_path.exists():
                self._bias_path.unlink()
        logger.info("决策验证器历史已清空")

    # ══════════════════════════════════════════════════════════════════════
    # 内部：记录与加载
    # ══════════════════════════════════════════════════════════════════════

    def _record_decision(self, decision_package: Dict[str, Any], actual_return_pct: float) -> None:
        """追加写入决策历史（JSON Lines）"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "stock_code": decision_package.get("stock_code", ""),
            "decision_date": decision_package.get("decision_date", ""),
            "market_cycle": decision_package.get("market_cycle", ""),
            "actual_return_pct": round(actual_return_pct, 4),
            "opinions": {
                aid: {
                    "signal": op.get("signal", 0),
                    "confidence": op.get("confidence", 0.5),
                    "reasoning": op.get("reasoning", "")[:200],  # 截断避免过大
                }
                for aid, op in decision_package.get("opinions", {}).items()
            },
            "final_decision": {
                "decision": decision_package.get("final_decision", {}).get("decision", 0),
                "confidence": decision_package.get("final_decision", {}).get("confidence", 0.5),
            },
        }
        with self._lock:
            with open(self._history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_decisions_for_stock(self, stock_code: str) -> List[Dict[str, Any]]:
        """加载某股票的全部历史决策"""
        decisions = []
        if not self._history_path.exists():
            return decisions
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("stock_code") == stock_code:
                        decisions.append(record)
        except Exception as e:
            logger.warning(f"加载决策历史失败: {e}")
        return decisions

    def _load_recent_decisions(self, n: int) -> List[Dict[str, Any]]:
        """加载最近 N 条决策"""
        decisions = []
        if not self._history_path.exists():
            return decisions
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    line = line.strip()
                    if line:
                        decisions.append(json.loads(line))
        except Exception as e:
            logger.warning(f"加载决策历史失败: {e}")
        return decisions

    def _save_bias_report(self, report: Dict[str, Any]) -> None:
        """保存偏差报告"""
        try:
            with self._lock:
                with open(self._bias_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"偏差报告保存失败: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # 内部：准确率分析
    # ══════════════════════════════════════════════════════════════════════

    def _analyze_agent_accuracy(self, stock_code: str, limit: int = 50) -> List[AgentPerformance]:
        """分析某股票最近 limit 条决策中各 Agent 的准确率"""
        decisions = self._load_decisions_for_stock(stock_code)[-limit:]
        return self._compute_backtest_performances(decisions)

    def _compute_backtest_performances(self, decisions: List[Dict[str, Any]]) -> List[AgentPerformance]:
        """从决策列表计算各 Agent 的绩效统计"""
        # 按 Agent 聚合
        agent_records: Dict[str, List[Dict]] = defaultdict(list)
        for d in decisions:
            actual = d.get("actual_return_pct", 0)
            for aid, op in d.get("opinions", {}).items():
                if aid == "RA-Agent":
                    continue
                agent_records[aid].append({
                    "signal": op.get("signal", 0),
                    "confidence": op.get("confidence", 0.5),
                    "actual_return_pct": actual,
                })

        performances = []
        for aid, records in sorted(agent_records.items()):
            perf = self._compute_single_agent_performance(aid, records)
            performances.append(perf)
        return performances

    @staticmethod
    def _compute_single_agent_performance(agent_id: str, records: List[Dict]) -> AgentPerformance:
        """计算单个 Agent 的绩效"""
        total = len(records)
        if total == 0:
            return AgentPerformance(agent_id=agent_id)

        correct = 0
        conf_correct = []
        conf_wrong = []
        buy_total = buy_correct = 0
        sell_total = sell_correct = 0
        hold_total = hold_correct = 0
        sig_dist = {"buy": 0, "sell": 0, "hold": 0}

        for r in records:
            signal = r["signal"]
            actual = r["actual_return_pct"]
            conf = r["confidence"]

            is_correct = DecisionValidator._is_prediction_correct(signal, actual)
            if is_correct:
                correct += 1
                conf_correct.append(conf)
            else:
                conf_wrong.append(conf)

            if signal == 1:
                sig_dist["buy"] += 1
                buy_total += 1
                if is_correct:
                    buy_correct += 1
            elif signal == -1:
                sig_dist["sell"] += 1
                sell_total += 1
                if is_correct:
                    sell_correct += 1
            else:
                sig_dist["hold"] += 1
                hold_total += 1
                if is_correct:
                    hold_correct += 1

        accuracy = correct / total if total > 0 else 0

        # 近期趋势：最近 5 条 vs 前 5 条
        recent_trend = "insufficient"
        if total >= 10:
            recent = sum(1 for r in records[-5:] if DecisionValidator._is_prediction_correct(r["signal"], r["actual_return_pct"]))
            older = sum(1 for r in records[-10:-5] if DecisionValidator._is_prediction_correct(r["signal"], r["actual_return_pct"]))
            if recent > older:
                recent_trend = "improving"
            elif recent < older:
                recent_trend = "degrading"
            else:
                recent_trend = "stable"

        return AgentPerformance(
            agent_id=agent_id,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=round(accuracy, 4),
            avg_confidence_when_correct=round(sum(conf_correct) / len(conf_correct), 4) if conf_correct else 0,
            avg_confidence_when_wrong=round(sum(conf_wrong) / len(conf_wrong), 4) if conf_wrong else 0,
            buy_accuracy=round(buy_correct / buy_total, 4) if buy_total > 0 else 0,
            sell_accuracy=round(sell_correct / sell_total, 4) if sell_total > 0 else 0,
            hold_accuracy=round(hold_correct / hold_total, 4) if hold_total > 0 else 0,
            signal_distribution=sig_dist,
            recent_trend=recent_trend,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 内部：系统性偏差检测
    # ══════════════════════════════════════════════════════════════════════

    def _detect_systematic_bias(self, performances: List[AgentPerformance]) -> List[BiasReport]:
        """检测系统性偏差"""
        biases = []
        for perf in performances:
            if perf.total_predictions < _BIAS_MIN_SAMPLES:
                continue

            # 偏差 1: 整体准确率过低
            if perf.accuracy < _BIAS_ACCURACY_THRESHOLD:
                biases.append(BiasReport(
                    agent_id=perf.agent_id,
                    bias_type="low_overall_accuracy",
                    severity="high" if perf.accuracy < 0.20 else "medium",
                    description=f"{perf.agent_id} 近期准确率仅 {perf.accuracy:.1%}（{_BIAS_MIN_SAMPLES}+ 样本），显著低于随机水平",
                    affected_samples=perf.total_predictions,
                    accuracy=perf.accuracy,
                    suggestion="建议检查该 Agent 的分析框架是否存在系统性盲点",
                    prompt_adjustment=f"在 {perf.agent_id} 的 system prompt 中增加 '质疑自身假设' 的环节，要求每次分析前主动列出 3 个可能推翻当前结论的反证。",
                    weight_adjustment=round(max(-0.08, -(0.5 - perf.accuracy) * 0.2), 4),
                ))

            # 偏差 2: 高置信度时反而更错（过度自信）
            if (perf.avg_confidence_when_correct < perf.avg_confidence_when_wrong
                    and perf.avg_confidence_when_wrong > 0.75):
                biases.append(BiasReport(
                    agent_id=perf.agent_id,
                    bias_type="overconfidence",
                    severity="medium",
                    description=f"{perf.agent_id} 错误预测时的平均置信度({perf.avg_confidence_when_wrong:.2f})高于正确时({perf.avg_confidence_when_correct:.2f})，存在过度自信",
                    affected_samples=perf.total_predictions,
                    accuracy=perf.accuracy,
                    suggestion="建议降低该 Agent 的置信度校准，或增加 '置信度必须匹配证据强度' 的约束",
                    prompt_adjustment=f"修改 {perf.agent_id} 的 system prompt：将 confidence 的判定标准收紧，要求 '只有当多个独立证据源一致支持时才允许 confidence ≥ 0.8'。",
                    weight_adjustment=-0.03,
                ))

            # 偏差 3: 买入准确率显著低于卖出（或反之）
            if perf.buy_accuracy > 0 and perf.sell_accuracy > 0:
                if abs(perf.buy_accuracy - perf.sell_accuracy) > 0.30:
                    direction = "买入" if perf.buy_accuracy < perf.sell_accuracy else "卖出"
                    biases.append(BiasReport(
                        agent_id=perf.agent_id,
                        bias_type="directional_bias",
                        severity="medium",
                        description=f"{perf.agent_id} 在 {direction} 方向的准确率显著偏低（买{perf.buy_accuracy:.0%} vs 卖{perf.sell_accuracy:.0%}）",
                        affected_samples=perf.total_predictions,
                        accuracy=perf.accuracy,
                        suggestion=f"检查该 Agent 在 {direction} 信号上的判断逻辑是否存在惯性偏差",
                        prompt_adjustment=f"在 {perf.agent_id} 的 system prompt 中增加对称性检验：要求对 '买入' 和 '卖出' 信号使用同等严格的证据标准。",
                        weight_adjustment=-0.02,
                    ))

            # 偏差 4: 持续恶化趋势
            if perf.recent_trend == "degrading" and perf.accuracy < 0.40:
                biases.append(BiasReport(
                    agent_id=perf.agent_id,
                    bias_type="degrading_performance",
                    severity="high",
                    description=f"{perf.agent_id} 准确率呈持续下降趋势，最近表现({perf.accuracy:.1%})显著恶化",
                    affected_samples=perf.total_predictions,
                    accuracy=perf.accuracy,
                    suggestion="市场环境可能发生变化，建议重新校准该 Agent 的分析参数",
                    prompt_adjustment=f"在 {perf.agent_id} 的 system prompt 开头增加市场适应性声明：'当前市场环境可能已发生变化，请优先使用最近 20 个交易日的数据做判断。'",
                    weight_adjustment=-0.05,
                ))

            # 偏差 5: 信号分布极端偏斜（如永远看多/看空）
            total_sig = sum(perf.signal_distribution.values())
            if total_sig > 0:
                max_ratio = max(v / total_sig for v in perf.signal_distribution.values())
                if max_ratio > 0.80:
                    dominant = max(perf.signal_distribution, key=perf.signal_distribution.get)
                    biases.append(BiasReport(
                        agent_id=perf.agent_id,
                        bias_type="signal_imbalance",
                        severity="medium",
                        description=f"{perf.agent_id} 信号分布严重偏斜：{dominant} 占比 {max_ratio:.1%}，缺乏多空平衡视角",
                        affected_samples=perf.total_predictions,
                        accuracy=perf.accuracy,
                        suggestion="该 Agent 可能存在确认偏误，只关注支持其固有观点的证据",
                        prompt_adjustment=f"在 {perf.agent_id} 的 system prompt 中强制要求：'无论初步判断如何，必须列出至少 1 个反向论据，且反向论据的权重不得低于总权重的 30%'。",
                        weight_adjustment=-0.03,
                    ))

        return biases

    def _suggest_prompt_adjustment(self, bias: BiasReport) -> str:
        """根据偏差生成 Prompt 微调建议"""
        return (
            f"【{bias.agent_id} | {bias.bias_type} | 严重度:{bias.severity}】\n"
            f"问题: {bias.description}\n"
            f"建议: {bias.suggestion}\n"
            f"Prompt 调整: {bias.prompt_adjustment}\n"
            f"权重调整: {bias.weight_adjustment:+.2f}"
        )

    def _generate_top_suggestions(self, performances: List[AgentPerformance], biases: List[BiasReport]) -> List[str]:
        """生成优先级排序的综合建议"""
        suggestions = []

        # 按严重度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_biases = sorted(biases, key=lambda b: severity_order.get(b.severity, 3))

        for bias in sorted_biases[:5]:  # 最多 5 条
            suggestions.append(self._suggest_prompt_adjustment(bias))

        # 补充：表现最好的 Agent 建议增加权重
        best = max(performances, key=lambda p: p.accuracy, default=None)
        if best and best.accuracy > 0.65 and best.total_predictions >= _BIAS_MIN_SAMPLES:
            suggestions.append(
                f"【{best.agent_id} 表现优异】近期准确率 {best.accuracy:.1%}，建议权重 +0.03~0.05"
            )

        return suggestions

    # ══════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_prediction_correct(signal: int, actual_return_pct: float) -> bool:
        """判断预测是否正确（与 DecisionEngine 保持一致）"""
        if signal == 1 and actual_return_pct > 5:
            return True
        if signal == -1 and actual_return_pct < -5:
            return True
        if signal == 0 and abs(actual_return_pct) < 5:
            return True
        return False

    @staticmethod
    def _perf_to_dict(perf: AgentPerformance) -> Dict[str, Any]:
        return {
            "agent_id": perf.agent_id,
            "total_predictions": perf.total_predictions,
            "correct_predictions": perf.correct_predictions,
            "accuracy": perf.accuracy,
            "avg_confidence_when_correct": perf.avg_confidence_when_correct,
            "avg_confidence_when_wrong": perf.avg_confidence_when_wrong,
            "buy_accuracy": perf.buy_accuracy,
            "sell_accuracy": perf.sell_accuracy,
            "hold_accuracy": perf.hold_accuracy,
            "signal_distribution": perf.signal_distribution,
            "recent_trend": perf.recent_trend,
        }

    @staticmethod
    def _bias_to_dict(bias: BiasReport) -> Dict[str, Any]:
        return asdict(bias)
