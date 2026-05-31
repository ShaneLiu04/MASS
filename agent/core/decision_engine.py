"""
MASS 决策引擎 v2.1
负责加权投票 + 风险过滤 + 动态权重调整 + 动态权重学习

优化：
- 引入动态权重学习：基于 Agent 历史准确率自动调整权重
- 权重持久化：进程重启不丢失学习结果
- 数据闭环：支持外部系统（回测/实盘）回填预测结果
"""
import json
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.core.blackboard import AgentOpinion
from config import (
    WEIGHT_MAP, CYCLE_WEIGHT_MAP, DATA_DIR,
    NONLINEAR_CONFIDENCE_ENABLED,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_HIGH_EXPONENT, CONFIDENCE_LOW_EXPONENT,
)


# ── 动态权重学习配置 ──
ACCURACY_HISTORY_MAXLEN = 100          # 每个 Agent 最多保存多少条历史记录
ACCURACY_MIN_SAMPLES = 20              # 最少需要多少条记录才启用动态调整
ACCURACY_ADJUSTMENT_RANGE = 0.10       # 单 Agent 最大调整幅度 ±10%
ACCURACY_ADJUSTMENT_FACTOR = 0.20      # 准确率差值乘数
DYNAMIC_WEIGHT_ENABLED = os.getenv("DYNAMIC_WEIGHT_ENABLED", "True").lower() == "true"
ACCURACY_STATE_PATH = Path(DATA_DIR) / "agent_accuracy_state.json"


@dataclass
class WeightedVote:
    """加权投票结果"""
    agent_id: str
    signal: int
    confidence: float
    weight: float
    weighted_signal: float


class DecisionEngine:
    """
    决策引擎 v2.1
    1. 根据宏观周期选择基础权重方案
    2. 应用 MA-Agent 的微调
    3. 结合历史准确率进行动态权重学习
    4. 计算加权信号
    5. 应用 RA-Agent 的风险过滤
    """

    # 参与投票的 Agent 前缀（RA-Agent 只用于风险过滤）
    VOTING_AGENTS = ["TA", "FA", "CA", "SA", "MA"]

    def __init__(self):
        self.default_weights = {
            "TA": 0.20, "FA": 0.20, "CA": 0.20,
            "SA": 0.15, "MA": 0.15, "RA": 0.10,
        }
        # ── 动态权重学习：Agent 历史准确率追踪 ──
        self.agent_accuracy_history: Dict[str, deque] = {
            agent: deque(maxlen=ACCURACY_HISTORY_MAXLEN)
            for agent in self.VOTING_AGENTS
        }
        self._accuracy_lock = threading.Lock()
        self._load_accuracy_state()

    # ══════════════════════════════════════════════════════════════════════
    # 动态权重学习：记录 & 计算
    # ══════════════════════════════════════════════════════════════════════

    def record_outcome(
        self,
        agent_id: str,
        predicted_signal: int,
        actual_return_pct: float,
    ) -> bool:
        """
        记录 Agent 预测结果与实际收益的对比，用于后续权重调整。

        判定规则：
        - signal=1（买入）时，actual_return_pct > +5%  算正确
        - signal=-1（卖出）时，actual_return_pct < -5% 算正确
        - signal=0（观望）时，|actual_return_pct| < 5% 算正确

        Args:
            agent_id: Agent 标识，如 "TA-Agent" 或 "TA"
            predicted_signal: 预测信号 (-1, 0, 1)
            actual_return_pct: 实际收益率（百分比，如 8.5 表示 +8.5%）

        Returns:
            是否成功记录
        """
        agent_key = self._normalize_agent_key(agent_id)
        if agent_key not in self.agent_accuracy_history:
            logger.debug(f"Agent {agent_id} 不在投票 Agent 列表中，跳过记录")
            return False

        correct = self._is_prediction_correct(predicted_signal, actual_return_pct)

        with self._accuracy_lock:
            self.agent_accuracy_history[agent_key].append(1 if correct else 0)
            self._save_accuracy_state()

        logger.info(
            f"记录 Agent 预测结果: {agent_key} | "
            f"predicted={predicted_signal}, actual={actual_return_pct:.2f}%, "
            f"correct={correct}, history_len={len(self.agent_accuracy_history[agent_key])}"
        )
        return True

    def compute_dynamic_weights(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        market_cycle: str = "",
    ) -> Dict[str, float]:
        """
        计算动态权重（结合历史准确率）。

        逻辑：
        1. 获取基础权重（周期权重或 default）
        2. 对历史记录 >= ACCURACY_MIN_SAMPLES 的 Agent，
           计算其准确率，与 0.5 基准比较，进行 ±ACCURACY_ADJUSTMENT_RANGE 的调整
        3. 所有权重 clamp 到 [0.05, 0.50]
        4. 归一化到总和为 1.0

        Args:
            base_weights: 可传入外部基础权重；None 则使用周期权重
            market_cycle: 市场周期标签，用于选择基础权重

        Returns:
            调整后的动态权重字典
        """
        weights = dict(base_weights) if base_weights else self._get_base_weights(market_cycle)

        with self._accuracy_lock:
            adjustments: Dict[str, float] = {}
            for agent, history in self.agent_accuracy_history.items():
                if len(history) >= ACCURACY_MIN_SAMPLES:
                    accuracy = sum(history) / len(history)
                    # 准确率高于 0.5 基准的 Agent 增加权重，低于则减少
                    adj = (accuracy - 0.5) * ACCURACY_ADJUSTMENT_FACTOR * 2
                    adj = max(-ACCURACY_ADJUSTMENT_RANGE, min(ACCURACY_ADJUSTMENT_RANGE, adj))
                    adjustments[agent] = adj
                else:
                    adjustments[agent] = 0.0

        # 应用调整并 clamp
        adjusted: Dict[str, float] = {}
        for agent, weight in weights.items():
            adj = adjustments.get(agent, 0.0)
            adjusted[agent] = max(0.05, min(0.50, weight + adj))

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

        logger.debug(f"动态权重计算完成: base={weights}, adjusted={adjusted}")
        return adjusted

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """获取各 Agent 当前准确率统计（用于诊断/监控）"""
        stats = {}
        with self._accuracy_lock:
            for agent, history in self.agent_accuracy_history.items():
                n = len(history)
                stats[agent] = {
                    "samples": n,
                    "correct": int(sum(history)),
                    "accuracy": round(sum(history) / n, 4) if n > 0 else None,
                    "ready": n >= ACCURACY_MIN_SAMPLES,
                }
        return stats

    def reset_accuracy_history(self, agent_id: Optional[str] = None) -> None:
        """重置准确率历史（用于测试或手动重置）"""
        with self._accuracy_lock:
            if agent_id:
                key = self._normalize_agent_key(agent_id)
                if key in self.agent_accuracy_history:
                    self.agent_accuracy_history[key].clear()
            else:
                for history in self.agent_accuracy_history.values():
                    history.clear()
            self._save_accuracy_state()
        logger.info(f"准确率历史已重置: agent={agent_id or 'ALL'}")

    # ── 内部辅助 ──

    @staticmethod
    def _normalize_agent_key(agent_id: str) -> str:
        """统一 Agent ID 格式：TA-Agent → TA"""
        return agent_id.replace("-Agent", "").strip()

    @staticmethod
    def _is_prediction_correct(predicted_signal: int, actual_return_pct: float) -> bool:
        """判断单条预测是否正确"""
        if predicted_signal == 1 and actual_return_pct > 5:
            return True
        if predicted_signal == -1 and actual_return_pct < -5:
            return True
        if predicted_signal == 0 and abs(actual_return_pct) < 5:
            return True
        return False

    def _save_accuracy_state(self) -> None:
        """将准确率历史持久化到 JSON 文件（进程重启不丢失）"""
        try:
            state = {
                agent: list(history)
                for agent, history in self.agent_accuracy_history.items()
            }
            ACCURACY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ACCURACY_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"准确率状态保存失败: {e}")

    def _load_accuracy_state(self) -> None:
        """从 JSON 文件恢复准确率历史"""
        if not ACCURACY_STATE_PATH.exists():
            return
        try:
            with open(ACCURACY_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            for agent, history in state.items():
                if agent in self.agent_accuracy_history and isinstance(history, list):
                    self.agent_accuracy_history[agent].extend(history)
            logger.info(f"准确率状态已恢复: {ACCURACY_STATE_PATH}")
        except Exception as e:
            logger.warning(f"准确率状态恢复失败: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # 加权决策
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def confidence_transform(conf: float) -> float:
        """
        非线性置信度变换

        - 高置信度 (≥0.8): conf^0.5 → 进一步放大（如 0.81 → 0.90）
        - 低置信度 (≤0.5): conf^2   → 大幅压缩（如 0.49 → 0.24）
        - 中间区间: 保持不变
        """
        if conf >= CONFIDENCE_HIGH_THRESHOLD:
            return conf ** CONFIDENCE_HIGH_EXPONENT
        elif conf <= CONFIDENCE_LOW_THRESHOLD:
            return conf ** CONFIDENCE_LOW_EXPONENT
        return conf

    def compute_weighted_decision(
        self,
        opinions: Dict[str, AgentOpinion],
        market_cycle: str = "",
        weight_adjustment: Optional[Dict[str, float]] = None,
        use_dynamic_weights: bool = True,
        use_nonlinear_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        计算加权决策

        Args:
            opinions: {agent_id: AgentOpinion}
            market_cycle: 市场周期标签
            weight_adjustment: MA-Agent 建议的权重微调
            use_dynamic_weights: 是否启用动态权重学习（默认 True）
            use_nonlinear_confidence: 是否启用非线性置信度加权（默认 True）

        Returns:
            加权决策结果
        """
        # 1. 获取基础权重
        base_weights = self._get_base_weights(market_cycle)

        # 2. 应用动态权重学习（若启用且数据足够）
        weights = dict(base_weights)
        if DYNAMIC_WEIGHT_ENABLED and use_dynamic_weights:
            try:
                dynamic_weights = self.compute_dynamic_weights(base_weights, market_cycle)
                weights = dynamic_weights
            except Exception as e:
                logger.warning(f"动态权重计算失败，回退到基础权重: {e}")

        # 3. 应用 MA-Agent 微调
        if weight_adjustment:
            for agent, delta in weight_adjustment.items():
                if agent in weights:
                    weights[agent] = max(0.05, min(0.50, weights[agent] + delta))

        # 4. 再次归一化（动态权重 + MA 微调后总和可能不为 1）
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        # 5. 计算加权投票（含非线性置信度变换）
        votes = []
        weighted_sum = 0.0
        confidence_sum = 0.0
        nonlinear_enabled = NONLINEAR_CONFIDENCE_ENABLED and use_nonlinear_confidence

        for agent_id, opinion in opinions.items():
            if agent_id == "RA-Agent":
                continue  # RA-Agent 只用于风险过滤，不参与投票

            agent_key = self._normalize_agent_key(agent_id)
            weight = weights.get(agent_key, 0.15)

            # 非线性置信度变换
            raw_conf = opinion.confidence
            if nonlinear_enabled:
                transformed_conf = self.confidence_transform(raw_conf)
            else:
                transformed_conf = raw_conf

            w_signal = opinion.signal * transformed_conf * weight
            weighted_sum += w_signal
            confidence_sum += transformed_conf * weight

            votes.append(WeightedVote(
                agent_id=agent_id,
                signal=opinion.signal,
                confidence=round(raw_conf, 3),
                weight=round(weight, 4),
                weighted_signal=round(w_signal, 4),
            ))

        # 6. 确定初步信号
        if weighted_sum > 0.15:
            preliminary_signal = 1
        elif weighted_sum < -0.15:
            preliminary_signal = -1
        else:
            preliminary_signal = 0

        # 7. 综合置信度
        overall_confidence = min(confidence_sum, 0.95)

        return {
            "preliminary_signal": preliminary_signal,
            "weighted_score": round(weighted_sum, 4),
            "overall_confidence": round(overall_confidence, 3),
            "votes": [
                {
                    "agent": v.agent_id,
                    "signal": v.signal,
                    "confidence": v.confidence,
                    "weight": v.weight,
                    "weighted": v.weighted_signal,
                }
                for v in votes
            ],
            "weights_used": weights,
            "weights_source": "dynamic" if (DYNAMIC_WEIGHT_ENABLED and use_dynamic_weights) else "static",
            "confidence_transform": {
                "enabled": nonlinear_enabled,
                "high_threshold": CONFIDENCE_HIGH_THRESHOLD,
                "low_threshold": CONFIDENCE_LOW_THRESHOLD,
            },
        }

    def apply_risk_filter(
        self,
        preliminary: Dict[str, Any],
        ra_opinion: Optional[AgentOpinion],
    ) -> Dict[str, Any]:
        """
        应用风险过滤

        Rules:
        - risk_level=5 → 强制观望
        - risk_level=4 + preliminary=1 → 降级为 0
        - confidence < 0.55 → 观望
        """
        result = dict(preliminary)

        if ra_opinion is None:
            return result

        ra_data = ra_opinion.raw_data
        risk_level = ra_data.get("risk_level", 3)
        max_position = ra_data.get("max_position_pct", 0.15)

        # 风险过滤规则
        if risk_level >= 5:
            result["final_signal"] = 0
            result["risk_override"] = "risk_level=5，强制观望"
        elif risk_level == 4 and result["preliminary_signal"] == 1:
            result["final_signal"] = 0
            result["risk_override"] = "risk_level=4，买入信号被降级为观望"
        elif result["overall_confidence"] < 0.55:
            result["final_signal"] = 0
            result["risk_override"] = "综合置信度不足，观望"
        else:
            result["final_signal"] = result["preliminary_signal"]
            result["risk_override"] = None

        result["max_position_pct"] = max_position
        result["risk_level"] = risk_level

        return result

    def _get_base_weights(self, market_cycle: str) -> Dict[str, float]:
        """根据市场周期获取基础权重"""
        weight_key = CYCLE_WEIGHT_MAP.get(market_cycle, "oscillation")
        return dict(WEIGHT_MAP.get(weight_key, self.default_weights))

    def calculate_scenario_expected_return(
        self,
        scenarios: Dict[str, Any],
    ) -> float:
        """
        计算情景分析的预期收益
        E(R) = Σ P_i * R_i
        """
        if not scenarios:
            return 0.0

        expected = 0.0
        for name, scenario in scenarios.items():
            prob = scenario.get("probability", 0.33)
            ret = scenario.get("return_pct", 0.0)
            expected += prob * ret

        return round(expected, 2)
