"""
MASS Agent 间通信引擎 (Inter-Agent Communication Engine)
v2.3 新增：
- 两轮分析机制：独立分析 → 通信摘要 → 修正分析
- 冲突检测与分歧量化
- 智能修正触发（仅对显著分歧的Agent执行修正）
"""
from typing import Dict, List, Optional, Any, Callable
from collections import Counter

from loguru import logger

from agent.core.blackboard import AgentOpinion


class AgentCommunicationEngine:
    """
    Agent 间通信引擎
    
    核心流程：
    1. 第一轮：所有 Agent 独立分析
    2. 构建通信摘要：提取各Agent的关键结论
    3. 冲突检测：识别与其他Agent显著分歧的个体
    4. 第二轮（可选）：对需要修正的Agent执行修正分析
    5. 结果合并：优先使用修正后的结论
    """

    # 分歧判定阈值
    SIGNAL_DIVERGENCE_THRESHOLD = 2  # signal 差值 >= 2 视为显著分歧（如 -1 vs 1）
    CONFIDENCE_LOW_THRESHOLD = 0.50   # 置信度低于此值视为不确定
    MAX_REVISIONS = 2                 # 每轮诊断最多修正的Agent数量

    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: AgentOrchestrator 实例，用于调用Agent和获取数据
        """
        self.orchestrator = orchestrator

    def run_two_round_analysis(
        self,
        stock_code: str,
        user_position: Optional[Dict] = None,
        progress_cb: Optional[Callable] = None,
    ) -> Dict[str, AgentOpinion]:
        """
        执行两轮分析
        
        Round 1: 所有 Agent 独立分析（与当前模式一致）
        Round 2: 对显著分歧的 Agent 执行修正分析
        
        Returns:
            Dict[str, AgentOpinion]: 最终观点（含修正标记）
        """
        # ── Round 1: 独立分析 ──
        logger.info(f"[{stock_code}] Round 1: 启动独立分析")
        if progress_cb:
            progress_cb("agent_start", "Round 1: 6大Agent独立分析中...", progress=32)

        round1_opinions = self.orchestrator._run_agents_parallel(
            stock_code, user_position, progress_cb=progress_cb
        )

        # 记录第一轮结果到blackboard（覆盖之前的）
        self._record_round1_opinions(stock_code, round1_opinions)

        # ── 构建通信摘要 ──
        communication_summary = self._build_communication_summary(round1_opinions)
        logger.debug(f"[{stock_code}] 通信摘要:\n{communication_summary[:500]}...")

        # ── 冲突检测 ──
        conflicts = self._detect_conflicts(round1_opinions)
        if not conflicts:
            logger.info(f"[{stock_code}] Agent间无明显分歧，跳过Round 2")
            return round1_opinions

        logger.info(
            f"[{stock_code}] 检测到 {len(conflicts)} 个Agent需要修正: {conflicts}"
        )

        # ── Round 2: 修正分析 ──
        if progress_cb:
            progress_cb("agent_revision", f"Round 2: {len(conflicts)}个Agent修正分析中...", progress=78)

        round2_opinions = self._run_revision_round(
            stock_code, round1_opinions, conflicts, communication_summary, user_position
        )

        # 统计修正效果
        revised_count = sum(
            1 for aid in conflicts
            if round2_opinions.get(aid, round1_opinions[aid]).is_revision
        )
        logger.info(f"[{stock_code}] Round 2 完成: {revised_count}/{len(conflicts)} 个Agent完成修正")

        return round2_opinions

    def _record_round1_opinions(self, stock_code: str, opinions: Dict[str, AgentOpinion]) -> None:
        """记录第一轮观点到blackboard（清空之前的可能存在的旧观点）"""
        snapshot = self.orchestrator.blackboard.get_snapshot(stock_code)
        self.orchestrator.blackboard.clear_stock(stock_code)
        if snapshot is not None:
            self.orchestrator.blackboard.publish_snapshot(snapshot)
        for agent_id, opinion in opinions.items():
            self.orchestrator.blackboard.submit_opinion(stock_code, opinion)

    def _build_communication_summary(self, opinions: Dict[str, AgentOpinion]) -> str:
        """
        构建 Agent 间通信摘要
        
        提取每个Agent的关键结论，形成结构化的通信文本
        """
        sig_map = {-1: "卖出", 0: "观望", 1: "买入"}
        parts = ["=== 其他分析师的初步结论 ===", ""]

        for agent_id, op in opinions.items():
            sig_text = sig_map.get(op.signal, "观望")
            parts.append(f"【{agent_id}】→ {sig_text} (置信度 {op.confidence:.0%})")
            parts.append(f"  核心理由: {op.reasoning[:120]}...")
            if op.key_factors:
                parts.append(f"  关键因子: {', '.join(op.key_factors[:3])}")
            if op.risk_flags:
                parts.append(f"  风险标记: {', '.join(op.risk_flags[:2])}")
            # 添加原始数据中的关键指标（如果存在）
            raw = op.raw_data or {}
            extras = []
            if "fundamental_score" in raw:
                extras.append(f"基本面评分: {raw['fundamental_score']}")
            if "capital_score" in raw:
                extras.append(f"资金评分: {raw['capital_score']}")
            if "sentiment_index" in raw:
                extras.append(f"情绪指数: {raw['sentiment_index']:.2f}")
            if "risk_level" in raw:
                extras.append(f"风险等级: {raw['risk_level']}")
            if extras:
                parts.append(f"  量化指标: {' | '.join(extras)}")
            parts.append("")

        # 添加整体统计
        signals = [op.signal for op in opinions.values() if op.agent_id != "RA-Agent"]
        if signals:
            counter = Counter(signals)
            majority = counter.most_common(1)[0]
            parts.append("=== 整体信号统计 ===")
            parts.append(f"买入: {counter.get(1, 0)}票 | 观望: {counter.get(0, 0)}票 | 卖出: {counter.get(-1, 0)}票")
            parts.append(f"多数意见: {sig_map.get(majority[0], '观望')} ({majority[1]}票)")
            parts.append("")

        return "\n".join(parts)

    def _detect_conflicts(self, opinions: Dict[str, AgentOpinion]) -> List[str]:
        """
        检测需要修正的 Agent
        
        修正触发条件（满足任一即可）：
        1. 该Agent信号与多数意见相反（差值 >= 2，如 -1 vs 1）
        2. 该Agent置信度异常低（< CONFIDENCE_LOW_THRESHOLD）
        3. 该Agent为关键Agent（TA/FA）且与Chairman初步判断方向相反
        
        排除RA-Agent（风控官不参与方向投票）
        """
        # 计算多数信号（排除RA-Agent）
        directional_agents = {
            aid: op for aid, op in opinions.items()
            if aid != "RA-Agent" and op.signal != 0
        }
        if not directional_agents:
            return []

        signals = [op.signal for op in directional_agents.values()]
        majority_signal = Counter(signals).most_common(1)[0][0] if signals else 0

        conflicts = []
        for agent_id, opinion in opinions.items():
            if agent_id == "RA-Agent":
                continue  # RA-Agent 不参与方向投票，不需要修正

            if self._needs_revision(agent_id, opinion, majority_signal, opinions):
                conflicts.append(agent_id)

        # 限制修正数量，优先修正置信度最低的
        if len(conflicts) > self.MAX_REVISIONS:
            conflicts = sorted(
                conflicts,
                key=lambda aid: opinions[aid].confidence
            )[:self.MAX_REVISIONS]

        return conflicts

    def _needs_revision(
        self,
        agent_id: str,
        opinion: AgentOpinion,
        majority_signal: int,
        all_opinions: Dict[str, AgentOpinion],
    ) -> bool:
        """判断单个Agent是否需要修正"""
        # 条件1: 信号与多数意见显著相反
        if opinion.signal != majority_signal:
            signal_diff = abs(opinion.signal - majority_signal)
            if signal_diff >= self.SIGNAL_DIVERGENCE_THRESHOLD:
                logger.debug(
                    f"{agent_id} 信号分歧: {opinion.signal} vs 多数 {majority_signal}"
                )
                return True

        # 条件2: 置信度异常低
        if opinion.confidence < self.CONFIDENCE_LOW_THRESHOLD:
            logger.debug(
                f"{agent_id} 置信度过低: {opinion.confidence:.2f}"
            )
            return True

        # 条件3: 关键Agent（TA/FA/CA）与至少2个其他Agent方向相反
        if agent_id in ("TA-Agent", "FA-Agent", "CA-Agent"):
            opposite_count = sum(
                1 for aid, op in all_opinions.items()
                if aid != agent_id and aid != "RA-Agent"
                and op.signal != 0 and op.signal != opinion.signal
            )
            if opposite_count >= 2:
                logger.debug(
                    f"{agent_id} 关键Agent孤立: {opposite_count}个Agent方向相反"
                )
                return True

        return False

    def _run_revision_round(
        self,
        stock_code: str,
        round1_opinions: Dict[str, AgentOpinion],
        conflicts: List[str],
        communication_summary: str,
        user_position: Optional[Dict] = None,
    ) -> Dict[str, AgentOpinion]:
        """
        执行第二轮修正分析
        
        仅对 conflicts 列表中的Agent执行修正，其他Agent保持原结论
        """
        snapshot = self.orchestrator.blackboard.get_snapshot(stock_code)
        final_opinions = dict(round1_opinions)

        for agent_id in conflicts:
            agent = self.orchestrator.agents.get(agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} 不存在，跳过修正")
                continue

            original = round1_opinions[agent_id]
            logger.info(
                f"[{stock_code}] {agent_id} 启动修正分析: "
                f"原始信号={original.signal}, 置信度={original.confidence:.2f}"
            )

            try:
                revised = agent.revise(
                    snapshot=snapshot,
                    original_opinion=original,
                    communication_summary=communication_summary,
                    user_position=user_position,
                )
                final_opinions[agent_id] = revised
                self.orchestrator.blackboard.submit_opinion(stock_code, revised)
                logger.info(
                    f"[{stock_code}] {agent_id} 修正完成: "
                    f"信号 {original.signal}→{revised.signal}, "
                    f"置信度 {original.confidence:.2f}→{revised.confidence:.2f}"
                )
            except Exception as e:
                logger.error(f"[{stock_code}] {agent_id} 修正分析异常: {e}")
                # 保持原始结论

        return final_opinions

    def get_conflict_report(
        self,
        opinions: Dict[str, AgentOpinion],
    ) -> Dict[str, Any]:
        """
        生成分歧报告（用于前端展示）
        
        Returns:
            {
                "has_conflict": bool,
                "majority_signal": int,
                "conflict_agents": [str],
                "revision_summary": str,
            }
        """
        conflicts = self._detect_conflicts(opinions)
        directional = {aid: op for aid, op in opinions.items() if aid != "RA-Agent" and op.signal != 0}
        signals = [op.signal for op in directional.values()]
        majority = Counter(signals).most_common(1)[0][0] if signals else 0

        sig_map = {-1: "卖出", 0: "观望", 1: "买入"}
        return {
            "has_conflict": len(conflicts) > 0,
            "majority_signal": majority,
            "majority_text": sig_map.get(majority, "观望"),
            "conflict_agents": conflicts,
            "revision_summary": (
                f"检测到 {len(conflicts)} 个Agent与多数意见({sig_map.get(majority)})存在分歧，"
                f"已执行修正分析"
            ) if conflicts else "Agent间观点一致，无需修正",
        }
