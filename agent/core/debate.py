"""
MASS Agent 辩论机制 (Debate Engine)
实现 Agent 间 1v1 质询与交叉验证
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from agent.core.blackboard import AgentOpinion
from agent.tools.llm_client import LLMClient


@dataclass
class DebateRound:
    """辩论轮次"""
    round_number: int
    challenger: str          # 发起质疑的Agent
    responder: str           # 被质疑的Agent
    challenge: str           # 质疑内容
    response: str            # 回应内容
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebateResult:
    """辩论结果"""
    topic: str               # 辩论主题
    rounds: List[DebateRound]
    summary: str             # 总结
    consensus_reached: bool  # 是否达成共识
    winner: Optional[str]    # 占优方（可选）
    confidence_delta: float  # 辩论后置信度变化


class DebateEngine:
    """
    Agent 辩论引擎
    
    当 Chairman 检测到Agent间观点冲突时，自动触发定向辩论。
    辩论过程会被记录并作为 Chairman 最终决策的额外输入。
    """
    
    # 冲突检测阈值
    SIGNAL_CONFLICT_THRESHOLD = 2   # signal差异 >= 2 (如 1 vs -1)
    CONFIDENCE_GAP_THRESHOLD = 0.3  # 置信度差距 >= 0.3
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.debate_history: List[DebateResult] = []
    
    def detect_conflicts(self, opinions: Dict[str, AgentOpinion]) -> List[Dict[str, Any]]:
        """
        检测Agent间的观点冲突
        
        Returns:
            冲突列表，每个冲突包含冲突双方和冲突原因
        """
        conflicts = []
        agent_ids = list(opinions.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a1, a2 = agent_ids[i], agent_ids[j]
                op1, op2 = opinions[a1], opinions[a2]
                
                signal_diff = abs(op1.signal - op2.signal)
                conf_gap = abs(op1.confidence - op2.confidence)
                
                # 判断是否为有效冲突
                is_conflict = False
                reason = []
                
                if signal_diff >= self.SIGNAL_CONFLICT_THRESHOLD:
                    is_conflict = True
                    reason.append(f"信号方向相反: {op1.signal} vs {op2.signal}")
                
                if conf_gap >= self.CONFIDENCE_GAP_THRESHOLD and signal_diff > 0:
                    is_conflict = True
                    reason.append(f"置信度差距大: {op1.confidence:.2f} vs {op2.confidence:.2f}")
                
                # 检查风险标记冲突
                op1_risk = set(op1.risk_flags)
                op2_risk = set(op2.risk_flags)
                if op1_risk != op2_risk and len(op1_risk | op2_risk) > 0:
                    unique_risks = op1_risk.symmetric_difference(op2_risk)
                    if len(unique_risks) >= 2:
                        is_conflict = True
                        reason.append(f"风险认知分歧: {len(unique_risks)}项差异")
                
                if is_conflict:
                    conflicts.append({
                        "agent_a": a1,
                        "agent_b": a2,
                        "opinion_a": op1,
                        "opinion_b": op2,
                        "reason": "; ".join(reason),
                        "severity": self._calculate_severity(signal_diff, conf_gap),
                    })
        
        # 按严重程度排序
        conflicts.sort(key=lambda x: x["severity"], reverse=True)
        logger.info(f"检测到 {len(conflicts)} 组Agent冲突")
        return conflicts
    
    def run_debate(
        self,
        topic: str,
        challenger_op: AgentOpinion,
        responder_op: AgentOpinion,
        context: str = "",
        max_rounds: int = 2,
    ) -> DebateResult:
        """
        运行一场Agent间辩论
        
        Args:
            topic: 辩论主题
            challenger_op: 质疑方观点
            responder_op: 被质疑方观点
            context: 额外上下文
            max_rounds: 最大轮数
        
        Returns:
            DebateResult
        """
        rounds = []
        
        for r in range(max_rounds):
            # 生成质疑
            challenge = self._generate_challenge(
                challenger_op, responder_op, context, r + 1
            )
            
            # 生成回应
            response = self._generate_response(
                responder_op, challenger_op, challenge, context, r + 1
            )
            
            rounds.append(DebateRound(
                round_number=r + 1,
                challenger=challenger_op.agent_id,
                responder=responder_op.agent_id,
                challenge=challenge,
                response=response,
            ))
        
        # 评估辩论结果
        consensus_reached, winner, confidence_delta = self._evaluate_debate(rounds)

        # 生成总结
        summary = self._summarize_debate(topic, rounds, winner)

        result = DebateResult(
            topic=topic,
            rounds=rounds,
            summary=summary,
            consensus_reached=consensus_reached,
            winner=winner,
            confidence_delta=confidence_delta,
        )
        
        self.debate_history.append(result)
        logger.info(f"辩论完成: {topic}, {len(rounds)}轮")
        return result
    
    def run_all_debates(
        self,
        opinions: Dict[str, AgentOpinion],
        context: str = "",
        max_debates: int = 2,
    ) -> List[DebateResult]:
        """
        自动检测并运行所有必要辩论
        
        Args:
            opinions: 所有Agent观点
            context: 股票上下文
            max_debates: 最多进行几场辩论
        
        Returns:
            辩论结果列表
        """
        conflicts = self.detect_conflicts(opinions)
        results = []
        
        for conflict in conflicts[:max_debates]:
            # 让置信度更高的Agent作为被质疑方（需要为观点辩护）
            op_a = conflict["opinion_a"]
            op_b = conflict["opinion_b"]
            
            if op_a.confidence > op_b.confidence:
                challenger, responder = op_b, op_a
            else:
                challenger, responder = op_a, op_b
            
            topic = f"{conflict['agent_a']} vs {conflict['agent_b']}: {conflict['reason']}"
            
            result = self.run_debate(
                topic=topic,
                challenger_op=challenger,
                responder_op=responder,
                context=context,
            )
            results.append(result)
        
        return results
    
    def _calculate_severity(self, signal_diff: int, conf_gap: float) -> int:
        """计算冲突严重程度"""
        severity = signal_diff * 10
        if conf_gap >= 0.5:
            severity += 5
        elif conf_gap >= 0.3:
            severity += 3
        return severity
    
    def _generate_challenge(
        self,
        challenger: AgentOpinion,
        responder: AgentOpinion,
        context: str,
        round_num: int,
    ) -> str:
        """生成质疑内容（LLM或规则引擎）"""
        if self.llm is None:
            return self._rule_based_challenge(challenger, responder, round_num)
        
        prompt = f"""你是一位投资辩论中的质疑方。

你的观点: {challenger.signal} ({'买入' if challenger.signal==1 else ('卖出' if challenger.signal==-1 else '观望')}), 置信度{challenger.confidence}
你的理由: {challenger.reasoning}

对方观点: {responder.signal}, 置信度{responder.confidence}
对方理由: {responder.reasoning}

请用1-2句话，基于数据和逻辑，指出对方分析中的漏洞或未被充分考虑的因素。
"""
        try:
            response = self.llm.chat(
                system="你是一位严谨的投资分析师，善于发现论证中的漏洞。",
                user=prompt,
                json_mode=False,
            )
            if isinstance(response, dict):
                return response.get("content", response.get("text", ""))
            return str(response)[:200]
        except Exception as e:
            logger.warning(f"LLM生成质疑失败: {e}")
            return self._rule_based_challenge(challenger, responder, round_num)
    
    def _generate_response(
        self,
        responder: AgentOpinion,
        challenger: AgentOpinion,
        challenge: str,
        context: str,
        round_num: int,
    ) -> str:
        """生成回应内容"""
        if self.llm is None:
            return self._rule_based_response(responder, challenge, round_num)
        
        prompt = f"""你是一位投资辩论中的辩护方。

你的观点: {responder.signal}, 置信度{responder.confidence}
你的理由: {responder.reasoning}

对方质疑: {challenge}

请用1-2句话回应质疑，解释为什么你的分析依然成立，或承认合理的顾虑。
"""
        try:
            response = self.llm.chat(
                system="你是一位理性的投资分析师，能够冷静回应质疑。",
                user=prompt,
                json_mode=False,
            )
            if isinstance(response, dict):
                return response.get("content", response.get("text", ""))
            return str(response)[:200]
        except Exception as e:
            logger.warning(f"LLM生成回应失败: {e}")
            return self._rule_based_response(responder, challenge, round_num)
    
    def _rule_based_challenge(
        self,
        challenger: AgentOpinion,
        responder: AgentOpinion,
        round_num: int,
    ) -> str:
        """基于规则的质疑生成"""
        templates = [
            f"你提到{'看多' if responder.signal==1 else '看空'}，但{challenger.key_factors[0] if challenger.key_factors else '某些因素'}似乎与你的判断矛盾。",
            f"你的置信度为{responder.confidence}，但在{'牛市' if challenger.signal==1 else '当前环境'}下，是否过于{'乐观' if responder.signal==1 else '悲观'}？",
            f"你忽略了{challenger.risk_flags[0] if challenger.risk_flags else '潜在风险'}，这是否会影响你的结论？",
        ]
        return templates[round_num % len(templates)]
    
    def _rule_based_response(
        self,
        responder: AgentOpinion,
        challenge: str,
        round_num: int,
    ) -> str:
        """基于规则的回应生成"""
        templates = [
            f"你的质疑有道理，但我考虑了{responder.key_factors[0] if responder.key_factors else '更多因素'}，整体判断不变。",
            f"该风险已在评估范围内，{responder.reasoning[:50] if responder.reasoning else '现有数据支持原观点'}。",
            f"确实需要警惕，但当前{'技术面' if 'TA' in responder.agent_id else '基本面'}信号仍然清晰。",
        ]
        return templates[round_num % len(templates)]
    
    def _evaluate_debate(self, rounds: List[DebateRound]) -> tuple:
        """
        评估辩论结果，返回 (consensus_reached, winner, confidence_delta)

        优先使用 LLM 进行多维度逻辑评估；LLM 不可用时降级到规则引擎。
        """
        if not rounds:
            return False, None, 0.0

        # 优先尝试 LLM 评估
        if self.llm is not None:
            try:
                return self._llm_evaluate_debate(rounds)
            except Exception as e:
                logger.warning(f"LLM 辩论评估失败，降级到规则引擎: {e}")

        return self._rule_based_evaluate(rounds)

    def _llm_evaluate_debate(self, rounds: List[DebateRound]) -> tuple:
        """
        使用 LLM 对辩论进行多维度逻辑评估（替代关键词匹配）。

        评估维度：
        1. 质疑方逻辑严密性（是否击中核心矛盾）
        2. 回应方论证充分性（是否有数据/逻辑支撑）
        3. 双方是否围绕核心分歧展开（是否跑题）
        4. 是否有新信息/新视角出现

        Returns:
            (consensus_reached, winner, confidence_delta)
        """
        # 构建辩论全文
        debate_lines = []
        for r in rounds:
            debate_lines.append(f"第{r.round_number}轮:")
            debate_lines.append(f"质疑方({r.challenger}): {r.challenge}")
            debate_lines.append(f"回应方({r.responder}): {r.response}")

        debate_text = "\n".join(debate_lines)
        eval_prompt = (
            "请评估以下投资辩论的质量。\n\n"
            f"辩论内容:\n{debate_text}\n\n"
            "请基于以下维度评估（每项1-5分）：\n"
            "1. 质疑方逻辑严密性（是否击中核心矛盾）\n"
            "2. 回应方论证充分性（是否有数据/逻辑支撑）\n"
            "3. 双方是否围绕核心分歧展开（是否跑题）\n"
            "4. 是否有新信息/新视角出现\n\n"
            "输出严格JSON格式（不要Markdown代码块）：\n"
            '{\n'
            '  "challenger_score": 3,\n'
            '  "responder_score": 4,\n'
            '  "consensus_reached": false,\n'
            '  "winner": "responder",\n'
            '  "reason": "回应方用具体数据支撑了观点，质疑方过于笼统"\n'
            '}\n'
            "\n注意：winner 只能取 'challenger' 或 'responder'，若势均力敌则 winner 为空字符串。"
        )

        response = self.llm.chat(
            system="你是一位严谨的投资辩论裁判，擅长评估论证质量。请只输出合法JSON。",
            user=eval_prompt,
            json_mode=True,
        )

        if isinstance(response, str):
            import json
            response = json.loads(response)

        challenger_score = float(response.get("challenger_score", 3))
        responder_score = float(response.get("responder_score", 3))
        winner_str = response.get("winner", "")
        consensus = bool(response.get("consensus_reached", False))
        reason = response.get("reason", "")

        logger.info(
            f"LLM 辩论评估完成: challenger={challenger_score}, responder={responder_score}, "
            f"winner={winner_str}, consensus={consensus}, reason={reason}"
        )

        last_round = rounds[-1]
        winner = None
        confidence_delta = 0.0

        if winner_str == "challenger":
            winner = last_round.challenger
            confidence_delta = 0.10
            consensus = False
        elif winner_str == "responder":
            winner = last_round.responder
            confidence_delta = 0.08
            consensus = consensus or True  # 回应方获胜视为有效防守/共识
        else:
            # 势均力敌
            consensus = False
            confidence_delta = 0.0

        return consensus, winner, confidence_delta

    def _rule_based_evaluate(self, rounds: List[DebateRound]) -> tuple:
        """
        基于规则的辩论评估（降级方案）。

        通过关键词匹配粗略判断辩论质量。
        """
        challenger_score = 0
        responder_score = 0
        concession_keywords = ["有道理", "承认", "需要警惕", "确实", "忽略", "矛盾"]
        defense_keywords = ["不变", "支持", "清晰", "充分", "整体判断", "评估范围"]

        for r in rounds:
            challenge = r.challenge.lower()
            response = r.response.lower()

            # 回应中若出现让步词，challenger 得分
            if any(kw in response for kw in concession_keywords):
                challenger_score += 2
            # 回应中若出现防守词，responder 得分
            if any(kw in response for kw in defense_keywords):
                responder_score += 1

            # 质疑是否击中关键
            if any(kw in challenge for kw in ["数据", "矛盾", "忽略", "未考虑"]):
                challenger_score += 1

        winner = None
        confidence_delta = 0.0
        consensus_reached = False

        if challenger_score > responder_score + 2:
            winner = rounds[-1].challenger
            confidence_delta = 0.10
        elif responder_score > challenger_score + 1:
            winner = rounds[-1].responder
            confidence_delta = 0.08
            consensus_reached = True
        else:
            confidence_delta = 0.0

        return consensus_reached, winner, confidence_delta

    def _summarize_debate(self, topic: str, rounds: List[DebateRound], winner: Optional[str] = None) -> str:
        """总结辩论"""
        if not rounds:
            return "无辩论内容"

        last_round = rounds[-1]
        if winner:
            return (
                f"辩论'{topic}'共{len(rounds)}轮。"
                f"{winner}的论证更具说服力。"
                f"建议Chairman在决策时重点参考{winner}的观点。"
            )
        return (
            f"辩论'{topic}'共{len(rounds)}轮。"
            f"最终{last_round.responder}回应了{last_round.challenger}的质疑。"
            "双方均未完全说服对方，建议Chairman综合权衡。"
        )
    
    def to_dict(self, result: DebateResult) -> Dict[str, Any]:
        """将辩论结果转为字典"""
        return {
            "topic": result.topic,
            "rounds": [
                {
                    "round": r.round_number,
                    "challenger": r.challenger,
                    "responder": r.responder,
                    "challenge": r.challenge,
                    "response": r.response,
                }
                for r in result.rounds
            ],
            "summary": result.summary,
            "consensus_reached": result.consensus_reached,
            "confidence_delta": result.confidence_delta,
        }
