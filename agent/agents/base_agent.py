"""
MASS Agent 抽象基类
所有Agent必须继承此类并实现 analyze 方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from loguru import logger

from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.core.agent_cache import AgentCache
from agent.core.prompt_compressor import PromptCompressor
from agent.tools.llm_client import LLMClient


class BaseAgent(ABC):
    """
    Agent抽象基类
    
    子类需要:
    1. 实现 analyze() 方法
    2. 在 agent/prompts/system/{agent_id}.md 放置系统提示词
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        model_params: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.llm = llm_client
        self.model_params = model_params or {}
        self.model = model  # 模型分层：可为该 Agent 指定专用模型
        self.system_prompt = self._load_prompt()
        # 懒加载优化组件
        self._agent_cache: Optional[AgentCache] = None
        self._prompt_compressor: Optional[PromptCompressor] = None
        logger.info(f"Agent {agent_id} 初始化完成 (model={model or 'default'})")
    
    def _load_prompt(self) -> str:
        """从文件加载系统提示词（支持热更新）"""
        path = Path(f"agent/prompts/system/{self.agent_id.lower().replace('-', '_')}.md")
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # 尝试其他命名格式
        alt_path = Path(f"agent/prompts/system/{self.agent_id.lower()}.md")
        if alt_path.exists():
            with open(alt_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        logger.warning(f"Agent {self.agent_id} 的系统提示词文件不存在: {path}")
        return self._default_prompt()
    
    @abstractmethod
    def analyze(
        self,
        snapshot: StockSnapshot,
        user_position: Optional[Dict] = None,
    ) -> AgentOpinion:
        """
        执行分析并返回Agent观点
        
        Args:
            snapshot: 股票数据快照
            user_position: 用户持仓信息（可选）
        
        Returns:
            AgentOpinion
        """
        pass
    
    def _call_llm(self, user_prompt: str, json_mode: bool = True) -> Dict[str, Any]:
        """调用LLM — 支持模型参数覆盖 + 模型分层"""
        return self.llm.chat(
            system=self.system_prompt,
            user=user_prompt,
            json_mode=json_mode,
            model=self.model,
            override_config=self.model_params if self.model_params else None,
        )
    
    def _build_default_opinion(
        self,
        signal: int,
        confidence: float,
        reasoning: str,
        raw_data: Dict[str, Any],
    ) -> AgentOpinion:
        """构建标准AgentOpinion"""
        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=raw_data.get("key_factors", []),
            risk_flags=raw_data.get("risk_flags", []),
            raw_data=raw_data,
        )
    
    @abstractmethod
    def _default_prompt(self) -> str:
        """默认系统提示词（当文件不存在时使用）"""
        pass
    
    # ── 缓存与 Prompt 压缩辅助 ──

    def _get_agent_cache(self) -> AgentCache:
        if self._agent_cache is None:
            self._agent_cache = AgentCache()
        return self._agent_cache

    def _get_prompt_compressor(self) -> PromptCompressor:
        if self._prompt_compressor is None:
            self._prompt_compressor = PromptCompressor()
        return self._prompt_compressor

    def _check_cache(
        self,
        stock_code: str,
        snapshot: StockSnapshot,
    ) -> Optional[AgentOpinion]:
        """检查是否有缓存的 AgentOpinion 可用"""
        return self._get_agent_cache().get(stock_code, self.agent_id, snapshot)

    def _save_cache(
        self,
        stock_code: str,
        snapshot: StockSnapshot,
        opinion: AgentOpinion,
    ) -> None:
        """将 AgentOpinion 存入缓存"""
        self._get_agent_cache().set(stock_code, self.agent_id, snapshot, opinion)

    def _build_compressed_user_prompt(
        self,
        snapshot: StockSnapshot,
        user_position: Optional[Dict] = None,
    ) -> str:
        """构建压缩后的用户提示词（减少 Token）"""
        base = self._get_prompt_compressor().compress_for_agent(snapshot, self.agent_id)
        if user_position:
            base += f"\n\n用户当前持仓: {user_position}"
        return base
    
    def revise(
        self,
        snapshot: StockSnapshot,
        original_opinion: "AgentOpinion",
        communication_summary: str,
        user_position: Optional[Dict] = None,
    ) -> "AgentOpinion":
        """
        修正分析 —— 基于其他Agent的结论重新审视自身观点
        
        Args:
            snapshot: 股票数据快照
            original_opinion: 该Agent的原始观点
            communication_summary: 其他Agent的通信摘要
            user_position: 用户持仓信息
        
        Returns:
            AgentOpinion（修正后的观点）
        """
        # 修正分析不使用缓存（上下文已变化）
        base_prompt = self._build_user_prompt(snapshot, user_position)
        revision_prompt = (
            f"{base_prompt}\n\n"
            f"{communication_summary}\n\n"
            "=== 修正分析要求 ===\n"
            "你注意到其他分析师的结论与你的初步判断可能存在分歧。\n"
            "请重新审视你的分析，考虑以下可能性：\n"
            "1. 你是否忽略了其他分析师发现的关键因素？\n"
            "2. 你的分析框架是否有盲区和偏差？\n"
            "3. 在什么条件下，其他分析师的结论会是正确的？\n"
            "4. 综合所有观点后，你的修正结论是什么？\n"
            "\n"
            "如果修正后观点与之前一致，请说明坚持理由；"
            "如果发生转变，请说明转变原因。\n"
        )
        
        try:
            response = self._call_llm(revision_prompt)
            parsed = self._safe_parse_llm_response(response)
            
            opinion = self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=f"[修正分析] {parsed['reasoning']}",
                raw_data={**parsed, "revision": True, "original_signal": original_opinion.signal},
            )
            # 标记为修正分析
            opinion.is_revision = True
            opinion.original_signal = original_opinion.signal
            opinion.revision_round = 2
            return opinion
        except Exception as e:
            logger.warning(f"{self.agent_id} 修正分析失败: {e}，使用原始结论")
            return original_opinion
    
    def _build_user_prompt(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> str:
        """构建用户提示词（子类可覆盖）"""
        context = snapshot.to_prompt_context()
        if user_position:
            context += f"\n\n用户当前持仓: {user_position}"
        return context
    
    def _safe_parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """安全解析LLM响应，确保关键字段存在"""
        defaults = {
            "signal": 0,
            "confidence": 0.5,
            "reasoning": "分析中...",
            "key_factors": [],
            "risk_flags": [],
        }
        result = {**defaults, **response}
        
        # 类型校验
        try:
            result["signal"] = int(result["signal"])
            if result["signal"] not in (-1, 0, 1):
                result["signal"] = 0
        except (ValueError, TypeError):
            result["signal"] = 0
        
        try:
            conf = float(result["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result["confidence"] = 0.5
        
        if not isinstance(result["key_factors"], list):
            result["key_factors"] = [str(result["key_factors"])]
        if not isinstance(result["risk_flags"], list):
            result["risk_flags"] = [str(result["risk_flags"])]
        
        return result
