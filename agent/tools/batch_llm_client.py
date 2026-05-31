"""
批量 LLM 推理客户端 + 批量 Agent 运行器

核心设计：
- BatchLLMClient: 将 N 个 Agent 的独立 LLM 请求合并为 1 次 API 调用
- BatchAgentRunner: 透明拦截所有 Agent 的 _call_llm，收集请求后批量调用，再回填结果
  （无需修改任何子类，保持原有的 analyze() 流程不变）

分组策略（默认）：
- Batch 1: TA-Agent + CA-Agent + SA-Agent（市场微观结构：技术/资金/情绪）
- Batch 2: FA-Agent + MA-Agent + RA-Agent（基本面与风控：财务/宏观/风险）

收益：6 次独立调用 → 2 次批量调用，延迟降低 20~30%（减少 4 次 RTT）
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, NamedTuple, Optional

from loguru import logger

from agent.tools.llm_client import LLMClient


class BatchRequest(NamedTuple):
    """单个 Agent 的批量请求单元"""
    agent_id: str
    system_prompt: str          # 该 Agent 的原始 system prompt
    user_prompt: str            # 该 Agent 的 user prompt（已包含数据上下文）
    model_params: Optional[Dict[str, Any]] = None


class BatchLLMClient:
    """
    批量 LLM 客户端

    用法：
        client = BatchLLMClient(llm_client)
        requests = [
            BatchRequest("TA-Agent", sys_ta, user_ta),
            BatchRequest("CA-Agent", sys_ca, user_ca),
            BatchRequest("SA-Agent", sys_sa, user_sa),
        ]
        results = client.batch_chat(requests)
        # results[0] == {"signal": 1, "confidence": 0.8, ...}
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def batch_chat(
        self,
        requests: List[BatchRequest],
        json_mode: bool = True,
        model: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行批量推理

        Args:
            requests: 每个 Agent 的请求单元
            json_mode: 是否强制 JSON 输出（默认 True）
            model: 覆盖模型
            override_config: 覆盖生成参数

        Returns:
            与 requests 顺序对应的解析结果列表
        """
        if not requests:
            return []

        if len(requests) == 1:
            # 单条请求降级为普通调用，避免 batch prompt 开销
            req = requests[0]
            return [self.llm.chat(
                system=req.system_prompt,
                user=req.user_prompt,
                json_mode=json_mode,
                model=model,
                override_config=override_config or req.model_params,
            )]

        master_prompt = self._build_master_prompt(requests)

        response = self.llm.chat(
            system=self._BATCH_SYSTEM_PROMPT,
            user=master_prompt,
            json_mode=json_mode,
            model=model,
            override_config=override_config,
        )

        return self._parse_results(response, requests)

    # ── 内部常量与辅助方法 ──

    _BATCH_SYSTEM_PROMPT = """你是一位多角色投资分析协调系统。你的任务是为多个独立的分析师角色同时进行分析，每个角色有独立的分析框架和输出要求。

核心规则：
1. 每个角色的分析必须相互独立，不受其他角色影响
2. 严格按照每个角色的输出格式要求生成结果
3. 所有结果必须封装在统一的 JSON 结构中
4. 只输出 JSON，不要任何解释文字

输出格式（严格 JSON）：
{
  "results": [
    {
      "agent_id": "角色标识",
      "signal": 1,              // -1(卖出), 0(观望), 1(买入)
      "confidence": 0.75,       // 0.0 ~ 1.0
      "reasoning": "分析理由（简洁）",
      "key_factors": ["关键因子1", "关键因子2"],
      "risk_flags": ["风险标志1"],
      // 其他角色特有的字段也可包含在 raw_data 中
    }
  ]
}
"""

    def _build_master_prompt(self, requests: List[BatchRequest]) -> str:
        """构造合并后的 master user prompt"""
        parts = [
            f"请为以下 {len(requests)} 个分析师角色独立进行分析。",
            "每个角色的分析数据和分析要求如下：\n",
        ]

        for idx, req in enumerate(requests, 1):
            parts.append(f"=== 角色 {idx}: {req.agent_id} ===")
            # 提取 system prompt 中的核心角色描述（取前 500 字符避免过长）
            role_desc = req.system_prompt.strip()
            if len(role_desc) > 500:
                role_desc = role_desc[:500] + "..."
            parts.append(f"【角色定义】\n{role_desc}")
            parts.append(f"\n【分析数据】\n{req.user_prompt}")
            parts.append("")

        parts.append("=" * 40)
        parts.append("请严格按 JSON 格式输出所有角色的分析结果，不要输出任何解释文字。")

        return "\n".join(parts)

    def _parse_results(
        self,
        response: Dict[str, Any],
        requests: List[BatchRequest],
    ) -> List[Dict[str, Any]]:
        """解析批量响应，确保与 requests 顺序对应"""
        if not isinstance(response, dict):
            logger.error(f"BatchLLM 响应不是 dict: {type(response)}")
            return [self._fallback_result(req.agent_id) for req in requests]

        results_map: Dict[str, Dict[str, Any]] = {}

        # 尝试解析 "results" 数组
        results_list = response.get("results", [])
        if not results_list and "signal" in response:
            # LLM 可能只返回了一个对象（常见于单条或格式错误）
            results_list = [response]

        for item in results_list:
            if isinstance(item, dict) and "agent_id" in item:
                agent_id = item["agent_id"]
                results_map[agent_id] = item

        # 按 requests 顺序组装返回
        output: List[Dict[str, Any]] = []
        for req in requests:
            if req.agent_id in results_map:
                output.append(results_map[req.agent_id])
            else:
                logger.warning(f"BatchLLM 响应中缺少 {req.agent_id}，使用降级结果")
                output.append(self._fallback_result(req.agent_id))

        return output

    @staticmethod
    def _fallback_result(agent_id: str) -> Dict[str, Any]:
        """批量解析失败时的降级结果"""
        return {
            "agent_id": agent_id,
            "signal": 0,
            "confidence": 0.3,
            "reasoning": "批量推理解析失败，降级为观望",
            "key_factors": ["批量解析异常"],
            "risk_flags": ["batch_parse_error"],
        }


class BatchAgentRunner:
    """
    批量 Agent 运行器 — 透明拦截所有 Agent 的 _call_llm，实现批量推理

    无需修改任何子类，保持原有的 analyze() 流程不变。
    内部通过临时替换 _call_llm 为拦截函数，收集所有 prompt 后批量调用 LLM，
    再回填结果让各 Agent 继续执行后处理逻辑。

    用法：
        from agent.tools.batch_llm_client import BatchAgentRunner, BatchLLMClient
        runner = BatchAgentRunner(BatchLLMClient(llm_client))
        opinions = runner.run(orchestrator.agents, snapshot, user_position)
    """

    def __init__(self, batch_llm_client: BatchLLMClient):
        self.batch_client = batch_llm_client
        self._original_methods: Dict[str, callable] = {}

    def run(
        self,
        agents: Dict[str, Any],
        snapshot: Any,
        user_position: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        批量运行所有 Agent。

        返回:
            Dict[str, AgentOpinion]
        """
        import queue

        q: queue.Queue = queue.Queue()
        results: Dict[str, Dict[str, Any]] = {}
        event = threading.Event()
        lock = threading.Lock()

        def _make_interceptor(agent_id: str, system_prompt: str):
            def _intercept(user_prompt: str, json_mode: bool = True) -> Dict[str, Any]:
                q.put((agent_id, system_prompt, user_prompt))
                with lock:
                    pass  # 确保 put 完成
                event.wait(timeout=120)
                return results.get(agent_id, BatchLLMClient._fallback_result(agent_id))
            return _intercept

        # 1. 替换每个 Agent 的 _call_llm 为拦截函数
        for aid, agent in agents.items():
            self._original_methods[aid] = agent._call_llm
            agent._call_llm = _make_interceptor(aid, agent.system_prompt)

        # 2. 启动 collector 线程（daemon，独立运行）
        def _collect():
            collected = []
            deadline = time.time() + 15  # 最多收集 15 秒
            while len(collected) < len(agents) and time.time() < deadline:
                try:
                    item = q.get(timeout=0.5)
                    collected.append(item)
                except queue.Empty:
                    continue

            # 分 batch 调用 LLM
            batch_size = 3
            for i in range(0, len(collected), batch_size):
                batch = collected[i:i + batch_size]
                batch_reqs = [
                    BatchRequest(aid, sys_p, user_p)
                    for aid, sys_p, user_p in batch
                ]
                try:
                    batch_results = self.batch_client.batch_chat(batch_reqs)
                    for (aid, _, _), result in zip(batch, batch_results):
                        results[aid] = result
                except Exception as e:
                    logger.error(f"BatchLLM 调用失败: {e}")
                    for aid, _, _ in batch:
                        results[aid] = BatchLLMClient._fallback_result(aid)

            # 未收集到的 Agent（可能 analyze 抛异常未调用 _call_llm）也填充降级结果
            for aid in agents:
                if aid not in results:
                    results[aid] = BatchLLMClient._fallback_result(aid)

            event.set()

        collector = threading.Thread(target=_collect, daemon=True)
        collector.start()

        # 3. 并行运行所有 Agent 的 analyze()
        opinions: Dict[str, Any] = {}
        # 延迟导入避免循环依赖
        from agent.core.blackboard import AgentOpinion

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {
                executor.submit(agent.analyze, snapshot, user_position): aid
                for aid, agent in agents.items()
            }
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    opinions[aid] = future.result(timeout=180)
                except Exception as e:
                    logger.error(f"Batch 模式下 {aid} 分析失败: {e}")
                    opinions[aid] = AgentOpinion(
                        agent_id=aid,
                        signal=0,
                        confidence=0.3,
                        reasoning=f"批量推理失败: {e}",
                        key_factors=["batch_error"],
                        risk_flags=[f"{aid} batch失败"],
                        raw_data={"error": str(e)},
                    )

        # 4. 恢复原始 _call_llm
        for aid, orig in self._original_methods.items():
            agents[aid]._call_llm = orig
        self._original_methods.clear()

        collector.join(timeout=5)
        return opinions
