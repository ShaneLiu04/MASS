"""
MASS Agent 编排器 (Orchestrator / Chairman)
负责：任务分解 → 并行派发 → 结果聚合 → 冲突仲裁 → 最终决策
"""
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from loguru import logger

from config import (
    CHAIRMAN_MODEL, AGENT_PARALLEL, MAX_CONCURRENT_AGENTS, DISCLAIMER,
    AGENT_INTER_COMMUNICATION,
    ADAPTIVE_CONCURRENCY, CONCURRENCY_MIN, CONCURRENCY_MAX,
    CONCURRENCY_SLOW_THRESHOLD, CONCURRENCY_FAST_THRESHOLD,
    CONCURRENCY_ERROR_HIGH, CONCURRENCY_ERROR_LOW, CONCURRENCY_WINDOW_SIZE,
    AGENT_MODEL_MAP, BATCH_INFERENCE_ENABLED, BATCH_INFERENCE_SIZE,
)
from agent.core.blackboard import Blackboard, StockSnapshot, AgentOpinion
from agent.core.decision_engine import DecisionEngine
from agent.core.debate import DebateEngine
from agent.core.communication import AgentCommunicationEngine
from agent.core.concurrency_controller import AdaptiveConcurrencyController
from agent.core.container import Container, get_container
from agent.tools.llm_client import LLMClient, MockLLMClient
from agent.tools.stock_data_tool import StockDataTool
from agent.tools.indicator_tool import IndicatorTool
from agent.tools.sentiment_tool import SentimentTool
from agent.agents import (
    TA_Agent, FA_Agent, CA_Agent, SA_Agent, MA_Agent, RA_Agent
)
from agent.core.exceptions import DataError
from agent.core.prediction_engine import PredictionEngine
from agent.models.agent_response import DecisionPackage, ChairmanDecision
from agent.models.prediction import PredictionResult

# ── 统一线程池 ──
# 动态计算 worker 数：CPU 核心 + 4（I/O 密集型场景），上限 32
_ORCHESTRATOR_MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)
_ORCHESTRATOR_EXECUTOR = ThreadPoolExecutor(
    max_workers=_ORCHESTRATOR_MAX_WORKERS,
    thread_name_prefix="orch-",
)

# ── 并发控制：自适应 vs 静态 ──
# 自适应并发控制器根据响应时间和错误率动态调整诊断并发限制
if ADAPTIVE_CONCURRENCY:
    _CONCURRENCY_CONTROLLER = AdaptiveConcurrencyController(
        min_concurrent=CONCURRENCY_MIN,
        max_concurrent=CONCURRENCY_MAX,
        slow_threshold=CONCURRENCY_SLOW_THRESHOLD,
        fast_threshold=CONCURRENCY_FAST_THRESHOLD,
        error_high=CONCURRENCY_ERROR_HIGH,
        error_low=CONCURRENCY_ERROR_LOW,
        window_size=CONCURRENCY_WINDOW_SIZE,
    )
    logger.info(f"自适应并发控制已启用: min={CONCURRENCY_MIN}, max={CONCURRENCY_MAX}")
else:
    # 静态信号量：限制同时进行的诊断请求数，避免无限堆积耗尽线程池
    # 设为线程池容量的一半，保留余量给内部计算任务
    _MAX_CONCURRENT_DIAGNOSES = max(2, _ORCHESTRATOR_MAX_WORKERS // 2)
    _DIAGNOSIS_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_DIAGNOSES)
    logger.info(f"静态并发控制: max_concurrent_diagnoses={_MAX_CONCURRENT_DIAGNOSES}")

_EXECUTOR_LOCK = threading.Lock()


class AgentOrchestrator:
    """
    Agent编排器 - 投资委员会主席
    """
    
    AGENT_CLASSES = {
        "TA-Agent": TA_Agent,
        "FA-Agent": FA_Agent,
        "CA-Agent": CA_Agent,
        "SA-Agent": SA_Agent,
        "MA-Agent": MA_Agent,
        "RA-Agent": RA_Agent,
    }
    
    def __init__(
        self,
        use_mock_llm: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
        container: Optional[Container] = None,
    ):
        self._container = container or get_container()
        self.blackboard = self._container.blackboard
        self.decision_engine = DecisionEngine()
        self.stock_tool = self._container.stock_data_tool
        self.indicator_tool = IndicatorTool()
        self.sentiment_tool = SentimentTool()
        self.prediction_engine = PredictionEngine()
        self.model_params = model_params or {}
        
        if use_mock_llm:
            self.llm = MockLLMClient()
            self.chairman_llm = MockLLMClient()
            logger.info("使用 Mock LLM 客户端")
        else:
            self.llm = LLMClient()
            self.chairman_llm = LLMClient()
            # Chairman 可以用更强的模型
            self.chairman_llm.config.model = CHAIRMAN_MODEL

        # 批量推理客户端（延迟初始化）
        self._batch_runner = None

        self.debate_engine = DebateEngine(llm_client=self.chairman_llm)

        # 初始化Agent间通信引擎
        self.communication_engine = AgentCommunicationEngine(self)

        # 初始化所有分析Agent，传入模型参数 + 模型分层
        self.agents = {}
        for agent_id, cls in self.AGENT_CLASSES.items():
            agent_model = AGENT_MODEL_MAP.get(agent_id)
            self.agents[agent_id] = cls(
                agent_id=agent_id,
                llm_client=self.llm,
                model_params=self.model_params,
                model=agent_model,
            )

        logger.info(
            f"AgentOrchestrator 初始化完成，Agent列表: {list(self.agents.keys())}, "
            f"Agent间通信: {'启用' if AGENT_INTER_COMMUNICATION else '禁用'}, "
            f"批量推理: {'启用' if BATCH_INFERENCE_ENABLED else '禁用'}, "
            f"模型分层: {AGENT_MODEL_MAP}"
        )
    
    def run_diagnosis(
        self,
        stock_code: str,
        stock_name: str = "",
        market_type: Optional[str] = None,
        user_position: Optional[Dict] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的多智能体诊断流程
        
        受并发控制器保护，限制并发诊断请求数，防止线程池耗尽。
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            market_type: 市场类型
            user_position: 用户当前持仓信息
        
        Returns:
            DecisionPackage 字典
        """
        # 如果传入了新的 model_params，需要重建Agent
        if model_params and model_params != self.model_params:
            self.model_params = model_params
            for agent_id, cls in self.AGENT_CLASSES.items():
                agent_model = AGENT_MODEL_MAP.get(agent_id)
                self.agents[agent_id] = cls(
                    agent_id=agent_id,
                    llm_client=self.llm,
                    model_params=model_params,
                    model=agent_model,
                )
            logger.info(f"模型参数已更新: {model_params}")

        # 并发控制：获取许可
        _start = time.time()
        _wait_start = time.time()
        if ADAPTIVE_CONCURRENCY:
            acquired = _CONCURRENCY_CONTROLLER.acquire(timeout=30.0)
        else:
            acquired = _DIAGNOSIS_SEMAPHORE.acquire(timeout=30.0)
        _wait_time = time.time() - _wait_start

        if not acquired:
            logger.warning(f"诊断请求过多，并发控制等待超时: {stock_code}")
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.record_result(time.time() - _start, False, _wait_time)
            raise RuntimeError("系统繁忙，当前诊断请求过多，请稍后重试")
        
        try:
            result = self._run_diagnosis_impl(stock_code, stock_name, market_type, user_position)
            _duration = time.time() - _start
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.record_result(_duration, True, _wait_time)
            return result
        except Exception:
            _duration = time.time() - _start
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.record_result(_duration, False, _wait_time)
            raise
        finally:
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.release()
            else:
                _DIAGNOSIS_SEMAPHORE.release()
    
    def _run_diagnosis_impl(
        self,
        stock_code: str,
        stock_name: str = "",
        market_type: Optional[str] = None,
        user_position: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """诊断实际实现（已被信号量保护）"""
        start_time = time.time()
        logger.info(f"开始诊断股票: {stock_code}")
        
        # Phase 1: 数据采集
        snapshot = self._create_snapshot(stock_code, stock_name)
        self.blackboard.publish_snapshot(snapshot)
        
        # Phase 2: Agent分析（可选两轮通信机制 / 批量推理）
        if AGENT_INTER_COMMUNICATION:
            opinions = self.communication_engine.run_two_round_analysis(
                stock_code, user_position
            )
        elif BATCH_INFERENCE_ENABLED and not use_mock_llm:
            opinions = self._run_agents_batch(stock_code, user_position)
        else:
            opinions = self._run_agents_parallel(stock_code, user_position)
        
        # Phase 3: 决策引擎加权投票
        ma_opinion = opinions.get("MA-Agent")
        market_cycle = ""
        weight_adj = None
        if ma_opinion:
            market_cycle = ma_opinion.raw_data.get("market_cycle", "")
            weight_adj = ma_opinion.raw_data.get("recommended_weight_adjustment", {})
        
        weighted = self.decision_engine.compute_weighted_decision(
            opinions, market_cycle, weight_adj
        )
        
        # Phase 4: 风险过滤
        ra_opinion = opinions.get("RA-Agent")
        filtered = self.decision_engine.apply_risk_filter(weighted, ra_opinion)
        
        # Phase 5: Chairman 综合决策
        final_decision = self._chairman_decide(
            stock_code, snapshot, opinions, filtered, user_position
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # 组装决策包
        package = DecisionPackage(
            stock_code=stock_code,
            stock_name=stock_name or snapshot.stock_name,
            current_price=snapshot.current_price,
            decision_date=datetime.now().strftime("%Y-%m-%d"),
            decision_time=datetime.now().strftime("%H:%M:%S"),
            market_cycle=market_cycle,
            opinions={k: v.to_dict() for k, v in opinions.items()},
            final_decision=final_decision,
            disclaimer=DISCLAIMER.strip(),
            processing_time_seconds=processing_time,
            data_summary=snapshot.to_summary(),
            risk_metrics=snapshot.risk_metrics,
            indicators=snapshot.indicators,
            data_quality=snapshot.data_quality,
        )
        
        logger.info(f"股票 {stock_code} 诊断完成，耗时 {processing_time}s，决策: {final_decision.decision}")
        
        return package.model_dump()

    def run_diagnosis_stream(
        self,
        stock_code: str,
        stock_name: str = "",
        market_type: Optional[str] = None,
        user_position: Optional[Dict] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        流式诊断生成器 — 在关键节点 yield 进度事件
        
        受并发控制器保护，限制并发诊断请求数。
        
        Yields: Dict with keys: stage, message, agent_id, progress, data
        """
        # 如果传入了新的 model_params，需要重建Agent
        if model_params and model_params != self.model_params:
            self.model_params = model_params
            for agent_id, cls in self.AGENT_CLASSES.items():
                self.agents[agent_id] = cls(agent_id, self.llm, model_params=model_params)
            logger.info(f"模型参数已更新: {model_params}")

        # 并发控制：获取许可
        _start = time.time()
        _wait_start = time.time()
        if ADAPTIVE_CONCURRENCY:
            acquired = _CONCURRENCY_CONTROLLER.acquire(timeout=30.0)
        else:
            acquired = _DIAGNOSIS_SEMAPHORE.acquire(timeout=30.0)
        _wait_time = time.time() - _wait_start

        if not acquired:
            logger.warning(f"流式诊断请求过多，并发控制等待超时: {stock_code}")
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.record_result(time.time() - _start, False, _wait_time)
            yield {"stage": "error", "message": "系统繁忙，当前诊断请求过多，请稍后重试", "progress": 0}
            return

        _has_error = False
        try:
            for event in self._run_diagnosis_stream_impl(
                stock_code, stock_name, market_type, user_position, model_params
            ):
                if event.get("stage") == "error":
                    _has_error = True
                yield event
        finally:
            _duration = time.time() - _start
            if ADAPTIVE_CONCURRENCY:
                _CONCURRENCY_CONTROLLER.record_result(_duration, not _has_error, _wait_time)
                _CONCURRENCY_CONTROLLER.release()
            else:
                _DIAGNOSIS_SEMAPHORE.release()
    
    def _run_diagnosis_stream_impl(
        self,
        stock_code: str,
        stock_name: str = "",
        market_type: Optional[str] = None,
        user_position: Optional[Dict] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """流式诊断实际实现（已被信号量保护）"""
        start_time = time.time()
        yield {"stage": "init", "message": f"开始诊断 {stock_code}...", "progress": 0}

        # Phase 1: 数据采集（并行获取）
        yield {"stage": "data", "message": "正在并行获取多源实时数据...", "progress": 5}
        try:
            snapshot = self._create_snapshot_internal(
                stock_code, stock_name,
                progress_cb=lambda stage, msg, pct: None,  # 内部有详细进度但流式只发关键节点
            )
            self.blackboard.publish_snapshot(snapshot)
            kline_count = len(snapshot.kline_df) if snapshot.kline_df is not None else 0
            yield {"stage": "data", "message": f"数据获取完成（K线{kline_count}条 / {len(snapshot.fundamentals or {})}个基本面字段）", "progress": 30}
        except DataError as e:
            yield {"stage": "error", "message": str(e), "progress": 0}
            return
        except Exception as e:
            yield {"stage": "error", "message": f"数据获取异常: {e}", "progress": 0}
            return

        # Phase 2: Agent分析（可选两轮通信机制 / 批量推理）
        if AGENT_INTER_COMMUNICATION:
            yield {"stage": "agent_start", "message": "Round 1: 6大Agent独立分析中...", "progress": 32}
            opinions = self.communication_engine.run_two_round_analysis(
                stock_code, user_position,
                progress_cb=lambda stage, msg, progress, **kwargs: None,
            )
            # 报告修正信息
            conflict_report = self.communication_engine.get_conflict_report(opinions)
            if conflict_report["has_conflict"]:
                yield {
                    "stage": "agent_revision",
                    "message": f"Round 2: {len(conflict_report['conflict_agents'])}个Agent执行修正分析: {', '.join(conflict_report['conflict_agents'])}",
                    "progress": 78,
                    "revision_summary": conflict_report["revision_summary"],
                }
        elif BATCH_INFERENCE_ENABLED:
            yield {"stage": "agent_start", "message": f"6大Agent批量推理中（{BATCH_INFERENCE_SIZE}个/批）...", "progress": 32}
            opinions = self._run_agents_batch(stock_code, user_position)
        else:
            yield {"stage": "agent_start", "message": "6大Agent并行分析中...", "progress": 32}
            opinions = self._run_agents_parallel(
                stock_code, user_position,
                progress_cb=lambda stage, msg, agent_id, progress: None,
            )

        # 逐个报告Agent结果（按固定顺序，让用户看到每个Agent的完成状态）
        agent_order = ["TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent", "RA-Agent"]
        agent_roles = {
            "TA-Agent": "技术面分析 — K线形态、均线系统、MACD/RSI/KDJ指标",
            "FA-Agent": "基本面分析 — 财务健康度、估值水平、盈利能力",
            "CA-Agent": "资金面分析 — 主力资金、北向资金、筹码分布",
            "SA-Agent": "情绪面分析 — 市场情绪、舆情热度、 crowd行为",
            "MA-Agent": "宏观分析 — 经济周期、行业景气度、政策环境",
            "RA-Agent": "风险分析 — 波动率、回撤控制、仓位管理",
        }
        for idx, agent_id in enumerate(agent_order):
            op = opinions.get(agent_id)
            if op:
                sig_map = {-1: "卖出", 0: "观望", 1: "买入"}
                sig_text = sig_map.get(op.signal, "观望")
                raw = op.raw_data or {}
                result_detail = {
                    "agent_id": agent_id,
                    "role": agent_roles.get(agent_id, ""),
                    "signal": op.signal,
                    "confidence": op.confidence,
                    "reasoning": op.reasoning[:200] + "..." if len(op.reasoning) > 200 else op.reasoning,
                    "key_factors": op.key_factors[:5] if op.key_factors else [],
                    "risk_flags": op.risk_flags[:3] if op.risk_flags else [],
                    "is_revision": op.is_revision,
                    "original_signal": op.original_signal,
                    "revision_round": op.revision_round,
                }
                # 各Agent特有分析指标
                if agent_id == "TA-Agent":
                    result_detail["indicators"] = {
                        "chart_patterns": raw.get("chart_patterns", []),
                        "trend_direction": raw.get("trend_direction", ""),
                        "target_price_low": raw.get("target_price_low"),
                        "target_price_high": raw.get("target_price_high"),
                        "stop_loss": raw.get("stop_loss"),
                    }
                elif agent_id == "FA-Agent":
                    result_detail["indicators"] = {
                        "fundamental_score": raw.get("fundamental_score"),
                        "sub_scores": raw.get("sub_scores", {}),
                        "valuation_gap": raw.get("valuation_gap", ""),
                    }
                elif agent_id == "CA-Agent":
                    result_detail["indicators"] = {
                        "capital_score": raw.get("capital_score"),
                        "smart_money_direction": raw.get("smart_money_direction", ""),
                        "retail_vs_institutional": raw.get("retail_vs_institutional", ""),
                    }
                elif agent_id == "SA-Agent":
                    result_detail["indicators"] = {
                        "sentiment_index": raw.get("sentiment_index"),
                        "sentiment_percentile": raw.get("sentiment_percentile"),
                        "crowd_behavior": raw.get("crowd_behavior", ""),
                    }
                elif agent_id == "MA-Agent":
                    result_detail["indicators"] = {
                        "market_cycle": raw.get("market_cycle", ""),
                        "sector_outlook": raw.get("sector_outlook", ""),
                        "macro_signal": raw.get("macro_signal"),
                    }
                elif agent_id == "RA-Agent":
                    result_detail["indicators"] = {
                        "risk_level": raw.get("risk_level"),
                        "max_position_pct": raw.get("max_position_pct"),
                        "risk_reward_ratio": raw.get("risk_reward_ratio"),
                        "black_scenarios": raw.get("black_scenarios", []),
                        "stress_test": raw.get("stress_test", {}),
                        "portfolio_risk": raw.get("portfolio_risk", {}),
                        "dynamic_stop_loss": raw.get("dynamic_stop_loss", {}),
                    }
                yield {
                    "stage": "agent",
                    "agent_id": agent_id,
                    "message": f"{agent_id} 分析完成 → {sig_text} (置信度 {op.confidence:.0%})",
                    "progress": 35 + idx * 8,
                    "agent_result": result_detail
                }

        # Phase 3: 决策引擎
        yield {"stage": "engine", "message": "决策引擎加权投票与风险过滤...", "progress": 85}
        ma_opinion = opinions.get("MA-Agent")
        market_cycle = ""
        weight_adj = None
        if ma_opinion:
            market_cycle = ma_opinion.raw_data.get("market_cycle", "")
            weight_adj = ma_opinion.raw_data.get("recommended_weight_adjustment", {})

        weighted = self.decision_engine.compute_weighted_decision(
            opinions, market_cycle, weight_adj
        )
        ra_opinion = opinions.get("RA-Agent")
        filtered = self.decision_engine.apply_risk_filter(weighted, ra_opinion)

        # Phase 4: Chairman 综合决策
        yield {"stage": "chairman", "message": "Chairman 综合各方观点形成最终决策...", "progress": 90}
        final_decision = self._chairman_decide(
            stock_code, snapshot, opinions, filtered, user_position
        )

        processing_time = round(time.time() - start_time, 2)

        # Phase 5: 组装决策包
        yield {"stage": "package", "message": "组装决策报告...", "progress": 98}
        package = DecisionPackage(
            stock_code=stock_code,
            stock_name=stock_name or snapshot.stock_name,
            current_price=snapshot.current_price,
            decision_date=datetime.now().strftime("%Y-%m-%d"),
            decision_time=datetime.now().strftime("%H:%M:%S"),
            market_cycle=market_cycle,
            opinions={k: v.to_dict() for k, v in opinions.items()},
            final_decision=final_decision,
            disclaimer=DISCLAIMER.strip(),
            processing_time_seconds=processing_time,
            data_summary=snapshot.to_summary(),
            data_quality=snapshot.data_quality,
        )

        logger.info(f"股票 {stock_code} 流式诊断完成，耗时 {processing_time}s")
        yield {"stage": "result", "message": "诊断完成", "progress": 100, "data": package.model_dump()}

    def run_diagnosis_background(
        self,
        stock_code: str,
        progress_queue: "queue.Queue",
        stock_name: str = "",
        market_type: Optional[str] = None,
        user_position: Optional[Dict] = None,
        model_params: Optional[Dict[str, Any]] = None,
        task_id: str = "",
    ) -> None:
        """
        后台线程执行诊断，将 SSE 事件推入 progress_queue + TaskTracker buffer。

        Queue 协议：
        - 每个事件是一个 dict
        - 最后一个事件为 None（哨兵，表示完成）
        - 异常时推送 {"stage": "error", ...} 再推送 None

        如果提供了 task_id，同步推送事件到 TaskTracker 的持久化 buffer，
        支持客户端断连后重连恢复。
        """
        from agent.core.task_tracker import task_tracker

        try:
            for event in self.run_diagnosis_stream(
                stock_code=stock_code,
                stock_name=stock_name,
                market_type=market_type,
                user_position=user_position,
                model_params=model_params,
            ):
                progress_queue.put(event)
                if task_id:
                    task_tracker.push_event(task_id, event)
        except Exception as e:
            logger.exception(f"后台诊断异常: {stock_code}")
            err_event = {"stage": "error", "message": str(e), "code": "BACKGROUND_ERROR"}
            progress_queue.put(err_event)
            if task_id:
                task_tracker.push_event(task_id, err_event)
                task_tracker.error_task(task_id)
        finally:
            progress_queue.put(None)  # 哨兵：流结束
            if task_id:
                task_tracker.finish_task(task_id)

    def _fetch_multi_timeframe_data(self, stock_code: str, daily_df: Any) -> Dict[str, Any]:
        """
        获取并计算多时间框架数据
        
        Returns:
            {
                "multi_timeframe": {...},
                "support_resistance": {...},
                "chart_patterns": [...],
                "ta_score": {...},
            }
        """
        result = {
            "multi_timeframe": {},
            "support_resistance": {},
            "chart_patterns": [],
            "ta_score": {},
        }
        
        try:
            # 1. 获取多周期K线
            tf_klines = self.stock_tool.get_kline_multi_timeframe(stock_code)
            
            # 2. 计算多周期指标
            result["multi_timeframe"] = self.indicator_tool.compute_multi_timeframe(
                daily_df=daily_df,
                weekly_df=tf_klines.get("weekly"),
                hourly_df=tf_klines.get("hourly"),
            )
            
            # 3. 计算支撑压力位
            daily_indicators = self.indicator_tool.compute_all(daily_df)
            if "error" not in daily_indicators:
                result["support_resistance"] = self.indicator_tool.compute_support_resistance(
                    daily_df, daily_indicators
                )
                
                # 4. 形态识别
                result["chart_patterns"] = self.indicator_tool.detect_chart_patterns(daily_df)
                
                # 5. 多因子评分
                result["ta_score"] = self.indicator_tool.compute_ta_score(
                    daily_indicators, result["support_resistance"]
                )
        except Exception as e:
            logger.warning(f"多时间框架数据获取/计算失败 [{stock_code}]: {e}")
        
        return result

    def _create_snapshot(self, stock_code: str, stock_name: str) -> StockSnapshot:
        """创建股票数据快照 — 全真实数据，处理缺失"""
        return self._create_snapshot_internal(stock_code, stock_name)

    def _create_snapshot_internal(
        self, stock_code: str, stock_name: str,
        progress_cb: Optional[Callable] = None,
    ) -> StockSnapshot:
        """创建股票数据快照 — 支持进度回调和并行获取"""
        logger.info(f"正在获取 {stock_code} 的数据快照...")

        def _prog(stage: str, msg: str, pct: int):
            if progress_cb:
                progress_cb(stage, msg, pct)

        # ── Phase 1: 并行获取所有数据源 ──
        _prog("data", "并行获取多源数据（K线、基本面、资金流向、市场情绪、宏观环境）...", 5)

        def _fetch_kline():
            df = self.stock_tool.get_kline(stock_code, days=120)
            if df is None:
                logger.warning(f"无法获取 {stock_code} 的历史行情数据，所有K线数据源均不可用，将继续使用其他维度数据")
            return df
        
        def _fetch_index_kline():
            """获取沪深300指数K线用于计算真实Beta"""
            try:
                # 沪深300指数代码: 000300
                index_df = self.stock_tool.get_kline("000300", days=120)
                if index_df is not None and len(index_df) >= 30:
                    return index_df
            except Exception as e:
                logger.debug(f"沪深300指数K线获取失败: {e}")
            return None

        def _fetch_fundamentals():
            result = self.stock_tool.get_fundamentals(stock_code)
            return result or {"stock_code": stock_code, "_meta": {"sources_succeeded": []}}

        def _fetch_fund_flow():
            result = self.stock_tool.get_fund_flow(stock_code)
            return result or {"stock_code": stock_code, "_meta": {"sources_succeeded": []}}

        def _fetch_sentiment_bundle():
            sentiment_data = self.stock_tool.get_sentiment_data(stock_code) or {}
            sentiment_news = self.sentiment_tool.fetch_stock_news(stock_code)
            sentiment_analysis = self.sentiment_tool.analyze_news(sentiment_news)
            sentiment_data.update({
                "news": sentiment_news[:5] if sentiment_news else [],
                "news_analysis": sentiment_analysis,
                "sector_heat": self.sentiment_tool.compute_sector_heat(f"{stock_code}板块"),
            })
            return sentiment_data

        def _fetch_market_context():
            result = self.stock_tool.get_market_context(stock_code)
            return result or {"_meta": {"sources_succeeded": []}}

        def _fetch_macro():
            result = self.stock_tool.get_macro_data()
            return result or {"_meta": {"sources_succeeded": []}}

        # 主任务：6 路数据源 fan-out + 指数K线（用于Beta计算）
        fetchers = {
            "kline": _fetch_kline,
            "fundamentals": _fetch_fundamentals,
            "fund_flow": _fetch_fund_flow,
            "sentiment": _fetch_sentiment_bundle,
            "market_context": _fetch_market_context,
            "macro": _fetch_macro,
            "index_kline": _fetch_index_kline,
        }

        parallel_results: Dict[str, Any] = {}
        kline_df: Any = None
        ind_future: Any = None
        risk_future: Any = None
        futures: Dict[Any, str] = {
            _ORCHESTRATOR_EXECUTOR.submit(fn): name for name, fn in fetchers.items()
        }

        # Phase 1: 等待所有数据源完成
        index_kline_df: Any = None
        for future in as_completed(futures):
            name = futures.pop(future)
            try:
                result = future.result()
                if name == "kline":
                    kline_df = result
                if name == "index_kline":
                    index_kline_df = result
                parallel_results[name] = result
                _prog("data", f"{name} 获取完成", 5 + len(parallel_results) * 3)
            except Exception as e:
                logger.warning(f"{name} 获取失败: {e}")
                if name == "kline":
                    kline_df = None
                elif name == "fundamentals":
                    parallel_results[name] = {"stock_code": stock_code, "_meta": {"sources_succeeded": []}}
                elif name == "fund_flow":
                    parallel_results[name] = {"stock_code": stock_code, "_meta": {"sources_succeeded": []}}
                elif name in ("sentiment", "market_context", "macro"):
                    parallel_results[name] = {"_meta": {"sources_succeeded": []}}

        # Phase 2: kline 和指数数据都到位后，提交指标+风险计算+多周期分析
        if kline_df is not None:
            ind_future = _ORCHESTRATOR_EXECUTOR.submit(
                self.indicator_tool.compute_all, kline_df
            )
            risk_future = _ORCHESTRATOR_EXECUTOR.submit(
                self.indicator_tool.compute_risk_metrics, kline_df, index_kline_df
            )
            
            # 新增：多周期K线获取与指标计算（并行）
            multi_tf_future = _ORCHESTRATOR_EXECUTOR.submit(
                self._fetch_multi_timeframe_data, stock_code, kline_df
            )
            
            try:
                parallel_results["indicators"] = ind_future.result()
                _prog("data", "indicators 计算完成", 30)
            except Exception as e:
                logger.warning(f"指标计算失败: {e}")
                parallel_results["indicators"] = {}
            try:
                parallel_results["risk_metrics"] = risk_future.result()
                _prog("data", "risk_metrics 计算完成", 32)
            except Exception as e:
                logger.warning(f"风险指标计算失败: {e}")
                parallel_results["risk_metrics"] = {}
            
            # 新增：合并多周期分析结果到 indicators
            try:
                multi_tf_data = multi_tf_future.result(timeout=15)
                if multi_tf_data:
                    indicators = parallel_results.get("indicators", {})
                    if isinstance(indicators, dict) and "error" not in indicators:
                        indicators["multi_timeframe"] = multi_tf_data.get("multi_timeframe", {})
                        indicators["support_resistance"] = multi_tf_data.get("support_resistance", {})
                        indicators["chart_patterns"] = multi_tf_data.get("chart_patterns", [])
                        indicators["ta_score"] = multi_tf_data.get("ta_score", {})
                        parallel_results["indicators"] = indicators
                        _prog("data", "多周期技术分析完成", 33)
            except Exception as e:
                logger.warning(f"多周期分析失败: {e}")

        fundamentals = parallel_results.get("fundamentals", {})
        fund_flow = parallel_results.get("fund_flow", {})
        sentiment_data = parallel_results.get("sentiment", {})
        market_context = parallel_results.get("market_context", {})
        macro_data = parallel_results.get("macro", {})

        # ── 情绪数据增强：从K线和资金流向补充情绪指标 ──
        # 当爬虫获取不到专门的情绪数据时，从已有真实行情数据推导情绪状态
        if not sentiment_data or isinstance(sentiment_data, dict) and len(sentiment_data) <= 3:
            sentiment_data = {}

        # 1. 从K线计算情绪指标
        if kline_df is not None and "kline_derived" not in str(sentiment_data.get("analysis_method", "")):
            try:
                kline_sentiment = self.sentiment_tool.compute_sentiment_from_kline(kline_df)
                if kline_sentiment and not kline_sentiment.get("_error"):
                    sentiment_data["kline_sentiment"] = kline_sentiment
                    # 将核心情绪指标提升到顶层，方便 SA_Agent 直接使用
                    sentiment_data["sentiment_index"] = kline_sentiment.get("sentiment_index", 0.0)
                    sentiment_data["sentiment_percentile"] = kline_sentiment.get("sentiment_percentile", 50)
                    sentiment_data["crowd_behavior"] = kline_sentiment.get("crowd_behavior", "未知")
                    sentiment_data["volatility_20d"] = kline_sentiment.get("volatility_annual", 0.0)
                    logger.info(f"情绪指标已从K线推导: sentiment_index={sentiment_data['sentiment_index']}, "
                               f"crowd={sentiment_data['crowd_behavior']}")
            except Exception as e:
                logger.warning(f"K线情绪指标计算失败: {e}")

        # 2. 从资金流向计算情绪指标
        if fund_flow and "fund_flow_derived" not in str(sentiment_data.get("analysis_method", "")):
            try:
                ff_sentiment = self.sentiment_tool.compute_sentiment_from_fund_flow(fund_flow)
                if ff_sentiment and not ff_sentiment.get("_error"):
                    sentiment_data["fund_flow_sentiment"] = ff_sentiment
                    logger.info(f"情绪指标已从资金流向推导: main_force={ff_sentiment.get('main_force_emotion')}")
            except Exception as e:
                logger.warning(f"资金流向情绪指标计算失败: {e}")

        # 指标和风险已在 fan-out 中并行计算完成，直接取结果
        indicators = parallel_results.get("indicators", {})
        risk_metrics = parallel_results.get("risk_metrics", {})

        current_price = indicators.get("current_price", 0.0)
        if current_price == 0 and kline_df is not None and not kline_df.empty:
            current_price = float(kline_df["close"].iloc[-1])

        # 生成数据质量报告
        _prog("data", "数据质量校验完成", 28)
        data_quality = self.stock_tool.get_data_quality_report(stock_code)

        # 确定真实股票名称：优先从 fundamentals，其次从 Sina 实时行情
        resolved_name = stock_name
        if not resolved_name or "模拟" in resolved_name or "mock" in resolved_name.lower():
            resolved_name = fundamentals.get("company_name", "")
        if not resolved_name or "模拟" in resolved_name or "mock" in resolved_name.lower():
            # 尝试从 Sina 行情获取名称
            try:
                sina_name = self.stock_tool.get_fundamentals(stock_code)
                if sina_name and isinstance(sina_name, dict):
                    resolved_name = sina_name.get("company_name", "")
            except Exception:
                pass
        if not resolved_name or "模拟" in resolved_name or "mock" in resolved_name.lower():
            resolved_name = stock_code
        
        return StockSnapshot(
            stock_code=stock_code,
            stock_name=resolved_name,
            current_price=current_price,
            kline_df=kline_df,
            indicators=indicators,
            fundamentals=fundamentals,
            fund_flow=fund_flow,
            market_context=market_context,
            sentiment_data=sentiment_data,
            macro_data=macro_data,
            risk_metrics=risk_metrics,
            data_quality=data_quality,
        )
    
    def _run_agents_parallel(
        self,
        stock_code: str,
        user_position: Optional[Dict] = None,
        progress_cb: Optional[Callable] = None,
    ) -> Dict[str, AgentOpinion]:
        """并行运行所有分析Agent，支持进度回调"""
        snapshot = self.blackboard.get_snapshot(stock_code)
        opinions = {}

        def _prog(agent_id: str, done: int, total: int):
            if progress_cb:
                base_pct = 30
                step = 50 // total
                pct = base_pct + done * step
                progress_cb("agent", f"{agent_id} 分析完成", agent_id=agent_id, progress=pct)

        if AGENT_PARALLEL:
            futures = {}
            for agent_id, agent in self.agents.items():
                future = _ORCHESTRATOR_EXECUTOR.submit(self._run_single_agent, agent, snapshot, user_position)
                futures[future] = agent_id

            completed = 0
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    opinion = future.result(timeout=120)
                    opinions[agent_id] = opinion
                    self.blackboard.submit_opinion(stock_code, opinion)
                    completed += 1
                    _prog(agent_id, completed, len(futures))
                    logger.info(f"{agent_id} 分析完成: signal={opinion.signal}, confidence={opinion.confidence}")
                except Exception as e:
                    logger.warning(f"{agent_id} 分析失败: {e}")
                    opinions[agent_id] = self._fallback_opinion(agent_id, str(e))
                    completed += 1
                    _prog(agent_id, completed, len(futures))
        else:
            for idx, (agent_id, agent) in enumerate(self.agents.items()):
                try:
                    opinion = self._run_single_agent(agent, snapshot, user_position)
                    opinions[agent_id] = opinion
                    self.blackboard.submit_opinion(stock_code, opinion)
                    _prog(agent_id, idx + 1, len(self.agents))
                except Exception as e:
                    logger.error(f"{agent_id} 分析失败: {e}")
                    opinions[agent_id] = self._fallback_opinion(agent_id, str(e))
                    _prog(agent_id, idx + 1, len(self.agents))

        return opinions

    def _run_agents_batch(
        self,
        stock_code: str,
        user_position: Optional[Dict] = None,
    ) -> Dict[str, AgentOpinion]:
        """
        批量推理运行所有 Agent — 透明拦截 _call_llm，合并为 2 次 API 调用

        无需修改子类，保持原有 analyze() 流程不变。
        """
        snapshot = self.blackboard.get_snapshot(stock_code)

        from agent.tools.batch_llm_client import BatchAgentRunner, BatchLLMClient

        if self._batch_runner is None:
            self._batch_runner = BatchAgentRunner(BatchLLMClient(self.llm))

        opinions = self._batch_runner.run(self.agents, snapshot, user_position)

        # 将结果提交到黑板
        for agent_id, opinion in opinions.items():
            self.blackboard.submit_opinion(stock_code, opinion)
            logger.info(f"{agent_id} 批量分析完成: signal={opinion.signal}, confidence={opinion.confidence}")

        return opinions

    def _run_single_agent(
        self,
        agent,
        snapshot: StockSnapshot,
        user_position: Optional[Dict] = None,
    ) -> AgentOpinion:
        """运行单个Agent — 集成 Agent 缓存"""
        from agent.core.agent_cache import AgentCache
        cache = AgentCache()

        # 1. 尝试命中缓存
        cached = cache.get(snapshot.stock_code, agent.agent_id, snapshot)
        if cached is not None:
            logger.info(f"{agent.agent_id} 缓存命中，跳过 LLM 调用")
            return cached

        # 2. 执行分析
        opinion = agent.analyze(snapshot, user_position)

        # 3. 存入缓存
        cache.set(snapshot.stock_code, agent.agent_id, snapshot, opinion)
        return opinion

    def _fallback_opinion(self, agent_id: str, error_msg: str) -> AgentOpinion:
        """Agent失败时的降级观点"""
        return AgentOpinion(
            agent_id=agent_id,
            signal=0,
            confidence=0.3,
            reasoning=f"分析过程中发生错误: {error_msg}。系统降级为观望。",
            key_factors=["分析异常"],
            risk_flags=[f"{agent_id} 分析失败"],
            raw_data={"error": error_msg},
        )
    
    def _chairman_decide(
        self,
        stock_code: str,
        snapshot: StockSnapshot,
        opinions: Dict[str, AgentOpinion],
        filtered_decision: Dict[str, Any],
        user_position: Optional[Dict] = None,
    ) -> ChairmanDecision:
        """
        Chairman 综合决策
        优先使用LLM生成高质量决策，失败时回退到规则引擎
        
        增强：冲突检测 → 组织辩论 → 根据辩论结果修正决策
        """
        # 1. 检测Agent间冲突并运行辩论
        debate_results = self.debate_engine.run_all_debates(
            opinions=opinions,
            context=f"股票: {stock_code} ({snapshot.stock_name})",
            max_debates=2,
        )
        
        # 2. 根据辩论结果调整加权决策
        adjusted_decision = self._apply_debate_results(filtered_decision, opinions, debate_results)
        
        try:
            return self._chairman_llm_decide(
                stock_code, snapshot, opinions, adjusted_decision, user_position, debate_results
            )
        except Exception as e:
            logger.error(f"Chairman LLM决策失败: {e}，回退到规则引擎")
            return self._chairman_rule_decide(adjusted_decision, opinions, debate_results)
    
    def _chairman_llm_decide(
        self,
        stock_code: str,
        snapshot: StockSnapshot,
        opinions: Dict[str, AgentOpinion],
        filtered_decision: Dict[str, Any],
        user_position: Optional[Dict] = None,
        debate_results: Optional[List[Any]] = None,
    ) -> ChairmanDecision:
        """使用LLM进行Chairman决策（含辩论结果输入）"""
        # 构建Prompt上下文
        context_parts = [
            f"股票代码: {stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            f"用户持仓: {user_position or '无持仓'}",
            "",
            "=== 各Agent观点 ===",
        ]
        
        for agent_id, op in opinions.items():
            context_parts.append(f"\n--- {agent_id} ---")
            context_parts.append(f"Signal: {op.signal} | Confidence: {op.confidence}")
            context_parts.append(f"Reasoning: {op.reasoning}")
            context_parts.append(f"Key Factors: {', '.join(op.key_factors)}")
            context_parts.append(f"Risk Flags: {', '.join(op.risk_flags)}")
            if op.raw_data:
                # 添加Agent特有的关键字段
                for k, v in op.raw_data.items():
                    if k not in ("reasoning", "key_factors", "risk_flags"):
                        context_parts.append(f"{k}: {v}")
        
        # 添加辩论结果到上下文
        if debate_results:
            context_parts.extend([
                "",
                "=== Agent间辩论结果 ===",
            ])
            for dr in debate_results:
                context_parts.append(f"- 主题: {dr.topic}")
                context_parts.append(f"  总结: {dr.summary}")
                if dr.winner:
                    context_parts.append(f"  占优方: {dr.winner} (置信度调整 +{dr.confidence_delta:.2f})")
                else:
                    context_parts.append(f"  结果: 势均力敌，无明确占优方")
        
        context_parts.extend([
            "",
            "=== 决策引擎输出 ===",
            f"Preliminary Signal: {filtered_decision.get('preliminary_signal')}",
            f"Final Signal: {filtered_decision.get('final_signal')}",
            f"Weighted Score: {filtered_decision.get('weighted_score')}",
            f"Overall Confidence: {filtered_decision.get('overall_confidence')}",
            f"Risk Level: {filtered_decision.get('risk_level')}",
            f"Max Position: {filtered_decision.get('max_position_pct')}",
        ])
        
        user_prompt = "\n".join(context_parts)
        
        # 加载Chairman系统提示词
        prompt_path = "agent/prompts/system/chairman.md"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = self._default_chairman_prompt()
        
        response = self.chairman_llm.chat(
            system=system_prompt,
            user=user_prompt,
            json_mode=True,
            model=CHAIRMAN_MODEL,
            override_config=self.model_params if self.model_params else None,
        )
        
        # 解析并校验
        try:
            decision = ChairmanDecision(**response)
        except Exception as e:
            logger.warning(f"Chairman响应校验失败: {e}，使用原始数据")
            decision = ChairmanDecision(
                decision=filtered_decision.get("final_signal", 0),
                confidence=filtered_decision.get("overall_confidence", 0.5),
                reasoning=str(response.get("reasoning", "Chairman响应异常，使用规则引擎结果")),
            )
        
        # 风险过滤：不能超过RA建议的最大仓位
        ra_op = opinions.get("RA-Agent")
        if ra_op and ra_op.raw_data:
            max_pos = ra_op.raw_data.get("max_position_pct", 0.5)
            if decision.position_pct > max_pos:
                decision.position_pct = round(max_pos * 0.8, 3)
        
        # confidence < 0.6 强制观望
        if decision.confidence < 0.60:
            decision.decision = 0
        
        return decision
    
    def _apply_debate_results(
        self,
        filtered: Dict[str, Any],
        opinions: Dict[str, AgentOpinion],
        debate_results: List[Any],
    ) -> Dict[str, Any]:
        """
        根据辩论结果调整加权决策。
        
        规则:
        - 若辩论有明确 winner，提升其对应 signal 的权重
        - 若辩论 consensus_reached=True，整体置信度 +0.05
        - 若冲突严重且无共识，整体置信度 -0.05（表示不确定性）
        """
        adjusted = dict(filtered)
        if not debate_results:
            return adjusted
        
        confidence = adjusted.get("overall_confidence", 0.5)
        
        for dr in debate_results:
            if dr.winner and dr.winner in opinions:
                winner_op = opinions[dr.winner]
                # 若 winner 建议买入/卖出，轻微推动 final_signal
                if winner_op.signal != 0 and adjusted.get("final_signal", 0) == 0:
                    adjusted["final_signal"] = winner_op.signal
                    adjusted["preliminary_signal"] = winner_op.signal
            
            if dr.consensus_reached:
                confidence += 0.05
            elif not dr.winner:
                confidence -= 0.05
        
        adjusted["overall_confidence"] = round(max(0.3, min(0.95, confidence)), 3)
        adjusted["debate_adjusted"] = True
        return adjusted
    
    def _chairman_rule_decide(
        self,
        filtered: Dict[str, Any],
        opinions: Dict[str, AgentOpinion],
        debate_results: Optional[List[Any]] = None,
    ) -> ChairmanDecision:
        """Chairman规则引擎降级方案（含辩论结果）"""
        final_signal = filtered.get("final_signal", 0)
        confidence = filtered.get("overall_confidence", 0.5)
        max_pos = filtered.get("max_position_pct", 0.10)
        
        # 收集共识因子
        consensus = []
        all_factors = []
        for op in opinions.values():
            all_factors.extend(op.key_factors)
        from collections import Counter
        factor_counts = Counter(all_factors)
        for factor, count in factor_counts.most_common(3):
            if count >= 2:
                consensus.append(factor)
        
        # 根据辩论结果构建分歧观点
        dissenting = []
        if debate_results:
            for dr in debate_results:
                if not dr.consensus_reached and not dr.winner:
                    dissenting.append(dr.topic)
        
        reasoning = filtered.get("risk_override") or "规则引擎综合决策"
        if debate_results:
            reasoning += f"（参考了{len(debate_results)}场Agent辩论）"
        
        return ChairmanDecision(
            decision=final_signal,
            confidence=confidence,
            position_pct=round(max_pos * 0.8, 3) if final_signal == 1 else 0,
            time_horizon="2-4周",
            expected_return_pct=10.0 if final_signal == 1 else 0,
            risk_adjusted_score=50,
            reasoning=reasoning,
            consensus_factors=consensus,
            dissenting_views=dissenting,
            scenario_analysis={
                "bull": {"probability": 0.25, "return_pct": 20},
                "base": {"probability": 0.50, "return_pct": 8},
                "bear": {"probability": 0.25, "return_pct": -10},
            },
            execution_plan=["建议观望"] if final_signal == 0 else ["分批建仓", "严格止损"],
        )
    
    def _default_chairman_prompt(self) -> str:
        """默认Chairman提示词（当文件不存在时使用）"""
        return """# Role
你是多智能体投研系统的投资委员会主席，负责综合各方观点做出最终投资决策。

# Context
你将收到所有分析师的观点和风险官的评估，请做出概率最优、风险可控的决策。

# Output Format (Strict JSON)
{
  "decision": 1,
  "confidence": 0.75,
  "position_pct": 0.10,
  "target_price": 20.00,
  "stop_loss": 15.00,
  "time_horizon": "2-4周",
  "expected_return_pct": 12.0,
  "risk_adjusted_score": 65,
  "reasoning": "综合决策理由",
  "consensus_factors": ["共识因子1"],
  "dissenting_views": [],
  "scenario_analysis": {
    "bull": {"probability": 0.25, "return_pct": 25},
    "base": {"probability": 0.50, "return_pct": 10},
    "bear": {"probability": 0.25, "return_pct": -8}
  },
  "execution_plan": ["执行步骤1"]
}

# Constraints
- decision 只能是 -1, 0, 1
- position_pct 不得超过风险官建议
- confidence < 0.60 时 decision 必须为 0
"""
