"""
MASS 股票预测引擎 v2.3
Enhanced: 分层 Prompt + 可调参数 + 置信度校准 + 多模型 fallback

性能:
- Token 消耗减 30-50%（按 horizon 分层裁剪数据）
- 系统提示词文件化，支持热更新
- 多模型 fallback，主模型故障自动切换

模型参数完全可调:
- temperature: 0.0 ~ 2.0
- top_p: 0.0 ~ 1.0
- max_tokens: 1024 ~ 8192
- frequency_penalty: -2.0 ~ 2.0
- presence_penalty: -2.0 ~ 2.0
"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from loguru import logger

from agent.tools.llm_client import LLMClient, LLMConfig
from agent.core.blackboard import StockSnapshot
from agent.models.prediction import PredictionResult
from config import PREDICTION_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS

# ── 默认系统提示词（当文件缺失时使用）──
_DEFAULT_SYSTEM_PROMPT = """# Role
你是一位顶级量化投资策略分析师，擅长基于多维度真实数据进行股票走势概率预测。

# Core Capabilities
- 技术面分析：K线形态、均线系统、成交量、技术指标
- 基本面分析：估值水平（PE/PB/ROE）、盈利能力、成长性
- 资金面分析：主力资金流向、北向资金、融资融券
- 情绪面分析：市场新闻情绪、板块热度
- 宏观面分析：政策环境、市场周期、行业景气度

# Prediction Principles
1. **严格基于输入数据**：绝不编造不存在的数据或事件
2. **概率化表达**：给出上涨/下跌/震荡的概率分布
3. **量化目标价**：基于技术面支撑压力位给出目标价区间
4. **风险前置**：明确列出主要风险因素和潜在催化事件
5. **推理透明**：详细说明预测逻辑链条

# Output Format (Strict JSON)
{
  "direction": "上涨",
  "confidence": 0.72,
  "target_price_low": 18.5,
  "target_price_high": 22.0,
  "stop_loss": 17.0,
  "risk_reward_ratio": 2.5,
  "holding_period_days": 5,
  "probability_up": 0.60,
  "probability_down": 0.25,
  "probability_sideways": 0.15,
  "key_drivers": ["因子1"],
  "risk_factors": ["风险1"],
  "catalyst_events": ["事件1"],
  "reasoning": "..."
}

# Constraints
- direction 只能是 "上涨"、"下跌"、"震荡" 之一
- confidence 范围 0.0-1.0
- probabilities 三项之和必须为 1.0
"""


class PredictionEngine:
    """
    股票预测引擎 v2.3

    输入: StockSnapshot + 可调参数
    输出: PredictionResult（结构化预测结果）

    增强特性:
    - 分层 Prompt：short/medium/long 各自裁剪数据，减少 token 30-50%
    - 风险偏好调节：conservative/moderate/aggressive
    - 投资风格调节：swing/trend/value
    - 置信度校准：基于数据质量加权
    - 多模型 fallback：主模型失败自动切换备模型
    """

    # Prompt 动态注入段的占位符（%s 占位符在 _build_injected_section 中填充）
    _RISK_TOLERANCE_SECTIONS: Dict[str, str] = {
        "conservative": (
            "# Risk Tolerance: CONSERVATIVE\n"
            "- Prioritize downside protection, weight probability_down +10%%\n"
            "- Narrow target price range by 15%%, tighter stop-loss\n"
            "- Only give directional call when 2+ dimensions align\n"
            "- Require risk-reward ratio > 2.5"
        ),
        "moderate": (
            "# Risk Tolerance: MODERATE\n"
            "- Balance risk and return, reflect data signals objectively\n"
            "- Target price range based on 1 standard deviation\n"
            "- At least 1 supporting dimension to give directional call\n"
            "- Require risk-reward ratio > 1.5"
        ),
        "aggressive": (
            "# Risk Tolerance: AGGRESSIVE\n"
            "- Prioritize upside capture, weight probability_up +10%%\n"
            "- Widen target price range by 20%%, tolerate higher volatility\n"
            "- Single strong signal sufficient for directional call\n"
            "- Require risk-reward ratio > 1.0"
        ),
    }

    _INVESTMENT_STYLE_SECTIONS: Dict[str, str] = {
        "swing": (
            "# Investment Style: SWING TRADING\n"
            "- Analysis weights: technical(40%%) + fund flow(30%%) + sentiment(20%%) + fundamentals(10%%)\n"
            "- Focus on short-term catalysts and volume-price coordination\n"
            "- Strict stop-loss discipline, holding_period_days: 1-10"
        ),
        "trend": (
            "# Investment Style: TREND FOLLOWING\n"
            "- Analysis weights: moving averages(35%%) + macro cycle(25%%) + fund flow(20%%) + patterns(20%%)\n"
            "- Focus on mid-to-long-term trend strength and persistence\n"
            "- Hold while trend intact, holding_period_days: 10-60"
        ),
        "value": (
            "# Investment Style: VALUE INVESTING\n"
            "- Analysis weights: fundamentals(45%%) + valuation(25%%) + macro(15%%) + technical(15%%)\n"
            "- Focus on PE/PB percentile, ROE, revenue growth\n"
            "- Low valuation + catalyst = buy signal, holding_period_days: 30-90"
        ),
    }

    _HORIZON_LABELS = {
        "short": "1-5 个交易日",
        "medium": "1-4 周",
        "long": "1-3 个月",
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self._fallback_llm: Optional[LLMClient] = None
        self._system_prompt: Optional[str] = None
        logger.info("PredictionEngine v2.3 初始化完成")

    # ══════════════════════════════════════════════════════════════════════
    # System Prompt (热更新)
    # ══════════════════════════════════════════════════════════════════════

    def _load_system_prompt(self) -> str:
        """从文件加载系统提示词，缺失时使用内嵌默认值"""
        if self._system_prompt is not None:
            return self._system_prompt
        prompt_path = Path("agent/prompts/system/prediction.md")
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._system_prompt = f.read()
            logger.debug("从文件加载预测系统提示词")
        else:
            self._system_prompt = _DEFAULT_SYSTEM_PROMPT
            logger.warning("prediction.md 不存在，使用内嵌默认提示词")
        return self._system_prompt

    def _build_system_prompt(
        self,
        risk_tolerance: str,
        investment_style: str,
        confidence_threshold: float,
    ) -> str:
        """构建带动态注入段的系统提示词"""
        template = self._load_system_prompt()

        risk_section = self._RISK_TOLERANCE_SECTIONS.get(
            risk_tolerance,
            self._RISK_TOLERANCE_SECTIONS["moderate"],
        )
        style_section = self._INVESTMENT_STYLE_SECTIONS.get(
            investment_style,
            self._INVESTMENT_STYLE_SECTIONS["swing"],
        )

        if confidence_threshold >= 0.75:
            conf_section = (
                "# Confidence Standard: STRICT\n"
                f"- confidence < {confidence_threshold} 时 direction 必须为 \"不确定\"\n"
                "- 仅当多维度信号强烈共振时才给出高置信度\n"
                "- 宁可观望，不可过度自信"
            )
        elif confidence_threshold >= 0.55:
            conf_section = (
                "# Confidence Standard: STANDARD\n"
                f"- confidence < {confidence_threshold} 时 direction 必须为 \"不确定\"\n"
                "- 信号清晰时给出明确方向，模糊时保持观望"
            )
        else:
            conf_section = (
                "# Confidence Standard: RELAXED\n"
                f"- confidence < {confidence_threshold} 时 direction 为 \"不确定\"\n"
                "- 允许在较弱信号下给出方向性判断\n"
                "- 适合探索性分析场景"
            )

        conf_rule = (
            f"confidence < {confidence_threshold} 时 direction 必须为 \"不确定\""
        )

        return (
            template
            .replace("{RISK_TOLERANCE_SECTION}", risk_section)
            .replace("{INVESTMENT_STYLE_SECTION}", style_section)
            .replace("{CONFIDENCE_SECTION}", conf_section)
            .replace("{CONFIDENCE_THRESHOLD_RULE}", conf_rule)
        )

    # ══════════════════════════════════════════════════════════════════════
    # Main API
    # ══════════════════════════════════════════════════════════════════════

    def predict(
        self,
        stock_code: str,
        stock_name: str,
        snapshot: StockSnapshot,
        horizon: str = "short",
        risk_tolerance: str = "moderate",
        investment_style: str = "swing",
        confidence_threshold: float = 0.6,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        执行股票走势预测

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            snapshot: 股票数据快照（真实数据）
            horizon: 预测周期 short/medium/long
            risk_tolerance: 风险偏好 conservative/moderate/aggressive
            investment_style: 投资风格 swing/trend/value
            confidence_threshold: 置信度阈值 (0.4-0.8)
            model_params: 覆盖模型参数

        Returns:
            PredictionResult
        """
        if horizon not in ("short", "medium", "long"):
            raise ValueError(f"horizon 必须是 short/medium/long，收到: {horizon}")
        if risk_tolerance not in ("conservative", "moderate", "aggressive"):
            raise ValueError(f"risk_tolerance 无效: {risk_tolerance}")
        if investment_style not in ("swing", "trend", "value"):
            raise ValueError(f"investment_style 无效: {investment_style}")

        start_time = datetime.now()
        logger.info(
            f"预测 {stock_code} | horizon={horizon} "
            f"risk={risk_tolerance} style={investment_style} threshold={confidence_threshold}"
        )

        # 1. 构建分层 Prompt
        system_prompt = self._build_system_prompt(
            risk_tolerance, investment_style, confidence_threshold
        )
        user_prompt = self._build_prediction_prompt(
            stock_code, stock_name, snapshot, horizon
        )
        prompt_tokens = self._estimate_tokens(system_prompt, user_prompt)

        # 2. LLM 调用（带 fallback）
        model_used, fallback_used, response = self._call_with_fallback(
            system_prompt, user_prompt, model_params
        )

        # 3. 解析并校验
        result = self._parse_prediction(
            response, stock_code, stock_name, horizon
        )

        # 4. 置信度校准
        dq_factor = self._compute_quality_factor(snapshot)
        calibrated = round(result.confidence * dq_factor, 4)
        if calibrated < confidence_threshold:
            result.direction = "不确定"

        # 5. 回填元信息
        result.confidence_calibrated = calibrated
        result.data_quality_factor = round(dq_factor, 4)
        result.risk_tolerance = risk_tolerance
        result.investment_style = investment_style
        result.confidence_threshold = confidence_threshold
        result.model_used = model_used
        result.fallback_used = fallback_used
        result.model_params = model_params or {}
        result.prediction_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        result.prompt_tokens_estimated = prompt_tokens
        result.disclaimer = (
            "本预测基于历史数据和模型分析，不构成投资建议。股市有风险，投资需谨慎。"
        )

        logger.info(
            f"预测完成: {stock_code} → {result.direction} "
            f"(raw={result.confidence:.2f}, calibrated={calibrated:.2f}, "
            f"factor={dq_factor:.2f}, tokens≈{prompt_tokens})"
        )
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Multi-model Fallback
    # ══════════════════════════════════════════════════════════════════════

    def _call_with_fallback(
        self,
        system: str,
        user: str,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        主模型 → 备模型 fallback 链。
        Returns: (model_name, fallback_used, response_dict)
        """
        # 尝试主模型
        try:
            response = self.llm.chat(
                system=system,
                user=user,
                json_mode=True,
                model=PREDICTION_MODEL,
                override_config=override_config,
            )
            return PREDICTION_MODEL, False, response
        except Exception as e:
            logger.warning(f"主模型 {PREDICTION_MODEL} 失败: {e}")

        # 尝试备模型
        fallback_model = self._get_fallback_model()
        if fallback_model and fallback_model != PREDICTION_MODEL:
            logger.info(f"切换到备模型: {fallback_model}")
            try:
                response = self.llm.chat(
                    system=system,
                    user=user,
                    json_mode=True,
                    model=fallback_model,
                    override_config=override_config,
                )
                return fallback_model, True, response
            except Exception as e:
                logger.error(f"备模型 {fallback_model} 也失败: {e}")

        raise RuntimeError(
            f"所有预测模型调用失败: primary={PREDICTION_MODEL}, "
            f"fallback={fallback_model}"
        )

    def _get_fallback_model(self) -> Optional[str]:
        """获取备模型名称"""
        import os
        return os.getenv("FALLBACK_PREDICTION_MODEL", "").strip() or None

    # ══════════════════════════════════════════════════════════════════════
    # Layered Prompt Construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_prediction_prompt(
        self,
        stock_code: str,
        stock_name: str,
        snapshot: StockSnapshot,
        horizon: str,
    ) -> str:
        """
        构建预测用户提示词 — 按 horizon 分层裁剪数据
        """
        lines = [
            f"# 股票预测任务",
            f"",
            f"股票代码: {stock_code}",
            f"股票名称: {stock_name}",
            f"预测周期: {horizon} ({self._HORIZON_LABELS.get(horizon, horizon)})",
            f"当前价格: {snapshot.current_price}",
            f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
        ]

        # ── K线摘要（所有周期都需要，但 tail 不同）──
        if snapshot.kline_df is not None and not snapshot.kline_df.empty:
            df = snapshot.kline_df
            tail_map = {"short": 10, "medium": 30, "long": 120}
            recent = df.tail(tail_map.get(horizon, 20))
            lines.extend([
                f"## 近期K线摘要（最近{len(recent)}个交易日）",
                f"- 最新收盘价: {df['close'].iloc[-1]}",
                f"- {len(recent)}日最高: {recent['high'].max()}",
                f"- {len(recent)}日最低: {recent['low'].min()}",
                f"- {len(recent)}日均量: {recent['volume'].mean():.0f}",
            ])
            if len(df) >= 20:
                chg = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                lines.append(f"- 20日涨跌幅: {chg:.2f}%")
            if "pct_change" in df.columns:
                lines.append(f"- 最新涨跌幅: {df['pct_change'].iloc[-1]}%")
            lines.append("")

        # ── 技术指标（所有周期）──
        if snapshot.indicators:
            lines.extend(["## 技术指标"])
            for key, val in snapshot.indicators.items():
                if isinstance(val, (int, float)):
                    lines.append(f"- {key}: {val}")
            lines.append("")

        # ── 资金流向（short + medium）──
        if horizon in ("short", "medium"):
            if snapshot.fund_flow and isinstance(snapshot.fund_flow, dict):
                flow = {k: v for k, v in snapshot.fund_flow.items()
                        if not k.startswith("_") and not isinstance(v, list)}
                if flow:
                    lines.extend(["## 资金流向"])
                    for k, v in flow.items():
                        if isinstance(v, (int, float, str)):
                            lines.append(f"- {k}: {v}")
                    lines.append("")

        # ── 市场情绪（short + medium）──
        if horizon in ("short", "medium"):
            if snapshot.sentiment_data and isinstance(snapshot.sentiment_data, dict):
                sentiment = {k: v for k, v in snapshot.sentiment_data.items()
                             if not k.startswith("_")}
                if sentiment:
                    lines.extend(["## 市场情绪"])
                    news = sentiment.get("news", [])
                    if news:
                        lines.append("- 近期新闻:")
                        for item in news[:5]:
                            title = item.get("title", "")
                            if title:
                                lines.append(f"  - {title}")
                    for k, v in sentiment.items():
                        if k not in ("news", "news_analysis", "sector_heat") and isinstance(v, (int, float, str)):
                            lines.append(f"- {k}: {v}")
                    lines.append("")

        # ── 基本面（medium + long）──
        if horizon in ("medium", "long"):
            if snapshot.fundamentals and isinstance(snapshot.fundamentals, dict):
                fund = {k: v for k, v in snapshot.fundamentals.items()
                        if not k.startswith("_") and k not in ("bid_ask", "quarterly_data")}
                if fund:
                    lines.extend(["## 基本面数据"])
                    key_metrics = [
                        "company_name", "industry", "pe_ttm", "pb", "roe",
                        "market_cap", "gross_margin", "net_margin",
                        "revenue_yoy", "profit_yoy",
                    ]
                    for k in key_metrics:
                        if k in fund and fund[k] is not None:
                            lines.append(f"- {k}: {fund[k]}")
                    lines.append("")

        # ── 宏观环境（long）──
        if horizon == "long":
            if snapshot.macro_data and isinstance(snapshot.macro_data, dict):
                macro = {k: v for k, v in snapshot.macro_data.items()
                         if not k.startswith("_")}
                if macro:
                    lines.extend(["## 宏观环境"])
                    for k, v in macro.items():
                        if isinstance(v, (int, float, str)):
                            lines.append(f"- {k}: {v}")
                    lines.append("")

            # 市场指数
            if snapshot.market_context and isinstance(snapshot.market_context, dict):
                ctx = {k: v for k, v in snapshot.market_context.items()
                       if not k.startswith("_")}
                indices = ctx.get("indices", {})
                if indices:
                    lines.extend(["## 大盘指数"])
                    for name, data in indices.items():
                        if isinstance(data, dict):
                            close = data.get("close", "N/A")
                            pct = data.get("pct_change", "N/A")
                            lines.append(f"- {name}: {close} ({pct})")
                    lines.append("")

        # ── 风险指标（所有周期）──
        if snapshot.risk_metrics:
            lines.extend(["## 风险指标"])
            for key, val in snapshot.risk_metrics.items():
                if isinstance(val, (int, float)):
                    lines.append(f"- {key}: {val}")
            lines.append("")

        lines.extend([
            "## 要求",
            f"请基于以上真实数据，对 {stock_code} ({stock_name}) 进行 {horizon} 周期走势预测。",
            "严格按照系统提示词中的JSON格式输出，不要输出任何其他内容。",
        ])

        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(system: str, user: str) -> int:
        """粗略估算 token 数（中文 ~1.5 字符/token，英文 ~4 字符/token）"""
        total_chars = len(system) + len(user)
        return int(total_chars / 2.5)

    # ══════════════════════════════════════════════════════════════════════
    # Confidence Calibration
    # ══════════════════════════════════════════════════════════════════════

    def _compute_quality_factor(self, snapshot: StockSnapshot) -> float:
        """
        基于数据质量计算置信度校准因子。

        因子 = 1.0 表示数据完整，< 1.0 表示有数据缺失。
        缺失越多，因子越低，预测置信度自动打折。
        """
        dq = snapshot.data_quality or {}
        data_types = dq.get("data_types", {})
        if not data_types:
            return 1.0

        weights = {
            "kline": 0.30,
            "fundamentals": 0.25,
            "fund_flow": 0.15,
            "sentiment": 0.10,
            "market_context": 0.10,
            "macro": 0.10,
        }

        total = 0.0
        weight_sum = 0.0
        for dtype, w in weights.items():
            info = data_types.get(dtype, {})
            status = info.get("status", "unavailable")
            completeness = info.get("completeness", 0.0)

            if status == "ok":
                total += w
            elif status == "partial":
                total += w * completeness
            # unavailable → 0 contribution

            weight_sum += w

        if weight_sum == 0:
            return 1.0

        return round(max(0.3, total / weight_sum), 4)

    # ══════════════════════════════════════════════════════════════════════
    # Response Parsing
    # ══════════════════════════════════════════════════════════════════════

    def _parse_prediction(
        self,
        response: Dict[str, Any],
        stock_code: str,
        stock_name: str,
        horizon: str,
    ) -> PredictionResult:
        """解析并校验预测结果"""
        # 概率归一化
        up = float(response.get("probability_up", 0))
        down = float(response.get("probability_down", 0))
        sideways = float(response.get("probability_sideways", 0))
        total = up + down + sideways
        if total > 0:
            up, down, sideways = up / total, down / total, sideways / total

        direction = response.get("direction", "不确定")
        if direction not in ("上涨", "下跌", "震荡", "不确定"):
            direction = "不确定"

        confidence = max(0.0, min(1.0, float(response.get("confidence", 0))))

        return PredictionResult(
            stock_code=stock_code,
            stock_name=stock_name,
            prediction_horizon=horizon,
            direction=direction,
            confidence=confidence,
            confidence_calibrated=confidence,
            data_quality_factor=1.0,
            target_price_low=response.get("target_price_low"),
            target_price_high=response.get("target_price_high"),
            stop_loss=response.get("stop_loss"),
            risk_reward_ratio=response.get("risk_reward_ratio"),
            probability_up=round(up, 4),
            probability_down=round(down, 4),
            probability_sideways=round(sideways, 4),
            holding_period_days=response.get("holding_period_days"),
            key_drivers=response.get("key_drivers", []),
            risk_factors=response.get("risk_factors", []),
            catalyst_events=response.get("catalyst_events", []),
            reasoning=response.get("reasoning", ""),
        )
