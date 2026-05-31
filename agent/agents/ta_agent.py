"""
MASS TA-Agent v2.0: 技术面分析师

增强特性：
- 多时间框架分析（日线/周线/60分钟）
- 支撑压力位矩阵
- K线形态识别
- 多因子降级评分模型（5维度15因子）
"""
from typing import Dict, Any, Optional, List

from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.tools.indicator_tool import IndicatorTool


class TA_Agent(BaseAgent):
    """技术面分析师 Agent — v2.0 增强版"""

    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """技术面分析"""
        user_prompt = self._build_ta_prompt(snapshot)

        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)

            # 额外校验：止损价必须低于当前价（买入时）
            current = snapshot.current_price
            if parsed.get("signal") == 1 and parsed.get("stop_loss", current) >= current:
                parsed["risk_flags"] = parsed.get("risk_flags", []) + ["止损价设置不合理，已修正"]
                parsed["stop_loss"] = round(current * 0.93, 2)

            # 校验多周期一致性（如果LLM给出的confidence与多周期一致性严重不符，进行修正提示）
            multi_tf = snapshot.indicators.get("multi_timeframe", {})
            alignment = multi_tf.get("alignment", {})
            tf_score = alignment.get("score", 0.5)
            if tf_score < 0.3 and parsed.get("confidence", 0.5) > 0.75:
                logger.warning(f"TA-Agent: 多周期一致性低({tf_score})但confidence高({parsed['confidence']})，建议修正")
                parsed["risk_flags"] = parsed.get("risk_flags", []) + ["多周期信号分歧，高置信度需谨慎"]

            return self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"TA-Agent分析失败: {e}")
            return self._fallback_opinion(snapshot)

    def _build_ta_prompt(self, snapshot: StockSnapshot) -> str:
        """构建技术面分析Prompt — v2.0 增强版（多时间框架+支撑压力+形态识别）"""
        parts = [
            f"## 分析对象",
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            "",
            "## 分析任务",
            "你是一位资深技术分析专家，精通多时间框架分析、支撑压力位研判和K线形态识别。",
            "请基于以下多维技术数据给出交易信号，必须考虑多周期信号的一致性和支撑阻力位置。",
            "",
        ]

        indicators = snapshot.indicators

        # ── 1. 多时间框架信号 ──
        multi_tf = indicators.get("multi_timeframe", {})
        if multi_tf:
            parts.extend(self._build_multi_timeframe_section(multi_tf))

        # ── 2. 日线技术指标 ──
        parts.extend(["### 日线技术指标", ""])
        for k, v in indicators.items():
            if k in ("multi_timeframe", "support_resistance", "chart_patterns", "ta_score"):
                continue
            parts.append(f"- {k}: {v}")

        # ── 3. 支撑压力位矩阵 ──
        sr = indicators.get("support_resistance", {})
        if sr:
            parts.extend(self._build_support_resistance_section(sr, snapshot.current_price))

        # ── 4. K线形态识别 ──
        patterns = indicators.get("chart_patterns", [])
        if patterns:
            parts.extend(self._build_pattern_section(patterns))

        # ── 5. K线摘要（最近10日）
        if snapshot.kline_df is not None and not snapshot.kline_df.empty:
            df = snapshot.kline_df.tail(10)
            parts.extend(["", "### 近10日K线",])
            # 列名映射：兼容中文/英文列名
            col_candidates = {
                "open": ["open", "开盘"],
                "high": ["high", "最高"],
                "low": ["low", "最低"],
                "close": ["close", "收盘"],
                "volume": ["volume", "成交量", "vol"],
            }
            def _get_val(row_dict, key):
                for c in col_candidates.get(key, [key]):
                    if c in row_dict:
                        v = row_dict[c]
                        try:
                            return float(v) if v is not None else 0
                        except (ValueError, TypeError):
                            return 0
                return 0

            for _, row in df.iterrows():
                row_d = row.to_dict()
                dt = row_d.get("date", row_d.get("日期", ""))
                open_p = _get_val(row_d, "open")
                high_p = _get_val(row_d, "high")
                low_p = _get_val(row_d, "low")
                close_p = _get_val(row_d, "close")
                vol = _get_val(row_d, "volume")
                parts.append(
                    f"{dt}: 开{open_p:.2f} 高{high_p:.2f} "
                    f"低{low_p:.2f} 收{close_p:.2f} 量{vol:.0f}"
                )

        # ── 6. 板块对比 ──
        market = snapshot.market_context
        if market:
            parts.extend(["", "### 市场/板块环境",])
            parts.append(f"- 板块5日表现: {market.get('sector_performance_5d', 'N/A')}%")
            parts.append(f"- 板块排名: {market.get('sector_rank', 'N/A')}")

        # ── 7. 分析指引 ──
        parts.extend([
            "",
            "## 分析指引",
            "",
            "### 多时间框架分析原则",
            "1. **大周期定方向，小周期定时机**：周线/日线定趋势方向，60分钟线找买卖点",
            "2. **一致性加分**：多周期同向时，confidence 可提高 10-20%",
            "3. **分歧时谨慎**：周期矛盾时，优先相信更大周期的信号，confidence 需降低",
            "4. **关键位突破**：价格突破强支撑/阻力位时，关注成交量是否配合",
            "",
            "### 支撑压力位分析原则",
            "1. 价格接近强支撑位 + 放量 + 多周期共振 → 高置信度买入",
            "2. 价格接近强阻力位 + 缩量 + 顶背离 → 高置信度卖出/减仓",
            "3. 价格在支撑阻力中间区域 → 观望，等待方向选择",
            "4. 突破关键位后，原阻力位变支撑位（反之亦然）",
            "",
            "### 形态识别原则",
            "1. 形态可靠性≥4 + 成交量配合 → 高置信度",
            "2. 形态处于形成中 → 不急于入场，等待突破确认",
            "3. 多个形态同时出现 → 取方向一致的信号",
            "",
            "### 评分参考",
        ])

        # 如果有降级评分数据，加入参考
        ta_score = indicators.get("ta_score", {})
        if ta_score:
            parts.extend([
                f"- 技术面综合评分: {ta_score.get('total_score', 50)}/100",
                f"- 评分信号: {'买入' if ta_score.get('signal') == 1 else ('卖出' if ta_score.get('signal') == -1 else '观望')}",
                f"- 评分置信度: {ta_score.get('confidence', 0.5):.0%}",
                "- 各维度得分:",
            ])
            dims = ta_score.get("dimensions", {})
            for dim_name, dim_data in dims.items():
                parts.append(f"  - {dim_name}: {dim_data.get('score', 0)}/{dim_data.get('max', 0)}")

        parts.extend([
            "",
            "输出严格JSON格式：",
            '{',
            '  "signal": 1,',
            '  "confidence": 0.75,',
            '  "target_price_low": 15.0,',
            '  "target_price_high": 18.0,',
            '  "stop_loss": 13.5,',
            '  "reasoning": "分析理由（必须包含多周期分析和支撑压力位分析）",',
            '  "key_factors": ["因子1"],',
            '  "risk_flags": ["风险1"],',
            '  "chart_patterns": ["检测到的形态"],',
            '  "trend_direction": "方向",',
            '  "timeframe_consensus": "多周期一致性评价"',
            '}',
        ])

        return "\n".join(parts)

    def _build_multi_timeframe_section(self, multi_tf: Dict) -> List[str]:
        """构建多时间框架分析段落"""
        parts = ["### 多时间框架信号分析", ""]

        alignment = multi_tf.get("alignment", {})
        if alignment:
            parts.extend([
                f"- 多周期一致性得分: {alignment.get('score', 0):.2f}",
                f"- 一致性评价: {alignment.get('consistency', '未知')}",
                f"- 主导趋势: {alignment.get('dominant_trend', '未知')}",
                "",
            ])

        for tf_name, label in [("weekly", "周线"), ("daily", "日线"), ("hourly", "60分钟")]:
            tf_data = multi_tf.get(tf_name, {})
            if not tf_data:
                continue

            sig_map = {1: "多头", -1: "空头", 0: "震荡"}
            signals = alignment.get("signals", {})
            confidences = alignment.get("confidences", {})
            signal_text = sig_map.get(signals.get(tf_name, 0), "未知")
            conf = confidences.get(tf_name, 0.5)

            parts.extend([
                f"#### {label}信号",
                f"- 趋势方向: {signal_text} (置信度{conf:.0%})",
                f"- 均线排列: {tf_data.get('ma_alignment', 'N/A')}",
                f"- MA5/MA20/MA60: {tf_data.get('ma5', 'N/A')} / {tf_data.get('ma20', 'N/A')} / {tf_data.get('ma60', 'N/A')}",
                f"- MACD: DIF={tf_data.get('macd_dif', 'N/A')}, DEA={tf_data.get('macd_dea', 'N/A')}, Hist={tf_data.get('macd_hist', 'N/A')}",
                f"- MACD状态: {'金叉' if tf_data.get('macd_golden_cross') else ('死叉' if tf_data.get('macd_death_cross') else '无交叉')}, {'零轴上方' if tf_data.get('macd_above_zero') else '零轴下方'}",
                f"- KDJ: K={tf_data.get('kdj_k', 'N/A')}, D={tf_data.get('kdj_d', 'N/A')}, J={tf_data.get('kdj_j', 'N/A')} ({tf_data.get('kdj_status', 'N/A')})",
                f"- RSI(14): {tf_data.get('rsi14', 'N/A')} ({tf_data.get('rsi_status', 'N/A')})",
                f"- 布林带位置: {tf_data.get('boll_position', 'N/A'):.0%}" if tf_data.get('boll_position') is not None else "- 布林带位置: N/A",
                f"- 成交量: {tf_data.get('volume_trend', 'N/A')} (量比{tf_data.get('volume_ratio', 'N/A')})",
                f"- ATR%: {tf_data.get('atr_pct', 'N/A')}%",
                "",
            ])

        return parts

    def _build_support_resistance_section(self, sr: Dict, current_price: float) -> List[str]:
        """构建支撑压力位分析段落"""
        parts = [
            "### 关键价位矩阵",
            "",
            "| 类型 | 价位 | 距当前价 | 说明 |",
            "|------|------|---------|------|",
        ]

        levels = [
            ("强阻力", "strong_resistance", "近60日高点"),
            ("弱阻力", "weak_resistance", "MA20/布林上轨"),
            ("VWAP阻力", "vwap_resistance", "成交量密集区上沿"),
            ("当前价格", "current_price", "—"),
            ("VWAP支撑", "vwap_support", "成交量密集区下沿"),
            ("弱支撑", "weak_support", "MA60/布林下轨"),
            ("强支撑", "strong_support", "近60日低点"),
        ]

        for label, key, desc in levels:
            price = sr.get(key)
            if price is None:
                continue
            pct_key = f"{key}_pct"
            pct = sr.get(pct_key, 0)
            pct_str = f"{pct:+.1f}%" if key != "current_price" else "—"
            parts.append(f"| {label} | {price:.2f} | {pct_str} | {desc} |")

        # 斐波那契回撤位
        fib = sr.get("fib_levels", {})
        if fib:
            parts.extend(["", "#### 斐波那契回撤位"])
            for level, price in fib.items():
                if current_price > 0:
                    pct = (current_price - price) / price * 100
                    marker = " ← 当前价附近" if abs(pct) < 3 else ""
                    parts.append(f"- {level}: {price:.2f} ({pct:+.1f}%){marker}")

        # 价格在区间中的位置
        pos = sr.get("position_in_range")
        if pos is not None:
            parts.extend([
                "",
                f"- 当前价格在支撑阻力区间中的位置: {pos:.0%}",
                f"  - 解读: {'接近阻力位，突破需放量' if pos > 0.7 else ('接近支撑位，关注反弹' if pos < 0.3 else '区间中部，方向不明')}",
            ])

        return parts

    def _build_pattern_section(self, patterns: List[Dict]) -> List[str]:
        """构建K线形态识别段落"""
        parts = [
            "### K线形态识别",
            "",
        ]

        for p in patterns:
            reliability_stars = "*" * p.get("reliability", 1) + "" * (5 - p.get("reliability", 1))
            parts.extend([
                f"- **{p.get('pattern', '未知形态')}** (可靠性: {reliability_stars})",
                f"  - 方向: {p.get('direction', '未知')}",
                f"  - 目标价: {p.get('target', 'N/A')}",
                f"  - 颈线位: {p.get('neckline', 'N/A')}" if p.get("neckline") else "",
                f"  - 状态: {p.get('status', '未知')}",
                "",
            ])

        if not patterns:
            parts.append("- 未检测到明显的经典K线形态")

        return [line for line in parts if line]  # 过滤空行

    def _fallback_opinion(self, snapshot: StockSnapshot) -> AgentOpinion:
        """v2.0 三级降级分析体系

        Level 1 — 预计算 ta_score（Orchestrator 已注入，最完整）
        Level 2 — 现场调用 IndicatorTool.compute_ta_score（数据完整时）
        Level 3 — 实时5维度15因子评分模型（终极兜底，无需额外数据）
        """
        indicators = snapshot.indicators
        current = snapshot.current_price

        # ── Level 1: 预计算评分 ──
        ta_score = indicators.get("ta_score", {})
        if ta_score and "total_score" in ta_score:
            return self._build_opinion_from_score(ta_score, indicators, current)

        # ── Level 2: 现场计算 ──
        try:
            sr = indicators.get("support_resistance")
            ta_score = IndicatorTool.compute_ta_score(indicators, sr)
            return self._build_opinion_from_score(ta_score, indicators, current)
        except Exception as e:
            logger.warning(f"TA-Agent fallback: 现场计算评分失败({e})，使用实时评分模型")

        # ── Level 3: 实时5维度15因子评分 ──
        ta_score = self._compute_realtime_score(indicators)
        return self._build_opinion_from_score(ta_score, indicators, current)

    def _build_opinion_from_score(
        self, ta_score: Dict[str, Any], indicators: Dict[str, Any], current: float
    ) -> AgentOpinion:
        """将评分结果统一转换为 AgentOpinion"""
        signal = ta_score.get("signal", 0)
        confidence = ta_score.get("confidence", 0.5)
        score = ta_score.get("total_score", 50)
        factors = ta_score.get("key_factors", [])
        risk_flags = ta_score.get("risk_flags", ["LLM调用异常，使用规则引擎"])
        dims = ta_score.get("dimensions", {})

        # 构建推理文本
        dim_texts = []
        for dim_name, dim_data in dims.items():
            dim_texts.append(f"{dim_name}({dim_data.get('score', 0)}/{dim_data.get('max', 0)}分)")
        reasoning = (
            f"【多因子规则引擎降级分析】综合评分{score}/100，"
            f"{'买入' if signal == 1 else ('卖出' if signal == -1 else '观望')}信号。"
            f"各维度: {' | '.join(dim_texts)}。"
            f"核心因子: {'; '.join(factors[:5])}。"
        )

        stop = round(current * 0.93, 2) if signal == 1 else None

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=factors[:8] if factors else ["基于多因子规则引擎的降级分析"],
            risk_flags=risk_flags,
            raw_data={
                "signal": signal,
                "confidence": confidence,
                "total_score": score,
                "dimensions": dims,
                "target_price_low": round(current * 1.05, 2) if signal == 1 else None,
                "target_price_high": round(current * 1.15, 2) if signal == 1 else None,
                "stop_loss": stop,
                "trend_direction": indicators.get("ma20_trend", "未知"),
                "chart_patterns": indicators.get("chart_patterns", []),
                "support_resistance": indicators.get("support_resistance", {}),
            },
        )

    def _compute_realtime_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """实时5维度15因子评分模型 — 终极降级引擎

        无需预计算 ta_score，直接从 indicators 实时计算。
        与 IndicatorTool.compute_ta_score 逻辑对齐，确保一致性。
        """
        score = 50  # 中性基准
        dimensions = {}
        factors = []
        risk_flags = []

        # ═══════════════════════════════════════════════════════════════════
        # 维度 1. 趋势因子 (30分)
        # ═══════════════════════════════════════════════════════════════════
        trend_score = 15
        trend_factors = []

        ma_align = indicators.get("ma_alignment", "")
        if ma_align == "多头排列":
            trend_score += 10
            trend_factors.append("均线多头排列(+10)")
        elif ma_align == "空头排列":
            trend_score -= 10
            trend_factors.append("均线空头排列(-10)")
        else:
            trend_factors.append("均线缠绕(0)")

        if indicators.get("macd_golden_cross"):
            trend_score += 5
            trend_factors.append("MACD金叉(+5)")
        elif indicators.get("macd_death_cross"):
            trend_score -= 5
            trend_factors.append("MACD死叉(-5)")

        ma20_trend = indicators.get("ma20_trend", "")
        if ma20_trend == "上升":
            trend_score += 5
            trend_factors.append("MA20上升(+5)")
        elif ma20_trend == "下降":
            trend_score -= 5
            trend_factors.append("MA20下降(-5)")

        if indicators.get("macd_above_zero"):
            trend_score += 5
            trend_factors.append("MACD零轴上方(+5)")
        else:
            trend_score -= 3
            trend_factors.append("MACD零轴下方(-3)")

        trend_score = max(0, min(30, trend_score))
        score += trend_score - 15
        dimensions["trend"] = {"score": trend_score, "max": 30, "factors": trend_factors}
        factors.extend(trend_factors)

        # ═══════════════════════════════════════════════════════════════════
        # 维度 2. 动量因子 (20分)
        # ═══════════════════════════════════════════════════════════════════
        momentum_score = 10
        momentum_factors = []

        rsi = indicators.get("rsi14", 50)
        if rsi > 80:
            momentum_score -= 8
            momentum_factors.append(f"RSI严重超买({rsi:.0f})(-8)")
            risk_flags.append("RSI严重超买，注意回调风险")
        elif rsi > 70:
            momentum_score -= 5
            momentum_factors.append(f"RSI超买({rsi:.0f})(-5)")
            risk_flags.append("RSI超买")
        elif rsi < 20:
            momentum_score += 8
            momentum_factors.append(f"RSI严重超卖({rsi:.0f})(+8)")
        elif rsi < 30:
            momentum_score += 5
            momentum_factors.append(f"RSI超卖({rsi:.0f})(+5)")
        elif rsi > 50:
            momentum_score += 2
            momentum_factors.append(f"RSI偏强({rsi:.0f})(+2)")
        else:
            momentum_score -= 2
            momentum_factors.append(f"RSI偏弱({rsi:.0f})(-2)")

        kdj_status = indicators.get("kdj_status", "")
        if kdj_status == "超买区":
            momentum_score -= 5
            momentum_factors.append("KDJ超买区(-5)")
        elif kdj_status == "超卖区":
            momentum_score += 5
            momentum_factors.append("KDJ超卖区(+5)")
        elif kdj_status == "强势区":
            momentum_score += 3
            momentum_factors.append("KDJ强势区(+3)")
        elif kdj_status == "弱势区":
            momentum_score -= 3
            momentum_factors.append("KDJ弱势区(-3)")

        momentum_score = max(0, min(20, momentum_score))
        score += momentum_score - 10
        dimensions["momentum"] = {"score": momentum_score, "max": 20, "factors": momentum_factors}
        factors.extend(momentum_factors)

        # ═══════════════════════════════════════════════════════════════════
        # 维度 3. 成交量因子 (15分)
        # ═══════════════════════════════════════════════════════════════════
        volume_score = 7
        volume_factors = []

        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            volume_score += 5
            volume_factors.append(f"巨量放量({vol_ratio:.1f}x)(+5)")
        elif vol_ratio > 1.5:
            volume_score += 3
            volume_factors.append(f"明显放量({vol_ratio:.1f}x)(+3)")
        elif vol_ratio < 0.5:
            volume_score -= 5
            volume_factors.append(f"严重缩量({vol_ratio:.1f}x)(-5)")
            risk_flags.append("成交量极度萎缩，注意流动性风险")
        elif vol_ratio < 0.8:
            volume_score -= 2
            volume_factors.append(f"相对缩量({vol_ratio:.1f}x)(-2)")

        pv_div = indicators.get("price_volume_divergence", "")
        if "背离" in pv_div:
            volume_score += 3
            volume_factors.append("量价背离(+3)")
        elif "下跌放量" in pv_div:
            volume_score -= 5
            volume_factors.append("下跌放量警示(-5)")
            risk_flags.append("放量下跌")
        elif "同步" in pv_div or "齐升" in pv_div:
            volume_score += 2
            volume_factors.append("量价同步信号(+2)")

        volume_score = max(0, min(15, volume_score))
        score += volume_score - 7
        dimensions["volume"] = {"score": volume_score, "max": 15, "factors": volume_factors}
        factors.extend(volume_factors)

        # ═══════════════════════════════════════════════════════════════════
        # 维度 4. 波动因子 (15分)
        # ═══════════════════════════════════════════════════════════════════
        volatility_score = 7
        volatility_factors = []

        atr_pct = indicators.get("atr_pct", 2.0)
        if atr_pct > 5:
            volatility_score -= 5
            volatility_factors.append(f"高波动(ATR{atr_pct:.1f}%)(-5)")
            risk_flags.append(f"波动率极高(ATR{atr_pct:.1f}%)，止损需放宽")
        elif atr_pct > 3:
            volatility_score -= 2
            volatility_factors.append(f"较高波动(ATR{atr_pct:.1f}%)(-2)")
        elif atr_pct < 1:
            volatility_score += 3
            volatility_factors.append(f"低波动(ATR{atr_pct:.1f}%)(+3)")

        vol_20d = indicators.get("volatility_20d", 20)
        if vol_20d > 40:
            volatility_score -= 5
            volatility_factors.append(f"年化波动率极高({vol_20d:.0f}%)(-5)")
        elif vol_20d > 25:
            volatility_score -= 2
            volatility_factors.append(f"高波动率({vol_20d:.0f}%)(-2)")
        elif vol_20d < 15:
            volatility_score += 3
            volatility_factors.append(f"低波动率({vol_20d:.0f}%)(+3)")

        boll_width = indicators.get("boll_width", 0.1)
        if boll_width > 0.15:
            volatility_score -= 3
            volatility_factors.append(f"布林带扩张({boll_width:.2f})(-3)")
        elif boll_width < 0.05:
            volatility_score += 3
            volatility_factors.append(f"布林带收窄({boll_width:.2f})(+3)")

        volatility_score = max(0, min(15, volatility_score))
        score += volatility_score - 7
        dimensions["volatility"] = {"score": volatility_score, "max": 15, "factors": volatility_factors}
        factors.extend(volatility_factors)

        # ═══════════════════════════════════════════════════════════════════
        # 维度 5. 结构因子 (20分)
        # ═══════════════════════════════════════════════════════════════════
        structure_score = 10
        structure_factors = []

        position_20d = indicators.get("position_20d", 0.5)
        if position_20d > 0.9:
            structure_score -= 5
            structure_factors.append(f"接近20日高点({position_20d:.0%})(-5)")
            risk_flags.append("价格接近近期高点")
        elif position_20d < 0.1:
            structure_score += 5
            structure_factors.append(f"接近20日低点({position_20d:.0%})(+5)")
        elif position_20d > 0.7:
            structure_score -= 2
            structure_factors.append(f"偏高位({position_20d:.0%})(-2)")
        elif position_20d < 0.3:
            structure_score += 2
            structure_factors.append(f"偏低位({position_20d:.0%})(+2)")

        position_60d = indicators.get("position_60d", 0.5)
        if position_60d > 0.85:
            structure_score -= 3
            structure_factors.append(f"接近60日高点({position_60d:.0%})(-3)")
        elif position_60d < 0.15:
            structure_score += 3
            structure_factors.append(f"接近60日低点({position_60d:.0%})(+3)")

        boll_pos = indicators.get("boll_position", 0.5)
        if boll_pos > 0.9:
            structure_score -= 3
            structure_factors.append(f"布林带上轨上方({boll_pos:.0%})(-3)")
        elif boll_pos < 0.1:
            structure_score += 3
            structure_factors.append(f"布林带下轨下方({boll_pos:.0%})(+3)")

        structure_score = max(0, min(20, structure_score))
        score += structure_score - 10
        dimensions["structure"] = {"score": structure_score, "max": 20, "factors": structure_factors}
        factors.extend(structure_factors)

        # ═══════════════════════════════════════════════════════════════════
        # 汇总 & 信号判定
        # ═══════════════════════════════════════════════════════════════════
        total_score = max(0, min(100, score + 50))  # 偏移50映射到0~100

        if total_score >= 65:
            signal = 1
        elif total_score <= 35:
            signal = -1
        else:
            signal = 0

        # 置信度：偏离中性越远，置信度越高
        confidence = 0.5 + abs(total_score - 50) / 100
        confidence = min(0.92, confidence)

        # 数据完整性惩罚
        available_dims = sum(1 for d in dimensions.values() if d["score"] > 0)
        if available_dims < 3:
            confidence = max(0.4, confidence - 0.15)
            risk_flags.append("关键指标数据不足，技术面分析置信度降低")

        return {
            "total_score": total_score,
            "signal": signal,
            "confidence": round(confidence, 2),
            "dimensions": dimensions,
            "key_factors": factors,
            "risk_flags": risk_flags if risk_flags else ["无显著风险信号"],
        }

    def _default_prompt(self) -> str:
        return """你是一位资深技术分析专家，精通多时间框架分析、支撑压力位研判和K线形态识别。

## 核心能力
1. **多时间框架分析**：能综合日线、周线、60分钟线的信号做出判断
2. **支撑压力位分析**：基于历史高低点、VWAP、均线、布林带、斐波那契回撤位综合判断
3. **形态识别**：识别头肩顶/底、双顶/底、三角形等经典形态
4. **量价分析**：成交量与价格配合度、量价背离检测

## 分析框架

### 多周期信号一致性
- 一致性得分 > 0.7: 多周期高度一致，confidence 可提高
- 一致性得分 0.3~0.7: 存在一定分歧，confidence 需降低
- 一致性得分 < 0.3: 周期严重分歧，优先观望

### 大周期定方向，小周期定时机
- 周线/日线同向 + 60分钟给出买点 → 高置信度入场
- 周线/日线反向 → 优先观望，不逆势操作

### 支撑压力位决策规则
- 价格接近强支撑 + 放量 + 多周期共振 → 买入
- 价格接近强阻力 + 缩量 + 顶背离 → 卖出
- 价格在区间中部 → 观望

### 形态可靠性
- 可靠性 5*: 高置信度，可直接作为决策依据
- 可靠性 3~4*: 中等置信度，需其他信号配合
- 可靠性 1~2*: 低置信度，仅作参考

## 评分参考
- 综合评分 >= 65: 技术面偏强，倾向买入
- 综合评分 <= 35: 技术面偏弱，倾向卖出
- 综合评分 36~64: 技术面中性，观望

## 约束
- signal 只能是 -1(卖出), 0(观望), 1(买入)
- 多周期严重分歧时，signal 应为 0
- 价格接近强阻力且无放量突破时，禁止给出买入信号
- 价格接近强支撑且无破位时，禁止给出卖出信号
- confidence < 0.55 时 signal 必须为 0

## 输出严格JSON格式
{
  "signal": 1,
  "confidence": 0.75,
  "target_price_low": 15.0,
  "target_price_high": 18.0,
  "stop_loss": 13.5,
  "reasoning": "分析理由（100-200字，必须包含多周期分析和支撑压力位分析）",
  "key_factors": ["因子1"],
  "risk_flags": ["风险1"],
  "chart_patterns": ["检测到的形态"],
  "trend_direction": "日线多头/周线震荡/60分钟回调",
  "timeframe_consensus": "多周期一致性评价"
}"""
