"""
MASS FA-Agent: 基本面分析师
优化重点：
1. 数据不完整时的推断能力（隐含ROE、估值联动）
2. PB异常检测与自动分析强化
3. 多层降级分析，绝不返回"数据不足"
4. 财务健康度红绿灯评估
"""
from typing import Dict, Any, Optional, List

from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion


class FA_Agent(BaseAgent):
    """基本面分析师 Agent"""

    # 行业平均估值参考表（作为fallback，优先使用snapshot中的真实行业数据）
    INDUSTRY_BENCHMARKS = {
        "银行": {"pe": 6.0, "pb": 0.7, "roe": 10.0},
        "保险": {"pe": 10.0, "pb": 1.2, "roe": 12.0},
        "证券": {"pe": 18.0, "pb": 1.5, "roe": 8.0},
        "房地产": {"pe": 12.0, "pb": 1.0, "roe": 8.0},
        "钢铁": {"pe": 10.0, "pb": 0.9, "roe": 9.0},
        "煤炭": {"pe": 8.0, "pb": 1.2, "roe": 15.0},
        "电力": {"pe": 15.0, "pb": 1.5, "roe": 10.0},
        "白酒": {"pe": 28.0, "pb": 6.0, "roe": 22.0},
        "食品饮料": {"pe": 25.0, "pb": 4.0, "roe": 15.0},
        "医药": {"pe": 30.0, "pb": 3.5, "roe": 12.0},
        "医疗器械": {"pe": 35.0, "pb": 4.0, "roe": 14.0},
        "半导体": {"pe": 50.0, "pb": 4.0, "roe": 10.0},
        "电子": {"pe": 35.0, "pb": 3.0, "roe": 10.0},
        "计算机": {"pe": 45.0, "pb": 3.5, "roe": 8.0},
        "新能源": {"pe": 25.0, "pb": 3.0, "roe": 12.0},
        "汽车": {"pe": 18.0, "pb": 1.8, "roe": 10.0},
        "化工": {"pe": 15.0, "pb": 1.5, "roe": 10.0},
        "机械": {"pe": 18.0, "pb": 2.0, "roe": 10.0},
        "建材": {"pe": 12.0, "pb": 1.2, "roe": 10.0},
        "纺织": {"pe": 15.0, "pb": 1.5, "roe": 9.0},
        "传媒": {"pe": 25.0, "pb": 2.0, "roe": 8.0},
        "通信": {"pe": 25.0, "pb": 2.0, "roe": 8.0},
        "军工": {"pe": 40.0, "pb": 3.0, "roe": 6.0},
        "有色": {"pe": 15.0, "pb": 1.8, "roe": 12.0},
        "石油": {"pe": 10.0, "pb": 1.2, "roe": 10.0},
        "航运": {"pe": 8.0, "pb": 1.2, "roe": 15.0},
        "港口": {"pe": 12.0, "pb": 1.0, "roe": 8.0},
        "零售": {"pe": 20.0, "pb": 1.5, "roe": 8.0},
        "旅游": {"pe": 25.0, "pb": 2.0, "roe": 8.0},
    }

    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """基本面分析"""
        user_prompt = self._build_fa_prompt(snapshot)

        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)

            # 确保关键字段存在
            score = parsed.get("fundamental_score", 50)
            parsed["fundamental_score"] = max(0, min(100, int(score)))

            # signal 与评分联动校验
            expected_signal = 1 if parsed["fundamental_score"] >= 75 else (-1 if parsed["fundamental_score"] <= 40 else 0)
            if parsed["signal"] != expected_signal:
                logger.warning(f"FA-Agent signal({parsed['signal']})与score({parsed['fundamental_score']})不匹配，已修正")
                parsed["signal"] = expected_signal

            # PE为负（亏损）时，禁止买入信号
            pe = snapshot.fundamentals.get("pe_ttm")
            if pe is not None and pe < 0 and parsed["signal"] == 1:
                logger.warning(f"FA-Agent: PE为负({pe})但信号为买入，修正为观望")
                parsed["signal"] = 0
                parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

            # PB异常高 + PE正常 → 强烈负面倾向修正
            pb = snapshot.fundamentals.get("pb")
            if pb is not None and pb > 10 and pe is not None and 0 < pe < 50:
                implied_roe = pb / pe if pe > 0 else 0
                # 隐含ROE<8%：盈利能力极差；隐含ROE>50%：不可持续或会计异常
                if (implied_roe < 0.08 or implied_roe > 0.50) and parsed["signal"] == 1:
                    logger.warning(f"FA-Agent: PB({pb})极高且隐含ROE({implied_roe:.1%})异常，修正信号")
                    parsed["signal"] = 0
                    parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

            # 确保 financial_health 存在
            if not parsed.get("financial_health"):
                parsed["financial_health"] = self._infer_financial_health(snapshot)

            # 确保 valuation_gap 存在
            if not parsed.get("valuation_gap"):
                parsed["valuation_gap"] = self._infer_valuation_gap(snapshot)

            return self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"FA-Agent分析失败: {e}")
            return self._fallback_opinion(snapshot)

    def _build_fa_prompt(self, snapshot: StockSnapshot) -> str:
        """构建基本面分析Prompt — 自动计算隐含指标、展示数据完整度"""
        fundamentals = snapshot.fundamentals
        parts = [
            f"## 分析对象",
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            f"所属行业: {fundamentals.get('industry', '未知')}",
            f"上市日期: {fundamentals.get('list_date', '未知')}",
            "",
            "## 分析任务",
            "你是一位深度价值分析师。请基于以下财务数据（部分可能缺失），",
            "通过估值联动关系和财务恒等式做出深度推断，评估企业内在价值与当前股价的匹配度。",
            "",
            "**重要：即使部分数据缺失，也必须基于已有数据做合理推断，禁止返回'数据缺失'、'无法判断'等消极表述。**",
            "",
        ]

        # ── 数据完整度统计 ──
        key_metrics = [
            "pe_ttm", "pb", "ps", "roe", "roic",
            "gross_margin", "net_margin", "debt_ratio",
            "current_ratio", "revenue_yoy", "profit_yoy",
            "operating_cash_flow", "free_cash_flow",
            "dividend_yield", "market_cap", "total_revenue",
        ]
        available = [k for k in key_metrics if k in fundamentals and fundamentals[k] is not None]
        completeness = len(available) / len(key_metrics)

        parts.extend([
            f"### 数据覆盖情况",
            f"- 可用指标数: {len(available)}/{len(key_metrics)} ({completeness:.0%})",
            f"- 可用指标: {', '.join(available) if available else '仅基础估值数据'}",
            "",
        ])

        # ── 1. 估值指标与隐含推导 ──
        parts.append("### 一、估值指标与隐含推导")
        pe = fundamentals.get("pe_ttm")
        pb = fundamentals.get("pb")
        ps = fundamentals.get("ps")

        if pe is not None:
            parts.append(f"- PE(TTM): {pe:.2f} {'(亏损)' if pe < 0 else ''}")
        if pb is not None:
            parts.append(f"- PB: {pb:.2f}")
        if ps is not None:
            parts.append(f"- PS: {ps:.2f}")

        # 隐含ROE推导
        if pb is not None and pe is not None and pe > 0:
            implied_roe = pb / pe
            parts.append(f"- **隐含ROE = PB/PE: {implied_roe:.2%}** (推导出的盈利能力指标)")
            if implied_roe < 0.08:
                parts.append(f"   隐含ROE极低，说明净资产回报率严重不足")
            elif implied_roe > 0.20:
                parts.append(f"   隐含ROE优秀")

        # 市值规模
        market_cap = fundamentals.get("market_cap")
        if market_cap:
            parts.append(f"- 总市值: {market_cap/1e8:.2f}亿")

        # 估值异常检测
        if pb is not None and pe is not None and pe > 0:
            if pb > 10 and pe < 50:
                parts.append(f"- ** 估值异常检测**: PB({pb:.2f})极高但PE({pe:.2f})正常，隐含ROE仅{implied_roe:.2%}，")
                parts.append(f"  可能原因：次新股净资产极小、壳资源炒作、或资产质量极差。需深入分析。")

        # ── 1.5 Forward 估值分析（新增） ──
        forecast = self._fetch_forecast_data(snapshot)
        if forecast:
            fm = self._compute_forward_metrics(snapshot, forecast)
            parts.extend([
                "",
                "### 一、Forward 估值与一致预期",
                f"- 研报覆盖数: {fm['research_report_count']} 家",
                f"- 机构评级分布: 买入/增持占比 {fm['rating_buy_pct']:.0%} ({forecast.get('rating_buy', 0)}买入 + {forecast.get('rating_add', 0)}增持)",
                f"- **一致预期强度**: {fm['consensus_strength']}",
            ])

            if fm["forward_pe_y1"]:
                parts.append(f"- Forward PE(当年): {fm['forward_pe_y1']:.1f}x")
            if fm["forward_pe_y2"]:
                parts.append(f"- Forward PE(次年): {fm['forward_pe_y2']:.1f}x")
            if fm["forward_pe_y3"]:
                parts.append(f"- Forward PE(第三年): {fm['forward_pe_y3']:.1f}x")

            if fm["eps_growth_y1y2"] is not None:
                parts.append(f"- 预测 EPS 增速(当年→次年): {fm['eps_growth_y1y2']:+.1f}%")
            if fm["eps_growth_y2y3"] is not None:
                parts.append(f"- 预测 EPS 增速(次年→第三年): {fm['eps_growth_y2y3']:+.1f}%")

            if fm["peg_y1"] is not None:
                parts.append(f"- PEG(当年): {fm['peg_y1']:.2f} {'' if fm['peg_y1'] < 1 else ('' if fm['peg_y1'] > 2 else '')}")
            if fm["peg_y2"] is not None:
                parts.append(f"- PEG(次年): {fm['peg_y2']:.2f} {'' if fm['peg_y2'] < 1 else ('' if fm['peg_y2'] > 2 else '')}")

            # Forward vs TTM 对比
            if fm["forward_pe_y1"] and pe and pe > 0:
                pe_discount = (1 - fm["forward_pe_y1"] / pe) * 100
                if pe_discount > 10:
                    parts.append(f"- Forward PE 较 TTM 折价 {pe_discount:.1f}% → 预期盈利改善")
                elif pe_discount < -10:
                    parts.append(f"- Forward PE 较 TTM 溢价 {abs(pe_discount):.1f}% → 预期盈利下滑")

            parts.append("")

        parts.append("")

        # ── 2. 盈利能力指标 ──
        has_profitability = any(
            fundamentals.get(k) is not None
            for k in ["roe", "roic", "gross_margin", "net_margin"]
        )
        if has_profitability:
            parts.append("### 二、盈利能力指标")
            roe = fundamentals.get("roe")
            if roe is not None:
                parts.append(f"- ROE: {roe:.2f}% {'' if roe > 15 else ('' if roe < 8 else '')}")
            roic = fundamentals.get("roic")
            if roic is not None:
                parts.append(f"- ROIC: {roic:.2f}%")
            gm = fundamentals.get("gross_margin")
            if gm is not None:
                parts.append(f"- 毛利率: {gm:.2f}%")
            nm = fundamentals.get("net_margin")
            if nm is not None:
                parts.append(f"- 净利率: {nm:.2f}%")
            parts.append("")

        # ── 3. 成长性指标 ──
        has_growth = any(
            fundamentals.get(k) is not None
            for k in ["revenue_yoy", "profit_yoy", "total_revenue"]
        )
        if has_growth:
            parts.append("### 三、成长性指标")
            rev_yoy = fundamentals.get("revenue_yoy")
            if rev_yoy is not None:
                parts.append(f"- 营收同比增长: {rev_yoy:.2f}%")
            prof_yoy = fundamentals.get("profit_yoy")
            if prof_yoy is not None:
                parts.append(f"- 净利润同比增长: {prof_yoy:.2f}%")
            total_rev = fundamentals.get("total_revenue")
            if total_rev is not None:
                parts.append(f"- 总营收: {total_rev/1e8:.2f}亿")
                if market_cap and total_rev > 0:
                    implied_ps = market_cap / total_rev
                    parts.append(f"- 隐含PS = 市值/营收: {implied_ps:.2f}")
            parts.append("")

        # ── 4. 财务安全指标 ──
        has_safety = any(
            fundamentals.get(k) is not None
            for k in ["debt_ratio", "current_ratio", "operating_cash_flow"]
        )
        if has_safety:
            parts.append("### 四、财务安全指标")
            debt = fundamentals.get("debt_ratio")
            if debt is not None:
                parts.append(f"- 资产负债率: {debt:.2f}% {'[危险]' if debt > 70 else ('[注意]' if debt > 50 else '[安全]')}")
            cr = fundamentals.get("current_ratio")
            if cr is not None:
                parts.append(f"- 流动比率: {cr:.2f} {'[安全]' if cr > 1.5 else ('[注意]' if cr > 1.0 else '[危险]')}")
            ocf = fundamentals.get("operating_cash_flow")
            if ocf is not None:
                parts.append(f"- 经营现金流: {ocf/1e8:.2f}亿 {'[安全]' if ocf > 0 else '[危险]'}")
            fcf = fundamentals.get("free_cash_flow")
            if fcf is not None:
                parts.append(f"- 自由现金流: {fcf/1e8:.2f}亿 {'[安全]' if fcf > 0 else '[危险]'}")
            parts.append("")

        # ── 5. 季度数据 ──
        quarters = fundamentals.get("quarterly_data", [])
        if quarters:
            parts.extend(["### 五、近8季度财务数据",])
            for q in quarters[:8]:
                parts.append(
                    f"- {q.get('quarter', '')}: 营收{q.get('revenue', 0):.2f} "
                    f"净利{q.get('net_profit', 0):.2f} "
                    f"毛利率{q.get('gross_margin', 0):.2f}% "
                    f"ROE{q.get('roe', 0):.2f}%"
                )
            parts.append("")

        # ── 5.5 财务趋势分析（新增） ──
        if quarters:
            trends = self._compute_financial_trends(quarters)
            parts.extend([
                "### 五、财务趋势分析（纵向对比）",
                "",
                f"- ROE趋势: **{trends['roe_trend']}** (最新{trends['latest_roe']:.1f}%)",
                f"- 毛利率趋势: **{trends['gross_margin_trend']}** (最新{trends['latest_gross_margin']:.1f}%)",
                f"- 营收增速: {trends['latest_revenue_growth']:+.1f}% (趋势{trends['revenue_growth_trend']})",
                f"- 利润增速: {trends['latest_profit_growth']:+.1f}% (趋势{trends['profit_growth_trend']})",
                f"- **盈利质量评分**: {trends['earnings_quality_score']}/100",
                "",
                "**趋势判断规则（供参考，分析时请结合以下信号）**：",
            ])

            if trends["continuous_roe_up"]:
                parts.append("-  ROE连续3季上升 → 盈利能力持续改善")
            if trends["continuous_gm_down"]:
                parts.append("-  毛利率连续下降 → 竞争恶化或成本上升")
            if trends["latest_revenue_growth"] > 0 and trends["latest_profit_growth"] < trends["latest_revenue_growth"] * 0.3:
                parts.append("-  营收增速显著高于利润增速 → 增收不增利，警惕")
            if trends["latest_profit_growth"] > trends["latest_revenue_growth"] and trends["latest_profit_growth"] > 0:
                parts.append("-  利润增速超营收增速 → 高质量增长")

            if trends["trend_flags"]:
                parts.extend(["", "**趋势信号**:",])
                for flag in trends["trend_flags"][:5]:
                    parts.append(f"- {flag}")

            parts.append("")

        # ── 6. 同业对比矩阵（新增） ──
        peer_parts = self._build_peer_comparison(snapshot)
        if peer_parts:
            parts.extend(peer_parts)

        # ── 7. 行业对比（使用真实数据或参考基准） ──
        industry = fundamentals.get("industry", "")
        parts.extend(["### 七、行业对比",])

        # 尝试从fundamentals中获取真实行业数据
        industry_pe = fundamentals.get("industry_pe")
        industry_pb = fundamentals.get("industry_pb")
        industry_roe = fundamentals.get("industry_roe")

        if industry_pe is not None or industry_pb is not None:
            if industry_pe is not None:
                parts.append(f"- 行业平均PE: {industry_pe:.2f}")
            if industry_pb is not None:
                parts.append(f"- 行业平均PB: {industry_pb:.2f}")
            if industry_roe is not None:
                parts.append(f"- 行业平均ROE: {industry_roe:.2f}%")
        else:
            # 使用参考基准
            benchmark = self._get_industry_benchmark(industry)
            parts.append(f"- 行业: {industry if industry else '未知'}")
            parts.append(f"- 参考行业平均PE: {benchmark['pe']:.1f} | PB: {benchmark['pb']:.1f} | ROE: {benchmark['roe']:.1f}%")
            parts.append(f"- *(注：行业数据为参考基准，实际可能有所差异)*")

        # 相对位置分析
        if pe is not None and pe > 0:
            ref_pe = industry_pe or self._get_industry_benchmark(industry)["pe"]
            pe_ratio = pe / ref_pe
            parts.append(f"- 当前PE相对行业: {pe_ratio:.2f}x {'(折价)' if pe_ratio < 0.8 else ('(溢价)' if pe_ratio > 1.2 else '(持平)')}")
        if pb is not None:
            ref_pb = industry_pb or self._get_industry_benchmark(industry)["pb"]
            pb_ratio = pb / ref_pb
            parts.append(f"- 当前PB相对行业: {pb_ratio:.2f}x {'(折价)' if pb_ratio < 0.8 else ('(溢价)' if pb_ratio > 1.2 else '(持平)')}")

        parts.append("")

        # ── 8. 分析指引 ──
        parts.extend([
            "## 分析指引",
            "",
            "1. **估值联动分析**：检查PE-PB-ROE三角关系。",
            "   - PB极高+PE正常 → 隐含ROE极低 → 基本面重大疑问",
            "   - PE极高+PB正常 → 盈利能力极弱但资产尚可",
            "   - 必须对异常估值组合给出明确解释",
            "",
            "2. **盈利能力评估**：ROE/隐含ROE是否>15%？毛利率趋势？",
            "",
            "3. **成长质量**：营收/利润增速匹配度？利润是否有现金流支撑？",
            "",
            "4. **财务安全**：负债率、流动性、现金流健康度（用红绿灯标注）。",
            "",
            "5. **估值水平**：相对行业和历史的折价/溢价。",
            "",
            "6. **数据不完整时的推断**：",
            "   - 当ROE缺失时，使用隐含ROE = PB/PE",
            "   - 当毛利率缺失时，通过PE和行业平均推断",
            '   - **禁止"数据缺失"、"无法判断"等消极表述**',
            "",
            "输出严格JSON格式：",
        ])

        return "\n".join(parts)

    def _fallback_opinion(self, snapshot: StockSnapshot) -> AgentOpinion:
        """多层降级分析 — 基于任何可用数据做有意义推断"""
        fundamentals = snapshot.fundamentals
        pe = fundamentals.get("pe_ttm")
        pb = fundamentals.get("pb")
        roe = fundamentals.get("roe")
        debt = fundamentals.get("debt_ratio")
        market_cap = fundamentals.get("market_cap")
        industry = fundamentals.get("industry", "")
        list_date = fundamentals.get("list_date", "")

        score = 50
        factors = []
        risk_flags = []
        signal = 0
        confidence = 0.55
        reasoning_parts = ["【规则引擎降级分析】"]
        financial_health = "[注意] 数据有限，基于可用指标推断"

        # ── 1. 估值分析（最可靠，优先使用）──
        if pe is not None and pb is not None:
            # PE-PB-ROE联动
            if pe > 0:
                implied_roe = pb / pe
                factors.append(f"隐含ROE={implied_roe:.2%}(PB{pb:.2f}/PE{pe:.2f})")

                # 估值评分
                valuation_score = 10
                ref = self._get_industry_benchmark(industry)

                # PE评估
                if pe < 0:
                    valuation_score -= 10
                    risk_flags.append("PE为负，公司处于亏损状态")
                    reasoning_parts.append(f"PE为负({pe:.2f})，公司亏损。")
                elif pe < ref["pe"] * 0.7:
                    valuation_score += 5
                    factors.append(f"PE({pe:.1f})低于行业平均({ref['pe']:.1f})")
                elif pe > ref["pe"] * 1.5:
                    valuation_score -= 5
                    risk_flags.append(f"PE({pe:.1f})高于行业平均({ref['pe']:.1f})")

                # PB评估
                if pb > 20:
                    valuation_score -= 15
                    risk_flags.append(f"PB({pb:.2f})极度异常，净资产溢价过高")
                    financial_health = "[危险] PB极度异常"
                elif pb > 10:
                    valuation_score -= 10
                    risk_flags.append(f"PB({pb:.2f})极高，净资产溢价异常")
                    financial_health = "[危险] PB极高，净资产溢价异常"
                elif pb > ref["pb"] * 2:
                    valuation_score -= 5
                    risk_flags.append(f"PB({pb:.2f})高于行业平均({ref['pb']:.1f})")

                # 隐含ROE异常时追加分析
                if implied_roe < 0.08:
                    risk_flags.append(f"隐含ROE仅{implied_roe:.2%}，盈利能力严重不足")
                    reasoning_parts.append(
                        f"PB高达{pb:.2f}，隐含ROE仅{implied_roe:.2%}，"
                        f"说明市场给予净资产极高溢价，基本面支撑不足。"
                    )
                elif implied_roe > 0.50:
                    reasoning_parts.append(
                        f"PB高达{pb:.2f}，隐含ROE高达{implied_roe:.2%}，"
                        f"远超正常水平(5%-30%)，极可能因净资产被压低或一次性收益导致，不可持续。"
                    )

                # 隐含ROE评估
                # 正常ROE区间约5%-30%，超出此范围均为异常
                if implied_roe < 0.05:
                    valuation_score -= 8
                    factors.append(f"隐含ROE极低({implied_roe:.2%})")
                elif implied_roe > 0.50:
                    valuation_score -= 10
                    risk_flags.append(f"隐含ROE高达{implied_roe:.2%}，不可持续或会计异常")
                    financial_health = "[危险] PB极高，隐含ROE异常"
                elif implied_roe > 0.30:
                    valuation_score -= 3
                    factors.append(f"隐含ROE偏高({implied_roe:.2%})，需警惕可持续性")
                elif implied_roe > 0.15:
                    valuation_score += 3
                    factors.append(f"隐含ROE良好({implied_roe:.2%})")

                score += valuation_score  # 允许负分，极端估值应拉低总分
            else:
                # PE为负（亏损）
                score -= 15
                risk_flags.append("PE为负，公司亏损")
                reasoning_parts.append(f"PE为负({pe:.2f})，公司处于亏损状态。")
                signal = -1
                financial_health = "[危险] 公司亏损"

        elif pe is not None:
            # 只有PE
            if pe < 0:
                score -= 15
                risk_flags.append("PE为负，公司亏损")
                reasoning_parts.append(f"PE为负({pe:.2f})，公司亏损。")
                signal = -1
            elif pe < 15:
                factors.append(f"PE较低({pe:.1f})")
                score += 10
            elif pe > 50:
                risk_flags.append(f"PE极高({pe:.1f})")
                score -= 10

        elif pb is not None:
            # 只有PB
            if pb > 5:
                risk_flags.append(f"PB较高({pb:.2f})")
                score -= 10
                reasoning_parts.append(f"PB({pb:.2f})较高，估值偏贵。")
            elif pb < 1:
                factors.append(f"PB较低({pb:.2f})")
                score += 5

        # ── 2. 盈利能力修正 ──
        if roe is not None:
            if roe > 15:
                score += 15
                factors.append(f"ROE优秀({roe:.1f}%)")
            elif roe > 10:
                score += 8
                factors.append(f"ROE良好({roe:.1f}%)")
            elif roe < 5:
                score -= 10
                risk_flags.append(f"ROE极低({roe:.1f}%)")
                reasoning_parts.append(f"ROE仅{roe:.1f}%，盈利能力弱。")

        # ── 3. 财务安全修正 ──
        if debt is not None:
            if debt > 70:
                score -= 10
                risk_flags.append(f"负债率高({debt:.1f}%)")
                financial_health = "[危险] 负债率过高"
            elif debt < 40:
                score += 5
                factors.append(f"负债率低({debt:.1f}%)")
                if "[危险]" not in financial_health:
                    financial_health = "[安全] 负债率安全"
            else:
                if "[危险]" not in financial_health:
                    financial_health = "[注意] 负债率中等"

        # ── 4. 规模与行业推断 ──
        if market_cap:
            if market_cap > 1e11:  # 1000亿
                factors.append("大盘蓝筹")
                score += 3
            elif market_cap < 5e9:  # 50亿
                risk_flags.append("小市值股票，流动性风险")

        # ── 4.5 一致预期修正（新增） ──
        try:
            forecast = self._fetch_forecast_data(snapshot)
            if forecast:
                fm = self._compute_forward_metrics(snapshot, forecast)
                report_count = fm.get("research_report_count", 0)
                rating_buy_pct = fm.get("rating_buy_pct", 0)
                forward_pe = fm.get("forward_pe_y1")
                eps_growth = fm.get("eps_growth_y1y2")
                peg = fm.get("peg_y1")

                # 研报覆盖度修正
                if report_count >= 20:
                    score += 3
                    factors.append(f"研报覆盖充分({report_count}家)")
                elif report_count < 5:
                    score -= 3
                    risk_flags.append("研报覆盖不足")

                # 机构评级修正
                if rating_buy_pct >= 0.8:
                    score += 5
                    factors.append("机构一致看多")
                elif rating_buy_pct <= 0.3 and report_count >= 5:
                    score -= 5
                    risk_flags.append("机构评级偏空")

                # Forward PE 修正
                if forward_pe and forward_pe > 0:
                    if forward_pe < 15:
                        score += 5
                        factors.append(f"Forward PE低({forward_pe:.1f}x)")
                    elif forward_pe > 40:
                        score -= 5
                        risk_flags.append(f"Forward PE高({forward_pe:.1f}x)")

                # PEG 修正
                if peg is not None:
                    if peg < 0.8:
                        score += 5
                        factors.append(f"PEG极低({peg:.2f})")
                    elif peg > 2.5:
                        score -= 5
                        risk_flags.append(f"PEG偏高({peg:.2f})")

                # 预期增速修正
                if eps_growth is not None:
                    if eps_growth > 30:
                        score += 5
                        factors.append(f"预期高增长({eps_growth:+.1f}%)")
                    elif eps_growth < -10:
                        score -= 5
                        risk_flags.append(f"预期盈利下滑({eps_growth:+.1f}%)")

                reasoning_parts.append(
                    f"一致预期: 覆盖{report_count}家, 看多占比{rating_buy_pct:.0%}, "
                    f"Forward PE{forward_pe if forward_pe else 'N/A'}x。"
                )
        except Exception:
            pass  # 一致预期失败不影响主流程

        # ── 5. 趋势分析修正（新增） ──
        quarters = fundamentals.get("quarterly_data", [])
        if quarters:
            try:
                trends = self._compute_financial_trends(quarters)
                eq_score = trends.get("earnings_quality_score", 50)

                # 盈利质量评分映射到基本面评分
                score += (eq_score - 50) * 0.3  # ±15分的影响

                # 趋势因子记录
                if trends.get("continuous_roe_up"):
                    factors.append("ROE连续改善")
                if trends.get("continuous_gm_down"):
                    risk_flags.append("毛利率连续下降")
                    score -= 5
                if trends.get("latest_profit_growth", 0) > trends.get("latest_revenue_growth", 0) > 0:
                    factors.append("高质量增长(利润增速>营收)")
                    score += 3
                if trends.get("latest_revenue_growth", 0) > 0 and trends.get("latest_profit_growth", 0) < trends.get("latest_revenue_growth", 0) * 0.3:
                    risk_flags.append("增收不增利")
                    score -= 5

                reasoning_parts.append(
                    f"财务趋势: ROE{trends['roe_trend']}, 毛利率{trends['gross_margin_trend']}, "
                    f"盈利质量评分{eq_score}。"
                )
            except Exception:
                pass  # 趋势分析失败不影响主流程

        # ── 6. 同业对比修正（新增） ──
        try:
            peer_parts = self._build_peer_comparison(snapshot)
            if peer_parts and hasattr(self, "_last_peer_score"):
                peer_score = getattr(self, "_last_peer_score", 50)
                peer_rankings = getattr(self, "_last_peer_rankings", {})

                # 同业评分映射到基本面评分（±10分）
                score += (peer_score - 50) * 0.2

                # 同业因子记录
                if "pe_ttm" in peer_rankings:
                    r = peer_rankings["pe_ttm"]
                    if r["percentile"] <= 30:
                        factors.append(f"PE同业折价(前{r['percentile']:.0f}%)")
                    elif r["percentile"] >= 70:
                        risk_flags.append(f"PE同业溢价(后{r['percentile']:.0f}%)")
                        score -= 3
                if "pb" in peer_rankings:
                    r = peer_rankings["pb"]
                    if r["percentile"] <= 30:
                        factors.append(f"PB同业折价(前{r['percentile']:.0f}%)")
                    elif r["percentile"] >= 70:
                        risk_flags.append(f"PB同业溢价(后{r['percentile']:.0f}%)")
                        score -= 3
                if "roe" in peer_rankings:
                    r = peer_rankings["roe"]
                    if r["percentile"] >= 70:
                        factors.append(f"ROE同业领先(前{r['percentile']:.0f}%)")
                        score += 3
                    elif r["percentile"] <= 30:
                        risk_flags.append(f"ROE同业落后(后{r['percentile']:.0f}%)")
                        score -= 3

                reasoning_parts.append(f"同业对比: 相对评分{peer_score}/100。")
        except Exception:
            pass  # 同业对比失败不影响主流程

        # ── 7. 最终信号确定 ──
        score = max(0, min(100, score))
        if signal == 0:
            signal = 1 if score >= 75 else (-1 if score <= 40 else 0)

        # 构建reasoning
        if len(reasoning_parts) == 1:
            # 没有详细的估值分析，构建通用描述
            desc_parts = []
            if factors:
                desc_parts.append("积极因素：" + "；".join(factors[:3]) + "。")
            if risk_flags:
                desc_parts.append("风险因素：" + "；".join(risk_flags[:3]) + "。")
            reasoning_parts.append("".join(desc_parts) if desc_parts else "基于有限财务数据的规则引擎推断。")

        reasoning = " ".join(reasoning_parts)

        # 构建sub_scores
        sub_scores = {
            "profitability": min(25, max(0, 15 if (roe and roe > 15) else (8 if (roe and roe > 8) else 5))),
            "growth": 10,  # 降级分析默认中等
            "safety": min(20, max(0, 15 if (debt and debt < 50) else (10 if (debt and debt < 70) else 5))),
            "valuation": min(20, max(0, 15 if (pe and 0 < pe < 20) else (10 if (pe and pe < 30) else 5))),
            "moat": 5,  # 降级分析默认最低
        }

        # 数据完整度限制最高分
        available_count = sum(1 for k in ["pe_ttm", "pb", "roe", "debt_ratio", "gross_margin"] if fundamentals.get(k) is not None)
        if available_count < 2:
            score = min(score, 60)
            confidence = 0.45
            risk_flags.append("可用财务指标极少，推断置信度低")

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            key_factors=factors if factors else ["基于有限数据的规则引擎推断"],
            risk_flags=risk_flags if risk_flags else ["降级分析，置信度有限"],
            raw_data={
                "signal": signal,
                "confidence": confidence,
                "fundamental_score": score,
                "sub_scores": sub_scores,
                "valuation_gap": self._infer_valuation_gap(snapshot),
                "financial_health": financial_health,
            },
        )

    def _compute_financial_trends(self, quarters: list) -> Dict[str, Any]:
        """计算财务指标趋势 — 纵向分析引擎

        从季度数据中提取趋势信号，包括：
        - 线性回归趋势方向（上升/下降/平稳）
        - 盈利质量评分（利润增速 vs 营收增速匹配度）
        - 连续改善/恶化检测
        """
        if not quarters or len(quarters) < 2:
            return {
                "roe_trend": "数据不足",
                "gross_margin_trend": "数据不足",
                "revenue_growth_trend": "数据不足",
                "profit_growth_trend": "数据不足",
                "latest_revenue_growth": 0.0,
                "latest_profit_growth": 0.0,
                "latest_roe": 0.0,
                "latest_gross_margin": 0.0,
                "earnings_quality_score": 50,
                "continuous_roe_up": False,
                "continuous_gm_down": False,
                "trend_flags": [],
            }

        # 按季度排序
        sorted_q = sorted(quarters, key=lambda x: x.get("quarter", ""))

        # 提取有效序列
        def extract_seq(key: str) -> list:
            return [q.get(key) for q in sorted_q if q.get(key) is not None]

        roes = extract_seq("roe")
        gms = extract_seq("gross_margin")
        revs = extract_seq("revenue")
        profits = extract_seq("net_profit")

        def trend_direction(values: list) -> str:
            """基于线性回归斜率判断趋势方向"""
            if len(values) < 3:
                return "数据不足"
            n = len(values)
            x = list(range(n))
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            denominator = n * sum_x2 - sum_x ** 2
            if denominator == 0:
                return "平稳"
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            # 以最新值为基准，判断相对变化率
            latest = values[-1]
            if latest == 0:
                return "平稳"
            if slope > abs(latest) * 0.03:   # 每季增长 > 3%
                return "上升"
            elif slope < -abs(latest) * 0.03:
                return "下降"
            return "平稳"

        def growth_rate(values: list) -> float:
            """计算最近两期的同比增长率"""
            if len(values) < 2:
                return 0.0
            prev, curr = values[-2], values[-1]
            if prev == 0:
                return 0.0
            return (curr - prev) / abs(prev) * 100

        # 检测连续改善/恶化（最近3期）
        def is_monotonic(values: list, direction: str, min_len: int = 3) -> bool:
            """检测序列是否单调递增/递减"""
            if len(values) < min_len:
                return False
            recent = values[-min_len:]
            if direction == "up":
                return all(recent[i] < recent[i + 1] for i in range(min_len - 1))
            elif direction == "down":
                return all(recent[i] > recent[i + 1] for i in range(min_len - 1))
            return False

        # ── 盈利质量评分 ──
        eq_score = 50
        trend_flags = []

        rev_growth = growth_rate(revs)
        profit_growth = growth_rate(profits)

        if len(revs) >= 2 and len(profits) >= 2:
            if profit_growth > rev_growth:
                eq_score += 15
                trend_flags.append("利润增速超营收，盈利质量改善(+15)")
            elif profit_growth < rev_growth * 0.3:
                eq_score -= 10
                trend_flags.append("增收不增利，盈利质量恶化(-10)")
            elif rev_growth > 0 and profit_growth > 0:
                eq_score += 5
                trend_flags.append("营收利润双增长(+5)")

        # ROE 连续改善检测
        roe_up = is_monotonic(roes, "up", min_len=3)
        if roe_up:
            eq_score += 10
            trend_flags.append("ROE连续3季上升，盈利能力持续改善(+10)")

        # 毛利率连续恶化检测
        gm_down = is_monotonic(gms, "down", min_len=2)
        if gm_down:
            eq_score -= 10
            trend_flags.append("毛利率连续下降，竞争或成本压力(-10)")

        # 营收连续增长检测
        rev_up = is_monotonic(revs, "up", min_len=3)
        if rev_up:
            eq_score += 5
            trend_flags.append("营收连续3季增长，成长性确认(+5)")

        # 利润连续下滑检测
        prof_down = is_monotonic(profits, "down", min_len=2)
        if prof_down:
            eq_score -= 8
            trend_flags.append("净利润连续下滑，经营压力加大(-8)")

        eq_score = max(0, min(100, eq_score))

        return {
            "roe_trend": trend_direction(roes),
            "gross_margin_trend": trend_direction(gms),
            "revenue_growth_trend": trend_direction(revs),
            "profit_growth_trend": trend_direction(profits),
            "latest_revenue_growth": round(rev_growth, 1),
            "latest_profit_growth": round(profit_growth, 1),
            "latest_roe": round(roes[-1], 1) if roes else 0.0,
            "latest_gross_margin": round(gms[-1], 1) if gms else 0.0,
            "earnings_quality_score": eq_score,
            "continuous_roe_up": roe_up,
            "continuous_gm_down": gm_down,
            "trend_flags": trend_flags,
        }

    def _fetch_forecast_data(self, snapshot: StockSnapshot) -> Optional[Dict[str, Any]]:
        """获取分析师一致预期数据

        优先从 fundamentals 中读取（Orchestrator 可能已预注入），
        否则现场调用 StockDataTool 获取。
        """
        # Level 1: 预计算数据
        forecast = snapshot.fundamentals.get("forecast")
        if forecast and isinstance(forecast, dict) and forecast.get("forecast_eps_y1"):
            return forecast

        # Level 2: 现场获取
        try:
            from agent.tools.stock_data_tool import StockDataTool
            tool = StockDataTool.get_instance()
            return tool.get_consensus_forecast(snapshot.stock_code)
        except Exception:
            return None

    def _compute_forward_metrics(
        self, snapshot: StockSnapshot, forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于一致预期计算 Forward 估值指标"""
        current_price = snapshot.current_price
        eps_y1 = forecast.get("forecast_eps_y1")
        eps_y2 = forecast.get("forecast_eps_y2")
        eps_y3 = forecast.get("forecast_eps_y3")

        result = {
            "forward_pe_y1": None,
            "forward_pe_y2": None,
            "forward_pe_y3": None,
            "eps_growth_y1y2": None,
            "eps_growth_y2y3": None,
            "peg_y1": None,
            "peg_y2": None,
            "consensus_strength": "未知",
            "research_report_count": forecast.get("research_report_count", 0),
            "rating_buy_pct": forecast.get("rating_buy_pct", 0),
        }

        # Forward PE
        if current_price and current_price > 0:
            if eps_y1 and eps_y1 > 0:
                result["forward_pe_y1"] = round(current_price / eps_y1, 2)
            if eps_y2 and eps_y2 > 0:
                result["forward_pe_y2"] = round(current_price / eps_y2, 2)
            if eps_y3 and eps_y3 > 0:
                result["forward_pe_y3"] = round(current_price / eps_y3, 2)

        # EPS 预测增速
        if eps_y1 and eps_y2 and eps_y1 > 0:
            result["eps_growth_y1y2"] = round((eps_y2 - eps_y1) / eps_y1 * 100, 1)
        if eps_y2 and eps_y3 and eps_y2 > 0:
            result["eps_growth_y2y3"] = round((eps_y3 - eps_y2) / eps_y2 * 100, 1)

        # PEG = Forward PE / 盈利增速
        if result["forward_pe_y1"] and result["eps_growth_y1y2"] and result["eps_growth_y1y2"] > 0:
            result["peg_y1"] = round(result["forward_pe_y1"] / result["eps_growth_y1y2"], 2)
        if result["forward_pe_y2"] and result["eps_growth_y2y3"] and result["eps_growth_y2y3"] > 0:
            result["peg_y2"] = round(result["forward_pe_y2"] / result["eps_growth_y2y3"], 2)

        # 一致预期强度判断
        report_count = forecast.get("research_report_count", 0)
        rating_buy_pct = forecast.get("rating_buy_pct", 0)
        if report_count >= 20 and rating_buy_pct >= 0.8:
            result["consensus_strength"] = "强烈看多"
        elif report_count >= 10 and rating_buy_pct >= 0.6:
            result["consensus_strength"] = "偏多"
        elif report_count >= 5 and rating_buy_pct <= 0.4:
            result["consensus_strength"] = "偏空"
        elif report_count < 5:
            result["consensus_strength"] = "覆盖不足"
        else:
            result["consensus_strength"] = "中性"

        return result

    def _get_peer_companies(
        self, industry: str, stock_code: str, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """获取同行业市值最接近的 peer 公司"""
        if not industry or not stock_code:
            return []
        try:
            from agent.tools.stock_data_tool import StockDataTool
            tool = StockDataTool.get_instance()
            return tool.get_industry_peers(stock_code, industry, top_n)
        except Exception:
            return []

    def _build_peer_comparison(self, snapshot: StockSnapshot) -> List[str]:
        """构建同业对比矩阵（含排名、Z-Score、百分位）"""
        fundamentals = snapshot.fundamentals
        industry = fundamentals.get("industry", "")
        stock_code = snapshot.stock_code

        peers = self._get_peer_companies(industry, stock_code, top_n=5)
        if not peers:
            return []

        # 对比指标定义
        metrics = ["pe_ttm", "pb", "roe", "gross_margin", "revenue_yoy", "debt_ratio"]
        metric_labels = {
            "pe_ttm": "PE(TTM)",
            "pb": "PB",
            "roe": "ROE",
            "gross_margin": "毛利率",
            "revenue_yoy": "营收增速",
            "debt_ratio": "负债率",
        }
        # 指标方向：True=越大越好，False=越小越好
        metric_higher_better = {
            "pe_ttm": False,
            "pb": False,
            "roe": True,
            "gross_margin": True,
            "revenue_yoy": True,
            "debt_ratio": False,
        }

        # 构建对比数据集
        peer_data = []
        for p in peers:
            peer_data.append({
                "code": p["stock_code"],
                "name": p["stock_name"],
                "market_cap": p.get("market_cap"),
                "pe_ttm": p.get("pe_ttm"),
                "pb": p.get("pb"),
            })

        # 加入当前股票
        current_data = {
            "code": stock_code,
            "name": snapshot.stock_name,
            "market_cap": fundamentals.get("market_cap"),
            "pe_ttm": fundamentals.get("pe_ttm"),
            "pb": fundamentals.get("pb"),
            "roe": fundamentals.get("roe"),
            "gross_margin": fundamentals.get("gross_margin"),
            "revenue_yoy": fundamentals.get("revenue_yoy"),
            "debt_ratio": fundamentals.get("debt_ratio"),
        }
        all_data = peer_data + [current_data]

        parts = ["### 同业对比矩阵（市值最接近的5家公司）", ""]
        parts.append("| 公司 | 市值(亿) | PE | PB |")
        parts.append("|------|---------|----|----|")

        for d in all_data:
            cap = d.get("market_cap")
            cap_str = f"{cap/1e8:.1f}" if cap else "N/A"
            pe = d.get("pe_ttm")
            pe_str = f"{pe:.1f}" if pe is not None else "N/A"
            pb_val = d.get("pb")
            pb_str = f"{pb_val:.2f}" if pb_val is not None else "N/A"
            marker = " ← 当前" if d["code"] == stock_code else ""
            parts.append(f"| {d['name']}{marker} | {cap_str} | {pe_str} | {pb_str} |")

        parts.append("")

        # 计算排名和 Z-Score
        rankings = {}
        for metric in metrics:
            values = []
            for d in all_data:
                v = d.get(metric)
                if v is not None:
                    values.append((d["code"], v))

            if len(values) < 3:
                continue

            codes, vals = zip(*values)
            vals = list(vals)
            n = len(vals)
            mean = sum(vals) / n
            std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0

            # 排序（考虑指标方向）
            reverse = metric_higher_better[metric]
            sorted_vals = sorted(values, key=lambda x: x[1], reverse=reverse)
            rank = next(i for i, (c, _) in enumerate(sorted_vals, 1) if c == stock_code)
            percentile = (n - rank + 1) / n * 100

            current_val = current_data.get(metric)
            zscore = (current_val - mean) / std if std > 0 and current_val is not None else 0

            rankings[metric] = {
                "rank": rank,
                "total": n,
                "percentile": percentile,
                "zscore": zscore,
                "mean": mean,
                "std": std,
                "value": current_val,
            }

        if rankings:
            parts.append("**当前股票在同业中的位置**:")
            for metric, r in rankings.items():
                label = metric_labels[metric]
                z_str = f"(Z={r['zscore']:+.2f})" if r["std"] > 0 else ""
                parts.append(
                    f"- {label}: 排名 {r['rank']}/{r['total']} (前{r['percentile']:.0f}%) {z_str}"
                )

            # 综合同业评分
            peer_score = 50
            peer_factors = []
            if "pe_ttm" in rankings:
                r = rankings["pe_ttm"]
                if r["percentile"] <= 30:  # PE 排名前30%（低PE）
                    peer_score += 10
                    peer_factors.append("PE同业折价(+10)")
                elif r["percentile"] >= 70:
                    peer_score -= 10
                    peer_factors.append("PE同业溢价(-10)")

            if "pb" in rankings:
                r = rankings["pb"]
                if r["percentile"] <= 30:
                    peer_score += 8
                    peer_factors.append("PB同业折价(+8)")
                elif r["percentile"] >= 70:
                    peer_score -= 8
                    peer_factors.append("PB同业溢价(-8)")

            if "roe" in rankings:
                r = rankings["roe"]
                if r["percentile"] >= 70:
                    peer_score += 10
                    peer_factors.append("ROE同业领先(+10)")
                elif r["percentile"] <= 30:
                    peer_score -= 8
                    peer_factors.append("ROE同业落后(-8)")

            if "gross_margin" in rankings:
                r = rankings["gross_margin"]
                if r["percentile"] >= 70:
                    peer_score += 5
                    peer_factors.append("毛利率同业领先(+5)")

            peer_score = max(0, min(100, peer_score))
            parts.append("")
            parts.append(f"- **同业相对评分**: {peer_score}/100")
            if peer_factors:
                parts.append(f"- **同业优势/劣势**: {'; '.join(peer_factors)}")

            # 保存 rankings 供 fallback 使用
            self._last_peer_rankings = rankings
            self._last_peer_score = peer_score

        parts.append("")
        return parts

    def _get_industry_benchmark(self, industry: str) -> Dict[str, float]:
        """获取行业估值基准"""
        for key, benchmark in self.INDUSTRY_BENCHMARKS.items():
            if key in industry:
                return benchmark
        # 默认基准
        return {"pe": 25.0, "pb": 2.5, "roe": 10.0}

    def _infer_financial_health(self, snapshot: StockSnapshot) -> str:
        """推断财务健康度（红绿灯）"""
        fundamentals = snapshot.fundamentals
        debt = fundamentals.get("debt_ratio")
        cr = fundamentals.get("current_ratio")
        ocf = fundamentals.get("operating_cash_flow")
        pb = fundamentals.get("pb")
        pe = fundamentals.get("pe_ttm")

        red_flags = 0
        green_flags = 0

        if debt is not None:
            if debt > 70:
                red_flags += 1
            elif debt < 50:
                green_flags += 1

        if cr is not None:
            if cr < 1.0:
                red_flags += 1
            elif cr > 1.5:
                green_flags += 1

        if ocf is not None:
            if ocf < 0:
                red_flags += 1
            else:
                green_flags += 1

        if pb is not None and pe is not None and pe > 0:
            implied_roe = pb / pe
            if pb > 10 and (implied_roe < 0.08 or implied_roe > 0.50):
                red_flags += 1

        if pe is not None and pe < 0:
            red_flags += 1

        if red_flags >= 2:
            return "[危险] 危险 | 多项财务指标警示"
        if green_flags >= 2 and red_flags == 0:
            return "[安全] 安全 | 财务指标健康"
        return "[注意] 注意 | 部分指标需关注"

    def _infer_valuation_gap(self, snapshot: StockSnapshot) -> str:
        """推断估值缺口描述"""
        fundamentals = snapshot.fundamentals
        pe = fundamentals.get("pe_ttm")
        pb = fundamentals.get("pb")
        industry = fundamentals.get("industry", "")
        ref = self._get_industry_benchmark(industry)

        parts = []

        if pe is not None and pb is not None and pe > 0:
            implied_roe = pb / pe
            if pb > 10 and pe < 50:
                if implied_roe < 0.08:
                    parts.append(
                        f"PB({pb:.2f})极高但PE({pe:.2f})正常，隐含ROE仅{implied_roe:.2%}，"
                        f"盈利能力极差，净资产溢价异常。"
                    )
                elif implied_roe > 0.50:
                    parts.append(
                        f"PB({pb:.2f})极高但PE({pe:.2f})正常，隐含ROE高达{implied_roe:.2%}，"
                        f"远超正常水平(5%-30%)，极可能因净资产被压低或一次性收益导致，不可持续。"
                    )
                else:
                    parts.append(
                        f"PB({pb:.2f})极高但PE({pe:.2f})正常，隐含ROE为{implied_roe:.2%}，"
                        f"需进一步分析净资产质量。"
                    )
            elif pe < 0:
                parts.append(f"PE为负({pe:.2f})，公司亏损，基本面恶化。")
            else:
                pe_ratio = pe / ref["pe"]
                pb_ratio = pb / ref["pb"] if pb else 1
                if pe_ratio < 0.7 and pb_ratio < 0.8:
                    parts.append(f"PE({pe:.1f})和PB({pb:.2f})均低于行业平均，估值有吸引力。")
                elif pe_ratio > 1.3 and pb_ratio > 1.3:
                    parts.append(f"PE({pe:.1f})和PB({pb:.2f})均高于行业平均，估值偏贵。")
                else:
                    parts.append(f"PE({pe:.1f})相对行业{pe_ratio:.1f}x，PB({pb:.2f})相对行业{pb_ratio:.1f}x。")
        elif pe is not None:
            if pe < 0:
                parts.append("公司亏损，无法评估估值吸引力。")
            else:
                parts.append(f"PE={pe:.1f}，{'低于' if pe < ref['pe'] else '高于'}行业平均({ref['pe']:.1f})。")
        elif pb is not None:
            parts.append(f"PB={pb:.2f}，{'低于' if pb < ref['pb'] else '高于'}行业平均({ref['pb']:.1f})。")
        else:
            parts.append("估值数据不可用，无法评估估值水平。")

        return " ".join(parts)

    def _default_prompt(self) -> str:
        return """你是一位深度价值分析师（Graham-Buffett学派），同时兼顾成长性评估。

## 核心能力
1. 通过财务数据判断企业内在价值与股价的匹配度
2. **从有限数据中推导深层结论**——即使部分指标缺失，也能通过估值联动关系推断
3. PE-PB-ROE三角分析：ROE ≈ PB/PE
4. 财务健康度红绿灯评估

## 评分模型（0-100）
- 盈利质量（25分）：ROE/隐含ROE、毛利率、净利率
- 成长性（25分）：营收/利润增速、增速持续性
- 财务安全（20分）：负债率、流动性、现金流
- 估值吸引力（20分）：PE/PB相对行业和历史
- 行业地位（10分）：市值、行业排名

## 约束
- score >= 75 → signal=1；score <= 40 → signal=-1；否则 0
- PE < 0（亏损）时，signal 不得为 1
- PB > 10 且 PE 正常时，必须深入分析隐含ROE并给出负面倾向
- **禁止使用"数据缺失"、"无法判断"等消极表述**
- 当数据不完整时，基于估值联动关系做合理推断

## 输出严格JSON格式
{
  "signal": 0,
  "confidence": 0.72,
  "fundamental_score": 55,
  "sub_scores": {"profitability":15, "growth":15, "safety":10, "valuation":10, "moat":5},
  "valuation_gap": "估值缺口描述",
  "financial_health": "[安全]/[注意]/[危险] 健康度描述",
  "reasoning": "详细分析（100-200字，必须包含具体数字和推断逻辑）",
  "key_factors": ["因素1", "因素2"],
  "risk_flags": ["风险1"]
}"""
