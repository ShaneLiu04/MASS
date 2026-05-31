"""
MASS CA-Agent: 资金面分析师 v2.1
支持: 筹码分布 + 北向资金深度 + 机构行为追踪 + 融资融券深度分析
"""
from typing import Dict, Any, Optional, List

from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.tools.stock_data_tool import StockDataTool


class CA_Agent(BaseAgent):
    """资金面分析师 Agent"""
    
    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """资金面分析"""
        user_prompt = self._build_ca_prompt(snapshot)
        
        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)
            
            # 校验 smart_money_direction 枚举
            valid_directions = ["强烈建仓", "建仓期", "观望", "派发期", "强烈派发"]
            smd = parsed.get("smart_money_direction", "观望")
            if smd not in valid_directions:
                parsed["smart_money_direction"] = "观望"
            
            return self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"CA-Agent分析失败: {e}")
            return self._fallback_opinion(snapshot)
    
    def _analyze_chip_distribution(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """筹码分布分析 — 基于K线和成交量估算主力成本与筹码状态

        核心指标：
        - CR90: 90%成本集中度 = (P95-P05)/当前价，越低越集中
        - 主力成本区: 近60日VWAP ± 5%
        - 获利盘比例: 历史收盘价低于当前价的占比
        - 筹码锁定度: 近期换手率相对前期下降程度
        """
        kline = snapshot.kline_df
        if kline is None or len(kline) < 30:
            return {
                "concentration": 50.0,
                "cr90": 50.0,
                "main_cost_low": 0.0,
                "main_cost_high": 0.0,
                "main_cost_center": 0.0,
                "price_to_cost_ratio": 0.0,
                "profit_ratio": 50.0,
                "lock_ratio": 30.0,
                "chip_status": "数据不足",
            }

        import pandas as pd

        # 使用近 60 日数据计算
        recent = kline.tail(60).copy()
        if "close" not in recent.columns or "volume" not in recent.columns:
            return {
                "concentration": 50.0, "cr90": 50.0,
                "main_cost_low": 0.0, "main_cost_high": 0.0,
                "main_cost_center": 0.0, "price_to_cost_ratio": 0.0,
                "profit_ratio": 50.0, "lock_ratio": 30.0,
                "chip_status": "数据不足",
            }

        closes = recent["close"].values
        volumes = recent["volume"].values
        current = snapshot.current_price

        # 1. 主力成本中心 = 成交量加权平均价 (VWAP)
        total_vol = volumes.sum()
        vwap = (closes * volumes).sum() / total_vol if total_vol > 0 else closes.mean()

        # 2. CR90 — 90% 成本集中度（更精确的筹码集中度指标）
        # 按成交量加权计算价格分位数
        # 方法：将每条 K 线视为一个价格点，权重为成交量
        sorted_idx = closes.argsort()
        sorted_prices = closes[sorted_idx]
        sorted_vols = volumes[sorted_idx]
        cumvol = sorted_vols.cumsum()
        cumvol_pct = cumvol / cumvol[-1] if cumvol[-1] > 0 else cumvol

        # 找到 5% 和 95% 分位价格
        p05_idx = max(0, (cumvol_pct < 0.05).sum() - 1)
        p95_idx = min(len(sorted_prices) - 1, (cumvol_pct <= 0.95).sum())
        p05 = float(sorted_prices[p05_idx]) if p05_idx >= 0 else sorted_prices[0]
        p95 = float(sorted_prices[p95_idx]) if p95_idx < len(sorted_prices) else sorted_prices[-1]

        cr90 = (p95 - p05) / current * 100 if current > 0 else 0

        # 3. 获利盘比例
        profit_ratio = (closes < current).mean() * 100

        # 4. 筹码锁定度 — 近期换手率下降程度
        if len(kline) >= 60:
            recent_vol_mean = kline.tail(20)["volume"].mean()
            prior_vol_mean = kline.tail(60).head(40)["volume"].mean()
            lock_ratio = max(0, (1 - recent_vol_mean / prior_vol_mean) * 100) if prior_vol_mean > 0 else 0
        else:
            lock_ratio = 30.0

        # 5. 筹码状态判断
        if cr90 < 10 and profit_ratio < 40:
            chip_status = "高度集中且套牢为主（主力吸筹期）"
        elif cr90 < 15 and profit_ratio > 60:
            chip_status = "高度集中且获利为主（拉升或派发期）"
        elif cr90 > 25:
            chip_status = "筹码分散（无主力控盘）"
        elif lock_ratio > 50:
            chip_status = "高度锁定（主力锁仓）"
        else:
            chip_status = "筹码相对分散"

        return {
            "concentration": round(cr90, 2),          # 旧的集中度字段（兼容）
            "cr90": round(cr90, 2),                   # 90%成本集中度
            "main_cost_low": round(vwap * 0.95, 2),
            "main_cost_high": round(vwap * 1.05, 2),
            "main_cost_center": round(vwap, 2),
            "price_to_cost_ratio": round((current - vwap) / vwap * 100, 1) if vwap > 0 else 0,
            "profit_ratio": round(profit_ratio, 1),
            "lock_ratio": round(min(100, lock_ratio), 1),
            "chip_status": chip_status,
        }

    def _count_consecutive_inflow(self, north_data: List[Dict[str, Any]]) -> int:
        """计算北向资金连续净流入天数"""
        if not north_data:
            return 0
        # 按日期排序，取最近的数据
        sorted_data = sorted(north_data, key=lambda x: x.get("date", ""), reverse=True)
        consecutive = 0
        for d in sorted_data:
            net = d.get("net", 0)
            if net > 0:
                consecutive += 1
            else:
                break
        return consecutive

    def _analyze_institution_behavior(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        机构行为追踪分析 — 方向二

        分析维度：
        1. 机构持仓季度变化（持股比例、占流通股比例变化）
        2. 持仓机构数量变化（新增/减少机构数）
        3. 基金重仓变化（基金家数、持股变动比例）
        """
        result = {
            "inst_hold": None,
            "fund_hold": None,
            "inst_signal": "neutral",
            "inst_score": 0,  # -10 ~ +10
        }

        try:
            tool = StockDataTool.get_instance()

            # 1. 机构持仓汇总
            inst = tool.get_institute_hold_summary(snapshot.stock_code)
            if inst:
                result["inst_hold"] = inst
                score = 0
                signal = "neutral"

                # 持股比例增幅
                hrc = inst.get("hold_ratio_change") or 0
                frc = inst.get("float_ratio_change") or 0
                icc = inst.get("institution_count_change") or 0

                if hrc > 2:
                    score += 5
                    signal = "strong_buy"
                elif hrc > 0.5:
                    score += 3
                    signal = "buy"
                elif hrc < -2:
                    score -= 5
                    signal = "strong_sell"
                elif hrc < -0.5:
                    score -= 3
                    signal = "sell"

                # 机构数变化
                if icc > 20:
                    score += 3
                elif icc > 5:
                    score += 1
                elif icc < -20:
                    score -= 3
                elif icc < -5:
                    score -= 1

                result["inst_score"] = max(-10, min(10, score))
                result["inst_signal"] = signal

            # 2. 基金重仓
            fund = tool.get_fund_holdings(snapshot.stock_code)
            if fund:
                result["fund_hold"] = fund
                # 如果基金数据与机构数据信号一致，增强评分
                fund_change_pct = fund.get("hold_change_pct") or 0
                if result["inst_signal"] in ("buy", "strong_buy") and fund_change_pct > 0:
                    result["inst_score"] = min(10, result["inst_score"] + 2)
                elif result["inst_signal"] in ("sell", "strong_sell") and fund_change_pct < 0:
                    result["inst_score"] = max(-10, result["inst_score"] - 2)

        except Exception as e:
            logger.debug(f"机构行为分析数据获取失败: {e}")

        return result

    def _analyze_margin_depth(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        融资融券深度分析 — 方向三

        分析维度：
        1. 融资余额/流通市值比率 → 杠杆水平
        2. 融资买入额/成交额比率 → 杠杆资金活跃度
        3. 融券余量变化 → 做空力量
        4. 融资余额趋势判断
        """
        result = {
            "margin_detail": None,
            "margin_ratio": None,
            "margin_buy_ratio": None,
            "leveraged_trend": "unknown",
            "short_pressure": "unknown",
            "margin_score": 0,  # -10 ~ +10
        }

        try:
            tool = StockDataTool.get_instance()
            margin = tool.get_margin_detail(snapshot.stock_code)
            if not margin:
                return result

            result["margin_detail"] = margin

            fund = snapshot.fund_flow
            fundamentals = snapshot.fundamentals or {}
            kline = snapshot.kline_df

            # 1. 融资余额 / 流通市值 比率
            float_cap = fundamentals.get("float_market_cap")
            margin_balance = margin.get("margin_balance")
            if float_cap and float_cap > 0 and margin_balance:
                result["margin_ratio"] = round(margin_balance / float_cap * 100, 3)

            # 2. 融资买入额 / 近期日均成交额 比率
            margin_buy = margin.get("margin_buy_amount")
            if kline is not None and not kline.empty and margin_buy:
                if "amount" in kline.columns:
                    avg_amount = kline.tail(5)["amount"].mean()
                elif "volume" in kline.columns and "close" in kline.columns:
                    # 估算成交额 = 成交量 * 收盘价
                    avg_amount = (kline.tail(5)["volume"] * kline.tail(5)["close"]).mean()
                else:
                    avg_amount = None

                if avg_amount and avg_amount > 0:
                    result["margin_buy_ratio"] = round(margin_buy / avg_amount * 100, 2)

            # 3. 杠杆趋势判断
            margin_change = fund.get("margin_balance_change", 0)
            if margin_change > 0:
                result["leveraged_trend"] = "上升"
            elif margin_change < 0:
                result["leveraged_trend"] = "下降"
            else:
                result["leveraged_trend"] = "平稳"

            # 4. 融券压力
            short_volume = margin.get("short_volume", 0)
            if short_volume and short_volume > 1000000:  # 100万股以上
                result["short_pressure"] = "高"
            elif short_volume and short_volume > 100000:
                result["short_pressure"] = "中"
            else:
                result["short_pressure"] = "低"

            # 5. 评分
            score = 0
            mr = result["margin_ratio"]
            mbr = result["margin_buy_ratio"]

            if mr is not None:
                if mr > 8:
                    score -= 4  # 杠杆过高，风险大
                    result["leveraged_trend"] = "过高杠杆"
                elif mr > 5:
                    score -= 2
                elif mr > 2:
                    score += 1  # 适度杠杆，活跃

            if mbr is not None:
                if mbr > 20:
                    score -= 3  # 过度杠杆买入
                elif mbr > 10:
                    score -= 1
                elif mbr > 3:
                    score += 2  # 健康杠杆参与

            if result["short_pressure"] == "高":
                score -= 3
            elif result["short_pressure"] == "中":
                score -= 1

            # 融资余额变化趋势修正
            if margin_change > 5000:
                score += 2 if score >= 0 else 1  # 杠杆加速上升，仅当原本偏积极时加分
            elif margin_change < -5000:
                score -= 2  # 去杠杆

            result["margin_score"] = max(-10, min(10, score))

        except Exception as e:
            logger.debug(f"融资融券深度分析失败: {e}")

        return result

    def _build_ca_prompt(self, snapshot: StockSnapshot) -> str:
        """构建资金面分析Prompt — v2.1 增强版（筹码分布 + 北向深度 + 机构行为 + 融资融券）"""
        parts = [
            f"## 分析对象",
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            "",
            "## 分析任务",
            "你是一位资深资金面分析师，精通主力资金行为识别、筹码分布分析、北向资金解读、机构持仓追踪和融资融券研判。",
            "请基于以下多维资金数据给出交易信号，必须综合考量筹码状态、主力成本位置、机构行为变化和杠杆水平。",
            "",
        ]

        fund = snapshot.fund_flow

        # ── 1. 资金流向总览 ──
        parts.append("### 一、资金流向总览")
        key_fields = [
            ("main_net_inflow_10d", "主力近10日净流入(万)"),
            ("main_inflow_days", "主力连续流入天数"),
            ("north_bound_5d", "北向近5日净流入(万)"),
            ("north_bound_30d", "北向近30日净流入(万)"),
            ("margin_balance_change", "融资融券余额变化(万)"),
            ("institution_research_count", "机构研报覆盖数"),
        ]
        for key, label in key_fields:
            if key in fund:
                parts.append(f"- {label}: {fund[key]}")
        parts.append("")

        # ── 2. 筹码分布分析 ──
        chip = self._analyze_chip_distribution(snapshot)
        parts.extend([
            "### 二、筹码分布分析",
            f"- **筹码集中度(CR90)**: {chip['cr90']:.1f}% (越低越集中)",
            f"  - 解读: {'高度集中' if chip['cr90'] < 15 else ('相对集中' if chip['cr90'] < 25 else '分散')}",
            f"- 主力成本区: {chip['main_cost_low']:.2f} ~ {chip['main_cost_high']:.2f} (中心 {chip['main_cost_center']:.2f})",
            f"- 当前价距主力成本: {chip['price_to_cost_ratio']:+.1f}%",
            f"- 获利盘比例: {chip['profit_ratio']:.1f}%",
            f"  - 解读: {'获利盘为主，注意抛压' if chip['profit_ratio'] > 70 else ('套牢盘为主，突破需放量' if chip['profit_ratio'] < 30 else '多空均衡')}",
            f"- 筹码锁定度: {chip['lock_ratio']:.1f}% {'(高锁定，主力锁仓)' if chip['lock_ratio'] > 50 else '(正常换手)'}",
            f"- **筹码状态**: {chip['chip_status']}",
            "",
            "**筹码分析指引**：",
            "1. CR90 < 15% + 当前价在主力成本区下方 + 获利盘 < 30% → 主力吸筹完成，拉升概率大",
            "2. CR90 < 15% + 获利盘 > 70% → 主力可能进入派发阶段，需警惕",
            "3. 当前价突破主力成本区 + 筹码锁定度 > 50% → 主升浪确认信号",
            "4. CR90 > 25% → 筹码分散，无主力控盘，趋势难以持续",
            "",
        ])

        # ── 3. 机构行为追踪（方向二 新增） ──
        inst_analysis = self._analyze_institution_behavior(snapshot)
        inst = inst_analysis.get("inst_hold")
        fund_h = inst_analysis.get("fund_hold")

        parts.append("### 三、机构行为追踪")
        if inst:
            parts.extend([
                f"- **报告期**: {inst.get('quarter', 'N/A')}",
                f"- 持仓机构数: {inst.get('institution_count', 'N/A')} (变化 {inst.get('institution_count_change', 0):+d})",
                f"- 机构持股比例: {inst.get('hold_ratio', 'N/A')}% (季度变化 {inst.get('hold_ratio_change', 0):+.2f}%)",
                f"- 占流通股比例: {inst.get('float_ratio', 'N/A')}% (季度变化 {inst.get('float_ratio_change', 0):+.2f}%)",
            ])
        else:
            parts.append("- 机构持仓数据暂不可用")

        if fund_h:
            parts.extend([
                f"- **基金重仓**: {fund_h.get('fund_count', 'N/A')} 家基金持有",
                f"- 持股变动: {fund_h.get('hold_change', 'N/A')} {fund_h.get('hold_change_pct', 'N/A')}%",
            ])
        else:
            parts.append("- 基金重仓数据暂不可用")

        parts.extend([
            "",
            "**机构行为指引**：",
            "- 机构持股比例季度增幅 > 2% + 机构数增加 > 20 家 → 机构强烈看好",
            "- 机构持股比例季度降幅 > 2% + 机构数减少 > 20 家 → 机构撤离",
            "- 基金重仓增持 + 机构持仓增持 → 双重机构看好信号",
            "- 机构持仓与主力流向方向一致 → 信号共振",
            "",
        ])

        # ── 4. 融资融券深度分析（方向三 新增） ──
        margin = self._analyze_margin_depth(snapshot)
        md = margin.get("margin_detail")

        parts.append("### 四、融资融券深度分析")
        if md:
            parts.extend([
                f"- **数据日期**: {md.get('date', 'N/A')}",
                f"- 融资余额: {md.get('margin_balance', 'N/A'):,.0f} 元",
                f"- 融资买入额: {md.get('margin_buy_amount', 'N/A'):,.0f} 元",
                f"- 融券余量: {md.get('short_volume', 'N/A'):,.0f} 股",
                f"- 融券余额: {md.get('short_balance', 'N/A'):,.0f} 元",
            ])
            if margin.get("margin_ratio") is not None:
                parts.append(f"- **融资余额/流通市值比率**: {margin['margin_ratio']:.3f}%")
            if margin.get("margin_buy_ratio") is not None:
                parts.append(f"- **融资买入额/成交额比率**: {margin['margin_buy_ratio']:.2f}%")
            parts.extend([
                f"- 杠杆趋势: {margin.get('leveraged_trend', 'N/A')}",
                f"- 融券压力: {margin.get('short_pressure', 'N/A')}",
            ])
        else:
            parts.append("- 融资融券明细数据暂不可用")

        # 如果有 margin_balance_change，也展示
        if "margin_balance_change" in fund:
            parts.append(f"- 融资余额变化: {fund['margin_balance_change']:+,.0f} 万")

        parts.extend([
            "",
            "**融资融券指引**：",
            "- 融资余额/流通市值 > 8% → 杠杆过高，需警惕强平风险",
            "- 融资余额/流通市值 2-5% → 健康杠杆水平",
            "- 融资买入额/成交额 > 20% → 杠杆资金过度涌入，短期见顶风险",
            "- 融资余额加速上升 + 股价滞涨 → 杠杆派发生态，危险信号",
            "- 融券余量大幅增加 → 做空力量增强，警惕回调",
            "",
        ])

        # ── 5. 每日流向 ──
        daily = fund.get("daily_flow", [])
        if daily:
            parts.extend([
                "### 五、近10日主力流向明细",
                "",
            ])
            for d in daily[-10:]:
                main_net = d.get("main_net_inflow", 0)
                retail_net = d.get("retail_net_inflow", 0)
                marker = "" if main_net > 0 and retail_net < 0 else ("" if main_net < 0 and retail_net > 0 else "")
                parts.append(
                    f"- {d.get('date', '')}: 主力{main_net:+.0f} 散户{retail_net:+.0f} {marker}"
                )
            parts.append("")

        # ── 6. 北向资金深度分析 ──
        north = fund.get("north_bound_detail", [])
        if north:
            north_sorted = sorted(north, key=lambda x: x.get("date", ""), reverse=True)
            net_5d = sum(d.get("net", 0) for d in north_sorted[:5])
            net_30d = sum(d.get("net", 0) for d in north_sorted[:30])
            consecutive = self._count_consecutive_inflow(north)
            parts.extend([
                "### 六、北向资金行为分析",
                f"- 近5日净流入: {net_5d:+.0f}万",
                f"- 近30日净流入: {net_30d:+.0f}万",
                f"- 连续净流入天数: {consecutive}天",
                f"- 北向持仓占比变化: {fund.get('north_holding_pct_change', 'N/A')}%",
                "",
                "**北向资金指引**：",
                "- 连续净流入 > 10 天 → 外资长线看好",
                "- 近30日大幅流入但近5日流出 → 外资短期获利了结",
                "- 北向与主力方向一致 → 信号共振，可信度高",
                "",
            ])

        # ── 7. 大宗交易 ──
        block = fund.get("block_trade", [])
        if block:
            parts.extend([
                "### 七、大宗交易",
                "",
            ])
            for b in block[-5:]:
                disc = b.get("discount", 0)
                disc_marker = "大幅折价" if disc < -5 else ("溢价" if disc > 0 else "")
                parts.append(
                    f"- {b.get('date', '')}: 成交额{b.get('volume', 0):.0f}万 "
                    f"折溢价{disc:+.1f}% {disc_marker}"
                )
            parts.append("")

        # ── 8. 分析指引 ──
        parts.extend([
            "## 资金面综合判断框架",
            "",
            "1. **主力资金方向**：连续流入天数 + 流入金额 + 散户反向行为",
            "2. **筹码状态**：集中度 + 获利盘 + 锁定度 + 主力成本位置",
            "3. **机构行为**：持仓比例变化 + 机构数量变化 + 基金重仓动向",
            "4. **杠杆水平**：融资余额/流通市值 + 融资买入活跃度 + 融券做空压力",
            "5. **北向资金**：短期(5日) vs 中期(30日)流向一致性",
            "6. **大宗交易**：折价幅度、频次、金额（折价>5%警惕机构出货）",
            "",
            "## 约束",
            "- smart_money_direction 必须从 [\"强烈建仓\", \"建仓期\", \"观望\", \"派发期\", \"强烈派发\"] 中选择",
            "- 筹码高度集中(CR90<15%) + 主力成本区下方 + 主力持续流入 + 机构增持 → 强烈建仓",
            "- 筹码分散(CR90>25%) + 获利盘>70% + 主力流出 + 机构减持 + 高杠杆 → 强烈派发",
            "",
            "输出严格JSON格式：",
            "{",
            '  "signal": 1,',
            '  "confidence": 0.80,',
            '  "capital_score": 82,',
            '  "reasoning": "分析理由（必须包含筹码分布、主力成本、机构行为和杠杆水平分析）",',
            '  "key_factors": ["因子1"],',
            '  "risk_flags": ["风险1"],',
            '  "smart_money_direction": "建仓期",',
            '  "retail_vs_institutional": "散户卖出，机构吸筹"',
            "}",
        ])

        return "\n".join(parts)
    
    def _fallback_opinion(self, snapshot: StockSnapshot) -> AgentOpinion:
        """v2.1 多维度降级分析 — 主力资金 + 筹码分布 + 北向资金 + 机构行为 + 融资融券"""
        fund = snapshot.fund_flow
        main_flow = fund.get("main_net_inflow_10d", 0)
        inflow_days = fund.get("main_inflow_days", 0)

        score = 50
        signal = 0
        factors = []
        risk_flags = []
        smart_money = "观望"
        extra_raw = {}

        # ── 1. 主力资金分析 ──
        if main_flow > 10000 and inflow_days >= 5:
            score += 15
            signal = 1
            factors.append("主力资金持续流入")
            smart_money = "建仓期"
        elif main_flow > 5000 and inflow_days >= 3:
            score += 8
            factors.append("主力资金流入")
        elif main_flow < -10000:
            score -= 15
            signal = -1
            factors.append("主力资金持续流出")
            smart_money = "派发期"
        elif main_flow < -5000:
            score -= 8
            risk_flags.append("主力资金流出")

        # ── 2. 筹码分布修正 ──
        try:
            chip = self._analyze_chip_distribution(snapshot)
            extra_raw["chip"] = chip
            cr90 = chip["cr90"]
            profit_ratio = chip["profit_ratio"]
            lock_ratio = chip["lock_ratio"]
            price_to_cost = chip["price_to_cost_ratio"]

            if cr90 < 15 and profit_ratio < 40 and price_to_cost < 5:
                score += 10
                factors.append("筹码集中且套牢为主（吸筹完成）")
                smart_money = "强烈建仓" if score > 70 else "建仓期"
            elif cr90 < 15 and profit_ratio > 70:
                score -= 10
                risk_flags.append("筹码集中但获利盘过高（派发风险）")
                smart_money = "派发期"
            elif cr90 > 25:
                score -= 5
                risk_flags.append("筹码分散（无主力控盘）")

            if lock_ratio > 50:
                score += 5
                factors.append("筹码高度锁定（主力锁仓）")
            elif lock_ratio < 10:
                score -= 3
                risk_flags.append("筹码松动（高换手）")

            if price_to_cost < -5:
                score += 3
                factors.append("当前价低于主力成本（被套空间）")
            elif price_to_cost > 15:
                score -= 5
                risk_flags.append("当前价远超主力成本（追高风险）")

        except Exception:
            pass

        # ── 3. 北向资金修正 ──
        north_5d = fund.get("north_bound_5d", 0)
        north_30d = fund.get("north_bound_30d", 0)
        if north_30d > 50000 and north_5d > 0:
            score += 5
            factors.append("北向资金中期大幅流入")
        elif north_30d < -30000:
            score -= 5
            risk_flags.append("北向资金中期流出")

        # ── 4. 机构行为修正（方向二） ──
        try:
            inst = self._analyze_institution_behavior(snapshot)
            extra_raw["institution"] = inst
            inst_score = inst.get("inst_score", 0)
            score += inst_score

            if inst_score >= 5:
                factors.append("机构持仓显著增持")
                if smart_money == "观望":
                    smart_money = "建仓期"
            elif inst_score >= 3:
                factors.append("机构持仓增持")
            elif inst_score <= -5:
                risk_flags.append("机构持仓显著减持")
                if smart_money == "观望":
                    smart_money = "派发期"
            elif inst_score <= -3:
                risk_flags.append("机构持仓减持")

            fund_h = inst.get("fund_hold")
            if fund_h:
                hold_change = fund_h.get("hold_change", "")
                if "增持" in hold_change or "新进" in hold_change:
                    factors.append("基金重仓增持")
                elif "减持" in hold_change:
                    risk_flags.append("基金重仓减持")

        except Exception:
            pass

        # ── 5. 融资融券修正（方向三） ──
        try:
            margin = self._analyze_margin_depth(snapshot)
            extra_raw["margin"] = margin
            margin_score = margin.get("margin_score", 0)
            score += margin_score

            if margin_score <= -5:
                risk_flags.append("杠杆水平过高或融券压力大")
            elif margin_score >= 3:
                factors.append("杠杆水平健康，融资活跃")

            if margin.get("leveraged_trend") == "过高杠杆":
                risk_flags.append("融资余额/流通市值比率过高，警惕强平")
            if margin.get("short_pressure") == "高":
                risk_flags.append("融券余量高，做空压力大")

        except Exception:
            pass

        # ── 6. 最终信号确定 ──
        score = max(0, min(100, score))
        if signal == 0:
            signal = 1 if score >= 70 else (-1 if score <= 35 else 0)

        confidence = 0.5 + abs(score - 50) / 100
        confidence = min(0.85, confidence)

        # 构建 reasoning
        reasoning_parts = ["【资金面规则引擎降级分析】"]
        if factors:
            reasoning_parts.append("积极信号: " + ";".join(factors[:4]) + "。")
        if risk_flags:
            reasoning_parts.append("风险提示: " + ";".join(risk_flags[:4]) + "。")

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning=" ".join(reasoning_parts),
            key_factors=factors if factors else ["资金面规则引擎分析"],
            risk_flags=risk_flags if risk_flags else ["LLM调用异常，使用规则引擎"],
            raw_data={
                "signal": signal,
                "confidence": confidence,
                "capital_score": score,
                "smart_money_direction": smart_money,
                **extra_raw,
            },
        )
    
    def _default_prompt(self) -> str:
        return """你是一位资金面分析师。请分析资金流向给出信号。

smart_money_direction 必须从 ["强烈建仓", "建仓期", "观望", "派发期", "强烈派发"] 中选择。

输出严格JSON格式：
{
  "signal": 1,
  "confidence": 0.80,
  "capital_score": 82,
  "reasoning": "理由",
  "key_factors": [],
  "risk_flags": [],
  "smart_money_direction": "建仓期",
  "retail_vs_institutional": "散户卖出，机构吸筹"
}"""
