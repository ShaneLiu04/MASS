"""
MASS RA-Agent: 风险控制官 (Risk Assessment Agent)
v2.0 增强：多情景压力测试、尾部风险指标、历史模拟法 VaR/CVaR、
       跳空风险分析、流动性风险量化、波动率期限结构
"""
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd
from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.models.agent_response import StressTestResult, StressScenario, PortfolioRiskResult, PortfolioPosition


class RA_Agent(BaseAgent):
    """
    风险控制官 Agent —— "先求不败，再求胜"

    v2.0 核心增强：
    1. 压力测试引擎：基准 / 乐观 / 悲观 / 黑天鹅 四情景分析
    2. 尾部风险指标：历史模拟法 VaR(95%) / CVaR(95%)
    3. 跳空风险统计：历史向上/向下跳空次数与平均幅度
    4. 流动性风险：Amihud 非流动性指标 + 成交额波动率
    5. 波动率期限结构：5日/20日/60日波动率对比
    6. 增强降级逻辑：LLM 失败时基于压力测试结果给出规则化仓位建议
    """

    # ── 情景参数（可配置） ──
    STRESS_BULL_MARKET_MOVE = 20.0    # 乐观情景市场涨幅(%)
    STRESS_BEAR_MARKET_MOVE = -20.0   # 悲观情景市场跌幅(%)
    STRESS_BLACK_SWAN_MOVE = -40.0    # 黑天鹅情景市场跌幅(%)
    MAX_DRAWDOWN_CAP = -80.0          # 最大回撤下限(%)
    BLOWUP_THRESHOLD = -50.0          # 爆仓阈值(%)

    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """执行风险分析，输出包含压力测试与组合风险的 AgentOpinion"""
        user_prompt = self._build_ra_prompt(snapshot, user_position)

        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)

            # ── 字段校验与截断 ──
            rl = parsed.get("risk_level", 3)
            parsed["risk_level"] = max(1, min(5, int(rl)))

            mpp = parsed.get("max_position_pct", 0.10)
            parsed["max_position_pct"] = max(0.05, min(0.50, float(mpp)))

            rrr = parsed.get("risk_reward_ratio", 1.0)
            parsed["risk_reward_ratio"] = max(0.0, float(rrr))

            # 事件风险强制提升 risk_level
            risk_flags = parsed.get("risk_flags", [])
            has_event_risk = any(kw in str(f) for f in risk_flags for kw in ("财报", "解禁", "监管", "问询", "ST", "退市"))
            if has_event_risk and parsed["risk_level"] < 3:
                parsed["risk_level"] = 3
                risk_flags.append("存在未披露重大事件，强制提升风险等级")
                parsed["risk_flags"] = risk_flags

            # 若 LLM 未返回 stress_test，用本地计算结果补全
            if "stress_test" not in parsed:
                stress = self._run_stress_tests(snapshot)
                parsed["stress_test"] = stress.model_dump()

            # 若 LLM 未返回 portfolio_risk，用本地计算结果补全
            if "portfolio_risk" not in parsed:
                portfolio = self._analyze_portfolio_risk(snapshot, user_position)
                parsed["portfolio_risk"] = portfolio.model_dump()

            # 若 LLM 未返回 dynamic_stop_loss，用本地计算结果补全
            if "dynamic_stop_loss" not in parsed:
                stop_loss_result = self._calculate_dynamic_stop_loss(snapshot)
                parsed["dynamic_stop_loss"] = stop_loss_result.model_dump()

            return self._build_default_opinion(
                signal=0,  # RA-Agent 只输出风险信息，不参与方向投票
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"RA-Agent 分析失败: {e}")
            return self._fallback_opinion(snapshot, user_position)

    # ═══════════════════════════════════════════════════════════════
    #  Prompt 构建
    # ═══════════════════════════════════════════════════════════════
    def _build_ra_prompt(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> str:
        """构建风险分析 Prompt —— 融合压力测试、尾部风险、流动性风险"""
        parts = [
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            "",
            "=== 基础风险指标 ===",
        ]

        risk = snapshot.risk_metrics
        key_fields = [
            "annual_volatility", "max_drawdown", "sharpe_ratio",
            "downside_std", "win_rate", "beta", "avg_amount_5d",
            "annual_return", "beta_source",
        ]
        for k in key_fields:
            if k in risk:
                parts.append(f"{k}: {risk[k]}")

        # 技术指标中的风险相关数据
        ind = snapshot.indicators
        if ind:
            parts.extend(["", "=== 技术指标风险数据 ==="])
            parts.append(f"ATR(14): {ind.get('atr14', 'N/A')} ({ind.get('atr_pct', 'N/A')}%)")
            parts.append(f"20日波动率: {ind.get('volatility_20d', 'N/A')}%")
            parts.append(f"60日最大回撤: {ind.get('max_drawdown_60d', 'N/A')}%")
            parts.append(f"量价背离: {ind.get('price_volume_divergence', 'N/A')}")

        # 价格位置
        parts.extend(["", "=== 价格位置 ==="])
        parts.append(f"20日位置: {ind.get('position_20d', 'N/A')}")
        parts.append(f"60日位置: {ind.get('position_60d', 'N/A')}")
        parts.append(f"20日高点: {ind.get('high_20d', 'N/A')}")
        parts.append(f"20日低点: {ind.get('low_20d', 'N/A')}")
        parts.append(f"60日高点: {ind.get('high_60d', 'N/A')}")
        parts.append(f"60日低点: {ind.get('low_60d', 'N/A')}")

        # ── 新增：压力测试场景 ──
        stress = self._run_stress_tests(snapshot)
        parts.extend([
            "",
            "=== 压力测试情景（多情景分析） ===",
            f"- 基准情景（概率 {stress.base.probability:.0%}）: 预期收益 {stress.base.expected_return:.1f}%, 最大回撤 {stress.base.max_drawdown:.1f}%",
            f"- 乐观情景（概率 {stress.bull.probability:.0%}）: 预期收益 {stress.bull.expected_return:.1f}%, 最大回撤 {stress.bull.max_drawdown:.1f}% | 触发: {stress.bull.trigger}",
            f"- 悲观情景（概率 {stress.bear.probability:.0%}）: 预期收益 {stress.bear.expected_return:.1f}%, 最大回撤 {stress.bear.max_drawdown:.1f}% | 触发: {stress.bear.trigger}",
            f"- 黑天鹅情景（概率 {stress.black_swan.probability:.0%}）: 预期收益 {stress.black_swan.expected_return:.1f}%, 最大回撤 {stress.black_swan.max_drawdown:.1f}% | 触发: {stress.black_swan.trigger}",
            "",
            "=== 尾部风险指标 ===",
            f"- VaR(95%): {stress.var_95:.2f}% （单日最大损失，{stress.var_method}模拟）",
            f"- CVaR(95%): {stress.cvar_95:.2f}% （极端情况平均损失）",
            f"- 历史最大连续亏损天数: {stress.max_consecutive_losses} 天",
            f"- 历史爆仓概率(>{abs(self.BLOWUP_THRESHOLD):.0f}%回撤): {stress.blowup_probability:.2f}%",
            "",
            "=== 跳空风险统计 ===",
        ])
        gap = stress.gap_risk
        parts.extend([
            f"- 向上跳空次数(120日): {gap.get('up_gaps', 'N/A')}, 平均幅度: {gap.get('avg_up_gap_pct', 'N/A')}%",
            f"- 向下跳空次数(120日): {gap.get('down_gaps', 'N/A')}, 平均幅度: {gap.get('avg_down_gap_pct', 'N/A')}%",
            f"- 最大单日向下跳空: {gap.get('max_down_gap_pct', 'N/A')}%",
            "",
            "=== 流动性风险指标 ===",
        ])
        liq = stress.liquidity_risk
        parts.extend([
            f"- Amihud非流动性指标: {liq.get('amihud_illiquidity', 'N/A')} (×10⁶)",
            f"- 成交额5日波动率: {liq.get('amount_volatility_5d', 'N/A')}%",
            f"- 近5日均成交额: {liq.get('avg_amount_5d', 'N/A')} 亿",
            f"- 成交额20日趋势: {liq.get('amount_trend_20d', 'N/A')}",
            "",
            "=== 波动率期限结构 ===",
        ])
        vts = stress.volatility_term_structure
        parts.extend([
            f"- 5日年化波动率: {vts.get('vol_5d', 'N/A')}%",
            f"- 20日年化波动率: {vts.get('vol_20d', 'N/A')}%",
            f"- 60日年化波动率: {vts.get('vol_60d', 'N/A')}%",
            f"- 波动率趋势(5d/20d): {vts.get('vol_trend_ratio', 'N/A')} ({vts.get('vol_trend_desc', 'N/A')})",
        ])

        # ── 新增：组合风险分析 ──
        portfolio = self._analyze_portfolio_risk(snapshot, user_position)
        if user_position and portfolio.existing_position_count > 0:
            parts.extend([
                "",
                "=== 组合风险分析（新增该股票对整体组合的影响） ===",
                f"- 现有持仓数量: {portfolio.existing_position_count} 只",
                f"- 新增后组合Beta: {portfolio.new_beta:.2f} (当前 {portfolio.current_beta:.2f}, 变动 {portfolio.beta_change_pct:+.1f}%) [{portfolio.beta_status}]",
                f"- 行业集中度(HHI): {portfolio.industry_hhi:.0f} (最大行业占比 {portfolio.max_industry_pct:.0f}%) [{portfolio.concentration_risk}]",
                f"- 行业重叠度: {portfolio.industry_overlap}",
                f"- 新增后组合年化波动率: {portfolio.new_volatility:.1f}% (当前 {portfolio.current_volatility:.1f}%)",
                f"- 组合VaR(95%): {portfolio.portfolio_var_95:.2f}%",
                f"- 边际风险贡献: {portfolio.marginal_risk_contribution:.2f}% (占组合总风险 {portfolio.risk_contribution_pct:.1f}%)",
                f"- 估算组合最大回撤: {portfolio.estimated_max_drawdown:.1f}%",
                f"- 新增股票权重: {portfolio.new_weight_pct:.1f}%",
                f"- 组合层面仓位约束: {portfolio.position_constraint} (建议上限 {portfolio.recommended_max_position:.0%})",
            ])
        elif user_position:
            parts.extend([
                "",
                "=== 组合风险分析 ===",
                "- 用户暂无其他持仓，该股票将成为首只持仓",
                f"- 建议首仓仓位不超过 {portfolio.recommended_max_position:.0%}，作为后续对比基准",
            ])

        # ── 新增：动态止损策略 ──
        dsl = self._calculate_dynamic_stop_loss(snapshot)
        parts.extend([
            "",
            "=== 动态止损策略（多策略对比） ===",
            f"- 【推荐】{dsl.recommended_strategy}: 止损价 {dsl.recommended_stop_loss:.2f} 元",
            f"  推荐理由: {dsl.recommendation_reason}",
            "",
            "- 波动率自适应止损:",
            f"  止损价 {dsl.volatility_adaptive.stop_price:.2f} (幅度 {dsl.volatility_adaptive.stop_pct:.1f}%) | {dsl.volatility_adaptive.description}",
            f"  适用: {dsl.volatility_adaptive.suitable_for}",
            "",
            "- ATR 止损 (多档):",
            f"  1x ATR: {dsl.atr_based_1x.stop_price:.2f} (幅度 {dsl.atr_based_1x.stop_pct:.1f}%)",
            f"  2x ATR: {dsl.atr_based_2x.stop_price:.2f} (幅度 {dsl.atr_based_2x.stop_pct:.1f}%) — 默认推荐",
            f"  3x ATR: {dsl.atr_based_3x.stop_price:.2f} (幅度 {dsl.atr_based_3x.stop_pct:.1f}%)",
            "",
            "- 移动止损:",
            f"  初始止损: {dsl.trailing.stop_price:.2f} | {dsl.trailing.description}",
            f"  上移规则: {dsl.trailing_rule}",
            "",
            "- 时间止损:",
            f"  止损价 {dsl.time_based.stop_price:.2f} | {dsl.time_based.description}",
            f"  规则: {dsl.time_stop_rule}",
            "",
            "- 技术位止损:",
            f"  前期低点: {dsl.technical_low.stop_price:.2f} (幅度 {dsl.technical_low.stop_pct:.1f}%)",
            f"  支撑位:   {dsl.technical_support.stop_price:.2f} (幅度 {dsl.technical_support.stop_pct:.1f}%)",
            f"  布林带下轨: {dsl.technical_bollinger.stop_price:.2f} (幅度 {dsl.technical_bollinger.stop_pct:.1f}%)",
        ])

        # 未来事件（模拟）
        parts.extend(["", "=== 未来30日潜在事件 ==="])
        parts.append("(模拟数据) 下一财报披露日: 约15日后")
        parts.append("请结合压力测试情景、组合风险分析与动态止损策略评估事件风险对仓位的实际影响。")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════════════
    #  压力测试引擎
    # ═══════════════════════════════════════════════════════════════
    def _run_stress_tests(self, snapshot: StockSnapshot) -> StressTestResult:
        """
        运行多情景压力测试

        四情景架构：
        - 基准：当前风险指标延续
        - 乐观：市场上涨 20%，个股弹性 = Beta
        - 悲观：市场下跌 20%，个股跌幅放大 1.5x
        - 黑天鹅：市场下跌 40%，个股跌幅放大 2.5x，系统性危机
        """
        kline = snapshot.kline_df
        risk = snapshot.risk_metrics

        vol = risk.get("annual_volatility", 20.0)
        beta = risk.get("beta", 1.0)
        base_dd = risk.get("max_drawdown", -15.0)

        # ── 情景计算 ──
        base = StressScenario(
            probability=0.50,
            expected_return=risk.get("annual_return", 0.0),
            max_drawdown=base_dd,
            trigger="当前风险指标延续",
        )

        bull = StressScenario(
            probability=0.25,
            expected_return=round(self.STRESS_BULL_MARKET_MOVE * beta, 1),
            max_drawdown=round(max(base_dd * 0.5, -20.0), 1),
            trigger=f"市场上涨{self.STRESS_BULL_MARKET_MOVE:.0f}%，个股Beta={beta:.2f}",
        )

        bear_dd = round(min(base_dd * 1.5, -30.0), 1)
        bear = StressScenario(
            probability=0.20,
            expected_return=round(self.STRESS_BEAR_MARKET_MOVE * beta, 1),
            max_drawdown=max(bear_dd, self.MAX_DRAWDOWN_CAP),
            trigger=f"市场下跌{abs(self.STRESS_BEAR_MARKET_MOVE):.0f}%，个股跌幅放大1.5x",
        )

        black_dd = round(min(base_dd * 2.5, -60.0), 1)
        black = StressScenario(
            probability=0.05,
            expected_return=round(self.STRESS_BLACK_SWAN_MOVE * beta, 1),
            max_drawdown=max(black_dd, self.MAX_DRAWDOWN_CAP),
            trigger="系统性金融危机/黑天鹅事件",
        )

        # ── 尾部风险计算 ──
        var_95, cvar_95, var_method = self._calculate_var_cvar(kline, vol)

        # ── 波动率期限结构 ──
        vts = self._calculate_volatility_term_structure(kline)

        # ── 跳空风险 ──
        gap_risk = self._analyze_gap_risk(kline)

        # ── 流动性风险 ──
        liquidity = self._analyze_liquidity_risk(kline, risk)

        return StressTestResult(
            base=base,
            bull=bull,
            bear=bear,
            black_swan=black,
            var_95=var_95,
            cvar_95=cvar_95,
            var_method=var_method,
            max_consecutive_losses=self._count_max_consecutive_losses(kline),
            blowup_probability=self._estimate_blowup_probability(kline, vol),
            gap_risk=gap_risk,
            liquidity_risk=liquidity,
            volatility_term_structure=vts,
        )

    def _calculate_var_cvar(self, kline: Optional[pd.DataFrame], annual_vol: float) -> Tuple[float, float, str]:
        """
        计算 VaR(95%) 与 CVaR(95%)

        策略：
        1. 优先历史模拟法（基于真实收益率分位数，无分布假设）
        2. 数据不足时回退正态法
        """
        if kline is not None and len(kline) >= 30 and "close" in kline.columns:
            close = kline["close"].values
            returns = np.diff(close) / close[:-1]
            if len(returns) >= 30:
                var_95 = round(np.percentile(returns, 5) * 100, 2)
                cvar_95 = round(np.mean(returns[returns <= np.percentile(returns, 5)]) * 100, 2)
                return var_95, cvar_95, "历史模拟"

        # 回退：正态近似（年化转日度：sqrt(252) ≈ 15.87，此处用 16）
        daily_vol = annual_vol / 16.0
        var_95 = round(-1.645 * daily_vol, 2)
        cvar_95 = round(-2.063 * daily_vol, 2)  # 正态分布尾部条件期望系数
        return var_95, cvar_95, "正态近似"

    def _count_max_consecutive_losses(self, kline: Optional[pd.DataFrame]) -> int:
        """统计历史最大连续亏损天数"""
        if kline is None or len(kline) < 2 or "close" not in kline.columns:
            return 0
        close = kline["close"].values
        returns = np.diff(close) / close[:-1]
        max_streak = current = 0
        for r in returns:
            if r < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return int(max_streak)

    def _estimate_blowup_probability(self, kline: Optional[pd.DataFrame], annual_vol: float) -> float:
        """
        估算历史爆仓概率

        定义：在 120 日窗口内，从任意高点回落超过 BLOWUP_THRESHOLD 的概率。
        简化模型：基于正态假设，计算日收益率突破阈值的概率，
        再用历史实际回撤频率校正。
        """
        if kline is not None and len(kline) >= 30 and "close" in kline.columns:
            close = kline["close"].values
            cummax = np.maximum.accumulate(close)
            drawdowns = (close - cummax) / cummax
            blowup_days = np.sum(drawdowns <= (self.BLOWUP_THRESHOLD / 100.0))
            prob = blowup_days / len(close) * 100.0
        else:
            # 无数据时用正态尾部概率近似
            # P(return < -50%) 在年化波动率 vol 下的日度概率
            daily_vol = annual_vol / 16.0
            # 假设 120 个交易日，累积跌幅阈值对应日均跌幅
            daily_threshold = abs(self.BLOWUP_THRESHOLD) / 120.0
            z = daily_threshold / max(daily_vol, 0.01)
            from math import erfc, sqrt
            prob = (0.5 * erfc(z / sqrt(2))) * 100.0 * 120.0

        return round(min(prob, 100.0), 2)

    def _analyze_gap_risk(self, kline: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        分析跳空风险

        统计：
        - 向上/向下跳空次数
        - 平均跳空幅度
        - 最大单日向下跳空
        """
        if kline is None or len(kline) < 5 or "open" not in kline.columns or "close" not in kline.columns:
            return {"up_gaps": 0, "down_gaps": 0, "avg_up_gap_pct": 0.0, "avg_down_gap_pct": 0.0, "max_down_gap_pct": 0.0}

        open_p = kline["open"].values
        close = kline["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = open_p[0]  # 首日无跳空

        gaps = (open_p - prev_close) / prev_close * 100.0
        up_gaps = gaps[gaps > 1.0]    # >1% 视为向上跳空
        down_gaps = gaps[gaps < -1.0]  # <-1% 视为向下跳空

        return {
            "up_gaps": int(len(up_gaps)),
            "down_gaps": int(len(down_gaps)),
            "avg_up_gap_pct": round(float(np.mean(up_gaps)), 2) if len(up_gaps) > 0 else 0.0,
            "avg_down_gap_pct": round(float(np.mean(down_gaps)), 2) if len(down_gaps) > 0 else 0.0,
            "max_down_gap_pct": round(float(np.min(down_gaps)), 2) if len(down_gaps) > 0 else 0.0,
        }

    def _analyze_liquidity_risk(self, kline: Optional[pd.DataFrame], risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        流动性风险分析

        指标：
        - Amihud 非流动性 = mean(|return| / amount) × 10⁶
        - 成交额 5 日波动率
        - 成交额 20 日趋势（线性回归斜率）
        """
        if kline is None or len(kline) < 5 or "close" not in kline.columns or "amount" not in kline.columns:
            avg_amount = risk_metrics.get("avg_amount_5d", 0.0)
            return {
                "amihud_illiquidity": 0.0,
                "amount_volatility_5d": 0.0,
                "avg_amount_5d": avg_amount,
                "amount_trend_20d": "数据不足",
            }

        close = kline["close"].values
        amount = kline["amount"].values
        returns = np.diff(close) / close[:-1]

        # Amihud 非流动性（基于最近 20 日）
        window = min(20, len(returns))
        abs_ret = np.abs(returns[-window:])
        amt = amount[-window:]
        # 避免除零
        amihud = np.mean(abs_ret / (amt + 1e-9)) * 1e6 if np.any(amt > 0) else 0.0

        # 成交额 5 日波动率（相对标准差）
        amt_5d = amount[-5:]
        amount_cv = (np.std(amt_5d) / (np.mean(amt_5d) + 1e-9)) * 100.0 if np.mean(amt_5d) > 0 else 0.0

        # 成交额 20 日趋势
        amt_20d = amount[-20:] if len(amount) >= 20 else amount
        if len(amt_20d) >= 5:
            x = np.arange(len(amt_20d))
            slope = np.polyfit(x, amt_20d, 1)[0]
            trend_desc = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
        else:
            trend_desc = "数据不足"

        avg_amount_5d = risk_metrics.get("avg_amount_5d", round(float(np.mean(amt_5d)) / 1e8, 2))

        return {
            "amihud_illiquidity": round(float(amihud), 4),
            "amount_volatility_5d": round(float(amount_cv), 2),
            "avg_amount_5d": avg_amount_5d,
            "amount_trend_20d": trend_desc,
        }

    def _calculate_volatility_term_structure(self, kline: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        波动率期限结构

        计算 5日 / 20日 / 60日 年化波动率，并判断趋势。
        """
        if kline is None or len(kline) < 10 or "close" not in kline.columns:
            return {"vol_5d": 0.0, "vol_20d": 0.0, "vol_60d": 0.0, "vol_trend_ratio": 1.0, "vol_trend_desc": "数据不足"}

        close = kline["close"].values
        returns = np.diff(close) / close[:-1]

        def _rolling_vol(ret: np.ndarray, window: int) -> float:
            if len(ret) < window:
                return 0.0
            return round(float(np.std(ret[-window:]) * np.sqrt(252) * 100), 2)

        vol_5d = _rolling_vol(returns, 5)
        vol_20d = _rolling_vol(returns, 20)
        vol_60d = _rolling_vol(returns, 60)

        ratio = vol_5d / (vol_20d + 1e-6)
        if ratio > 1.3:
            desc = "短期波动率飙升（风险积聚）"
        elif ratio < 0.7:
            desc = "短期波动率回落（趋于平静）"
        else:
            desc = "波动率期限结构平稳"

        return {
            "vol_5d": vol_5d,
            "vol_20d": vol_20d,
            "vol_60d": vol_60d,
            "vol_trend_ratio": round(ratio, 2),
            "vol_trend_desc": desc,
        }

    # ═══════════════════════════════════════════════════════════════
    #  组合风险分析引擎
    # ═══════════════════════════════════════════════════════════════
    def _analyze_portfolio_risk(self, snapshot: StockSnapshot, user_position: Optional[Dict]) -> PortfolioRiskResult:
        """
        分析新增该股票对整体组合的风险影响

        核心指标：
        1. 组合 Beta（当前 vs 新增后）
        2. 行业集中度（HHI + 最大行业占比）
        3. 组合波动率（简化协方差矩阵）
        4. 组合 VaR(95%)
        5. 边际风险贡献
        6. 估算组合最大回撤
        7. 仓位约束（基于组合风险）
        """
        if not user_position:
            return PortfolioRiskResult()

        existing = user_position.get("positions", [])
        if not existing:
            # 首仓场景
            planned = user_position.get("planned_investment", 0)
            total = user_position.get("total_portfolio_value", planned)
            new_weight = (planned / total * 100) if total > 0 else 0.0
            return PortfolioRiskResult(
                new_weight_pct=round(new_weight, 1),
                recommended_max_position=0.10,
                position_constraint="首仓建议",
            )

        # ── 解析现有持仓 ──
        positions: List[PortfolioPosition] = []
        total_value = 0.0
        for p in existing:
            pos = PortfolioPosition(
                code=p.get("code", ""),
                name=p.get("name", ""),
                value=float(p.get("value", 0)),
                weight=0.0,  # 稍后计算
                beta=float(p.get("beta", 1.0)),
                volatility=float(p.get("volatility", 20.0)),
                industry=p.get("industry", "未知"),
                max_drawdown=float(p.get("max_drawdown", -15.0)),
            )
            positions.append(pos)
            total_value += pos.value

        # 新增股票
        planned_investment = float(user_position.get("planned_investment", 0))
        new_total = total_value + planned_investment
        if new_total <= 0:
            return PortfolioRiskResult()

        # 计算权重
        for pos in positions:
            pos.weight = pos.value / new_total
        new_weight = planned_investment / new_total

        # 新增股票的指标
        new_beta = snapshot.risk_metrics.get("beta", 1.0)
        new_vol = snapshot.risk_metrics.get("annual_volatility", 20.0)
        new_industry = snapshot.fundamentals.get("industry", "未知")
        new_dd = snapshot.risk_metrics.get("max_drawdown", -15.0)

        # ── 1. 组合 Beta ──
        current_beta = sum(p.value / total_value * p.beta for p in positions) if total_value > 0 else 1.0
        new_portfolio_beta = sum(p.weight * p.beta for p in positions) + new_weight * new_beta
        beta_change_pct = ((new_portfolio_beta - current_beta) / max(abs(current_beta), 0.01)) * 100.0

        if new_portfolio_beta > 1.5:
            beta_status = "超标"
        elif new_portfolio_beta > 1.2:
            beta_status = "偏高"
        else:
            beta_status = "正常"

        # ── 2. 行业集中度 ──
        industry_values: Dict[str, float] = {}
        for p in positions:
            industry_values[p.industry] = industry_values.get(p.industry, 0.0) + p.value
        # 加入新增股票
        industry_values[new_industry] = industry_values.get(new_industry, 0.0) + planned_investment

        max_industry_pct = max(industry_values.values()) / new_total * 100.0
        # HHI = sum((weight_i)^2) * 10000
        industry_weights = [v / new_total for v in industry_values.values()]
        hhi = sum(w ** 2 for w in industry_weights) * 10000.0

        # 行业重叠度
        existing_industries = {p.industry for p in positions}
        if new_industry in existing_industries:
            overlap_ratio = industry_values.get(new_industry, 0) / new_total
            if overlap_ratio > 0.5:
                industry_overlap = "高度重叠"
            else:
                industry_overlap = "部分重叠"
        else:
            industry_overlap = "无"

        # 投资组合语境下 HHI 阈值适当放宽（vs 反垄断标准）
        if hhi > 3500:
            concentration_risk = "高"
        elif hhi > 2200:
            concentration_risk = "中"
        else:
            concentration_risk = "低"

        # ── 3. 组合波动率（简化协方差矩阵） ──
        # 构建所有资产（现有 + 新增）的参数数组
        all_weights = [p.weight for p in positions] + [new_weight]
        all_betas = [p.beta for p in positions] + [new_beta]
        all_vols = [p.volatility for p in positions] + [new_vol]
        all_industries = [p.industry for p in positions] + [new_industry]
        n = len(all_weights)

        # 市场波动率代理（使用指数典型值或已有数据的平均）
        market_vol = 20.0  # 沪深300典型年化波动率

        # 构建协方差矩阵：Σ_ij = β_i * β_j * σ_m² + σ_idio_i * σ_idio_j * ρ_idio_ij
        # 简化：总波动率分解为系统性 + 非系统性
        # σ_idio² = σ_total² - β² * σ_m²
        idio_vols = []
        for i in range(n):
            idio_var = max(0.0, (all_vols[i] ** 2) - (all_betas[i] ** 2) * (market_vol ** 2))
            idio_vols.append(np.sqrt(idio_var))

        # 相关性矩阵
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if all_industries[i] == all_industries[j]:
                    # 同行业：高相关性
                    corr = 0.7 + 0.3 * min(all_betas[i], all_betas[j]) / max(all_betas[i], all_betas[j], 0.01)
                else:
                    # 不同行业：通过 Beta 传导的市场相关性
                    corr = 0.3 * min(all_betas[i], all_betas[j]) / max(all_betas[i], all_betas[j], 0.01)
                corr_matrix[i, j] = corr_matrix[j, i] = min(corr, 0.99)

        # 协方差矩阵
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = corr_matrix[i, j] * all_vols[i] * all_vols[j]

        # 组合波动率
        w = np.array(all_weights)
        portfolio_var = float(w @ cov_matrix @ w)
        new_portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0

        # 当前组合波动率（不含新增）
        if len(positions) > 0:
            w_curr = np.array([p.value / total_value for p in positions])
            cov_curr = cov_matrix[:len(positions), :len(positions)]
            curr_var = float(w_curr @ cov_curr @ w_curr)
            current_portfolio_vol = np.sqrt(curr_var) if curr_var > 0 else 0.0
        else:
            current_portfolio_vol = 0.0

        # ── 4. 组合 VaR(95%) ──
        # 日度 VaR = 1.645 * portfolio_vol / sqrt(252)
        portfolio_var_95 = round(-1.645 * new_portfolio_vol / np.sqrt(252), 2)

        # ── 5. 边际风险贡献 ──
        # MRC_i = (Σw)_i / portfolio_vol
        sigma_w = cov_matrix @ w
        mrc_new = float(sigma_w[-1]) / max(new_portfolio_vol, 0.01)
        risk_contribution_pct = (w[-1] * mrc_new / max(new_portfolio_vol, 0.01)) * 100.0

        # ── 6. 估算组合最大回撤 ──
        # 简化：基于历史最大回撤和压力测试的加权
        all_dds = [p.max_drawdown for p in positions] + [new_dd]
        # 组合回撤近似：考虑相关性后的加权平均
        dd_contrib = sum(w[i] * all_dds[i] for i in range(n))
        # 分散化效益：持仓越多，回撤越接近 Beta 驱动的市场回撤
        diversification_factor = min(1.0, 0.5 + 0.5 / np.sqrt(n))
        estimated_dd = dd_contrib * diversification_factor

        # ── 7. 仓位约束 ──
        constraint_flags = []
        rec_max_pos = 0.20  # 默认建议

        if new_portfolio_beta > 1.5:
            rec_max_pos = min(rec_max_pos, 0.10)
            constraint_flags.append("组合Beta超标")
        elif new_portfolio_beta > 1.2:
            rec_max_pos = min(rec_max_pos, 0.15)
            constraint_flags.append("组合Beta偏高")

        if hhi > 3500:
            rec_max_pos = min(rec_max_pos, 0.10)
            constraint_flags.append("行业过度集中")
        elif hhi > 2200:
            rec_max_pos = min(rec_max_pos, 0.15)
            constraint_flags.append("行业集中度偏高")

        if industry_overlap == "高度重叠":
            rec_max_pos = min(rec_max_pos, 0.12)
            constraint_flags.append("同行业高度重叠")

        if new_portfolio_vol > 35:
            rec_max_pos = min(rec_max_pos, 0.10)
            constraint_flags.append("组合波动率过高")

        if constraint_flags:
            position_constraint = "限制: " + ", ".join(constraint_flags)
        else:
            position_constraint = "可配置"

        return PortfolioRiskResult(
            current_beta=round(current_beta, 2),
            new_beta=round(new_portfolio_beta, 2),
            beta_change_pct=round(beta_change_pct, 1),
            beta_status=beta_status,
            industry_hhi=round(hhi, 1),
            max_industry_pct=round(max_industry_pct, 1),
            industry_overlap=industry_overlap,
            concentration_risk=concentration_risk,
            current_volatility=round(current_portfolio_vol, 1),
            new_volatility=round(new_portfolio_vol, 1),
            portfolio_var_95=portfolio_var_95,
            marginal_risk_contribution=round(mrc_new, 2),
            risk_contribution_pct=round(risk_contribution_pct, 1),
            estimated_max_drawdown=round(estimated_dd, 1),
            position_constraint=position_constraint,
            recommended_max_position=round(max(0.05, rec_max_pos), 3),
            existing_position_count=len(positions),
            new_weight_pct=round(new_weight * 100, 1),
        )

    # ═══════════════════════════════════════════════════════════════
    #  动态止损引擎
    # ═══════════════════════════════════════════════════════════════
    def _calculate_dynamic_stop_loss(self, snapshot: StockSnapshot) -> "DynamicStopLossResult":
        """
        动态止损策略引擎

        计算多种止损策略并基于市场环境智能推荐：
        1. 波动率自适应：根据年化波动率动态调整止损比例
        2. ATR 多档：1x/2x/3x ATR
        3. 移动止损：趋势行情中锁定利润
        4. 时间止损：避免时间成本
        5. 技术位止损：前期低点、支撑位、布林带下轨
        """
        from agent.models.agent_response import DynamicStopLossResult, StopLossStrategy

        current = snapshot.current_price
        atr = snapshot.indicators.get("atr14", current * 0.03)
        atr_pct = snapshot.indicators.get("atr_pct", 3.0)
        vol = snapshot.risk_metrics.get("annual_volatility", 20.0)
        beta = snapshot.risk_metrics.get("beta", 1.0)
        low_20d = snapshot.indicators.get("low_20d", current * 0.90)
        low_60d = snapshot.indicators.get("low_60d", current * 0.85)
        boll_lower = snapshot.indicators.get("boll_lower", current * 0.92)
        ma_alignment = snapshot.indicators.get("ma_alignment", "")
        trend_direction = snapshot.indicators.get("trend_direction", "")

        # ── 1. 波动率自适应止损 ──
        # 逻辑：vol < 15% → -5%, 15-30% → -7%, 30-45% → -10%, >45% → -12%
        if vol < 15:
            vol_stop_pct = 5.0
        elif vol < 30:
            vol_stop_pct = 7.0
        elif vol < 45:
            vol_stop_pct = 10.0
        else:
            vol_stop_pct = 12.0
        vol_stop = round(current * (1 - vol_stop_pct / 100), 2)
        volatility_adaptive = StopLossStrategy(
            stop_price=vol_stop,
            stop_pct=-vol_stop_pct,
            strategy_type="volatility_adaptive",
            description=f"根据年化波动率 {vol:.1f}% 自适应调整止损比例 (-{vol_stop_pct:.0f}%)",
            pros="自动适应不同波动率环境，低波动收紧、高波动放宽",
            cons="对突发跳空保护不足",
            suitable_for="所有市场环境，尤其适合波动率变化剧烈的股票",
        )

        # ── 2. ATR 多档止损 ──
        atr1_stop = round(current - atr, 2)
        atr1_pct = round(-atr / current * 100, 1)
        atr2_stop = round(current - 2 * atr, 2)
        atr2_pct = round(-2 * atr / current * 100, 1)
        atr3_stop = round(current - 3 * atr, 2)
        atr3_pct = round(-3 * atr / current * 100, 1)

        atr_1x = StopLossStrategy(
            stop_price=atr1_stop,
            stop_pct=atr1_pct,
            strategy_type="atr_1x",
            description="1x ATR 止损，紧跟随",
            pros="反应灵敏，快速截断亏损",
            cons="震荡市容易被洗出",
            suitable_for="低波动、趋势明确的股票",
        )
        atr_2x = StopLossStrategy(
            stop_price=atr2_stop,
            stop_pct=atr2_pct,
            strategy_type="atr_2x",
            description="2x ATR 止损，平衡型",
            pros="兼顾灵敏度与抗噪能力",
            cons="极端行情下止损偏慢",
            suitable_for="大多数股票，默认推荐",
        )
        atr_3x = StopLossStrategy(
            stop_price=atr3_stop,
            stop_pct=atr3_pct,
            strategy_type="atr_3x",
            description="3x ATR 止损，宽松型",
            pros="抗震荡能力强，适合高波动股票",
            cons="单笔亏损幅度较大",
            suitable_for="高波动、高Beta的股票",
        )

        # ── 3. 移动止损 ──
        trailing_initial = atr2_stop
        trailing_pct = atr2_pct
        trailing_rule = (
            f"初始止损 {trailing_initial:.2f} 元；"
            f"股价每上涨 1x ATR (+{atr:.2f}元)，止损上移 0.5x ATR (+{atr*0.5:.2f}元)；"
            f"止损永不下移。"
        )
        trailing = StopLossStrategy(
            stop_price=trailing_initial,
            stop_pct=trailing_pct,
            strategy_type="trailing",
            description="上涨后自动上移止损，锁定利润",
            pros="让利润奔跑，同时保护已得收益",
            cons="震荡市容易被洗出，需趋势配合",
            suitable_for="多头排列、趋势向上的股票",
        )

        # ── 4. 时间止损 ──
        # 基于历史胜率和波动率计算期望持有时间
        win_rate = snapshot.risk_metrics.get("win_rate", 50.0)
        if win_rate > 55:
            time_window = 15  # 高胜率，给15天
            time_stop_pct = 5.0
        elif win_rate > 45:
            time_window = 10
            time_stop_pct = 7.0
        else:
            time_window = 7
            time_stop_pct = 10.0
        time_stop = round(current * (1 - time_stop_pct / 100), 2)
        time_rule = f"若 {time_window} 个交易日内未上涨 {time_stop_pct:.0f}% 以上，无论盈亏均触发止损"
        time_based = StopLossStrategy(
            stop_price=time_stop,
            stop_pct=-time_stop_pct,
            strategy_type="time_based",
            description=f"{time_window} 日内未达预期即止损，避免时间成本",
            pros="强制资金轮动，避免死仓",
            cons="可能错过慢牛启动",
            suitable_for="高轮动策略、短线交易",
        )

        # ── 5. 技术位止损 ──
        # 前期低点
        tech_low_stop = round(low_20d * 0.98, 2)  # 前期低点下方 2%
        tech_low_pct = round((tech_low_stop - current) / current * 100, 1)
        technical_low = StopLossStrategy(
            stop_price=tech_low_stop,
            stop_pct=tech_low_pct,
            strategy_type="technical_low",
            description=f"20日低点 {low_20d:.2f} 下方 2%",
            pros="符合技术派交易习惯，支撑明确",
            cons="若支撑位失效，亏损已较大",
            suitable_for="技术分析为主的交易者",
        )

        # 支撑位（简化：使用 60 日低点或布林带下轨的较高者）
        support_level = max(low_60d, boll_lower)
        tech_support_stop = round(support_level * 0.97, 2)
        tech_support_pct = round((tech_support_stop - current) / current * 100, 1)
        technical_support = StopLossStrategy(
            stop_price=tech_support_stop,
            stop_pct=tech_support_pct,
            strategy_type="technical_support",
            description=f"关键支撑位 {support_level:.2f} 下方 3%",
            pros="利用市场共识支撑，心理关口有效",
            cons="假突破频繁，需结合成交量确认",
            suitable_for="有明显支撑压力位的股票",
        )

        # 布林带下轨
        tech_boll_stop = round(boll_lower * 0.99, 2)
        tech_boll_pct = round((tech_boll_stop - current) / current * 100, 1)
        technical_bollinger = StopLossStrategy(
            stop_price=tech_boll_stop,
            stop_pct=tech_boll_pct,
            strategy_type="technical_bollinger",
            description=f"布林带下轨 {boll_lower:.2f} 下方 1%",
            pros="统计学支撑，超跌反弹概率高",
            cons="趋势行情中下轨会不断下移",
            suitable_for="震荡市、均值回归策略",
        )

        # ── 智能推荐 ──
        recommended = "atr_based_2x"
        reason = "默认推荐 2x ATR 止损，兼顾灵敏度与抗噪能力"

        # 高波动环境 (>35%) → 3x ATR 或 波动率自适应
        if vol > 35:
            recommended = "atr_based_3x"
            reason = f"高波动环境(年化{vol:.0f}%)，3x ATR 可避免日常噪音洗出"
        # 低波动环境 (<15%) + 高胜率 → 1x ATR
        elif vol < 15 and snapshot.risk_metrics.get("win_rate", 50) > 55:
            recommended = "atr_based_1x"
            reason = "低波动高胜率环境，1x ATR 可快速截断亏损"
        # 明确多头趋势 → 移动止损
        elif ma_alignment == "多头排列" or trend_direction == "上升趋势":
            recommended = "trailing"
            reason = "多头排列/上升趋势，移动止损可让利润奔跑"
        # 震荡市 + 布林带有效 → 技术位止损
        elif vol < 25 and abs(tech_boll_pct) < 12:
            recommended = "technical_bollinger"
            reason = "震荡市环境，布林带下轨提供统计学支撑"
        # Beta 高 (>1.3) → 波动率自适应
        elif beta > 1.3:
            recommended = "volatility_adaptive"
            reason = f"高Beta({beta:.2f})股票波动大，波动率自适应止损更稳健"

        rec_stop = {
            "volatility_adaptive": volatility_adaptive.stop_price,
            "atr_based_1x": atr_1x.stop_price,
            "atr_based_2x": atr_2x.stop_price,
            "atr_based_3x": atr_3x.stop_price,
            "trailing": trailing.stop_price,
            "time_based": time_based.stop_price,
            "technical_support": technical_support.stop_price,
            "technical_bollinger": technical_bollinger.stop_price,
            "technical_low": technical_low.stop_price,
        }.get(recommended, atr2_stop)

        return DynamicStopLossResult(
            volatility_adaptive=volatility_adaptive,
            atr_based_1x=atr_1x,
            atr_based_2x=atr_2x,
            atr_based_3x=atr_3x,
            trailing=trailing,
            time_based=time_based,
            technical_support=technical_support,
            technical_bollinger=technical_bollinger,
            technical_low=technical_low,
            recommended_strategy=recommended,
            recommended_stop_loss=rec_stop,
            recommendation_reason=reason,
            trailing_rule=trailing_rule,
            time_stop_rule=time_rule,
        )

    # ═══════════════════════════════════════════════════════════════
    #  降级逻辑
    # ═══════════════════════════════════════════════════════════════
    def _fallback_opinion(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """
        降级分析 —— LLM 调用失败时的规则引擎

        v2.0 增强：
        1. 基于压力测试结果动态调整风险等级
        2. 考虑 VaR、连续亏损天数、跳空风险
        3. 更精细的仓位公式
        """
        risk = snapshot.risk_metrics
        vol = risk.get("annual_volatility", 20.0)
        drawdown = risk.get("max_drawdown", -15.0)

        # 运行压力测试（纯本地计算，不依赖 LLM）
        try:
            stress = self._run_stress_tests(snapshot)
        except Exception as e:
            logger.warning(f"压力测试降级计算失败: {e}")
            stress = StressTestResult()

        # ── 风险等级计算 ──
        # 基础分（波动率 + 回撤）
        risk_level = 3
        if vol > 40 or drawdown < -30:
            risk_level = 5
        elif vol > 30 or drawdown < -20:
            risk_level = 4
        elif vol < 15 and drawdown > -10:
            risk_level = 2
        elif vol < 10 and drawdown > -5:
            risk_level = 1

        # 压力测试校正
        if stress.black_swan.max_drawdown < -60:
            risk_level = min(5, risk_level + 1)
        if stress.cvar_95 < -5.0:  # 极端日均损失 >5%
            risk_level = min(5, risk_level + 1)
        if stress.max_consecutive_losses >= 7:
            risk_level = min(5, risk_level + 1)
        if stress.blowup_probability > 5.0:
            risk_level = min(5, risk_level + 1)

        # 事件风险兜底
        risk_flags = ["LLM调用异常，使用规则引擎（含压力测试）"]
        if stress.max_consecutive_losses >= 5:
            risk_flags.append(f"历史最大连续亏损{stress.max_consecutive_losses}天，注意止损纪律")
        if stress.gap_risk.get("down_gaps", 0) >= 3:
            risk_flags.append(f"近120日出现{stress.gap_risk['down_gaps']}次向下跳空，隔夜风险显著")

        # ── 仓位计算 ──
        # 基础仓位 = 20% - (risk_level - 1) * 3%
        base_pos = max(0.05, min(0.50, 0.20 - (risk_level - 1) * 0.03))
        # CVaR 校正：极端损失越大，仓位越低
        cvar_adj = 1.0
        if stress.cvar_95 < -3.0:
            cvar_adj = max(0.3, 1.0 + (stress.cvar_95 + 1.0) / 5.0)  # 例如 cvar=-5% -> adj=0.6
        # 流动性校正
        liq = stress.liquidity_risk
        liq_adj = 1.0
        if liq.get("amount_volatility_5d", 0) > 50:
            liq_adj = 0.8

        # ── 组合风险约束 ──
        try:
            portfolio = self._analyze_portfolio_risk(snapshot, user_position)
        except Exception as e:
            logger.warning(f"组合风险降级计算失败: {e}")
            portfolio = PortfolioRiskResult()

        # 组合风险校正：若组合层面建议更保守，取更严格值
        portfolio_adj = 1.0
        if portfolio.beta_status == "超标":
            portfolio_adj = 0.6
            risk_flags.append("组合Beta超标，强制压缩仓位")
        elif portfolio.beta_status == "偏高":
            portfolio_adj = 0.8
            risk_flags.append("组合Beta偏高，建议降低仓位")
        if portfolio.concentration_risk == "高":
            portfolio_adj = min(portfolio_adj, 0.6)
            risk_flags.append("行业过度集中，强制分散")
        elif portfolio.concentration_risk == "中":
            portfolio_adj = min(portfolio_adj, 0.8)
            risk_flags.append("行业集中度偏高")

        max_pos = round(base_pos * cvar_adj * liq_adj * portfolio_adj, 3)
        max_pos = max(0.05, min(0.50, max_pos))

        # 若组合风险有更严格的建议上限，取二者较小值
        max_pos = min(max_pos, portfolio.recommended_max_position)

        # ── 动态止损建议 ──
        try:
            dsl = self._calculate_dynamic_stop_loss(snapshot)
        except Exception as e:
            logger.warning(f"动态止损降级计算失败: {e}")
            from agent.models.agent_response import DynamicStopLossResult
            dsl = DynamicStopLossResult()

        stop_loss = dsl.recommended_stop_loss

        # ── 盈亏比 ──
        target_price = snapshot.current_price * (1 + vol / 100.0)
        risk_reward = round((target_price - snapshot.current_price) / max(snapshot.current_price - stop_loss, 0.01), 2)

        raw_data = {
            "risk_level": risk_level,
            "max_position_pct": max_pos,
            "risk_reward_ratio": risk_reward,
            "recommended_stop_loss": stop_loss,
            "black_scenarios": ["业绩不及预期", "市场系统性风险", "流动性危机"],
            "position_sizing_formula": (
                f"基础仓位={base_pos:.0%} × CVaR校正{cvar_adj:.2f} × "
                f"流动性校正{liq_adj:.2f} × 组合校正{portfolio_adj:.2f} = {max_pos:.0%}"
            ),
            "stress_test": stress.model_dump(),
            "portfolio_risk": portfolio.model_dump(),
            "dynamic_stop_loss": dsl.model_dump(),
            "key_factors": [
                f"年化波动率: {vol}%",
                f"最大回撤: {drawdown}%",
                f"VaR(95%): {stress.var_95}%",
                f"CVaR(95%): {stress.cvar_95}%",
                f"最大连续亏损: {stress.max_consecutive_losses}天",
                f"组合Beta: {portfolio.new_beta} ({portfolio.beta_status})",
                f"行业集中度: {portfolio.concentration_risk}",
                f"推荐止损策略: {dsl.recommended_strategy} ({dsl.recommended_stop_loss}元)",
            ],
            "risk_flags": risk_flags,
        }

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=0,
            confidence=0.55,
            reasoning=(
                f"【规则引擎降级】基于压力测试、组合风险与动态止损的风险评估："
                f"风险等级={risk_level}，"
                f"VaR(95%)={stress.var_95}%，"
                f"CVaR(95%)={stress.cvar_95}%，"
                f"组合Beta={portfolio.new_beta}({portfolio.beta_status})，"
                f"行业集中度={portfolio.concentration_risk}，"
                f"推荐止损策略={dsl.recommended_strategy}({dsl.recommended_stop_loss}元)，"
                f"建议最大仓位={max_pos:.0%}。"
            ),
            key_factors=raw_data["key_factors"],
            risk_flags=risk_flags,
            raw_data=raw_data,
        )

    # ═══════════════════════════════════════════════════════════════
    #  默认系统提示词
    # ═══════════════════════════════════════════════════════════════
    def _default_prompt(self) -> str:
        return """# Role
你是一位以"先求不败，再求胜"为信条的资深风控总监。你的职责不是寻找机会，而是识别所有可能导致亏损的情景，并给出严格的仓位和止损纪律。

# Context
你将收到：
1. 该股票近 120 日价格序列
2. 已计算的量化指标：年化波动率、最大回撤、Beta、夏普比率、下行标准差
3. 该股票与大盘的相关性矩阵
4. 近期重大事件风险（财报发布日、解禁日、监管问询等）
5. 若用户已有持仓：当前持仓成本、仓位占比、组合现有 Beta
6. 【v2.0】四情景压力测试结果（基准/乐观/悲观/黑天鹅）
7. 【v2.0】尾部风险指标：VaR(95%)、CVaR(95%)、最大连续亏损天数、爆仓概率
8. 【v2.0】跳空风险统计与流动性风险指标
9. 【v2.0】波动率期限结构（5日/20日/60日）
10. 【v2.1】组合风险分析：组合Beta（当前vs新增后）、行业集中度(HHI)、组合波动率、组合VaR、边际风险贡献、估算组合最大回撤、仓位约束
11. 【v2.2】动态止损策略：波动率自适应、ATR 1x/2x/3x、移动止损、时间止损、技术位止损（前期低点/支撑位/布林带下轨）

# Analysis Framework
1. **波动率风险**：当前波动率是否处于历史高位？5日/20日/60日波动率期限结构是否显示风险积聚？
2. **回撤风险**：基于历史回撤与压力测试悲观情景，在当前价位买入后潜在最大亏损？
3. **尾部风险**：VaR 与 CVaR 暗示的极端损失水平？历史连续亏损天数是否考验资金耐力？
4. **跳空风险**：历史向下跳空次数与幅度，隔夜持仓风险是否可控？
5. **流动性风险**：Amihud 非流动性指标与成交额波动是否支持计划仓位？
6. **组合风险**：
   - 新增后组合 Beta 是否超标（>1.5）或偏高（>1.2）？
   - 行业集中度 HHI 是否过高（>2500 高集中度）？
   - 新增股票与现有持仓的行业重叠度？
   - 边际风险贡献是否过高（>组合总风险的 30%）？
   - 组合层面仓位约束是否限制了个股仓位？
7. **事件风险**：未来 30 日内是否有财报/解禁/股东大会等不确定性事件？
8. **压力测试结论**：在四情景下仓位是否都能存活？黑天鹅情景下是否会爆仓？
9. **组合风险结论**：新增该股票后，组合是否在可承受风险范围内？
10. **止损策略选择**：
    - 高波动(>35%) → 3x ATR 或波动率自适应
    - 低波动(<15%)+高胜率 → 1x ATR
    - 多头排列/上升趋势 → 移动止损
    - 震荡市 → 技术位止损（布林带下轨）
    - 高Beta(>1.3) → 波动率自适应
    - 默认 → 2x ATR

# Output Format (Strict JSON)
```json
{
  "signal": 0,
  "confidence": 0.80,
  "risk_level": 3,
  "max_position_pct": 0.15,
  "recommended_stop_loss": 12.50,
  "risk_reward_ratio": 2.1,
  "reasoning": "当前波动率处于 90 日 70% 分位，ATR 显示日均波幅 4.5%，未来 10 日有季报披露，建议仓位不超过 15%，止损设于前期低点下方 2%。",
  "key_factors": [
    "波动率处于 90 日 70% 分位",
    "未来 10 日财报披露，业绩不确定性高",
    "近 5 日均成交额 2.1 亿，流动性一般"
  ],
  "risk_flags": [
    "下行标准差显著高于上行标准差",
    "当前价位距离 120 日低点仅 8%，下方空间打开后止损盘密集"
  ],
  "black_scenarios": [
    "业绩不及预期导致跌停",
    "行业政策突发收紧"
  ],
  "position_sizing_formula": "凯利公式修正版：f = (2.1*0.55 - 0.45) / 2.1 ≈ 0.14，再乘以 0.8 保守系数 → 11%",
  "stress_test": {
    "base": {"probability": 0.5, "expected_return": 0, "max_drawdown": -15, "trigger": "基准情景"},
    "bull": {"probability": 0.25, "expected_return": 20, "max_drawdown": -5, "trigger": "市场上涨20%"},
    "bear": {"probability": 0.2, "expected_return": -20, "max_drawdown": -30, "trigger": "市场下跌20%"},
    "black_swan": {"probability": 0.05, "expected_return": -40, "max_drawdown": -60, "trigger": "系统性金融危机"},
    "var_95": -2.5,
    "cvar_95": -3.8,
    "var_method": "历史模拟",
    "max_consecutive_losses": 5,
    "blowup_probability": 2.1,
    "gap_risk": {"up_gaps": 2, "down_gaps": 3, "avg_up_gap_pct": 1.5, "avg_down_gap_pct": -2.1, "max_down_gap_pct": -4.5},
    "liquidity_risk": {"amihud_illiquidity": 0.0123, "amount_volatility_5d": 25.5, "avg_amount_5d": 2.1, "amount_trend_20d": "下降"},
    "volatility_term_structure": {"vol_5d": 28.5, "vol_20d": 22.1, "vol_60d": 20.3, "vol_trend_ratio": 1.29, "vol_trend_desc": "短期波动率飙升（风险积聚）"}
  },
  "portfolio_risk": {
    "current_beta": 1.0,
    "new_beta": 1.15,
    "beta_change_pct": 15.0,
    "beta_status": "正常",
    "industry_hhi": 1200.0,
    "max_industry_pct": 25.0,
    "industry_overlap": "部分重叠",
    "concentration_risk": "低",
    "current_volatility": 18.0,
    "new_volatility": 19.5,
    "portfolio_var_95": -2.0,
    "marginal_risk_contribution": 3.2,
    "risk_contribution_pct": 18.5,
    "estimated_max_drawdown": -14.0,
    "position_constraint": "可配置",
    "recommended_max_position": 0.15,
    "existing_position_count": 3,
    "new_weight_pct": 12.5
  },
  "dynamic_stop_loss": {
    "volatility_adaptive": {"stop_price": 13.95, "stop_pct": -7.0, "strategy_type": "volatility_adaptive", "description": "根据年化波动率 25% 自适应调整止损比例 (-7%)", "pros": "自动适应不同波动率环境", "cons": "对突发跳空保护不足", "suitable_for": "所有市场环境"},
    "atr_based_1x": {"stop_price": 14.35, "stop_pct": -4.3, "strategy_type": "atr_1x", "description": "1x ATR 止损，紧跟随", "pros": "反应灵敏", "cons": "容易被洗出", "suitable_for": "低波动股票"},
    "atr_based_2x": {"stop_price": 13.70, "stop_pct": -8.7, "strategy_type": "atr_2x", "description": "2x ATR 止损，平衡型", "pros": "兼顾灵敏度与抗噪", "cons": "极端行情偏慢", "suitable_for": "大多数股票"},
    "atr_based_3x": {"stop_price": 13.05, "stop_pct": -13.0, "strategy_type": "atr_3x", "description": "3x ATR 止损，宽松型", "pros": "抗震荡强", "cons": "单笔亏损大", "suitable_for": "高波动股票"},
    "trailing": {"stop_price": 13.70, "stop_pct": -8.7, "strategy_type": "trailing", "description": "上涨后自动上移止损", "pros": "锁定利润", "cons": "震荡市易洗出", "suitable_for": "趋势向上股票"},
    "time_based": {"stop_price": 14.25, "stop_pct": -5.0, "strategy_type": "time_based", "description": "10日内未涨即止损", "pros": "避免时间成本", "cons": "可能错过慢牛", "suitable_for": "高轮动策略"},
    "technical_support": {"stop_price": 13.60, "stop_pct": -9.3, "strategy_type": "technical_support", "description": "关键支撑位下方3%", "pros": "心理关口有效", "cons": "假突破频繁", "suitable_for": "有明显支撑位的股票"},
    "technical_bollinger": {"stop_price": 13.80, "stop_pct": -8.0, "strategy_type": "technical_bollinger", "description": "布林带下轨下方1%", "pros": "统计学支撑", "cons": "趋势市下轨下移", "suitable_for": "震荡市"},
    "technical_low": {"stop_price": 13.50, "stop_pct": -10.0, "strategy_type": "technical_low", "description": "20日低点下方2%", "pros": "符合技术派习惯", "cons": "亏损已较大", "suitable_for": "技术分析为主"},
    "recommended_strategy": "atr_based_2x",
    "recommended_stop_loss": 13.70,
    "recommendation_reason": "默认推荐 2x ATR 止损，兼顾灵敏度与抗噪能力",
    "trailing_rule": "初始止损 13.70 元；股价每上涨 1x ATR (+0.65元)，止损上移 0.5x ATR (+0.33元)；止损永不下移。",
    "time_stop_rule": "若 10 个交易日内未上涨 5% 以上，无论盈亏均触发止损"
  }
}
```

# Constraints
- risk_level 必须是 1（极低）到 5（极高）的整数。
- max_position_pct 必须在 0.05 ~ 0.50 之间，且不得超过 portfolio_risk.recommended_max_position。
- risk_reward_ratio 必须基于目标价和止损价计算。
- recommended_stop_loss 必须与 dynamic_stop_loss.recommended_stop_loss 一致。
- 若存在未披露的财报/监管问询/ST/退市风险，risk_level 不得低于 3。
- 当 VaR(95%) < -4% 或 CVaR(95%) < -6% 时，max_position_pct 建议不超过 0.10。
- 当 blowup_probability > 5% 时，risk_level 建议不低于 4。
- 当组合 Beta > 1.5 或行业 HHI > 2500 时，max_position_pct 建议不超过 0.10。
- stress_test 字段必须完整返回，数值与 Prompt 中提供的一致。
- portfolio_risk 字段必须完整返回（含用户持仓时），数值与 Prompt 中提供的一致。
- dynamic_stop_loss 字段必须完整返回，包含至少 8 种策略及推荐策略。
"""
