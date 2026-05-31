"""
MASS SA-Agent: 情绪面分析师 v2.3
支持: K线推导情绪 + 资金流向情绪 + 新闻情感 + 社交媒体情绪 + 情绪动量 + 板块/大盘对比
"""
from typing import Dict, Any, Optional

from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.tools.sentiment_tool import SentimentTool


class SA_Agent(BaseAgent):
    """情绪面分析师 Agent"""

    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """情绪面分析"""
        user_prompt = self._build_sa_prompt(snapshot)

        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)

            # 校验 sentiment_index 范围
            si = parsed.get("sentiment_index", 0.0)
            parsed["sentiment_index"] = max(-1.0, min(1.0, float(si if si is not None else 0.0)))

            # 校验 sentiment_percentile
            sp = parsed.get("sentiment_percentile", 50)
            parsed["sentiment_percentile"] = max(0, min(100, int(sp if sp is not None else 50)))

            # 情绪极端值与信号联动（逆向投资逻辑）
            if parsed["sentiment_percentile"] > 80 and parsed["signal"] == 1:
                logger.warning("SA-Agent: 情绪过热但信号为买入，修正为观望")
                parsed["signal"] = 0
                parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

            if parsed["sentiment_percentile"] < 20 and parsed["signal"] == -1:
                logger.warning("SA-Agent: 情绪过冷但信号为卖出，修正为观望")
                parsed["signal"] = 0
                parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

            # 确保 contrarian_opportunity 存在
            if not parsed.get("contrarian_opportunity"):
                parsed["contrarian_opportunity"] = self._infer_contrarian_opportunity(parsed)

            # 确保 crowd_behavior 不为空或未知
            cb = parsed.get("crowd_behavior", "")
            if not cb or cb == "未知":
                parsed["crowd_behavior"] = self._infer_crowd_behavior(snapshot, parsed)

            return self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"SA-Agent分析失败: {e}")
            return self._fallback_opinion(snapshot)

    def _fetch_social_media_data(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """获取社交媒体情绪数据（延迟初始化 SentimentTool）"""
        try:
            tool = SentimentTool()
            return tool.fetch_social_media_sentiment(snapshot.stock_code)
        except Exception as e:
            logger.debug(f"社交媒体情绪获取失败: {e}")
            return {"_error": str(e)}

    def _compute_relative_sentiment(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        相对情绪对比分析 — 方向三

        比较个股情绪 vs 板块情绪 vs 大盘情绪，识别：
        1. 个股是否被过度炒作（个股情绪 >> 板块/大盘）
        2. 个股是否存在独立利好/利空（个股情绪与板块/大盘背离）
        3. 系统性风险/机会（大盘情绪极端时的个股相对表现）
        """
        result = {
            "individual_sentiment": 0.0,
            "sector_sentiment": 0.0,
            "market_sentiment": 0.0,
            "relative_to_sector": 0.0,
            "relative_to_market": 0.0,
            "relative_score": 0,       # -10 ~ +10
            "relative_signal": "neutral",
            "relative_interpretation": "数据不足",
            "market_context": {},
        }

        # 1. 提取个股情绪
        sentiment = snapshot.sentiment_data
        kline_sent = sentiment.get("kline_sentiment", {})
        individual = kline_sent.get("sentiment_index", 0.0)
        result["individual_sentiment"] = individual

        # 2. 获取市场和板块情绪
        try:
            tool = SentimentTool()
            industry = snapshot.fundamentals.get("industry", "") if snapshot.fundamentals else ""
            market_sector = tool.fetch_market_and_sector_sentiment(
                snapshot.stock_code, industry=industry
            )

            if market_sector.get("_error"):
                return result

            result["market_context"] = market_sector
            result["sector_sentiment"] = market_sector.get("sector_sentiment_index", 0.0)
            result["market_sentiment"] = market_sector.get("market_sentiment_index", 0.0)

            sector_score = market_sector.get("sector_sentiment_score", 50.0)
            market_score = market_sector.get("market_sentiment_score", 50.0)
            sector_name = market_sector.get("sector_name", "未知板块")

            # 3. 计算相对偏差
            result["relative_to_sector"] = round(
                individual - result["sector_sentiment"], 3
            )
            result["relative_to_market"] = round(
                individual - result["market_sentiment"], 3
            )

            # 4. 评分与信号
            rel_sector = result["relative_to_sector"]
            rel_market = result["relative_to_market"]
            score = 0
            signal = "neutral"

            # 个股情绪显著强于板块和大盘
            if rel_sector > 0.3 and rel_market > 0.3:
                score = 8
                signal = "strong_overheat"
                interpretation = (
                    f"个股情绪显著强于板块({rel_sector:+.2f})和大盘({rel_market:+.2f})，"
                    f"可能存在独立利好驱动，但也需警惕过度炒作"
                )
            elif rel_sector > 0.2:
                score = 5
                signal = "overheat_vs_sector"
                interpretation = (
                    f"个股情绪强于板块({rel_sector:+.2f})，"
                    f"可能存在板块内独立行情或资金集中炒作"
                )
            elif rel_market > 0.2:
                score = 5
                signal = "overheat_vs_market"
                interpretation = (
                    f"个股情绪强于大盘({rel_market:+.2f})，"
                    f"可能受益于独立利好或资金抱团"
                )
            # 个股情绪显著弱于板块和大盘
            elif rel_sector < -0.3 and rel_market < -0.3:
                score = -8
                signal = "strong_underperformance"
                interpretation = (
                    f"个股情绪显著弱于板块({rel_sector:+.2f})和大盘({rel_market:+.2f})，"
                    f"可能存在独立利空或资金撤离"
                )
            elif rel_sector < -0.2:
                score = -5
                signal = "underperform_sector"
                interpretation = (
                    f"个股情绪弱于板块({rel_sector:+.2f})，"
                    f"可能跑输板块 peers"
                )
            elif rel_market < -0.2:
                score = -5
                signal = "underperform_market"
                interpretation = (
                    f"个股情绪弱于大盘({rel_market:+.2f})，"
                    f"可能受个股特定利空影响"
                )
            # 同步
            elif abs(rel_sector) < 0.1 and abs(rel_market) < 0.1:
                score = 0
                signal = "in_sync"
                interpretation = "个股情绪与板块、大盘基本一致，无显著背离"
            else:
                score = 2 if rel_sector > 0 else -2
                signal = "mild_divergence"
                interpretation = f"个股情绪与板块/大盘存在轻微偏差（板块{rel_sector:+.2f}, 大盘{rel_market:+.2f})"

            # 5. 系统性极端情景修正
            market_si = result["market_sentiment"]
            if abs(market_si) > 0.6:
                # 大盘情绪极端时，个股相对表现更有意义
                if market_si > 0.6 and individual < 0:
                    score -= 3
                    interpretation += "；注意：大盘狂热但个股冷淡，可能资金正在从该个股流出"
                elif market_si < -0.6 and individual > 0:
                    score += 3
                    interpretation += "；注意：大盘恐慌但个股坚挺，可能存在强支撑或独立逻辑"

            result["relative_score"] = score
            result["relative_signal"] = signal
            result["relative_interpretation"] = interpretation

        except Exception as e:
            logger.debug(f"相对情绪对比计算失败: {e}")

        return result

    def _compute_sentiment_momentum(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        情绪动量分析 — 方向二

        不仅看情绪的绝对水平（百分位），更要看情绪的变化速度（动量）。
        核心洞察：情绪加速向极端 = 即将反转；情绪从极端缓和 = 反转确认。

        计算维度：
        1. K线情绪动量：从近N日K线逐日推导情绪指数，计算变化速度
        2. 量价动量：成交量+价格偏离度的变化速度
        3. 社交媒体动量：参与意愿的变化趋势（如有数据）
        """
        result = {
            "momentum_1d": 0.0,
            "momentum_3d": 0.0,
            "momentum_5d": 0.0,
            "momentum_10d": 0.0,
            "momentum_direction": "未知",
            "acceleration": 0.0,       # 动量的二阶导（加速度）
            "momentum_signal": "数据不足",
            "reversal_probability": 0.0,       # 基于动量的反转概率估计
            "si_history": [],          # 近10日情绪指数序列
            "si_current": 0.0,
        }

        kline = snapshot.kline_df
        if kline is None or len(kline) < 20:
            return result

        try:
            import pandas as pd
            import numpy as np

            df = kline.copy()
            for col in ["close", "high", "low", "open", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close", "volume"])

            if len(df) < 20:
                return result

            # ── 1. 逐日计算简化情绪指数 ──
            # 基于：涨跌幅(+/-)、成交量相对20日均值、价格偏离MA20
            # 每日情绪 = 0.4*涨跌幅_norm + 0.3*成交量_norm + 0.3*偏离度_norm
            si_history = []
            min_window = 20
            for i in range(min_window, len(df) + 1):
                window = df.iloc[i - min_window:i]
                if len(window) < min_window:
                    continue

                close_series = window["close"]
                vol_series = window["volume"]

                # 当日涨跌幅（归一化到 [-1, 1]）
                if len(close_series) >= 2:
                    daily_return = close_series.iloc[-1] / close_series.iloc[-2] - 1
                    return_norm = max(-1.0, min(1.0, daily_return / 0.1))  # 10% 归一化到 1
                else:
                    return_norm = 0.0

                # 成交量相对20日均值（归一化）
                vol_ma20 = vol_series.mean()
                vol_today = vol_series.iloc[-1]
                if vol_ma20 > 0:
                    vol_ratio = vol_today / vol_ma20
                    vol_norm = max(-1.0, min(1.0, (vol_ratio - 1.0) / 1.5))  # 1.5倍归一化到 1
                else:
                    vol_norm = 0.0

                # 价格偏离 MA20（归一化）
                ma20 = close_series.mean()
                price = close_series.iloc[-1]
                if ma20 > 0:
                    dev = (price / ma20 - 1)
                    dev_norm = max(-1.0, min(1.0, dev / 0.15))  # 15% 归一化到 1
                else:
                    dev_norm = 0.0

                # 综合简化情绪指数
                daily_si = return_norm * 0.40 + vol_norm * 0.30 + dev_norm * 0.30
                si_history.append(round(daily_si, 3))

            if len(si_history) < 5:
                return result

            # 取最近 10 日
            si_history = si_history[-10:]
            result["si_history"] = si_history
            result["si_current"] = si_history[-1]

            # ── 2. 计算多周期动量 ──
            result["momentum_1d"] = round(si_history[-1] - si_history[-2], 3) if len(si_history) >= 2 else 0.0
            result["momentum_3d"] = round(si_history[-1] - si_history[-4], 3) if len(si_history) >= 4 else 0.0
            result["momentum_5d"] = round(si_history[-1] - si_history[-6], 3) if len(si_history) >= 6 else 0.0
            result["momentum_10d"] = round(si_history[-1] - si_history[0], 3) if len(si_history) >= 10 else 0.0

            # ── 3. 计算加速度（动量的二阶导） ──
            if len(si_history) >= 5:
                mom_recent = si_history[-1] - si_history[-3]  # 近3日动量
                mom_prior = si_history[-3] - si_history[-5]    # 前3日动量
                result["acceleration"] = round(mom_recent - mom_prior, 3)

            # ── 4. 动量方向判断 ──
            mom_5d = result["momentum_5d"]
            accel = result["acceleration"]
            si_curr = result["si_current"]

            if abs(mom_5d) > 0.3 and abs(accel) > 0.15:
                result["momentum_direction"] = "加速" + ("升温" if mom_5d > 0 else "降温")
            elif abs(mom_5d) > 0.15:
                result["momentum_direction"] = "缓和" + ("升温" if mom_5d > 0 else "降温")
            else:
                result["momentum_direction"] = "平稳"

            # ── 5. 动量信号 ──
            if abs(si_curr) > 0.5 and abs(mom_5d) > 0.3:
                result["momentum_signal"] = "情绪正在极端化，需警惕反转"
            elif abs(si_curr) > 0.5 and mom_5d * si_curr < 0:
                # 情绪在极端区但动量反向 → 反转信号
                result["momentum_signal"] = "情绪从极端开始回落，反转可能正在发生"
            elif abs(si_curr) < 0.3 and abs(mom_5d) > 0.3:
                result["momentum_signal"] = "情绪快速偏离中性，关注方向持续性"
            elif abs(mom_5d) < 0.15:
                result["momentum_signal"] = "情绪趋于平稳，暂无明确动量方向"
            else:
                result["momentum_signal"] = "情绪温和变化，观察中"

            # ── 6. 反转概率估计（启发式） ──
            reversal_prob = 0.0
            if abs(si_curr) > 0.6:
                # 情绪越极端，反转概率越高
                base_prob = (abs(si_curr) - 0.6) / 0.4 * 0.4  # 0.6→0%, 1.0→40%
                if mom_5d * si_curr < 0:
                    # 动量已经开始反向
                    reversal_prob = base_prob + 0.25
                elif abs(mom_5d) > 0.3:
                    # 动量加速向极端 → 即将反转
                    reversal_prob = base_prob + 0.15
                else:
                    reversal_prob = base_prob
            else:
                if abs(mom_5d) > 0.4 and abs(accel) > 0.2:
                    reversal_prob = 0.15  # 快速偏离中性也可能短期反转

            result["reversal_probability"] = round(min(0.95, reversal_prob), 2)

            # ── 7. 尝试融合社交媒体动量 ──
            try:
                tool = SentimentTool()
                social = tool.fetch_social_media_sentiment(snapshot.stock_code)
                if not social.get("_error"):
                    pw_change = social.get("participation_will_change", 0)
                    pw = social.get("participation_will", 50)
                    # 如果社交媒体动量与K线动量方向一致，增强信号
                    if pw_change > 5 and result["momentum_5d"] > 0:
                        result["momentum_signal"] += "（社交媒体共振加速）"
                    elif pw_change < -5 and result["momentum_5d"] < 0:
                        result["momentum_signal"] += "（社交媒体共振降温）"
                    elif pw_change > 5 and result["momentum_5d"] < 0:
                        result["momentum_signal"] += "（注意：社交媒体与量价情绪背离）"
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"情绪动量计算失败: {e}")

        return result

    def _build_sa_prompt(self, snapshot: StockSnapshot) -> str:
        """构建情绪面分析Prompt v2.1 — 整合K线推导、资金流向、新闻、社交媒体情绪"""
        parts = [
            f"## 分析对象",
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"当前价格: {snapshot.current_price}",
            "",
            "## 分析任务",
            "你是一位资深行为金融学家和情绪面分析专家。请基于以下多维情绪指标，"
            "深度分析当前市场情绪状态，识别 crowd 行为模式，并给出逆向交易信号。",
            "",
            "分析时必须考虑：",
            "1. 量价关系揭示的真实情绪（而非表面价格涨跌）",
            "2. 主力资金与散户行为的背离（机构 vs crowd）",
            "3. 情绪极端值与价格反转的概率关系",
            "4. 不同时间维度的情绪一致性（短期冲动 vs 中期趋势）",
            "5. 社交媒体情绪与真实市场行为的背离（高关注≠高价值）",
            "6. 必须区分'机构驱动上涨'和'散户FOMO推动'，后者即使情绪高也应给负面 signal",
            "",
        ]

        sentiment = snapshot.sentiment_data

        # ── 1. K线推导情绪指标 ──
        kline_sent = sentiment.get("kline_sentiment", {})
        if kline_sent and not kline_sent.get("_error"):
            parts.extend([
                "### 一、量价行为情绪（从真实K线推导）",
                f"- 综合情绪指数: {kline_sent.get('sentiment_index', 'N/A')} "
                f"(范围[-1,1]，越接近-1越恐慌，越接近1越狂热)",
                f"- 情绪百分位: {kline_sent.get('sentiment_percentile', 'N/A')}/100 "
                f"(>80情绪过热，<20情绪过冷)",
                f"- 年化波动率: {kline_sent.get('volatility_annual', 'N/A')} "
                f"(高波动=恐慌或狂热，低波动=压抑)",
                f"- 成交量情绪: {kline_sent.get('volume_sentiment', 'N/A')}",
                f"- 5日成交量/20日均量: {kline_sent.get('volume_ma5_ratio', 'N/A')}",
                f"- 趋势情绪: {kline_sent.get('trend_emotion', 'N/A')}",
                f"- 价格偏离MA20: {kline_sent.get('price_dev_ma20', 'N/A')}%",
                f"- 价格偏离MA60: {kline_sent.get('price_dev_ma60', 'N/A')}%",
                f"- 连续上涨天数: {kline_sent.get('consecutive_up', 'N/A')}",
                f"- 连续下跌天数: {kline_sent.get('consecutive_down', 'N/A')}",
                f"- 5日涨跌幅: {kline_sent.get('price_change_5d_pct', 'N/A')}%",
                f"- 20日涨跌幅: {kline_sent.get('price_change_20d_pct', 'N/A')}%",
                f"- 推断 crowd 行为: {kline_sent.get('crowd_behavior', 'N/A')}",
                "",
            ])

        # ── 2. 情绪动量分析（v2.2 新增） ──
        momentum = self._compute_sentiment_momentum(snapshot)
        if momentum.get("si_history"):
            hist_str = ", ".join(f"{v:+.2f}" for v in momentum["si_history"])
            parts.extend([
                "### 二、情绪动量分析（变化速度）",
                f"- **当前情绪指数**: {momentum['si_current']:+.2f}",
                f"- 近10日情绪序列: [{hist_str}]",
                f"- 1日动量: {momentum['momentum_1d']:+.3f}",
                f"- 3日动量: {momentum['momentum_3d']:+.3f}",
                f"- 5日动量: {momentum['momentum_5d']:+.3f}",
                f"- 10日动量: {momentum['momentum_10d']:+.3f}",
                f"- 加速度: {momentum['acceleration']:+.3f} (动量的二阶导)",
                f"- **动量方向**: {momentum['momentum_direction']}",
                f"- **动量信号**: {momentum['momentum_signal']}",
                f"- 基于动量的反转概率估计: {momentum['reversal_probability']:.0%}",
                "",
                "**情绪动量指引**：",
                "- 情绪指数 > 0.5 + 5日动量 > +0.3 → 加速狂热，高度警惕反转",
                "- 情绪指数 < -0.5 + 5日动量 < -0.3 → 加速恐慌，关注逆向买入机会",
                "- 情绪在极端区但动量反向（如指数>0.5但动量<0）→ 反转可能正在发生",
                "- 加速度 > 0.15 且方向与当前情绪一致 → 情绪在加速极端化",
                "- 加速度 < -0.15 → 情绪极端化的速度在减缓，可能接近拐点",
                "- 动量平稳(|5日动量|<0.15) → 趋势可能延续，暂无反转压力",
                "",
            ])

        # ── 3. 相对情绪对比（v2.3 新增） ──
        relative = self._compute_relative_sentiment(snapshot)
        market_ctx = relative.get("market_context", {})
        if not market_ctx.get("_error") and (relative.get("sector_sentiment") != 0.0 or relative.get("market_sentiment") != 0.0):
            parts.extend([
                "### 三、相对情绪对比（个股 vs 板块 vs 大盘）",
                f"- **个股情绪指数**: {relative['individual_sentiment']:+.2f}",
                f"- **板块情绪指数**: {relative['sector_sentiment']:+.2f} ({market_ctx.get('sector_name', 'N/A')})",
                f"- **大盘情绪指数**: {relative['market_sentiment']:+.2f} (趋势: {market_ctx.get('market_trend', 'N/A')})",
                f"- **个股 vs 板块偏差**: {relative['relative_to_sector']:+.2f}",
                f"- **个股 vs 大盘偏差**: {relative['relative_to_market']:+.2f}",
                f"- **相对信号**: {relative['relative_signal']}",
                f"- **解读**: {relative['relative_interpretation']}",
                "",
                "**相对情绪指引**：",
                "- 个股情绪 > 板块+0.3 且 > 大盘+0.3 → 个股可能被过度炒作，或存在独立利好",
                "- 个股情绪 < 板块-0.3 且 < 大盘-0.3 → 个股可能存在独立利空，或资金正在撤离",
                "- 个股情绪与板块/大盘同步(|偏差|<0.1) → 个股受系统性情绪主导",
                "- 大盘情绪极端(>0.6或<-0.6) + 个股反向 → 关注独立逻辑或资金调仓",
                "- 板块强于大盘 + 个股强于板块 → 三重共振，趋势最强势",
                "- 板块弱于大盘 + 个股弱于板块 → 三重弱势，回避为宜",
                "",
            ])

        # ── 4. 资金流向情绪 ──
        ff_sent = sentiment.get("fund_flow_sentiment", {})
        if ff_sent and not ff_sent.get("_error"):
            parts.extend([
                "### 四、资金流向情绪（从真实资金流向推导）",
                f"- 主力情绪: {ff_sent.get('main_force_emotion', 'N/A')} "
                f"(得分: {ff_sent.get('main_force_score', 'N/A')})",
                f"- 散户情绪: {ff_sent.get('retail_emotion', 'N/A')} "
                f"(得分: {ff_sent.get('retail_score', 'N/A')})",
                f"- 大单特征: {ff_sent.get('large_order_emotion', 'N/A')}",
                f"- 资金流向综合情绪: {ff_sent.get('fund_flow_sentiment', 'N/A')}",
                "",
            ])

        # ── 5. 社交媒体情绪（v2.1 新增） ──
        social = self._fetch_social_media_data(snapshot)
        if not social.get("_error"):
            rank_direction = "上升" if social.get('rank_change', 0) > 0 else "下降"
            parts.extend([
                "### 五、社交媒体情绪（东方财富千股千评）",
                f"- **综合社交媒体情绪指数**: {social.get('social_sentiment_index', 'N/A')} "
                f"(范围[-1,1]，基于综合得分+关注指数+机构参与度+参与意愿)",
                f"- 千股千评综合得分: {social.get('social_sentiment_score', 'N/A')}/100",
                f"- 用户关注指数: {social.get('attention_index', 'N/A')}",
                f"- 机构参与度: {social.get('institution_participation', 'N/A')}%",
                f"- 市场排名: {social.get('market_rank', 'N/A')} (较上期{rank_direction}{abs(social.get('rank_change', 0))}名)",
                f"- 市场参与意愿: {social.get('participation_will', 'N/A')} (5日均{social.get('participation_will_5d_avg', 'N/A')})",
                f"- 参与意愿变化: {social.get('participation_will_change', 0):+.1f}",
                f"- 评分趋势: {social.get('score_trend', 'N/A')}",
                f"- 关注趋势: {social.get('attention_trend', 'N/A')}",
            ])
            if social.get("anomaly_flag"):
                parts.append(f"- ** 异常情绪标记**: {social.get('anomaly_reason', '')}")
            parts.extend([
                "",
                "**社交媒体情绪指引**：",
                "- 综合得分 > 80 + 机构参与度 > 60 → 机构与散户共振看好",
                "- 关注指数 > 90 + 机构参与度 < 30 → 散户热炒，警惕反向操作",
                "- 参与意愿激增 + 排名飙升但评分低迷 → 疑似流量炒作",
                "- 社交媒体情绪与K线推导情绪背离时，优先相信量价行为",
                "- 高关注指数但价格横盘 → 有人造势，注意消息面风险",
                "",
            ])

        # ── 6. 爬虫原始情绪数据 ──
        if any(k in sentiment for k in ["social_sentiment_7d", "search_index_change", "news_count_7d"]):
            parts.extend([
                "### 六、舆情数据",
                f"- 社交媒体情绪7日: {sentiment.get('social_sentiment_7d', 'N/A')}",
                f"- 搜索指数变化: {sentiment.get('search_index_change', 'N/A')}%",
                f"- 新闻数量7日: {sentiment.get('news_count_7d', 'N/A')}",
                "",
            ])

        # ── 7. 新闻情感分析 ──
        news_analysis = sentiment.get("news_analysis", {})
        if news_analysis and news_analysis.get("sentiment_score") is not None:
            parts.extend([
                "### 七、新闻情感分析",
                f"- 平均情感得分: {news_analysis.get('sentiment_score', 'N/A')}",
                f"- 正面: {news_analysis.get('positive_count', 0)} | "
                f"负面: {news_analysis.get('negative_count', 0)} | "
                f"中性: {news_analysis.get('neutral_count', 0)}",
                f"- 主导基调: {news_analysis.get('dominant_tone', 'N/A')}",
                f"- 分析方法: {news_analysis.get('analysis_method', 'N/A')}",
                "",
            ])

        # 新闻标题
        news = sentiment.get("news", [])
        if news:
            parts.extend(["### 近期新闻标题",])
            for n in news[:5]:
                parts.append(f"- [{n.get('source', '')}] {n.get('title', '')}")
            parts.append("")

        # ── 7. 分析指引 ──
        parts.extend([
            "## 分析指引",
            "请按以下框架进行分析：",
            "",
            "1. **情绪状态诊断**：当前处于情绪周期的哪个阶段？",
            "   - 极端恐慌（sentiment_percentile < 20）→ 逆向买入机会",
            "   - 恐慌消退（20-40）→ 观望或试探",
            "   - 中性（40-60）→ 等待方向",
            "   - 乐观膨胀（60-80）→ 警惕过热",
            "   - 极端狂热（>80）→ 逆向卖出信号",
            "",
            "2. **情绪动量分析**：情绪的变化速度比绝对水平更重要",
            "   - 情绪加速向极端（动量>0.3且方向与当前情绪一致）→ 即将反转",
            "   - 情绪从极端回落（动量反向）→ 反转确认中",
            "   - 动量平稳（|5日动量|<0.15）→ 趋势延续，暂无反转压力",
            "   - 加速度由正转负 → 情绪极端化的动力在衰竭",
            "",
            "3. **相对情绪对比**：个股情绪是否与板块/大盘同步？",
            "   - 个股显著强于板块+大盘（偏差>+0.3）→ 可能存在独立利好或过度炒作",
            "   - 个股显著弱于板块+大盘（偏差<-0.3）→ 可能存在独立利空或资金撤离",
            "   - 大盘极端 + 个股反向 → 独立逻辑验证（强支撑或资金调仓）",
            "   - 板块>大盘>个股 或 板块<大盘<个股 → 同向层级，趋势一致性强",
            "",
            "4. **Crowd 行为识别**：当前主导 crowd 行为是什么？",
            "   - FOMO追涨 / 恐慌抛售 / 盲目抄底 / 获利了结 / 普遍观望 / 机构吸筹",
            "   - 社交媒体关注热度是否与实际基本面匹配？",
            "   - 散户和机构的行为是否一致？存在哪些背离？",
            "   - 量价关系是否支持当前 crowd 行为判断？",
            "",
            "5. **社交媒体情绪研判**：",
            "   - 高关注 + 低机构参与 = 散户热炒，反向信号",
            "   - 高关注 + 高机构参与 = 共振看好，趋势可能延续",
            "   - 参与意愿与价格背离 = 有人造势，警惕消息风险",
            "",
            "6. **逆向信号判断**：",
            "   - 情绪是否达到极端值（百分位<20 或 >80）？",
            "   - 情绪动量是否支持反转？（加速极端化=即将反转，动量反向=反转中）",
            "   - 个股情绪相对板块/大盘是否极端偏离？（独立炒作的逆向信号）",
            "   - 量价关系是否支持情绪反转？（放量滞涨/缩量止跌等）",
            "   - 主力资金流向是否与 crowd 情绪相反？（机构吸筹 vs 散户恐慌）",
            "   - 社交媒体情绪是否过度放大？（发帖量激增但价格未动）",
            "",
            "7. **时间维度一致性**：",
            "   - 短期（5日）情绪与中期（20日）趋势是否一致？",
            "   - 社交媒体热度持续性如何？（突发 vs 持续）",
            "   - 不一致时哪个更可靠？（优先相信量价行为 + 动量变化）",
            "",
            "8. **风险提示**:",
            "   - 当前情绪面最大的风险是什么？",
            "   - 哪些信号会改变你的判断？",
            "",
            "输出严格JSON格式：",
        ])

        return "\n".join(parts)

    def _fallback_opinion(self, snapshot: StockSnapshot) -> AgentOpinion:
        """基于已计算情绪指标的降级分析（不再返回'数据不足'或'未知'）"""
        sentiment = snapshot.sentiment_data

        # ── 优先使用K线推导的情绪指标 ──
        kline_sent = sentiment.get("kline_sentiment", {})
        if kline_sent and not kline_sent.get("_error"):
            si = kline_sent.get("sentiment_index", 0.0)
            sp = kline_sent.get("sentiment_percentile", 50)
            crowd = kline_sent.get("crowd_behavior", "")
            if not crowd or crowd == "未知":
                crowd = self._infer_crowd_behavior_from_kline(kline_sent)

            signal = 0
            if sp > 80:
                signal = -1  # 情绪过热，逆向卖出
            elif sp < 20:
                signal = 1   # 情绪过冷，逆向买入

            # 考虑资金流向修正
            ff_sent = sentiment.get("fund_flow_sentiment", {})
            ff_score = ff_sent.get("fund_flow_sentiment", 0.0) if ff_sent else 0.0
            if signal == 0 and abs(ff_score) > 0.5:
                signal = 1 if ff_score < -0.3 else (-1 if ff_score > 0.3 else 0)

            # ── v2.1 新增：社交媒体情绪修正 ──
            social = self._fetch_social_media_data(snapshot)
            social_raw = {}
            if not social.get("_error"):
                social_raw = social
                social_index = social.get("social_sentiment_index", 0.0)
                attention = social.get("attention_index", 50.0)
                inst_part = social.get("institution_participation", 50.0)
                anomaly = social.get("anomaly_flag", False)

                # 社交媒体情绪与K线情绪背离修正
                if si > 0.5 and social_index < -0.3:
                    # K线显示狂热但社交媒体冷淡 → 可能是假突破
                    signal = -1 if signal == 1 else signal
                    crowd += "（注意：量价情绪与社交媒体情绪背离）"
                elif si < -0.5 and social_index > 0.3:
                    # K线显示恐慌但社交媒体乐观 → 可能是诱空
                    signal = 1 if signal == -1 else signal
                    crowd += "（注意：量价情绪与社交媒体情绪背离）"

                # 异常情绪标记：高关注低机构 → 反向信号增强
                if anomaly and attention > 90 and inst_part < 30:
                    if signal == 0:
                        signal = -1  # 散户热炒，倾向卖出
                    confidence_adj = -0.1
                else:
                    confidence_adj = 0.0
            else:
                confidence_adj = 0.0

            # ── v2.2 新增：情绪动量修正 ──
            momentum = self._compute_sentiment_momentum(snapshot)
            momentum_raw = momentum if momentum.get("si_history") else {}
            if momentum_raw:
                mom_5d = momentum_raw.get("momentum_5d", 0.0)
                accel = momentum_raw.get("acceleration", 0.0)
                rev_prob = momentum_raw.get("reversal_probability", 0.0)
                si_curr = momentum_raw.get("si_current", si)

                # 动量加速向极端 → 增强逆向信号
                if abs(si_curr) > 0.5 and mom_5d * si_curr > 0 and abs(mom_5d) > 0.3:
                    # 情绪在极端区且加速向更极端 → 强烈反向信号
                    if signal == 0:
                        signal = -1 if si_curr > 0 else 1
                    crowd += f"（情绪动量加速{'狂热' if si_curr > 0 else '恐慌'}，反转概率{rev_prob:.0%}）"
                    confidence_adj += min(0.1, rev_prob * 0.2)

                # 动量从极端回落 → 确认反转
                elif abs(si_curr) > 0.5 and mom_5d * si_curr < 0:
                    # 情绪在极端区但动量反向 → 反转确认
                    if signal == 0:
                        signal = -1 if si_curr > 0 else 1
                    crowd += f"（情绪从极端回落，反转概率{rev_prob:.0%}）"
                    confidence_adj += min(0.05, rev_prob * 0.1)

                # 加速度衰竭 → 情绪极端化动力减弱
                elif abs(si_curr) > 0.5 and accel < -0.15 and mom_5d * si_curr > 0:
                    # 仍在向极端但速度在减慢 → 接近拐点
                    if signal == 0:
                        signal = -1 if si_curr > 0 else 1
                    crowd += "（情绪极端化速度在衰竭，接近拐点）"

            # ── v2.3 新增：相对情绪修正 ──
            relative = self._compute_relative_sentiment(snapshot)
            relative_raw = relative if relative.get("market_context") else {}
            if relative_raw and not relative_raw.get("market_context", {}).get("_error"):
                rel_sector = relative.get("relative_to_sector", 0.0)
                rel_market = relative.get("relative_to_market", 0.0)
                rel_score = relative.get("relative_score", 0)

                # 个股情绪显著强于板块和大盘 → 警惕过度炒作
                if rel_sector > 0.3 and rel_market > 0.3:
                    if si > 0.5:
                        # 个股狂热 + 显著强于环境 = 高度警惕反向
                        if signal == 0 or signal == 1:
                            signal = -1
                        crowd += "（个股情绪显著强于板块/大盘，警惕独立炒作）"
                        confidence_adj += 0.05
                    elif si > 0:
                        # 个股偏乐观但强于环境
                        if signal == 0:
                            signal = -1  # 倾向谨慎
                        crowd += "（个股情绪相对板块/大盘过热）"

                # 个股情绪显著弱于板块和大盘 → 可能存在独立利空
                elif rel_sector < -0.3 and rel_market < -0.3:
                    if si < -0.5:
                        # 个股恐慌 + 显著弱于环境 = 可能错杀，关注逆向买入
                        if signal == 0:
                            signal = 1
                        crowd += "（个股情绪显著弱于板块/大盘，可能存在独立利空或错杀）"
                        confidence_adj += 0.03
                    elif si < 0:
                        if signal == 0:
                            signal = 1
                        crowd += "（个股情绪相对板块/大盘过冷）"

                # 大盘极端 + 个股反向 → 独立逻辑验证
                market_si = relative.get("market_sentiment", 0.0)
                if abs(market_si) > 0.6:
                    if market_si > 0.6 and relative.get("individual_sentiment", 0) < 0:
                        # 大盘狂热但个股冷淡 → 资金流出该个股
                        if signal == 0:
                            signal = -1
                        crowd += "（大盘狂热但个股冷淡，注意资金调仓）"
                    elif market_si < -0.6 and relative.get("individual_sentiment", 0) > 0:
                        # 大盘恐慌但个股坚挺 → 强支撑或独立利好
                        if signal == 0:
                            signal = 1
                        crowd += "（大盘恐慌但个股坚挺，存在强支撑逻辑）"
                        confidence_adj += 0.05

            confidence = 0.55 + abs(si) * 0.3 + confidence_adj
            confidence = max(0.4, min(0.9, confidence))

            contrarian = self._infer_contrarian_opportunity({
                "sentiment_percentile": sp,
                "sentiment_index": si,
                "crowd_behavior": crowd,
            })

            reasoning = (
                f"【规则引擎降级分析】基于量价行为推导：{crowd}。"
                f"情绪指数{si:.2f}（百分位{sp}），"
                f"{'情绪过热' if sp > 80 else ('情绪过冷' if sp < 20 else '情绪中性')}。"
            )
            if ff_sent:
                reasoning += f"资金流向：{ff_sent.get('main_force_emotion', '中性')}。"
            if social_raw:
                reasoning += f"社交媒体情绪指数：{social_raw.get('social_sentiment_index', 'N/A')}。"
            if momentum_raw:
                reasoning += f"5日情绪动量：{momentum_raw.get('momentum_5d', 0):+.3f}。"
            if relative_raw:
                reasoning += f"相对板块偏差：{relative_raw.get('relative_to_sector', 0):+.2f}。"

            return AgentOpinion(
                agent_id=self.agent_id,
                signal=signal,
                confidence=round(confidence, 2),
                reasoning=reasoning,
                key_factors=[
                    f"量价情绪指数: {si}",
                    f"情绪百分位: {sp}",
                    f"crowd行为: {crowd}",
                ],
                risk_flags=["LLM调用异常，使用K线推导的规则引擎降级分析"],
                raw_data={
                    "signal": signal,
                    "confidence": confidence,
                    "sentiment_index": si,
                    "sentiment_percentile": sp,
                    "crowd_behavior": crowd,
                    "contrarian_opportunity": contrarian,
                    "social_media": social_raw,
                    "momentum": momentum_raw,
                    "relative_sentiment": relative_raw,
                },
            )

        # ── 次优：基于资金流向 ──
        ff_sent = sentiment.get("fund_flow_sentiment", {})
        if ff_sent and not ff_sent.get("_error"):
            ff_score = ff_sent.get("fund_flow_sentiment", 0.0)
            main_force = ff_sent.get("main_force_emotion", "中性")
            retail = ff_sent.get("retail_emotion", "中性")

            signal = 0
            if ff_score > 0.5:
                signal = -1
            elif ff_score < -0.5:
                signal = 1

            # 机构吸筹 vs 散户恐慌 → 买入信号
            if "吸筹" in main_force or ("恐慌" in retail and "流入" in main_force):
                signal = 1

            crowd = f"主力{main_force}，散户{retail}"
            confidence = 0.55
            contrarian = "主力行为与散户情绪背离，关注机构动向" if signal != 0 else "情绪中性，暂无明确逆向机会"

            return AgentOpinion(
                agent_id=self.agent_id,
                signal=signal,
                confidence=confidence,
                reasoning=f"【规则引擎降级分析】K线情绪指标不可用，基于资金流向推断：{crowd}。综合得分{ff_score:.2f}。",
                key_factors=[
                    f"主力情绪: {main_force}",
                    f"散户情绪: {retail}",
                    f"资金流向得分: {ff_score}",
                ],
                risk_flags=["K线情绪数据缺失，仅基于资金流向做降级推断"],
                raw_data={
                    "signal": signal,
                    "confidence": confidence,
                    "sentiment_index": 0.0,
                    "sentiment_percentile": 50,
                    "crowd_behavior": crowd,
                    "contrarian_opportunity": contrarian,
                },
            )

        # ── 再次：基于新闻/舆情 ──
        news_analysis = sentiment.get("news_analysis", {})
        if news_analysis and news_analysis.get("sentiment_score") is not None:
            ns = news_analysis.get("sentiment_score", 0.0)
            dominant = news_analysis.get("dominant_tone", "中性")

            signal = 0
            if ns > 0.5:
                signal = -1  # 新闻过于乐观 → 逆向卖出
            elif ns < -0.5:
                signal = 1   # 新闻过于悲观 → 逆向买入

            crowd = f"新闻主导基调：{dominant}"
            confidence = 0.5
            contrarian = "新闻情绪极端，关注与实际价量的背离" if signal != 0 else "新闻情绪中性，暂无明确逆向机会"

            return AgentOpinion(
                agent_id=self.agent_id,
                signal=signal,
                confidence=confidence,
                reasoning=f"【规则引擎降级分析】基于新闻情感推断：{dominant}（得分{ns:.2f}）。量价与资金流向数据均不可用。",
                key_factors=[
                    f"新闻情感得分: {ns}",
                    f"主导基调: {dominant}",
                ],
                risk_flags=["量价及资金流向数据缺失，仅基于新闻情感做降级推断"],
                raw_data={
                    "signal": signal,
                    "confidence": confidence,
                    "sentiment_index": ns,
                    "sentiment_percentile": 50,
                    "crowd_behavior": crowd,
                    "contrarian_opportunity": contrarian,
                },
            )

        # ── 最弱降级：基于任何可用数据的保守推断 ──
        # 绝不返回"数据不足"或"未知"
        signal = 0
        confidence = 0.4
        crowd = "可用情绪数据有限，市场状态不明朗"
        contrarian = "数据有限，建议等待更明确的情绪极端信号后再做逆向操作"

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=confidence,
            reasoning="【规则引擎降级分析】情绪指标均不可用，基于保守原则建议观望。建议待量价数据恢复后再做情绪面判断。",
            key_factors=["情绪指标数据暂不可用"],
            risk_flags=["情绪数据缺失，信号置信度低"],
            raw_data={
                "signal": signal,
                "confidence": confidence,
                "sentiment_index": 0.0,
                "sentiment_percentile": 50,
                "crowd_behavior": crowd,
                "contrarian_opportunity": contrarian,
            },
        )

    # ── 辅助推断方法 ──

    def _infer_crowd_behavior(self, snapshot: StockSnapshot, parsed: Dict[str, Any]) -> str:
        """基于 snapshot 和 parsed 推断 crowd 行为，禁止返回'未知'"""
        kline_sent = snapshot.sentiment_data.get("kline_sentiment", {})
        if kline_sent:
            return self._infer_crowd_behavior_from_kline(kline_sent)

        ff_sent = snapshot.sentiment_data.get("fund_flow_sentiment", {})
        if ff_sent:
            main_force = ff_sent.get("main_force_emotion", "")
            retail = ff_sent.get("retail_emotion", "")
            if "吸筹" in main_force:
                return "机构吸筹"
            if "恐慌" in retail:
                return "散户恐慌抛售"
            if "跟风" in retail:
                return "散户FOMO追涨"
            return f"主力{main_force}，散户{retail}"

        sp = parsed.get("sentiment_percentile", 50)
        if sp > 80:
            return "市场情绪过热，可能是FOMO追涨"
        if sp < 20:
            return "市场情绪过冷，可能是恐慌抛售"
        return "市场情绪中性，观望为主"

    def _infer_crowd_behavior_from_kline(self, kline_sent: Dict[str, Any]) -> str:
        """从K线情绪指标推断 crowd 行为"""
        volume_sent = kline_sent.get("volume_sentiment", "")
        consecutive_up = kline_sent.get("consecutive_up", 0) or 0
        consecutive_down = kline_sent.get("consecutive_down", 0) or 0
        price_dev = kline_sent.get("price_dev_ma20", 0) or 0
        si = kline_sent.get("sentiment_index", 0) or 0

        if "放量上涨" in volume_sent and price_dev > 5:
            return "FOMO追涨" if si > 0.5 else "放量反弹"
        if "放量下跌" in volume_sent and consecutive_down >= 2:
            return "恐慌抛售"
        if "缩量下跌" in volume_sent and price_dev < -5:
            return "低迷观望或筑底"
        if "缩量上涨" in volume_sent:
            return "机构控盘拉升或流动性不足"
        if consecutive_up >= 3:
            return "趋势追涨情绪"
        if consecutive_down >= 3:
            return "连续下跌后的悲观情绪"
        if "横盘" in volume_sent:
            return "普遍观望"
        return "情绪中性，无明显 crowd 偏向"

    def _infer_contrarian_opportunity(self, parsed: Dict[str, Any]) -> str:
        """基于解析结果推断逆向交易机会"""
        sp = parsed.get("sentiment_percentile", 50)
        si = parsed.get("sentiment_index", 0.0)
        crowd = parsed.get("crowd_behavior", "")

        if sp > 80:
            return f"情绪过热（百分位{sp}），crowd处于{crowd}状态，关注反向卖出机会"
        if sp < 20:
            return f"情绪过冷（百分位{sp}），crowd处于{crowd}状态，关注反向买入机会"
        if "FOMO" in crowd or "追涨" in crowd:
            return "散户FOMO推动，非机构驱动，警惕回调"
        if "恐慌" in crowd or "抛售" in crowd:
            return "恐慌情绪蔓延，可能是机构吸筹窗口"
        if "吸筹" in crowd:
            return "机构吸筹阶段，可能与散户情绪背离，关注后续拉升"
        return "情绪尚未到达极端，暂无明确逆向交易机会"

    def _default_prompt(self) -> str:
        return """你是资深行为金融学家，专精于A股市场情绪面分析与逆向投资。

## 核心能力
1. 从量价行为推断市场真实情绪（而非表面价格涨跌）
2. 识别 crowd 行为模式（FOMO、恐慌、观望、抄底、获利了结）
3. 区分"机构驱动"和"散户 crowd 驱动"的情绪
4. 判断情绪极端值与价格反转的概率关系

## 分析框架
- **情绪指数** [-1.0, 1.0]: -1=极度恐慌, 0=中性, 1=极度狂热
- **情绪百分位** [0, 100]: <20=情绪过冷(逆向买入), >80=情绪过热(逆向卖出)
- **Crowd 行为**: 必须基于量价数据给出具体描述，禁止返回"未知"
- **逆向信号**: 情绪极端时给出反向信号（别人恐惧我贪婪，别人贪婪我恐惧）

## 约束
- 当情绪百分位 > 80 时，signal 倾向于 -1（卖出/观望）
- 当情绪百分位 < 20 时，signal 倾向于 1（买入）
- 必须区分"机构吸筹"和"散户FOMO"的本质差异
- 当数据部分缺失时，基于已有数据做合理推断，绝不返回"数据不足"
- crowd_behavior 字段禁止返回"未知"

## 输出严格JSON格式
{
  "signal": 0,
  "confidence": 0.65,
  "sentiment_index": 0.72,
  "sentiment_percentile": 85,
  "reasoning": "详细分析理由（100-200字，必须包含具体的量价证据和crowd行为判断）",
  "key_factors": ["因素1", "因素2"],
  "risk_flags": ["风险1"],
  "crowd_behavior": "具体crowd行为描述（禁止'未知'）",
  "contrarian_opportunity": "逆向交易机会描述"
}"""
