"""
MASS 舆情/新闻情感分析工具 — 全真实数据版本
零容忍虚假数据，所有新闻必须来自真实媒体

架构:
- 默认使用增强规则引擎（否定词 + 程度副词 + 转折句处理）
- 支持配置切换到 LLM 引擎（Deepseek 等）进行高精度分析
- 新增：从K线/成交量/资金流向推断情绪指标（不依赖外部爬虫）
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from loguru import logger

from agent.crawlers import CrawlerRegistry


class SentimentTool:
    """情感分析工具 — 只分析真实新闻，绝不生成虚假新闻"""

    # ── 情感词典 ──
    POSITIVE_WORDS = [
        "上涨", "涨停", "利好", "突破", "反弹", "走强", "强劲", "超预期",
        "增长", "盈利", "净利润", "分红", "增持", "买入", "推荐", "看好",
        "创新高", "放量", "资金流入", "主力买入", "北向资金", "机构调研",
        "业绩大增", "订单饱满", "产能扩张", "政策利好", "行业景气",
    ]

    NEGATIVE_WORDS = [
        "下跌", "跌停", "利空", "破位", "回调", "走弱", "疲软", "不及预期",
        "下滑", "亏损", "亏损扩大", "减持", "卖出", "回避", "看空",
        "创新低", "缩量", "资金流出", "主力卖出", "大股东减持", "监管问询",
        "业绩下滑", "订单减少", "产能过剩", "政策收紧", "行业低迷",
        "暴跌", "崩盘", "腰斩", "退市", "ST", "暴雷", "财务造假",
    ]

    # ── 否定词 ──
    # 按长度降序排列，优先匹配长词（如"并没有"优先于"没有"）
    NEGATION_WORDS = [
        "并不是", "并没有", "不曾", "不再", "不必", "不宜",
        "不", "没", "无", "未", "非", "勿", "别",
        "没有", "不是", "并未", "难以", "不能", "不会",
        "不要", "缺乏", "不足", "不够", "不及", "不如",
    ]

    # ── 程度副词 ──
    INTENSIFIER_WORDS = {
        "极其": 2.0, "非常": 1.75, "十分": 1.5, "特别": 1.5,
        "相当": 1.5, "比较": 1.25, "较为": 1.25, "明显": 1.25,
        "大幅": 1.5, "剧烈": 1.75, "显著": 1.5, "强劲": 1.5,
        "略微": 0.75, "轻微": 0.75, "稍有": 0.75,
        "微": 0.5, "略": 0.75, "有点": 0.75, "有些": 0.75,
    }

    # ── 转折词（转折后的内容权重更高） ──
    TRANSITION_WORDS = [
        "但是", "但", "然而", "不过", "只是", "可是", "却",
        "反而", "尽管", "虽然", "虽说", "固然",
    ]

    # 否定翻转窗口：情感词前面多少字内出现否定词视为有效否定
    NEGATION_WINDOW = 6

    def __init__(self, use_llm: Optional[bool] = None):
        self._cache = {}
        self._registry = CrawlerRegistry.get_instance()
        # 配置开关：环境变量 > 传入参数 > 默认 False（规则引擎优先）
        self._use_llm = use_llm if use_llm is not None else (
            os.getenv("USE_LLM_SENTIMENT", "false").lower() == "true"
        )

    # ── 公共接口 ──

    def analyze_news(self, news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析新闻列表的情感倾向 — 只分析真实新闻

        Args:
            news_list: [{"title": str, "content": str, "time": str, "source": str, "url": str}, ...]

        Returns:
            情感分析结果（向后兼容原有字段）
        """
        if not news_list:
            return {
                "sentiment_score": None,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "dominant_tone": "无数据",
                "summary": "无新闻数据（所有新闻源均不可用）",
            }

        texts = [f"{n.get('title', '')} {n.get('content', '')}" for n in news_list]

        if self._use_llm:
            scores = self._analyze_with_llm(texts)
            explanations = ["LLM分析"] * len(scores)
        else:
            scores_exps = [self._enhanced_score(t) for t in texts]
            scores = [s for s, _ in scores_exps]
            explanations = [e for _, e in scores_exps]

        pos_count = sum(1 for s in scores if s > 0.2)
        neg_count = sum(1 for s in scores if s < -0.2)
        neu_count = len(scores) - pos_count - neg_count

        detailed = [
            {
                "title": news_list[i].get("title", "")[:60],
                "score": round(scores[i], 3),
                "explanation": explanations[i],
            }
            for i in range(len(news_list))
        ]

        avg_score = sum(scores) / len(scores) if scores else None

        avg_score = sum(scores) / len(scores) if scores else None

        if avg_score is None:
            tone = "无数据"
        elif avg_score > 0.3:
            tone = "偏正面"
        elif avg_score < -0.3:
            tone = "偏负面"
        else:
            tone = "中性"

        return {
            "sentiment_score": round(avg_score, 3) if avg_score is not None else None,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "dominant_tone": tone,
            "summary": f"共分析{len(news_list)}条真实新闻，正面{pos_count}条，负面{neg_count}条"
                      + (f"，平均情感得分{avg_score:.3f}" if avg_score is not None else ""),
            "analysis_method": "llm" if self._use_llm else "enhanced_rule",
            "details": detailed[:5],  # 前5条详情，供调试
        }

    def analyze_social_media(self, texts: List[str]) -> Dict[str, Any]:
        """分析社交媒体文本情感"""
        if not texts:
            return {"sentiment_index": None, "sample_count": 0}

        if self._use_llm:
            scores = self._analyze_with_llm(texts)
        else:
            scores = [self._enhanced_score(t)[0] for t in texts]
        return {
            "sentiment_index": round(sum(scores) / len(scores), 3),
            "sentiment_std": round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5, 3
            ) if scores else 0,
            "positive_ratio": round(sum(1 for s in scores if s > 0) / len(scores), 3),
            "sample_count": len(texts),
        }

    # ════════════════════════════════════════════════════════════════════
    # 情绪指标计算 — 从K线/成交量/资金流向推断市场情绪
    # ════════════════════════════════════════════════════════════════════

    def compute_sentiment_from_kline(self, kline_df) -> Dict[str, Any]:
        """
        从K线数据计算多维情绪指标。
        不依赖外部爬虫，完全基于已有真实行情数据推导情绪状态。
        """
        import pandas as pd
        import numpy as np

        if kline_df is None or len(kline_df) < 20:
            return {"_error": "K线数据不足，无法计算情绪指标"}

        df = kline_df.copy()
        # 确保必要列存在
        required = ["close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                return {"_error": f"K线数据缺少{col}字段"}

        # 转换为数值型
        for col in ["close", "high", "low", "open", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close", "volume"])
        if len(df) < 20:
            return {"_error": "有效K线数据不足"}

        # ── 1. 波动率情绪 ──
        returns = df["close"].pct_change().dropna()
        volatility_20d = returns.tail(20).std() * (252 ** 0.5)  # 年化波动率
        volatility_5d = returns.tail(5).std() * (252 ** 0.5)
        # 波动率情绪：极低=压抑(可能爆发)，极高=恐慌/狂热，适中=平静
        vol_score = 0.0
        if volatility_20d > 0.5:  # 极度波动
            vol_score = -0.7 if returns.tail(5).mean() < 0 else 0.7
        elif volatility_20d > 0.3:  # 高波动
            vol_score = -0.4 if returns.tail(5).mean() < 0 else 0.4
        elif volatility_20d < 0.15:  # 极低波动
            vol_score = 0.0  # 压抑观望

        # ── 2. 成交量情绪 ──
        vol_ma20 = df["volume"].tail(20).mean()
        vol_ma5 = df["volume"].tail(5).mean()
        vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
        price_change_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) >= 6 else 0

        volume_sentiment = "中性"
        vol_score_val = 0.0
        if vol_ratio > 1.5 and price_change_5d > 0.03:
            volume_sentiment = "放量上涨（FOMO追涨/机构进场）"
            vol_score_val = 0.6
        elif vol_ratio > 1.5 and price_change_5d < -0.03:
            volume_sentiment = "放量下跌（恐慌抛售）"
            vol_score_val = -0.6
        elif vol_ratio < 0.7 and price_change_5d > 0:
            volume_sentiment = "缩量上涨（上涨乏力/观望）"
            vol_score_val = 0.2
        elif vol_ratio < 0.7 and price_change_5d < 0:
            volume_sentiment = "缩量下跌（抛压减弱/可能企稳）"
            vol_score_val = -0.2
        elif vol_ratio > 1.5 and abs(price_change_5d) < 0.01:
            volume_sentiment = "放量横盘（多空激烈博弈）"
            vol_score_val = 0.0

        # ── 3. 趋势情绪（价格偏离度） ──
        ma20 = df["close"].tail(20).mean()
        ma60 = df["close"].tail(60).mean() if len(df) >= 60 else ma20
        price = df["close"].iloc[-1]
        dev_ma20 = (price / ma20 - 1) if ma20 > 0 else 0
        dev_ma60 = (price / ma60 - 1) if ma60 > 0 else 0

        trend_emotion = "中性"
        trend_score = 0.0
        if dev_ma20 > 0.1 and dev_ma60 > 0.15:
            trend_emotion = "严重超买（情绪狂热）"
            trend_score = 0.8
        elif dev_ma20 > 0.05:
            trend_emotion = "偏乐观"
            trend_score = 0.4
        elif dev_ma20 < -0.1 and dev_ma60 < -0.15:
            trend_emotion = "严重超卖（情绪恐慌）"
            trend_score = -0.8
        elif dev_ma20 < -0.05:
            trend_emotion = "偏悲观"
            trend_score = -0.4

        # ── 4. 连续涨跌天数 ──
        df["daily_return"] = df["close"].pct_change()
        recent_returns = df["daily_return"].dropna().tail(10)
        consecutive_up = 0
        consecutive_down = 0
        for r in reversed(recent_returns):
            if r > 0:
                consecutive_up += 1
                consecutive_down = 0
            elif r < 0:
                consecutive_down += 1
                consecutive_up = 0
            else:
                break

        streak_emotion = ""
        streak_score = 0.0
        if consecutive_up >= 5:
            streak_emotion = f"连续{consecutive_up}天上涨（FOMO情绪累积）"
            streak_score = 0.7
        elif consecutive_up >= 3:
            streak_emotion = f"连续{consecutive_up}天上涨（乐观情绪）"
            streak_score = 0.4
        elif consecutive_down >= 5:
            streak_emotion = f"连续{consecutive_down}天下跌（恐慌蔓延）"
            streak_score = -0.7
        elif consecutive_down >= 3:
            streak_emotion = f"连续{consecutive_down}天下跌（悲观情绪）"
            streak_score = -0.4

        # ── 5. 综合情绪指数 ──
        composite = (vol_score * 0.2 + vol_score_val * 0.25 +
                     trend_score * 0.35 + streak_score * 0.2)
        composite = max(-1.0, min(1.0, composite))

        # 情绪百分位（将 composite 映射到 0-100）
        percentile = int((composite + 1.0) / 2.0 * 100)

        # ── 6. 推断 crowd 行为 ──
        crowd_behavior = self._infer_crowd_behavior(
            vol_ratio, volatility_20d, dev_ma20, consecutive_up, consecutive_down,
            price_change_5d
        )

        return {
            "sentiment_index": round(composite, 3),
            "sentiment_percentile": percentile,
            "volatility_annual": round(volatility_20d, 3),
            "volatility_sentiment": round(vol_score, 2),
            "volume_ma5_ratio": round(vol_ratio, 2),
            "volume_sentiment": volume_sentiment,
            "volume_sentiment_score": round(vol_score_val, 2),
            "price_dev_ma20": round(dev_ma20 * 100, 2),  # 百分比
            "price_dev_ma60": round(dev_ma60 * 100, 2),
            "trend_emotion": trend_emotion,
            "trend_score": round(trend_score, 2),
            "consecutive_up": consecutive_up,
            "consecutive_down": consecutive_down,
            "streak_emotion": streak_emotion,
            "streak_score": round(streak_score, 2),
            "price_change_5d_pct": round(price_change_5d * 100, 2),
            "crowd_behavior": crowd_behavior,
            "analysis_method": "kline_derived",
            "data_quality": "real",
        }

    def _infer_crowd_behavior(self, vol_ratio, vol_annual, dev_ma20,
                              consec_up, consec_down, price_change_5d) -> str:
        """
        基于量价行为推断 crowd 行为模式。
        返回标准化描述，供 LLM 进一步推理。
        """
        # FOMO 追涨
        if consec_up >= 3 and vol_ratio > 1.3 and dev_ma20 > 0.05:
            return "FOMO追涨（散户跟风买入，情绪过热）"
        # 恐慌抛售
        if consec_down >= 3 and vol_ratio > 1.3:
            return "恐慌抛售（散户割肉，情绪崩溃）"
        # 机构建仓
        if vol_ratio > 1.5 and abs(price_change_5d) < 0.02 and dev_ma20 > -0.03:
            return "机构吸筹（放量横盘，主力暗中建仓）"
        # 观望等待
        if vol_ratio < 0.6 and abs(dev_ma20) < 0.03:
            return "普遍观望（成交萎缩，多空双方均不活跃）"
        # 获利了结
        if consec_up >= 4 and vol_ratio > 1.2 and price_change_5d > 0.05:
            return "获利了结（前期涨幅较大，资金开始兑现）"
        # 抄底行为
        if consec_down >= 4 and vol_ratio > 1.0 and dev_ma20 < -0.08:
            return "抄底试探（跌幅较大，部分资金开始试探性买入）"
        # 洗盘震仓
        if vol_ratio > 1.8 and abs(price_change_5d) < 0.03 and vol_annual > 0.4:
            return "剧烈洗盘（高波动+放量震荡，清洗浮筹）"
        # 默认
        if dev_ma20 > 0.03:
            return "偏乐观持仓（价格站上均线，市场情绪偏多）"
        elif dev_ma20 < -0.03:
            return "偏悲观持仓（价格跌破均线，市场情绪偏空）"
        return "情绪中性（多空力量均衡，等待方向选择）"

    def compute_sentiment_from_fund_flow(self, fund_flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        从资金流向数据提取情绪信号。
        """
        if not fund_flow or not isinstance(fund_flow, dict):
            return {"_error": "资金流向数据不可用"}

        # 提取关键字段
        main_net = fund_flow.get("main_net_inflow")
        retail_net = fund_flow.get("retail_net_inflow")
        large_order = fund_flow.get("large_order_ratio")

        if main_net is None:
            return {"_error": "主力资金流向数据缺失"}

        # 主力情绪
        main_emotion = "中性"
        main_score = 0.0
        try:
            mn = float(main_net)
            if mn > 50000000:  # 5000万
                main_emotion = "主力大幅净流入（机构强烈看好）"
                main_score = 0.8
            elif mn > 10000000:  # 1000万
                main_emotion = "主力净流入（机构看多）"
                main_score = 0.4
            elif mn < -50000000:
                main_emotion = "主力大幅净流出（机构撤离）"
                main_score = -0.8
            elif mn < -10000000:
                main_emotion = "主力净流出（机构看空）"
                main_score = -0.4
            else:
                main_emotion = "主力流向平稳"
                main_score = 0.0
        except (TypeError, ValueError):
            main_emotion = "主力数据异常"
            main_score = 0.0

        # 散户情绪
        retail_emotion = "中性"
        retail_score = 0.0
        if retail_net is not None:
            try:
                rn = float(retail_net)
                if rn > 0 and main_score > 0:
                    retail_emotion = "散户跟风买入（需警惕）"
                    retail_score = -0.3  # 散户跟风通常是反向指标
                elif rn < 0 and main_score < 0:
                    retail_emotion = "散户恐慌抛售（可能接近底部）"
                    retail_score = 0.3
                elif rn > 0 and main_score < 0:
                    retail_emotion = "散户接盘（主力出货）"
                    retail_score = -0.5
                elif rn < 0 and main_score > 0:
                    retail_emotion = "散户割肉（主力吸筹）"
                    retail_score = 0.5
            except (TypeError, ValueError):
                pass

        # 大单比例
        large_emotion = ""
        if large_order is not None:
            try:
                lo = float(large_order)
                if lo > 0.6:
                    large_emotion = f"大单主导({lo:.0%})，机构控盘度高"
                elif lo < 0.4:
                    large_emotion = f"小单主导({1-lo:.0%})，散户交易活跃"
                else:
                    large_emotion = f"大单占比{lo:.0%}，机构散户博弈均衡"
            except (TypeError, ValueError):
                pass

        return {
            "main_force_emotion": main_emotion,
            "main_force_score": round(main_score, 2),
            "retail_emotion": retail_emotion,
            "retail_score": round(retail_score, 2),
            "large_order_emotion": large_emotion,
            "fund_flow_sentiment": round(main_score + retail_score * 0.3, 2),
            "analysis_method": "fund_flow_derived",
        }

    def fetch_stock_news(self, stock_code: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取个股相关新闻 — 100%真实新闻

        策略:
        1. 优先通过爬虫获取真实新闻
        2. 没有任何新闻时返回空列表（绝不编造）
        3. 从所有可用源聚合新闻
        """
        news = []
        sources = ["sina", "eastmoney", "ths"]

        for source_name in sources:
            try:
                crawler = self._registry.get_crawler(source_name)
                if crawler:
                    result = crawler.fetch(stock_code, "news")
                    if result and isinstance(result, list):
                        news.extend(result)
                        logger.debug(f"[{source_name}] 获取 {len(result)} 条新闻")
            except Exception as e:
                logger.debug(f"[{source_name}] 新闻获取失败: {e}")

        # 去重（按标题）
        seen_titles = set()
        unique_news = []
        for item in news:
            title = item.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(item)

        if not unique_news:
            logger.warning(f"无法获取 {stock_code} 的真实新闻，所有新闻源均不可用")
            return []

        return unique_news[:count]

    # ════════════════════════════════════════════════════════════════════
    # 社交媒体情绪分析 — 东方财富千股千评
    # ════════════════════════════════════════════════════════════════════

    def fetch_social_media_sentiment(self, stock_code: str) -> Dict[str, Any]:
        """
        获取个股社交媒体情绪数据（东方财富千股千评体系）

        数据源：
        1. stock_comment_em — 全市场千股千评快照（综合得分、关注指数、机构参与度、排名）
        2. stock_comment_detail_scrd_desire_em — 市场参与意愿历史趋势
        3. stock_comment_detail_scrd_focus_em — 用户关注指数历史
        4. stock_comment_detail_zhpj_lspf_em — 综合评分历史
        5. stock_comment_detail_zlkp_jgcyd_em — 机构参与度历史

        Returns:
            {
                "social_sentiment_index": float,  # [-1, 1] 综合社交媒体情绪指数
                "social_sentiment_score": float,  # [0, 100] 综合得分
                "attention_index": float,         # 关注指数
                "institution_participation": float,       # 机构参与度(%)
                "market_rank": int,               # 目前排名
                "rank_change": int,               # 排名变化（上升为正）
                "participation_will": float,      # 最新市场参与意愿
                "participation_will_5d_avg": float,       # 5日平均参与意愿
                "participation_will_change": float,       # 参与意愿变化
                "score_trend": str,               # 评分趋势：上升/下降/平稳
                "attention_trend": str,           # 关注趋势
                "anomaly_flag": bool,             # 异常情绪标记
                "anomaly_reason": str,            # 异常原因
            }
        """
        if not stock_code:
            return {"_error": "股票代码为空"}

        cache_key = f"social_sentiment:{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = {
            "social_sentiment_index": 0.0,
            "social_sentiment_score": 50.0,
            "attention_index": 50.0,
            "institution_participation": 50.0,
            "market_rank": -1,
            "rank_change": 0,
            "participation_will": 50.0,
            "participation_will_5d_avg": 50.0,
            "participation_will_change": 0.0,
            "score_trend": "平稳",
            "attention_trend": "平稳",
            "anomaly_flag": False,
            "anomaly_reason": "",
        }

        try:
            import akshare as ak

            # 1. 获取千股千评快照
            comment_df = ak.stock_comment_em()
            if comment_df is not None and not comment_df.empty:
                row = comment_df[comment_df["代码"] == stock_code]
                if not row.empty:
                    r = row.iloc[0]
                    result["social_sentiment_score"] = self._safe_float(r.get("综合得分"), 50.0)
                    result["attention_index"] = self._safe_float(r.get("关注指数"), 50.0)
                    result["institution_participation"] = self._safe_float(r.get("机构参与度"), 50.0) * 100
                    result["market_rank"] = int(r.get("目前排名", -1) or -1)
                    result["rank_change"] = int(r.get("上升", 0) or 0)

            # 2. 获取市场参与意愿历史
            try:
                desire_df = ak.stock_comment_detail_scrd_desire_em(symbol=stock_code)
                if desire_df is not None and not desire_df.empty:
                    latest = desire_df.iloc[-1]
                    result["participation_will"] = self._safe_float(latest.get("参与意愿"), 50.0)
                    result["participation_will_5d_avg"] = self._safe_float(latest.get("5日平均参与意愿"), 50.0)
                    result["participation_will_change"] = self._safe_float(latest.get("参与意愿变化"), 0.0)

                    # 检测趋势
                    if len(desire_df) >= 10:
                        recent = desire_df.tail(5)["参与意愿"].mean()
                        prior = desire_df.tail(10).head(5)["参与意愿"].mean()
                        if recent > prior * 1.1:
                            result["score_trend"] = "上升"
                        elif recent < prior * 0.9:
                            result["score_trend"] = "下降"
            except Exception:
                pass

            # 3. 获取用户关注指数历史
            try:
                focus_df = ak.stock_comment_detail_scrd_focus_em(symbol=stock_code)
                if focus_df is not None and not focus_df.empty:
                    latest_focus = focus_df.iloc[-1].get("用户关注指数", 50.0)
                    result["attention_index"] = self._safe_float(latest_focus, result["attention_index"])

                    if len(focus_df) >= 10:
                        recent_f = focus_df.tail(5)["用户关注指数"].mean()
                        prior_f = focus_df.tail(10).head(5)["用户关注指数"].mean()
                        if recent_f > prior_f * 1.05:
                            result["attention_trend"] = "上升"
                        elif recent_f < prior_f * 0.95:
                            result["attention_trend"] = "下降"
            except Exception:
                pass

            # 4. 综合社交媒体情绪指数 [-1, 1]
            # 权重: 综合得分40% + 关注指数20% + 机构参与度20% + 参与意愿20%
            score_norm = (result["social_sentiment_score"] - 50) / 50  # [-1, 1]
            attention_norm = (result["attention_index"] - 50) / 50
            inst_norm = (result["institution_participation"] - 50) / 50
            will_norm = (result["participation_will"] - 50) / 50

            social_index = (
                score_norm * 0.40 +
                attention_norm * 0.20 +
                inst_norm * 0.20 +
                will_norm * 0.20
            )
            result["social_sentiment_index"] = round(max(-1.0, min(1.0, social_index)), 3)

            # 5. 异常情绪检测
            result["anomaly_flag"], result["anomaly_reason"] = self._detect_sentiment_anomaly(
                result, desire_df if 'desire_df' in dir() and desire_df is not None else None
            )

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"获取社交媒体情绪数据失败({stock_code}): {e}")
            return {"_error": str(e), **result}

    def _detect_sentiment_anomaly(self, result: Dict[str, Any], desire_df=None) -> tuple:
        """
        检测社交媒体情绪异常

        异常模式：
        1. 关注指数激增但价格未动 → 有人造势
        2. 参与意愿与机构参与度背离 → crowd vs 机构分歧
        3. 排名大幅上升但综合得分下降 → 流量炒作
        """
        anomaly = False
        reasons = []

        attention = result.get("attention_index", 50.0)
        participation = result.get("participation_will", 50.0)
        inst_part = result.get("institution_participation", 50.0)
        rank_change = result.get("rank_change", 0)
        score = result.get("social_sentiment_score", 50.0)

        # 模式1: 关注指数极高(>90)但机构参与度低(<30) → 散户热炒
        if attention > 90 and inst_part < 30:
            anomaly = True
            reasons.append("高关注低机构参与：散户热炒迹象")

        # 模式2: 排名大幅上升(>500名)但得分下降 → 流量炒作
        if rank_change > 500 and score < 50:
            anomaly = True
            reasons.append("排名飙升但评分低迷：疑似流量炒作")

        # 模式3: 参与意愿与机构参与度严重背离
        if participation > 70 and inst_part < 30:
            anomaly = True
            reasons.append("参与意愿高但机构参与度低：crowd情绪与机构背离")
        elif participation < 30 and inst_part > 70:
            anomaly = True
            reasons.append("机构参与度高但市场意愿低：机构暗中布局")

        # 模式4: 基于历史数据的突兀变化
        if desire_df is not None and len(desire_df) >= 20:
            recent_will = desire_df.tail(5)["参与意愿"].mean()
            prior_will = desire_df.tail(20).head(15)["参与意愿"].mean()
            if prior_will > 0 and recent_will > prior_will * 1.5:
                anomaly = True
                reasons.append(f"参与意愿短期激增{((recent_will/prior_will-1)*100):.0f}%：情绪异常放大")

        return anomaly, ";".join(reasons) if reasons else ""

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        """安全转换浮点数"""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # ════════════════════════════════════════════════════════════════════
    # 市场情绪 / 板块情绪 — 相对情绪对比基础
    # ════════════════════════════════════════════════════════════════════

    def fetch_market_and_sector_sentiment(
        self, stock_code: str, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取市场整体情绪和个股所属板块情绪

        数据来源：EastMoneyCrawler 的市场环境数据
        - 大盘指数（上证、深证、创业板）涨跌幅 → 市场情绪
        - 板块资金流向排行 → 板块情绪

        Args:
            stock_code: 6 位股票代码
            industry: 所属行业名称（可选，用于匹配板块）

        Returns:
            {
                "market_sentiment_index": float,   # [-1, 1] 大盘情绪
                "market_sentiment_score": float,   # [0, 100] 大盘情绪得分
                "market_trend": str,               # 上涨/下跌/震荡
                "market_up_down_ratio": float,     # 涨跌家数比
                "sector_sentiment_index": float,   # [-1, 1] 板块情绪
                "sector_sentiment_score": float,   # [0, 100] 板块情绪得分
                "sector_name": str,                # 匹配到的板块名
                "sector_rank": int,                # 板块排名
                "sector_main_inflow": float,       # 板块主力净流入
                "relative_individual_sector": float,       # 个股相对板块（预留，由SA-Agent计算）
                "_error": str,                     # 如有错误
            }
        """
        result = {
            "market_sentiment_index": 0.0,
            "market_sentiment_score": 50.0,
            "market_trend": "未知",
            "market_up_down_ratio": 1.0,
            "sector_sentiment_index": 0.0,
            "sector_sentiment_score": 50.0,
            "sector_name": industry or "未知",
            "sector_rank": -1,
            "sector_main_inflow": 0.0,
        }

        try:
            # 通过 CrawlerRegistry 获取市场环境数据
            market_data = self._registry.fetch_merge(stock_code, "market_context")
            if not market_data:
                return {"_error": "市场环境数据不可用", **result}

            # 1. 计算大盘情绪
            indices = market_data.get("indices", {})
            if indices:
                market_sent = self.compute_market_sentiment(indices)
                result.update(market_sent)

            # 2. 计算板块情绪
            sectors = market_data.get("sector_top", [])
            if sectors and industry:
                sector_sent = self.compute_sector_sentiment(sectors, industry)
                result.update(sector_sent)
            elif sectors:
                # 没有行业信息时，取涨幅最高的板块作为参考
                top_sector = max(sectors, key=lambda x: x.get("pct_change", 0))
                result["sector_sentiment_index"] = max(-1.0, min(1.0, top_sector.get("pct_change", 0) / 0.03))
                result["sector_sentiment_score"] = 50 + result["sector_sentiment_index"] * 50
                result["sector_name"] = top_sector.get("name", "未知")
                result["sector_main_inflow"] = top_sector.get("main_net_inflow", 0)

            return result

        except Exception as e:
            logger.warning(f"获取市场/板块情绪失败({stock_code}): {e}")
            return {"_error": str(e), **result}

    @staticmethod
    def compute_market_sentiment(indices: Dict[str, Any]) -> Dict[str, Any]:
        """
        从大盘指数数据计算市场情绪指数

        权重：上证40% + 深证30% + 创业板20% + 科创50/沪深300 10%
        """
        weights = {
            "上证指数": 0.40,
            "深证成指": 0.30,
            "创业板指": 0.20,
            "科创50": 0.05,
            "沪深300": 0.05,
        }

        weighted_pct = 0.0
        total_weight = 0.0
        up_count_total = 0
        down_count_total = 0

        for name, weight in weights.items():
            idx = indices.get(name)
            if idx and isinstance(idx, dict):
                pct = idx.get("pct_change", 0)
                weighted_pct += pct * weight
                total_weight += weight
                up_count_total += idx.get("up_count", 0)
                down_count_total += idx.get("down_count", 0)

        if total_weight == 0:
            return {
                "market_sentiment_index": 0.0,
                "market_sentiment_score": 50.0,
                "market_trend": "未知",
                "market_up_down_ratio": 1.0,
            }

        # 归一化到 [-1, 1]：3% 涨跌幅 = 情绪指数 1.0
        sentiment_index = max(-1.0, min(1.0, weighted_pct / 0.03))
        sentiment_score = 50 + sentiment_index * 50

        # 涨跌家数比
        total_stocks = up_count_total + down_count_total
        up_down_ratio = up_count_total / max(1, down_count_total)

        if weighted_pct > 0.01:
            trend = "上涨"
        elif weighted_pct < -0.01:
            trend = "下跌"
        else:
            trend = "震荡"

        return {
            "market_sentiment_index": round(sentiment_index, 3),
            "market_sentiment_score": round(sentiment_score, 1),
            "market_trend": trend,
            "market_up_down_ratio": round(up_down_ratio, 2),
        }

    @staticmethod
    def compute_sector_sentiment(sectors: List[Dict[str, Any]], industry: str) -> Dict[str, Any]:
        """
        从板块数据中匹配个股所属行业，计算板块情绪

        匹配逻辑：
        1. 精确匹配行业名
        2. 模糊匹配（包含关系）
        3. 取排名最前且名称相关的板块
        """
        if not sectors or not industry:
            return {
                "sector_sentiment_index": 0.0,
                "sector_sentiment_score": 50.0,
                "sector_name": industry or "未知",
                "sector_rank": -1,
                "sector_main_inflow": 0.0,
            }

        matched = None
        for i, sec in enumerate(sectors):
            sec_name = sec.get("name", "")
            # 精确匹配或包含
            if industry in sec_name or sec_name in industry:
                matched = sec
                matched["_rank"] = i + 1
                break

        if not matched:
            # 取涨幅最高的作为参考
            matched = max(sectors, key=lambda x: x.get("pct_change", 0))
            matched["_rank"] = -1

        pct = matched.get("pct_change", 0)
        sentiment_index = max(-1.0, min(1.0, pct / 0.04))  # 4% 归一化到 1.0
        sentiment_score = 50 + sentiment_index * 50

        return {
            "sector_sentiment_index": round(sentiment_index, 3),
            "sector_sentiment_score": round(sentiment_score, 1),
            "sector_name": matched.get("name", industry),
            "sector_rank": matched.get("_rank", -1),
            "sector_main_inflow": matched.get("main_net_inflow", 0),
        }

    # ── 核心分析引擎 ──

    def _enhanced_score(self, text: str) -> Tuple[float, str]:
        """
        增强规则情感打分 — 支持否定翻转、程度加权、转折句处理

        Returns:
            (score, explanation)
        """
        if not text:
            return 0.0, "空文本"

        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text]

        total_weighted_score = 0.0
        total_weight = 0.0
        explanations = []

        for sentence in sentences:
            s_score, s_weight, s_exp = self._score_sentence(sentence)
            if s_weight > 0:
                total_weighted_score += s_score * s_weight
                total_weight += s_weight
                if s_exp:
                    explanations.append(s_exp)

        if total_weight == 0:
            return 0.0, "未匹配到情感词"

        final_score = total_weighted_score / total_weight
        # 压缩到 [-1, 1]
        final_score = max(-1.0, min(1.0, final_score))
        return final_score, "; ".join(explanations[:3])

    def _score_sentence(self, sentence: str) -> Tuple[float, float, str]:
        """
        对单一句子打分

        Returns:
            (raw_score_sum, weight_sum, explanation)
        """
        # 检查是否有转折词 —— 转折后的内容权重翻倍
        transition_pos = -1
        for tword in self.TRANSITION_WORDS:
            pos = sentence.find(tword)
            if pos != -1:
                transition_pos = max(transition_pos, pos + len(tword))
                break

        pos_hits = []
        neg_hits = []

        # 正面词匹配
        for w in self.POSITIVE_WORDS:
            for m in re.finditer(re.escape(w), sentence):
                start = max(0, m.start() - self.NEGATION_WINDOW)
                prefix = sentence[start:m.start()]
                flipped = self._has_negation(prefix)
                intensifier = self._find_intensifier(prefix)
                weight = 1.0
                # 转折后权重翻倍
                if transition_pos != -1 and m.start() >= transition_pos:
                    weight *= 2.0
                pos_hits.append((m.start(), 1.0 if not flipped else -1.0, intensifier, weight, w, flipped))

        # 负面词匹配
        for w in self.NEGATIVE_WORDS:
            for m in re.finditer(re.escape(w), sentence):
                start = max(0, m.start() - self.NEGATION_WINDOW)
                prefix = sentence[start:m.start()]
                flipped = self._has_negation(prefix)
                intensifier = self._find_intensifier(prefix)
                weight = 1.0
                if transition_pos != -1 and m.start() >= transition_pos:
                    weight *= 2.0
                neg_hits.append((m.start(), -1.0 if not flipped else 1.0, intensifier, weight, w, flipped))

        # 去重：同一位置只保留最长匹配（避免"上涨"和"涨"重复计分）
        all_hits = sorted(pos_hits + neg_hits, key=lambda x: x[0])
        deduped = []
        last_end = -1
        for hit in all_hits:
            pos = hit[0]
            if pos >= last_end:
                deduped.append(hit)
                last_end = pos + len(hit[4])

        if not deduped:
            return 0.0, 0.0, ""

        score_sum = 0.0
        weight_sum = 0.0
        detail_parts = []
        for pos, polarity, intensifier, weight, word, flipped in deduped:
            adjusted = polarity * intensifier * weight
            score_sum += adjusted
            weight_sum += weight
            label = "正" if polarity > 0 else "负"
            if flipped:
                label += "(被否定翻转)"
            detail_parts.append(f"{word}({label},强度{intensifier:.2f})")

        return score_sum, weight_sum, ",".join(detail_parts)

    def _split_sentences(self, text: str) -> List[str]:
        """简单分句"""
        # 按常见中文标点分句
        parts = re.split(r'[。！？；\n]', text)
        return [p.strip() for p in parts if p.strip()]

    # 单字否定词常见的"非否定"嵌入组合（避免"非常"中的"非"被误判）
    _NEGATION_FALSE_POSITIVES = {
        '非': {'非常', '除非', '无非', '若非'},
        '无': {'无论', '无语', '无敌', '无限', '无线'},
    }

    def _has_negation(self, prefix: str) -> bool:
        """检查前缀中是否包含有效否定词（避免部分匹配如'非常'中的'非'）"""
        for neg in self.NEGATION_WORDS:
            idx = prefix.rfind(neg)
            if idx == -1:
                continue
            # 多字否定词直接生效
            if len(neg) > 1:
                return True
            # 单字否定词：检查是否属于常见非否定嵌入词
            false_pos = self._NEGATION_FALSE_POSITIVES.get(neg)
            if false_pos:
                # 检查前后文是否构成长度为2或3的假否定词
                for length in (2, 3):
                    substr = prefix[idx:idx + length]
                    if substr in false_pos:
                        break
                else:
                    return True
            else:
                return True
        return False

    def _find_intensifier(self, prefix: str) -> float:
        """查找前缀中的程度副词，返回强度系数"""
        best = 1.0
        for word, factor in self.INTENSIFIER_WORDS.items():
            if word in prefix:
                # 取最大的 intensifier
                if factor > best:
                    best = factor
        return best

    # ── LLM 回退/增强层（长期方案） ──

    def _analyze_with_llm(self, texts: List[str]) -> List[float]:
        """
        使用 DeepSeek V4 Pro 进行批量高精度情感分析。
        从逐条分析升级为批量综合摘要分析，充分利用 LLM 深度推理能力。
        """
        if not texts:
            return []

        try:
            from agent.tools.llm_client import LLMClient
            client = LLMClient()
        except Exception as e:
            logger.warning(f"LLM 情感分析初始化失败，回退到规则引擎: {e}")
            return [self._enhanced_score(t)[0] for t in texts]

        # ── 策略：先批量综合摘要，再逐条精细校准 ──
        # Step 1: 批量摘要分析（一次性送入所有新闻，让 LLM 做整体判断）
        batch_text = "\n".join([f"{i+1}. {t[:300]}" for i, t in enumerate(texts) if t.strip()])

        system_prompt = (
            "你是资深中文金融舆情分析专家，拥有10年A股市场研究经验。"
            "你的任务是对一组股票相关新闻进行深度情感分析。"
            "分析时必须考虑：否定词翻转、反讽语气、政策语境、行业周期、市场预期。"
        )

        user_prompt = (
            f"以下是一组股票相关新闻标题/摘要（共{len(texts)}条），请进行深度分析：\n\n"
            f"{batch_text}\n\n"
            "请输出 JSON 格式：\n"
            "{\n"
            '  "overall_sentiment": float,  // 整体情感倾向 [-1.0, 1.0]，-1=极度看空，0=中性，1=极度看多\n'
            '  "overall_confidence": float,  // 置信度 [0, 1]\n'
            '  "dominant_theme": str,        // 主导主题（如：业绩超预期、政策利好、行业景气等）\n'
            '  "market_expectation": str,    // 市场预期描述\n'
            '  "risk_signals": [str],        // 风险信号列表\n'
            '  "opportunity_signals": [str], // 机会信号列表\n'
            '  "per_item_scores": [          // 每条新闻的情感打分\n'
            '    {"index": int, "score": float, "reason": str}\n'
            '  ]\n'
            "}\n"
            "注意：\n"
            "1. 如果新闻中同时存在利好和利空，要根据权重和时效性综合判断\n"
            "2. 政策类新闻权重高于普通新闻\n"
            "3. 业绩类新闻要考虑是否超预期（不仅看绝对值）\n"
            "4. 如果所有新闻均为中性或无实质信息，overall_sentiment 应为 0.0\n"
        )

        try:
            result = client.chat(
                system=system_prompt,
                user=user_prompt,
                json_mode=True,
            )
            if isinstance(result, dict):
                overall = float(result.get("overall_sentiment", 0))
                per_item = result.get("per_item_scores", [])

                scores = []
                item_map = {}
                for item in per_item:
                    idx = item.get("index", 0)
                    item_map[idx] = float(item.get("score", overall))

                for i in range(len(texts)):
                    # 优先使用 LLM 逐条打分，缺失时用整体打分回退
                    score = item_map.get(i + 1, overall)
                    scores.append(max(-1.0, min(1.0, score)))
                return scores
        except Exception as e:
            logger.warning(f"LLM 批量情感分析失败，回退规则引擎: {e}")

        # 回退到规则引擎
        return [self._enhanced_score(t)[0] for t in texts]
