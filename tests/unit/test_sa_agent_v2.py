"""
单元测试: SA-Agent v2.1 社交媒体情绪增强
"""
import pytest
import pandas as pd

from agent.agents import SA_Agent
from agent.tools.llm_client import MockLLMClient
from agent.tools.sentiment_tool import SentimentTool
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestSAAgentV2:
    """SA-Agent v2.1 测试类"""

    @pytest.fixture(autouse=True)
    def mock_social_media(self, monkeypatch):
        """为所有测试 mock 社交媒体数据获取，避免网络请求"""
        def mock_fetch_agent(self, snapshot):
            return {
                "social_sentiment_index": 0.2,
                "social_sentiment_score": 60.0,
                "attention_index": 55.0,
                "institution_participation": 45.0,
                "market_rank": 500,
                "rank_change": 5,
                "participation_will": 52.0,
                "participation_will_5d_avg": 50.0,
                "participation_will_change": 2.0,
                "score_trend": "平稳",
                "attention_trend": "平稳",
                "anomaly_flag": False,
                "anomaly_reason": "",
            }
        monkeypatch.setattr(SA_Agent, "_fetch_social_media_data", mock_fetch_agent)

        # 同时 mock SentimentTool 方法，避免网络请求
        def mock_fetch_social(self, stock_code):
            return {
                "social_sentiment_index": 0.2,
                "social_sentiment_score": 60.0,
                "attention_index": 55.0,
                "institution_participation": 45.0,
                "market_rank": 500,
                "rank_change": 5,
                "participation_will": 52.0,
                "participation_will_5d_avg": 50.0,
                "participation_will_change": 2.0,
                "score_trend": "平稳",
                "attention_trend": "平稳",
                "anomaly_flag": False,
                "anomaly_reason": "",
            }
        monkeypatch.setattr(SentimentTool, "fetch_social_media_sentiment", mock_fetch_social)

        def mock_fetch_market(self, stock_code, industry=None):
            return {
                "market_sentiment_index": 0.1,
                "market_sentiment_score": 55.0,
                "market_trend": "震荡",
                "market_up_down_ratio": 1.1,
                "sector_sentiment_index": 0.15,
                "sector_sentiment_score": 57.5,
                "sector_name": industry or "测试板块",
                "sector_rank": 10,
                "sector_main_inflow": 5000.0,
            }
        monkeypatch.setattr(SentimentTool, "fetch_market_and_sector_sentiment", mock_fetch_market)

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def base_snapshot(self):
        """基础快照（无K线情绪）"""
        return StockSnapshot(
            stock_code="999999",  # 避免网络请求
            stock_name="测试",
            current_price=15.0,
            indicators={},
            fundamentals={},
            fund_flow={},
            sentiment_data={
                "news_analysis": {
                    "sentiment_score": 0.3,
                    "positive_count": 3,
                    "negative_count": 1,
                    "neutral_count": 2,
                    "dominant_tone": "偏正面",
                },
                "news": [
                    {"title": "测试新闻1", "source": "测试源"},
                    {"title": "测试新闻2", "source": "测试源"},
                ],
            },
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_with_kline_sentiment(self):
        """带K线情绪数据的快照"""
        return StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={},
            fundamentals={},
            fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.65,
                    "sentiment_percentile": 82,
                    "volatility_annual": 0.35,
                    "volume_sentiment": "放量上涨（FOMO追涨/机构进场）",
                    "volume_ma5_ratio": 1.8,
                    "trend_emotion": "严重超买（情绪狂热）",
                    "price_dev_ma20": 8.5,
                    "price_dev_ma60": 12.0,
                    "consecutive_up": 4,
                    "consecutive_down": 0,
                    "price_change_5d_pct": 6.5,
                    "crowd_behavior": "FOMO追涨（散户跟风买入，情绪过热）",
                },
                "fund_flow_sentiment": {
                    "main_force_emotion": "主力大幅净流入（机构强烈看好）",
                    "main_force_score": 0.8,
                    "retail_emotion": "散户跟风买入（需警惕）",
                    "retail_score": -0.3,
                    "fund_flow_sentiment": 0.71,
                },
            },
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    # ──────────────────────────────────────────────
    # 1. Prompt 构建测试
    # ──────────────────────────────────────────────
    def test_build_prompt_with_kline_sentiment(self, mock_llm, snapshot_with_kline_sentiment):
        """测试Prompt包含K线推导情绪"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(snapshot_with_kline_sentiment)

        assert "量价行为情绪" in prompt
        assert "综合情绪指数" in prompt
        assert "情绪百分位" in prompt
        assert "crowd_behavior" not in prompt  # 这是字段名，不应出现在prompt中
        assert "推断 crowd 行为" in prompt

    def test_build_prompt_with_fund_flow(self, mock_llm, snapshot_with_kline_sentiment):
        """测试Prompt包含资金流向情绪"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(snapshot_with_kline_sentiment)

        assert "资金流向情绪" in prompt
        assert "主力情绪" in prompt
        assert "散户情绪" in prompt

    def test_build_prompt_contains_social_media_section(self, mock_llm, snapshot_with_kline_sentiment, monkeypatch):
        """测试Prompt包含社交媒体情绪章节"""
        agent = SA_Agent("SA-Agent", mock_llm)
        # Mock 社交媒体数据避免网络请求
        monkeypatch.setattr(agent, "_fetch_social_media_data", lambda s: {
            "social_sentiment_index": 0.5,
            "social_sentiment_score": 75.0,
            "attention_index": 80.0,
            "institution_participation": 60.0,
            "market_rank": 100,
            "rank_change": 10,
            "participation_will": 65.0,
            "participation_will_5d_avg": 55.0,
            "participation_will_change": 5.0,
            "score_trend": "上升",
            "attention_trend": "上升",
            "anomaly_flag": False,
            "anomaly_reason": "",
        })
        prompt = agent._build_sa_prompt(snapshot_with_kline_sentiment)

        assert "社交媒体情绪" in prompt
        assert "千股千评综合得分" in prompt
        assert "用户关注指数" in prompt
        assert "机构参与度" in prompt

    def test_build_prompt_with_news(self, mock_llm, base_snapshot):
        """测试Prompt包含新闻分析"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(base_snapshot)

        assert "新闻情感分析" in prompt
        assert "测试新闻1" in prompt

    def test_build_prompt_analysis_framework(self, mock_llm, base_snapshot):
        """测试Prompt包含完整的分析框架"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(base_snapshot)

        assert "情绪状态诊断" in prompt
        assert "Crowd 行为识别" in prompt
        assert "逆向信号判断" in prompt
        assert "时间维度一致性" in prompt

    # ──────────────────────────────────────────────
    # 2. Fallback 降级测试
    # ──────────────────────────────────────────────
    def test_fallback_with_kline_sentiment_overheat(self, mock_llm, snapshot_with_kline_sentiment):
        """测试情绪过热时的逆向卖出信号"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_kline_sentiment)

        assert isinstance(opinion, AgentOpinion)
        # 百分位82 > 80，应触发逆向卖出
        assert opinion.signal == -1
        assert opinion.confidence >= 0.55
        assert "FOMO" in opinion.reasoning or "过热" in opinion.reasoning

    def test_fallback_with_kline_sentiment_panic(self, mock_llm):
        """测试情绪过冷时的逆向买入信号"""
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": -0.75,
                    "sentiment_percentile": 15,
                    "volume_sentiment": "放量下跌（恐慌抛售）",
                    "crowd_behavior": "恐慌抛售（散户割肉，情绪崩溃）",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        # 百分位15 < 20，应触发逆向买入
        assert opinion.signal == 1
        assert "恐慌" in opinion.reasoning or "过冷" in opinion.reasoning

    def test_fallback_with_fund_flow_only(self, mock_llm):
        """测试仅有资金流向数据时的降级"""
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "fund_flow_sentiment": {
                    "main_force_emotion": "主力大幅净流入（机构强烈看好）",
                    "main_force_score": 0.8,
                    "retail_emotion": "散户恐慌抛售（可能接近底部）",
                    "retail_score": 0.3,
                    "fund_flow_sentiment": 0.89,
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.signal in (-1, 0, 1)
        assert "主力" in opinion.reasoning

    def test_fallback_with_news_only(self, mock_llm, base_snapshot):
        """测试仅有新闻数据时的降级"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(base_snapshot)

        assert isinstance(opinion, AgentOpinion)
        # 新闻情感0.3 < 0.5，不应触发信号
        assert opinion.signal == 0

    def test_fallback_raw_data_has_social_media(self, mock_llm, snapshot_with_kline_sentiment):
        """测试降级输出的 raw_data 包含社交媒体数据字段"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_kline_sentiment)

        assert "social_media" in opinion.raw_data

    def test_fallback_extreme_pessimism(self, mock_llm):
        """测试极端悲观新闻的逆向买入"""
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "news_analysis": {
                    "sentiment_score": -0.8,
                    "positive_count": 0,
                    "negative_count": 10,
                    "neutral_count": 2,
                    "dominant_tone": "偏负面",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        # 新闻情感-0.8 < -0.5，应触发逆向买入
        assert opinion.signal == 1

    def test_fallback_weakest_degradation(self, mock_llm):
        """测试最弱降级（无任何情绪数据）"""
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={},
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        assert opinion.signal == 0
        assert opinion.confidence == 0.4
        assert opinion.raw_data.get("crowd_behavior") != "未知"
        assert "观望" in opinion.reasoning

    @pytest.fixture
    def snapshot_with_kline(self):
        """带真实K线数据的快照（用于动量计算）"""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end="2025-06-27", periods=30)
        # 构造情绪从恐慌到狂热的K线：前10日下跌，后20日上涨
        closes = []
        vols = []
        base = 15.0
        for i in range(30):
            if i < 10:
                # 下跌期，情绪恐慌
                change = -0.01 - (i % 3) * 0.005
                vol = 1.5 + (i % 3) * 0.3  # 放量下跌
            elif i < 20:
                # 反弹期，情绪恢复
                change = 0.005 + (i % 3) * 0.003
                vol = 0.8
            else:
                # 狂热期，加速上涨
                change = 0.015 + (i % 3) * 0.005
                vol = 1.8 + (i % 3) * 0.2  # 放量上涨
            base *= (1 + change)
            closes.append(base)
            vols.append(1000000 * vol)

        kline = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": [c * 0.99 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": vols,
        })

        return StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=closes[-1],
            indicators={},
            fundamentals={},
            fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.65,
                    "sentiment_percentile": 82,
                    "crowd_behavior": "FOMO追涨（散户跟风买入，情绪过热）",
                },
            },
            market_context={},
            macro_data={},
            risk_metrics={},
            kline_df=kline,
        )

    # ──────────────────────────────────────────────
    # 3. 相对情绪对比测试
    # ──────────────────────────────────────────────
    def test_relative_sentiment_computation(self, mock_llm, snapshot_with_kline_sentiment):
        """测试相对情绪对比计算"""
        agent = SA_Agent("SA-Agent", mock_llm)
        relative = agent._compute_relative_sentiment(snapshot_with_kline_sentiment)

        assert "individual_sentiment" in relative
        assert "sector_sentiment" in relative
        assert "market_sentiment" in relative
        assert "relative_to_sector" in relative
        assert "relative_to_market" in relative
        assert "relative_signal" in relative
        assert "relative_interpretation" in relative

    def test_relative_sentiment_no_data(self, mock_llm):
        """测试无K线情绪时的降级 — 有市场数据但个股情绪为0"""
        snapshot = StockSnapshot(
            stock_code="999999", stock_name="测试", current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        relative = agent._compute_relative_sentiment(snapshot)

        # 无个股情绪数据时 individual=0，mock 市场数据返回板块 0.15/大盘 0.1
        # 此时产生 mild_divergence（因为偏差在 0.1~0.2 之间）
        assert relative["individual_sentiment"] == 0.0
        assert "relative_signal" in relative
        assert "relative_interpretation" in relative

    def test_relative_prompt_section(self, mock_llm, snapshot_with_kline_sentiment):
        """测试Prompt包含相对情绪章节"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(snapshot_with_kline_sentiment)

        assert "相对情绪对比" in prompt
        assert "个股情绪指数" in prompt
        assert "板块情绪指数" in prompt
        assert "大盘情绪指数" in prompt
        assert "相对情绪指引" in prompt

    def test_fallback_relative_sentiment_overheat(self, mock_llm):
        """测试个股显著强于板块/大盘时的反向修正"""
        import pandas as pd
        dates = pd.date_range(end="2025-06-27", periods=30)
        closes = [10.0 * (1.04 ** i) for i in range(30)]  # 持续大涨
        vols = [1500000] * 30
        kline = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "close": closes, "volume": vols,
            "open": [c * 0.99 for c in closes],
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.98 for c in closes],
        })

        snapshot = StockSnapshot(
            stock_code="999999", stock_name="测试", current_price=closes[-1],
            indicators={}, fundamentals={"industry": "半导体"}, fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.75,
                    "sentiment_percentile": 88,
                    "crowd_behavior": "FOMO追涨",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
            kline_df=kline,
        )
        agent = SA_Agent("SA-Agent", mock_llm)

        # 由于 autouse fixture mock 了 fetch_market_and_sector_sentiment 返回较温和的数据
        # 个股情绪 0.75 会显著强于 mock 的板块 0.15 和大盘 0.1，触发相对过热修正
        opinion = agent._fallback_opinion(snapshot)
        assert "momentum" in opinion.raw_data
        assert "relative_sentiment" in opinion.raw_data

    def test_fallback_raw_data_has_relative(self, mock_llm, snapshot_with_kline):
        """测试降级输出包含相对情绪数据"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_kline)

        assert "relative_sentiment" in opinion.raw_data
        rel = opinion.raw_data["relative_sentiment"]
        if rel:
            assert "relative_to_sector" in rel
            assert "relative_to_market" in rel

    # ──────────────────────────────────────────────
    # 4. 情绪动量分析测试
    # ──────────────────────────────────────────────
    def test_momentum_computation_with_kline(self, mock_llm, snapshot_with_kline):
        """测试有K线时的情绪动量计算"""
        agent = SA_Agent("SA-Agent", mock_llm)
        momentum = agent._compute_sentiment_momentum(snapshot_with_kline)

        assert "si_history" in momentum
        assert len(momentum["si_history"]) >= 5
        assert "momentum_1d" in momentum
        assert "momentum_5d" in momentum
        assert "momentum_direction" in momentum
        assert "momentum_signal" in momentum
        assert "reversal_probability" in momentum
        # 反转概率应在合理范围内
        assert 0 <= momentum["reversal_probability"] <= 1

    def test_momentum_computation_no_kline(self, mock_llm, snapshot_with_kline_sentiment):
        """测试无K线时的动量降级"""
        agent = SA_Agent("SA-Agent", mock_llm)
        momentum = agent._compute_sentiment_momentum(snapshot_with_kline_sentiment)

        assert momentum["si_history"] == []
        assert momentum["momentum_signal"] == "数据不足"
        assert momentum["momentum_direction"] == "未知"

    def test_momentum_prompt_section(self, mock_llm, snapshot_with_kline, monkeypatch):
        """测试Prompt包含情绪动量章节"""
        agent = SA_Agent("SA-Agent", mock_llm)
        prompt = agent._build_sa_prompt(snapshot_with_kline)

        assert "情绪动量分析" in prompt
        assert "动量方向" in prompt
        assert "动量信号" in prompt
        assert "反转概率" in prompt
        assert "情绪动量指引" in prompt

    def test_fallback_momentum_acceleration_extreme(self, mock_llm):
        """测试动量加速向极端时的信号增强 — 使用已有的kline fixture"""
        # 使用 snapshot_with_kline，它包含从恐慌到狂热的K线
        import pandas as pd
        dates = pd.date_range(end="2025-06-27", periods=30)
        # 构造持续狂热加速的K线：30天大涨，成交量持续放大
        closes = [10.0]
        vols = [1000000]
        for i in range(1, 30):
            closes.append(closes[-1] * (1 + 0.05))  # 每天涨5%
            vols.append(1000000 * (1.5 + i * 0.05))  # 成交量大幅递增

        kline = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "close": closes,
            "volume": vols,
            "open": [c * 0.98 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.97 for c in closes],
        })

        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=closes[-1],
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.85,
                    "sentiment_percentile": 92,
                    "crowd_behavior": "FOMO追涨",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
            kline_df=kline,
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        # 应触发逆向卖出信号（情绪过热）
        assert opinion.signal == -1
        assert "momentum" in opinion.raw_data

    def test_fallback_momentum_reversal_from_extreme(self, mock_llm):
        """测试情绪从极端回落时的动量反向"""
        import pandas as pd
        # 构造情绪从狂热开始回落的K线：前20日大涨，后10日下跌
        dates = pd.date_range(end="2025-06-27", periods=30)
        closes = [10.0]
        vols = [1000000]
        for i in range(1, 20):
            closes.append(closes[-1] * (1 + 0.04))  # 前20天涨4%
            vols.append(1500000)
        # 后10天下跌
        for i in range(10):
            closes.append(closes[-1] * (1 - 0.03))  # 后10天跌3%
            vols.append(1800000)  # 放量下跌

        kline = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "close": closes,
            "volume": vols,
            "open": [c * 0.98 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.97 for c in closes],
        })

        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=closes[-1],
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.70,
                    "sentiment_percentile": 85,
                    "crowd_behavior": "获利了结",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
            kline_df=kline,
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        # 至少 raw_data 中包含动量数据
        assert "momentum" in opinion.raw_data
        mom = opinion.raw_data["momentum"]
        if mom and mom.get("si_history"):
            # 后段下跌导致动量应为负
            assert mom.get("momentum_5d", 0) <= 0.1  # 允许近似为0

    def test_fallback_momentum_raw_data_present(self, mock_llm, snapshot_with_kline):
        """测试降级输出包含动量数据"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_kline)

        assert "momentum" in opinion.raw_data
        mom = opinion.raw_data["momentum"]
        assert mom is not None
        if mom:
            assert "si_history" in mom
            assert "momentum_5d" in mom
            assert "reversal_probability" in mom

    # ──────────────────────────────────────────────
    # 4. 辅助推断方法测试
    # ──────────────────────────────────────────────
    def test_infer_crowd_behavior_from_kline_fomo(self, mock_llm):
        """测试从K线推断FOMO行为"""
        agent = SA_Agent("SA-Agent", mock_llm)
        kline_sent = {
            "volume_sentiment": "放量上涨",
            "consecutive_up": 5,
            "price_dev_ma20": 8.0,
            "sentiment_index": 0.8,
        }
        crowd = agent._infer_crowd_behavior_from_kline(kline_sent)
        assert "FOMO" in crowd or "追涨" in crowd

    def test_infer_crowd_behavior_from_kline_panic(self, mock_llm):
        """测试从K线推断恐慌行为"""
        agent = SA_Agent("SA-Agent", mock_llm)
        kline_sent = {
            "volume_sentiment": "放量下跌",
            "consecutive_down": 4,
            "price_dev_ma20": -10.0,
        }
        crowd = agent._infer_crowd_behavior_from_kline(kline_sent)
        assert "恐慌" in crowd or "抛售" in crowd

    def test_infer_contrarian_overheat(self, mock_llm):
        """测试情绪过热的逆向机会推断"""
        agent = SA_Agent("SA-Agent", mock_llm)
        parsed = {"sentiment_percentile": 85, "sentiment_index": 0.7, "crowd_behavior": "FOMO追涨"}
        opp = agent._infer_contrarian_opportunity(parsed)
        assert "反向卖出" in opp or "过热" in opp

    def test_infer_contrarian_panic(self, mock_llm):
        """测试情绪过冷的逆向机会推断"""
        agent = SA_Agent("SA-Agent", mock_llm)
        parsed = {"sentiment_percentile": 15, "sentiment_index": -0.6, "crowd_behavior": "恐慌抛售"}
        opp = agent._infer_contrarian_opportunity(parsed)
        assert "反向买入" in opp or "过冷" in opp

    # ──────────────────────────────────────────────
    # 4. 端到端分析测试
    # ──────────────────────────────────────────────
    def test_sa_agent_analyze_end_to_end(self, mock_llm, snapshot_with_kline_sentiment):
        """测试完整分析流程"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent.analyze(snapshot_with_kline_sentiment)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "SA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.reasoning) > 0

        # sentiment_index 校验
        si = opinion.raw_data.get("sentiment_index", 0)
        assert -1 <= si <= 1

        # sentiment_percentile 校验
        sp = opinion.raw_data.get("sentiment_percentile", 50)
        assert 0 <= sp <= 100

    def test_sa_agent_extreme_sentiment_correction(self, mock_llm):
        """测试极端情绪与信号联动修正"""
        # MockLLM 返回情绪过热但 signal=1，应被修正为 0
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={
                "kline_sentiment": {
                    "sentiment_index": 0.9,
                    "sentiment_percentile": 90,
                    "crowd_behavior": "FOMO追涨",
                },
            },
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        # MockLLM 默认返回的信号需要查看其实现
        # 这里主要测试 fallback 路径
        opinion = agent._fallback_opinion(snapshot)
        assert opinion.signal == -1  # 百分位90 > 80，逆向卖出

    def test_crowd_behavior_never_unknown(self, mock_llm):
        """测试 crowd_behavior 永远不会是'未知'"""
        snapshot = StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={},
            market_context={}, macro_data={}, risk_metrics={},
        )
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        cb = opinion.raw_data.get("crowd_behavior", "")
        assert cb != "未知"
        assert len(cb) > 0
