"""
单元测试: CA-Agent v2.1 资金面分析增强
覆盖: 筹码分布、机构行为追踪、融资融券深度分析
"""
import pytest
import pandas as pd

from agent.agents import CA_Agent
from agent.tools.llm_client import MockLLMClient
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestCAAgentV2:
    """CA-Agent v2.1 测试类"""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def base_snapshot(self):
        """基础快照（无K线）"""
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            indicators={},
            fundamentals={
                "float_market_cap": 500_0000_0000,  # 500亿流通市值
            },
            fund_flow={
                "main_net_inflow_10d": 10000,
                "main_inflow_days": 7,
                "margin_balance_change": 2000,
                "daily_flow": [
                    {"date": "2025-06-20", "main_net_inflow": 1000, "retail_net_inflow": -500},
                    {"date": "2025-06-21", "main_net_inflow": 1200, "retail_net_inflow": -600},
                ],
            },
            sentiment_data={},
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_with_kline(self):
        """带K线的快照（用于筹码分析和融资融券比率计算）"""
        dates = pd.date_range(end="2025-06-27", periods=60)
        kline = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": [14.5 + i * 0.01 for i in range(60)],
            "high": [14.6 + i * 0.01 for i in range(60)],
            "low": [14.4 + i * 0.01 for i in range(60)],
            "close": [14.5 + i * 0.01 for i in range(60)],
            "volume": [1000000 + (i % 10) * 50000 for i in range(60)],
            "amount": [14500000 + i * 10000 for i in range(60)],
            "turnover": [2.0 + (i % 5) * 0.1 for i in range(60)],
        })
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            indicators={},
            fundamentals={
                "float_market_cap": 500_0000_0000,
            },
            fund_flow={
                "main_net_inflow_10d": 10000,
                "main_inflow_days": 7,
                "margin_balance_change": 2000,
            },
            sentiment_data={},
            market_context={},
            macro_data={},
            risk_metrics={},
            kline_df=kline,
        )

    # ──────────────────────────────────────────────
    # 1. 筹码分布分析测试
    # ──────────────────────────────────────────────
    def test_chip_distribution_with_kline(self, mock_llm, snapshot_with_kline):
        """测试有K线时的筹码分布计算"""
        agent = CA_Agent("CA-Agent", mock_llm)
        chip = agent._analyze_chip_distribution(snapshot_with_kline)

        assert "cr90" in chip
        assert "main_cost_center" in chip
        assert "profit_ratio" in chip
        assert "lock_ratio" in chip
        assert "chip_status" in chip

        # CR90 应该在合理范围内
        assert 0 <= chip["cr90"] <= 100
        # 主力成本中心应该接近价格范围
        assert 10 <= chip["main_cost_center"] <= 20
        # 获利盘比例在 0-100 之间
        assert 0 <= chip["profit_ratio"] <= 100

    def test_chip_distribution_no_kline(self, mock_llm, base_snapshot):
        """测试无K线时的降级处理"""
        agent = CA_Agent("CA-Agent", mock_llm)
        chip = agent._analyze_chip_distribution(base_snapshot)

        assert chip["chip_status"] == "数据不足"
        assert chip["cr90"] == 50.0  # 默认值

    def test_chip_distribution_short_kline(self, mock_llm):
        """测试K线数据不足30条时的降级"""
        kline = pd.DataFrame({
            "close": [15.0] * 20,
            "volume": [1000000] * 20,
        })
        snapshot = StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
            kline_df=kline,
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        chip = agent._analyze_chip_distribution(snapshot)
        assert chip["chip_status"] == "数据不足"

    # ──────────────────────────────────────────────
    # 2. 机构行为分析测试
    # ──────────────────────────────────────────────
    def test_institution_behavior_no_data(self, mock_llm):
        """测试无机构数据时的降级处理 — 使用不存在的股票代码"""
        snapshot = StockSnapshot(
            stock_code="999999",  # 不存在的代码
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        inst = agent._analyze_institution_behavior(snapshot)

        assert inst["inst_signal"] == "neutral"
        assert inst["inst_score"] == 0
        assert inst["inst_hold"] is None
        assert inst["fund_hold"] is None

    # ──────────────────────────────────────────────
    # 3. 融资融券深度分析测试
    # ──────────────────────────────────────────────
    def test_margin_depth_with_kline(self, mock_llm, snapshot_with_kline):
        """测试有K线和基本面数据时的融资融券分析"""
        agent = CA_Agent("CA-Agent", mock_llm)
        margin = agent._analyze_margin_depth(snapshot_with_kline)

        # 没有真实融资融券数据，应该返回默认值
        assert "margin_detail" in margin
        assert "leveraged_trend" in margin
        assert "short_pressure" in margin
        assert "margin_score" in margin

    def test_margin_depth_no_data(self, mock_llm):
        """测试无融资融券数据时的降级 — 使用不存在的股票代码"""
        snapshot = StockSnapshot(
            stock_code="999999",  # 不存在的代码
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={},
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        margin = agent._analyze_margin_depth(snapshot)

        assert margin["margin_detail"] is None
        assert margin["leveraged_trend"] == "unknown"
        assert margin["margin_score"] == 0

    # ──────────────────────────────────────────────
    # 4. Prompt 构建测试
    # ──────────────────────────────────────────────
    def test_build_prompt_contains_chip(self, mock_llm, snapshot_with_kline):
        """测试Prompt包含筹码分布内容"""
        agent = CA_Agent("CA-Agent", mock_llm)
        prompt = agent._build_ca_prompt(snapshot_with_kline)

        assert "筹码集中度(CR90)" in prompt
        assert "主力成本区" in prompt
        assert "获利盘比例" in prompt

    def test_build_prompt_contains_institution(self, mock_llm, base_snapshot):
        """测试Prompt包含机构行为内容"""
        agent = CA_Agent("CA-Agent", mock_llm)
        prompt = agent._build_ca_prompt(base_snapshot)

        assert "机构行为追踪" in prompt
        assert "机构行为指引" in prompt

    def test_build_prompt_contains_margin(self, mock_llm, base_snapshot):
        """测试Prompt包含融资融券内容"""
        agent = CA_Agent("CA-Agent", mock_llm)
        prompt = agent._build_ca_prompt(base_snapshot)

        assert "融资融券深度分析" in prompt
        assert "融资融券指引" in prompt

    def test_build_prompt_contains_northbound(self, mock_llm, base_snapshot):
        """测试Prompt包含北向资金内容 — 需提供 north_bound_detail"""
        snapshot = StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            indicators={}, fundamentals={},
            fund_flow={
                "main_net_inflow_10d": 10000,
                "main_inflow_days": 7,
                "north_bound_detail": [
                    {"date": "2025-06-24", "net": 5000},
                    {"date": "2025-06-25", "net": 3000},
                ],
            },
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        prompt = agent._build_ca_prompt(snapshot)

        assert "北向资金行为分析" in prompt
        assert "北向资金指引" in prompt

    # ──────────────────────────────────────────────
    # 5. Fallback 降级测试
    # ──────────────────────────────────────────────
    def test_fallback_with_main_flow(self, mock_llm):
        """测试主力流入时的降级信号 — 隔离其他维度影响"""
        snapshot = StockSnapshot(
            stock_code="999999",  # 避免获取真实融资融券/机构数据
            stock_name="测试",
            current_price=15.0,
            indicators={}, fundamentals={},
            fund_flow={
                "main_net_inflow_10d": 15000,
                "main_inflow_days": 7,
            },
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.signal == 1  # 主力流入>10000且天数>=5
        assert opinion.confidence >= 0.5
        assert "主力资金持续流入" in opinion.reasoning

    def test_fallback_with_outflow(self, mock_llm):
        """测试主力流出时的降级信号"""
        snapshot = StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=15.0,
            indicators={}, fundamentals={}, fund_flow={
                "main_net_inflow_10d": -15000,
                "main_inflow_days": 2,
            },
            sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
        )
        agent = CA_Agent("CA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot)

        assert opinion.signal == -1
        assert "主力资金持续流出" in opinion.reasoning

    def test_fallback_raw_data_structure(self, mock_llm, snapshot_with_kline):
        """测试降级输出的 raw_data 结构完整性"""
        agent = CA_Agent("CA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_kline)

        assert opinion.raw_data is not None
        assert "capital_score" in opinion.raw_data
        assert "smart_money_direction" in opinion.raw_data
        assert "chip" in opinion.raw_data  # 筹码数据应存在

    # ──────────────────────────────────────────────
    # 6. 端到端分析测试
    # ──────────────────────────────────────────────
    def test_ca_agent_analyze_end_to_end(self, mock_llm, snapshot_with_kline):
        """测试完整分析流程"""
        agent = CA_Agent("CA-Agent", mock_llm)
        opinion = agent.analyze(snapshot_with_kline)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "CA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.reasoning) > 0

        # 校验 smart_money_direction 枚举
        valid = ["强烈建仓", "建仓期", "观望", "派发期", "强烈派发"]
        smd = opinion.raw_data.get("smart_money_direction", "观望")
        assert smd in valid
