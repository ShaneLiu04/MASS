"""
单元测试: MA-Agent v2.1 行业景气度周期定位
"""
import pytest

from agent.agents import MA_Agent
from agent.tools.llm_client import MockLLMClient
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestMAAgentV2:
    """MA-Agent v2.1 测试类"""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def snapshot_with_quarters(self):
        """带季度数据的快照（盈利上升趋势 = 复苏期）"""
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            indicators={},
            fundamentals={
                "industry": "银行",
                "quarterly_data": [
                    {"quarter": "2023Q1", "revenue": 40, "net_profit": 6, "roe": 10, "gross_margin": 30},
                    {"quarter": "2023Q2", "revenue": 42, "net_profit": 6.5, "roe": 11, "gross_margin": 31},
                    {"quarter": "2023Q3", "revenue": 45, "net_profit": 7.2, "roe": 12, "gross_margin": 32},
                    {"quarter": "2023Q4", "revenue": 48, "net_profit": 8.0, "roe": 13, "gross_margin": 33},
                    {"quarter": "2024Q1", "revenue": 52, "net_profit": 8.8, "roe": 14, "gross_margin": 34},
                ],
            },
            fund_flow={},
            sentiment_data={},
            market_context={
                "indices": {
                    "上证指数": {"close": 3000, "pct_change": 0.01, "up_count": 2000, "down_count": 1500},
                },
                "sector_top": [
                    {"name": "银行", "pct_change": 0.02, "main_net_inflow": 2e8},
                    {"name": "半导体", "pct_change": 0.05, "main_net_inflow": 1e8},
                ],
            },
            macro_data={
                "pmi": 51.5,
                "policy_stance": "宽松",
                "bond_yield_10y": 2.5,
                "bond_yield_trend": "下行",
            },
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_declining(self):
        """盈利下滑的快照（衰退期）"""
        return StockSnapshot(
            stock_code="000002",
            stock_name="测试衰退",
            current_price=10.0,
            indicators={},
            fundamentals={
                "industry": "房地产",
                "quarterly_data": [
                    {"quarter": "2023Q1", "revenue": 100, "net_profit": 20, "gross_margin": 40},
                    {"quarter": "2023Q2", "revenue": 95, "net_profit": 17, "gross_margin": 37},
                    {"quarter": "2023Q3", "revenue": 88, "net_profit": 14, "gross_margin": 33},
                    {"quarter": "2023Q4", "revenue": 80, "net_profit": 10, "gross_margin": 28},
                ],
            },
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_no_quarters(self):
        """无季度数据的快照"""
        return StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={},
            fundamentals={},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    # ──────────────────────────────────────────────
    # 1. 行业景气度周期分析测试
    # ──────────────────────────────────────────────
    def test_industry_cycle_recovery(self, mock_llm, snapshot_with_quarters):
        """测试盈利上升 = 复苏期"""
        agent = MA_Agent("MA-Agent", mock_llm)
        cycle = agent._analyze_industry_cycle(snapshot_with_quarters)

        assert cycle["stage"] in ("复苏期", "繁荣期", "复苏期/繁荣期")
        assert cycle["profit_trend"] == "上升"
        assert cycle["revenue_trend"] == "上升"
        assert cycle["margin_trend"] == "上升"
        assert cycle["composite_score"] > 50
        assert cycle["cycle_phase_num"] in (2, 3)

    def test_industry_cycle_recession(self, mock_llm, snapshot_declining):
        """测试盈利下滑 = 衰退期"""
        agent = MA_Agent("MA-Agent", mock_llm)
        cycle = agent._analyze_industry_cycle(snapshot_declining)

        assert "衰退" in cycle["stage"]
        assert cycle["profit_trend"] == "下降"
        assert cycle["composite_score"] < 50
        assert cycle["cycle_phase_num"] == 4

    def test_industry_cycle_no_data(self, mock_llm, snapshot_no_quarters):
        """测试无季度数据时的降级"""
        agent = MA_Agent("MA-Agent", mock_llm)
        cycle = agent._analyze_industry_cycle(snapshot_no_quarters)

        assert cycle["stage"] == "数据不足"
        assert cycle["composite_score"] == 50

    def test_industry_cycle_with_roe_fallback(self, mock_llm):
        """测试无毛利率但有ROE时的降级处理"""
        snapshot = StockSnapshot(
            stock_code="000003", stock_name="测试ROE", current_price=15.0,
            indicators={}, fund_flow={}, sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
            fundamentals={
                "industry": "电力",
                "quarterly_data": [
                    {"quarter": "2023Q1", "revenue": 50, "net_profit": 8, "roe": 10},
                    {"quarter": "2023Q2", "revenue": 55, "net_profit": 9, "roe": 11},
                    {"quarter": "2023Q3", "revenue": 60, "net_profit": 10, "roe": 12},
                ],
            },
        )
        agent = MA_Agent("MA-Agent", mock_llm)
        cycle = agent._analyze_industry_cycle(snapshot)

        # ROE 被用作毛利率的代理
        assert cycle["margin_trend"] != "未知"
        assert cycle["stage"] != "数据不足"

    # ──────────────────────────────────────────────
    # 2. Prompt 构建测试
    # ──────────────────────────────────────────────
    def test_build_prompt_contains_industry_cycle(self, mock_llm, snapshot_with_quarters):
        """测试Prompt包含行业景气度章节"""
        agent = MA_Agent("MA-Agent", mock_llm)
        prompt = agent._build_ma_prompt(snapshot_with_quarters)

        assert "行业景气度周期定位" in prompt
        assert "周期综合评分" in prompt
        assert "周期定位规则" in prompt

    def test_build_prompt_no_quarters(self, mock_llm, snapshot_no_quarters):
        """测试无季度数据时Prompt显示数据不足"""
        agent = MA_Agent("MA-Agent", mock_llm)
        prompt = agent._build_ma_prompt(snapshot_no_quarters)

        # Prompt 中应包含行业周期定位参考，但实际数据为空/不足
        assert "行业景气度周期定位" in prompt
        assert "周期综合评分" not in prompt or "数据不足" in prompt

    # ──────────────────────────────────────────────
    # 3. Fallback 降级测试
    # ──────────────────────────────────────────────
    def test_fallback_with_recovery_cycle(self, mock_llm, snapshot_with_quarters):
        """测试复苏期时的行业周期修正"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_quarters)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert "industry_cycle" in opinion.raw_data
        cycle = opinion.raw_data["industry_cycle"]
        assert cycle["stage"] in ("复苏期", "繁荣期", "复苏期/繁荣期")

    def test_fallback_with_recession_cycle(self, mock_llm, snapshot_declining):
        """测试衰退期时的行业周期修正"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_declining)

        assert isinstance(opinion, AgentOpinion)
        assert "industry_cycle" in opinion.raw_data
        cycle = opinion.raw_data["industry_cycle"]
        assert "衰退" in cycle["stage"]

    def test_fallback_reasoning_contains_cycle(self, mock_llm, snapshot_with_quarters):
        """测试降级推理包含行业周期信息"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_with_quarters)

        assert "行业周期" in opinion.reasoning

    def test_fallback_no_cycle_data(self, mock_llm, snapshot_no_quarters):
        """测试无季度数据时不包含行业周期"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_no_quarters)

        assert "industry_cycle" not in opinion.raw_data or not opinion.raw_data.get("industry_cycle")

    # ──────────────────────────────────────────────
    # 4. 端到端分析测试
    # ──────────────────────────────────────────────
    def test_ma_agent_analyze_end_to_end(self, mock_llm, snapshot_with_quarters):
        """测试完整分析流程"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent.analyze(snapshot_with_quarters)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.reasoning) > 0

        # 校验 market_cycle 枚举
        valid_cycles = ["复苏早期", "复苏晚期", "过热", "滞胀", "衰退早期", "衰退晚期"]
        if opinion.raw_data:
            assert opinion.raw_data.get("market_cycle", "") in valid_cycles
            assert opinion.raw_data.get("macro_score", 0) is not None
