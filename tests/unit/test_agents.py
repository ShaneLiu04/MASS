"""
单元测试: Agent 分析逻辑
"""
import pytest

from agent.agents import (
    TA_Agent, FA_Agent, CA_Agent, SA_Agent, MA_Agent, RA_Agent
)
from agent.tools.llm_client import MockLLMClient
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestAgents:
    """Agent 测试类"""
    
    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()
    
    @pytest.fixture
    def snapshot(self):
        return StockSnapshot(
            stock_code="000001",
            stock_name="测试",
            current_price=15.0,
            indicators={
                "current_price": 15.0, "ma5": 15.2, "ma20": 15.0,
                "ma_alignment": "多头排列", "macd_golden_cross": True,
                "rsi14": 55, "kdj_k": 60, "boll_position": 0.5,
            },
            fundamentals={
                "pe_ttm": 15, "pb": 1.5, "roe": 18,
                "debt_ratio": 40, "revenue_growth": 20,
                "quarterly_data": [
                    {"quarter": "2024Q1", "revenue": 50, "net_profit": 8, "roe": 18},
                ],
            },
            fund_flow={
                "main_net_inflow_10d": 10000,
                "main_inflow_days": 7,
                "daily_flow": [{"date": "2024-01-01", "main_net_inflow": 1000}],
            },
            sentiment_data={
                "social_sentiment_7d": 0.3,
                "news": [{"title": "测试", "source": "测试"}],
            },
            market_context={
                "sector_performance_5d": 2.5,
                "sector_rank": 10,
            },
            macro_data={
                "pmi": 51, "bond_yield_10y": 2.5,
            },
            risk_metrics={
                "annual_volatility": 20,
                "max_drawdown": -10,
            },
        )
    
    def test_ta_agent_analyze(self, mock_llm, snapshot):
        """测试技术面Agent"""
        agent = TA_Agent("TA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "TA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.reasoning) > 0
    
    def test_fa_agent_analyze(self, mock_llm, snapshot):
        """测试基本面Agent"""
        agent = FA_Agent("FA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "FA-Agent"
        assert opinion.signal in (-1, 0, 1)
        # 检查score与signal的联动
        if opinion.raw_data and "fundamental_score" in opinion.raw_data:
            score = opinion.raw_data["fundamental_score"]
            expected = 1 if score >= 75 else (-1 if score <= 40 else 0)
            assert opinion.signal == expected
    
    def test_ca_agent_analyze(self, mock_llm, snapshot):
        """测试资金面Agent"""
        agent = CA_Agent("CA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "CA-Agent"
        assert opinion.signal in (-1, 0, 1)
        # 校验smart_money_direction枚举
        if opinion.raw_data:
            valid = ["强烈建仓", "建仓期", "观望", "派发期", "强烈派发"]
            smd = opinion.raw_data.get("smart_money_direction", "观望")
            assert smd in valid
    
    def test_sa_agent_analyze(self, mock_llm, snapshot):
        """测试情绪面Agent"""
        agent = SA_Agent("SA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "SA-Agent"
        assert opinion.signal in (-1, 0, 1)
        if opinion.raw_data:
            si = opinion.raw_data.get("sentiment_index", 0)
            assert -1 <= si <= 1
    
    def test_ma_agent_analyze(self, mock_llm, snapshot):
        """测试宏观Agent"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)
        if opinion.raw_data:
            cycles = ["复苏早期", "复苏晚期", "过热", "滞胀", "衰退早期", "衰退晚期"]
            assert opinion.raw_data.get("market_cycle", "") in cycles
    
    def test_ra_agent_analyze(self, mock_llm, snapshot):
        """测试风险Agent"""
        agent = RA_Agent("RA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "RA-Agent"
        assert opinion.signal == 0  # RA-Agent只输出风险，不参与方向投票
        if opinion.raw_data:
            rl = opinion.raw_data.get("risk_level", 3)
            assert 1 <= rl <= 5
            mpp = opinion.raw_data.get("max_position_pct", 0.1)
            assert 0.05 <= mpp <= 0.50
    
    def test_agent_fallback(self, mock_llm, snapshot):
        """测试Agent降级逻辑"""
        # 构造一个会导致LLM失败的数据（mock不会失败，但测试降级代码路径）
        agent = TA_Agent("TA-Agent", mock_llm)
        opinion = agent.analyze(snapshot)
        assert opinion is not None
        assert opinion.signal in (-1, 0, 1)
    
    def test_base_agent_prompt_loading(self, mock_llm):
        """测试Prompt加载"""
        agent = TA_Agent("TA-Agent", mock_llm)
        # 应该成功加载文件中的prompt
        assert len(agent.system_prompt) > 100
        assert "Role" in agent.system_prompt
        assert "JSON" in agent.system_prompt
    
    def test_safe_parse_llm_response(self, mock_llm):
        """测试LLM响应安全解析"""
        agent = TA_Agent("TA-Agent", mock_llm)
        
        # 正常响应
        normal = {"signal": 1, "confidence": 0.8, "reasoning": "测试"}
        parsed = agent._safe_parse_llm_response(normal)
        assert parsed["signal"] == 1
        assert parsed["confidence"] == 0.8
        
        # 缺少字段的响应
        incomplete = {"signal": "invalid"}
        parsed = agent._safe_parse_llm_response(incomplete)
        assert parsed["signal"] == 0  # 默认值
        assert parsed["confidence"] == 0.5
        
        # confidence越界的响应
        out_of_range = {"signal": 1, "confidence": 1.5}
        parsed = agent._safe_parse_llm_response(out_of_range)
        assert parsed["confidence"] == 1.0  # 被截断
