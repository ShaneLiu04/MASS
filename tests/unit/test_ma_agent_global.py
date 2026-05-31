"""
单元测试: MA-Agent v2.3 全球经济联动分析
"""
import pytest

from agent.agents import MA_Agent
from agent.tools.llm_client import MockLLMClient
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestMAAgentGlobalV3:
    """MA-Agent v2.3 全球经济联动分析测试类"""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def snapshot_electronics_export(self):
        """电子（出口型）+ 人民币贬值"""
        return StockSnapshot(
            stock_code="000100",
            stock_name="TCL科技",
            current_price=5.0,
            indicators={},
            fundamentals={"industry": "电子"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={"rmb_trend": "贬值", "oil_trend": "下跌"},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_airline_import(self):
        """航空（进口型）+ 人民币升值 + 油价上涨"""
        return StockSnapshot(
            stock_code="600029",
            stock_name="南方航空",
            current_price=8.0,
            indicators={},
            fundamentals={"industry": "航空"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={"rmb_trend": "升值", "oil_trend": "上涨"},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_steel_commodity(self):
        """钢铁（铁矿石敏感）+ 铁矿石上涨 + 油价上涨"""
        return StockSnapshot(
            stock_code="000709",
            stock_name="河钢股份",
            current_price=3.0,
            indicators={},
            fundamentals={"industry": "钢铁"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={"oil_trend": "上涨", "iron_trend": "上涨"},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_semiconductor_us(self):
        """半导体 + 美股数据"""
        return StockSnapshot(
            stock_code="688981",
            stock_name="中芯国际",
            current_price=50.0,
            indicators={},
            fundamentals={"industry": "半导体"},
            fund_flow={},
            sentiment_data={},
            market_context={"us_market": {"pct_change": 0.03}},
            macro_data={},
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_no_global_data(self):
        """无任何全球经济数据"""
        return StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={},
            fundamentals={"industry": "银行"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={},
            risk_metrics={},
        )

    # ──────────────────────────────────────────────
    # 1. 汇率敏感度配置测试
    # ──────────────────────────────────────────────
    def test_fx_sensitivity_export(self, mock_llm):
        """电子为出口型行业，汇率敏感度 0.85"""
        agent = MA_Agent("MA-Agent", mock_llm)
        fx = agent._get_fx_sensitivity("电子")
        assert fx["type"] == "export"
        assert fx["sensitivity"] == 0.85

    def test_fx_sensitivity_import(self, mock_llm):
        """航空为进口型行业，汇率敏感度 0.85"""
        agent = MA_Agent("MA-Agent", mock_llm)
        fx = agent._get_fx_sensitivity("航空")
        assert fx["type"] == "import"
        assert fx["sensitivity"] == 0.85

    def test_fx_sensitivity_unknown(self, mock_llm):
        """未知行业返回中性"""
        agent = MA_Agent("MA-Agent", mock_llm)
        fx = agent._get_fx_sensitivity("未知")
        assert fx["type"] == "neutral"
        assert fx["sensitivity"] == 0.0

    # ──────────────────────────────────────────────
    # 2. 大宗商品敏感度测试
    # ──────────────────────────────────────────────
    def test_commodity_sensitivity_airline(self, mock_llm):
        """航空对油价敏感度 0.90"""
        agent = MA_Agent("MA-Agent", mock_llm)
        comm = agent._get_commodity_sensitivity("航空")
        assert comm["oil"] == 0.90

    def test_commodity_sensitivity_steel(self, mock_llm):
        """钢铁对铁矿石敏感度 0.85"""
        agent = MA_Agent("MA-Agent", mock_llm)
        comm = agent._get_commodity_sensitivity("钢铁")
        assert comm["iron"] == 0.85

    def test_commodity_sensitivity_none(self, mock_llm):
        """银行无大宗商品敏感度"""
        agent = MA_Agent("MA-Agent", mock_llm)
        comm = agent._get_commodity_sensitivity("银行")
        assert comm == {}

    # ──────────────────────────────────────────────
    # 3. 美股映射测试
    # ──────────────────────────────────────────────
    def test_us_stock_mapping_semiconductor(self, mock_llm):
        """半导体映射到美股半导体/SOX，相关系数 0.75"""
        agent = MA_Agent("MA-Agent", mock_llm)
        us = agent._get_us_stock_mapping("半导体")
        assert "SOX" in us["sector"]
        assert us["correlation"] == 0.75

    def test_us_stock_mapping_bank(self, mock_llm):
        """银行映射到金融/XLF"""
        agent = MA_Agent("MA-Agent", mock_llm)
        us = agent._get_us_stock_mapping("银行")
        assert "XLF" in us["sector"]

    # ──────────────────────────────────────────────
    # 4. 全球经济联动分析核心测试
    # ──────────────────────────────────────────────
    def test_global_economy_export_weak_rmb(self, mock_llm, snapshot_electronics_export):
        """出口型 + 人民币贬值 = 汇率利好"""
        agent = MA_Agent("MA-Agent", mock_llm)
        ge = agent._analyze_global_economy(snapshot_electronics_export)

        assert ge["industry"] == "电子"
        fx = ge["fx_analysis"]
        assert fx["industry_type"] == "export"
        assert fx["impact_score"] > 0  # 出口型+贬值=利好
        assert "利好" in fx["reasoning"] or "受益" in fx["reasoning"]
        assert ge["global_score"] > 0

    def test_global_economy_import_strong_rmb_oil_up(self, mock_llm, snapshot_airline_import):
        """进口型 + 人民币升值 + 油价上涨 = 汇率利好但油价利空"""
        agent = MA_Agent("MA-Agent", mock_llm)
        ge = agent._analyze_global_economy(snapshot_airline_import)

        assert ge["industry"] == "航空"
        fx = ge["fx_analysis"]
        assert fx["impact_score"] > 0  # 进口型+升值=利好
        comm = ge["commodity_analysis"]
        assert comm["total_impact"] > 0  # 油价上涨对航空为利空(正值)
        # 汇率利好 + 油价利空，综合看哪个更大

    def test_global_economy_steel_commodity_up(self, mock_llm, snapshot_steel_commodity):
        """钢铁 + 铁矿石上涨 + 油价上涨 = 原材料成本压力"""
        agent = MA_Agent("MA-Agent", mock_llm)
        ge = agent._analyze_global_economy(snapshot_steel_commodity)

        assert ge["industry"] == "钢铁"
        comm = ge["commodity_analysis"]
        assert comm["total_impact"] > 0  # 原材料上涨=利空(正值)
        # 应有铁矿石和油价两条影响
        assert len(comm["impacts"]) >= 1

    def test_global_economy_us_stock_mapping(self, mock_llm, snapshot_semiconductor_us):
        """半导体 + 美股上涨 = 正向传导"""
        agent = MA_Agent("MA-Agent", mock_llm)
        ge = agent._analyze_global_economy(snapshot_semiconductor_us)

        assert ge["industry"] == "半导体"
        us = ge["us_stock_mapping"]
        assert "SOX" in us["us_sector"]
        assert us["impact_score"] > 0  # 美股上涨=正向传导
        assert "大涨" in us["reasoning"] or "正向传导" in us["reasoning"]

    def test_global_economy_no_data(self, mock_llm, snapshot_no_global_data):
        """无全球数据时返回中性"""
        agent = MA_Agent("MA-Agent", mock_llm)
        ge = agent._analyze_global_economy(snapshot_no_global_data)

        assert ge["global_score"] == 0.0
        assert ge["global_direction"] == "中性"
        assert ge["global_score_100"] == 50

    # ──────────────────────────────────────────────
    # 5. Prompt 构建测试
    # ──────────────────────────────────────────────
    def test_build_prompt_contains_global(self, mock_llm, snapshot_electronics_export):
        """测试Prompt包含全球经济联动章节"""
        agent = MA_Agent("MA-Agent", mock_llm)
        prompt = agent._build_ma_prompt(snapshot_electronics_export)

        assert "全球经济联动分析" in prompt
        assert "汇率敏感性分析" in prompt
        assert "美股映射与联动传导" in prompt
        assert "全球经济联动综合得分" in prompt
        assert "全球经济联动纳入量化对冲规则" in prompt

    # ──────────────────────────────────────────────
    # 6. Fallback 降级测试
    # ──────────────────────────────────────────────
    def test_fallback_with_global_score(self, mock_llm, snapshot_electronics_export):
        """测试Fallback中全球经济评分修正"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_electronics_export)

        assert isinstance(opinion, AgentOpinion)
        assert "global_economy" in opinion.raw_data
        ge = opinion.raw_data["global_economy"]
        # 电子(出口型)+贬值=汇率利好，油价下跌=成本利好，综合应非零
        assert ge["global_score"] != 0.0

    def test_fallback_no_global_data(self, mock_llm, snapshot_no_global_data):
        """测试无全球数据时Fallback正常工作"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_no_global_data)

        assert isinstance(opinion, AgentOpinion)
        # 即使无宏观数据，美股映射仍可能存在（如银行→XLF）
        # 关键是 Fallback 不崩溃且信号合理
        assert opinion.signal in (-1, 0, 1)
        assert opinion.agent_id == "MA-Agent"

    # ──────────────────────────────────────────────
    # 7. 端到端分析测试
    # ──────────────────────────────────────────────
    def test_ma_agent_analyze_with_global(self, mock_llm, snapshot_semiconductor_us):
        """测试完整分析流程包含全球数据"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent.analyze(snapshot_semiconductor_us)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)
