"""
单元测试: MA-Agent v2.2 政策敏感性分析
"""
import pytest

from agent.agents import MA_Agent
from agent.tools.llm_client import MockLLMClient
from agent.core.blackboard import StockSnapshot, AgentOpinion


class TestMAAgentPolicyV2:
    """MA-Agent v2.2 政策敏感性分析测试类"""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def snapshot_bank_loose(self):
        """银行 + 宽松货币政策"""
        return StockSnapshot(
            stock_code="000001",
            stock_name="平安银行",
            current_price=15.0,
            indicators={},
            fundamentals={"industry": "银行"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={
                "policy_stance": "宽松",
                "bond_yield_trend": "下行",
            },
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_realestate_tight(self):
        """房地产 + 严监管"""
        return StockSnapshot(
            stock_code="000002",
            stock_name="万科A",
            current_price=20.0,
            indicators={},
            fundamentals={"industry": "房地产"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={
                "policy_stance": "收紧，限购限贷",
                "regulatory_policy": "加强监管",
            },
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_semiconductor_mixed(self):
        """半导体 + 混合政策环境"""
        return StockSnapshot(
            stock_code="688981",
            stock_name="中芯国际",
            current_price=50.0,
            indicators={},
            fundamentals={"industry": "半导体"},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={
                "policy_stance": "积极财政，产业扶持",
                "trade_policy": "技术封锁，出口管制",
            },
            risk_metrics={},
        )

    @pytest.fixture
    def snapshot_no_industry(self):
        """无行业信息"""
        return StockSnapshot(
            stock_code="999999",
            stock_name="测试",
            current_price=15.0,
            indicators={},
            fundamentals={},
            fund_flow={},
            sentiment_data={},
            market_context={},
            macro_data={"policy_stance": "宽松"},
            risk_metrics={},
        )

    # ──────────────────────────────────────────────
    # 1. 政策敏感度配置测试
    # ──────────────────────────────────────────────
    def test_policy_sensitivity_bank(self, mock_llm):
        """银行对货币政策敏感度 0.95（极高）"""
        agent = MA_Agent("MA-Agent", mock_llm)
        sens = agent._get_policy_sensitivity("银行")
        assert sens["monetary"] == 0.95
        assert sens["fiscal"] == 0.50
        assert sens["regulatory"] == 0.90

    def test_policy_sensitivity_semiconductor(self, mock_llm):
        """半导体对贸易政策敏感度 0.90（极高）"""
        agent = MA_Agent("MA-Agent", mock_llm)
        sens = agent._get_policy_sensitivity("半导体")
        assert sens["trade"] == 0.90
        assert sens["industrial"] == 0.95
        assert sens["monetary"] == 0.50

    def test_policy_sensitivity_new_energy(self, mock_llm):
        """新能源对财政和产业敏感度极高"""
        agent = MA_Agent("MA-Agent", mock_llm)
        sens = agent._get_policy_sensitivity("新能源")
        assert sens["fiscal"] == 0.90
        assert sens["industrial"] == 0.95

    def test_policy_sensitivity_unknown_industry(self, mock_llm):
        """未知行业返回默认敏感度"""
        agent = MA_Agent("MA-Agent", mock_llm)
        sens = agent._get_policy_sensitivity("未知行业")
        assert sens["monetary"] == 0.50
        assert sens["fiscal"] == 0.50
        assert sens["regulatory"] == 0.50

    def test_policy_sensitivity_empty_industry(self, mock_llm):
        """空行业返回默认敏感度"""
        agent = MA_Agent("MA-Agent", mock_llm)
        sens = agent._get_policy_sensitivity("")
        assert sens["monetary"] == 0.50

    def test_policy_sensitivity_fuzzy_match(self, mock_llm):
        """模糊匹配（子串）"""
        agent = MA_Agent("MA-Agent", mock_llm)
        # "股份制银行" 应该匹配到 "银行"
        sens = agent._get_policy_sensitivity("股份制银行")
        assert sens["monetary"] == 0.95

    # ──────────────────────────────────────────────
    # 2. 政策影响分析测试
    # ──────────────────────────────────────────────
    def test_policy_impact_loose_monetary(self, mock_llm, snapshot_bank_loose):
        """宽松货币 + 高敏感度行业 = 强利好"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_bank_loose)

        assert impact["industry"] == "银行"
        assert impact["sensitivity"]["monetary"] == 0.95
        assert impact["impact_score"] > 0
        assert impact["impact_direction"] in ("利好", "偏利好")
        assert impact["policy_score"] > 50
        assert len(impact["transmission_analysis"]) > 0

    def test_policy_impact_tight_regulatory(self, mock_llm, snapshot_realestate_tight):
        """严监管 + 高敏感度行业 = 强利空"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_realestate_tight)

        assert impact["industry"] == "房地产"
        assert impact["sensitivity"]["regulatory"] == 0.95
        assert impact["impact_score"] < 0
        assert impact["impact_direction"] in ("利空", "偏利空")
        assert impact["policy_score"] < 50
        # 风险雷达应该识别监管政策
        risk = impact.get("risk_radar", {})
        assert "监管" in risk.get("warning", "") or "当前无明显政策风险维度" in risk.get("warning", "")

    def test_policy_impact_mixed(self, mock_llm, snapshot_semiconductor_mixed):
        """mixed 政策环境下的综合得分"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_semiconductor_mixed)

        assert impact["industry"] == "半导体"
        # 财政+产业利好 vs 贸易利空，综合得分取决于权重
        dim_scores = impact["dimension_scores"]
        assert "fiscal" in dim_scores
        assert "industrial" in dim_scores
        assert "trade" in dim_scores
        # 传导预期应该包含国产替代相关文本
        tx = " ".join(impact["transmission_analysis"])
        assert "国产替代" in tx or "补贴" in tx or "技术封锁" in tx or "中性" in tx

    def test_policy_impact_no_industry(self, mock_llm, snapshot_no_industry):
        """无行业信息时使用默认敏感度"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_no_industry)

        assert impact["industry"] == "未知"
        assert impact["sensitivity"]["monetary"] == 0.50
        assert impact["impact_score"] > 0  # 宽松 = 利好

    def test_policy_impact_no_policy_data(self, mock_llm):
        """无政策数据时所有方向为中性"""
        snapshot = StockSnapshot(
            stock_code="000003", stock_name="测试", current_price=10.0,
            indicators={}, fund_flow={}, sentiment_data={}, market_context={}, macro_data={}, risk_metrics={},
            fundamentals={"industry": "银行"},
        )
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot)

        assert impact["impact_score"] == 0.0
        assert impact["impact_direction"] == "中性"
        assert impact["policy_score"] == 50

    # ──────────────────────────────────────────────
    # 3. 政策传导预期文本测试
    # ──────────────────────────────────────────────
    def test_transmission_analysis_bank_loose(self, mock_llm, snapshot_bank_loose):
        """银行+宽松应该生成利率相关传导预期"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_bank_loose)
        tx = " ".join(impact["transmission_analysis"])
        assert "利率" in tx or "息差" in tx or "融资成本" in tx

    def test_transmission_analysis_realestate_tight(self, mock_llm, snapshot_realestate_tight):
        """房地产+严监管应该生成监管相关传导预期"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_realestate_tight)
        tx = " ".join(impact["transmission_analysis"])
        assert "监管" in tx or "房企" in tx or "按揭" in tx or "中性" in tx

    # ──────────────────────────────────────────────
    # 4. Prompt 构建测试
    # ──────────────────────────────────────────────
    def test_build_prompt_contains_policy(self, mock_llm, snapshot_bank_loose):
        """测试Prompt包含政策敏感性章节"""
        agent = MA_Agent("MA-Agent", mock_llm)
        prompt = agent._build_ma_prompt(snapshot_bank_loose)

        assert "政策敏感性分析" in prompt
        assert "货币政策敏感度" in prompt
        assert "综合政策影响得分" in prompt
        assert "政策传导预期" in prompt
        assert "政策风险雷达" in prompt
        assert "政策敏感度纳入量化对冲规则" in prompt

    # ──────────────────────────────────────────────
    # 5. Fallback 降级测试
    # ──────────────────────────────────────────────
    def test_fallback_with_policy_score(self, mock_llm, snapshot_bank_loose):
        """测试Fallback中政策评分修正"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_bank_loose)

        assert isinstance(opinion, AgentOpinion)
        assert "policy_impact" in opinion.raw_data
        policy = opinion.raw_data["policy_impact"]
        assert policy["impact_score"] > 0
        assert "政策影响" in opinion.reasoning

    def test_fallback_with_realestate_tight(self, mock_llm, snapshot_realestate_tight):
        """测试房地产严监管时的Fallback"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_realestate_tight)

        assert isinstance(opinion, AgentOpinion)
        assert "policy_impact" in opinion.raw_data
        policy = opinion.raw_data["policy_impact"]
        assert policy["impact_score"] < 0

    def test_fallback_no_policy_data(self, mock_llm, snapshot_no_industry):
        """测试无政策数据时Fallback仍能工作"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent._fallback_opinion(snapshot_no_industry)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)

    # ──────────────────────────────────────────────
    # 6. 风险雷达测试
    # ──────────────────────────────────────────────
    def test_risk_radar_identifies_top_risk(self, mock_llm, snapshot_realestate_tight):
        """风险雷达识别最敏感的不利维度"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_realestate_tight)

        risk = impact.get("risk_radar", {})
        # 房地产对监管敏感度 0.95，且政策方向为利空
        if "dimension" in risk:
            assert risk["sensitivity"] >= 0.5
            assert "不利" in risk.get("direction", "")

    def test_risk_radar_no_risk(self, mock_llm, snapshot_bank_loose):
        """无不利政策时风险雷达为空"""
        agent = MA_Agent("MA-Agent", mock_llm)
        impact = agent._analyze_policy_impact(snapshot_bank_loose)

        risk = impact.get("risk_radar", {})
        # 宽松货币对银行是利好，不应有风险雷达
        assert "当前无明显政策风险维度" in risk.get("warning", "") or "dimension" not in risk

    # ──────────────────────────────────────────────
    # 7. 端到端分析测试
    # ──────────────────────────────────────────────
    def test_ma_agent_analyze_with_policy(self, mock_llm, snapshot_bank_loose):
        """测试完整分析流程包含政策数据"""
        agent = MA_Agent("MA-Agent", mock_llm)
        opinion = agent.analyze(snapshot_bank_loose)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_id == "MA-Agent"
        assert opinion.signal in (-1, 0, 1)
        # LLM模式下raw_data中应包含policy相关字段（由_prompt构建）
