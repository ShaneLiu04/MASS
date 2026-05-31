"""
集成测试: Orchestrator 完整流程

注意: 本文件所有测试依赖外部网络（akshare 数据获取），
      默认被 pytest 跳过，需显式运行 `pytest -m integration`
"""
import pytest
from agent.core.orchestrator import AgentOrchestrator


@pytest.mark.integration
class TestOrchestrator:
    """编排器集成测试 — 需外部网络"""
    
    @pytest.fixture
    def orchestrator(self):
        return AgentOrchestrator(use_mock_llm=True)
    
    def test_full_diagnosis(self, orchestrator):
        """测试完整诊断流程"""
        result = orchestrator.run_diagnosis(
            stock_code="000001",
            stock_name="平安银行",
        )
        
        # 基本结构检查
        assert result["stock_code"] == "000001"
        assert result["stock_name"] == "平安银行"
        assert result["current_price"] > 0
        assert "decision_date" in result
        assert "decision_time" in result
        
        # 最终决策检查
        fd = result["final_decision"]
        assert fd["decision"] in (-1, 0, 1)
        assert 0 <= fd["confidence"] <= 1
        assert "reasoning" in fd
        
        # 各Agent观点检查
        opinions = result["opinions"]
        assert len(opinions) == 6
        
        expected_agents = ["TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent", "RA-Agent"]
        for agent_id in expected_agents:
            assert agent_id in opinions
            op = opinions[agent_id]
            assert "signal" in op
            assert "confidence" in op
            assert "reasoning" in op
            assert "key_factors" in op
            assert "risk_flags" in op
    
    def test_diagnosis_with_user_position(self, orchestrator):
        """测试带持仓的诊断"""
        user_position = {
            "cost": 15.0,
            "shares": 1000,
            "current_value": 15000,
        }
        
        result = orchestrator.run_diagnosis(
            stock_code="000001",
            user_position=user_position,
        )
        
        assert result["stock_code"] == "000001"
        assert "final_decision" in result
    
    def test_data_summary(self, orchestrator):
        """测试数据摘要"""
        result = orchestrator.run_diagnosis(stock_code="000001")
        
        ds = result.get("data_summary", {})
        assert "indicator_names" in ds
        assert "fundamental_keys" in ds
        assert "fund_flow_keys" in ds
    
    def test_processing_time(self, orchestrator):
        """测试处理时间"""
        result = orchestrator.run_diagnosis(stock_code="000001")
        
        pt = result.get("processing_time_seconds", 0)
        assert pt >= 0
        # Mock模式网络正常时应很快，真实数据源重试可能耗时至 30s
        assert pt < 30
    
    def test_blackboard_integration(self, orchestrator):
        """测试黑板集成"""
        from agent.core.blackboard import Blackboard
        
        # 运行诊断前清空
        bb = Blackboard()
        bb.clear_stock("000001")
        
        orchestrator.run_diagnosis(stock_code="000001")
        
        # 检查黑板数据
        snapshot = bb.get_snapshot("000001")
        assert snapshot is not None
        assert snapshot.stock_code == "000001"
        
        opinions = bb.get_opinions("000001")
        assert len(opinions) == 6
    
    def test_disclaimer_present(self, orchestrator):
        """测试免责声明"""
        result = orchestrator.run_diagnosis(stock_code="000001")
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 0
    
    def test_multiple_stocks(self, orchestrator):
        """测试多股票诊断"""
        stocks = ["000001", "000002", "600000"]
        results = []
        
        for code in stocks:
            result = orchestrator.run_diagnosis(stock_code=code)
            results.append(result)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["stock_code"] == stocks[i]
            assert "final_decision" in result
    
    def test_risk_filter(self, orchestrator):
        """测试风险过滤"""
        result = orchestrator.run_diagnosis(stock_code="000001")
        
        fd = result["final_decision"]
        ra_opinion = result["opinions"].get("RA-Agent", {})
        
        if ra_opinion and ra_opinion.get("raw_data"):
            risk_level = ra_opinion["raw_data"].get("risk_level", 3)
            # 如果风险等级=5，decision应该不是1
            if risk_level == 5:
                assert fd["decision"] != 1
            
            # 仓位不应超过RA建议
            max_pos = ra_opinion["raw_data"].get("max_position_pct", 0.5)
            assert fd.get("position_pct", 0) <= max_pos
    
    def test_scenario_analysis(self, orchestrator):
        """测试情景分析"""
        result = orchestrator.run_diagnosis(stock_code="000001")
        
        fd = result["final_decision"]
        scenarios = fd.get("scenario_analysis", {})
        
        # 检查是否有情景分析
        if scenarios:
            total_prob = sum(s.get("probability", 0) for s in scenarios.values())
            # 概率之和应接近1
            assert abs(total_prob - 1.0) < 0.1 or total_prob == 0
