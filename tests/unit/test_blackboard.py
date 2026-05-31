"""
单元测试: 共享黑板 (Blackboard)
"""
import pytest
from datetime import datetime
import pandas as pd

from agent.core.blackboard import Blackboard, StockSnapshot, AgentOpinion


class TestBlackboard:
    """Blackboard 测试类"""
    
    def test_get_instance_singleton(self):
        """测试 get_instance() 返回全局单例"""
        bb1 = Blackboard.get_instance()
        bb2 = Blackboard.get_instance()
        assert bb1 is bb2
    
    def test_direct_instantiation_isolation(self):
        """测试直接实例化创建独立实例（测试隔离）"""
        bb1 = Blackboard()
        bb2 = Blackboard()
        assert bb1 is not bb2
    
    def test_publish_snapshot(self):
        """测试发布快照"""
        bb = Blackboard()  # 独立实例，避免测试间状态污染
        
        snapshot = StockSnapshot(
            stock_code="000001",
            stock_name="测试",
            current_price=10.0,
        )
        bb.publish_snapshot(snapshot)
        
        retrieved = bb.get_snapshot("000001")
        assert retrieved is not None
        assert retrieved.stock_code == "000001"
        assert retrieved.current_price == 10.0
    
    def test_submit_opinion(self):
        """测试提交观点"""
        bb = Blackboard()
        bb.clear_stock("000001")
        
        # 先发布快照
        bb.publish_snapshot(StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=10.0
        ))
        
        opinion = AgentOpinion(
            agent_id="TA-Agent",
            signal=1,
            confidence=0.8,
            reasoning="测试",
        )
        bb.submit_opinion("000001", opinion)
        
        opinions = bb.get_opinions("000001")
        assert len(opinions) == 1
        assert opinions[0].agent_id == "TA-Agent"
        assert opinions[0].signal == 1
    
    def test_get_opinion_by_agent(self):
        """测试按Agent获取观点"""
        bb = Blackboard()
        bb.clear_stock("000001")
        bb.publish_snapshot(StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=10.0
        ))
        
        bb.submit_opinion("000001", AgentOpinion("TA-Agent", 1, 0.8, "测试"))
        bb.submit_opinion("000001", AgentOpinion("FA-Agent", 0, 0.6, "测试"))
        
        ta_op = bb.get_opinion_by_agent("000001", "TA-Agent")
        assert ta_op is not None
        assert ta_op.signal == 1
        
        missing = bb.get_opinion_by_agent("000001", "XX-Agent")
        assert missing is None
    
    def test_thread_safety(self):
        """测试线程安全（基础）"""
        bb = Blackboard()
        bb.clear_stock("000001")
        bb.publish_snapshot(StockSnapshot(
            stock_code="000001", stock_name="测试", current_price=10.0
        ))
        
        import threading
        opinions = []
        
        def submit():
            for i in range(10):
                op = AgentOpinion(f"Agent-{i}", 1, 0.5, "test")
                bb.submit_opinion("000001", op)
                opinions.append(op)
        
        threads = [threading.Thread(target=submit) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        final_ops = bb.get_opinions("000001")
        assert len(final_ops) == 30
    
    def test_to_summary(self):
        """测试快照摘要"""
        snapshot = StockSnapshot(
            stock_code="000001",
            stock_name="测试",
            current_price=10.0,
            indicators={"ma5": 10},
        )
        summary = snapshot.to_summary()
        assert summary["stock_code"] == "000001"
        assert "indicator_names" in summary
        assert "ma5" in summary["indicator_names"]
    
    def test_opinion_to_dict(self):
        """测试观点序列化"""
        op = AgentOpinion(
            agent_id="TA-Agent",
            signal=1,
            confidence=0.8,
            reasoning="测试理由",
            key_factors=["因子1"],
            risk_flags=["风险1"],
        )
        d = op.to_dict()
        assert d["agent_id"] == "TA-Agent"
        assert d["signal"] == 1
        assert d["key_factors"] == ["因子1"]
        assert "timestamp" in d
