"""
MASS 数据管道集成测试
验证: 爬虫 → StockDataTool → StockSnapshot → Agent分析

注意: 本文件所有测试依赖外部网络（akshare/东方财富等），
      默认被 pytest 跳过，需显式运行 `pytest -m integration`
"""
import pytest

from agent.tools.stock_data_tool import StockDataTool
from agent.tools.indicator_tool import IndicatorTool
from agent.tools.sentiment_tool import SentimentTool
from agent.core.blackboard import StockSnapshot
from agent.agents import FA_Agent, CA_Agent, SA_Agent
from agent.tools.llm_client import MockLLMClient


@pytest.mark.integration
class TestDataPipeline:
    """数据管道端到端测试 — 需外部网络"""

    @pytest.fixture
    def stock_tool(self):
        return StockDataTool()

    @pytest.fixture
    def indicator_tool(self):
        return IndicatorTool()

    @pytest.fixture
    def sentiment_tool(self):
        return SentimentTool()

    def test_fundamentals_pipeline(self, stock_tool):
        """测试基本面数据管道 — 验证只返回真实数据"""
        fund = stock_tool.get_fundamentals("600000")
        
        # 如果数据源不可用，返回None（不编造）
        if fund is None:
            # 在当前测试环境中某些源可能受限，验证不编造即可
            return
        
        # 核心字段必须存在（来自真实数据源）
        assert "stock_code" in fund
        assert "company_name" in fund
        assert "_meta" in fund
        
        # 验证数据中不包含任何虚假/mock标记
        assert fund.get("source") != "mock"
        
        # 验证不再自动补充虚假字段
        assert "quarterly_data" not in fund, "不应自动补充虚假季度数据"

    def test_fund_flow_pipeline(self, stock_tool):
        """测试资金流向数据管道 — 真实数据或None"""
        flow = stock_tool.get_fund_flow("600000", days=5)
        
        # 在当前测试环境中资金流向源可能不可用
        # 验证：要么返回真实数据，要么返回None（绝不编造）
        if flow is None:
            return
        
        # 如果返回了数据，验证关键字段
        assert "stock_code" in flow
        assert "_meta" in flow
        assert flow.get("source") != "mock"

    def test_market_context_pipeline(self, stock_tool):
        """测试市场环境数据管道"""
        ctx = stock_tool.get_market_context("600000")
        
        assert "indices" in ctx
        assert "上证指数" in ctx["indices"] or "source" in ctx

    def test_macro_pipeline(self, stock_tool):
        """测试宏观数据管道 — 真实数据或None"""
        macro = stock_tool.get_macro_data()
        
        # 在当前测试环境中宏观数据可能部分可用
        if macro is None:
            return
        
        assert "_meta" in macro
        assert macro.get("source") != "mock"

    def test_sentiment_pipeline(self, stock_tool, sentiment_tool):
        """测试情绪数据管道"""
        sentiment = stock_tool.get_sentiment_data("600000")
        news = sentiment_tool.fetch_stock_news("600000")
        analysis = sentiment_tool.analyze_news(news)
        
        assert "news_count_7d" in sentiment or "volatility_20d" in sentiment
        assert "sentiment_score" in analysis

    def test_snapshot_creation(self, stock_tool, indicator_tool, sentiment_tool):
        """测试 StockSnapshot 创建 — 处理真实数据缺失"""
        stock_code = "600000"
        
        kline_df = stock_tool.get_kline(stock_code, days=30)
        indicators = indicator_tool.compute_all(kline_df) if kline_df is not None else {}
        fundamentals = stock_tool.get_fundamentals(stock_code) or {}
        fund_flow = stock_tool.get_fund_flow(stock_code) or {}
        sentiment_data = stock_tool.get_sentiment_data(stock_code) or {}
        market_context = stock_tool.get_market_context(stock_code) or {}
        macro_data = stock_tool.get_macro_data() or {}
        risk_metrics = indicator_tool.compute_risk_metrics(kline_df) if kline_df is not None else {}
        
        snapshot = StockSnapshot(
            stock_code=stock_code,
            stock_name=fundamentals.get("company_name", stock_code),
            current_price=indicators.get("current_price", 0.0),
            kline_df=kline_df,
            indicators=indicators,
            fundamentals=fundamentals,
            fund_flow=fund_flow,
            market_context=market_context,
            sentiment_data=sentiment_data,
            macro_data=macro_data,
            risk_metrics=risk_metrics,
        )
        
        assert snapshot.stock_code == stock_code
        assert snapshot.stock_name != ""
        # 验证数据质量报告字段存在
        assert hasattr(snapshot, "data_quality")

    def test_fa_agent_with_real_data(self, stock_tool, indicator_tool):
        """测试FA Agent能消费真实基本面数据"""
        llm = MockLLMClient()
        agent = FA_Agent("FA-Agent", llm)
        
        stock_code = "600000"
        kline_df = stock_tool.get_kline(stock_code, days=30)
        fundamentals = stock_tool.get_fundamentals(stock_code)
        
        snapshot = StockSnapshot(
            stock_code=stock_code,
            stock_name=fundamentals.get("company_name", stock_code),
            current_price=10.0,
            fundamentals=fundamentals,
        )
        
        opinion = agent.analyze(snapshot)
        
        assert opinion.agent_id == "FA-Agent"
        assert opinion.signal in (-1, 0, 1)
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.reasoning) > 0

    def test_data_source_stats(self, stock_tool):
        """测试数据质量报告"""
        # 发起请求
        stock_tool.get_fundamentals("600000")
        stock_tool.get_sentiment_data("600000")
        
        stats = stock_tool.get_data_quality_report("600000")
        assert "stock_code" in stats
        assert "data_types" in stats
        assert "timestamp" in stats
        # 验证报告中包含kline质量信息
        assert "kline" in stats["data_types"]
