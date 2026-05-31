"""
单元测试: 技术指标计算
"""
import pytest
import numpy as np
import pandas as pd

from agent.tools.indicator_tool import IndicatorTool


class TestIndicators:
    """技术指标测试类"""
    
    @pytest.fixture
    def sample_df(self):
        """生成60日模拟K线数据"""
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="B")
        base = 15.0
        
        closes = [base]
        for _ in range(59):
            closes.append(closes[-1] * (1 + np.random.normal(0, 0.02)))
        closes = np.array(closes)
        
        opens = closes * (1 + np.random.normal(0, 0.005, 60))
        highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.02, 60))
        lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.02, 60))
        volumes = np.random.randint(1000000, 5000000, 60)
        
        return pd.DataFrame({
            "date": dates,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes,
            "amount": np.round(volumes * closes, 2),
        })
    
    def test_compute_all_basic(self, sample_df):
        """测试指标计算基本功能"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert "error" not in result
        assert result["current_price"] > 0
        assert "ma5" in result
        assert "ma20" in result
        assert "ma60" in result
        assert "boll_upper" in result
        assert "boll_lower" in result
        assert "kdj_k" in result
        assert "macd_dif" in result
        assert "rsi14" in result
    
    def test_ma_calculation(self, sample_df):
        """测试MA计算准确性"""
        result = IndicatorTool.compute_all(sample_df)
        closes = sample_df["close"].values
        
        expected_ma5 = np.mean(closes[-5:])
        expected_ma20 = np.mean(closes[-20:])
        
        assert abs(result["ma5"] - expected_ma5) < 0.1
        assert abs(result["ma20"] - expected_ma20) < 0.1
    
    def test_boll_calculation(self, sample_df):
        """测试布林带计算"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert result["boll_upper"] > result["boll_mid"]
        assert result["boll_mid"] > result["boll_lower"]
        assert result["boll_width"] > 0
    
    def test_kdj_range(self, sample_df):
        """测试KDJ值范围"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert 0 <= result["kdj_k"] <= 100
        assert 0 <= result["kdj_d"] <= 100
        # J值可能超出0-100范围
    
    def test_rsi_range(self, sample_df):
        """测试RSI值范围"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert 0 <= result["rsi14"] <= 100
    
    def test_macd_consistency(self, sample_df):
        """测试MACD一致性"""
        result = IndicatorTool.compute_all(sample_df)
        
        # MACD柱状图 = 2 * (DIF - DEA)
        expected_hist = 2 * (result["macd_dif"] - result["macd_dea"])
        assert abs(result["macd_hist"] - expected_hist) < 0.01
    
    def test_volume_analysis(self, sample_df):
        """测试成交量分析"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert result["volume_ratio"] > 0
        assert result["volume_trend"] in ["放量", "缩量", "正常"]
    
    def test_atr_positive(self, sample_df):
        """测试ATR为正"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert result["atr14"] > 0
        assert result["atr_pct"] > 0
    
    def test_price_position(self, sample_df):
        """测试价格位置"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert 0 <= result["position_20d"] <= 1
        assert 0 <= result["position_60d"] <= 1
    
    def test_empty_data(self):
        """测试空数据处理"""
        empty_df = pd.DataFrame()
        result = IndicatorTool.compute_all(empty_df)
        
        assert "error" in result
    
    def test_short_data(self):
        """测试数据不足"""
        short_df = pd.DataFrame({
            "close": [10, 11, 12],
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "volume": [100, 200, 300],
        })
        result = IndicatorTool.compute_all(short_df)
        
        assert "error" in result
    
    def test_risk_metrics(self, sample_df):
        """测试风险指标"""
        result = IndicatorTool.compute_risk_metrics(sample_df)
        
        assert "error" not in result
        assert result["annual_volatility"] >= 0
        assert result["max_drawdown"] <= 0
        assert result["sharpe_ratio"] is not None
    
    def test_ma_alignment(self, sample_df):
        """测试均线排列判断"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert result["ma_alignment"] in ["多头排列", "空头排列", "缠绕/整理"]
    
    def test_divergence_detection(self, sample_df):
        """测试背离检测"""
        result = IndicatorTool.compute_all(sample_df)
        
        assert result["price_volume_divergence"] in [
            "价涨量缩（顶背离风险）",
            "价跌量增（恐慌抛售或吸筹）",
            "量价齐升",
            "量价齐跌",
            "数据不足",
        ]
