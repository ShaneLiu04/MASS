"""
MASS Agent 响应模型 - Pydantic v2
严格定义每个Agent的输出结构，用于JSON校验
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class BaseOpinion(BaseModel):
    """Agent观点基类"""
    signal: int = Field(..., ge=-1, le=1, description="-1卖出, 0观望, 1买入")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(..., min_length=10, description="自然语言推理过程")
    key_factors: List[str] = Field(default_factory=list, description="关键因子")
    risk_flags: List[str] = Field(default_factory=list, description="风险提示")


class TAOpinion(BaseOpinion):
    """技术面Agent输出"""
    target_price_low: Optional[float] = Field(None, description="目标价下限")
    target_price_high: Optional[float] = Field(None, description="目标价上限")
    stop_loss: Optional[float] = Field(None, description="止损价")
    chart_patterns: List[str] = Field(default_factory=list, description="识别到的形态")
    trend_direction: str = Field("unknown", description="趋势方向")

    @field_validator('target_price_low', 'target_price_high')
    @classmethod
    def validate_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('价格必须大于0')
        return v


class SubScores(BaseModel):
    """基本面子评分"""
    profitability: int = Field(0, ge=0, le=25)
    growth: int = Field(0, ge=0, le=25)
    safety: int = Field(0, ge=0, le=20)
    valuation: int = Field(0, ge=0, le=20)
    moat: int = Field(0, ge=0, le=10)


class FAOpinion(BaseOpinion):
    """基本面Agent输出"""
    fundamental_score: int = Field(0, ge=0, le=100, description="综合评分")
    sub_scores: SubScores = Field(default_factory=SubScores)
    valuation_gap: str = Field("", description="估值分析")


class CAOpinion(BaseOpinion):
    """资金面Agent输出"""
    capital_score: int = Field(0, ge=0, le=100, description="资金评分")
    smart_money_direction: str = Field("观望", description="主力方向")
    retail_vs_institutional: str = Field("", description="散户vs机构")


class SAOpinion(BaseOpinion):
    """情绪面Agent输出"""
    sentiment_index: float = Field(0.0, ge=-1.0, le=1.0, description="情绪指数")
    sentiment_percentile: int = Field(50, ge=0, le=100, description="情绪分位")
    crowd_behavior: str = Field("", description=" crowd行为")
    contrarian_opportunity: str = Field("", description="逆向机会")


class WeightAdjustment(BaseModel):
    """权重调整"""
    TA: float = Field(0.0, ge=-0.2, le=0.2)
    FA: float = Field(0.0, ge=-0.2, le=0.2)
    CA: float = Field(0.0, ge=-0.2, le=0.2)
    SA: float = Field(0.0, ge=-0.2, le=0.2)
    MA: float = Field(0.0, ge=-0.2, le=0.2)
    RA: float = Field(0.0, ge=-0.2, le=0.2)


class MAOpinion(BaseOpinion):
    """宏观Agent输出"""
    market_cycle: str = Field("", description="市场周期")
    cycle_confidence: float = Field(0.0, ge=0.0, le=1.0)
    sector_outlook: str = Field("中性", description="行业展望")
    style_alignment: float = Field(0.5, ge=0.0, le=1.0)
    macro_signal: int = Field(0, ge=-1, le=1)
    recommended_weight_adjustment: WeightAdjustment = Field(default_factory=WeightAdjustment)


class StressScenario(BaseModel):
    """压力测试单情景"""
    probability: float = Field(0.25, ge=0.0, le=1.0, description="假设概率")
    expected_return: float = Field(0.0, description="预期收益率(%)")
    max_drawdown: float = Field(0.0, description="最大回撤(%)")
    trigger: str = Field("", description="触发条件描述")


class StressTestResult(BaseModel):
    """压力测试完整结果"""
    base: StressScenario = Field(default_factory=lambda: StressScenario(probability=0.5, expected_return=0.0, max_drawdown=-15.0, trigger="基准情景"))
    bull: StressScenario = Field(default_factory=lambda: StressScenario(probability=0.25, expected_return=20.0, max_drawdown=-5.0, trigger="市场上涨20%"))
    bear: StressScenario = Field(default_factory=lambda: StressScenario(probability=0.20, expected_return=-20.0, max_drawdown=-30.0, trigger="市场下跌20%"))
    black_swan: StressScenario = Field(default_factory=lambda: StressScenario(probability=0.05, expected_return=-40.0, max_drawdown=-60.0, trigger="系统性金融危机"))
    var_95: float = Field(0.0, description="VaR 95% (%)")
    cvar_95: float = Field(0.0, description="CVaR 95% (%)")
    var_method: str = Field("historical", description="VaR计算方法")
    max_consecutive_losses: int = Field(0, ge=0, description="历史最大连续亏损天数")
    blowup_probability: float = Field(0.0, ge=0.0, le=100.0, description="历史爆仓概率(%)")
    gap_risk: Dict[str, Any] = Field(default_factory=dict, description="跳空风险统计")
    liquidity_risk: Dict[str, Any] = Field(default_factory=dict, description="流动性风险指标")
    volatility_term_structure: Dict[str, Any] = Field(default_factory=dict, description="波动率期限结构")


class PortfolioPosition(BaseModel):
    """组合中的单个持仓"""
    code: str = Field("", description="股票代码")
    name: str = Field("", description="股票名称")
    value: float = Field(0.0, ge=0.0, description="持仓市值")
    weight: float = Field(0.0, ge=0.0, le=1.0, description="权重")
    beta: float = Field(1.0, description="个股Beta")
    volatility: float = Field(20.0, ge=0.0, description="年化波动率(%)")
    industry: str = Field("", description="所属行业")
    max_drawdown: float = Field(-15.0, description="历史最大回撤(%)")


class PortfolioRiskResult(BaseModel):
    """组合风险分析结果"""
    # 组合Beta
    current_beta: float = Field(1.0, description="当前组合Beta")
    new_beta: float = Field(1.0, description="新增后组合Beta")
    beta_change_pct: float = Field(0.0, description="Beta变动幅度(%)")
    beta_status: str = Field("正常", description="Beta状态: 正常/偏高/超标")

    # 行业集中度
    industry_hhi: float = Field(0.0, description="行业赫芬达尔指数(HHI)")
    max_industry_pct: float = Field(0.0, ge=0.0, le=100.0, description="最大行业占比(%)")
    industry_overlap: str = Field("无", description="行业重叠度: 无/部分/高度重叠")
    concentration_risk: str = Field("低", description="集中度风险: 低/中/高")

    # 组合波动率与VaR
    current_volatility: float = Field(20.0, ge=0.0, description="当前组合年化波动率(%)")
    new_volatility: float = Field(20.0, ge=0.0, description="新增后组合年化波动率(%)")
    portfolio_var_95: float = Field(0.0, description="组合VaR 95% (%)")

    # 边际风险贡献
    marginal_risk_contribution: float = Field(0.0, description="边际风险贡献(%)")
    risk_contribution_pct: float = Field(0.0, description="风险贡献占比(%)")

    # 组合回撤估计
    estimated_max_drawdown: float = Field(-15.0, description="估算组合最大回撤(%)")

    # 仓位约束
    position_constraint: str = Field("可配置", description="仓位约束建议")
    recommended_max_position: float = Field(0.10, ge=0.05, le=0.50, description="考虑组合后的建议最大仓位")

    # 持仓明细摘要
    existing_position_count: int = Field(0, ge=0, description="现有持仓数量")
    new_weight_pct: float = Field(0.0, ge=0.0, le=100.0, description="新增股票权重(%)")


class StopLossStrategy(BaseModel):
    """单止损策略"""
    stop_price: float = Field(0.0, description="止损价格")
    stop_pct: float = Field(0.0, description="止损幅度(%)")
    strategy_type: str = Field("", description="策略类型")
    description: str = Field("", description="策略描述")
    pros: str = Field("", description="优点")
    cons: str = Field("", description="缺点")
    suitable_for: str = Field("", description="适用场景")


class DynamicStopLossResult(BaseModel):
    """动态止损策略结果"""
    # 多种止损策略
    volatility_adaptive: StopLossStrategy = Field(default_factory=StopLossStrategy, description="波动率自适应止损")
    atr_based_1x: StopLossStrategy = Field(default_factory=StopLossStrategy, description="ATR 1x 止损")
    atr_based_2x: StopLossStrategy = Field(default_factory=StopLossStrategy, description="ATR 2x 止损")
    atr_based_3x: StopLossStrategy = Field(default_factory=StopLossStrategy, description="ATR 3x 止损")
    trailing: StopLossStrategy = Field(default_factory=StopLossStrategy, description="移动止损")
    time_based: StopLossStrategy = Field(default_factory=StopLossStrategy, description="时间止损")
    technical_support: StopLossStrategy = Field(default_factory=StopLossStrategy, description="技术位止损(支撑位)")
    technical_bollinger: StopLossStrategy = Field(default_factory=StopLossStrategy, description="技术位止损(布林带下轨)")
    technical_low: StopLossStrategy = Field(default_factory=StopLossStrategy, description="技术位止损(前期低点)")

    # 推荐策略
    recommended_strategy: str = Field("atr_based_2x", description="推荐策略名称")
    recommended_stop_loss: float = Field(0.0, description="推荐止损价格")
    recommendation_reason: str = Field("", description="推荐理由")

    # 动态调整规则
    trailing_rule: str = Field("", description="移动止损上移规则")
    time_stop_rule: str = Field("", description="时间止损规则")


class RAOpinion(BaseOpinion):
    """风险Agent输出"""
    risk_level: int = Field(3, ge=1, le=5, description="风险等级")
    max_position_pct: float = Field(0.10, ge=0.05, le=0.50, description="最大仓位")
    recommended_stop_loss: Optional[float] = Field(None, description="建议止损")
    risk_reward_ratio: float = Field(1.0, ge=0.0, description="盈亏比")
    black_scenarios: List[str] = Field(default_factory=list, description="黑天鹅场景")
    position_sizing_formula: str = Field("", description="仓位公式说明")
    stress_test: StressTestResult = Field(default_factory=StressTestResult, description="压力测试结果")
    portfolio_risk: PortfolioRiskResult = Field(default_factory=PortfolioRiskResult, description="组合风险分析")
    dynamic_stop_loss: DynamicStopLossResult = Field(default_factory=DynamicStopLossResult, description="动态止损策略")


class Scenario(BaseModel):
    """情景分析"""
    probability: float = Field(0.33, ge=0.0, le=1.0)
    return_pct: float = Field(0.0)


class DissentingView(BaseModel):
    """异议记录"""
    agent: str = Field(...)
    view: str = Field(...)
    chairman_response: str = Field(...)


class ChairmanDecision(BaseModel):
    """Chairman最终决策"""
    decision: int = Field(0, ge=-1, le=1)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    position_pct: float = Field(0.0, ge=0.0, le=0.50)
    target_price: Optional[float] = Field(None)
    stop_loss: Optional[float] = Field(None)
    time_horizon: str = Field("")
    expected_return_pct: float = Field(0.0)
    risk_adjusted_score: int = Field(50, ge=0, le=100)
    reasoning: str = Field("")
    consensus_factors: List[str] = Field(default_factory=list)
    dissenting_views: List[DissentingView] = Field(default_factory=list)
    scenario_analysis: Dict[str, Scenario] = Field(default_factory=dict)
    execution_plan: List[str] = Field(default_factory=list)


class DecisionPackage(BaseModel):
    """完整决策包 - 最终返回给前端"""
    stock_code: str = Field(...)
    stock_name: str = Field("")
    current_price: float = Field(0.0)
    decision_date: str = Field("")
    decision_time: str = Field("")
    market_cycle: str = Field("")
    
    # 各Agent观点
    opinions: Dict[str, Any] = Field(default_factory=dict)
    
    # Chairman决策
    final_decision: ChairmanDecision = Field(default_factory=ChairmanDecision)
    
    # 元信息
    disclaimer: str = Field("")
    version: str = Field("1.0.0")
    processing_time_seconds: float = Field(0.0)
    
    # 原始数据摘要
    data_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # 风险指标
    risk_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # 技术指标
    indicators: Dict[str, Any] = Field(default_factory=dict)
    
    # 数据质量报告
    data_quality: Dict[str, Any] = Field(default_factory=dict)
