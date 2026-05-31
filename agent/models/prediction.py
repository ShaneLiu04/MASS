"""
MASS 股票预测模型 - Pydantic v2
定义 PredictionEngine 的输出结构 (v2.3 enhanced)
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class PredictionResult(BaseModel):
    """股票预测结果 — v2.3 增强版：新增止损、风险收益比、校准置信度"""
    stock_code: str = Field(..., description="股票代码")
    stock_name: str = Field("", description="股票名称")
    prediction_horizon: str = Field(..., description="预测周期: short/medium/long")

    # ── 预测核心 ──
    direction: str = Field(..., description="预测方向: 上涨/下跌/震荡/不确定")
    confidence: float = Field(..., ge=0.0, le=1.0, description="原始 LLM 置信度")
    confidence_calibrated: float = Field(0.0, ge=0.0, le=1.0, description="数据质量校准后的置信度")
    data_quality_factor: float = Field(1.0, description="校准因子 (0-1)")

    # ── 价格目标 ──
    target_price_low: Optional[float] = Field(None, description="目标价下限")
    target_price_high: Optional[float] = Field(None, description="目标价上限")
    stop_loss: Optional[float] = Field(None, description="建议止损价")
    risk_reward_ratio: Optional[float] = Field(None, description="风险收益比")

    # ── 概率分布 ──
    probability_up: float = Field(0.0, ge=0.0, le=1.0, description="上涨概率")
    probability_down: float = Field(0.0, ge=0.0, le=1.0, description="下跌概率")
    probability_sideways: float = Field(0.0, ge=0.0, le=1.0, description="震荡概率")

    # ── 时间与交易 ──
    holding_period_days: Optional[int] = Field(None, description="建议持有天数")

    # ── 分析详情 ──
    key_drivers: List[str] = Field(default_factory=list, description="核心驱动因素")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素")
    catalyst_events: List[str] = Field(default_factory=list, description="潜在催化事件")
    reasoning: str = Field("", description="详细推理过程")

    # ── 请求上下文回显 ──
    risk_tolerance: str = Field("moderate", description="风险偏好: conservative/moderate/aggressive")
    investment_style: str = Field("swing", description="投资风格: swing/trend/value")
    confidence_threshold: float = Field(0.6, description="置信度阈值")

    # ── 性能度量 ──
    model_used: str = Field("", description="实际使用的模型")
    fallback_used: bool = Field(False, description="是否触发了备模型")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="模型参数")
    prompt_tokens_estimated: Optional[int] = Field(None, description="估算 Prompt token 数")
    prediction_time: str = Field("", description="预测时间")
    disclaimer: str = Field("", description="免责声明")

    @field_validator('probability_up', 'probability_down', 'probability_sideways')
    @classmethod
    def round_probabilities(cls, v: float) -> float:
        return round(v, 4)
