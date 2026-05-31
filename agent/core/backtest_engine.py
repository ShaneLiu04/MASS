"""
MASS 量化回测引擎 v2.0
基于真实历史K线的策略回测框架 — 增强版

支持策略:
- ma_cross:     MA5/MA20均线交叉（金叉买入，死叉卖出）
- momentum_rsi: RSI动量（超卖买入，超买卖出）
- macd_signal:  MACD信号（DIF上穿DEA买入，下穿卖出）
- bollinger_breakout: 布林带突破（突破上轨买入，跌破中轨卖出）
- multi_factor: 多因子共振（MA金叉 + RSI未超买 + MACD红柱放大）

增强特性:
- LLM策略表现解读
- 月度收益矩阵
- 逐笔交易分析
- 回撤曲线
"""
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003      # 手续费 0.03%
    slippage: float = 0.001              # 滑点 0.1%
    stop_loss_pct: float = 0.08          # 止损 8%
    take_profit_pct: float = 0.20        # 止盈 20%
    position_size: float = 1.0           # 仓位比例 100%
    multi_agent_lookback: int = 20       # multi_agent 策略 outcome 验证天数


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    type: str          # "买入" / "卖出"
    price: float
    shares: int
    value: float
    commission: float
    reason: str


@dataclass
class BacktestResult:
    """回测结果 — v2.0 增强版"""
    stock_code: str
    strategy: str
    strategy_desc: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    buy_hold_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    volatility_annual: float
    trade_count: int
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    dates: List[str]
    data_source: str = "真实历史K线"
    # ── v2.0 新增 ──
    monthly_returns: List[Dict[str, Any]] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trade_analysis: Dict[str, Any] = field(default_factory=dict)
    llm_explanation: str = ""
    llm_prediction: Optional[Dict[str, Any]] = None
    # ── v2.1 新增：回测闭环验证 ──
    validation_report: Dict[str, Any] = field(default_factory=dict)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算 MACD"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist


def _compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """计算布林带"""
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


class BacktestEngine:
    """回测引擎 v2.0"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        stock_code: str,
        strategy: str = "ma_cross",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        llm_explanation: bool = True,
        llm_prediction: Optional[Dict[str, Any]] = None,
        decision_engine=None,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            df: K线 DataFrame (open, high, low, close, volume, date)
            stock_code: 股票代码
            strategy: 策略名称（新增 multi_agent）
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            llm_explanation: 是否生成LLM策略解读
            llm_prediction: 预计算的LLM预测结果（可选）
            decision_engine: DecisionEngine 实例（multi_agent 策略必需）
        """
        df = df.copy()

        # 日期过滤
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]

        if len(df) < 30:
            raise ValueError("过滤后数据不足30条，无法回测")

        # multi_agent 策略校验
        if strategy == "multi_agent" and decision_engine is None:
            raise ValueError("multi_agent 策略必须提供 decision_engine 实例")

        # 策略选择
        strategy_fn, strategy_desc = self._get_strategy(strategy)

        # 执行回测
        result = self._backtest_loop(
            df, strategy_fn, stock_code, strategy, strategy_desc,
            decision_engine=decision_engine,
        )

        # 附加LLM预测（如提供）
        if llm_prediction:
            result.llm_prediction = llm_prediction

        # 生成LLM策略解读
        if llm_explanation:
            try:
                result.llm_explanation = self._generate_llm_explanation(result, df)
            except Exception as e:
                logger.warning(f"LLM策略解读生成失败: {e}")
                result.llm_explanation = "（LLM解读服务暂不可用）"

        # ── 回测闭环：决策验证器分析 Agent 表现 ──
        if strategy == "multi_agent" and decision_engine is not None:
            try:
                from agent.core.validator import DecisionValidator
                validator = DecisionValidator()
                validation = validator.validate_backtest(result, decision_engine)
                result.validation_report = validation
            except Exception as e:
                logger.warning(f"回测验证分析失败: {e}")
                result.validation_report = {"status": "error", "message": str(e)}

        return result

    def _get_strategy(self, name: str):
        """获取策略函数和描述"""
        strategies = {
            "ma_cross": (self._strategy_ma_cross, "MA5/MA20均线交叉：金叉买入，死叉卖出"),
            "momentum_rsi": (self._strategy_momentum_rsi, "RSI动量：超卖(<30)买入，超买(>70)卖出"),
            "macd_signal": (self._strategy_macd, "MACD信号：DIF上穿DEA买入，下穿卖出"),
            "bollinger_breakout": (self._strategy_bollinger, "布林带突破：突破上轨买入，跌破中轨卖出"),
            "multi_factor": (self._strategy_multi_factor, "多因子共振：MA金叉 + RSI未超买 + MACD红柱放大"),
            "multi_agent": (self._strategy_multi_agent, "多Agent加权决策回测：基于模拟Agent信号 + DecisionEngine动态权重"),
        }
        return strategies.get(name, strategies["ma_cross"])

    # ══════════════════════════════════════════════════════════════════════
    # 策略实现
    # ══════════════════════════════════════════════════════════════════════

    def _strategy_ma_cross(self, df: pd.DataFrame, i: int, position: int) -> tuple:
        """均线交叉策略: (signal, reason)"""
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20).mean()

        ma5 = df['ma5'].iloc[i]
        ma20 = df['ma20'].iloc[i]
        prev_ma5 = df['ma5'].iloc[i - 1]
        prev_ma20 = df['ma20'].iloc[i - 1]

        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(prev_ma5) or pd.isna(prev_ma20):
            return 0, ""

        if prev_ma5 <= prev_ma20 and ma5 > ma20 and position == 0:
            return 1, "MA5上穿MA20（金叉）"
        if prev_ma5 >= prev_ma20 and ma5 < ma20 and position == 1:
            return -1, "MA5下穿MA20（死叉）"
        return 0, ""

    def _strategy_momentum_rsi(self, df: pd.DataFrame, i: int, position: int) -> tuple:
        """RSI动量策略"""
        if 'rsi14' not in df.columns:
            df['rsi14'] = _compute_rsi(df['close'], 14)

        rsi = df['rsi14'].iloc[i]
        prev_rsi = df['rsi14'].iloc[i - 1]

        if pd.isna(rsi) or pd.isna(prev_rsi):
            return 0, ""

        if prev_rsi < 30 and rsi >= 30 and position == 0:
            return 1, "RSI从超卖区回升（<30→≥30）"
        if prev_rsi > 70 and rsi <= 70 and position == 1:
            return -1, "RSI从超买区回落（>70→≤70）"
        return 0, ""

    def _strategy_macd(self, df: pd.DataFrame, i: int, position: int) -> tuple:
        """MACD信号策略"""
        if 'dif' not in df.columns or 'dea' not in df.columns:
            dif, dea, _ = _compute_macd(df['close'])
            df['dif'] = dif
            df['dea'] = dea

        dif = df['dif'].iloc[i]
        dea = df['dea'].iloc[i]
        prev_dif = df['dif'].iloc[i - 1]
        prev_dea = df['dea'].iloc[i - 1]

        if pd.isna(dif) or pd.isna(dea) or pd.isna(prev_dif) or pd.isna(prev_dea):
            return 0, ""

        if prev_dif <= prev_dea and dif > dea and position == 0:
            return 1, "DIF上穿DEA（MACD金叉）"
        if prev_dif >= prev_dea and dif < dea and position == 1:
            return -1, "DIF下穿DEA（MACD死叉）"
        return 0, ""

    def _strategy_bollinger(self, df: pd.DataFrame, i: int, position: int) -> tuple:
        """布林带突破策略"""
        if 'bb_upper' not in df.columns:
            upper, mid, lower = _compute_bollinger(df['close'])
            df['bb_upper'] = upper
            df['bb_mid'] = mid
            df['bb_lower'] = lower

        close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i - 1]
        upper = df['bb_upper'].iloc[i]
        mid = df['bb_mid'].iloc[i]
        prev_upper = df['bb_upper'].iloc[i - 1]
        prev_mid = df['bb_mid'].iloc[i - 1]

        if pd.isna(upper) or pd.isna(mid):
            return 0, ""

        # 突破上轨买入
        if prev_close <= prev_upper and close > upper and position == 0:
            return 1, "价格突破布林带上轨"
        # 跌破中轨卖出
        if prev_close >= prev_mid and close < mid and position == 1:
            return -1, "价格跌破布林带中轨"
        return 0, ""

    def _strategy_multi_factor(self, df: pd.DataFrame, i: int, position: int) -> tuple:
        """多因子共振策略：MA金叉 + RSI未超买 + MACD红柱放大"""
        # 预计算指标
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20).mean()
        if 'rsi14' not in df.columns:
            df['rsi14'] = _compute_rsi(df['close'], 14)
        if 'macd_hist' not in df.columns:
            _, _, hist = _compute_macd(df['close'])
            df['macd_hist'] = hist

        ma5 = df['ma5'].iloc[i]
        ma20 = df['ma20'].iloc[i]
        prev_ma5 = df['ma5'].iloc[i - 1]
        prev_ma20 = df['ma20'].iloc[i - 1]

        rsi = df['rsi14'].iloc[i]
        prev_rsi = df['rsi14'].iloc[i - 1]

        hist = df['macd_hist'].iloc[i]
        prev_hist = df['macd_hist'].iloc[i - 1]

        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(rsi) or pd.isna(hist):
            return 0, ""

        # 买入条件：MA金叉 + RSI从超卖区回升（<50）+ MACD红柱放大
        ma_golden = prev_ma5 <= prev_ma20 and ma5 > ma20
        rsi_ok = prev_rsi < 50 and rsi >= 30 and rsi < 70
        macd_ok = prev_hist <= 0 and hist > 0

        if position == 0 and ma_golden and rsi_ok and macd_ok:
            return 1, "三因子共振：MA金叉 + RSI回升 + MACD转红"

        # 卖出条件：任一因子反向
        ma_dead = prev_ma5 >= prev_ma20 and ma5 < ma20
        rsi_overbought = prev_rsi > 70 and rsi <= 70
        macd_weak = prev_hist > 0 and hist < 0

        if position == 1 and (ma_dead or rsi_overbought or macd_weak):
            reasons = []
            if ma_dead:
                reasons.append("MA死叉")
            if rsi_overbought:
                reasons.append("RSI超买回落")
            if macd_weak:
                reasons.append("MACD转绿")
            return -1, " + ".join(reasons)

        return 0, ""

    def _strategy_multi_agent(self, df: pd.DataFrame, i: int, position: int, decision_engine) -> tuple:
        """
        多Agent加权决策策略（回测专用）
        基于历史技术指标模拟各 Agent 信号，通过 DecisionEngine 生成交易信号。
        """
        from agent.core.blackboard import AgentOpinion

        opinions = self._simulate_agent_opinions(df, i)
        if not opinions:
            return 0, ""

        weighted = decision_engine.compute_weighted_decision(
            opinions, market_cycle="", use_dynamic_weights=True, use_nonlinear_confidence=True
        )
        ra_opinion = opinions.get("RA-Agent")
        filtered = decision_engine.apply_risk_filter(weighted, ra_opinion)

        final_signal = filtered.get("final_signal", 0)
        if final_signal == 1 and position == 0:
            return 1, f"MultiAgent买入 | 得分:{weighted.get('weighted_score', 0):.3f} | 置信度:{weighted.get('overall_confidence', 0):.2f}"
        if final_signal == -1 and position == 1:
            return -1, f"MultiAgent卖出 | 得分:{weighted.get('weighted_score', 0):.3f} | 置信度:{weighted.get('overall_confidence', 0):.2f}"
        return 0, ""

    def _simulate_agent_opinions(self, df: pd.DataFrame, i: int) -> Dict[str, "AgentOpinion"]:
        """
        基于历史技术指标模拟各 Agent 的预测信号（用于回测）。
        不使用 LLM，完全基于可复现的技术规则近似 Agent 逻辑。
        """
        from agent.core.blackboard import AgentOpinion

        close = df['close']
        vol = df.get('volume', pd.Series(np.ones(len(df)), index=df.index))
        price = close.iloc[i]

        # 预计算通用指标
        if 'ma5' not in df.columns:
            df['ma5'] = close.rolling(window=5).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = close.rolling(window=20).mean()
        if 'rsi14' not in df.columns:
            df['rsi14'] = _compute_rsi(close, 14)
        if 'macd_hist' not in df.columns:
            _, _, df['macd_hist'] = _compute_macd(close)
        if 'bb_upper' not in df.columns:
            df['bb_upper'], df['bb_mid'], df['bb_lower'] = _compute_bollinger(close)

        ma5 = df['ma5'].iloc[i]
        ma20 = df['ma20'].iloc[i]
        prev_ma5 = df['ma5'].iloc[i - 1]
        prev_ma20 = df['ma20'].iloc[i - 1]
        rsi = df['rsi14'].iloc[i]
        prev_rsi = df['rsi14'].iloc[i - 1]
        hist = df['macd_hist'].iloc[i]
        prev_hist = df['macd_hist'].iloc[i - 1]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]

        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(rsi):
            return {}

        opinions: Dict[str, AgentOpinion] = {}

        # ── TA-Agent：技术面 ──
        ta_score = 0
        ta_reasons = []
        # 趋势分权重更高
        if ma5 > ma20:
            ta_score += 2
            ta_reasons.append("MA5>MA20")
        else:
            ta_score -= 2
            ta_reasons.append("MA5<MA20")
        if prev_ma5 <= prev_ma20 and ma5 > ma20:
            ta_score += 1
            ta_reasons.append("金叉")
        if prev_ma5 >= prev_ma20 and ma5 < ma20:
            ta_score -= 1
            ta_reasons.append("死叉")
        # RSI 只在与趋势反向时提供信号（避免趋势中 RSI 极端抵消趋势分）
        if rsi < 30 and ma5 <= ma20:
            ta_score += 1
            ta_reasons.append("RSI超卖+下跌")
        elif rsi > 70 and ma5 >= ma20:
            ta_score -= 1
            ta_reasons.append("RSI超买+上涨")
        if hist > 0 and prev_hist <= 0:
            ta_score += 1
            ta_reasons.append("MACD转红")
        elif hist < 0 and prev_hist >= 0:
            ta_score -= 1
            ta_reasons.append("MACD转绿")

        if ta_score >= 2:
            ta_signal, ta_conf = 1, min(0.65 + ta_score * 0.04, 0.90)
        elif ta_score <= -2:
            ta_signal, ta_conf = -1, min(0.65 + abs(ta_score) * 0.04, 0.90)
        else:
            ta_signal, ta_conf = 0, 0.60
        opinions["TA-Agent"] = AgentOpinion(
            agent_id="TA-Agent", signal=ta_signal, confidence=round(ta_conf, 3),
            reasoning="; ".join(ta_reasons),
            raw_data={"ta_score": ta_score},
        )

        # ── FA-Agent：基本面（代理：价格与长期均线关系 + 波动率）──
        if 'ma60' not in df.columns:
            df['ma60'] = close.rolling(window=60).mean()
        ma60 = df['ma60'].iloc[i]
        if not pd.isna(ma60):
            deviation = (price - ma60) / ma60
            if deviation < -0.10:
                fa_signal, fa_conf = 1, min(0.65 + abs(deviation), 0.90)
                fa_reason = f"价格低于MA60 {abs(deviation)*100:.1f}%，估值偏低"
            elif deviation > 0.15:
                fa_signal, fa_conf = -1, min(0.65 + deviation, 0.90)
                fa_reason = f"价格高于MA60 {deviation*100:.1f}%，估值偏高"
            else:
                fa_signal, fa_conf = 0, 0.60
                fa_reason = "价格在MA60附近，估值中性"
        else:
            # MA60 不足时回退到 MA20
            deviation = (price - ma20) / ma20
            if deviation < -0.08:
                fa_signal, fa_conf = 1, min(0.60 + abs(deviation), 0.85)
                fa_reason = f"价格低于MA20 {abs(deviation)*100:.1f}%，短期估值偏低"
            elif deviation > 0.12:
                fa_signal, fa_conf = -1, min(0.60 + deviation, 0.85)
                fa_reason = f"价格高于MA20 {deviation*100:.1f}%，短期估值偏高"
            else:
                fa_signal, fa_conf = 0, 0.55
                fa_reason = "价格在MA20附近，估值中性"
        opinions["FA-Agent"] = AgentOpinion(
            agent_id="FA-Agent", signal=fa_signal, confidence=round(fa_conf, 3),
            reasoning=fa_reason,
            raw_data={"deviation_from_ma60": round(deviation, 4) if not pd.isna(ma60) else None},
        )

        # ── CA-Agent：资金面（代理：成交量变化率）──
        if i >= 6:
            avg_vol = vol.iloc[i-5:i].mean()
            prev_avg_vol = vol.iloc[i-6:i-1].mean()
            if prev_avg_vol > 0:
                vol_change = (avg_vol - prev_avg_vol) / prev_avg_vol
            else:
                vol_change = 0
        else:
            vol_change = 0
        if vol_change > 0.20 and price > ma5:
            ca_signal, ca_conf = 1, min(0.60 + vol_change, 0.85)
            ca_reason = f"成交量放大{vol_change*100:.0f}%，资金流入"
        elif vol_change > 0.20 and price < ma5:
            ca_signal, ca_conf = -1, min(0.60 + vol_change, 0.85)
            ca_reason = f"成交量放大{vol_change*100:.0f}%，资金出逃"
        else:
            ca_signal, ca_conf = 0, 0.55
            ca_reason = "成交量平稳"
        opinions["CA-Agent"] = AgentOpinion(
            agent_id="CA-Agent", signal=ca_signal, confidence=round(ca_conf, 3),
            reasoning=ca_reason,
            raw_data={"volume_change_pct": round(vol_change * 100, 2)},
        )

        # ── SA-Agent：情绪面（代理：RSI极端 + 波动率）──
        if i >= 20:
            ret_20d = close.iloc[i-19:i+1].pct_change().dropna()
            vol_20d = ret_20d.std() * np.sqrt(252) if len(ret_20d) > 1 else 0
        else:
            vol_20d = 0
        sa_score = 0
        sa_reasons = []
        if rsi < 25:
            sa_score += 2
            sa_reasons.append("极度悲观（RSI<25）")
        elif rsi > 75:
            sa_score -= 2
            sa_reasons.append("极度乐观（RSI>75）")
        if vol_20d > 0.50:
            sa_score += 1 if rsi < 50 else -1
            sa_reasons.append("高波动")
        if sa_score >= 1:
            sa_signal, sa_conf = 1, 0.75
        elif sa_score <= -1:
            sa_signal, sa_conf = -1, 0.75
        else:
            sa_signal, sa_conf = 0, 0.60
        opinions["SA-Agent"] = AgentOpinion(
            agent_id="SA-Agent", signal=sa_signal, confidence=round(sa_conf, 3),
            reasoning="; ".join(sa_reasons) if sa_reasons else "情绪中性",
            raw_data={"volatility_20d": round(vol_20d, 4)},
        )

        # ── MA-Agent：宏观面（代理：大盘趋势 + 周期推断）──
        if i >= 60:
            trend_60d = (close.iloc[i] - close.iloc[i-60]) / close.iloc[i-60]
        else:
            trend_60d = 0
        if trend_60d > 0.15:
            ma_cycle = "复苏晚期"
            ma_signal, ma_conf = 1, 0.65
            ma_reason = "中期趋势向上，宏观偏多"
        elif trend_60d < -0.15:
            ma_cycle = "衰退早期"
            ma_signal, ma_conf = -1, 0.65
            ma_reason = "中期趋势向下，宏观偏空"
        else:
            ma_cycle = "振荡"
            ma_signal, ma_conf = 0, 0.5
            ma_reason = "中期趋势不明，宏观中性"
        opinions["MA-Agent"] = AgentOpinion(
            agent_id="MA-Agent", signal=ma_signal, confidence=round(ma_conf, 3),
            reasoning=ma_reason,
            raw_data={
                "market_cycle": ma_cycle,
                "trend_60d": round(trend_60d, 4),
            },
        )

        # ── RA-Agent：风险面（代理：波动率 + 回撤）──
        if i >= 20:
            recent = close.iloc[i-20:i+1]
            peak = recent.max()
            drawdown = (peak - price) / peak if peak > 0 else 0
        else:
            drawdown = 0
        if drawdown > 0.15 or vol_20d > 0.60:
            risk_level = 5
            ra_signal = 0
            ra_conf = 0.80
            ra_reason = f"高风险：回撤{drawdown*100:.1f}%，波动率{vol_20d*100:.1f}%"
        elif drawdown > 0.08 or vol_20d > 0.40:
            risk_level = 4
            ra_signal = 0
            ra_conf = 0.70
            ra_reason = f"中高风险：回撤{drawdown*100:.1f}%，波动率{vol_20d*100:.1f}%"
        else:
            risk_level = 2
            ra_signal = 0
            ra_conf = 0.60
            ra_reason = "风险可控"
        opinions["RA-Agent"] = AgentOpinion(
            agent_id="RA-Agent", signal=ra_signal, confidence=round(ra_conf, 3),
            reasoning=ra_reason,
            raw_data={
                "risk_level": risk_level,
                "max_position_pct": 0.10 if risk_level >= 4 else 0.20,
                "drawdown": round(drawdown, 4),
            },
        )

        return opinions

    # ══════════════════════════════════════════════════════════════════════
    # 回测主循环
    # ══════════════════════════════════════════════════════════════════════

    def _backtest_loop(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable,
        stock_code: str,
        strategy_name: str,
        strategy_desc: str,
        decision_engine=None,
    ) -> BacktestResult:
        """回测主循环"""
        cfg = self.config
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        dates = df['date'].astype(str).tolist() if 'date' in df.columns else [str(i) for i in range(len(df))]

        capital = cfg.initial_capital
        position = 0          # 0空仓, 1持仓
        shares = 0
        entry_price = 0.0
        entry_date = ""
        trades: List[TradeRecord] = []
        equity = [capital]

        # ── multi_agent 策略专用：Agent 预测结果跟踪与数据闭环 ──
        is_multi_agent = strategy_name == "multi_agent"
        pending_outcomes: deque = deque()  # [(day_index, agent_id, predicted_signal, price_at_signal), ...]
        lookback = cfg.multi_agent_lookback

        for i in range(1, len(df)):
            current_price = close[i]
            current_high = high[i]

            # 风控：止损/止盈检查（持仓时）
            if position == 1 and shares > 0:
                loss_pct = (entry_price - current_price) / entry_price
                profit_pct = (current_price - entry_price) / entry_price

                if loss_pct >= cfg.stop_loss_pct:
                    sell_price = current_price * (1 - cfg.slippage)
                    value = shares * sell_price
                    commission = value * cfg.commission_rate
                    capital = value - commission
                    trades.append(TradeRecord(
                        date=dates[i], type="卖出", price=round(sell_price, 2),
                        shares=shares, value=round(value, 2),
                        commission=round(commission, 2), reason=f"止损触发（跌幅{loss_pct*100:.1f}%）",
                    ))
                    position = 0
                    shares = 0
                    equity.append(capital)
                    continue

                if profit_pct >= cfg.take_profit_pct:
                    sell_price = current_price * (1 - cfg.slippage)
                    value = shares * sell_price
                    commission = value * cfg.commission_rate
                    capital = value - commission
                    trades.append(TradeRecord(
                        date=dates[i], type="卖出", price=round(sell_price, 2),
                        shares=shares, value=round(value, 2),
                        commission=round(commission, 2), reason=f"止盈触发（涨幅{profit_pct*100:.1f}%）",
                    ))
                    position = 0
                    shares = 0
                    equity.append(capital)
                    continue

            # ── 策略信号 ──
            if is_multi_agent and decision_engine is not None:
                signal, reason = self._strategy_multi_agent(df, i, position, decision_engine)
            else:
                signal, reason = strategy_fn(df, i, position)

            if signal == 1 and position == 0:
                buy_price = current_price * (1 + cfg.slippage)
                # 考虑手续费后计算最大可买股数
                effective_price = buy_price * (1 + cfg.commission_rate)
                max_shares = int((capital * cfg.position_size) / effective_price)
                if max_shares > 0:
                    cost = max_shares * buy_price
                    commission = cost * cfg.commission_rate
                    if capital >= cost + commission:
                        shares = max_shares
                        capital -= (cost + commission)
                        entry_price = buy_price
                        entry_date = dates[i]
                        position = 1
                        trades.append(TradeRecord(
                            date=dates[i], type="买入", price=round(buy_price, 2),
                            shares=shares, value=round(cost, 2),
                            commission=round(commission, 2), reason=reason,
                        ))

            elif signal == -1 and position == 1 and shares > 0:
                sell_price = current_price * (1 - cfg.slippage)
                value = shares * sell_price
                commission = value * cfg.commission_rate
                capital = value - commission
                trades.append(TradeRecord(
                    date=dates[i], type="卖出", price=round(sell_price, 2),
                    shares=shares, value=round(value, 2),
                    commission=round(commission, 2), reason=reason,
                ))
                position = 0
                shares = 0

            # ── multi_agent 数据闭环：记录到期的 Agent 预测结果 ──
            if is_multi_agent and decision_engine is not None:
                # 检查并记录到期的预测结果
                while pending_outcomes and pending_outcomes[0][0] <= i - lookback:
                    record_day, agent_id, predicted_signal, signal_price = pending_outcomes.popleft()
                    future_price = close[i]
                    actual_return_pct = (future_price - signal_price) / signal_price * 100
                    decision_engine.record_outcome(agent_id, predicted_signal, actual_return_pct)

                # 如果今日发出了非观望信号，保存各 Agent 的预测以待后续验证
                if signal != 0 and i + lookback < len(df):
                    opinions = self._simulate_agent_opinions(df, i)
                    for agent_id, op in opinions.items():
                        if agent_id != "RA-Agent":
                            pending_outcomes.append((i, agent_id, op.signal, current_price))

            # 计算当日权益
            if position == 1 and shares > 0:
                current_value = shares * current_price + capital
            else:
                current_value = capital
            equity.append(current_value)

        # 最后一天：处理剩余 pending_outcomes + 平仓
        if is_multi_agent and decision_engine is not None:
            final_price = close[-1]
            while pending_outcomes:
                record_day, agent_id, predicted_signal, signal_price = pending_outcomes.popleft()
                actual_return_pct = (final_price - signal_price) / signal_price * 100
                decision_engine.record_outcome(agent_id, predicted_signal, actual_return_pct)

        if position == 1 and shares > 0:
            final_price = close[-1] * (1 - cfg.slippage)
            value = shares * final_price
            commission = value * cfg.commission_rate
            capital = value - commission
            trades.append(TradeRecord(
                date=dates[-1], type="卖出", price=round(final_price, 2),
                shares=shares, value=round(value, 2),
                commission=round(commission, 2), reason="回测结束平仓",
            ))
            equity[-1] = capital

        # ── 计算绩效指标 ──
        equity_arr = np.array(equity)
        daily_rets = np.diff(equity_arr) / equity_arr[:-1]
        valid_rets = daily_rets[np.isfinite(daily_rets)]

        total_return = (equity_arr[-1] / cfg.initial_capital - 1) * 100
        buy_hold_return = (close[-1] - close[0]) / close[0] * 100

        # Sharpe
        if len(valid_rets) > 0 and np.std(valid_rets) > 0:
            sharpe = (np.mean(valid_rets) / np.std(valid_rets)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 最大回撤 & 回撤曲线
        cummax = np.maximum.accumulate(equity_arr)
        drawdowns = (cummax - equity_arr) / cummax
        max_dd = np.max(drawdowns) * 100
        drawdown_curve = [round(d * 100, 2) for d in drawdowns.tolist()]

        # 胜率（按交易盈亏）
        trade_profits = []
        buy_value = 0
        for t in trades:
            if t.type == "买入":
                buy_value = t.value
            elif t.type == "卖出" and buy_value > 0:
                profit = t.value - t.commission - buy_value
                trade_profits.append(profit)
                buy_value = 0

        win_count = sum(1 for p in trade_profits if p > 0)
        win_rate = (win_count / len(trade_profits) * 100) if trade_profits else 0

        # 年化波动率
        vol = np.std(valid_rets) * np.sqrt(252) * 100 if len(valid_rets) > 0 else 0

        # ── v2.0 新增：交易分析 ──
        trade_analysis = self._analyze_trades(trades, dates)

        # ── v2.0 新增：月度收益 ──
        monthly_returns = self._compute_monthly_returns(dates, equity_arr)

        return BacktestResult(
            stock_code=stock_code,
            strategy=strategy_name,
            strategy_desc=strategy_desc,
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=cfg.initial_capital,
            final_capital=round(equity_arr[-1], 2),
            total_return_pct=round(total_return, 2),
            buy_hold_return_pct=round(buy_hold_return, 2),
            excess_return_pct=round(total_return - buy_hold_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            win_rate=round(win_rate, 1),
            volatility_annual=round(vol, 2),
            trade_count=len(trades),
            trades=[self._trade_to_dict(t) for t in trades[:50]],
            equity_curve=[round(v, 2) for v in equity_arr.tolist()],
            dates=dates,
            monthly_returns=monthly_returns,
            drawdown_curve=drawdown_curve,
            trade_analysis=trade_analysis,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 交易分析
    # ══════════════════════════════════════════════════════════════════════

    def _analyze_trades(self, trades: List[TradeRecord], dates: List[str]) -> Dict[str, Any]:
        """逐笔交易统计分析"""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_holding_days": 0,
                "best_trade": None,
                "worst_trade": None,
                "avg_profit_per_trade": 0,
                "profit_factor": 0,
            }

        profits = []
        holding_days = []
        buy_value = 0
        buy_date = ""
        best_trade = {"profit": -float('inf'), "date": ""}
        worst_trade = {"profit": float('inf'), "date": ""}
        gross_profit = 0
        gross_loss = 0

        for t in trades:
            if t.type == "买入":
                buy_value = t.value
                buy_date = t.date
            elif t.type == "卖出" and buy_value > 0:
                profit = t.value - t.commission - buy_value
                profits.append(profit)

                # 持仓天数
                try:
                    bd = datetime.strptime(buy_date, "%Y-%m-%d")
                    sd = datetime.strptime(t.date, "%Y-%m-%d")
                    holding_days.append((sd - bd).days)
                except ValueError:
                    holding_days.append(0)

                # 最佳/最差交易
                if profit > best_trade["profit"]:
                    best_trade = {"profit": round(profit, 2), "date": t.date, "return_pct": round(profit / buy_value * 100, 2)}
                if profit < worst_trade["profit"]:
                    worst_trade = {"profit": round(profit, 2), "date": t.date, "return_pct": round(profit / buy_value * 100, 2)}

                if profit > 0:
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)

                buy_value = 0

        winning = sum(1 for p in profits if p > 0)
        losing = sum(1 for p in profits if p <= 0)
        avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0
        avg_profit = sum(profits) / len(profits) if profits else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else round(gross_profit, 2)

        return {
            "total_trades": len(profits),
            "winning_trades": winning,
            "losing_trades": losing,
            "avg_holding_days": round(avg_holding, 1),
            "best_trade": best_trade if best_trade["profit"] > -float('inf') else None,
            "worst_trade": worst_trade if worst_trade["profit"] < float('inf') else None,
            "avg_profit_per_trade": round(avg_profit, 2),
            "profit_factor": profit_factor,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }

    def _compute_monthly_returns(self, dates: List[str], equity_arr: np.ndarray) -> List[Dict[str, Any]]:
        """计算月度收益矩阵"""
        if len(dates) < 30:
            return []

        try:
            df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity_arr})
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            # 每月末权益值
            monthly = df.groupby(["year", "month"]).last().reset_index()
            monthly["prev_equity"] = monthly["equity"].shift(1)
            monthly["return_pct"] = (monthly["equity"] / monthly["prev_equity"] - 1) * 100

            # 去掉第一个不完整月份
            monthly = monthly.dropna(subset=["return_pct"])

            return [
                {
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "return_pct": round(row["return_pct"], 2),
                    "equity": round(row["equity"], 2),
                }
                for _, row in monthly.iterrows()
            ]
        except Exception as e:
            logger.warning(f"月度收益计算失败: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════
    # LLM 策略解读
    # ══════════════════════════════════════════════════════════════════════

    def _generate_llm_explanation(self, result: BacktestResult, df: pd.DataFrame) -> str:
        """
        调用LLM生成策略表现解读。
        当 LLM 不可用时优雅降级。
        """
        try:
            from agent.tools.llm_client import LLMClient
            llm = LLMClient()
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败: {e}")
            return "（LLM服务暂不可用，无法生成策略解读）"

        # 构建精简的上下文
        kline_summary = self._build_kline_summary(df)
        trade_summary = result.trade_analysis

        system_prompt = self._load_backtest_prompt()

        user_prompt = f"""# 回测结果摘要

## 基本信息
- 股票代码: {result.stock_code}
- 策略: {result.strategy_desc}
- 回测区间: {result.start_date} ~ {result.end_date}
- 初始资金: {result.initial_capital:,.0f}元

## 绩效指标
- 累计收益: {result.total_return_pct:.2f}%
- 买入持有收益: {result.buy_hold_return_pct:.2f}%
- 超额收益: {result.excess_return_pct:.2f}%
- 夏普比率: {result.sharpe_ratio:.2f}
- 最大回撤: {result.max_drawdown_pct:.2f}%
- 胜率: {result.win_rate:.1f}%
- 交易次数: {result.trade_count}

## 交易统计
- 盈利交易: {trade_summary.get('winning_trades', 0)}
- 亏损交易: {trade_summary.get('losing_trades', 0)}
- 平均持仓天数: {trade_summary.get('avg_holding_days', 0)}
- 盈亏比: {trade_summary.get('profit_factor', 0)}
- 最佳单笔: {trade_summary.get('best_trade', {})}
- 最差单笔: {trade_summary.get('worst_trade', {})}

## K线特征
{kline_summary}

请基于以上数据，以中文生成一段专业的策略表现解读（300-500字），包括：
1. 策略在该股票上的有效性评估
2. 与买入持有策略的对比分析
3. 该策略适合的市场环境
4. 风险提示与改进建议
"""

        try:
            response = llm.chat(
                system=system_prompt,
                user=user_prompt,
                json_mode=False,
            )
            explanation = response.get("content", "") if isinstance(response, dict) else str(response)
            return explanation.strip() if explanation else "（LLM返回空解读）"
        except Exception as e:
            logger.warning(f"LLM策略解读调用失败: {e}")
            return "（LLM策略解读服务暂不可用）"

    @staticmethod
    def _load_backtest_prompt() -> str:
        """加载回测解读系统提示词"""
        from pathlib import Path
        prompt_path = Path("agent/prompts/system/backtest_explanation.md")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        # 默认提示词
        return """你是一位资深量化策略分析师，擅长评估交易策略的历史表现并给出专业解读。

要求：
- 基于真实数据，客观分析策略优劣
- 指出策略适合/不适合的市场环境
- 给出具体、可操作的改进建议
- 语言专业但通俗易懂
- 严禁编造不存在的数据
"""

    @staticmethod
    def _build_kline_summary(df: pd.DataFrame) -> str:
        """构建K线摘要用于LLM提示词"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series())

            total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
            max_price = high.max()
            min_price = low.min()
            avg_volume = volume.mean() if not volume.empty else 0

            # 20日波动率
            if len(close) >= 20:
                returns_20d = close.pct_change().tail(20)
                vol_20d = returns_20d.std() * np.sqrt(252) * 100
            else:
                vol_20d = 0

            return f"""- 区间总涨跌幅: {total_return:.2f}%
- 最高价: {max_price:.2f}
- 最低价: {min_price:.2f}
- 平均成交量: {avg_volume:,.0f}
- 20日年化波动率: {vol_20d:.2f}%
- 数据条数: {len(df)}"""
        except Exception:
            return "- K线摘要暂不可用"

    @staticmethod
    def _trade_to_dict(t: TradeRecord) -> Dict[str, Any]:
        return {
            "date": t.date,
            "type": t.type,
            "price": t.price,
            "shares": t.shares,
            "value": t.value,
            "commission": t.commission,
            "reason": t.reason,
        }
