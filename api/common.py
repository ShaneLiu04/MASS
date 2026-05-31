"""
MASS API 共享模块
提供跨 Blueprint 的辅助函数、线程池和延迟初始化工具
"""
import json
import time
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import jsonify, request, current_app
from loguru import logger

from agent.core.orchestrator import AgentOrchestrator
from agent.core.cache import cache
from agent.models.database import Database
from api.middleware import RateLimiter

# ── 线程池 ──

# 组合分析专用线程池（限制并发，避免耗尽 waitress 线程）
PORTFOLIO_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="portfolio-")

# SSE 诊断后台线程池 — 将 70-120s 的长任务从 Waitress 线程剥离
DIAGNOSIS_EXECUTOR = ThreadPoolExecutor(max_workers=10, thread_name_prefix="diagnosis-bg-")

# 数据库保存后台线程池 — fire-and-forget，从响应路径剥离 SQLite 写入耗时
DB_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-save-")

# ── 延迟初始化 ──

_orchestrator: Optional[AgentOrchestrator] = None
_db: Optional[Database] = None


def get_orchestrator() -> AgentOrchestrator:
    """延迟初始化编排器（线程安全由 GIL + 原子赋值保证）"""
    global _orchestrator
    if _orchestrator is None:
        use_mock = current_app.config.get('USE_MOCK_LLM', False)
        _orchestrator = AgentOrchestrator(use_mock_llm=use_mock)
    return _orchestrator


def get_db() -> Database:
    """延迟初始化数据库连接"""
    global _db
    if _db is None:
        _db = Database()
    return _db


# ── 辅助函数 ──

def safe_save_decision(result: Dict[str, Any]) -> None:
    """后台线程安全保存决策，异常不传播"""
    try:
        get_db().save_decision(result)
    except Exception as e:
        logger.warning(f"异步保存决策失败: {e}")


def safe_save_prediction(response: Dict[str, Any]) -> None:
    """后台保存预测记录"""
    try:
        from agent.models.prediction import PredictionResult
        record = PredictionResult(**response)
        get_db().save_prediction(record)
    except Exception as e:
        logger.warning(f"异步保存预测记录失败: {e}")


def generate_cache_key(prefix: str, **kwargs) -> str:
    """生成缓存键"""
    sorted_items = sorted(kwargs.items())
    return f"{prefix}:{json.dumps(sorted_items, ensure_ascii=False)}"


def _build_agent_matrix(results: list) -> Dict[str, Any]:
    """构建多智能体共识矩阵数据"""
    AGENT_ORDER = ["TA-Agent", "FA-Agent", "CA-Agent", "SA-Agent", "MA-Agent", "RA-Agent"]
    stocks = []
    for r in results:
        opinions = r.get("opinions", {})
        fd = r.get("final_decision", {})
        stock_matrix = {
            "code": r.get("stock_code", ""),
            "name": r.get("stock_name", r.get("stock_code", "")),
            "chairman_decision": fd.get("decision", 0),
            "chairman_confidence": fd.get("confidence", 0),
            "signals": {},
            "confidences": {},
            "raw_highlights": {},
            "risk_override": None,
            "chairman_reasoning": fd.get("reasoning", "")[:200],
        }
        for aid in AGENT_ORDER:
            op = opinions.get(aid, {})
            stock_matrix["signals"][aid] = op.get("signal", 0)
            stock_matrix["confidences"][aid] = op.get("confidence", 0)
            # 提取各Agent核心亮点
            raw = op.get("raw_data", {})
            hl = {}
            if aid == "TA-Agent":
                hl = {
                    "chart_patterns": raw.get("chart_patterns", [])[:2],
                    "trend_direction": raw.get("trend_direction", ""),
                    "target_price_low": raw.get("target_price_low"),
                    "stop_loss": raw.get("stop_loss"),
                }
            elif aid == "FA-Agent":
                hl = {
                    "fundamental_score": raw.get("fundamental_score"),
                    "sub_scores": raw.get("sub_scores", {}),
                    "valuation_gap": raw.get("valuation_gap", "")[:40],
                }
            elif aid == "CA-Agent":
                hl = {
                    "capital_score": raw.get("capital_score"),
                    "smart_money_direction": raw.get("smart_money_direction", ""),
                    "retail_vs_institutional": raw.get("retail_vs_institutional", ""),
                }
            elif aid == "SA-Agent":
                hl = {
                    "sentiment_index": raw.get("sentiment_index"),
                    "sentiment_percentile": raw.get("sentiment_percentile"),
                    "crowd_behavior": raw.get("crowd_behavior", ""),
                }
            elif aid == "MA-Agent":
                hl = {
                    "market_cycle": raw.get("market_cycle", ""),
                    "sector_outlook": raw.get("sector_outlook", ""),
                    "style_alignment": raw.get("style_alignment"),
                }
            elif aid == "RA-Agent":
                hl = {
                    "risk_level": raw.get("risk_level", 3),
                    "max_position_pct": raw.get("max_position_pct"),
                    "risk_reward_ratio": raw.get("risk_reward_ratio"),
                    "black_scenarios": raw.get("black_scenarios", [])[:2],
                }
            stock_matrix["raw_highlights"][aid] = {k: v for k, v in hl.items() if v is not None and v != ""}
        # 风险降级检测
        ra_raw = opinions.get("RA-Agent", {}).get("raw_data", {})
        rl = ra_raw.get("risk_level", 3)
        if rl >= 5:
            stock_matrix["risk_override"] = f"风险等级={rl}，强制观望"
        elif rl == 4 and stock_matrix["chairman_decision"] == 0:
            # 检查原始加权信号是否可能是买入（这里只能推断）
            pass
        stocks.append(stock_matrix)
    return {"agents": AGENT_ORDER, "stocks": stocks}


def _detect_conflicts(results: list) -> Dict[str, Any]:
    """检测Agent间冲突"""
    conflicts = []
    for r in results:
        opinions = r.get("opinions", {})
        fd = r.get("final_decision", {})
        agents_sig = {}
        for aid, op in opinions.items():
            if aid == "RA-Agent":
                continue
            sig = op.get("signal", 0)
            conf = op.get("confidence", 0)
            if sig != 0 and conf >= 0.5:
                agents_sig[aid] = {"signal": sig, "confidence": conf, "reasoning": op.get("reasoning", "")[:60]}
        # 检测冲突：存在看多和看空的Agent
        bulls = [(a, d) for a, d in agents_sig.items() if d["signal"] == 1]
        bears = [(a, d) for a, d in agents_sig.items() if d["signal"] == -1]
        if bulls and bears:
            # 取最强烈的冲突双方
            bull = max(bulls, key=lambda x: x[1]["confidence"])
            bear = max(bears, key=lambda x: x[1]["confidence"])
            final_sig = fd.get("decision", 0)
            resolution = "观望"
            if final_sig == 1:
                resolution = f"Chairman最终看多，{bull[0]}的观点占了上风"
            elif final_sig == -1:
                resolution = f"Chairman最终看空，{bear[0]}的观点占了上风"
            else:
                resolution = "Chairman决定观望，多空力量相互抵消"
            conflicts.append({
                "stock_code": r.get("stock_code", ""),
                "stock_name": r.get("stock_name", r.get("stock_code", "")),
                "agents": [bull[0], bear[0]],
                "description": f"{bull[0]}看多(买入，置信度{bull[1]['confidence']:.0%})，{bear[0]}看空(卖出，置信度{bear[1]['confidence']:.0%})",
                "resolution": resolution,
            })
    return {"total_conflicts": len(conflicts), "conflicts": conflicts[:5]}


def _build_weight_scheme(results: list) -> Dict[str, Any]:
    """根据组合中多数股票的market_cycle推断权重方案"""
    from config import WEIGHT_MAP, CYCLE_WEIGHT_MAP
    cycles = []
    for r in results:
        mc = r.get("market_cycle", "")
        if mc:
            cycles.append(mc)
    if not cycles:
        cycle = "未知"
        scheme_name = "oscillation"
    else:
        from collections import Counter
        cycle = Counter(cycles).most_common(1)[0][0]
        scheme_name = CYCLE_WEIGHT_MAP.get(cycle, "oscillation")
    weights = WEIGHT_MAP.get(scheme_name, WEIGHT_MAP["oscillation"])
    desc_map = {
        "bull_trend": "趋势市特征，提升技术面(TA)和资金面(CA)权重，降低风控(RA)权重",
        "bull_value": "价值市特征，提升基本面(FA)权重，均衡配置其他维度",
        "bear_defense": "防御市特征，大幅提升风控(RA)权重，降低技术面(TA)权重",
        "oscillation": "震荡市特征，均衡配置各维度，注重资金面(CA)和情绪面(SA)",
    }
    return {
        "cycle": cycle,
        "scheme_name": scheme_name,
        "weights": weights,
        "description": desc_map.get(scheme_name, "均衡配置"),
    }


def _build_portfolio_scenarios(results: list) -> Dict[str, Any]:
    """聚合组合情景分析"""
    bull_w, base_w, bear_w = 0, 0, 0
    bull_prob, base_prob, bear_prob = 0, 0, 0
    total_pos = 0
    for r in results:
        fd = r.get("final_decision", {})
        pos_pct = fd.get("position_pct", 0)
        total_pos += pos_pct
        scenarios = fd.get("scenario_analysis", {})
        bull = scenarios.get("bull", {})
        base = scenarios.get("base", {})
        bear = scenarios.get("bear", {})
        bull_w += (bull.get("return_pct", 0) or 0) * pos_pct
        base_w += (base.get("return_pct", 0) or 0) * pos_pct
        bear_w += (bear.get("return_pct", 0) or 0) * pos_pct
        bull_prob += (bull.get("probability", 0) or 0) * pos_pct
        base_prob += (base.get("probability", 0) or 0) * pos_pct
        bear_prob += (bear.get("probability", 0) or 0) * pos_pct
    if total_pos <= 0:
        total_pos = len(results) * 0.1
        for r in results:
            fd = r.get("final_decision", {})
            scenarios = fd.get("scenario_analysis", {})
            bull = scenarios.get("bull", {})
            base = scenarios.get("base", {})
            bear = scenarios.get("bear", {})
            bull_w += (bull.get("return_pct", 0) or 0) * 0.1
            base_w += (base.get("return_pct", 0) or 0) * 0.1
            bear_w += (bear.get("return_pct", 0) or 0) * 0.1
            bull_prob += (bull.get("probability", 0) or 0) * 0.1
            base_prob += (base.get("probability", 0) or 0) * 0.1
            bear_prob += (bear.get("probability", 0) or 0) * 0.1
    return {
        "bull": {
            "probability": round(bull_prob / total_pos, 2),
            "return_pct": round(bull_w / total_pos, 1),
            "description": "若市场进入强势上涨阶段的组合表现预估",
        },
        "base": {
            "probability": round(base_prob / total_pos, 2),
            "return_pct": round(base_w / total_pos, 1),
            "description": "基准情景下的组合表现预估",
        },
        "bear": {
            "probability": round(bear_prob / total_pos, 2),
            "return_pct": round(bear_w / total_pos, 1),
            "description": "若市场走弱或进入调整阶段的组合表现预估",
        },
    }


def _build_chairman_advice(risk: Dict[str, Any], results: list) -> List[str]:
    """基于规则的组合级Chairman建议"""
    advice = []
    n = risk.get("total_stocks", 0)
    buy_pct = risk.get("signal_distribution", {}).get("buy_pct", 0)
    sell_pct = risk.get("signal_distribution", {}).get("sell_pct", 0)
    avg_risk = risk.get("avg_risk_level", 3)
    concentration = risk.get("portfolio_concentration_risk", "低")
    industries = risk.get("industry_distribution", {})
    max_ind_pct = max(industries.values()) / n * 100 if industries and n else 0

    if avg_risk >= 4:
        advice.append("[警告] 组合平均风险等级较高，建议降低整体仓位或增加低风险防御性持仓")
    elif avg_risk <= 2:
        advice.append("[提示] 组合风险可控，可适当增加权益类配置以提升收益")

    if concentration == "高":
        advice.append("[警告] 组合仓位集中度较高，建议分散配置以降低单一标的大幅波动的影响")

    if max_ind_pct > 50:
        top_ind = max(industries, key=industries.get)
        advice.append(f"[警告] 行业集中度风险：{top_ind}占比超过50%，建议跨行业分散")

    if buy_pct >= 60:
        advice.append("[机会] 组合整体看多信号强烈，当前处于进攻姿态，注意设置好止损")
    elif sell_pct >= 40:
        advice.append("[风险] 组合卖出信号较多，建议逐步减仓或转向防御性板块")
    elif buy_pct >= 30 and sell_pct >= 30:
        advice.append("[观望] 组合内部分歧较大，多空交织，建议控制仓位观望为主")

    if not advice:
        advice.append("[提示] 组合配置均衡，当前无重大风险或机会提示")

    return advice


def _build_rebalance_suggestions(
    results: list,
    industries: Dict[str, int],
    concentration: str,
    n: int,
    buy_count: int,
    sell_count: int,
    hold_count: int,
) -> List[Dict[str, Any]]:
    """
    基于多维度深度分析的再平衡建议
    
    不只是简单的买入/卖出信号，而是从风险、基本面、资金、情绪、技术面背离、
    组合集中度、风险收益比等维度给出 actionable 的调仓建议。
    """
    suggestions = []
    
    # ── 维度1: 个股风险驱动 ──
    for r in results:
        code = r.get("stock_code", "")
        name = r.get("stock_name", code)
        fd = r.get("final_decision", {})
        dec = fd.get("decision", 0)
        pos_pct = fd.get("position_pct", 0)
        opinions = r.get("opinions", {})
        ra_raw = opinions.get("RA-Agent", {}).get("raw_data", {})
        fa_raw = opinions.get("FA-Agent", {}).get("raw_data", {})
        ca_raw = opinions.get("CA-Agent", {}).get("raw_data", {})
        sa_raw = opinions.get("SA-Agent", {}).get("raw_data", {})
        ta_sig = opinions.get("TA-Agent", {}).get("signal", 0)
        ta_conf = opinions.get("TA-Agent", {}).get("confidence", 0)
        fa_sig = opinions.get("FA-Agent", {}).get("signal", 0)
        fa_conf = opinions.get("FA-Agent", {}).get("confidence", 0)
        
        risk_level = ra_raw.get("risk_level", 3)
        fundamental_score = fa_raw.get("fundamental_score", 50)
        smart_money = ca_raw.get("smart_money_direction", "")
        sentiment_idx = sa_raw.get("sentiment_index", 0)
        risk_reward = ra_raw.get("risk_reward_ratio", 1.5)
        
        # 1.1 极高风险 → 强制减仓/清仓
        if risk_level >= 5:
            suggestions.append({
                "code": code, "name": name,
                "action": "清仓", "priority": 10,
                "reason": f"RA-Agent判定风险等级={risk_level}(极高)，建议清仓避险",
                "detail": f"该持仓风险等级为最高档，波动率或回撤可能超出承受范围"
            })
        # 1.2 高风险 + 已有仓位 → 减仓至安全线
        elif risk_level == 4 and pos_pct > 0.05:
            suggestions.append({
                "code": code, "name": name,
                "action": "减仓", "priority": 9,
                "reason": f"风险等级={risk_level}(高)，建议将仓位从{pos_pct*100:.0f}%降至5%以下",
                "detail": f"高风险持仓应严格控制仓位，降低组合整体波动"
            })
        
        # 2. 基本面恶化
        if fundamental_score <= 35 and dec != -1:
            suggestions.append({
                "code": code, "name": name,
                "action": "减仓", "priority": 8,
                "reason": f"基本面评分仅{fundamental_score}分(低于35)，财务质量堪忧",
                "detail": f"FA-Agent认为该股票长期价值支撑不足，建议降低配置权重"
            })
        
        # 3. 主力资金强烈派发
        if smart_money in ("强烈派发", "派发期") and dec != -1:
            suggestions.append({
                "code": code, "name": name,
                "action": "减仓", "priority": 8,
                "reason": f"CA-Agent监测到主力资金处于'{smart_money}'阶段",
                "detail": f"机构资金正在流出，后续上涨动力可能不足，建议跟随减仓"
            })
        
        # 4. 情绪过热逆向保护
        if sentiment_idx > 0.6 and dec == 1:
            suggestions.append({
                "code": code, "name": name,
                "action": "止盈", "priority": 7,
                "reason": f"SA-Agent情绪指数={sentiment_idx:.2f}，市场处于过热状态",
                "detail": f"情绪面极度乐观时往往是阶段性高点，建议分批止盈而非追涨"
            })
        
        # 5. 技术面与基本面严重背离
        if ta_sig == 1 and ta_conf >= 0.6 and fa_sig == -1 and fa_conf >= 0.5:
            suggestions.append({
                "code": code, "name": name,
                "action": "观望", "priority": 7,
                "reason": "TA-Agent强烈看多但FA-Agent看空，技术反弹缺乏基本面支撑",
                "detail": f"技术面与基本面存在严重冲突，建议暂不操作，等待基本面验证"
            })
        
        # 6. 风险收益比过低
        if risk_reward < 1.0 and pos_pct > 0.1:
            suggestions.append({
                "code": code, "name": name,
                "action": "替换", "priority": 6,
                "reason": f"风险收益比仅{risk_reward:.1f}，承担的风险与潜在收益不匹配",
                "detail": f"建议将该仓位替换为风险收益比更优的标的"
            })
        
        # 7. 买入信号 + 低风险的优质标的
        if dec == 1 and risk_level <= 2 and fundamental_score >= 70:
            suggestions.append({
                "code": code, "name": name,
                "action": "加仓", "priority": 5,
                "reason": f"低风险(risk={risk_level})高基本面({fundamental_score}分)优质标的",
                "detail": f"基本面扎实且风险可控，可适度提高仓位占比"
            })
    
    # ── 维度2: 组合层面建议 ──
    
    # 8. 行业过度集中
    if industries:
        max_ind_count = max(industries.values())
        max_ind = max(industries, key=industries.get)
        if max_ind_count / n > 0.5:
            suggestions.append({
                "code": "", "name": max_ind,
                "action": "分散", "priority": 9,
                "reason": f"{max_ind}行业占比{max_ind_count/n*100:.0f}%，集中度风险较高",
                "detail": f"该行业在组合中占据主导地位，一旦行业性回调将拖累整体收益，建议增配其他行业"
            })
    
    # 9. 组合缺乏进攻性
    if n >= 3 and buy_count == 0 and hold_count >= n * 0.7:
        suggestions.append({
            "code": "", "name": "组合整体",
            "action": "审视", "priority": 6,
            "reason": f"组合中{buy_count}只买入信号，{hold_count}只观望，缺乏进攻性标的",
            "detail": f"当前配置过于保守，预期收益可能低于市场平均，建议审视持仓结构"
        })
    
    # 10. 组合空仓比例过高
    if n >= 3 and sell_count >= n * 0.4:
        suggestions.append({
            "code": "", "name": "组合整体",
            "action": "防御", "priority": 8,
            "reason": f"组合中{sell_count}/{n}只发出卖出信号，整体看空情绪浓厚",
            "detail": f"多只股票同时看空，可能预示系统性风险，建议降低整体仓位至50%以下"
        })
    
    # 11. 组合仓位集中度过高
    if concentration == "高":
        suggestions.append({
            "code": "", "name": "组合整体",
            "action": "分散", "priority": 7,
            "reason": "组合推荐仓位集中度为'高'，风险过于集中在少数标的上",
            "detail": f"建议将单只持仓占比控制在20%以内，通过分散降低单一黑天鹅事件的影响"
        })
    
    # 按优先级排序，高优先级在前
    suggestions.sort(key=lambda x: (-x["priority"], x["code"]))
    return suggestions


def aggregate_portfolio_risk(results: list) -> Dict[str, Any]:
    """汇总组合风险 — 增强版：增加多智能体共识矩阵、冲突检测、情景分析、Chairman建议"""
    if not results:
        return {}

    total_expected_return = 0
    total_position_pct = 0
    total_cost_value = 0
    risk_levels = []
    signals = []
    industries = {}
    risk_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    weighted_return = 0
    weighted_risk = 0
    market_cycles = []

    for r in results:
        fd = r.get("final_decision", {})
        exp_ret = fd.get("expected_return_pct", 0)
        pos_pct = fd.get("position_pct", 0.1)
        total_expected_return += exp_ret
        total_position_pct += pos_pct
        weighted_return += exp_ret * pos_pct

        # 成本市值
        user_pos = r.get("user_position", {})
        cost_val = user_pos.get("current_value", 0)
        total_cost_value += cost_val

        opinions = r.get("opinions", {})
        ra = opinions.get("RA-Agent", {})
        rl = 3
        if ra and ra.get("raw_data"):
            rl = ra["raw_data"].get("risk_level", 3)
        risk_levels.append(rl)
        risk_dist[rl] = risk_dist.get(rl, 0) + 1
        weighted_risk += rl * pos_pct

        signals.append(fd.get("decision", 0))

        # 行业分布
        industry = r.get("data_summary", {}).get("industry", "未知")
        if not industry:
            industry = "未知"
        industries[industry] = industries.get(industry, 0) + 1

        # 市场周期
        mc = r.get("market_cycle", "")
        if mc:
            market_cycles.append(mc)

    buy_count = sum(1 for s in signals if s == 1)
    sell_count = sum(1 for s in signals if s == -1)
    hold_count = sum(1 for s in signals if s == 0)
    n = len(results)

    # 集中度
    concentration = "低"
    if total_position_pct > 0.6:
        concentration = "高"
    elif total_position_pct > 0.4:
        concentration = "中"

    # 再平衡建议 — 多维度深度分析
    rebalance = _build_rebalance_suggestions(results, industries, concentration, n, buy_count, sell_count, hold_count)

    # 市场周期共识
    from collections import Counter
    market_cycle_consensus = Counter(market_cycles).most_common(1)[0][0] if market_cycles else "未知"

    # 组装基础风险数据
    base_risk = {
        "total_stocks": n,
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "hold_signals": hold_count,
        "signal_distribution": {
            "buy": buy_count,
            "hold": hold_count,
            "sell": sell_count,
            "buy_pct": round(buy_count / n * 100, 1) if n else 0,
            "hold_pct": round(hold_count / n * 100, 1) if n else 0,
            "sell_pct": round(sell_count / n * 100, 1) if n else 0,
        },
        "avg_expected_return": round(total_expected_return / n, 2) if n else 0,
        "weighted_expected_return": round(weighted_return / total_position_pct, 2) if total_position_pct > 0 else 0,
        "total_recommended_position": round(total_position_pct, 3),
        "total_cost_value": round(total_cost_value, 2),
        "max_risk_level": max(risk_levels) if risk_levels else 3,
        "avg_risk_level": round(sum(risk_levels) / len(risk_levels), 1) if risk_levels else 3,
        "weighted_risk_level": round(weighted_risk / total_position_pct, 1) if total_position_pct > 0 else 3,
        "risk_distribution": risk_dist,
        "portfolio_concentration_risk": concentration,
        "diversification_score": round(min(n / 10 * 100, 100), 1),
        "industry_distribution": industries,
        "rebalance_suggestions": rebalance,
        "market_cycle_consensus": market_cycle_consensus,
    }

    # 增强数据
    base_risk["agent_matrix"] = _build_agent_matrix(results)
    base_risk["conflict_summary"] = _detect_conflicts(results)
    base_risk["portfolio_weight_scheme"] = _build_weight_scheme(results)
    base_risk["portfolio_scenarios"] = _build_portfolio_scenarios(results)
    base_risk["chairman_portfolio_advice"] = _build_chairman_advice(base_risk, results)

    return base_risk
