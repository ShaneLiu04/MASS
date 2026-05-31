"""
MASS Flask Blueprint: backtest_bp — v2.0 增强版
集成 LLM 预测 + 策略解读 + 增强回测结果
仅支持真实K线数据，拒绝任何模拟数据回退
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from loguru import logger

from api.common import (
    get_orchestrator,
    cache,
)
from api.middleware import RateLimiter


backtest_bp = Blueprint('backtest', __name__, url_prefix='/api/agent')

# 支持的策略列表（用于前端展示）
SUPPORTED_STRATEGIES = [
    {"id": "ma_cross", "name": "MA均线交叉", "desc": "MA5/MA20金叉买入，死叉卖出"},
    {"id": "momentum_rsi", "name": "RSI动量", "desc": "RSI超卖(<30)买入，超买(>70)卖出"},
    {"id": "macd_signal", "name": "MACD信号", "desc": "DIF上穿DEA买入，下穿卖出"},
    {"id": "bollinger_breakout", "name": "布林带突破", "desc": "突破上轨买入，跌破中轨卖出"},
    {"id": "multi_factor", "name": "多因子共振", "desc": "MA金叉+RSI未超买+MACD红柱放大"},
    {"id": "multi_agent", "name": "多Agent加权决策", "desc": "模拟6大Agent信号+DecisionEngine动态权重+数据闭环学习"},
]


@backtest_bp.route('/backtest/strategies', methods=['GET'])
def list_strategies():
    """获取支持的策略列表"""
    return jsonify({
        "code": "OK",
        "strategies": SUPPORTED_STRATEGIES,
    })


@backtest_bp.route('/backtest/run', methods=['POST'])
@RateLimiter().limit(max_requests=10, window=60)
def run_backtest():
    """
    量化回测接口 v2.0 — 基于真实历史K线的多策略回测引擎 + LLM预测

    Request:
        {
            "stock_code": "000001",
            "strategy": "ma_cross",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
            "commission_rate": 0.0003,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.20,
            "include_llm_explanation": true,
            "include_llm_prediction": true,
            "prediction_horizon": "short"
        }

    支持策略:
        - ma_cross:         MA5/MA20均线交叉
        - momentum_rsi:     RSI动量
        - macd_signal:      MACD信号
        - bollinger_breakout: 布林带突破
        - multi_factor:     多因子共振
    """
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()
        strategy = data.get('strategy', 'ma_cross')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not stock_code:
            return jsonify({"error": "stock_code 不能为空", "code": "MISSING_STOCK_CODE"}), 400

        stock_code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
        if not stock_code.isdigit() or len(stock_code) != 6:
            return jsonify({"error": "stock_code 必须是6位数字", "code": "INVALID_STOCK_CODE"}), 400

        valid_strategies = [s["id"] for s in SUPPORTED_STRATEGIES]
        if strategy not in valid_strategies:
            return jsonify({
                "error": f"不支持的策略: {strategy}",
                "code": "INVALID_STRATEGY",
                "supported": valid_strategies,
            }), 400

        include_llm_explanation = data.get('include_llm_explanation', True)
        include_llm_prediction = data.get('include_llm_prediction', True)
        prediction_horizon = data.get('prediction_horizon', 'short')

        # ── 获取历史K线数据（仅真实数据，拒绝模拟） ──
        from agent.tools.stock_data_tool import StockDataTool
        from agent.core.backtest_engine import BacktestEngine, BacktestConfig

        stock_tool = StockDataTool()
        kline_df = stock_tool.get_kline(stock_code, days=365)
        data_source = "真实历史K线"

        if kline_df is None or len(kline_df) < 60:
            logger.error(f"真实K线获取失败 [{stock_code}]，拒绝模拟数据回退")
            return jsonify({
                "code": "DATA_UNAVAILABLE",
                "error": "无法获取足够的历史K线数据进行回测",
                "message": "真实数据源暂不可用，请检查股票代码或稍后重试",
            }), 503

        # 确保必要列存在
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in kline_df.columns:
                return jsonify({
                    "code": "DATA_INVALID",
                    "error": f"K线数据缺少必要字段: {col}",
                }), 503

        # ── 构建回测配置 ──
        config = BacktestConfig(
            initial_capital=float(data.get('initial_capital', 100000)),
            commission_rate=float(data.get('commission_rate', 0.0003)),
            stop_loss_pct=float(data.get('stop_loss_pct', 0.08)),
            take_profit_pct=float(data.get('take_profit_pct', 0.20)),
        )

        # ── 可选：LLM预测 ──
        llm_prediction = None
        if include_llm_prediction:
            try:
                orchestrator = get_orchestrator()
                snapshot = orchestrator._create_snapshot(
                    stock_code=stock_code,
                    stock_name=data.get('stock_name', ''),
                )
                pred_result = orchestrator.prediction_engine.predict(
                    stock_code=stock_code,
                    stock_name=snapshot.stock_name,
                    snapshot=snapshot,
                    horizon=prediction_horizon,
                )
                llm_prediction = pred_result.model_dump()
            except Exception as e:
                logger.warning(f"回测中LLM预测获取失败: {e}")
                llm_prediction = None

        # ── multi_agent 策略：注入 DecisionEngine ──
        decision_engine = None
        if strategy == "multi_agent":
            from agent.core.decision_engine import DecisionEngine
            decision_engine = DecisionEngine()

        # ── 执行回测 ──
        engine = BacktestEngine(config)
        result = engine.run(
            df=kline_df,
            stock_code=stock_code,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            llm_explanation=include_llm_explanation,
            llm_prediction=llm_prediction,
            decision_engine=decision_engine,
        )

        # ── multi_agent 策略：附加动态权重诊断信息 ──
        dynamic_weight_info = None
        if strategy == "multi_agent" and decision_engine is not None:
            dynamic_weight_info = {
                "accuracy_stats": decision_engine.get_accuracy_stats(),
                "dynamic_weights": decision_engine.compute_dynamic_weights(market_cycle=""),
            }

        # ── 组装响应 ──
        response = {
            "code": "OK",
            "simulation": False,
            "data_source": "真实历史K线",
            "stock_code": result.stock_code,
            "strategy": result.strategy,
            "strategy_desc": result.strategy_desc,
            "period": {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "trade_days": len(result.equity_curve),
            },
            "capital": {
                "initial": result.initial_capital,
                "final": result.final_capital,
            },
            "metrics": {
                "total_return": result.total_return_pct,
                "buy_hold_return": result.buy_hold_return_pct,
                "excess_return": result.excess_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "volatility": result.volatility_annual,
                "trade_count": result.trade_count,
            },
            "equity_curve": {
                "dates": result.dates,
                "values": result.equity_curve,
            },
            "drawdown_curve": {
                "dates": result.dates,
                "values": result.drawdown_curve,
            },
            "trades": result.trades,
            "trade_analysis": result.trade_analysis,
            "monthly_returns": result.monthly_returns,
            "llm_explanation": result.llm_explanation,
        }

        if result.llm_prediction:
            response["llm_prediction"] = {
                "direction": result.llm_prediction.get("direction", "不确定"),
                "confidence": result.llm_prediction.get("confidence_calibrated", 0),
                "target_price_low": result.llm_prediction.get("target_price_low"),
                "target_price_high": result.llm_prediction.get("target_price_high"),
                "stop_loss": result.llm_prediction.get("stop_loss"),
                "risk_reward_ratio": result.llm_prediction.get("risk_reward_ratio"),
                "probability_up": result.llm_prediction.get("probability_up", 0),
                "probability_down": result.llm_prediction.get("probability_down", 0),
                "probability_sideways": result.llm_prediction.get("probability_sideways", 0),
                "holding_period_days": result.llm_prediction.get("holding_period_days"),
                "key_drivers": result.llm_prediction.get("key_drivers", []),
                "risk_factors": result.llm_prediction.get("risk_factors", []),
                "reasoning": result.llm_prediction.get("reasoning", ""),
            }

        if dynamic_weight_info:
            response["dynamic_weight_info"] = dynamic_weight_info

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e), "code": "INVALID_PARAMETER"}), 400
    except Exception as e:
        logger.exception("回测接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


@backtest_bp.route('/backtest/compare', methods=['POST'])
@RateLimiter().limit(max_requests=5, window=60)
def compare_strategies():
    """
    多策略对比回测 — 同时运行多个策略并返回对比结果

    Request:
        {
            "stock_code": "000001",
            "strategies": ["ma_cross", "macd_signal", "multi_factor"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000
        }
    """
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()
        strategies = data.get('strategies', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not stock_code:
            return jsonify({"error": "stock_code 不能为空", "code": "MISSING_STOCK_CODE"}), 400

        stock_code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
        if not stock_code.isdigit() or len(stock_code) != 6:
            return jsonify({"error": "stock_code 必须是6位数字", "code": "INVALID_STOCK_CODE"}), 400

        if not strategies or len(strategies) < 2:
            return jsonify({"error": "请至少选择2个策略进行对比", "code": "NEED_MORE_STRATEGIES"}), 400

        if len(strategies) > 3:
            return jsonify({"error": "最多支持3个策略同时对比", "code": "TOO_MANY_STRATEGIES"}), 400

        valid_strategies = [s["id"] for s in SUPPORTED_STRATEGIES]
        invalid = [s for s in strategies if s not in valid_strategies]
        if invalid:
            return jsonify({
                "error": f"不支持的策略: {invalid}",
                "code": "INVALID_STRATEGY",
                "supported": valid_strategies,
            }), 400

        # ── 获取K线数据 ──
        from agent.tools.stock_data_tool import StockDataTool
        from agent.core.backtest_engine import BacktestEngine, BacktestConfig

        stock_tool = StockDataTool()
        kline_df = stock_tool.get_kline(stock_code, days=365)

        if kline_df is None or len(kline_df) < 60:
            return jsonify({
                "code": "DATA_UNAVAILABLE",
                "error": "无法获取足够的历史K线数据进行回测",
            }), 503

        # ── 执行多策略回测 ──
        config = BacktestConfig(
            initial_capital=float(data.get('initial_capital', 100000)),
            commission_rate=float(data.get('commission_rate', 0.0003)),
            stop_loss_pct=float(data.get('stop_loss_pct', 0.08)),
            take_profit_pct=float(data.get('take_profit_pct', 0.20)),
        )

        # multi_agent 策略需要 DecisionEngine
        decision_engine = None
        if "multi_agent" in strategies:
            from agent.core.decision_engine import DecisionEngine
            decision_engine = DecisionEngine()

        results = []
        for strategy in strategies:
            engine = BacktestEngine(config)
            result = engine.run(
                df=kline_df,
                stock_code=stock_code,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                llm_explanation=False,
                llm_prediction=None,
                decision_engine=decision_engine if strategy == "multi_agent" else None,
            )
            results.append({
                "strategy": result.strategy,
                "strategy_desc": result.strategy_desc,
                "total_return": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "trade_count": result.trade_count,
                "excess_return": result.excess_return_pct,
                "equity_curve": result.equity_curve,
                "dates": result.dates,
            })

        # 排序：按累计收益降序
        results.sort(key=lambda x: x["total_return"], reverse=True)

        return jsonify({
            "code": "OK",
            "stock_code": stock_code,
            "period": {"start_date": start_date, "end_date": end_date},
            "results": results,
        })

    except Exception as e:
        logger.exception("策略对比接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500
