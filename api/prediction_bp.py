"""
MASS Flask Blueprint: prediction_bp
"""
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from loguru import logger

from api.common import (
    get_orchestrator, get_db,
    safe_save_decision, safe_save_prediction,
    generate_cache_key, aggregate_portfolio_risk,
    cache, PORTFOLIO_EXECUTOR, DIAGNOSIS_EXECUTOR, DB_SAVE_EXECUTOR,
)
from api.middleware import RateLimiter

prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/agent')

@prediction_bp.route('/predict', methods=['POST'])
@RateLimiter().limit(max_requests=15, window=60)
def predict_stock():
    """
    股票走势预测 v2.3 — 分层 Prompt + 可调参数 + 置信度校准 + 多模型 fallback

    Request:
        {
            "stock_code": "600000",
            "stock_name": "浦发银行",           // optional
            "horizon": "short",                 // short(1-5天)/medium(1-4周)/long(1-3月)
            "risk_tolerance": "moderate",        // conservative/moderate/aggressive (new)
            "investment_style": "swing",         // swing/trend/value (new)
            "confidence_threshold": 0.6,         // 0.4-0.8, 低于此值方向降级为"不确定" (new)
            "force_refresh": false,              // optional
            "model_params": {                    // optional
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 4096
            }
        }

    Response:
        PredictionResult (v2.3 enhanced, 含 stop_loss/risk_reward_ratio/confidence_calibrated)
    """
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()

        if not stock_code:
            return jsonify({"error": "stock_code 不能为空", "code": "MISSING_STOCK_CODE"}), 400

        stock_code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
        if not stock_code.isdigit() or len(stock_code) != 6:
            return jsonify({"error": "stock_code 必须是6位数字", "code": "INVALID_STOCK_CODE"}), 400

        horizon = data.get('horizon', 'short')
        if horizon not in ('short', 'medium', 'long'):
            return jsonify({"error": "horizon 必须是 short/medium/long 之一", "code": "INVALID_HORIZON"}), 400

        risk_tolerance = data.get('risk_tolerance', 'moderate')
        if risk_tolerance not in ('conservative', 'moderate', 'aggressive'):
            return jsonify({"error": "risk_tolerance 必须是 conservative/moderate/aggressive 之一", "code": "INVALID_RISK_TOLERANCE"}), 400

        investment_style = data.get('investment_style', 'swing')
        if investment_style not in ('swing', 'trend', 'value'):
            return jsonify({"error": "investment_style 必须是 swing/trend/value 之一", "code": "INVALID_INVESTMENT_STYLE"}), 400

        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        confidence_threshold = max(0.3, min(0.9, confidence_threshold))

        force_refresh = data.get('force_refresh', False)
        model_params = data.get('model_params')

        # ── 缓存检查 ──
        cache_key = generate_cache_key(
            "predict_v2", code=stock_code, horizon=horizon,
            risk=risk_tolerance, style=investment_style,
            date=datetime.now().strftime("%Y%m%d%H"),
        )
        if not force_refresh and not model_params:
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"预测缓存命中: {stock_code}/{horizon}/{risk_tolerance}/{investment_style}")
                cached["from_cache"] = True
                return jsonify(cached)

        orchestrator = _get_orchestrator()
        snapshot = orchestrator._create_snapshot(
            stock_code=stock_code,
            stock_name=data.get('stock_name', ''),
        )

        result = orchestrator.prediction_engine.predict(
            stock_code=stock_code,
            stock_name=data.get('stock_name', snapshot.stock_name),
            snapshot=snapshot,
            horizon=horizon,
            risk_tolerance=risk_tolerance,
            investment_style=investment_style,
            confidence_threshold=confidence_threshold,
            model_params=model_params,
        )

        response = result.model_dump()
        response["from_cache"] = False

        if not model_params:
            cache.set(cache_key, response, ttl=300)

        # ── fire-and-forget 保存预测记录 ──
        _DB_SAVE_EXECUTOR.submit(safe_save_prediction, response)

        return jsonify(response)

    except Exception as e:
        from agent.core.exceptions import DataError
        if isinstance(e, DataError):
            return jsonify({
                "error": str(e),
                "code": "DATA_UNAVAILABLE",
                "message": "该股票的关键数据暂时无法获取，请稍后重试。",
            }), 503
        logger.exception("预测接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


