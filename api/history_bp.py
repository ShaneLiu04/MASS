"""
MASS Flask Blueprint: history_bp
"""
from flask import Blueprint, jsonify, request, current_app
from loguru import logger

from api.common import (
    get_orchestrator, get_db,
    safe_save_decision, safe_save_prediction,
    generate_cache_key, aggregate_portfolio_risk,
    cache, PORTFOLIO_EXECUTOR, DIAGNOSIS_EXECUTOR, DB_SAVE_EXECUTOR,
)
from api.middleware import RateLimiter

history_bp = Blueprint('history', __name__, url_prefix='/api/agent')

@history_bp.route('/decisions/history', methods=['GET'])
def decision_history():
    """获取历史决策记录"""
    try:
        stock_code = request.args.get('stock_code', '').strip() or None
        limit = min(int(request.args.get('limit', 50)), 200)
        offset = int(request.args.get('offset', 0))
        
        db = get_db()
        decisions = db.get_decisions(stock_code, limit, offset)
        
        # 格式化输出
        formatted = []
        for d in decisions:
            formatted.append({
                "id": d["id"],
                "stock_code": d["stock_code"],
                "stock_name": d["stock_name"],
                "decision_date": d["decision_date"],
                "decision": d["decision"],
                "confidence": d["confidence"],
                "position_pct": d["position_pct"],
                "expected_return_pct": d["expected_return_pct"],
                "validated": d["validated"],
                "created_at": d["created_at"],
            })
        
        return jsonify({
            "decisions": formatted,
            "count": len(formatted),
            "limit": limit,
            "offset": offset,
        })
    
    except Exception as e:
        logger.exception("历史记录接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


@history_bp.route('/decisions/<int:decision_id>', methods=['GET'])
def decision_detail(decision_id: int):
    """获取单条决策详情"""
    try:
        db = get_db()
        decision = db.get_decision_by_id(decision_id)
        
        if decision is None:
            return jsonify({"error": "决策记录不存在", "code": "NOT_FOUND"}), 404
        
        # 解析 raw_json
        try:
            raw = json.loads(decision.get("raw_json", "{}"))
            decision["parsed_data"] = raw
        except:
            decision["parsed_data"] = {}
        
        return jsonify(decision)
    
    except Exception as e:
        logger.exception("决策详情接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


