"""
MASS Flask Blueprint: portfolio_bp
"""
import json
from concurrent.futures import as_completed

from flask import Blueprint, jsonify, request, current_app, Response, stream_with_context
from loguru import logger

from api.common import (
    get_orchestrator, get_db,
    safe_save_decision, safe_save_prediction,
    generate_cache_key, aggregate_portfolio_risk,
    cache, PORTFOLIO_EXECUTOR, DIAGNOSIS_EXECUTOR, DB_SAVE_EXECUTOR,
)
from api.middleware import RateLimiter

portfolio_bp = Blueprint('portfolio', __name__, url_prefix='/api/agent')

@portfolio_bp.route('/stock/name', methods=['GET'])
def get_stock_name():
    """查询股票名称"""
    code = request.args.get('code', '').strip()
    if not code or len(code) != 6:
        return jsonify({"error": "股票代码必须是6位数字", "code": "INVALID_CODE"}), 400
    try:
        from agent.crawlers.eastmoney import EastMoneyCrawler
        crawler = EastMoneyCrawler()
        data = crawler.fetch(code, "fundamentals")
        if data and data.get("company_name"):
            return jsonify({"code": code, "name": data["company_name"]})
        return jsonify({"code": code, "name": ""})
    except Exception as e:
        logger.warning(f"查询股票名称失败 {code}: {e}")
        return jsonify({"code": code, "name": ""})

@portfolio_bp.route('/portfolio/analyze', methods=['POST'])
@RateLimiter().limit(max_requests=10, window=60)
def analyze_portfolio():
    """
    组合级多智能体分析 — 并行诊断
    
    Request:
        {
            "holdings": [
                {"code": "000001", "cost": 15.2, "shares": 1000, "name": "平安银行"},
                ...
            ]
        }
    """
    try:
        # Robust JSON parsing
        data = request.get_json(silent=True)
        if data is None and request.data:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        
        holdings = data.get('holdings', [])
        
        if not holdings:
            return jsonify({"error": "holdings 不能为空", "code": "MISSING_HOLDINGS"}), 400
        
        if len(holdings) > 20:
            return jsonify({"error": "持仓数量不能超过20只", "code": "TOO_MANY_HOLDINGS"}), 400
        
        orchestrator = get_orchestrator()
        results = []
        errors = []
        
        def _diagnose_one(holding: dict) -> dict:
            """单只股票诊断任务"""
            code = holding.get('code', '').strip()
            if not code:
                return {"_error": True, "code": "", "error": "empty code"}
            
            user_position = {
                "cost": holding.get('cost', 0),
                "shares": holding.get('shares', 0),
                "current_value": holding.get('cost', 0) * holding.get('shares', 0),
            }
            
            result = orchestrator.run_diagnosis(
                stock_code=code,
                stock_name=holding.get('name', ''),
                user_position=user_position,
            )
            return result
        
        # 并行执行所有诊断任务
        futures = {
            PORTFOLIO_EXECUTOR.submit(_diagnose_one, h): h
            for h in holdings if h.get('code', '').strip()
        }
        
        for future in as_completed(futures):
            holding = futures[future]
            code = holding.get('code', '').strip()
            try:
                result = future.result(timeout=180)  # 单只最大3分钟
                if result.get("_error"):
                    errors.append({"code": code, "error": result.get("error", "unknown")})
                else:
                    results.append(result)
            except Exception as e:
                logger.warning(f"组合分析中 {code} 失败: {e}")
                errors.append({"code": code, "error": str(e)})
        
        # 组合风险汇总
        portfolio_risk = aggregate_portfolio_risk(results)
        
        return jsonify({
            "holdings_analysis": results,
            "portfolio_risk": portfolio_risk,
            "errors": errors,
            "total": len(holdings),
            "success": len(results),
            "failed": len(errors),
        })
    
    except Exception as e:
        logger.exception("组合分析接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


@portfolio_bp.route('/portfolio/analyze/stream', methods=['POST'])
@RateLimiter().limit(max_requests=10, window=60)
def analyze_portfolio_stream():
    """
    组合级多智能体分析 — SSE 流式响应
    每完成一只股票即推送一次事件，最后推送汇总结果
    """
    # Robust JSON parsing
    data = request.get_json(silent=True)
    if data is None and request.data:
        try:
            data = json.loads(request.data.decode('utf-8'))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    
    holdings = data.get('holdings', [])
    
    if not holdings:
        def _err():
            yield f"data: {json.dumps({'stage': 'error', 'message': 'holdings 不能为空', 'code': 'MISSING_HOLDINGS'}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(_err()), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    
    if len(holdings) > 20:
        def _err():
            yield f"data: {json.dumps({'stage': 'error', 'message': '持仓数量不能超过20只', 'code': 'TOO_MANY_HOLDINGS'}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(_err()), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    
    orchestrator = get_orchestrator()
    
    def _diagnose_one(holding: dict) -> dict:
        code = holding.get('code', '').strip()
        if not code:
            return {"_error": True, "code": "", "error": "empty code"}
        user_position = {
            "cost": holding.get('cost', 0),
            "shares": holding.get('shares', 0),
            "current_value": holding.get('cost', 0) * holding.get('shares', 0),
        }
        result = orchestrator.run_diagnosis(
            stock_code=code,
            stock_name=holding.get('name', ''),
            user_position=user_position,
        )
        return result
    
    def generate():
        total = len(holdings)
        valid_holdings = [h for h in holdings if h.get('code', '').strip()]
        total = len(valid_holdings)
        
        # start event
        yield f"data: {json.dumps({'stage': 'start', 'message': f'开始并行诊断 {total} 只持仓', 'total': total, 'progress': 0}, ensure_ascii=False)}\n\n"
        
        futures = {
            PORTFOLIO_EXECUTOR.submit(_diagnose_one, h): h
            for h in valid_holdings
        }
        
        results = []
        errors = []
        completed = 0
        
        for future in as_completed(futures):
            holding = futures[future]
            code = holding.get('code', '').strip()
            name = holding.get('name', '') or code
            completed += 1
            progress = int((completed / total) * 100)
            try:
                result = future.result(timeout=180)
                if result.get("_error"):
                    err_msg = result.get("error", "unknown")
                    errors.append({"code": code, "error": err_msg})
                    yield f"data: {json.dumps({'stage': 'stock_error', 'message': f'{name} 诊断失败', 'code': code, 'name': name, 'error': err_msg, 'progress': progress, 'completed': completed, 'total': total}, ensure_ascii=False)}\n\n"
                else:
                    results.append(result)
                    yield f"data: {json.dumps({'stage': 'stock_done', 'message': f'{name} 诊断完成', 'code': code, 'name': name, 'progress': progress, 'completed': completed, 'total': total, 'result': result}, ensure_ascii=False, default=str)}\n\n"
            except Exception as e:
                logger.warning(f"组合分析流式中 {code} 失败: {e}")
                err_msg = str(e)
                errors.append({"code": code, "error": err_msg})
                yield f"data: {json.dumps({'stage': 'stock_error', 'message': f'{name} 诊断失败', 'code': code, 'name': name, 'error': err_msg, 'progress': progress, 'completed': completed, 'total': total}, ensure_ascii=False)}\n\n"
        
        # 汇总
        portfolio_risk = aggregate_portfolio_risk(results)
        yield f"data: {json.dumps({'stage': 'summary', 'message': '组合分析完成', 'progress': 100, 'completed': completed, 'total': total, 'portfolio_risk': portfolio_risk, 'errors': errors}, ensure_ascii=False, default=str)}\n\n"
        yield f"data: {json.dumps({'stage': 'done', 'message': '分析结束', 'progress': 100}, ensure_ascii=False)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@portfolio_bp.route('/positions', methods=['GET', 'POST'])
def virtual_positions():
    """模拟持仓管理"""
    db = get_db()
    username = request.args.get('username', 'default')
    
    if request.method == 'GET':
        positions = db.get_virtual_positions(username)
        total_value = sum(p["entry_price"] * p["shares"] for p in positions)
        return jsonify({
            "positions": positions,
            "total_positions": len(positions),
            "total_value": round(total_value, 2),
        })
    
    elif request.method == 'POST':
        data = request.get_json(force=True) or {}
        
        # 校验必填字段
        required = ['stock_code', 'entry_price', 'shares']
        for field in required:
            if not data.get(field):
                return jsonify({"error": f"缺少必填字段: {field}", "code": "MISSING_FIELD"}), 400
        
        position_id = db.add_virtual_position(data)
        return jsonify({"success": True, "position_id": position_id})


@portfolio_bp.route('/positions/<int:position_id>/close', methods=['POST'])
def close_position(position_id: int):
    """平仓"""
    try:
        data = request.get_json(force=True) or {}
        exit_price = data.get('exit_price', 0)
        
        if exit_price <= 0:
            return jsonify({"error": "exit_price 必须大于0", "code": "INVALID_PRICE"}), 400
        
        db = get_db()
        db.close_virtual_position(position_id, exit_price)
        
        return jsonify({"success": True, "message": "平仓成功"})
    
    except Exception as e:
        logger.exception("平仓接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500
