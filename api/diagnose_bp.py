"""
MASS Flask Blueprint: diagnose_bp
"""
import json
import queue
import time
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

diagnose_bp = Blueprint('diagnose', __name__, url_prefix='/api/agent')

@diagnose_bp.route('/diagnose', methods=['POST'])
@RateLimiter().limit(max_requests=10, window=60)
def diagnose_stock():
    """
    单只股票多智能体诊断
    
    Request:
        {
            "stock_code": "000001",
            "stock_name": "平安银行",  // optional
            "market_type": null,       // optional
            "user_position": null,     // optional
            "force_refresh": false     // optional, 强制刷新缓存
        }
    
    Response:
        DecisionPackage
    """
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()
        force_refresh = data.get('force_refresh', False)
        
        if not stock_code:
            return jsonify({"error": "stock_code 不能为空", "code": "MISSING_STOCK_CODE"}), 400
        
        # 清理股票代码
        stock_code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
        if not stock_code.isdigit() or len(stock_code) != 6:
            return jsonify({"error": "stock_code 必须是6位数字", "code": "INVALID_STOCK_CODE"}), 400
        
        # 缓存检查
        cache_key = generate_cache_key("diagnose", code=stock_code, date=datetime.now().strftime("%Y%m%d%H"))
        
        if not force_refresh:
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"返回缓存结果: {stock_code}")
                cached["from_cache"] = True
                return jsonify(cached)
        
        # 执行诊断（支持模型参数覆盖）
        model_params = data.get('model_params')
        orchestrator = get_orchestrator()
        result = orchestrator.run_diagnosis(
            stock_code=stock_code,
            stock_name=data.get('stock_name', ''),
            market_type=data.get('market_type'),
            user_position=data.get('user_position'),
            model_params=model_params,
        )
        
        result["from_cache"] = False

        # fire-and-forget：数据库写入从响应路径剥离，不阻塞客户端
        DB_SAVE_EXECUTOR.submit(safe_save_decision, result)

        # 写入缓存 (5分钟)
        cache.set(cache_key, result, ttl=300)
        
        return jsonify(result)
    
    except Exception as e:
        from agent.core.exceptions import DataError
        if isinstance(e, DataError):
            logger.error(f"关键数据缺失，诊断无法继续: {e}")
            return jsonify({
                "error": str(e),
                "code": "DATA_UNAVAILABLE",
                "message": "该股票的关键数据暂时无法获取，请稍后重试。系统只提供真实数据，绝不编造。"
            }), 503
        logger.exception("诊断接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


@diagnose_bp.route('/blackboard/clear', methods=['POST'])
@RateLimiter().limit(max_requests=5, window=60)
def clear_blackboard():
    """清理黑板（管理员接口）"""
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code')
        
        from agent.core.blackboard import Blackboard
        bb = Blackboard.get_instance()
        
        if stock_code:
            bb.clear_stock(stock_code)
            message = f"已清理 {stock_code} 的黑板数据"
        else:
            codes = bb.get_all_stock_codes()
            for code in codes:
                bb.clear_stock(code)
            message = f"已清理全部 {len(codes)} 只股票的黑板数据"
        
        # 同时清理缓存
        cache.clear()
        
        return jsonify({"success": True, "message": message})
    
    except Exception as e:
        logger.exception("清理黑板接口异常")
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500


@diagnose_bp.route('/diagnose/stream', methods=['POST'])
@RateLimiter().limit(max_requests=10, window=60)
def diagnose_stream():
    """
    SSE 流式诊断接口 — 异步后台执行 + 结果缓存 + 跨页面重连

    架构：
    - 优先检查缓存（与 /api/agent/diagnose 共享缓存键）
    - 缓存命中 → 直接推送单个 result 事件
    - task_id 重连 → 回放 TaskTracker buffer，继续接收实时事件
    - 新任务 → 提交到 DIAGNOSIS_EXECUTOR 后台线程池执行
    """
    from flask import Response, stream_with_context
    from agent.core.task_tracker import task_tracker

    data = request.get_json(force=True) or {}
    stock_code = data.get('stock_code', '').strip()

    if not stock_code:
        return jsonify({"error": "stock_code 不能为空", "code": "MISSING_STOCK_CODE"}), 400

    stock_code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if not stock_code.isdigit() or len(stock_code) != 6:
        return jsonify({"error": "stock_code 必须是6位数字", "code": "INVALID_STOCK_CODE"}), 400

    force_refresh = data.get('force_refresh', False)
    reconnect_task_id = data.get('task_id', '').strip()
    cache_key = generate_cache_key("diagnose", code=stock_code, date=datetime.now().strftime("%Y%m%d%H"))

    # ── 重连模式：task_id 有效 → 回放 buffer + 继续实时事件 ──
    if reconnect_task_id and not force_refresh:
        existing = task_tracker.get_task(reconnect_task_id)
        if existing is not None:
            logger.info(f"SSE 重连: task_id={reconnect_task_id}, status={existing.status}, buffer={len(existing.buffer)} events")

            def generate_reconnect():
                # 1) 回放已缓冲的历史事件
                buffered = existing.get_buffered()
                if buffered:
                    yield f"data: {json.dumps({'stage': 'reconnect', 'message': f'重连成功，回放 {len(buffered)} 个事件', 'progress': 0, 'replay_count': len(buffered)}, ensure_ascii=False)}\n\n"
                    for evt in buffered:
                        yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
                # 2) 如果任务已完成，直接结束
                if existing.status in ("completed", "error"):
                    yield f"data: {json.dumps({'stage': 'done', 'message': '任务已结束（重连）', 'progress': 100}, ensure_ascii=False)}\n\n"
                    return
                # 3) 任务仍在运行 → 继续监听实时事件（通过统一队列）
                progress_queue: queue.Queue = queue.Queue()
                orchestrator = get_orchestrator()
                # 添加 TaskTracker 轮询包装
                def _wrap_reconnect():
                    last_idx = len(buffered)
                    # 先发完 buffer 中已有的
                    while True:
                        time.sleep(0.5)
                        new_buf = existing.get_buffered()
                        if len(new_buf) > last_idx:
                            for evt in new_buf[last_idx:]:
                                progress_queue.put(evt)
                            last_idx = len(new_buf)
                        if existing.status in ("completed", "error"):
                            progress_queue.put(None)
                            return
                DIAGNOSIS_EXECUTOR.submit(_wrap_reconnect)

                last_heartbeat = time.time()
                while True:
                    try:
                        evt = progress_queue.get(timeout=1.0)
                        if evt is None:
                            break
                        yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
                        last_heartbeat = time.time()
                    except queue.Empty:
                        if time.time() - last_heartbeat >= 15:
                            yield ": heartbeat\n\n"
                            last_heartbeat = time.time()
                return

            return Response(
                stream_with_context(generate_reconnect()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'X-Task-ID': reconnect_task_id}
            )

    # ── 缓存命中：直接推送最终结果 ──
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"SSE 缓存命中: {stock_code}")
            cached["from_cache"] = True

            def generate_cache_hit():
                yield f"data: {json.dumps({'stage': 'cache_hit', 'message': '缓存命中，直接返回结果', 'progress': 100}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'stage': 'result', 'message': '诊断完成（缓存）', 'progress': 100, 'data': cached}, ensure_ascii=False, default=str)}\n\n"

            return Response(
                stream_with_context(generate_cache_hit()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
            )

    # ── 新任务：注册 + 后台线程池异步执行 ──
    new_task_id = task_tracker.start_task(stock_code, "diagnosis")
    progress_queue: queue.Queue = queue.Queue()
    orchestrator = get_orchestrator()
    bg_future = DIAGNOSIS_EXECUTOR.submit(
        orchestrator.run_diagnosis_background,
        stock_code=stock_code,
        progress_queue=progress_queue,
        stock_name=data.get('stock_name', ''),
        market_type=data.get('market_type'),
        user_position=data.get('user_position'),
        model_params=data.get('model_params'),
        task_id=new_task_id,
    )

    def generate():
        heartbeat_interval = 15
        last_heartbeat = time.time()
        last_result = None
        while True:
            try:
                event = progress_queue.get(timeout=1.0)
                if event is None:
                    try:
                        bg_future.result(timeout=0)
                    except Exception as e:
                        logger.warning(f"后台诊断任务异常: {e}")
                    if last_result and last_result.get("data"):
                        cache.set(cache_key, last_result["data"], ttl=300)
                    break
                if event.get("stage") == "result":
                    last_result = event
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                last_heartbeat = time.time()
            except queue.Empty:
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'X-Task-ID': new_task_id,
        }
    )


@diagnose_bp.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态（跨页面重连前检查）"""
    from agent.core.task_tracker import task_tracker
    task = task_tracker.get_task(task_id)
    if task is None:
        return jsonify({"error": "任务不存在或已过期", "code": "TASK_NOT_FOUND"}), 404
    return jsonify({
        "task_id": task.task_id,
        "stock_code": task.stock_code,
        "task_type": task.task_type,
        "status": task.status,
        "events_count": len(task.buffer),
        "started_at": task.started_at,
    })


@diagnose_bp.route('/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """取消正在运行的任务"""
    from agent.core.task_tracker import task_tracker
    ok = task_tracker.cancel_task(task_id)
    if ok:
        return jsonify({"success": True, "message": "任务已取消"})
    return jsonify({"error": "任务不存在", "code": "TASK_NOT_FOUND"}), 404


@diagnose_bp.route('/concurrency/stats', methods=['GET'])
@RateLimiter().limit(max_requests=30, window=60)
def concurrency_stats():
    """
    获取自适应并发控制器统计信息
    
    Response:
        {
            "adaptive_enabled": bool,
            "stats": {
                "current_limit": int,
                "min_concurrent": int,
                "max_concurrent": int,
                "avg_response_time": float,
                "error_rate": float,
                "total_requests": int,
                ...
            }
        }
    """
    from agent.core.orchestrator import ADAPTIVE_CONCURRENCY
    if ADAPTIVE_CONCURRENCY:
        from agent.core.orchestrator import _CONCURRENCY_CONTROLLER
        stats = _CONCURRENCY_CONTROLLER.get_stats()
    else:
        from agent.core.orchestrator import _MAX_CONCURRENT_DIAGNOSES
        stats = {
            "current_limit": _MAX_CONCURRENT_DIAGNOSES,
            "max_concurrent": _MAX_CONCURRENT_DIAGNOSES,
            "mode": "static",
        }
    return jsonify({
        "adaptive_enabled": ADAPTIVE_CONCURRENCY,
        "stats": stats,
    })


@diagnose_bp.route('/concurrency/reset', methods=['POST'])
@RateLimiter().limit(max_requests=5, window=60)
def concurrency_reset():
    """重置自适应并发控制器（管理员/调试接口）"""
    from agent.core.orchestrator import ADAPTIVE_CONCURRENCY
    if not ADAPTIVE_CONCURRENCY:
        return jsonify({"error": "自适应并发控制未启用", "code": "NOT_ENABLED"}), 400
    from agent.core.orchestrator import _CONCURRENCY_CONTROLLER
    _CONCURRENCY_CONTROLLER.reset()
    return jsonify({"success": True, "message": "并发控制器已重置"})


@diagnose_bp.route('/debate/simulate', methods=['POST'])
def simulate_debate():
    """模拟Agent间辩论"""
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()
        if not stock_code:
            return jsonify({"error": "stock_code 不能为空"}), 400
        
        orchestrator = get_orchestrator()
        result = orchestrator.run_diagnosis(stock_code=stock_code)
        
        from agent.core.blackboard import Blackboard
        bb = Blackboard.get_instance()
        opinions = bb.get_opinions(stock_code)
        
        from agent.core.debate import DebateEngine
        debate_engine = DebateEngine()
        opinions_dict = {op.agent_id: op for op in opinions}
        debate_results = debate_engine.run_all_debates(
            opinions=opinions_dict,
            context=f"股票: {stock_code}",
        )
        
        return jsonify({
            "stock_code": stock_code,
            "conflicts_detected": len(debate_results),
            "debates": [debate_engine.to_dict(r) for r in debate_results],
        })
    except Exception as e:
        logger.exception("辩论模拟接口异常")
        return jsonify({"error": str(e)}), 500

