"""
MASS Flask Blueprint: system_bp
"""
import glob
import sys
import os as _os
import threading
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

system_bp = Blueprint('system', __name__, url_prefix='/api/agent')

# ── 系统统计缓存与后台刷新 ──

_SYSTEM_STATS_REFRESH_INTERVAL = 30  # 秒
_SYSTEM_STATS_CACHE_LOCK = threading.Lock()
_SYSTEM_STATS_CACHE = {
    "data": None,
    "updated_at": 0,
    "error": None,
    "refresh_count": 0,
}
_SYSTEM_STATS_THREAD: threading.Thread | None = None
_SYSTEM_STATS_THREAD_STOP = threading.Event()


def _compute_system_stats() -> dict:
    """
    计算完整的系统统计信息。
    此函数可能耗时数秒，必须在后台线程中调用，禁止在请求处理线程中直接调用。
    """
    stats_package = {
        "recent_logs": [],
        "database": {},
        "blackboard": {},
        "cache": {},
        "requests_history": [],
        "diagnose_trend": [],
        "direction_distribution": {},
        "agent_signals": {},
        "hot_stocks": [],
        "data_sources": {},
        "system": {},
        "server_time": datetime.now().isoformat(),
        "version": "2.1.0",
    }

    # 1) 数据库统计
    try:
        db = get_db()
        stats_package["database"] = db.get_stats()
    except Exception as e:
        logger.warning(f"[stats-cache] 数据库统计失败: {e}")
        stats_package["database"] = {"error": str(e)}

    # 2) Blackboard & Cache
    try:
        from agent.core.blackboard import Blackboard
        stats_package["blackboard"] = Blackboard.get_instance().get_stats()
    except Exception as e:
        logger.warning(f"[stats-cache] Blackboard 统计失败: {e}")

    try:
        stats_package["cache"] = cache.get_stats()
    except Exception as e:
        logger.warning(f"[stats-cache] Cache 统计失败: {e}")

    # 3) 详细数据库查询（在一个连接内完成）
    try:
        db = get_db()
        with db._get_connection() as conn:
            # 最近诊断请求
            recent_rows = conn.execute(
                """SELECT stock_code, stock_name, decision_date, decision_time,
                          decision, confidence, validated
                   FROM agent_decisions
                   ORDER BY created_at DESC
                   LIMIT 20"""
            ).fetchall()
            stats_package["requests_history"] = [dict(row) for row in recent_rows]

            # 最近7天诊断趋势
            trend_rows = conn.execute(
                """SELECT decision_date, COUNT(*) as count,
                          AVG(confidence) as avg_confidence
                   FROM agent_decisions
                   WHERE decision_date >= date('now', '-6 days')
                   GROUP BY decision_date
                   ORDER BY decision_date"""
            ).fetchall()
            stats_package["diagnose_trend"] = [dict(row) for row in trend_rows]

            # 决策方向分布
            direction_rows = conn.execute(
                """SELECT decision, COUNT(*) as count
                   FROM agent_decisions
                   GROUP BY decision"""
            ).fetchall()
            stats_package["direction_distribution"] = {
                str(row['decision']): row['count'] for row in direction_rows
            }

            # Agent 信号分布
            agent_signal_rows = conn.execute(
                """SELECT agent_id, signal, COUNT(*) as count, AVG(confidence) as avg_confidence
                   FROM agent_opinions
                   GROUP BY agent_id, signal"""
            ).fetchall()
            agent_signals = {}
            for row in agent_signal_rows:
                aid = row['agent_id']
                if aid not in agent_signals:
                    agent_signals[aid] = {'signals': {}, 'avg_confidence': 0, 'total': 0}
                agent_signals[aid]['signals'][str(row['signal'])] = row['count']
                agent_signals[aid]['avg_confidence'] = round(row['avg_confidence'] or 0, 2)
                agent_signals[aid]['total'] += row['count']
            stats_package["agent_signals"] = agent_signals

            # 热门股票 Top 10
            hot_rows = conn.execute(
                """SELECT stock_code, stock_name, COUNT(*) as count
                   FROM agent_decisions
                   GROUP BY stock_code
                   ORDER BY count DESC
                   LIMIT 10"""
            ).fetchall()
            hot_stocks = []
            for row in hot_rows:
                name = row['stock_name']
                code = row['stock_code']
                if not name or name == code or "模拟" in name or "mock" in name.lower():
                    name = code
                hot_stocks.append({"stock_code": code, "stock_name": name, "count": row['count']})
            stats_package["hot_stocks"] = hot_stocks
    except Exception as e:
        logger.warning(f"[stats-cache] 数据库详细查询失败: {e}")

    # 4) 数据源健康状态
    try:
        from agent.tools.stock_data_tool import StockDataTool
        tool = StockDataTool()
        registry = tool._registry
        source_health = registry.health_check()
        stats_package["data_sources"] = {
            name: {
                "healthy": healthy,
                "priority": next(
                    (c.priority for c in registry._crawlers if c.name == name), 0
                ),
            }
            for name, healthy in source_health.items()
        }
    except Exception as e:
        logger.warning(f"[stats-cache] 数据源健康检查失败: {e}")

    # 5) 最近日志（限制读取行数，避免大文件）
    try:
        log_files = sorted(glob.glob("logs/mass_*.log"), reverse=True)
        if log_files:
            with open(log_files[0], "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                stats_package["recent_logs"] = [
                    line.strip() for line in lines[-30:] if line.strip()
                ]
    except Exception as e:
        logger.warning(f"[stats-cache] 读取日志失败: {e}")

    # 6) 系统信息（psutil）
    try:
        import psutil
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / 1024 / 1024
        stats_package["system"] = {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "memory_mb": round(mem_mb, 1),
            "cpu_percent": round(proc.cpu_percent(interval=0.1), 1),
            "pid": proc.pid,
            "cwd": str(_os.getcwd()),
            "start_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
        }
    except Exception:
        stats_package["system"] = {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        }

    return stats_package


def _refresh_system_stats_loop() -> None:
    """后台守护线程：定期刷新系统统计缓存。"""
    global _SYSTEM_STATS_CACHE
    logger.info("[stats-cache] 后台统计刷新线程已启动")
    while not _SYSTEM_STATS_THREAD_STOP.is_set():
        try:
            data = _compute_system_stats()
            with _SYSTEM_STATS_CACHE_LOCK:
                _SYSTEM_STATS_CACHE["data"] = data
                _SYSTEM_STATS_CACHE["updated_at"] = time.time()
                _SYSTEM_STATS_CACHE["error"] = None
                _SYSTEM_STATS_CACHE["refresh_count"] += 1
        except Exception as e:
            logger.exception(f"[stats-cache] 后台刷新失败: {e}")
            with _SYSTEM_STATS_CACHE_LOCK:
                _SYSTEM_STATS_CACHE["error"] = str(e)
        # 等待间隔（支持被事件提前唤醒以优雅退出）
        _SYSTEM_STATS_THREAD_STOP.wait(_SYSTEM_STATS_REFRESH_INTERVAL)
    logger.info("[stats-cache] 后台统计刷新线程已退出")


def _ensure_stats_refresh_thread() -> None:
    """确保后台统计刷新线程已启动（线程安全，幂等）。"""
    global _SYSTEM_STATS_THREAD
    with _SYSTEM_STATS_CACHE_LOCK:
        if _SYSTEM_STATS_THREAD is not None and _SYSTEM_STATS_THREAD.is_alive():
            return
        _SYSTEM_STATS_THREAD_STOP.clear()
        t = threading.Thread(target=_refresh_system_stats_loop, name="stats-refresh", daemon=True)
        t.start()
        _SYSTEM_STATS_THREAD = t
        logger.info("[stats-cache] 已启动后台统计刷新线程")


@system_bp.route('/stats', methods=['GET'])
@RateLimiter().limit(max_requests=30, window=60)
def system_stats():
    """
    获取系统统计信息 — 返回预计算缓存结果，响应时间 < 10ms。
    首次调用时若缓存为空，将同步计算一次作为兜底。
    """
    _ensure_stats_refresh_thread()

    with _SYSTEM_STATS_CACHE_LOCK:
        cached_data = _SYSTEM_STATS_CACHE["data"]
        updated_at = _SYSTEM_STATS_CACHE["updated_at"]
        last_error = _SYSTEM_STATS_CACHE["error"]
        refresh_count = _SYSTEM_STATS_CACHE["refresh_count"]

    # 首次启动兜底：若缓存完全为空，同步计算一次
    if cached_data is None:
        logger.info("[stats] 缓存为空，执行同步兜底计算")
        try:
            cached_data = _compute_system_stats()
            with _SYSTEM_STATS_CACHE_LOCK:
                _SYSTEM_STATS_CACHE["data"] = cached_data
                _SYSTEM_STATS_CACHE["updated_at"] = time.time()
                _SYSTEM_STATS_CACHE["error"] = None
                _SYSTEM_STATS_CACHE["refresh_count"] += 1
            updated_at = _SYSTEM_STATS_CACHE["updated_at"]
            last_error = None
            refresh_count = _SYSTEM_STATS_CACHE["refresh_count"]
        except Exception as e:
            logger.exception(f"[stats] 同步兜底计算失败: {e}")
            return jsonify({
                "error": str(e),
                "code": "INTERNAL_ERROR",
                "message": "统计信息计算失败，请稍后重试",
            }), 500

    age_seconds = round(time.time() - updated_at, 1) if updated_at else None
    response = {
        **cached_data,
        "cached_at": datetime.fromtimestamp(updated_at).isoformat() if updated_at else None,
        "cache_age_seconds": age_seconds,
        "stale": age_seconds > _SYSTEM_STATS_REFRESH_INTERVAL * 2 if age_seconds is not None else True,
        "refresh_count": refresh_count,
    }
    if last_error:
        response["_last_refresh_error"] = last_error

    return jsonify(response)



@system_bp.route('/settings', methods=['GET'])
def get_settings():
    """获取当前系统配置（API Key 脱敏）"""
    try:
        from agent.core.system_config import get_system_config
        cfg_mgr = get_system_config()
        config = cfg_mgr.get_config_for_display()
        return jsonify({
            "code": "OK",
            "data": config,
        })
    except Exception as e:
        logger.exception("获取配置失败")
        return jsonify({"code": "ERROR", "error": str(e)}), 500


@system_bp.route('/settings', methods=['POST'])
def save_settings():
    """
    保存系统配置

    Request:
        {
            "provider": "deepseek",
            "api_key": "sk-...",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-v4-pro",
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 4096,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": 60,
            "max_retries": 3,
            "use_mock": false
        }
    """
    try:
        from agent.core.system_config import get_system_config, LLMRuntimeConfig

        data = request.get_json(force=True) or {}
        cfg_mgr = get_system_config()

        # 获取当前配置作为基础
        current = cfg_mgr.get_llm_config()

        # 如果前端传了空 api_key 或脱敏值，保留旧值
        new_api_key = data.get("api_key", "").strip()
        if not new_api_key or "•" in new_api_key or "****" in new_api_key:
            new_api_key = current.api_key

        config = LLMRuntimeConfig(
            provider=data.get("provider", current.provider),
            api_key=new_api_key or current.api_key,
            base_url=data.get("base_url", current.base_url),
            model=data.get("model", current.model),
            temperature=float(data.get("temperature", current.temperature)),
            top_p=float(data.get("top_p", current.top_p)),
            max_tokens=int(data.get("max_tokens", current.max_tokens)),
            frequency_penalty=float(data.get("frequency_penalty", current.frequency_penalty)),
            presence_penalty=float(data.get("presence_penalty", current.presence_penalty)),
            timeout=int(data.get("timeout", current.timeout)),
            max_retries=int(data.get("max_retries", current.max_retries)),
            use_mock=data.get("use_mock", current.use_mock),
        )

        # 校验
        valid, msg = cfg_mgr.validate_config(config)
        if not valid:
            return jsonify({"code": "INVALID_CONFIG", "error": msg}), 400

        # 保存
        if cfg_mgr.save_llm_config(config):
            return jsonify({
                "code": "OK",
                "message": "配置已保存",
                "data": cfg_mgr.get_config_for_display(),
            })
        else:
            return jsonify({"code": "SAVE_FAILED", "error": "保存失败"}), 500

    except Exception as e:
        logger.exception("保存配置失败")
        return jsonify({"code": "ERROR", "error": str(e)}), 500


@system_bp.route('/settings/test', methods=['POST'])
def test_llm_connection():
    """测试 LLM 连接是否可用"""
    try:
        from agent.core.system_config import get_system_config, LLMRuntimeConfig
        from agent.tools.llm_client import LLMClient

        data = request.get_json(force=True) or {}
        cfg_mgr = get_system_config()
        current = cfg_mgr.get_llm_config()

        # 构建测试配置
        test_api_key = data.get("api_key", current.api_key).strip()
        if not test_api_key or "•" in test_api_key or "****" in test_api_key:
            test_api_key = current.api_key

        test_config = LLMRuntimeConfig(
            provider=data.get("provider", current.provider),
            api_key=test_api_key,
            base_url=data.get("base_url", current.base_url) or current.base_url,
            model=data.get("model", current.model),
            temperature=0.2,
            top_p=1.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=30,
            max_retries=1,
            use_mock=False,
        )

        if not test_config.api_key:
            return jsonify({
                "code": "MISSING_API_KEY",
                "success": False,
                "message": "API Key 不能为空",
            }), 400

        # 创建临时客户端测试连接
        from agent.tools.llm_client import LLMConfig
        client = LLMClient(config=LLMConfig(
            provider=test_config.provider,
            api_key=test_config.api_key,
            base_url=test_config.base_url,
            model=test_config.model,
            temperature=test_config.temperature,
            top_p=test_config.top_p,
            max_tokens=test_config.max_tokens,
            timeout=test_config.timeout,
            max_retries=test_config.max_retries,
        ))

        result = client.chat(
            system="你是一个测试助手，请回复：连接成功",
            user="测试",
            json_mode=False,
        )

        return jsonify({
            "code": "OK",
            "success": True,
            "message": "LLM 连接测试成功",
            "response_preview": str(result)[:200] if result else "",
        })

    except Exception as e:
        logger.warning(f"LLM 连接测试失败: {e}")
        return jsonify({
            "code": "CONNECTION_FAILED",
            "success": False,
            "message": f"连接测试失败: {str(e)}",
        }), 200  # 返回 200 但标记 success=False，方便前端处理
