"""
MASS: Multi-Agent Stock System - Flask 主应用入口 v2.1 (with Auth)
"""
import json as _json
import os
import sys
from datetime import datetime

from flask import Flask, render_template, jsonify, request, redirect, url_for
from loguru import logger

os.makedirs('logs', exist_ok=True)
logger.add("logs/mass_{time}.log", rotation="10 MB", retention="7 days", level="INFO")
logger.add("logs/mass_error_{time}.log", rotation="10 MB", retention="7 days", level="ERROR")

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, SECRET_KEY
from api.agent_bp import (
    diagnose_bp,
    portfolio_bp,
    history_bp,
    prediction_bp,
    backtest_bp,
    report_bp,
    system_bp,
)
from api.middleware import register_middleware
from agent.auth import AuthManager, require_login

# ── 可选加速：orjson (Rust) 替代 stdlib json，序列化速度 2-5× ──
_ORIGINAL_DUMPS = _json.dumps


def _make_numpy_default(orig_default):
    """包装原始 default，优先处理 numpy/pandas 类型"""
    def _default(o):
        try:
            import numpy as np
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        if orig_default:
            return orig_default(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return _default


try:
    import orjson as _orjson

    def _fast_dumps(obj, **kwargs):
        orig_default = kwargs.get('default')
        _default = _make_numpy_default(orig_default)
        new_kwargs = {**kwargs, 'default': _default}

        # 透明回退：非默认可选参数走 stdlib（确保兼容 indent/ensure_ascii 等）
        if any(k in kwargs for k in ('indent', 'separators', 'sort_keys')):
            return _ORIGINAL_DUMPS(obj, **new_kwargs)
        try:
            _opt = _orjson.OPT_NON_STR_KEYS
            if hasattr(_orjson, 'OPT_SERIALIZE_NUMPY'):
                _opt |= _orjson.OPT_SERIALIZE_NUMPY
            return _orjson.dumps(
                obj,
                default=_default,
                option=_opt,
            ).decode()
        except (TypeError, ValueError):
            return _ORIGINAL_DUMPS(obj, **new_kwargs)

    _json.dumps = _fast_dumps
    logger.info("orjson 加速已启用")
except ImportError:
    logger.info("orjson 未安装，使用 stdlib json")


def _warmup_cache() -> None:
    """后台线程：预加载热门股票数据，消除首次请求冷启动延迟"""
    hot_stocks = os.getenv("WARMUP_STOCKS", "")
    codes = [s.strip() for s in hot_stocks.split(",") if s.strip()]
    if not codes:
        return
    logger.info(f"缓存预热开始: {codes}")
    from agent.core.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(use_mock_llm=os.getenv('USE_MOCK_LLM', 'True').lower() == 'true')
    for code in codes:
        try:
            orch._create_snapshot(code, "")
            logger.info(f"预热完成: {code}")
        except Exception as e:
            logger.warning(f"预热跳过 {code}: {e}")
    logger.info("缓存预热结束")


def create_app() -> Flask:
    """Flask应用工厂"""
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    app.config['USE_MOCK_LLM'] = os.getenv('USE_MOCK_LLM', 'True').lower() == 'true'

    # ── 生产配置：减少模板/静态资源的重复加载 ──
    if not app.debug:
        app.config['TEMPLATES_AUTO_RELOAD'] = False
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # 静态资源缓存 1h

    register_middleware(app)
    # 注册所有领域 Blueprint（URL 前缀均为 /api/agent）
    app.register_blueprint(diagnose_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(system_bp)

    # ── 后台预热：不阻塞服务启动 ──
    import threading
    threading.Thread(target=_warmup_cache, daemon=True, name="warmup").start()
    
    # ========== 认证页面路由 ==========
    
    @app.route('/login')
    def login_page():
        """登录页面"""
        if AuthManager.is_authenticated():
            return redirect(url_for('dashboard'))
        return render_template('login.html')
    
    @app.route('/')
    @require_login
    def index():
        """首页重定向到仪表盘"""
        return redirect(url_for('dashboard'))
    
    @app.route('/dashboard')
    @require_login
    def dashboard():
        """工作台仪表盘"""
        user = AuthManager.get_current_user()
        return render_template('dashboard.html', user=user)
    
    # ========== 功能页面路由 (登录保护) ==========
    
    @app.route('/agent/trading')
    @require_login
    def agent_trading():
        return render_template('agent_trading.html')
    
    @app.route('/agent/portfolio')
    @require_login
    def agent_portfolio():
        return render_template('agent_portfolio.html')
    
    @app.route('/agent/backtest')
    @require_login
    def agent_backtest():
        return render_template('agent_backtest.html')
    
    @app.route('/agent/history')
    @require_login
    def agent_history():
        return render_template('agent_history.html')
    
    @app.route('/agent/monitor')
    @require_login
    def agent_monitor():
        return render_template('agent_monitor.html')
    
    @app.route('/agent/report')
    @require_login
    def agent_report():
        return render_template('agent_report.html')
    
    @app.route('/settings')
    @require_login
    def settings_page():
        """系统设置页面 — LLM / API Key 配置"""
        return render_template('settings.html')
    
    # ========== API 认证路由 ==========
    
    @app.route('/api/auth/login', methods=['POST'])
    def api_login():
        """API 登录"""
        data = request.get_json() or {}
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        user = AuthManager.authenticate(username, password)
        if user:
            AuthManager.login(username)
            logger.info(f"用户登录成功: {username}")
            return jsonify({
                "success": True,
                "message": "登录成功",
                "user": user
            })
        logger.warning(f"登录失败: {username}")
        return jsonify({
            "success": False,
            "message": "用户名或密码错误"
        }), 401
    
    @app.route('/api/auth/logout', methods=['POST'])
    def api_logout():
        """API 登出"""
        user = AuthManager.get_current_user()
        if user:
            logger.info(f"用户登出: {user.get('username')}")
        AuthManager.logout()
        return jsonify({"success": True, "message": "已退出登录"})
    
    @app.route('/api/auth/me')
    def api_me():
        """获取当前用户信息"""
        user = AuthManager.get_current_user()
        if user:
            return jsonify({"authenticated": True, "user": user})
        return jsonify({"authenticated": False, "user": None})
    
    # ========== 系统状态路由 ==========
    
    @app.route('/api/health')
    def health_check():
        from agent.core.cache import cache
        from agent.core.blackboard import Blackboard
        bb = Blackboard()
        return jsonify({
            "status": "healthy", "version": "2.1.0",
            "timestamp": datetime.now().isoformat(),
            "mock_mode": app.config['USE_MOCK_LLM'],
            "cache": cache.get_stats(),
            "blackboard": bb.get_stats(),
        })
    
    @app.route('/api/status')
    def system_status():
        from agent.models.database import Database
        db = Database()
        return jsonify({
            "status": "running", "version": "2.1.0",
            "timestamp": datetime.now().isoformat(),
            "mock_mode": app.config['USE_MOCK_LLM'],
            "database": db.get_stats(),
            "environment": {
                "llm_provider": os.getenv('LLM_PROVIDER', 'mock'),
                "model": os.getenv('DEFAULT_MODEL', 'mock'),
                "parallel": os.getenv('AGENT_PARALLEL', 'True'),
            },
        })
    
    @app.route('/api/market/indices')
    def market_indices():
        """获取A股主要指数实时行情"""
        from agent.crawlers.sina import SinaCrawler
        try:
            crawler = SinaCrawler()
            result = crawler.fetch("", "market_context")
            if result and "indices" in result:
                # 只返回需要的三个指数
                all_indices = result["indices"]
                filtered = {}
                for key in ["上证指数", "创业板指", "沪深300"]:
                    if key in all_indices:
                        filtered[key] = all_indices[key]
                return jsonify({
                    "code": "OK",
                    "data": filtered,
                    "timestamp": datetime.now().isoformat(),
                })
            return jsonify({
                "code": "DATA_UNAVAILABLE",
                "data": {},
                "timestamp": datetime.now().isoformat(),
            }), 503
        except Exception as e:
            logger.warning(f"获取指数行情失败: {e}")
            return jsonify({
                "code": "DATA_UNAVAILABLE",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }), 503
    
    return app


if __name__ == '__main__':
    app = create_app()
    logger.info(f"MASS v2.1 服务启动于 http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
