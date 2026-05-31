"""
MASS Flask Blueprint 聚合入口
按领域拆分为多个子 Blueprint，保持 URL 前缀一致
"""
from flask import Blueprint

from api.diagnose_bp import diagnose_bp
from api.portfolio_bp import portfolio_bp
from api.history_bp import history_bp
from api.prediction_bp import prediction_bp
from api.backtest_bp import backtest_bp
from api.report_bp import report_bp
from api.system_bp import system_bp

# 聚合 Blueprint：所有子模块共享 /api/agent 前缀
# 子模块各自独立注册路由，Flask 自动合并
agent_bp = Blueprint('agent', __name__, url_prefix='/api/agent')

# 将各子 Blueprint 的路由注册到聚合 Blueprint 中
# 注：Flask 不支持 Blueprint 嵌套注册，因此 app.py 需要分别注册每个子 Blueprint
# 这里保留 agent_bp 变量作为向后兼容的聚合引用

__all__ = [
    "agent_bp",
    "diagnose_bp",
    "portfolio_bp",
    "history_bp",
    "prediction_bp",
    "backtest_bp",
    "report_bp",
    "system_bp",
]
