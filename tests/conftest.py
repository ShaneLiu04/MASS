"""
MASS 测试配置
Pytest 共享 fixtures — 数据库隔离、Mock LLM、同步后台执行器
"""
import os
import tempfile
from typing import Dict, Any
import pytest

# 测试环境基础配置（不强制 Mock LLM，允许真实路径测试）
os.environ['FLASK_DEBUG'] = 'False'


class _SyncExecutor:
    """同步执行器：将后台任务在当前线程同步执行，避免测试中多线程数据库冲突"""

    def submit(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
        return _FakeFuture()

    def shutdown(self, wait=True):
        pass


class _FakeFuture:
    def result(self, timeout=None):
        return None


@pytest.fixture
def test_db(monkeypatch):
    """
    测试用隔离数据库。
    通过修改 config.DATABASE_PATH 让所有代码路径（包括蓝图中的 get_db()）
    自动使用同一个临时 SQLite 文件，避免多实例连接隔离问题。
    """
    import config as cfg_module
    import agent.models.database as db_module
    import api.common as common_module

    fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # 重定向全局数据库路径
    original_path = cfg_module.DATABASE_PATH
    monkeypatch.setattr(cfg_module, "DATABASE_PATH", tmp_path)
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path)

    # 强制重置已缓存的数据库实例
    monkeypatch.setattr(common_module, "_db", None)

    db = db_module.Database()

    yield db

    # 清理
    db._pool.close()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


@pytest.fixture
def app(monkeypatch, test_db):
    """创建测试用 Flask 应用（数据库隔离 + 同步后台执行器）"""
    import api.common

    # 同步化后台线程池，避免测试中的多线程竞争
    monkeypatch.setattr(api.common, "DB_SAVE_EXECUTOR", _SyncExecutor())
    monkeypatch.setattr(api.common, "PORTFOLIO_EXECUTOR", _SyncExecutor())

    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    app.config['USE_MOCK_LLM'] = True
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def mock_llm():
    """Mock LLM 客户端"""
    from agent.tools.llm_client import MockLLMClient
    return MockLLMClient()


@pytest.fixture
def sample_stock_snapshot():
    """示例股票数据快照（固定种子，可复现）"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from agent.core.blackboard import StockSnapshot

    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.now(), periods=60, freq="B")
    prices = 15.0 + np.cumsum(rng.standard_normal(60) * 0.3)

    df = pd.DataFrame({
        "date": dates,
        "open": prices + rng.standard_normal(60) * 0.1,
        "high": prices + np.abs(rng.standard_normal(60)) * 0.3 + 0.1,
        "low": prices - np.abs(rng.standard_normal(60)) * 0.3 - 0.1,
        "close": prices,
        "volume": rng.integers(1000000, 5000000, 60),
        "amount": rng.integers(10000000, 50000000, 60),
    })

    return StockSnapshot(
        stock_code="000001",
        stock_name="测试股票",
        current_price=15.5,
        kline_df=df,
        indicators={
            "ma5": 15.2, "ma20": 15.0, "ma60": 14.8,
            "macd_golden_cross": True, "rsi14": 55,
        },
        fundamentals={
            "pe_ttm": 15.0, "pb": 1.5, "roe": 18.0,
            "industry": "测试行业",
        },
        fund_flow={
            "main_net_inflow_10d": 5000,
            "main_inflow_days": 7,
        },
        market_context={
            "sector_performance_5d": 3.5,
            "sector_rank": 5,
        },
        sentiment_data={
            "social_sentiment_7d": 0.3,
            "news_count_7d": 10,
        },
        macro_data={
            "pmi": 51.5, "bond_yield_10y": 2.5,
        },
        risk_metrics={
            "annual_volatility": 25.0,
            "max_drawdown": -12.0,
        },
    )
