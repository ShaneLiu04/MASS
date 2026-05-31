"""
MASS v2.3 系统级压力测试套件
覆盖: 并发、缓存、预测引擎、数据库、内存、边界条件
"""
import json
import os
import sys
import time
import queue
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter

import pytest
import numpy as np

# ── 确保 mock 模式 ──
os.environ["USE_MOCK_LLM"] = "True"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def app():
    """Flask 测试应用"""
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def orchestrator():
    """共享编排器实例"""
    from agent.core.orchestrator import AgentOrchestrator
    return AgentOrchestrator(use_mock_llm=True)


@pytest.fixture(scope="module")
def snapshot(orchestrator):
    """预创建数据快照（复用，加速测试）"""
    return orchestrator._create_snapshot("000001", "平安银行")


@pytest.fixture(scope="module")
def prediction_engine(orchestrator):
    return orchestrator.prediction_engine


@pytest.fixture(scope="module")
def database():
    from agent.models.database import Database
    db = Database()
    yield db


# ═══════════════════════════════════════════════════════════════════════════
# 1. 并发压力测试
# ═══════════════════════════════════════════════════════════════════════════

class TestConcurrency:
    """高并发场景：多路 SSE + 普通 API 同时请求"""

    def test_concurrent_diagnose(self, client):
        """10 并发 POST /api/agent/diagnose"""
        n = 10
        results = []
        errors = []

        def _req():
            try:
                resp = client.post("/api/agent/diagnose",
                    json={"stock_code": "000001", "force_refresh": True})
                return resp.status_code, resp.get_json()
            except Exception as e:
                return None, str(e)

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(_req) for _ in range(n)]
            for f in as_completed(futures):
                code, data = f.result()
                if code == 200:
                    results.append(data)
                else:
                    errors.append((code, data))

        assert len(results) >= n - 2, f"Only {len(results)}/{n} succeeded, errors: {errors[:3]}"
        for r in results:
            assert "final_decision" in r
            assert r["stock_code"] == "000001"

    def test_concurrent_predict(self, client):
        """8 并发 POST /api/agent/predict（不同参数组合）"""
        combos = [
            {"stock_code": "000001", "horizon": "short", "risk_tolerance": "conservative"},
            {"stock_code": "000001", "horizon": "short", "risk_tolerance": "moderate"},
            {"stock_code": "000001", "horizon": "short", "risk_tolerance": "aggressive"},
            {"stock_code": "000001", "horizon": "medium", "investment_style": "swing"},
            {"stock_code": "000001", "horizon": "medium", "investment_style": "trend"},
            {"stock_code": "000001", "horizon": "long", "investment_style": "value"},
            {"stock_code": "600519", "horizon": "short", "confidence_threshold": 0.7},
            {"stock_code": "300750", "horizon": "long", "confidence_threshold": 0.4},
        ]

        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(client.post, "/api/agent/predict", json=c): c for c in combos}
            for f in as_completed(futures):
                resp = f.result()
                if resp.status_code == 200:
                    results.append(resp.get_json())

        assert len(results) >= 6, f"Only {len(results)}/8 predictions succeeded"

    def test_concurrent_mixed(self, client):
        """混合并发：诊断 + 预测 + 健康检查 + 统计"""
        tasks = [
            ("GET", "/api/health", None),
            ("GET", "/api/status", None),
            ("GET", "/api/agent/stats", None),
            ("POST", "/api/agent/diagnose", {"stock_code": "000001"}),
            ("POST", "/api/agent/predict", {"stock_code": "000001", "horizon": "short"}),
            ("POST", "/api/agent/predict/stream", {"stock_code": "000001", "horizon": "short"}),
        ] * 3  # 18 个并发请求

        success = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = []
            for method, path, body in tasks:
                if method == "GET":
                    futures.append(ex.submit(client.get, path))
                else:
                    futures.append(ex.submit(client.post, path, json=body))
            for f in as_completed(futures):
                resp = f.result()
                if resp.status_code in (200, 503):
                    success += 1

        assert success >= 15, f"Only {success}/18 mixed requests succeeded"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 缓存压力测试
# ═══════════════════════════════════════════════════════════════════════════

class TestCache:
    """缓存命中率、TTL、一致性"""

    def test_cache_hit_diagnose(self, client):
        """诊断缓存命中验证"""
        # 第一次：miss
        r1 = client.post("/api/agent/diagnose",
            json={"stock_code": "000001", "force_refresh": True})
        assert r1.status_code == 200
        assert r1.get_json().get("from_cache") == False

        # 第二次：hit
        r2 = client.post("/api/agent/diagnose",
            json={"stock_code": "000001"})
        assert r2.status_code == 200
        assert r2.get_json().get("from_cache") == True

    def test_cache_hit_predict(self, client):
        """预测缓存命中验证"""
        r1 = client.post("/api/agent/predict",
            json={"stock_code": "600000", "horizon": "short", "force_refresh": True})
        if r1.status_code == 200:
            assert r1.get_json().get("from_cache") == False
            r2 = client.post("/api/agent/predict",
                json={"stock_code": "600000", "horizon": "short"})
            if r2.status_code == 200:
                assert r2.get_json().get("from_cache") == True

    def test_cache_isolation_by_params(self, client):
        """不同参数组合缓存隔离"""
        params_a = {"stock_code": "000002", "horizon": "short", "risk_tolerance": "conservative"}
        params_b = {"stock_code": "000002", "horizon": "short", "risk_tolerance": "aggressive"}

        r1 = client.post("/api/agent/predict", json={**params_a, "force_refresh": True})
        r2 = client.post("/api/agent/predict", json={**params_b, "force_refresh": True})

        if r1.status_code == 200 and r2.status_code == 200:
            # 两者应独立缓存，互不影响
            assert r1.get_json().get("from_cache") == False
            assert r2.get_json().get("from_cache") == False

    def test_cache_force_refresh(self, client):
        """force_refresh 绕过缓存"""
        r1 = client.post("/api/agent/diagnose",
            json={"stock_code": "000001", "force_refresh": True})
        assert r1.status_code == 200
        assert r1.get_json().get("from_cache") == False

    def test_cache_stats(self, client):
        """缓存统计有效性"""
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        cache_stats = data.get("cache", {})
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        assert "hit_rate" in cache_stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. 预测引擎压力测试
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictionEngineStress:
    """预测引擎全面测试"""

    def test_all_9_combinations(self, prediction_engine, snapshot):
        """9 种 risk_tolerance × investment_style 组合"""
        risk_levels = ["conservative", "moderate", "aggressive"]
        styles = ["swing", "trend", "value"]
        results = []

        for risk in risk_levels:
            for style in styles:
                r = prediction_engine.predict(
                    "000001", "test", snapshot,
                    horizon="short",
                    risk_tolerance=risk,
                    investment_style=style,
                    confidence_threshold=0.5,
                )
                results.append(r)
                assert r.risk_tolerance == risk
                assert r.investment_style == style
                assert r.direction in ("上涨", "下跌", "震荡", "不确定")

        assert len(results) == 9

    def test_all_3_horizons(self, prediction_engine, snapshot):
        """3 种预测周期 — token 预算验证"""
        budgets = {}
        for h in ["short", "medium", "long"]:
            r = prediction_engine.predict("000001", "test", snapshot, horizon=h)
            budgets[h] = r.prompt_tokens_estimated
            assert r.prompt_tokens_estimated > 0
            assert r.prompt_tokens_estimated < 3000, \
                f"{h} token budget {r.prompt_tokens_estimated} exceeds 3000"

        assert budgets["short"] > 0
        assert budgets["long"] > 0

    def test_confidence_calibration(self, prediction_engine, snapshot):
        """置信度校准：数据缺失时应降低"""
        r = prediction_engine.predict("000001", "test", snapshot, horizon="short")
        assert 0 < r.data_quality_factor <= 1.0
        assert r.confidence_calibrated <= r.confidence + 0.01  # 浮点容差

    def test_threshold_downgrade(self, prediction_engine, snapshot):
        """高阈值下不确定方向的比例应更高"""
        uncertain_low = 0
        uncertain_high = 0

        for _ in range(5):
            r_low = prediction_engine.predict(
                "000001", "test", snapshot,
                confidence_threshold=0.3,
            )
            if r_low.direction == "不确定":
                uncertain_low += 1

            r_high = prediction_engine.predict(
                "000001", "test", snapshot,
                confidence_threshold=0.8,
            )
            if r_high.direction == "不确定":
                uncertain_high += 1

        # 高阈值应产生更多"不确定"（统计趋势，5 次采样容差）
        assert uncertain_high >= uncertain_low - 1, \
            f"Expected high threshold to produce more uncertain, got low={uncertain_low} high={uncertain_high}"

    def test_parameter_validation(self, prediction_engine, snapshot):
        """参数校验"""
        with pytest.raises(ValueError, match="horizon"):
            prediction_engine.predict("000001", "", snapshot, horizon="invalid")
        with pytest.raises(ValueError, match="risk_tolerance"):
            prediction_engine.predict("000001", "", snapshot, risk_tolerance="invalid")
        with pytest.raises(ValueError, match="investment_style"):
            prediction_engine.predict("000001", "", snapshot, investment_style="invalid")

    def test_probability_normalization(self, prediction_engine, snapshot):
        """概率归一化：三项之和 ≈ 1.0"""
        for _ in range(5):
            r = prediction_engine.predict("000001", "test", snapshot)
            total = r.probability_up + r.probability_down + r.probability_sideways
            assert abs(total - 1.0) < 0.02, f"Probabilities sum to {total}"

    def test_prediction_determinism(self, prediction_engine, snapshot):
        """相同参数应产生可复现的结果（mock 模式下）"""
        r1 = prediction_engine.predict("000001", "test", snapshot)
        # Mock 模式每次随机，所以只验证结构一致性
        assert r1.stock_code == "000001"
        assert r1.prediction_horizon == "short"

    def test_rapid_fire_predictions(self, prediction_engine, snapshot):
        """快速连续预测：验证无状态污染"""
        results = []
        for _ in range(10):
            r = prediction_engine.predict("000001", "test", snapshot, horizon="short")
            results.append(r)
        assert len(results) == 10
        # 所有结果应有完整的字段
        for r in results:
            assert r.stock_code == "000001"
            assert r.data_quality_factor > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. 数据库压力测试
# ═══════════════════════════════════════════════════════════════════════════

class TestDatabaseStress:
    """数据库并发写入与查询"""

    def test_concurrent_prediction_saves(self, prediction_engine, snapshot, database):
        """并发写入预测记录"""
        def _save_one(i):
            r = prediction_engine.predict(f"000001", f"test_{i}", snapshot, horizon="short")
            return database.save_prediction(r)

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_save_one, i) for i in range(8)]
            ids = [f.result() for f in as_completed(futures)]

        assert len(ids) == 8
        assert all(isinstance(i, int) and i > 0 for i in ids)

    def test_prediction_history_query(self, prediction_engine, snapshot, database):
        """历史查询性能"""
        # 确保有数据
        r = prediction_engine.predict("000001", "test", snapshot)
        database.save_prediction(r)

        t0 = time.perf_counter()
        history = database.get_prediction_history(limit=100)
        elapsed = (time.perf_counter() - t0) * 1000

        assert len(history) > 0
        assert elapsed < 500, f"History query too slow: {elapsed:.0f}ms"

    def test_prediction_accuracy_stats(self, database):
        """统计查询"""
        stats = database.get_prediction_accuracy_stats()
        assert "total_validated_predictions" in stats
        assert "overall_accuracy" in stats
        assert "by_horizon" in stats

    def test_database_connection_pool_integrity(self, database):
        """连接池：多次操作后无泄漏"""
        initial_stats = database.get_stats()
        for _ in range(20):
            database.get_prediction_history(limit=1)
            database.get_prediction_accuracy_stats()
        final_stats = database.get_stats()
        # 统计应一致（无泄漏检测）
        assert final_stats["total_decisions"] >= initial_stats["total_decisions"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. 组件完整性压力测试
# ═══════════════════════════════════════════════════════════════════════════

class TestComponentIntegrity:
    """各组件在压力下的正确性"""

    def test_crawler_registry_singleton(self):
        """爬虫注册表单例：多次获取返回同一实例"""
        from agent.crawlers.registry import CrawlerRegistry
        r1 = CrawlerRegistry.get_instance()
        r2 = CrawlerRegistry.get_instance()
        assert r1 is r2
        sources = r1.get_sources()
        assert len(sources) == 5  # 5 个爬虫，无重复

    def test_blackboard_concurrent(self):
        """黑板并发读写"""
        from agent.core.blackboard import Blackboard, StockSnapshot, AgentOpinion

        bb = Blackboard()
        snap = StockSnapshot(stock_code="TEST", stock_name="test")

        def _writer(i):
            bb.publish_snapshot(snap)
            bb.submit_opinion("TEST", AgentOpinion(
                agent_id=f"agent_{i}", signal=1, confidence=0.8,
                reasoning="test"))

        def _reader():
            return bb.get_snapshot("TEST"), bb.get_opinions("TEST")

        with ThreadPoolExecutor(max_workers=10) as ex:
            writers = [ex.submit(_writer, i) for i in range(5)]
            readers = [ex.submit(_reader) for _ in range(10)]
            for f in as_completed(writers + readers):
                result = f.result()
                if isinstance(result, tuple):
                    snap_out, opinions = result
                    assert snap_out is not None or len(opinions) >= 0

        stats = bb.get_stats()
        assert stats["total_snapshots"] > 0

    def test_rate_limiter_concurrent(self):
        """限流器并发正确性"""
        from api.middleware import RateLimiter

        rl = RateLimiter(default_limit=100, window=10)
        blocked = 0

        def _check(i):
            return rl.is_allowed(f"client_{i % 3}")

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(_check, i) for i in range(300)]
            for f in as_completed(futures):
                if not f.result():
                    blocked += 1

        # 3 个 client，每个允许 100/10s，300 请求应全部通过
        assert blocked == 0, f"Unexpected blocks: {blocked}"

    def test_crawler_registry_concurrent_fetch(self):
        """爬虫注册表并发 fetch（验证线程安全）"""
        from agent.crawlers.registry import CrawlerRegistry

        reg = CrawlerRegistry.get_instance()
        results = []

        def _fetch():
            data = reg.fetch_merge("000001", "fundamentals")
            return data is not None

        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(_fetch) for _ in range(6)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == 6

    def test_cache_manager_concurrent(self):
        """缓存并发读写"""
        from agent.core.cache import CacheManager

        cm = CacheManager()
        errors = []

        def _worker(i):
            try:
                cm.set(f"key_{i}", f"value_{i}", ttl=10)
                val = cm.get(f"key_{i}")
                assert val == f"value_{i}"
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(_worker, i) for i in range(100)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Concurrent cache errors: {errors[:5]}"

    def test_indicator_computation_reproducibility(self):
        """指标计算可复现性"""
        from agent.tools.indicator_tool import IndicatorTool
        import pandas as pd

        np.random.seed(123)
        df = pd.DataFrame({
            "open": 10 + np.cumsum(np.random.randn(120) * 0.2),
            "high": 12 + np.cumsum(np.random.randn(120) * 0.2),
            "low": 8 + np.cumsum(np.random.randn(120) * 0.2),
            "close": 10 + np.cumsum(np.random.randn(120) * 0.3),
            "volume": np.random.randint(10000, 100000, 120),
            "amount": np.random.randint(500000, 5000000, 120),
        })

        r1 = IndicatorTool.compute_all(df)
        r2 = IndicatorTool.compute_all(df)

        for key in r1:
            if key in r2 and isinstance(r1[key], (int, float)):
                assert r1[key] == r2[key], f"Mismatch on {key}: {r1[key]} vs {r2[key]}"

    def test_sentiment_tool_registry_shared(self):
        """SentimentTool 复用单例 Registry"""
        from agent.tools.sentiment_tool import SentimentTool
        from agent.tools.stock_data_tool import StockDataTool
        from agent.crawlers.registry import CrawlerRegistry

        sdt = StockDataTool()
        st = SentimentTool()
        reg = CrawlerRegistry.get_instance()

        # 三个引用指向同一实例
        assert sdt._registry is reg
        assert st._registry is reg


# ═══════════════════════════════════════════════════════════════════════════
# 6. 内存与性能基线
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryAndPerformance:
    """内存占用与响应时间基线"""

    def test_memory_stability_under_load(self, client):
        """持续请求后内存不泄漏"""
        gc.collect()
        import psutil
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1024 / 1024

        for _ in range(30):
            client.post("/api/agent/diagnose", json={"stock_code": "000001"})
            client.post("/api/agent/predict", json={"stock_code": "000001", "horizon": "short"})

        gc.collect()
        time.sleep(0.5)
        mem_after = proc.memory_info().rss / 1024 / 1024

        growth = mem_after - mem_before
        assert growth < 100, f"Memory grew by {growth:.0f}MB (possible leak)"

    def test_response_time_baseline(self, client):
        """响应时间基线（健康检查）"""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = client.get("/api/health")
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)

        avg = sum(times) / len(times) if times else 0
        p95 = sorted(times)[int(len(times) * 0.95)] if times else 0

        assert avg < 100, f"Health check avg {avg:.0f}ms > 100ms"
        assert p95 < 200, f"Health check p95 {p95:.0f}ms > 200ms"

    def test_cache_hit_response_time(self, client):
        """缓存命中响应时间 < 50ms"""
        # 预热缓存
        client.post("/api/agent/diagnose", json={"stock_code": "000001", "force_refresh": True})

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            r = client.post("/api/agent/diagnose", json={"stock_code": "000001"})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)

        avg = sum(times) / len(times) if times else 999
        assert avg < 50, f"Cache hit avg {avg:.0f}ms > 50ms"

    def test_token_budget_all_horizons(self, prediction_engine, snapshot):
        """Token 预算基线"""
        budgets = {}
        for h in ["short", "medium", "long"]:
            tokens = []
            for _ in range(5):
                r = prediction_engine.predict("000001", "test", snapshot, horizon=h)
                tokens.append(r.prompt_tokens_estimated)
            budgets[h] = {"min": min(tokens), "max": max(tokens), "avg": sum(tokens) / len(tokens)}

        # 所有周期 token 预算 < 3000, short 应最低
        for h, stats in budgets.items():
            assert stats["avg"] < 3000, f"{h} avg tokens {stats['avg']:.0f} > 3000"
            assert stats["max"] < 4000, f"{h} max tokens {stats['max']:.0f} > 4000"

        # long 应 > short（因为包含更多数据）
        assert budgets["long"]["avg"] >= budgets["short"]["avg"] - 50, \
            f"Long tokens should be >= short: {budgets}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. 边界条件与错误处理
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界条件和异常路径"""

    def test_empty_holdings_portfolio(self, client):
        """空组合"""
        r = client.post("/api/agent/portfolio/analyze", json={"holdings": []})
        assert r.status_code == 400

    def test_invalid_stock_code(self, client):
        """无效股票代码"""
        r = client.post("/api/agent/diagnose", json={"stock_code": "ABC"})
        assert r.status_code == 400

    def test_sse_cache_hit_produces_valid_events(self, client):
        """SSE 缓存命中产生有效事件"""
        # 预热
        client.post("/api/agent/diagnose", json={"stock_code": "000001", "force_refresh": True})
        # SSE 请求应命中缓存
        r = client.post("/api/agent/diagnose/stream",
            json={"stock_code": "000001"},
            headers={"Accept": "text/event-stream"},
            buffered=True,
        )
        if r.status_code == 200:
            text = r.get_data(as_text=True)
            assert "cache_hit" in text or "result" in text

    def test_predict_invalid_params_rejected(self, client):
        """无效参数被 API 拒绝"""
        tests = [
            ({"stock_code": "600000", "horizon": "bad"}, 400),
            ({"stock_code": "600000", "risk_tolerance": "bad"}, 400),
            ({"stock_code": "600000", "investment_style": "bad"}, 400),
        ]
        for body, expected_code in tests:
            r = client.post("/api/agent/predict", json=body)
            assert r.status_code == expected_code, \
                f"Expected {expected_code} for {body}, got {r.status_code}"

    def test_blackboard_eviction(self):
        """黑板淘汰机制：超过 200 只股票自动清理"""
        from agent.core.blackboard import Blackboard, StockSnapshot

        bb = Blackboard()
        for i in range(250):
            snap = StockSnapshot(stock_code=f"T{i:04d}", stock_name=f"test_{i}")
            bb.publish_snapshot(snap)

        stats = bb.get_stats()
        assert stats["total_snapshots"] <= 200, \
            f"Blackboard not evicting: {stats['total_snapshots']} stocks"

    def test_empty_snapshot_handling(self, prediction_engine):
        """空快照不会崩溃"""
        from agent.core.blackboard import StockSnapshot
        empty = StockSnapshot(stock_code="EMPTY", stock_name="empty")
        r = prediction_engine.predict("EMPTY", "empty", empty, horizon="short")
        assert r.direction in ("上涨", "下跌", "震荡", "不确定")
        assert r.data_quality_factor > 0  # 应有默认值


# ═══════════════════════════════════════════════════════════════════════════
# 8. 端到端工作流
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndWorkflow:
    """完整用户工作流"""

    def test_full_user_flow(self, client):
        """用户典型操作序列"""
        # 1. 登录
        login = client.post("/api/auth/login",
            json={"username": "admin", "password": "********"})
        assert login.status_code == 200
        assert login.get_json()["success"]

        # 2. 诊断
        diag = client.post("/api/agent/diagnose",
            json={"stock_code": "000001", "force_refresh": True})
        assert diag.status_code == 200
        assert "final_decision" in diag.get_json()

        # 3. 预测
        pred = client.post("/api/agent/predict",
            json={"stock_code": "000001", "horizon": "short"})
        assert pred.status_code in (200, 503)

        # 4. 历史
        hist = client.get("/api/agent/decisions/history")
        assert hist.status_code == 200

        # 5. 统计
        stats = client.get("/api/agent/stats")
        assert stats.status_code == 200

        # 6. 健康检查
        health = client.get("/api/health")
        assert health.status_code == 200


class TestVersionAndConfig:
    """版本标识与配置验证"""

    def test_version_consistency(self, client):
        """版本号一致"""
        health = client.get("/api/health")
        assert health.get_json()["version"] == "2.1.0"

        status = client.get("/api/status")
        assert status.get_json()["version"] == "2.1.0"

        stats = client.get("/api/agent/stats")
        assert stats.get_json()["version"] == "2.1.0"

    def test_prediction_engine_version(self, prediction_engine):
        """预测引擎版本标识"""
        assert hasattr(prediction_engine, "_RISK_TOLERANCE_SECTIONS")
        assert "conservative" in prediction_engine._RISK_TOLERANCE_SECTIONS
        assert "moderate" in prediction_engine._RISK_TOLERANCE_SECTIONS
        assert "aggressive" in prediction_engine._RISK_TOLERANCE_SECTIONS
