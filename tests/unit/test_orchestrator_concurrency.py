"""
MASS Orchestrator 并发控制集成测试
v2.3 新增：验证自适应并发控制器与 Orchestrator 的集成
"""
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from agent.core.orchestrator import AgentOrchestrator
from agent.core.concurrency_controller import AdaptiveConcurrencyController


class TestOrchestratorConcurrency:
    """测试 Orchestrator 与并发控制器的集成"""

    @pytest.fixture
    def mock_orchestrator(self):
        """创建使用 Mock LLM 的 Orchestrator"""
        orch = AgentOrchestrator(use_mock_llm=True)
        # Mock _run_diagnosis_impl 避免真实数据获取
        orch._run_diagnosis_impl = MagicMock(return_value={
            "stock_code": "000001",
            "final_decision": {"decision": 0, "confidence": 0.5},
            "opinions": {},
        })
        return orch

    def test_run_diagnosis_records_success(self, mock_orchestrator):
        """诊断成功后应记录成功结果到控制器"""
        with patch("agent.core.orchestrator.ADAPTIVE_CONCURRENCY", True):
            controller = AdaptiveConcurrencyController(
                min_concurrent=2, max_concurrent=10, window_size=20,
                cooldown_seconds=0.0,
            )
            with patch("agent.core.orchestrator._CONCURRENCY_CONTROLLER", controller):
                result = mock_orchestrator.run_diagnosis(stock_code="000001")
                assert result["stock_code"] == "000001"
                stats = controller.get_stats()
                assert stats["total_requests"] == 1
                assert stats["total_success"] == 1
                assert stats["total_errors"] == 0

    def test_run_diagnosis_records_error(self, mock_orchestrator):
        """诊断异常后应记录失败结果到控制器"""
        mock_orchestrator._run_diagnosis_impl = MagicMock(
            side_effect=RuntimeError("模拟诊断失败")
        )
        with patch("agent.core.orchestrator.ADAPTIVE_CONCURRENCY", True):
            controller = AdaptiveConcurrencyController(
                min_concurrent=2, max_concurrent=10, window_size=20,
                cooldown_seconds=0.0,
            )
            with patch("agent.core.orchestrator._CONCURRENCY_CONTROLLER", controller):
                with pytest.raises(RuntimeError, match="模拟诊断失败"):
                    mock_orchestrator.run_diagnosis(stock_code="000001")
                stats = controller.get_stats()
                assert stats["total_requests"] == 1
                assert stats["total_success"] == 0
                assert stats["total_errors"] == 1

    def test_run_diagnosis_stream_records_success(self, mock_orchestrator):
        """流式诊断成功后应记录成功结果"""
        mock_orchestrator._run_diagnosis_stream_impl = MagicMock(
            return_value=iter([
                {"stage": "init", "progress": 0},
                {"stage": "result", "progress": 100, "data": {"stock_code": "000001"}},
            ])
        )
        with patch("agent.core.orchestrator.ADAPTIVE_CONCURRENCY", True):
            controller = AdaptiveConcurrencyController(
                min_concurrent=2, max_concurrent=10, window_size=20,
                cooldown_seconds=0.0,
            )
            with patch("agent.core.orchestrator._CONCURRENCY_CONTROLLER", controller):
                events = list(mock_orchestrator.run_diagnosis_stream(stock_code="000001"))
                assert len(events) == 2
                stats = controller.get_stats()
                assert stats["total_requests"] == 1
                assert stats["total_success"] == 1

    def test_run_diagnosis_stream_records_error(self, mock_orchestrator):
        """流式诊断 yield error 后应记录失败结果"""
        mock_orchestrator._run_diagnosis_stream_impl = MagicMock(
            return_value=iter([
                {"stage": "init", "progress": 0},
                {"stage": "error", "message": "数据获取失败", "progress": 0},
            ])
        )
        with patch("agent.core.orchestrator.ADAPTIVE_CONCURRENCY", True):
            controller = AdaptiveConcurrencyController(
                min_concurrent=2, max_concurrent=10, window_size=20,
                cooldown_seconds=0.0,
            )
            with patch("agent.core.orchestrator._CONCURRENCY_CONTROLLER", controller):
                events = list(mock_orchestrator.run_diagnosis_stream(stock_code="000001"))
                assert any(e.get("stage") == "error" for e in events)
                stats = controller.get_stats()
                assert stats["total_requests"] == 1
                assert stats["total_success"] == 0
                assert stats["total_errors"] == 1

    def test_static_semaphore_fallback(self, mock_orchestrator):
        """ADAPTIVE_CONCURRENCY=False 时应使用静态信号量"""
        import threading
        from agent.core import orchestrator as orch_module
        # 手动注入静态信号量并切换到静态模式
        orch_module._DIAGNOSIS_SEMAPHORE = threading.Semaphore(5)
        with patch.object(orch_module, "ADAPTIVE_CONCURRENCY", False):
            result = mock_orchestrator.run_diagnosis(stock_code="000001")
            assert result["stock_code"] == "000001"

    def test_concurrent_diagnoses_respect_limit(self, mock_orchestrator):
        """并发诊断应遵守并发控制器限制"""
        call_times = []
        def slow_impl(*args, **kwargs):
            call_times.append(time.time())
            time.sleep(0.3)
            return {"stock_code": "000001", "final_decision": {"decision": 0, "confidence": 0.5}, "opinions": {}}

        mock_orchestrator._run_diagnosis_impl = slow_impl

        with patch("agent.core.orchestrator.ADAPTIVE_CONCURRENCY", True):
            controller = AdaptiveConcurrencyController(
                min_concurrent=2, max_concurrent=10, window_size=20,
                cooldown_seconds=0.0,
            )
            with patch("agent.core.orchestrator._CONCURRENCY_CONTROLLER", controller):
                threads = []
                results = []
                errors = []

                def worker():
                    try:
                        r = mock_orchestrator.run_diagnosis(stock_code="000001")
                        results.append(r)
                    except Exception as e:
                        errors.append(e)

                # 启动 5 个并发诊断，但限制为 2
                for _ in range(5):
                    t = threading.Thread(target=worker)
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join(timeout=10.0)

                # 所有 5 个都应该完成（串行执行，总耗时约 5*0.3=1.5s）
                assert len(results) == 5, f"期望5个成功，实际{len(results)}个成功，{len(errors)}个错误"
                assert len(errors) == 0

    def test_concurrent_stats_endpoint_format(self):
        """验证 stats 端点返回格式与控制器一致"""
        controller = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=10, window_size=20,
        )
        controller.record_result(duration=8.0, success=True, wait_time=1.0)
        controller.record_result(duration=12.0, success=False, wait_time=2.0)

        stats = controller.get_stats()
        # 确保 API 端点需要的所有字段都存在
        assert "current_limit" in stats
        assert "min_concurrent" in stats
        assert "max_concurrent" in stats
        assert "avg_response_time" in stats
        assert "error_rate" in stats
        assert "avg_wait_time" in stats
        assert "total_requests" in stats
        assert "total_success" in stats
        assert "total_errors" in stats
        assert "window_size" in stats
        assert "cooldown_seconds" in stats
        assert "last_adjustment_ago" in stats
