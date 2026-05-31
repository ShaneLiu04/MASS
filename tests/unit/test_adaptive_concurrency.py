"""
MASS 自适应并发控制器单元测试
v2.3 新增
"""
import threading
import time
import pytest

from agent.core.concurrency_controller import AdaptiveConcurrencyController


class TestAdaptiveConcurrencyController:
    """测试 AdaptiveConcurrencyController 的核心功能"""

    def test_init_defaults(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=10, window_size=50
        )
        assert ctrl.min_concurrent == 2
        assert ctrl.max_concurrent == 10
        assert ctrl.get_limit() == 2  # 初始化为最小值

    def test_init_min_greater_than_max(self):
        """min > max 时，max 应该被修正为至少 min"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=5, max_concurrent=3, window_size=10
        )
        assert ctrl.min_concurrent == 5
        assert ctrl.max_concurrent == 5

    def test_acquire_release_basic(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=5, window_size=10
        )
        # 初始限制为 2
        assert ctrl.acquire(timeout=1.0) is True
        assert ctrl.acquire(timeout=1.0) is True
        # 第3次应该超时（因为限制是2）
        assert ctrl.acquire(timeout=0.2) is False

        ctrl.release()
        assert ctrl.acquire(timeout=0.5) is True

    def test_release_never_exceeds_max(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=1, max_concurrent=3, window_size=10
        )
        ctrl.release()  # 从1增加到2
        ctrl.release()  # 从2增加到3
        ctrl.release()  # 不能超过max，仍为3
        assert ctrl.get_limit() == 3

    def test_record_result_success_no_adjust(self):
        """窗口不足时不应调整"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=10, window_size=50
        )
        for i in range(5):
            ctrl.record_result(duration=5.0, success=True, wait_time=0.0)
        # 窗口不足10条，不应调整
        assert ctrl.get_limit() == 2

    def test_record_result_slow_degradation(self):
        """响应时间慢 → 降级"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2,
            max_concurrent=10,
            slow_threshold=5.0,
            fast_threshold=1.0,
            error_high=0.20,
            error_low=0.05,
            window_size=20,
            cooldown_seconds=0.0,  # 关闭冷却期便于测试
        )
        # 先提升到较高水平（通过快速成功）
        for i in range(12):
            ctrl.record_result(duration=0.5, success=True, wait_time=0.0)
        # 现在限制应该已经增加（因为响应快且错误率低）
        limit_after_fast = ctrl.get_limit()
        assert limit_after_fast > 2

        # 然后大量慢响应
        for i in range(15):
            ctrl.record_result(duration=10.0, success=True, wait_time=0.0)
        limit_after_slow = ctrl.get_limit()
        # 应该降级
        assert limit_after_slow < limit_after_fast

    def test_record_result_error_degradation(self):
        """错误率高 → 大幅降级"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2,
            max_concurrent=10,
            slow_threshold=30.0,
            fast_threshold=10.0,
            error_high=0.20,
            error_low=0.05,
            window_size=20,
            cooldown_seconds=0.0,
        )
        # 先提升到较高水平
        for i in range(12):
            ctrl.record_result(duration=5.0, success=True, wait_time=0.0)
        limit_before = ctrl.get_limit()
        assert limit_before > 2

        # 大量错误（>20%）
        for i in range(15):
            ctrl.record_result(duration=5.0, success=False, wait_time=0.0)
        limit_after = ctrl.get_limit()
        # 错误率高应该大幅降级（-2）
        assert limit_after <= limit_before - 1

    def test_record_result_wait_time_degradation(self):
        """队列等待时间长 → 降级"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2,
            max_concurrent=10,
            slow_threshold=30.0,
            fast_threshold=10.0,
            error_high=0.20,
            error_low=0.05,
            window_size=20,
            cooldown_seconds=0.0,
        )
        # 先提升
        for i in range(12):
            ctrl.record_result(duration=5.0, success=True, wait_time=0.0)
        limit_before = ctrl.get_limit()
        assert limit_before > 2

        # 大量长等待时间
        for i in range(15):
            ctrl.record_result(duration=5.0, success=True, wait_time=8.0)
        limit_after = ctrl.get_limit()
        assert limit_after < limit_before

    def test_cooldown_prevents_adjustment(self):
        """冷却期内不应调整"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2,
            max_concurrent=10,
            slow_threshold=5.0,
            fast_threshold=1.0,
            error_high=0.20,
            error_low=0.05,
            window_size=20,
            cooldown_seconds=10.0,  # 长冷却期
        )
        for i in range(12):
            ctrl.record_result(duration=0.5, success=True, wait_time=0.0)
        limit_after_fast = ctrl.get_limit()
        assert limit_after_fast > 2

        # 紧接着慢响应（在冷却期内）
        for i in range(5):
            ctrl.record_result(duration=20.0, success=True, wait_time=0.0)
        # 冷却期内不应调整
        assert ctrl.get_limit() == limit_after_fast

    def test_get_stats(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=10, window_size=20
        )
        ctrl.record_result(duration=5.0, success=True, wait_time=1.0)
        ctrl.record_result(duration=3.0, success=False, wait_time=0.5)

        stats = ctrl.get_stats()
        assert stats["current_limit"] == 2
        assert stats["min_concurrent"] == 2
        assert stats["max_concurrent"] == 10
        assert stats["total_requests"] == 2
        assert stats["total_success"] == 1
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 0.5
        assert stats["avg_response_time"] == 4.0
        assert stats["avg_wait_time"] == 0.75
        assert stats["window_size"] == 2
        assert "last_adjustment_ago" in stats

    def test_reset(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=2, max_concurrent=10, window_size=20, cooldown_seconds=0.0
        )
        for i in range(15):
            ctrl.record_result(duration=0.5, success=True, wait_time=0.0)
        assert ctrl.get_limit() > 2
        assert ctrl.get_stats()["total_requests"] == 15

        ctrl.reset()
        assert ctrl.get_limit() == 2
        assert ctrl.get_stats()["total_requests"] == 0
        assert ctrl.get_stats()["window_size"] == 0

    def test_concurrent_acquire_release(self):
        """多线程并发获取和释放"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=5, max_concurrent=5, window_size=10
        )
        results = []
        threads = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()  # 等待所有线程就绪，同时开始
            ok = ctrl.acquire(timeout=0.02)  # 超时很短，确保只有初始5个能获取
            results.append(ok)
            if ok:
                time.sleep(0.5)  # 持有较长时间，确保其他线程在此期间无法获取
                ctrl.release()

        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 5个成功，5个失败（因为限制是5）
        assert sum(results) == 5
        assert len([r for r in results if not r]) == 5

    def test_acquire_timeout_records_wait_time(self):
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=1, max_concurrent=5, window_size=10
        )
        ctrl.acquire(timeout=1.0)  # 占住唯一许可
        # 再尝试获取，会超时
        wait_start = time.time()
        result = ctrl.acquire(timeout=0.3)
        wait_time = time.time() - wait_start
        assert result is False
        # 等待时间应该被记录
        stats = ctrl.get_stats()
        assert stats["avg_wait_time"] > 0

        ctrl.release()

    def test_limit_never_below_min(self):
        """降级不应低于最小值"""
        ctrl = AdaptiveConcurrencyController(
            min_concurrent=3,
            max_concurrent=10,
            slow_threshold=1.0,
            fast_threshold=0.5,
            error_high=0.10,
            error_low=0.01,
            window_size=20,
            cooldown_seconds=0.0,
        )
        # 大量失败 + 慢响应
        for i in range(30):
            ctrl.record_result(duration=50.0, success=False, wait_time=10.0)
        assert ctrl.get_limit() >= 3
