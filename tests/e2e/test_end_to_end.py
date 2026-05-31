"""
端到端测试: 完整用户流程模拟
"""
import json
import pytest


class TestEndToEnd:
    """端到端测试"""
    
    def test_complete_user_journey(self, client):
        """
        模拟完整用户旅程:
        1. 健康检查
        2. 诊断单只股票
        3. 查看历史记录
        4. 组合分析
        5. 查看统计
        """
        # Step 1: 健康检查
        r1 = client.get('/api/health')
        assert r1.status_code == 200
        health = json.loads(r1.data)
        assert health["status"] == "healthy"
        
        # Step 2: 诊断股票
        r2 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000001"}),
            content_type='application/json',
        )
        assert r2.status_code == 200
        diagnose = json.loads(r2.data)
        assert diagnose["stock_code"] == "000001"
        decision_id = diagnose.get("id")  # 如果有返回id
        
        # Step 3: 查看历史记录
        r3 = client.get('/api/agent/decisions/history?limit=5')
        assert r3.status_code == 200
        history = json.loads(r3.data)
        assert "decisions" in history
        
        # Step 4: 组合分析
        r4 = client.post(
            '/api/agent/portfolio/analyze',
            data=json.dumps({
                "holdings": [
                    {"code": "000001", "cost": 15.0, "shares": 1000},
                    {"code": "000002", "cost": 20.0, "shares": 500},
                ]
            }),
            content_type='application/json',
        )
        assert r4.status_code == 200
        portfolio = json.loads(r4.data)
        assert len(portfolio["holdings_analysis"]) == 2
        assert "portfolio_risk" in portfolio
        
        # Step 5: 查看统计
        r5 = client.get('/api/agent/stats')
        assert r5.status_code == 200
        stats = json.loads(r5.data)
        assert "database" in stats
        assert "cache" in stats
    
    def test_frontend_pages(self, client):
        """测试前端页面可访问"""
        pages = [
            '/',
            '/agent/trading',
        ]
        
        for page in pages:
            response = client.get(page)
            assert response.status_code == 200
            # 检查是否包含关键内容
            assert b'MASS' in response.data or b'mass' in response.data.lower()
    
    def test_api_response_headers(self, client):
        """测试API响应头"""
        response = client.get('/api/health')
        
        assert 'X-Request-ID' in response.headers
        assert 'X-Response-Time' in response.headers
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_concurrent_diagnosis(self, client):
        """测试并发诊断请求"""
        import threading
        
        results = []
        errors = []
        
        def diagnose(code):
            try:
                r = client.post(
                    '/api/agent/diagnose',
                    data=json.dumps({"stock_code": code}),
                    content_type='application/json',
                )
                results.append((code, r.status_code))
            except Exception as e:
                errors.append((code, str(e)))
        
        codes = [f"{i:06d}" for i in range(1, 6)]
        threads = [threading.Thread(target=diagnose, args=(c,)) for c in codes]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"并发请求出错: {errors}"
        assert len(results) == 5
        for code, status in results:
            assert status == 200, f"股票 {code} 请求失败: {status}"
    
    def test_error_recovery(self, client):
        """测试错误恢复能力"""
        # 发送无效请求
        r1 = client.post('/api/agent/diagnose', data='invalid json')
        assert r1.status_code in (400, 500)
        
        # 系统应该仍然可用
        r2 = client.get('/api/health')
        assert r2.status_code == 200
        
        # 发送正常请求
        r3 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000001"}),
            content_type='application/json',
        )
        assert r3.status_code == 200
    
    def test_cache_behavior(self, client):
        """测试缓存行为"""
        import time
        
        # 第一次请求
        start1 = time.time()
        r1 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000009"}),
            content_type='application/json',
        )
        duration1 = time.time() - start1
        
        # 第二次请求（缓存）
        start2 = time.time()
        r2 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000009"}),
            content_type='application/json',
        )
        duration2 = time.time() - start2
        
        data2 = json.loads(r2.data)
        
        # Mock模式下缓存和非缓存都很快，但结构上应该正确
        assert data2.get("from_cache") == True
    
    def test_decision_package_structure(self, client):
        """测试决策包完整结构"""
        response = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000001"}),
            content_type='application/json',
        )
        
        data = json.loads(response.data)
        
        # 检查所有必需字段
        required_top = [
            "stock_code", "stock_name", "current_price",
            "decision_date", "decision_time", "market_cycle",
            "opinions", "final_decision", "disclaimer",
            "version", "processing_time_seconds", "data_summary",
        ]
        for field in required_top:
            assert field in data, f"缺少字段: {field}"
        
        # 检查final_decision结构
        fd = data["final_decision"]
        required_fd = ["decision", "confidence", "reasoning"]
        for field in required_fd:
            assert field in fd, f"final_decision 缺少字段: {field}"
        
        # 检查opinions结构
        for agent_id, opinion in data["opinions"].items():
            assert "signal" in opinion
            assert "confidence" in opinion
            assert "reasoning" in opinion
            assert "key_factors" in opinion
            assert "risk_flags" in opinion
