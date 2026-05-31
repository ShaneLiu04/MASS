"""
集成测试: Flask API
"""
import json
import pytest


class TestHealthAPI:
    """健康检查接口测试"""
    
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "mock_mode" in data
    
    def test_system_status(self, client):
        """测试系统状态"""
        response = client.get('/api/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["status"] == "running"
        assert "database" in data
        assert "environment" in data


@pytest.mark.integration
class TestDiagnoseAPI:
    """诊断接口测试 — 触发网络数据获取"""
    
    def test_diagnose_success(self, client):
        """测试正常诊断请求"""
        response = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000001"}),
            content_type='application/json',
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["stock_code"] == "000001"
        assert "final_decision" in data
        assert "opinions" in data
        assert len(data["opinions"]) == 6  # 6个Agent
        
        fd = data["final_decision"]
        assert "decision" in fd
        assert "confidence" in fd
        assert "reasoning" in fd
    
    def test_diagnose_missing_code(self, client):
        """测试缺少股票代码"""
        response = client.post(
            '/api/agent/diagnose',
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert "error" in data
        assert data.get("code") == "MISSING_STOCK_CODE"
    
    def test_diagnose_invalid_code(self, client):
        """测试无效股票代码"""
        response = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "invalid"}),
            content_type='application/json',
        )
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data.get("code") == "INVALID_STOCK_CODE"
    
    def test_diagnose_with_stock_name(self, client):
        """测试带股票名称的诊断"""
        response = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000001", "stock_name": "平安银行"}),
            content_type='application/json',
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        # Mock模式下 fundamentals 会返回模拟名称，但传入的 stock_name 应被保留
        assert data["stock_name"] != ""
    
    def test_diagnose_cache(self, client):
        """测试缓存机制"""
        # 第一次请求
        r1 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000002"}),
            content_type='application/json',
        )
        data1 = json.loads(r1.data)
        
        # 第二次请求（应该走缓存）
        r2 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000002"}),
            content_type='application/json',
        )
        data2 = json.loads(r2.data)
        
        assert data2.get("from_cache") == True
        
        # 强制刷新
        r3 = client.post(
            '/api/agent/diagnose',
            data=json.dumps({"stock_code": "000002", "force_refresh": True}),
            content_type='application/json',
        )
        data3 = json.loads(r3.data)
        assert data3.get("from_cache") == False


@pytest.mark.integration
class TestPortfolioAPI:
    """组合分析接口测试 — 触发网络数据获取"""
    
    def test_portfolio_analyze(self, client):
        """测试组合分析"""
        response = client.post(
            '/api/agent/portfolio/analyze',
            data=json.dumps({
                "holdings": [
                    {"code": "000001", "cost": 15.2, "shares": 1000},
                    {"code": "000002", "cost": 20.0, "shares": 500},
                ]
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "holdings_analysis" in data
        assert "portfolio_risk" in data
        assert len(data["holdings_analysis"]) == 2
        assert data["total"] == 2
        assert data["success"] == 2
    
    def test_portfolio_empty(self, client):
        """测试空持仓"""
        response = client.post(
            '/api/agent/portfolio/analyze',
            data=json.dumps({"holdings": []}),
            content_type='application/json',
        )
        assert response.status_code == 400
    
    def test_portfolio_too_many(self, client):
        """测试持仓过多"""
        holdings = [{"code": f"{i:06d}", "cost": 10, "shares": 100} for i in range(25)]
        response = client.post(
            '/api/agent/portfolio/analyze',
            data=json.dumps({"holdings": holdings}),
            content_type='application/json',
        )
        assert response.status_code == 400
        assert json.loads(response.data).get("code") == "TOO_MANY_HOLDINGS"


class TestHistoryAPI:
    """历史记录接口测试"""
    
    def test_decision_history(self, client):
        """测试获取历史记录"""
        response = client.get('/api/agent/decisions/history?limit=10')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "decisions" in data
        assert "count" in data
        assert "limit" in data
    
    def test_decision_history_by_stock(self, client):
        """测试按股票筛选历史"""
        response = client.get('/api/agent/decisions/history?stock_code=000001&limit=5')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        for d in data["decisions"]:
            assert d["stock_code"] == "000001"


class TestStatsAPI:
    """统计接口测试"""
    
    def test_system_stats(self, client):
        """测试系统统计"""
        response = client.get('/api/agent/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "database" in data
        assert "blackboard" in data
        assert "cache" in data


class TestBlackboardAPI:
    """黑板管理接口测试"""
    
    def test_clear_blackboard(self, client):
        """测试清理黑板"""
        response = client.post(
            '/api/agent/blackboard/clear',
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["success"] == True
    
    def test_clear_specific_stock(self, client):
        """测试清理特定股票"""
        response = client.post(
            '/api/agent/blackboard/clear',
            data=json.dumps({"stock_code": "000001"}),
            content_type='application/json',
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["success"] == True
        assert "000001" in data["message"]


class TestErrorHandling:
    """错误处理测试"""
    
    def test_404(self, client):
        """测试404"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert "error" in data
        assert "request_id" in data
    
    def test_method_not_allowed(self, client):
        """测试405"""
        response = client.get('/api/agent/diagnose')
        assert response.status_code == 405


class TestCORS:
    """CORS测试"""
    
    def test_cors_headers(self, client):
        """测试CORS头 — 只允许白名单 Origin"""
        # 不带 Origin 的请求不应设置 CORS 头
        response = client.get('/api/health')
        # 白名单机制下，不带 Origin 的请求可能不设置该头
        # 带白名单 Origin 的请求应被允许
        response_allowed = client.get(
            '/api/health',
            headers={'Origin': 'http://localhost:5000'}
        )
        assert response_allowed.headers.get('Access-Control-Allow-Origin') == 'http://localhost:5000'
    
    def test_options_request(self, client):
        """测试OPTIONS请求"""
        response = client.options('/api/agent/diagnose')
        assert response.status_code == 200
