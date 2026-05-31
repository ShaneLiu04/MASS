"""
MASS Prediction API 集成测试
验证: /api/agent/predict 端点

注意: 本文件所有测试依赖外部网络（akshare 数据获取），
      默认被 pytest 跳过，需显式运行 `pytest -m integration`
"""
import json
import pytest


@pytest.mark.integration
class TestPredictAPI:
    """预测 API 集成测试 — 需外部网络"""

    @pytest.fixture
    def client(self):
        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_predict_success(self, client):
        """测试预测成功"""
        resp = client.post(
            "/api/agent/predict",
            json={"stock_code": "600000", "horizon": "short"},
        )
        assert resp.status_code in (200, 503)
        data = resp.get_json()
        if resp.status_code == 200:
            assert data.get("stock_code") == "600000"
            assert "direction" in data
            assert "confidence" in data
            assert 0 <= data.get("confidence", 0) <= 1
            assert "probability_up" in data
            assert "probability_down" in data
            assert "probability_sideways" in data
            assert "key_drivers" in data
            assert "risk_factors" in data
            assert "reasoning" in data
            assert data.get("model_used") == "deepseek-v4-pro"
        else:
            assert data.get("code") == "DATA_UNAVAILABLE"

    def test_predict_with_model_params(self, client):
        """测试带模型参数的预测"""
        resp = client.post(
            "/api/agent/predict",
            json={
                "stock_code": "600000",
                "horizon": "medium",
                "model_params": {"temperature": 0.5, "top_p": 0.9},
            },
        )
        assert resp.status_code in (200, 503)
        data = resp.get_json()
        if resp.status_code == 200:
            assert data.get("prediction_horizon") == "medium"
            assert data.get("model_params", {}).get("temperature") == 0.5

    def test_predict_missing_code(self, client):
        """测试缺少股票代码"""
        resp = client.post("/api/agent/predict", json={"horizon": "short"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "MISSING_STOCK_CODE"

    def test_predict_invalid_horizon(self, client):
        """测试无效的预测周期"""
        resp = client.post(
            "/api/agent/predict",
            json={"stock_code": "600000", "horizon": "invalid"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_HORIZON"

    def test_predict_invalid_stock_code(self, client):
        """测试无效的股票代码"""
        resp = client.post(
            "/api/agent/predict",
            json={"stock_code": "ABC123", "horizon": "short"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_STOCK_CODE"

    def test_predict_long_horizon(self, client):
        """测试长期预测"""
        resp = client.post(
            "/api/agent/predict",
            json={"stock_code": "000001", "horizon": "long"},
        )
        assert resp.status_code in (200, 503)
        data = resp.get_json()
        if resp.status_code == 200:
            assert data.get("prediction_horizon") == "long"
