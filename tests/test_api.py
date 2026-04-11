"""api/serve.py — FastAPI 엔드포인트 단위 테스트 (서버 없이 TestClient 사용)"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.serve import app

client = TestClient(app)


def test_health_no_model(tmp_path, monkeypatch):
    """배포된 모델이 없을 때 health는 model_deployed=False를 반환해야 한다."""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_deployed"] is False
    assert body["model_version"] is None


def test_model_info_no_model(tmp_path, monkeypatch):
    """배포된 모델이 없을 때 /model/info는 503을 반환해야 한다."""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    r = client.get("/model/info")
    assert r.status_code == 503


def test_predict_no_model(tmp_path, monkeypatch):
    """배포된 모델이 없을 때 /predict는 503을 반환해야 한다."""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    r = client.post("/predict", json={"features": [{"a": 1.0}]})
    assert r.status_code == 503


def test_predict_empty_features(tmp_path, monkeypatch):
    """빈 features 리스트는 422를 반환해야 한다."""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 422


def test_predict_with_mock_model(tmp_path, monkeypatch):
    """모델이 있을 때 /predict는 predictions를 반환해야 한다."""
    monkeypatch.setattr(
        "api.serve.get_active_model_info",
        lambda: {"version": "v1", "model_name": "mock"},
    )
    monkeypatch.setattr(
        "api.serve.predict",
        lambda X: np.array([0, 1]),
    )
    r = client.post("/predict", json={"features": [{"a": 1.0}, {"a": 2.0}]})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"] == [0, 1]
    assert body["n_samples"] == 2
    assert body["model_version"] == "v1"


def test_predict_proba_with_mock_model(tmp_path, monkeypatch):
    """/predict/proba는 확률 배열을 반환해야 한다."""
    monkeypatch.setattr(
        "api.serve.get_active_model_info",
        lambda: {"version": "v1", "model_name": "mock"},
    )
    monkeypatch.setattr(
        "api.serve.predict_proba",
        lambda X: np.array([[0.8, 0.2], [0.3, 0.7]]),
    )
    r = client.post("/predict/proba", json={"features": [{"a": 1.0}, {"a": 2.0}]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 2
    assert abs(sum(body["probabilities"][0]) - 1.0) < 1e-4
