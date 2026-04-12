"""agents/ds_development_agent.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ── _parse_response ───────────────────────────────────────────────────────────

def test_parse_valid_response():
    from agents.ds_development_agent import _parse_response
    raw = json.dumps({
        "target_column": "SeriousDlqin2yrs",
        "models_to_try": ["hist_gradient_boosting", "lightgbm"],
        "primary_metric": "roc_auc",
        "feature_exclusions": [],
        "reason": "기본 설정",
        "confidence": 0.9,
    })
    result = _parse_response(raw)
    assert result["target_column"] == "SeriousDlqin2yrs"
    assert "hist_gradient_boosting" in result["models_to_try"]
    assert result["confidence"] == 0.9


def test_parse_filters_invalid_models():
    from agents.ds_development_agent import _parse_response, DEFAULT_PLAN
    raw = json.dumps({
        "target_column": "SeriousDlqin2yrs",
        "models_to_try": ["unknown_model", "another_invalid"],  # 모두 유효하지 않음
        "primary_metric": "roc_auc",
        "feature_exclusions": [],
        "reason": "테스트",
        "confidence": 0.8,
    })
    result = _parse_response(raw)
    # 유효 모델 < 2개 → default로 교체
    assert result["models_to_try"] == DEFAULT_PLAN["models_to_try"]


def test_parse_keeps_valid_models_only():
    from agents.ds_development_agent import _parse_response
    raw = json.dumps({
        "target_column": "SeriousDlqin2yrs",
        "models_to_try": ["hist_gradient_boosting", "invalid_model", "xgboost"],
        "primary_metric": "roc_auc",
        "feature_exclusions": [],
        "reason": "테스트",
        "confidence": 0.7,
    })
    result = _parse_response(raw)
    assert "invalid_model" not in result["models_to_try"]
    assert "hist_gradient_boosting" in result["models_to_try"]
    assert "xgboost" in result["models_to_try"]


def test_parse_invalid_json_returns_default():
    from agents.ds_development_agent import _parse_response, DEFAULT_PLAN
    result = _parse_response("not valid json {{")
    assert result["target_column"] == DEFAULT_PLAN["target_column"]
    assert result["confidence"] == 0.0
    assert "파싱 실패" in result["reason"]


def test_parse_missing_fields_use_defaults():
    from agents.ds_development_agent import _parse_response, DEFAULT_PLAN
    # 필드 일부 누락
    raw = json.dumps({"reason": "부분 응답"})
    result = _parse_response(raw)
    assert result["target_column"] == DEFAULT_PLAN["target_column"]
    assert result["primary_metric"] == DEFAULT_PLAN["primary_metric"]


# ── plan_experiment fallback ──────────────────────────────────────────────────

def test_plan_experiment_use_agent_false_returns_default():
    from agents.ds_development_agent import plan_experiment, DEFAULT_PLAN
    result = plan_experiment(
        columns=["col1", "col2", "SeriousDlqin2yrs"],
        n_samples=10000,
        n_features=10,
        imbalance_ratio=0.07,
        use_agent=False,
    )
    assert result["target_column"] == DEFAULT_PLAN["target_column"]
    assert result["models_to_try"] == DEFAULT_PLAN["models_to_try"]


def test_plan_experiment_llm_failure_returns_default():
    from agents.ds_development_agent import plan_experiment, DEFAULT_PLAN
    with patch("agents.ds_development_agent._call_llm", side_effect=Exception("API 오류")):
        result = plan_experiment(
            columns=["col1", "SeriousDlqin2yrs"],
            n_samples=5000,
            n_features=5,
            imbalance_ratio=0.07,
        )
    assert result["target_column"] == DEFAULT_PLAN["target_column"]
    assert result["models_to_try"] == DEFAULT_PLAN["models_to_try"]


def test_plan_experiment_llm_success():
    from agents.ds_development_agent import plan_experiment
    mock_response = json.dumps({
        "target_column": "SeriousDlqin2yrs",
        "models_to_try": ["hist_gradient_boosting", "xgboost"],
        "primary_metric": "roc_auc",
        "feature_exclusions": ["id"],
        "reason": "LLM 정상 응답",
        "confidence": 0.95,
    })
    with patch("agents.ds_development_agent._call_llm", return_value=mock_response):
        result = plan_experiment(
            columns=["id", "age", "SeriousDlqin2yrs"],
            n_samples=15000,
            n_features=2,
            imbalance_ratio=0.07,
        )
    assert result["target_column"] == "SeriousDlqin2yrs"
    assert result["confidence"] == 0.95
    assert "id" in result["feature_exclusions"]


# ── DEFAULT_PLAN 불변성 ───────────────────────────────────────────────────────

def test_default_plan_not_mutated_by_fallback():
    """plan_experiment 반환값 수정이 DEFAULT_PLAN에 영향 없어야 함"""
    from agents.ds_development_agent import plan_experiment, DEFAULT_PLAN
    result = plan_experiment(
        columns=["col"], n_samples=100, n_features=1,
        imbalance_ratio=0.1, use_agent=False
    )
    result["models_to_try"].append("mutated")
    assert "mutated" not in DEFAULT_PLAN["models_to_try"]


# ── _build_schema_summary ─────────────────────────────────────────────────────

def test_build_schema_summary_is_valid_json():
    from agents.ds_development_agent import _build_schema_summary
    summary = _build_schema_summary(["col1", "col2"], 10000, 2, 0.07)
    parsed = json.loads(summary)
    assert parsed["n_samples"] == 10000
    assert parsed["imbalance_ratio"] == 0.07
