"""agents/operations_agent.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
# interpret_monitoring — 상태 해석
# ══════════════════════════════════════════════════════════════════════════════

GOOD_METRICS = {"roc_auc": 0.88, "error_rate": 0.12, "psi_mean": 0.05, "ks_pvalue_min": 0.30}
BAD_METRICS  = {"roc_auc": 0.60, "error_rate": 0.40, "psi_mean": 0.35, "ks_pvalue_min": 0.01}
EDGE_METRICS = {"roc_auc": 0.76, "error_rate": 0.24, "psi_mean": 0.19, "ks_pvalue_min": 0.06}


def test_rule_based_monitoring_healthy():
    from agents.operations_agent import interpret_monitoring
    result = interpret_monitoring("v1", GOOD_METRICS, use_agent=False)
    assert result["status"] == "healthy"
    assert result["confidence"] == 1.0


def test_rule_based_monitoring_critical():
    from agents.operations_agent import interpret_monitoring
    result = interpret_monitoring("v1", BAD_METRICS, use_agent=False)
    assert result["status"] == "critical"


def test_rule_based_monitoring_degraded():
    from agents.operations_agent import interpret_monitoring
    # 경계 근처 — degraded 예상 (10% 이내 위반)
    edge = {"roc_auc": 0.74, "error_rate": 0.26, "psi_mean": 0.21, "ks_pvalue_min": 0.04}
    result = interpret_monitoring("v1", edge, use_agent=False)
    assert result["status"] in ("degraded", "critical")


def test_monitoring_llm_success():
    from agents.operations_agent import interpret_monitoring
    mock_resp = json.dumps({"status": "degraded", "reason": "PSI 상승", "confidence": 0.85})
    with patch("agents.operations_agent._call_llm", return_value=mock_resp):
        result = interpret_monitoring("v1", EDGE_METRICS)
    assert result["status"] == "degraded"
    assert result["confidence"] == 0.85


def test_monitoring_llm_failure_falls_back():
    from agents.operations_agent import interpret_monitoring
    with patch("agents.operations_agent._call_llm", side_effect=Exception("API 오류")):
        result = interpret_monitoring("v1", GOOD_METRICS)
    # rule-based fallback → healthy
    assert result["status"] == "healthy"


def test_parse_monitoring_invalid_status_defaults_to_critical():
    from agents.operations_agent import _parse_monitoring_response
    raw = json.dumps({"status": "unknown_status", "reason": "?", "confidence": 0.5})
    result = _parse_monitoring_response(raw)
    assert result["status"] == "critical"


def test_parse_monitoring_invalid_json():
    from agents.operations_agent import _parse_monitoring_response
    result = _parse_monitoring_response("{{bad")
    assert result["status"] == "critical"
    assert result["confidence"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# decide_action — 액션 결정
# ══════════════════════════════════════════════════════════════════════════════

def test_rule_based_action_retrain_on_psi_drift(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1"}))

    from agents.operations_agent import decide_action
    metrics = {"roc_auc": 0.80, "error_rate": 0.20, "psi_mean": 0.25, "ks_pvalue_min": 0.02}
    result = decide_action("v1", [], {}, {}, metrics, use_agent=False)
    assert result["action"] == "retrain"


def test_rule_based_action_rollback_on_roc_drop_no_psi(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1"}))

    from agents.operations_agent import decide_action
    metrics = {"roc_auc": 0.60, "error_rate": 0.40, "psi_mean": 0.05, "ks_pvalue_min": 0.30}
    result = decide_action("v1", [], {}, {}, metrics, use_agent=False)
    assert result["action"] == "rollback"


def test_rule_based_action_promote_challenger_on_severe_drift(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({
        "active_model_version": "v1",
        "challenger_model_version": "v2",
        "challenger_traffic_weight": 0.1,
    }))

    from agents.operations_agent import decide_action
    metrics = {"roc_auc": 0.78, "error_rate": 0.22, "psi_mean": 0.40, "ks_pvalue_min": 0.01}
    result = decide_action("v1", [], {}, {}, metrics, use_agent=False)
    assert result["action"] == "promote_challenger"


def test_action_llm_success(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1"}))

    mock_resp = json.dumps({
        "action": "retrain",
        "affected_features": ["age", "DebtRatio"],
        "cause": "고령층 분포 변화",
        "recommendation": "최근 데이터로 재학습",
        "confidence": 0.9,
    })
    with patch("agents.operations_agent._call_llm", return_value=mock_resp):
        from agents.operations_agent import decide_action
        result = decide_action("v1", ["age", "DebtRatio"], {}, {}, BAD_METRICS)
    assert result["action"] == "retrain"
    assert "age" in result["affected_features"]
    assert result["confidence"] == 0.9


def test_action_llm_failure_falls_back_to_retrain(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1"}))

    with patch("agents.operations_agent._call_llm", side_effect=Exception("API 오류")):
        from agents.operations_agent import decide_action
        result = decide_action("v1", [], {}, {}, BAD_METRICS)
    assert result["action"] in ("retrain", "rollback")  # rule-based fallback


def test_parse_action_invalid_action_defaults_to_retrain():
    from agents.operations_agent import _parse_action_response
    raw = json.dumps({"action": "explode", "cause": "?", "confidence": 0.5})
    result = _parse_action_response(raw)
    assert result["action"] == "retrain"


def test_parse_action_invalid_json():
    from agents.operations_agent import _parse_action_response
    result = _parse_action_response("{{bad")
    assert result["action"] == "retrain"
    assert result["confidence"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# analyze_drift — drift_agent 호환 API
# ══════════════════════════════════════════════════════════════════════════════

def test_analyze_drift_backward_compatible_keys(tmp_path, monkeypatch):
    import agents.operations_agent as oa
    monkeypatch.setattr(oa, "FLAGS_PATH", tmp_path / "flags.json")
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1"}))

    mock_resp = json.dumps({
        "action": "retrain",
        "affected_features": ["age"],
        "cause": "분포 변화",
        "recommendation": "재학습 필요",
        "confidence": 0.9,
    })
    with patch("agents.operations_agent._call_llm", return_value=mock_resp):
        from agents.operations_agent import analyze_drift
        result = analyze_drift("v1", ["age"], {"age": 0.3}, {"age": 0.01}, BAD_METRICS)

    # drift_agent와 동일한 키 구조
    assert "affected_features" in result
    assert "cause" in result
    assert "recommendation" in result
    # action 키는 없어야 함 (backward compat)
    assert "action" not in result
