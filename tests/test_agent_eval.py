"""agents/eval/ — decision_logger + retrospective 단위 테스트"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ── decision_logger ───────────────────────────────────────────────────────────

def test_log_deploy_decision_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.eval.decision_logger.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("agents.eval.decision_logger.DEPLOY_LOG", tmp_path / "deploy_decisions.jsonl")

    from agents.eval.decision_logger import log_deploy_decision
    decision_id = log_deploy_decision(
        model_version="v2",
        decision="hold",
        confidence=0.9,
        reason="성능 동일",
        context={"candidate_roc_auc": 0.88, "champion_roc_auc": 0.88},
    )

    assert decision_id.startswith("deploy-v2-")
    log_file = tmp_path / "deploy_decisions.jsonl"
    assert log_file.exists()
    records = [json.loads(l) for l in log_file.read_text().splitlines() if l]
    assert len(records) == 1
    assert records[0]["decision"] == "hold"
    assert records[0]["outcome"] is None


def test_log_deploy_decision_appends(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.eval.decision_logger.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("agents.eval.decision_logger.DEPLOY_LOG", tmp_path / "deploy_decisions.jsonl")

    from agents.eval.decision_logger import log_deploy_decision
    log_deploy_decision("v1", "deploy", 0.95, "성능 우수", {})
    log_deploy_decision("v2", "hold", 0.90, "성능 동일", {})

    records = [json.loads(l) for l in (tmp_path / "deploy_decisions.jsonl").read_text().splitlines() if l]
    assert len(records) == 2


def test_update_deploy_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.eval.decision_logger.RESULTS_DIR", tmp_path)
    log_file = tmp_path / "deploy_decisions.jsonl"
    monkeypatch.setattr("agents.eval.decision_logger.DEPLOY_LOG", log_file)

    from agents.eval.decision_logger import log_deploy_decision, update_deploy_outcome, load_deploy_decisions
    log_deploy_decision("v1", "deploy", 0.95, "성능 우수", {})

    result = update_deploy_outcome("v1", "survived", monitor_result={"roc_auc": 0.88})
    assert result is True

    records = load_deploy_decisions()
    assert records[0]["outcome"] == "survived"
    assert records[0]["outcome_updated_at"] is not None
    assert records[0]["monitor_result"]["roc_auc"] == 0.88


def test_update_deploy_outcome_only_nulls(tmp_path, monkeypatch):
    """이미 outcome이 있는 레코드는 업데이트 안 함"""
    monkeypatch.setattr("agents.eval.decision_logger.RESULTS_DIR", tmp_path)
    log_file = tmp_path / "deploy_decisions.jsonl"
    monkeypatch.setattr("agents.eval.decision_logger.DEPLOY_LOG", log_file)

    from agents.eval.decision_logger import log_deploy_decision, update_deploy_outcome, load_deploy_decisions
    log_deploy_decision("v1", "deploy", 0.95, "성능 우수", {})
    update_deploy_outcome("v1", "survived")
    update_deploy_outcome("v1", "drift_detected")  # 이미 채워진 것은 무시

    records = load_deploy_decisions()
    assert records[0]["outcome"] == "survived"  # 처음 값 유지


def test_log_drift_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.eval.decision_logger.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("agents.eval.decision_logger.DRIFT_LOG", tmp_path / "drift_analyses.jsonl")

    from agents.eval.decision_logger import log_drift_analysis
    analysis_id = log_drift_analysis(
        model_version="v2",
        affected_features=["age", "DebtRatio"],
        cause="연령 분포 변화",
        recommendation="재학습 필요",
        metrics={"roc_auc": 0.58, "psi_mean": 1.46},
    )

    assert analysis_id.startswith("drift-v2-")
    records = [json.loads(l) for l in (tmp_path / "drift_analyses.jsonl").read_text().splitlines() if l]
    assert records[0]["affected_features"] == ["age", "DebtRatio"]


def test_load_returns_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.eval.decision_logger.DEPLOY_LOG", tmp_path / "nonexistent.jsonl")
    monkeypatch.setattr("agents.eval.decision_logger.DRIFT_LOG", tmp_path / "nonexistent2.jsonl")

    from agents.eval.decision_logger import load_deploy_decisions, load_drift_analyses
    assert load_deploy_decisions() == []
    assert load_drift_analyses() == []


# ── retrospective ─────────────────────────────────────────────────────────────

def test_analyze_deploy_empty():
    from agents.eval.retrospective import analyze_deploy_decisions
    stats = analyze_deploy_decisions([])
    assert stats["total_decisions"] == 0
    assert stats["deploy_accuracy"] is None


def test_analyze_deploy_accuracy():
    from agents.eval.retrospective import analyze_deploy_decisions
    records = [
        {"decision": "deploy", "confidence": 0.95, "outcome": "survived"},
        {"decision": "deploy", "confidence": 0.80, "outcome": "drift_detected"},
        {"decision": "hold",   "confidence": 0.90, "outcome": None},
    ]
    stats = analyze_deploy_decisions(records)
    assert stats["total_decisions"] == 3
    assert stats["deploy_accuracy"] == 0.5  # 1/2 deployed survived
    assert stats["pending_outcome"] == 1
    assert stats["held_count"] == 1


def test_analyze_deploy_no_outcomes():
    from agents.eval.retrospective import analyze_deploy_decisions
    records = [
        {"decision": "deploy", "confidence": 0.9, "outcome": None},
        {"decision": "hold",   "confidence": 0.8, "outcome": None},
    ]
    stats = analyze_deploy_decisions(records)
    assert stats["deploy_accuracy"] is None  # 결과 없음
    assert stats["pending_outcome"] == 2


def test_analyze_drift_top_features():
    from agents.eval.retrospective import analyze_drift_analyses
    records = [
        {"affected_features": ["age", "DebtRatio"], "retrain_triggered": True},
        {"affected_features": ["age", "MonthlyIncome"], "retrain_triggered": True},
    ]
    stats = analyze_drift_analyses(records)
    assert stats["total_analyses"] == 2
    assert stats["retrain_triggered_count"] == 2
    top_features = [f for f, _ in stats["top_affected_features"]]
    assert "age" in top_features  # age가 2번으로 최다
