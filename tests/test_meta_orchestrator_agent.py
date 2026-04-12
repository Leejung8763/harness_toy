"""agents/meta_orchestrator_agent.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── _should_call_ds_dev ───────────────────────────────────────────────────────

def test_should_call_ds_dev_first_deploy(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": []}))
    assert mo._should_call_ds_dev("ci_cd", 44089) is True


def test_should_call_ds_dev_new_dataset(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
    ]}))
    # 다른 dataset_id → 호출해야 함
    assert mo._should_call_ds_dev("ci_cd", 99999) is True


def test_should_not_call_ds_dev_same_dataset(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
    ]}))
    assert mo._should_call_ds_dev("ci_cd", 44089) is False


def test_should_call_ds_dev_new_experiment_trigger(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
    ]}))
    # new_experiment 트리거는 항상 호출
    assert mo._should_call_ds_dev("new_experiment", 44089) is True


# ── orchestrate — DS Development 레이어 스킵 ──────────────────────────────────

def test_orchestrate_skips_ds_dev_when_deployed_same_dataset(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
    ]}))

    mock_plan = MagicMock()
    mock_plan.skip_stages = []
    mock_plan.stages = ["data_eng", "train", "evaluate", "deploy", "monitor"]
    mock_plan.reason = "test"
    mock_plan.should_run = lambda s: True

    with patch("agents.meta_orchestrator_agent.plan_pipeline", return_value=mock_plan):
        result = mo.orchestrate("ci_cd", dataset_id=44089, use_agent=False)

    assert "ds_development" not in result.layers_called
    assert "pipeline" in result.layers_called


def test_orchestrate_calls_ds_dev_on_first_deploy(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": []}))

    mock_run_plan = MagicMock()
    mock_run_plan.skip_stages = []
    mock_run_plan.stages = ["train", "evaluate", "deploy", "monitor"]
    mock_run_plan.reason = "test"

    mock_exp_plan = {
        "target_column": "SeriousDlqin2yrs",
        "models_to_try": ["hist_gradient_boosting"],
        "primary_metric": "roc_auc",
        "feature_exclusions": [],
        "reason": "mock",
        "confidence": 0.9,
    }

    with patch("agents.meta_orchestrator_agent.plan_pipeline", return_value=mock_run_plan), \
         patch("agents.meta_orchestrator_agent.plan_experiment", return_value=mock_exp_plan):
        result = mo.orchestrate("ci_cd", dataset_id=44089, use_agent=True)

    assert "ds_development" in result.layers_called
    assert result.experiment_plan["target_column"] == "SeriousDlqin2yrs"


# ── orchestrate — Pipeline 레이어 fallback ────────────────────────────────────

def test_orchestrate_pipeline_failure_falls_back(tmp_path, monkeypatch):
    import agents.meta_orchestrator_agent as mo
    monkeypatch.setattr(mo, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
    ]}))

    with patch("agents.meta_orchestrator_agent.plan_pipeline", side_effect=Exception("API 오류")):
        result = mo.orchestrate("ci_cd", dataset_id=44089)

    # fallback: 전체 실행
    assert result.run_plan.skip_stages == []
    assert "pipeline" not in result.layers_called


# ── OrchestratorResult ────────────────────────────────────────────────────────

def test_orchestrator_result_should_run():
    from agents.meta_orchestrator_agent import OrchestratorResult
    from agents.pipeline_agent import RunPlan
    run_plan = RunPlan(
        stages=["train", "deploy"],
        skip_stages=["data_eng"],
        reason="test",
    )
    result = OrchestratorResult(run_plan=run_plan, experiment_plan={})
    assert not result.should_run("data_eng")
    assert result.should_run("train")


def test_orchestrator_result_summary():
    from agents.meta_orchestrator_agent import OrchestratorResult
    from agents.pipeline_agent import RunPlan
    run_plan = RunPlan(
        stages=["train", "deploy"],
        skip_stages=["data_eng"],
        reason="신선",
    )
    result = OrchestratorResult(
        run_plan=run_plan,
        experiment_plan={},
        layers_called=["ds_development", "pipeline"],
    )
    summary = result.summary()
    assert "data_eng" in summary
    assert "pipeline" in summary


# ── pipeline_server 연동 확인 ─────────────────────────────────────────────────

def test_pipeline_server_uses_meta_orchestrator():
    """pipeline_server.py가 meta_orchestrator_agent를 import하는지 확인"""
    server_code = Path("api/pipeline_server.py").read_text()
    assert "meta_orchestrator_agent" in server_code
    assert "orchestrate" in server_code
