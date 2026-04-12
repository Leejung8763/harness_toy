"""agents/pipeline_agent.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
# plan_pipeline — 스테이지 계획 (orchestrator 역할)
# ══════════════════════════════════════════════════════════════════════════════

def test_plan_pipeline_use_agent_false_ci_cd_full_run():
    from agents.pipeline_agent import plan_pipeline
    plan = plan_pipeline("ci_cd", use_agent=False)
    assert plan.skip_stages == []
    assert set(plan.stages) == {"data_eng", "train", "evaluate", "deploy", "monitor"}


def test_plan_pipeline_use_agent_false_drift_fresh_store_skips_data_eng(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "FEATURE_STORE_DIR", tmp_path / "features")
    meta_dir = tmp_path / "features" / "44089" / "v1"
    meta_dir.mkdir(parents=True)
    from datetime import datetime, timezone
    (meta_dir / "meta.json").write_text(json.dumps({
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_train": 10000,
    }))
    plan = pa.plan_pipeline("drift", dataset_id=44089, use_agent=False)
    assert "data_eng" in plan.skip_stages
    assert "data_eng" not in plan.stages


def test_plan_pipeline_use_agent_false_no_store_full_run(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "FEATURE_STORE_DIR", tmp_path / "empty")
    plan = pa.plan_pipeline("drift", dataset_id=44089, use_agent=False)
    assert plan.skip_stages == []


def test_plan_pipeline_llm_success():
    from agents.pipeline_agent import plan_pipeline
    mock_response = json.dumps({
        "skip_stages": ["data_eng"],
        "reason": "Feature Store 신선",
        "confidence": 0.95,
    })
    with patch("agents.pipeline_agent._call_llm", return_value=mock_response):
        plan = plan_pipeline("drift")
    assert "data_eng" in plan.skip_stages
    assert plan.confidence == 0.95


def test_plan_pipeline_llm_failure_falls_back(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "FEATURE_STORE_DIR", tmp_path / "empty")
    with patch("agents.pipeline_agent._call_llm", side_effect=Exception("API 오류")):
        plan = pa.plan_pipeline("ci_cd")
    # ci_cd → rule-based fallback → 전체 실행
    assert plan.skip_stages == []


def test_parse_stage_plan_filters_invalid_stages():
    from agents.pipeline_agent import _parse_stage_plan
    raw = json.dumps({
        "skip_stages": ["data_eng", "invalid_stage"],
        "reason": "테스트",
        "confidence": 0.8,
    })
    plan = _parse_stage_plan(raw, "drift")
    assert "invalid_stage" not in plan.skip_stages
    assert "data_eng" in plan.skip_stages


def test_parse_stage_plan_invalid_json_full_fallback():
    from agents.pipeline_agent import _parse_stage_plan, ALL_STAGES
    plan = _parse_stage_plan("{{invalid}}", "manual")
    assert set(plan.stages) == set(ALL_STAGES)
    assert plan.confidence == 0.0


def test_run_plan_should_run():
    from agents.pipeline_agent import RunPlan
    plan = RunPlan(stages=["train", "deploy"], skip_stages=["data_eng"], reason="test")
    assert not plan.should_run("data_eng")
    assert plan.should_run("train")


# ══════════════════════════════════════════════════════════════════════════════
# judge_deployment — 배포 판단 (deploy_agent 역할)
# ══════════════════════════════════════════════════════════════════════════════

def test_judge_deployment_no_context_returns_safe_default(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    (tmp_path / "registry.json").write_text(json.dumps({"models": []}))
    from agents.pipeline_agent import judge_deployment
    result = judge_deployment("v99")
    assert result["decision"] == "start_ab_test"
    assert result["confidence"] == 0.0


def test_judge_deployment_rule_based_first_deploy(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.85}},
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    from agents.pipeline_agent import judge_deployment
    result = judge_deployment("v1", use_agent=False)
    assert result["decision"] == "deploy"  # 첫 배포, baseline 충족


def test_judge_deployment_rule_based_better_model_deploys(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "metrics": {"roc_auc": 0.85}},
        {"version": "v2", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.860}},
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    from agents.pipeline_agent import judge_deployment
    result = judge_deployment("v2", use_agent=False)
    assert result["decision"] == "deploy"


def test_judge_deployment_rule_based_similar_model_ab_test(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "deployed", "metrics": {"roc_auc": 0.850}},
        {"version": "v2", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.8505}},  # 차이 0.0005 < 0.001
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    from agents.pipeline_agent import judge_deployment
    result = judge_deployment("v2", use_agent=False)
    assert result["decision"] == "start_ab_test"


def test_judge_deployment_rule_based_below_baseline_rejects(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.65}},  # baseline 미달
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    from agents.pipeline_agent import judge_deployment
    result = judge_deployment("v1", use_agent=False)
    assert result["decision"] == "reject"


def test_judge_deployment_llm_success(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.85}},
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    mock_resp = json.dumps({"decision": "deploy", "reason": "충분히 높음", "confidence": 0.9})
    with patch("agents.pipeline_agent._call_llm", return_value=mock_resp):
        result = pa.judge_deployment("v1")
    assert result["decision"] == "deploy"
    assert result["confidence"] == 0.9


def test_judge_deployment_llm_failure_uses_rule_based(tmp_path, monkeypatch):
    import agents.pipeline_agent as pa
    monkeypatch.setattr(pa, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(pa, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(pa, "METADATA_PATH", tmp_path / "runs.json")

    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"version": "v1", "status": "evaluated_pass",
         "model_name": "hgb", "dataset_name": "credit",
         "metrics": {"roc_auc": 0.85}},
    ]}))
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"evaluate": {"roc_auc_baseline": 0.70}, "monitor": {"roc_auc_min": 0.75}}
    ))
    (tmp_path / "runs.json").write_text(json.dumps([]))

    with patch("agents.pipeline_agent._call_llm", side_effect=Exception("API 오류")):
        result = pa.judge_deployment("v1")
    # rule-based fallback: 첫 배포 + baseline 충족 → deploy
    assert result["decision"] == "deploy"


def test_parse_deploy_response_invalid_decision_defaults_to_ab_test():
    from agents.pipeline_agent import _parse_deploy_response
    raw = json.dumps({"decision": "unknown_value", "reason": "?", "confidence": 0.5})
    result = _parse_deploy_response(raw)
    assert result["decision"] == "start_ab_test"


def test_parse_deploy_response_invalid_json_defaults_to_ab_test():
    from agents.pipeline_agent import _parse_deploy_response
    result = _parse_deploy_response("{{bad json")
    assert result["decision"] == "start_ab_test"
    assert result["confidence"] == 0.0
