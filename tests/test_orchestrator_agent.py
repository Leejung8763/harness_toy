"""agents/orchestrator_agent.py — 파이프라인 오케스트레이터 단위 테스트"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.orchestrator_agent import (
    RunPlan,
    _gather_context,
    _parse_response,
    _rule_based_plan,
    plan_pipeline,
)

# ── RunPlan ───────────────────────────────────────────────────────────────────

def test_run_plan_should_run():
    plan = RunPlan(stages=["train", "evaluate"], skip_stages=["data_eng"], reason="test")
    assert plan.should_run("train") is True
    assert plan.should_run("data_eng") is False


def test_run_plan_full_pipeline():
    plan = RunPlan(stages=["data_eng", "train", "evaluate", "deploy", "monitor"],
                   skip_stages=[], reason="full")
    assert plan.should_run("data_eng") is True
    assert plan.should_run("monitor") is True


# ── _parse_response ───────────────────────────────────────────────────────────

def test_parse_valid_skip_data_eng():
    raw = '{"skip_stages": ["data_eng"], "reason": "Feature Store 신선", "confidence": 0.9}'
    plan = _parse_response(raw, "drift")
    assert "data_eng" not in plan.stages
    assert "data_eng" in plan.skip_stages
    assert plan.confidence == 0.9


def test_parse_empty_skip():
    raw = '{"skip_stages": [], "reason": "전체 실행", "confidence": 1.0}'
    plan = _parse_response(raw, "ci_cd")
    assert plan.skip_stages == []
    assert len(plan.stages) == 5


def test_parse_invalid_stage_name_ignored():
    """존재하지 않는 스테이지명은 무시"""
    raw = '{"skip_stages": ["nonexistent", "data_eng"], "reason": "test", "confidence": 0.8}'
    plan = _parse_response(raw, "drift")
    assert "nonexistent" not in plan.skip_stages
    assert "data_eng" in plan.skip_stages


def test_parse_invalid_json_returns_full_plan():
    plan = _parse_response("not json", "drift")
    assert plan.skip_stages == []
    assert len(plan.stages) == 5
    assert plan.confidence == 0.0


# ── _rule_based_plan ──────────────────────────────────────────────────────────

def test_rule_based_ci_cd_always_full():
    context = {"feature_store": {"exists": True, "age_hours": 1.0}, "registry": {}, "recent_runs": []}
    plan = _rule_based_plan("ci_cd", context)
    assert plan.skip_stages == []
    assert len(plan.stages) == 5


def test_rule_based_drift_fresh_store_skips_data_eng():
    context = {"feature_store": {"exists": True, "age_hours": 5.0}, "registry": {}, "recent_runs": []}
    plan = _rule_based_plan("drift", context)
    assert "data_eng" in plan.skip_stages
    assert "train" in plan.stages


def test_rule_based_drift_stale_store_full_run():
    context = {"feature_store": {"exists": True, "age_hours": 30.0}, "registry": {}, "recent_runs": []}
    plan = _rule_based_plan("drift", context)
    assert plan.skip_stages == []


def test_rule_based_drift_no_store_full_run():
    context = {"feature_store": {"exists": False, "age_hours": None}, "registry": {}, "recent_runs": []}
    plan = _rule_based_plan("drift", context)
    assert plan.skip_stages == []


def test_rule_based_manual_fresh_store_skips():
    context = {"feature_store": {"exists": True, "age_hours": 10.0}, "registry": {}, "recent_runs": []}
    plan = _rule_based_plan("manual", context)
    assert "data_eng" in plan.skip_stages


# ── plan_pipeline (LLM mocking) ───────────────────────────────────────────────

def test_plan_pipeline_use_agent_false():
    """use_agent=False면 rule-based 계획 반환"""
    with patch("agents.orchestrator_agent._gather_context") as mock_ctx:
        mock_ctx.return_value = {
            "feature_store": {"exists": True, "age_hours": 2.0},
            "registry": {},
            "recent_runs": [],
        }
        plan = plan_pipeline("drift", dataset_id=44089, use_agent=False)

    assert isinstance(plan, RunPlan)
    assert "data_eng" in plan.skip_stages


def test_plan_pipeline_llm_success():
    """LLM 성공 시 파싱된 계획 반환"""
    llm_response = '{"skip_stages": ["data_eng"], "reason": "신선한 Feature Store", "confidence": 0.95}'
    with patch("agents.orchestrator_agent._call_llm", return_value=llm_response), \
         patch("agents.orchestrator_agent._gather_context") as mock_ctx:
        mock_ctx.return_value = {
            "feature_store": {"exists": True, "age_hours": 3.0},
            "registry": {},
            "recent_runs": [],
        }
        plan = plan_pipeline("drift", dataset_id=44089, use_agent=True)

    assert "data_eng" in plan.skip_stages
    assert plan.confidence == 0.95


def test_plan_pipeline_llm_failure_fallback():
    """LLM 실패 시 rule-based fallback"""
    with patch("agents.orchestrator_agent._call_llm", side_effect=Exception("timeout")), \
         patch("agents.orchestrator_agent._gather_context") as mock_ctx:
        mock_ctx.return_value = {
            "feature_store": {"exists": True, "age_hours": 2.0},
            "registry": {},
            "recent_runs": [],
        }
        plan = plan_pipeline("drift", dataset_id=44089, use_agent=True)

    assert isinstance(plan, RunPlan)
    # fallback rule-based: fresh store → skip data_eng
    assert "data_eng" in plan.skip_stages
