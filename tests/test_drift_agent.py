"""agents/drift_agent.py — 드리프트 분석 에이전트 단위 테스트"""

from unittest.mock import MagicMock, patch

import pytest

from agents.drift_agent import _build_prompt, _parse_response, analyze_drift

FEATURE_NAMES = ["age", "RevolvingUtilization", "DebtRatio"]

PSI_PER_FEATURE = {
    "age": 0.35,
    "RevolvingUtilization": 0.12,
    "DebtRatio": 0.03,
}

KS_PVALUE_PER_FEATURE = {
    "age": 0.001,
    "RevolvingUtilization": 0.04,
    "DebtRatio": 0.60,
}

METRICS = {
    "roc_auc": 0.58,
    "error_rate": 0.44,
    "psi_mean": 0.17,
    "ks_pvalue_min": 0.001,
}


# ── _build_prompt ─────────────────────────────────────────────────────────────

def test_build_prompt_contains_version():
    """프롬프트에 모델 버전 포함"""
    prompt = _build_prompt("v3", FEATURE_NAMES, PSI_PER_FEATURE, KS_PVALUE_PER_FEATURE, METRICS)
    assert "v3" in prompt


def test_build_prompt_contains_features():
    """프롬프트에 피처명과 PSI 값 포함"""
    prompt = _build_prompt("v1", FEATURE_NAMES, PSI_PER_FEATURE, KS_PVALUE_PER_FEATURE, METRICS)
    assert "age" in prompt
    assert "0.3500" in prompt


def test_build_prompt_drift_indicator():
    """PSI > 0.2 피처는 🔴 위험으로 표시"""
    prompt = _build_prompt("v1", FEATURE_NAMES, PSI_PER_FEATURE, KS_PVALUE_PER_FEATURE, METRICS)
    assert "🔴 위험" in prompt


def test_build_prompt_contains_roc_auc():
    """프롬프트에 ROC-AUC 지표 포함"""
    prompt = _build_prompt("v1", FEATURE_NAMES, PSI_PER_FEATURE, KS_PVALUE_PER_FEATURE, METRICS)
    assert "0.58" in prompt


def test_build_prompt_sorted_by_psi():
    """PSI 높은 피처가 먼저 나열됨"""
    prompt = _build_prompt("v1", FEATURE_NAMES, PSI_PER_FEATURE, KS_PVALUE_PER_FEATURE, METRICS)
    idx_age = prompt.index("age")
    idx_debt = prompt.index("DebtRatio")
    assert idx_age < idx_debt  # age(PSI=0.35) > DebtRatio(PSI=0.03)


# ── _parse_response ───────────────────────────────────────────────────────────

def test_parse_valid_json():
    """정상 JSON 응답 파싱"""
    raw = '{"affected_features": ["age"], "cause": "연령 분포 변화", "recommendation": "최근 데이터 재학습"}'
    result = _parse_response(raw)
    assert result["affected_features"] == ["age"]
    assert result["cause"] == "연령 분포 변화"
    assert result["recommendation"] == "최근 데이터 재학습"


def test_parse_missing_fields():
    """필드 누락 시 기본값 반환"""
    raw = '{"affected_features": ["age"]}'
    result = _parse_response(raw)
    assert result["cause"] == "분석 불가"
    assert result["recommendation"] == ""


def test_parse_invalid_json():
    """파싱 불가한 문자열은 cause에 오류 메시지 반환"""
    result = _parse_response("not json at all")
    assert result["affected_features"] == []
    assert "파싱 실패" in result["cause"]


def test_parse_empty_affected_features():
    """빈 affected_features 허용"""
    raw = '{"affected_features": [], "cause": "미상", "recommendation": ""}'
    result = _parse_response(raw)
    assert result["affected_features"] == []


# ── analyze_drift (LLM 호출 mocking) ─────────────────────────────────────────

def test_analyze_drift_success(tmp_path, monkeypatch):
    """LLM 호출 성공 시 결과 반환 및 출력"""
    llm_response = '{"affected_features": ["age"], "cause": "연령 분포 급변", "recommendation": "최신 데이터 포함 재학습"}'

    with patch("agents.drift_agent._call_llm", return_value=llm_response), \
         patch("agents.drift_agent._log_to_metadata"):
        result = analyze_drift(
            model_version="v2",
            feature_names=FEATURE_NAMES,
            psi_per_feature=PSI_PER_FEATURE,
            ks_pvalue_per_feature=KS_PVALUE_PER_FEATURE,
            metrics=METRICS,
        )

    assert result["affected_features"] == ["age"]
    assert "연령" in result["cause"]


def test_analyze_drift_llm_failure_returns_fallback():
    """LLM 호출 실패 시 fallback 결과 반환 (예외 전파 없음)"""
    with patch("agents.drift_agent._call_llm", side_effect=Exception("API timeout")), \
         patch("agents.drift_agent._log_to_metadata"):
        result = analyze_drift(
            model_version="v2",
            feature_names=FEATURE_NAMES,
            psi_per_feature=PSI_PER_FEATURE,
            ks_pvalue_per_feature=KS_PVALUE_PER_FEATURE,
            metrics=METRICS,
        )

    assert isinstance(result, dict)
    assert "cause" in result


def test_analyze_drift_logs_to_metadata():
    """분석 결과가 ml_metadata에 기록됨"""
    llm_response = '{"affected_features": ["age"], "cause": "드리프트", "recommendation": "재학습"}'

    with patch("agents.drift_agent._call_llm", return_value=llm_response) as _, \
         patch("agents.drift_agent._log_to_metadata") as mock_log:
        analyze_drift(
            model_version="v2",
            feature_names=FEATURE_NAMES,
            psi_per_feature=PSI_PER_FEATURE,
            ks_pvalue_per_feature=KS_PVALUE_PER_FEATURE,
            metrics=METRICS,
        )

    mock_log.assert_called_once()
    # _log_to_metadata(version, metrics, analysis) 세 인자
    version_arg, metrics_arg, analysis_arg = mock_log.call_args[0]
    assert version_arg == "v2"
    assert "affected_features" in analysis_arg
    assert analysis_arg["cause"] == "드리프트"
