"""
agents/operations_agent.py — Operations 레이어 에이전트

역할:
  - drift_agent 역할을 흡수 + 모니터링 해석 + 액션 결정으로 확장
  - monitor.py의 드리프트 감지 결과를 받아 두 가지 판단 수행

두 가지 서브 판단:
  1. interpret_monitoring()  — 모니터링 지표 상태 해석 (healthy/degraded/critical)
  2. decide_action()         — 드리프트 원인 분석 + 액션 결정
       retrain           : 데이터 변화 → 재학습 필요
       rollback          : 급격한 성능 저하 → 즉시 이전 버전 복구
       hold              : 경미한 이상 → 추가 모니터링
       promote_challenger: A/B 테스트 중 챌린저가 더 안정적

하네스 역할:
  - interpret_monitoring fallback: 지표값 직접 비교 (rule-based)
  - decide_action fallback: "retrain" (보수적 안전 기본값)
  - 두 판단 모두 독립 fallback — 한 쪽 실패가 다른 쪽에 영향 없음
  - 실제 롤백/트리거 실행은 monitor.py가 담당 (에이전트는 판단만)

기존 에이전트 호환:
  - drift_agent.analyze_drift() 시그니처 유지 (backward compatible)

참고 문서: agents/AGENTS_SPEC.md, docs/QUALITY_SCORE.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np

THRESHOLDS_PATH = Path("config/thresholds.json")
METADATA_PATH = Path("ml_metadata/runs.json")
REGISTRY_PATH = Path("registry/model_registry.json")
FLAGS_PATH = Path("feature_flags/flags.json")

MonitoringStatus = Literal["healthy", "degraded", "critical"]
ActionDecision = Literal["retrain", "rollback", "hold", "promote_challenger"]


# ── 공유 헬퍼 ─────────────────────────────────────────────────────────────────

def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _load_thresholds() -> dict:
    return json.loads(THRESHOLDS_PATH.read_text()) if THRESHOLDS_PATH.exists() else {}


def _load_flags() -> dict:
    return json.loads(FLAGS_PATH.read_text()) if FLAGS_PATH.exists() else {}


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"models": []}


# ══════════════════════════════════════════════════════════════════════════════
# 1. interpret_monitoring — 모니터링 상태 해석
# ══════════════════════════════════════════════════════════════════════════════

def interpret_monitoring(
    model_version: str,
    metrics: dict,
    use_agent: bool = True,
) -> dict:
    """
    모니터링 지표를 해석해 시스템 상태를 반환합니다.

    Args:
        model_version: 모니터링 중인 모델 버전
        metrics:       monitor.py의 _evaluate_round() 반환값
        use_agent:     False면 rule-based fallback만 실행

    Returns:
        {
            "status": "healthy" | "degraded" | "critical",
            "reason": str,
            "confidence": float,
        }
    """
    thresholds = _load_thresholds().get("monitor", {})

    if not use_agent:
        return _rule_based_monitoring(metrics, thresholds)

    try:
        prompt = _build_monitoring_prompt(model_version, metrics, thresholds)
        raw = _call_llm(prompt)
        result = _parse_monitoring_response(raw)
        _print_monitoring(model_version, result)
        return result
    except Exception as e:
        print(f"  ⚠️  operations_agent(monitoring) 실패: {e} → rule-based 판단")
        return _rule_based_monitoring(metrics, thresholds)


def _build_monitoring_prompt(version: str, metrics: dict, thresholds: dict) -> str:
    roc_min = thresholds.get("roc_auc_min", 0.75)
    err_max = thresholds.get("error_rate_max", 0.25)
    psi_thr = thresholds.get("psi_threshold", 0.20)
    ks_thr = thresholds.get("ks_pvalue_min", 0.05)

    return f"""당신은 MLOps 운영 에이전트입니다. 모니터링 지표를 해석하세요.

## 모델: {version}
## 현재 지표
- ROC-AUC: {metrics.get('roc_auc', 'N/A')} (기준 ≥ {roc_min})
- Error Rate: {metrics.get('error_rate', 'N/A')} (기준 ≤ {err_max})
- PSI mean: {metrics.get('psi_mean', 'N/A')} (기준 ≤ {psi_thr})
- KS p-value min: {metrics.get('ks_pvalue_min', 'N/A')} (기준 ≥ {ks_thr})

## 상태 정의
- healthy:  모든 지표 기준 충족
- degraded: 1개 지표 경계 근처 (10% 이내 위반 또는 경미한 위반)
- critical: 1개 이상 지표 명확히 위반

## 응답 (JSON만)
{{
  "status": "healthy" | "degraded" | "critical",
  "reason": "한국어 1문장",
  "confidence": 0.0~1.0
}}"""


def _parse_monitoring_response(raw: str) -> dict:
    try:
        data = json.loads(raw)
        status = data.get("status", "critical")
        if status not in ("healthy", "degraded", "critical"):
            status = "critical"
        return {
            "status": status,
            "reason": data.get("reason", "판단 불가"),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception:
        return {"status": "critical", "reason": f"파싱 실패: {raw[:80]}", "confidence": 0.0}


def _rule_based_monitoring(metrics: dict, thresholds: dict) -> dict:
    """rule-based 상태 판단 — 임계값 직접 비교"""
    roc_min = thresholds.get("roc_auc_min", 0.75)
    err_max = thresholds.get("error_rate_max", 0.25)
    psi_thr = thresholds.get("psi_threshold", 0.20)
    ks_thr = thresholds.get("ks_pvalue_min", 0.05)

    roc = metrics.get("roc_auc", 0.0)
    err = metrics.get("error_rate", 1.0)
    psi = metrics.get("psi_mean", 1.0)
    ks = metrics.get("ks_pvalue_min", 0.0)

    violations = []
    if roc < roc_min:
        violations.append(f"ROC-AUC({roc:.4f})<{roc_min}")
    if err > err_max:
        violations.append(f"ErrorRate({err:.4f})>{err_max}")
    if psi > psi_thr:
        violations.append(f"PSI({psi:.4f})>{psi_thr}")
    if ks < ks_thr:
        violations.append(f"KS-p({ks:.4f})<{ks_thr}")

    if not violations:
        return {"status": "healthy", "reason": "모든 지표 정상", "confidence": 1.0}

    # 경계 근처(10% 이내) 위반이면 degraded, 명확한 위반이면 critical
    severe = any([
        roc < roc_min * 0.9,
        err > err_max * 1.1,
        psi > psi_thr * 1.5,
        ks < ks_thr * 0.5,
    ])
    status = "critical" if severe else "degraded"
    return {
        "status": status,
        "reason": f"위반 항목: {', '.join(violations)}",
        "confidence": 1.0,
    }


def _print_monitoring(version: str, result: dict) -> None:
    icon = {"healthy": "✅", "degraded": "⚠️", "critical": "🚨"}.get(result["status"], "❓")
    print(f"\n  [operations_agent] {icon} 모니터링 상태: {result['status'].upper()} "
          f"(신뢰도: {result['confidence']:.0%})")
    print(f"  판단: {result['reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. decide_action — 드리프트 원인 분석 + 액션 결정
# ══════════════════════════════════════════════════════════════════════════════

def decide_action(
    model_version: str,
    feature_names: list[str],
    psi_per_feature: dict[str, float],
    ks_pvalue_per_feature: dict[str, float],
    metrics: dict,
    use_agent: bool = True,
) -> dict:
    """
    드리프트 원인을 분석하고 취할 액션을 결정합니다.

    Args:
        model_version:         드리프트가 감지된 모델 버전
        feature_names:         피처 이름 목록
        psi_per_feature:       피처별 PSI 값
        ks_pvalue_per_feature: 피처별 KS-test p-value
        metrics:               해당 라운드 성능 지표
        use_agent:             False면 rule-based fallback만 실행

    Returns:
        {
            "action": "retrain"|"rollback"|"hold"|"promote_challenger",
            "affected_features": list[str],
            "cause": str,
            "recommendation": str,
            "confidence": float,
        }
    """
    flags = _load_flags()
    has_challenger = bool(flags.get("challenger_model_version"))

    if not use_agent:
        result = _rule_based_action(metrics, has_challenger)
    else:
        try:
            prompt = _build_action_prompt(
                model_version, feature_names,
                psi_per_feature, ks_pvalue_per_feature,
                metrics, has_challenger,
            )
            raw = _call_llm(prompt)
            result = _parse_action_response(raw)
        except Exception as e:
            print(f"  ⚠️  operations_agent(action) 실패: {e} → rule-based fallback")
            result = _rule_based_action(metrics, has_challenger)

    _print_action(model_version, result)
    _log_to_metadata(model_version, metrics, result)
    return result


def _build_action_prompt(
    version: str,
    feature_names: list[str],
    psi_per_feature: dict[str, float],
    ks_pvalue_per_feature: dict[str, float],
    metrics: dict,
    has_challenger: bool,
) -> str:
    thresholds = _load_thresholds().get("monitor", {})
    psi_thr = thresholds.get("psi_threshold", 0.20)

    sorted_psi = sorted(psi_per_feature.items(), key=lambda x: x[1], reverse=True)
    psi_table = "\n".join(
        f"  {name}: PSI={psi:.4f} "
        f"{'🔴' if psi > psi_thr else '🟡' if psi > 0.1 else '🟢'}  "
        f"KS-p={ks_pvalue_per_feature.get(name, 1.0):.4f}"
        for name, psi in sorted_psi[:5]
    )

    challenger_note = (
        "\n⚡ A/B 테스트 진행 중: 챌린저 모델이 더 안정적이면 promote_challenger 선택 가능"
        if has_challenger else ""
    )

    return f"""당신은 MLOps 운영 에이전트입니다. 드리프트 감지 후 취할 액션을 결정하세요.

## 모델: {version}
## 성능 지표
- ROC-AUC: {metrics.get('roc_auc', 'N/A')} / Error Rate: {metrics.get('error_rate', 'N/A')}
- PSI mean: {metrics.get('psi_mean', 'N/A')} / KS p-min: {metrics.get('ks_pvalue_min', 'N/A')}

## 피처별 분포 변화 (상위 5개)
{psi_table}
{challenger_note}

## 액션 정의
- retrain:            데이터 분포 변화 → 새 데이터로 재학습 (가장 일반적)
- rollback:           ROC-AUC 급락 + PSI 낮음 → 데이터 문제 아닌 모델 문제, 즉시 이전 버전 복구
- hold:               경미한 이상 → 다음 라운드까지 추가 관찰
- promote_challenger: A/B 중 챌린저가 더 안정적 → 챌린저로 즉시 교체{' (현재 해당 없음)' if not has_challenger else ''}

## 응답 (JSON만)
{{
  "action": "retrain" | "rollback" | "hold" | "promote_challenger",
  "affected_features": ["영향 큰 피처 최대 3개"],
  "cause": "드리프트 원인 추정 (한국어 1~2문장)",
  "recommendation": "재학습 시 고려 사항 (한국어 1문장)",
  "confidence": 0.0~1.0
}}"""


def _parse_action_response(raw: str) -> dict:
    try:
        data = json.loads(raw)
        action = data.get("action", "retrain")
        if action not in ("retrain", "rollback", "hold", "promote_challenger"):
            action = "retrain"
        return {
            "action": action,
            "affected_features": data.get("affected_features", []),
            "cause": data.get("cause", "분석 불가"),
            "recommendation": data.get("recommendation", ""),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception:
        return {
            "action": "retrain",
            "affected_features": [],
            "cause": f"파싱 실패: {raw[:80]}",
            "recommendation": "",
            "confidence": 0.0,
        }


def _rule_based_action(metrics: dict, has_challenger: bool) -> dict:
    """rule-based 액션 결정 — 지표 패턴으로 판단"""
    roc = metrics.get("roc_auc", 0.0)
    psi = metrics.get("psi_mean", 0.0)

    # PSI 낮은데 ROC-AUC 급락 → 데이터 문제 아닌 모델 문제 → rollback
    if roc < 0.65 and psi < 0.10:
        return {
            "action": "rollback",
            "affected_features": [],
            "cause": f"ROC-AUC 급락({roc:.4f})이나 PSI 낮음({psi:.4f}) — 모델 자체 문제",
            "recommendation": "이전 버전 복구 후 원인 분석",
            "confidence": 1.0,
        }
    # A/B 테스트 중 + 심각한 드리프트 → 챌린저 승격 고려
    if has_challenger and psi > 0.30:
        return {
            "action": "promote_challenger",
            "affected_features": [],
            "cause": f"심각한 드리프트(PSI={psi:.4f}) + 챌린저 대기 중",
            "recommendation": "챌린저 모델로 즉시 교체",
            "confidence": 0.8,
        }
    # 기본: 재학습
    return {
        "action": "retrain",
        "affected_features": [],
        "cause": f"데이터 분포 변화 감지 (PSI={psi:.4f})",
        "recommendation": "최신 데이터로 재학습",
        "confidence": 1.0,
    }


def _print_action(version: str, result: dict) -> None:
    icon = {
        "retrain": "🔄", "rollback": "↩️",
        "hold": "⏸️", "promote_challenger": "⚡",
    }.get(result["action"], "❓")
    print(f"\n  [operations_agent] {icon} 액션 결정: {result['action'].upper()} "
          f"(신뢰도: {result['confidence']:.0%})")
    if result["affected_features"]:
        print(f"  영향 피처: {', '.join(result['affected_features'])}")
    print(f"  원인: {result['cause']}")
    if result["recommendation"]:
        print(f"  제안: {result['recommendation']}")


def _log_to_metadata(version: str, metrics: dict, result: dict) -> None:
    try:
        from ml_metadata import store as metadata_store
        metadata_store.log_monitoring({
            "model_version": version,
            "event": "operations_action",
            **metrics,
            "action_decision": result,
        })
    except Exception:
        pass

    try:
        from agents.eval.decision_logger import log_drift_analysis
        log_drift_analysis(
            model_version=version,
            affected_features=result.get("affected_features", []),
            cause=result.get("cause", ""),
            recommendation=result.get("recommendation", ""),
            metrics=metrics,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 3. analyze_drift — drift_agent 호환 API (backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_drift(
    model_version: str,
    feature_names: list[str],
    psi_per_feature: dict[str, float],
    ks_pvalue_per_feature: dict[str, float],
    metrics: dict,
) -> dict:
    """
    drift_agent.analyze_drift()와 동일한 시그니처 — backward compatible.
    내부적으로 decide_action()을 호출합니다.

    Returns:
        {
            "affected_features": list[str],
            "cause": str,
            "recommendation": str,
        }
    """
    result = decide_action(
        model_version=model_version,
        feature_names=feature_names,
        psi_per_feature=psi_per_feature,
        ks_pvalue_per_feature=ks_pvalue_per_feature,
        metrics=metrics,
    )
    return {
        "affected_features": result["affected_features"],
        "cause": result["cause"],
        "recommendation": result["recommendation"],
    }
