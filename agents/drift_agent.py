"""
agents/drift_agent.py — 드리프트 원인 분석 LLM 에이전트

역할:
  - monitor.py에서 드리프트 감지 시 호출
  - PSI/KS-test 결과와 피처별 분포 변화를 분석
  - 드리프트 원인을 자연어로 설명하고 재학습 방향 제안

하네스 역할:
  - 에이전트 분석 실패해도 롤백/트리거는 이미 완료된 상태
  - 분석 결과는 참고용 — 실제 롤백/재학습 결정은 rule-based로 처리됨

참고 문서: agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

THRESHOLDS_PATH = Path("config/thresholds.json")
METADATA_PATH = Path("ml_metadata/runs.json")


def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


def analyze_drift(
    model_version: str,
    feature_names: list[str],
    psi_per_feature: dict[str, float],
    ks_pvalue_per_feature: dict[str, float],
    metrics: dict,
) -> dict:
    """
    드리프트 원인을 LLM으로 분석합니다.

    Args:
        model_version: 드리프트가 감지된 모델 버전
        feature_names: 피처 이름 목록
        psi_per_feature: 피처별 PSI 값
        ks_pvalue_per_feature: 피처별 KS-test p-value
        metrics: 해당 라운드 성능 지표

    Returns:
        {
            "affected_features": ["feature1", ...],  # 영향 피처 목록
            "cause": "원인 설명 (한국어)",
            "recommendation": "재학습 방향 제안 (한국어)",
        }
    """
    prompt = _build_prompt(model_version, feature_names, psi_per_feature, ks_pvalue_per_feature, metrics)
    response = _call_llm(prompt)
    result = _parse_response(response)
    _print_analysis(result)
    _log_to_metadata(model_version, metrics, result)
    return result


def _build_prompt(
    version: str,
    feature_names: list[str],
    psi_per_feature: dict[str, float],
    ks_pvalue_per_feature: dict[str, float],
    metrics: dict,
) -> str:
    thresholds = {}
    if THRESHOLDS_PATH.exists():
        thresholds = json.loads(THRESHOLDS_PATH.read_text()).get("monitor", {})

    psi_threshold = thresholds.get("psi_threshold", 0.20)
    ks_threshold = thresholds.get("ks_pvalue_min", 0.05)

    # 피처별 PSI 내림차순 정렬
    sorted_psi = sorted(psi_per_feature.items(), key=lambda x: x[1], reverse=True)
    psi_table = "\n".join(
        f"  - {name}: PSI={psi:.4f} {'🔴 위험' if psi > psi_threshold else '🟡 주의' if psi > 0.1 else '🟢 정상'}"
        f"  KS-p={ks_pvalue_per_feature.get(name, 1.0):.4f}"
        for name, psi in sorted_psi
    )

    return f"""당신은 MLOps 드리프트 분석 에이전트입니다.
아래 정보를 바탕으로 데이터 드리프트의 원인을 분석하세요.

## 드리프트 감지 정보
- 모델 버전: {version}
- ROC-AUC: {metrics.get('roc_auc', 'N/A')} (임계값: {thresholds.get('roc_auc_min', 0.75)})
- Error Rate: {metrics.get('error_rate', 'N/A')} (임계값: {thresholds.get('error_rate_max', 0.25)})
- PSI mean: {metrics.get('psi_mean', 'N/A')} (임계값: {psi_threshold})

## 피처별 분포 변화 (PSI 높은 순)
{psi_table}

## PSI 해석 기준
- PSI < 0.10: 변화 없음
- PSI 0.10~0.20: 중간 변화
- PSI > 0.20: 유의미한 드리프트

## 응답 형식 (JSON만 반환)
{{
  "affected_features": ["가장 영향 큰 피처명 최대 3개"],
  "cause": "드리프트 원인 추정 (한국어 1~2문장, 데이터 특성 기반)",
  "recommendation": "재학습 시 고려할 사항 (한국어 1문장)"
}}"""


def _call_llm(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_response(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return {
            "affected_features": data.get("affected_features", []),
            "cause": data.get("cause", "분석 불가"),
            "recommendation": data.get("recommendation", ""),
        }
    except Exception:
        return {
            "affected_features": [],
            "cause": f"응답 파싱 실패: {raw[:100]}",
            "recommendation": "",
        }


def _print_analysis(result: dict) -> None:
    print(f"\n  [drift_agent] 📊 드리프트 분석 완료")
    if result["affected_features"]:
        print(f"  영향 피처: {', '.join(result['affected_features'])}")
    print(f"  원인: {result['cause']}")
    if result["recommendation"]:
        print(f"  제안: {result['recommendation']}")


def _log_to_metadata(version: str, metrics: dict, analysis: dict) -> None:
    try:
        from ml_metadata import store as metadata_store
        metadata_store.log_monitoring({
            "model_version": version,
            "event": "drift_analysis",
            **metrics,
            "drift_analysis": analysis,
        })
    except Exception:
        pass
