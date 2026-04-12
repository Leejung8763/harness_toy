"""
agents/ds_development_agent.py — DS Development 레이어 에이전트

역할:
  - 데이터셋 스키마 + docs/DS_GUIDE.md를 LLM에 전달
  - 타겟 변수 / 모델 후보 / 평가 지표 / 피처 제외 목록 결정
  - 결과를 harness_pipeline.py 또는 meta_orchestrator_agent가 소비

하네스 역할:
  - LLM 실패 시 코드에 하드코딩된 DEFAULT_PLAN으로 즉시 fallback
  - DS_GUIDE.md는 LLM 컨텍스트 전용 (파싱 없음)

참고 문서: docs/DS_GUIDE.md, agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

DS_GUIDE_PATH = Path("docs/DS_GUIDE.md")
VALID_MODELS = {
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "random_forest",
    "logistic_regression",
}

# Harness fallback — LLM 실패 시 항상 이 값으로 동작
DEFAULT_PLAN: dict = {
    "target_column": "SeriousDlqin2yrs",
    "models_to_try": [
        "hist_gradient_boosting",
        "lightgbm",
        "xgboost",
        "random_forest",
        "logistic_regression",
    ],
    "primary_metric": "roc_auc",
    "feature_exclusions": [],
    "reason": "default plan (fallback)",
    "confidence": 1.0,
}


def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


# ── 컨텍스트 수집 ─────────────────────────────────────────────────────────────

def _build_schema_summary(columns: list[str], n_samples: int, n_features: int, imbalance_ratio: float) -> str:
    return json.dumps({
        "columns": columns,
        "n_samples": n_samples,
        "n_features": n_features,
        "imbalance_ratio": round(imbalance_ratio, 3),
    }, ensure_ascii=False)


def _load_guide() -> str:
    if DS_GUIDE_PATH.exists():
        return DS_GUIDE_PATH.read_text(encoding="utf-8")
    return "(DS_GUIDE.md 없음 — default 사용)"


# ── LLM 호출 ──────────────────────────────────────────────────────────────────

def _build_prompt(schema_summary: str, guide: str) -> str:
    return f"""당신은 MLOps 시스템의 DS Development 에이전트입니다.
아래 데이터셋 스키마와 DS 가이드를 참고해서 실험 설정을 결정하세요.

## 데이터셋 스키마
{schema_summary}

## DS 가이드 (docs/DS_GUIDE.md)
{guide}

## 응답 형식 (JSON만, 다른 텍스트 없이)
{{
  "target_column": "타겟 컬럼명",
  "models_to_try": ["모델키1", "모델키2", ...],
  "primary_metric": "평가 지표",
  "feature_exclusions": ["제외할 컬럼명", ...],
  "reason": "판단 근거 (한국어, 2~3문장)",
  "confidence": 0.0~1.0
}}

사용 가능한 모델 키: hist_gradient_boosting, lightgbm, xgboost, random_forest, logistic_regression
최소 2개 이상의 모델을 선택하세요."""


def _call_llm(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_response(raw: str) -> dict:
    try:
        data = json.loads(raw)

        # models_to_try 검증 — 알 수 없는 모델 키 제거
        models = [m for m in data.get("models_to_try", []) if m in VALID_MODELS]
        if len(models) < 2:
            models = DEFAULT_PLAN["models_to_try"]

        return {
            "target_column": data.get("target_column", DEFAULT_PLAN["target_column"]),
            "models_to_try": models,
            "primary_metric": data.get("primary_metric", DEFAULT_PLAN["primary_metric"]),
            "feature_exclusions": data.get("feature_exclusions", []),
            "reason": data.get("reason", "판단 불가"),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception:
        return {**DEFAULT_PLAN, "reason": f"응답 파싱 실패: {raw[:100]}", "confidence": 0.0}


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_plan(plan: dict) -> None:
    print(f"  🎯 타겟: {plan['target_column']}")
    print(f"  🤖 모델: {', '.join(plan['models_to_try'])}")
    print(f"  📊 지표: {plan['primary_metric']}")
    if plan["feature_exclusions"]:
        print(f"  🚫 제외 피처: {', '.join(plan['feature_exclusions'])}")
    print(f"  💬 근거: {plan['reason']}")
    print(f"  📈 신뢰도: {plan['confidence']:.0%}")


# ── 공개 API ──────────────────────────────────────────────────────────────────

def plan_experiment(
    columns: list[str],
    n_samples: int,
    n_features: int,
    imbalance_ratio: float,
    use_agent: bool = True,
) -> dict:
    """
    데이터셋 스키마 기반으로 실험 설정을 결정합니다.

    Args:
        columns:          전체 컬럼 목록 (타겟 포함)
        n_samples:        학습 샘플 수
        n_features:       입력 피처 수
        imbalance_ratio:  양성 클래스 비율 (0.0~1.0)
        use_agent:        False면 DEFAULT_PLAN 즉시 반환

    Returns:
        {
            "target_column": str,
            "models_to_try": list[str],
            "primary_metric": str,
            "feature_exclusions": list[str],
            "reason": str,
            "confidence": float,
        }
    """
    if not use_agent:
        print("  ℹ️  ds_development_agent 비활성화 → default plan 사용")
        import copy
        return copy.deepcopy(DEFAULT_PLAN)

    try:
        schema_summary = _build_schema_summary(columns, n_samples, n_features, imbalance_ratio)
        guide = _load_guide()
        prompt = _build_prompt(schema_summary, guide)
        raw = _call_llm(prompt)
        plan = _parse_response(raw)
        _print_plan(plan)
        return plan
    except Exception as e:
        print(f"  ⚠️  ds_development_agent 실패: {e} → default plan 사용")
        import copy
        return copy.deepcopy(DEFAULT_PLAN)
