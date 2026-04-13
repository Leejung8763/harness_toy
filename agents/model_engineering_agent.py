"""
agents/model_engineering_agent.py — 모델 타입 및 하이퍼파라미터 결정 에이전트

판단 내용:
  - 사용할 모델 타입 (hist_gradient_boosting / lightgbm / random_forest / logistic_regression)
  - 하이퍼파라미터 후보 (learning_rate, max_depth, n_estimators 등)

하네스 역할:
  - LLM 반환값은 ModelPlan(Pydantic)으로 즉시 검증 — 잘못된 값은 경계에서 차단
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 모델 인스턴스 생성은 core/model_engineering.py가 담당

참고: agents/AGENTS_SPEC.md, agents/schemas.py
"""

from __future__ import annotations

import json
import subprocess

from pydantic import ValidationError

from agents.schemas import DataEngPlan, ModelPlan

DEFAULT_PLAN = ModelPlan(
    params={"learning_rate": 0.05, "max_iter": 200, "max_depth": 6, "min_samples_leaf": 20},
    reason="default plan (fallback)",
)


def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


def _call_llm(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(
    n_samples: int,
    n_features: int,
    n_classes: int,
    class_balance: dict,
    preprocessing_plan: DataEngPlan,
) -> str:
    return f"""You are an ML model selection expert.

Dataset profile:
- Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}
- Class balance: {class_balance}
- Preprocessing applied: scaler={preprocessing_plan.scaler}, features_dropped={len(preprocessing_plan.feature_exclusions)}

Task: binary/multiclass classification.

Choose the best model and initial hyperparameters. Return JSON:
- "model_type": one of {sorted(VALID_MODELS)}
- "params": dict of hyperparameters appropriate for the chosen model
  - hist_gradient_boosting: learning_rate, max_iter, max_depth, min_samples_leaf
  - lightgbm: learning_rate, n_estimators, max_depth, num_leaves
  - random_forest: n_estimators, max_depth, min_samples_leaf
  - logistic_regression: C, max_iter, solver
- "reason": brief explanation
- "confidence": float 0-1
"""


def plan_model(
    n_samples: int,
    n_features: int,
    n_classes: int,
    class_balance: dict,
    preprocessing_plan: DataEngPlan,
    use_agent: bool = True,
) -> ModelPlan:
    """
    모델 타입과 초기 하이퍼파라미터를 결정합니다.

    Returns:
        ModelPlan (Pydantic) — 하네스가 계약 검증 완료된 판단
    """
    if not use_agent:
        return ModelPlan(reason="use_agent=False (rule-based)", params=DEFAULT_PLAN.params)

    try:
        prompt = _build_prompt(n_samples, n_features, n_classes, class_balance, preprocessing_plan)
        raw = _call_llm(prompt)
        data = json.loads(raw)

        # ── Pydantic 계약 검증: 잘못된 model_type, 범위 이탈 값은 여기서 즉시 차단 ──
        return ModelPlan(**data)

    except (ValidationError, Exception) as e:
        print(f"[model_engineering_agent] fallback: {e}")
        return ModelPlan(params=DEFAULT_PLAN.params, reason=f"fallback due to: {e}")
