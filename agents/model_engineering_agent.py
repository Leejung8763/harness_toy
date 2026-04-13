"""
agents/model_engineering_agent.py — 모델 타입 및 하이퍼파라미터 결정 에이전트

판단 내용:
  - 사용할 모델 타입 (hist_gradient_boosting / lightgbm / random_forest / logistic_regression)
  - 하이퍼파라미터 후보 (learning_rate, max_depth, n_estimators 등)

하네스 역할:
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 모델 인스턴스 생성은 core/model_engineering.py가 담당

참고: agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess

import pandas as pd

VALID_MODELS = {
    "hist_gradient_boosting",
    "lightgbm",
    "random_forest",
    "logistic_regression",
}

DEFAULT_PLAN: dict = {
    "model_type": "hist_gradient_boosting",
    "params": {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
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
    preprocessing_plan: dict,
) -> str:
    return f"""You are an ML model selection expert.

Dataset profile:
- Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}
- Class balance: {class_balance}
- Preprocessing applied: scaler={preprocessing_plan.get('scaler')}, features_dropped={len(preprocessing_plan.get('feature_exclusions', []))}

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
    preprocessing_plan: dict,
    use_agent: bool = True,
) -> dict:
    """
    모델 타입과 초기 하이퍼파라미터를 결정합니다.

    Returns:
        dict with keys: model_type, params, reason, confidence
    """
    if not use_agent:
        return {**DEFAULT_PLAN, "reason": "use_agent=False (rule-based)"}

    try:
        prompt = _build_prompt(n_samples, n_features, n_classes, class_balance, preprocessing_plan)
        raw = _call_llm(prompt)
        plan = json.loads(raw)

        if plan.get("model_type") not in VALID_MODELS:
            plan["model_type"] = DEFAULT_PLAN["model_type"]
        plan.setdefault("params", DEFAULT_PLAN["params"])
        plan.setdefault("reason", "LLM decision")
        plan.setdefault("confidence", 0.8)

        return plan

    except Exception as e:
        print(f"[model_engineering_agent] fallback: {e}")
        return {**DEFAULT_PLAN, "reason": f"fallback due to: {e}"}
