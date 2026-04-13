"""
agents/train_agent.py — 학습 전략 결정 에이전트

판단 내용:
  - CV folds 수
  - stratify 여부
  - early_stopping 사용 여부 (트리 계열 모델용)
  - 평가 지표 (primary_metric)

하네스 역할:
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 학습 실행은 core/train.py가 담당

참고: agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess

DEFAULT_PLAN: dict = {
    "cv_folds": 5,
    "stratify": True,
    "early_stopping": False,
    "primary_metric": "roc_auc",
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
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(
    n_samples: int,
    n_classes: int,
    class_balance: dict,
    model_plan: dict,
) -> str:
    return f"""You are an ML training strategy expert.

Dataset: {n_samples} samples, {n_classes} classes
Class balance: {class_balance}
Model: {model_plan.get('model_type')} with params {model_plan.get('params')}

Decide the training strategy. Return JSON:
- "cv_folds": int (3-10), more folds for small datasets
- "stratify": bool, true if class imbalance > 2:1
- "early_stopping": bool, true only for gradient boosting models with large n_estimators
- "primary_metric": one of "roc_auc", "f1", "accuracy"
- "reason": brief explanation
- "confidence": float 0-1
"""


def plan_training(
    n_samples: int,
    n_classes: int,
    class_balance: dict,
    model_plan: dict,
    use_agent: bool = True,
) -> dict:
    """
    학습 전략을 결정합니다.

    Returns:
        dict with keys: cv_folds, stratify, early_stopping, primary_metric, reason, confidence
    """
    if not use_agent:
        return {**DEFAULT_PLAN, "reason": "use_agent=False (rule-based)"}

    try:
        prompt = _build_prompt(n_samples, n_classes, class_balance, model_plan)
        raw = _call_llm(prompt)
        plan = json.loads(raw)

        plan["cv_folds"] = max(3, min(10, int(plan.get("cv_folds", 5))))
        plan["stratify"] = bool(plan.get("stratify", True))
        plan["early_stopping"] = bool(plan.get("early_stopping", False))
        plan["primary_metric"] = plan.get("primary_metric", "roc_auc") if plan.get("primary_metric") in ("roc_auc", "f1", "accuracy") else "roc_auc"
        plan.setdefault("reason", "LLM decision")
        plan.setdefault("confidence", 0.8)

        return plan

    except Exception as e:
        print(f"[train_agent] fallback: {e}")
        return {**DEFAULT_PLAN, "reason": f"fallback due to: {e}"}
