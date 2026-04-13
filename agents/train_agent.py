"""
agents/train_agent.py — 학습 전략 결정 에이전트

판단 내용:
  - CV folds 수
  - stratify 여부
  - early_stopping 사용 여부 (트리 계열 모델용)
  - 평가 지표 (primary_metric)

하네스 역할:
  - LLM 반환값은 TrainPlan(Pydantic)으로 즉시 검증 — 잘못된 값은 경계에서 차단
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 학습 실행은 core/train.py가 담당

참고: agents/AGENTS_SPEC.md, agents/schemas.py
"""

from __future__ import annotations

import json
import subprocess

from pydantic import ValidationError

from agents._spec_loader import load as _load_spec
from agents.schemas import ModelPlan, TrainPlan

DEFAULT_PLAN = TrainPlan(reason="default plan (fallback)")

_SPEC = _load_spec("train_agent")


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
    messages = []
    if _SPEC:
        messages.append({"role": "system", "content": _SPEC})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(
    n_samples: int,
    n_classes: int,
    class_balance: dict,
    model_plan: ModelPlan,
    hint: str = "",
) -> str:
    hint_section = f"\nOrchestrator hint: {hint}" if hint else ""
    return f"""You are an ML training strategy expert.{hint_section}

Dataset: {n_samples} samples, {n_classes} classes
Class balance: {class_balance}
Model: {model_plan.model_type} with params {model_plan.params}

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
    model_plan: ModelPlan,
    hint: str = "",
    use_agent: bool = True,
) -> TrainPlan:
    """
    학습 전략을 결정합니다.

    Returns:
        TrainPlan (Pydantic) — 하네스가 계약 검증 완료된 판단
    """
    if not use_agent:
        return TrainPlan(reason="use_agent=False (rule-based)")

    try:
        prompt = _build_prompt(n_samples, n_classes, class_balance, model_plan, hint)
        raw = _call_llm(prompt)
        data = json.loads(raw)

        # ── Pydantic 계약 검증: cv_folds 범위(3-10), metric 허용값 등 즉시 차단 ──
        return TrainPlan(**data)

    except (ValidationError, Exception) as e:
        print(f"[train_agent] fallback: {e}")
        return TrainPlan(reason=f"fallback due to: {e}")
