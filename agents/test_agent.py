"""
agents/test_agent.py — 평가 결과 해석 및 합격/불합격 판단 에이전트

판단 내용:
  - 합격(pass) / 불합격(fail) 판단
  - 실패 원인 분석 및 개선 제안
  - 재학습 필요 여부

하네스 역할:
  - LLM 반환값은 TestVerdict(Pydantic)으로 즉시 검증 — 잘못된 값은 경계에서 차단
  - LLM 실패 시 ROC-AUC > 0.70 rule-based fallback
  - 실제 평가 실행은 core/test.py가 담당

참고: agents/AGENTS_SPEC.md, agents/schemas.py
"""

from __future__ import annotations

import json
import subprocess

from pydantic import ValidationError

from agents._spec_loader import load as _load_spec
from agents.schemas import ModelPlan, TestVerdict, TrainPlan

BASELINE_ROC_AUC = 0.70

_SPEC = _load_spec("test_agent")


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
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(metrics: dict, model_plan: ModelPlan, train_plan: TrainPlan, hint: str = "") -> str:
    hint_section = f"\nOrchestrator hint: {hint}" if hint else ""
    return f"""You are an ML quality gate expert.{hint_section}

Model: {model_plan.model_type}
Training: {train_plan.cv_folds}-fold CV, metric={train_plan.primary_metric}

Evaluation results:
{json.dumps(metrics, indent=2)}

Baseline threshold: ROC-AUC >= {BASELINE_ROC_AUC}

Judge the model quality. Return JSON:
- "verdict": "pass" or "fail"
- "reason": explain why pass or fail
- "suggestions": list of improvement suggestions if fail (empty list if pass)
- "retrain_needed": bool
- "confidence": float 0-1
"""


def _rule_based_verdict(metrics: dict) -> TestVerdict:
    roc_auc = metrics.get("roc_auc", 0.0)
    passed = roc_auc >= BASELINE_ROC_AUC
    return TestVerdict(
        verdict="pass" if passed else "fail",
        reason=f"rule-based: roc_auc={roc_auc:.4f} {'≥' if passed else '<'} {BASELINE_ROC_AUC}",
        suggestions=[] if passed else [f"ROC-AUC {roc_auc:.4f} is below baseline {BASELINE_ROC_AUC}. Consider feature engineering or a different model."],
        retrain_needed=not passed,
        confidence=1.0,
    )


def judge_results(
    metrics: dict,
    model_plan: ModelPlan,
    train_plan: TrainPlan,
    hint: str = "",
    use_agent: bool = True,
) -> TestVerdict:
    """
    평가 결과를 해석하고 합격/불합격을 판단합니다.

    Returns:
        TestVerdict (Pydantic) — 하네스가 계약 검증 완료된 판단
    """
    if not use_agent:
        return _rule_based_verdict(metrics)

    try:
        prompt = _build_prompt(metrics, model_plan, train_plan, hint)
        raw = _call_llm(prompt)
        data = json.loads(raw)

        # ── Pydantic 계약 검증: verdict는 반드시 "pass"/"fail" 중 하나 ──
        return TestVerdict(**data)

    except (ValidationError, Exception) as e:
        print(f"[test_agent] fallback: {e}")
        return _rule_based_verdict(metrics)
