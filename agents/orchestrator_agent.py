"""
agents/orchestrator_agent.py — 파이프라인 총괄 에이전트

역할:
  - 사용자의 자연어 지시를 받아 ML 파이프라인 실행 계획을 수립
  - 각 하위 에이전트(data_eng, model_eng, train, test)에 전달할 힌트 생성
  - 실제 에이전트 실행은 run_pipeline.py가 담당 (에이전트는 계획만)

하네스 역할:
  - LLM 반환값은 OrchestratorPlan(Pydantic)으로 즉시 검증
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - specs/orchestrator.md를 system message로 주입

참고: agents/specs/orchestrator.md, agents/schemas.py
"""

from __future__ import annotations

import json
import subprocess

from pydantic import ValidationError

from agents._spec_loader import load as _load_spec
from agents.schemas import OrchestratorPlan

DEFAULT_PLAN = OrchestratorPlan(reason="default plan (fallback)")

_SPEC = _load_spec("orchestrator_agent")


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
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(instruction: str, dataset_id: int) -> str:
    return f"""You are an ML pipeline orchestrator.

User instruction: "{instruction}"
Dataset ID: {dataset_id}

Based on the instruction, create an execution plan for the ML pipeline.
Return JSON:
- "goal": brief description of the ML objective
- "stages": list of stages to run in order (always ["data_eng", "model_eng", "train", "test"])
- "hints": dict with context hints for each stage based on the instruction
  - "data_eng": hint about preprocessing (e.g., domain-specific concerns)
  - "model_eng": hint about model selection (e.g., interpretability needs)
  - "train": hint about training strategy (e.g., class imbalance handling)
  - "test": hint about evaluation criteria (e.g., which metric matters most)
- "reason": brief explanation of the plan
- "confidence": float 0-1
"""


def plan(
    instruction: str,
    dataset_id: int = 44089,
    use_agent: bool = True,
) -> OrchestratorPlan:
    """
    사용자 지시를 받아 파이프라인 실행 계획을 수립합니다.

    Args:
        instruction: 사용자 자연어 지시 (예: "credit 데이터로 연체 여부 예측 모델 만들어줘")
        dataset_id:  OpenML 데이터셋 ID
        use_agent:   False면 rule-based fallback만 실행

    Returns:
        OrchestratorPlan (Pydantic) — 하네스가 계약 검증 완료된 실행 계획
    """
    if not use_agent:
        return OrchestratorPlan(reason="use_agent=False (rule-based)")

    try:
        prompt = _build_prompt(instruction, dataset_id)
        raw = _call_llm(prompt)
        data = json.loads(raw)

        # ── Pydantic 계약 검증 ──
        return OrchestratorPlan(**data)

    except (ValidationError, Exception) as e:
        print(f"[orchestrator_agent] fallback: {e}")
        return OrchestratorPlan(reason=f"fallback due to: {e}")
