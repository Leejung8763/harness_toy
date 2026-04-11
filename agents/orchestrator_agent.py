"""
agents/orchestrator_agent.py — 파이프라인 오케스트레이터 에이전트

역할:
  - 파이프라인 실행 전 "무엇을 실행할지" 결정
  - trigger_reason + 시스템 상태를 LLM에게 전달 → 최적 스테이지 계획 반환
  - Feature Store가 신선하면 data_eng 스킵 등 리소스 최적화

하네스 역할:
  - 에이전트 실패 시 전체 파이프라인 실행 (full fallback)
  - 스테이지 스킵 여부만 결정, 실제 실행은 harness_pipeline.py가 담당

참고 문서: agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

THRESHOLDS_PATH = Path("config/thresholds.json")
REGISTRY_PATH = Path("registry/model_registry.json")
METADATA_PATH = Path("ml_metadata/runs.json")
FEATURE_STORE_DIR = Path("feature_store/features")

TriggerReason = Literal["drift", "ci_cd", "manual", "scheduled"]

ALL_STAGES = ["data_eng", "train", "evaluate", "deploy", "monitor"]


@dataclass
class RunPlan:
    stages: list[str]
    skip_stages: list[str]
    reason: str
    confidence: float = 1.0
    trigger_reason: str = ""

    def should_run(self, stage: str) -> bool:
        return stage not in self.skip_stages


def plan_pipeline(
    trigger_reason: TriggerReason,
    dataset_id: int = 44089,
    use_agent: bool = True,
) -> RunPlan:
    """
    파이프라인 실행 계획을 수립합니다.

    Args:
        trigger_reason: 트리거 원인 ("drift" | "ci_cd" | "manual" | "scheduled")
        dataset_id: 학습에 사용할 데이터셋 ID
        use_agent: False면 rule-based fallback만 실행

    Returns:
        RunPlan: 실행할 스테이지 목록 및 근거
    """
    context = _gather_context(dataset_id)

    if not use_agent:
        return _rule_based_plan(trigger_reason, context)

    try:
        prompt = _build_prompt(trigger_reason, dataset_id, context)
        response = _call_llm(prompt)
        plan = _parse_response(response, trigger_reason)
        _print_plan(plan)
        return plan
    except Exception as e:
        print(f"  ⚠️  orchestrator_agent 실패: {e} → full pipeline 실행")
        return _rule_based_plan(trigger_reason, context)


# ── Context 수집 ──────────────────────────────────────────────────────────────

def _gather_context(dataset_id: int) -> dict:
    return {
        "feature_store": _get_feature_store_info(dataset_id),
        "registry": _get_registry_info(),
        "recent_runs": _get_recent_runs(n=3),
    }


def _get_feature_store_info(dataset_id: int) -> dict:
    meta_path = FEATURE_STORE_DIR / str(dataset_id) / "v1" / "meta.json"
    if not meta_path.exists():
        return {"exists": False, "age_hours": None, "dataset_id": dataset_id}

    meta = json.loads(meta_path.read_text())
    created_at = datetime.fromisoformat(meta["created_at"])
    # timezone-aware 비교
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    age_hours = (now - created_at).total_seconds() / 3600

    return {
        "exists": True,
        "age_hours": round(age_hours, 1),
        "n_train": meta.get("n_train"),
        "dataset_id": dataset_id,
        "created_at": meta["created_at"],
    }


def _get_registry_info() -> dict:
    if not REGISTRY_PATH.exists():
        return {"deployed": None, "total_models": 0}

    registry = json.loads(REGISTRY_PATH.read_text())
    models = registry.get("models", [])
    deployed = next((m for m in reversed(models) if m["status"] == "deployed"), None)

    return {
        "deployed": deployed,
        "total_models": len(models),
    }


def _get_recent_runs(n: int = 3) -> list[dict]:
    if not METADATA_PATH.exists():
        return []

    runs = json.loads(METADATA_PATH.read_text())
    if not isinstance(runs, list):
        return []

    pipeline_runs = [r for r in runs if r.get("stage") == "pipeline"]
    return pipeline_runs[-n:]


# ── Prompt 작성 ───────────────────────────────────────────────────────────────

def _build_prompt(trigger_reason: str, dataset_id: int, context: dict) -> str:
    fs = context["feature_store"]
    reg = context["registry"]
    runs = context["recent_runs"]

    deployed_info = "없음"
    if reg.get("deployed"):
        d = reg["deployed"]
        deployed_info = f"{d['version']} (ROC-AUC={d['metrics'].get('roc_auc', 'N/A')}, trained_at={d.get('trained_at', 'N/A')[:10]})"

    fs_info = "없음"
    if fs.get("exists"):
        fs_info = f"있음 (age={fs['age_hours']}h, n_train={fs.get('n_train')})"

    runs_info = f"{len(runs)}개 최근 파이프라인 실행" if runs else "없음"

    return f"""당신은 MLOps 파이프라인 오케스트레이터 에이전트입니다.
아래 시스템 상태를 분석하고 파이프라인 실행 계획을 결정하세요.

## 트리거 정보
- 원인: {trigger_reason}
- 대상 dataset_id: {dataset_id}

## 현재 시스템 상태
- 배포된 모델: {deployed_info}
- Feature Store: {fs_info}
- 최근 파이프라인 실행: {runs_info}

## 실행 가능한 스테이지
{ALL_STAGES}

## 스테이지 스킵 판단 기준
- data_eng 스킵 조건: Feature Store가 존재하고 age < 24h인 경우 (드리프트/재학습 트리거에서)
- ci_cd 트리거: 코드 변경이므로 항상 전체 실행
- drift 트리거: Feature Store 신선하면 data_eng 스킵 가능
- manual/scheduled: 상태 보고 판단

## 응답 형식 (JSON만 반환)
{{
  "skip_stages": ["스킵할 스테이지명 리스트, 없으면 빈 배열"],
  "reason": "결정 근거 (한국어 1~2문장)",
  "confidence": 0.0~1.0
}}"""


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# ── 응답 파싱 ─────────────────────────────────────────────────────────────────

def _parse_response(raw: str, trigger_reason: str) -> RunPlan:
    try:
        data = json.loads(raw)
        skip = [s for s in data.get("skip_stages", []) if s in ALL_STAGES]
        stages = [s for s in ALL_STAGES if s not in skip]
        return RunPlan(
            stages=stages,
            skip_stages=skip,
            reason=data.get("reason", ""),
            confidence=float(data.get("confidence", 1.0)),
            trigger_reason=trigger_reason,
        )
    except Exception:
        return RunPlan(
            stages=ALL_STAGES[:],
            skip_stages=[],
            reason=f"응답 파싱 실패 — 전체 실행",
            confidence=0.0,
            trigger_reason=trigger_reason,
        )


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_plan(trigger_reason: str, context: dict) -> RunPlan:
    """LLM 없이 규칙 기반으로 실행 계획을 수립합니다."""
    fs = context["feature_store"]

    if trigger_reason == "ci_cd":
        return RunPlan(
            stages=ALL_STAGES[:],
            skip_stages=[],
            reason="CI/CD 트리거 — 코드 변경으로 전체 파이프라인 실행",
            trigger_reason=trigger_reason,
        )

    # drift / manual / scheduled — Feature Store 신선하면 data_eng 스킵
    if fs.get("exists") and fs.get("age_hours", 999) < 24:
        return RunPlan(
            stages=[s for s in ALL_STAGES if s != "data_eng"],
            skip_stages=["data_eng"],
            reason=f"Feature Store 신선 (age={fs['age_hours']}h) — data_eng 스킵",
            trigger_reason=trigger_reason,
        )

    return RunPlan(
        stages=ALL_STAGES[:],
        skip_stages=[],
        reason="Feature Store 없음/오래됨 — 전체 파이프라인 실행",
        trigger_reason=trigger_reason,
    )


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_plan(plan: RunPlan) -> None:
    skip_str = f"스킵: {plan.skip_stages}" if plan.skip_stages else "전체 실행"
    print(f"\n  [orchestrator] 📋 실행 계획 — {skip_str}")
    print(f"  근거: {plan.reason}")
    print(f"  실행 스테이지: {plan.stages}")
