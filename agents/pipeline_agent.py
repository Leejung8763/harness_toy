"""
agents/pipeline_agent.py — Automated Pipeline 레이어 에이전트

역할:
  - orchestrator_agent + deploy_agent 두 역할을 단일 파일로 통합
  - 파이프라인 전체 맥락을 공유해 두 판단의 일관성 확보

두 가지 서브 판단:
  1. plan_pipeline()  — 어떤 스테이지를 실행할지 (orchestrator 역할)
  2. judge_deployment() — 모델을 배포할지 (deploy_agent 역할)

하네스 역할:
  - plan_pipeline fallback: rule-based 전체 실행
  - judge_deployment fallback: "start_ab_test" (보수적 안전 기본값)
  - 두 판단 모두 독립적으로 fallback — 한 쪽 실패가 다른 쪽에 영향 없음

기존 에이전트 호환:
  - orchestrator_agent.plan_pipeline() 시그니처 유지
  - deploy_agent.judge_deployment() 시그니처 유지
  - 두 모듈 모두 독립 실행 가능 (backward compatible)

참고 문서: agents/AGENTS_SPEC.md, docs/QUALITY_SCORE.md
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

THRESHOLDS_PATH = Path("config/thresholds.json")
REGISTRY_PATH = Path("registry/model_registry.json")
METADATA_PATH = Path("ml_metadata/runs.json")
FEATURE_STORE_DIR = Path("feature_store/features")
QUALITY_SCORE_PATH = Path("docs/QUALITY_SCORE.md")

TriggerReason = Literal["drift", "ci_cd", "manual", "scheduled"]
DeployDecision = Literal["deploy", "start_ab_test", "reject"]

ALL_STAGES = ["data_eng", "train", "evaluate", "deploy", "monitor"]


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class RunPlan:
    """파이프라인 실행 계획 (orchestrator 역할)"""
    stages: list[str]
    skip_stages: list[str]
    reason: str
    confidence: float = 1.0
    trigger_reason: str = ""

    def should_run(self, stage: str) -> bool:
        return stage not in self.skip_stages


@dataclass
class DeployJudgment:
    """배포 판단 결과 (deploy_agent 역할)"""
    decision: DeployDecision
    reason: str
    confidence: float
    candidate_version: str = ""


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


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"models": []}


def _load_thresholds() -> dict:
    return json.loads(THRESHOLDS_PATH.read_text()) if THRESHOLDS_PATH.exists() else {}


def _load_recent_runs(n: int = 3) -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    runs = json.loads(METADATA_PATH.read_text())
    if not isinstance(runs, list):
        return []
    return [r for r in runs if r.get("stage") == "pipeline"][-n:]


def _get_feature_store_info(dataset_id: int) -> dict:
    meta_path = FEATURE_STORE_DIR / str(dataset_id) / "v1" / "meta.json"
    if not meta_path.exists():
        return {"exists": False, "age_hours": None}
    meta = json.loads(meta_path.read_text())
    created_at = datetime.fromisoformat(meta["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - created_at).total_seconds() / 3600
    return {"exists": True, "age_hours": round(age_hours, 1), "n_train": meta.get("n_train")}


# ══════════════════════════════════════════════════════════════════════════════
# 1. plan_pipeline — 스테이지 실행 계획 (orchestrator 역할)
# ══════════════════════════════════════════════════════════════════════════════

def plan_pipeline(
    trigger_reason: TriggerReason,
    dataset_id: int = 44089,
    use_agent: bool = True,
) -> RunPlan:
    """
    파이프라인 실행 계획을 수립합니다.

    Args:
        trigger_reason: 트리거 원인 ("drift" | "ci_cd" | "manual" | "scheduled")
        dataset_id:     학습에 사용할 데이터셋 ID
        use_agent:      False면 rule-based fallback만 실행

    Returns:
        RunPlan: 실행 스테이지 목록 및 스킵 근거
    """
    fs_info = _get_feature_store_info(dataset_id)
    registry = _load_registry()
    recent_runs = _load_recent_runs()
    context = {"feature_store": fs_info, "registry": registry, "recent_runs": recent_runs}

    if not use_agent:
        return _rule_based_plan(trigger_reason, fs_info)

    try:
        prompt = _build_stage_plan_prompt(trigger_reason, dataset_id, context)
        raw = _call_llm(prompt)
        plan = _parse_stage_plan(raw, trigger_reason)
        _print_run_plan(plan)
        return plan
    except Exception as e:
        print(f"  ⚠️  pipeline_agent(stage_plan) 실패: {e} → full pipeline 실행")
        return _rule_based_plan(trigger_reason, fs_info)


def _build_stage_plan_prompt(trigger_reason: str, dataset_id: int, ctx: dict) -> str:
    fs = ctx["feature_store"]
    reg = ctx["registry"]
    models = reg.get("models", [])
    deployed = next((m for m in reversed(models) if m["status"] == "deployed"), None)

    deployed_info = "없음"
    if deployed:
        deployed_info = (f"{deployed['version']} "
                         f"(ROC-AUC={deployed['metrics'].get('roc_auc', 'N/A')}, "
                         f"trained_at={deployed.get('trained_at', 'N/A')[:10]})")

    fs_info = (f"있음 (age={fs['age_hours']}h, n_train={fs.get('n_train')})"
               if fs.get("exists") else "없음")

    return f"""당신은 MLOps 파이프라인 에이전트입니다.
시스템 상태를 분석해 불필요한 스테이지를 스킵하세요.

## 트리거
- 원인: {trigger_reason}
- dataset_id: {dataset_id}

## 시스템 상태
- 배포 모델: {deployed_info}
- Feature Store: {fs_info}

## 스테이지 스킵 규칙
- data_eng 스킵 조건: Feature Store가 있고 age < 24h (단, ci_cd 트리거는 항상 전체 실행)

## 응답 (JSON만)
{{
  "skip_stages": ["스킵할 스테이지명, 없으면 빈 배열"],
  "reason": "한국어 1~2문장",
  "confidence": 0.0~1.0
}}"""


def _parse_stage_plan(raw: str, trigger_reason: str) -> RunPlan:
    try:
        data = json.loads(raw)
        skip = [s for s in data.get("skip_stages", []) if s in ALL_STAGES]
        stages = [s for s in ALL_STAGES if s not in skip]
        return RunPlan(
            stages=stages, skip_stages=skip,
            reason=data.get("reason", ""),
            confidence=float(data.get("confidence", 1.0)),
            trigger_reason=trigger_reason,
        )
    except Exception:
        return RunPlan(stages=ALL_STAGES[:], skip_stages=[],
                       reason="파싱 실패 — 전체 실행", confidence=0.0,
                       trigger_reason=trigger_reason)


def _rule_based_plan(trigger_reason: str, fs_info: dict) -> RunPlan:
    """LLM 없이 규칙 기반 실행 계획"""
    if trigger_reason == "ci_cd":
        return RunPlan(stages=ALL_STAGES[:], skip_stages=[],
                       reason="CI/CD 트리거 — 전체 실행",
                       trigger_reason=trigger_reason)
    age = fs_info.get("age_hours")
    if fs_info.get("exists") and age is not None and age < 24:
        return RunPlan(
            stages=[s for s in ALL_STAGES if s != "data_eng"],
            skip_stages=["data_eng"],
            reason=f"Feature Store 신선 (age={age}h) — data_eng 스킵",
            trigger_reason=trigger_reason,
        )
    return RunPlan(stages=ALL_STAGES[:], skip_stages=[],
                   reason="Feature Store 없음/오래됨 — 전체 실행",
                   trigger_reason=trigger_reason)


def _print_run_plan(plan: RunPlan) -> None:
    skip_str = f"스킵: {plan.skip_stages}" if plan.skip_stages else "전체 실행"
    print(f"\n  [pipeline_agent] 📋 실행 계획 — {skip_str}")
    print(f"  근거: {plan.reason}")
    print(f"  실행 스테이지: {plan.stages}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. judge_deployment — 배포 판단 (deploy_agent 역할)
# ══════════════════════════════════════════════════════════════════════════════

def judge_deployment(
    candidate_version: str,
    use_agent: bool = True,
) -> dict:
    """
    후보 모델의 배포 여부를 LLM이 판단합니다.

    Args:
        candidate_version: 평가할 모델 버전 (예: "v3")
        use_agent:         False면 rule-based fallback만 실행

    Returns:
        {"decision": "deploy"|"start_ab_test"|"reject", "reason": str, "confidence": float}
    """
    ctx = _build_deploy_context(candidate_version)
    if ctx is None:
        return {"decision": "start_ab_test",
                "reason": "모델 정보를 찾을 수 없습니다.", "confidence": 0.0}

    if not use_agent:
        result = _rule_based_deploy(ctx)
    else:
        try:
            prompt = _build_deploy_prompt(ctx)
            raw = _call_llm(prompt)
            result = _parse_deploy_response(raw)
        except Exception as e:
            print(f"  ⚠️  pipeline_agent(deploy) 실패: {e} → rule-based fallback")
            result = _rule_based_deploy(ctx)

    _print_deploy_judgment(candidate_version, result)
    _log_deploy_decision(candidate_version, result, ctx)
    return result


def _build_deploy_context(version: str) -> dict | None:
    registry = _load_registry()
    candidate = next((m for m in registry["models"] if m["version"] == version), None)
    if candidate is None:
        return None

    champion = next(
        (m for m in reversed(registry["models"]) if m["status"] == "deployed"), None
    )
    runs = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else []
    latest_training = next(
        (r for r in reversed(runs) if r.get("stage") == "training"), {}
    )
    return {
        "candidate": candidate,
        "champion": champion,
        "latest_training": latest_training,
        "thresholds": _load_thresholds(),
    }


def _build_deploy_prompt(ctx: dict) -> str:
    candidate = ctx["candidate"]
    champion = ctx["champion"]
    training = ctx["latest_training"]
    thresholds = ctx["thresholds"]

    baseline = thresholds.get("evaluate", {}).get("roc_auc_baseline", 0.70)
    roc_min = thresholds.get("monitor", {}).get("roc_auc_min", 0.75)

    champion_info = (
        f"{champion['version']} (ROC-AUC: {champion.get('metrics', {}).get('roc_auc', 'N/A')})"
        if champion else "없음 (첫 배포)"
    )

    return f"""당신은 MLOps 배포 판단 에이전트입니다.

## 후보 모델
- 버전: {candidate['version']} / 모델: {candidate.get('model_name', 'N/A')}
- ROC-AUC: {candidate.get('metrics', {}).get('roc_auc', 'N/A')}

## 챔피언
{champion_info}

## 학습 정보
- Feature Store 사용: {training.get('feature_store_hit', 'N/A')}
- 학습 샘플 수: {training.get('n_train_samples', 'N/A')}

## 임계값
- 최소 ROC-AUC (첫 배포): {baseline}
- 운영 최소 ROC-AUC: {roc_min}

## 판단 기준
1. ROC-AUC 차이 > 0.001 → deploy
2. ROC-AUC 차이 ≤ 0.001 → start_ab_test
3. ROC-AUC < {baseline} → reject
4. 첫 배포 (챔피언 없음) + ROC-AUC ≥ {baseline} → deploy

## 응답 (JSON만)
{{
  "decision": "deploy" | "start_ab_test" | "reject",
  "reason": "한국어 1~2문장",
  "confidence": 0.0~1.0
}}"""


def _parse_deploy_response(raw: str) -> dict:
    try:
        data = json.loads(raw)
        decision = data.get("decision", "start_ab_test")
        if decision not in ("deploy", "start_ab_test", "reject"):
            decision = "start_ab_test"
        return {
            "decision": decision,
            "reason": data.get("reason", "판단 불가"),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception:
        return {"decision": "start_ab_test",
                "reason": f"파싱 실패: {raw[:100]}", "confidence": 0.0}


def _rule_based_deploy(ctx: dict) -> dict:
    """LLM 없이 규칙 기반 배포 판단"""
    candidate = ctx["candidate"]
    champion = ctx["champion"]
    thresholds = ctx["thresholds"]
    baseline = thresholds.get("evaluate", {}).get("roc_auc_baseline", 0.70)

    cand_auc = candidate.get("metrics", {}).get("roc_auc", 0.0)

    if cand_auc < baseline:
        return {"decision": "reject",
                "reason": f"ROC-AUC({cand_auc:.4f}) < baseline({baseline})",
                "confidence": 1.0}
    if champion is None:
        return {"decision": "deploy",
                "reason": "첫 배포 — baseline 충족",
                "confidence": 1.0}

    champ_auc = champion.get("metrics", {}).get("roc_auc", 0.0)
    diff = cand_auc - champ_auc
    if diff > 0.001:
        return {"decision": "deploy",
                "reason": f"ROC-AUC 개선 ({champ_auc:.4f}→{cand_auc:.4f}, +{diff:.4f})",
                "confidence": 1.0}
    return {"decision": "start_ab_test",
            "reason": f"ROC-AUC 차이({diff:+.4f}) ≤ 0.001 — A/B 테스트로 검증",
            "confidence": 1.0}


def _print_deploy_judgment(version: str, result: dict) -> None:
    icon = {"deploy": "🚀", "start_ab_test": "🔬", "reject": "🚫"}.get(
        result["decision"], "❓"
    )
    print(f"\n  [pipeline_agent] {icon} 배포 판단: {result['decision'].upper()} "
          f"(신뢰도: {result['confidence']:.0%})")
    print(f"  근거: {result['reason']}")


def _log_deploy_decision(version: str, result: dict, ctx: dict) -> None:
    try:
        from agents.eval.decision_logger import log_deploy_decision
        champion = ctx.get("champion")
        log_deploy_decision(
            model_version=version,
            decision=result["decision"],
            confidence=result["confidence"],
            reason=result["reason"],
            context={
                "candidate_roc_auc": ctx["candidate"].get("metrics", {}).get("roc_auc"),
                "champion_roc_auc": champion.get("metrics", {}).get("roc_auc") if champion else None,
                "champion_version": champion["version"] if champion else None,
            },
        )
    except Exception:
        pass
