"""
agents/meta_orchestrator_agent.py — 최상위 레이어 조율 에이전트

역할:
  - mlops.png의 3개 레이어를 총괄 조율
  - DS Development → Automated Pipeline → Operations 순서로 레이어 에이전트에 위임
  - 단일 진입점(orchestrate())으로 전체 파이프라인 실행 계획 수립

레이어 에이전트 위임:
  - DS Development: ds_development_agent.plan_experiment()
      → 첫 배포 또는 새 데이터셋일 때만 호출 (비용 최적화)
  - Automated Pipeline: pipeline_agent.plan_pipeline()
      → 항상 호출 (스테이지 최적화)
  - Operations: operations_agent (monitor.py가 이상 감지 시 호출)
      → 이 에이전트는 조율 결과를 반환, 실제 호출은 monitor.py 담당

하네스 역할:
  - 레이어 에이전트 하나가 실패해도 나머지는 독립 실행
  - 전체 실패 시 full pipeline + default experiment plan으로 fallback

참고 문서: agents/AGENTS_SPEC.md, docs/DS_GUIDE.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.ds_development_agent import DEFAULT_PLAN, plan_experiment
from agents.pipeline_agent import RunPlan, plan_pipeline

REGISTRY_PATH = Path("registry/model_registry.json")
FEATURE_STORE_DIR = Path("feature_store/features")

TriggerReason = Literal["drift", "ci_cd", "manual", "scheduled", "new_experiment"]


# ── 결과 컨테이너 ──────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """meta_orchestrator_agent의 통합 실행 계획"""

    run_plan: RunPlan
    """pipeline_agent가 결정한 스테이지 실행 계획"""

    experiment_plan: dict
    """ds_development_agent가 결정한 실험 설정
    (첫 배포 / 새 데이터셋 아닐 경우 DEFAULT_PLAN 그대로)"""

    layers_called: list[str] = field(default_factory=list)
    """실제로 호출된 레이어 에이전트 목록 (로깅/디버깅용)"""

    def should_run(self, stage: str) -> bool:
        return self.run_plan.should_run(stage)

    def summary(self) -> str:
        skip = self.run_plan.skip_stages
        layers = ", ".join(self.layers_called) if self.layers_called else "none"
        skip_str = f"스킵: {skip}" if skip else "전체 실행"
        return f"[meta_orchestrator] {skip_str} | 호출 레이어: {layers}"


# ── 진입점 ────────────────────────────────────────────────────────────────────

def orchestrate(
    trigger_reason: TriggerReason,
    dataset_id: int = 44089,
    use_agent: bool = True,
) -> OrchestratorResult:
    """
    3개 레이어 에이전트를 조율해 통합 실행 계획을 수립합니다.

    Args:
        trigger_reason: 트리거 원인
        dataset_id:     학습에 사용할 데이터셋 ID
        use_agent:      False면 모든 레이어 에이전트를 rule-based fallback으로 실행

    Returns:
        OrchestratorResult: run_plan + experiment_plan
    """
    layers_called: list[str] = []

    # ── Layer 1: DS Development ───────────────────────────────────────────────
    experiment_plan = _call_ds_development(trigger_reason, dataset_id, use_agent, layers_called)

    # ── Layer 2: Automated Pipeline ───────────────────────────────────────────
    run_plan = _call_pipeline(trigger_reason, dataset_id, use_agent, layers_called)

    result = OrchestratorResult(
        run_plan=run_plan,
        experiment_plan=experiment_plan,
        layers_called=layers_called,
    )

    _print_result(result)
    return result


# ── 레이어별 호출 로직 ─────────────────────────────────────────────────────────

def _call_ds_development(
    trigger_reason: str,
    dataset_id: int,
    use_agent: bool,
    layers_called: list[str],
) -> dict:
    """
    DS Development 레이어 — 아래 조건에서만 호출 (비용 최적화):
      - 첫 배포 (registry에 deployed 모델 없음)
      - 새 데이터셋 (현재 deployed 모델과 dataset_id 다름)
      - 명시적 실험 재설정 (trigger_reason == "new_experiment")
    """
    if not _should_call_ds_dev(trigger_reason, dataset_id):
        return _load_current_experiment_plan()

    try:
        import copy
        from data.loader import load_dataset
        split = load_dataset(dataset_id)
        all_columns = list(split.X_train.columns) + ["SeriousDlqin2yrs"]
        imbalance = float(split.y_train.mean())

        plan = plan_experiment(
            columns=all_columns,
            n_samples=len(split.X_train),
            n_features=len(split.X_train.columns),
            imbalance_ratio=imbalance,
            use_agent=use_agent,
        )
        layers_called.append("ds_development")
        return plan
    except Exception as e:
        print(f"  ⚠️  DS Development 레이어 실패: {e} → default experiment plan")
        import copy
        return copy.deepcopy(DEFAULT_PLAN)


def _call_pipeline(
    trigger_reason: str,
    dataset_id: int,
    use_agent: bool,
    layers_called: list[str],
) -> RunPlan:
    """Automated Pipeline 레이어 — 항상 호출"""
    try:
        plan = plan_pipeline(
            trigger_reason=trigger_reason,
            dataset_id=dataset_id,
            use_agent=use_agent,
        )
        layers_called.append("pipeline")
        return plan
    except Exception as e:
        print(f"  ⚠️  Pipeline 레이어 실패: {e} → full pipeline fallback")
        from agents.pipeline_agent import ALL_STAGES
        return RunPlan(
            stages=ALL_STAGES[:],
            skip_stages=[],
            reason=f"Pipeline 레이어 실패 — 전체 실행",
            trigger_reason=trigger_reason,
        )


# ── 조건 판단 ─────────────────────────────────────────────────────────────────

def _should_call_ds_dev(trigger_reason: str, dataset_id: int) -> bool:
    """DS Development 레이어를 호출해야 하는지 판단합니다."""
    # 명시적 실험 재설정 트리거
    if trigger_reason == "new_experiment":
        return True

    registry = _load_registry()
    models = registry.get("models", [])
    deployed = next((m for m in reversed(models) if m["status"] == "deployed"), None)

    # 첫 배포 (deployed 모델 없음)
    if deployed is None:
        return True

    # 새 데이터셋 (현재 배포 모델과 다른 dataset_id)
    if deployed.get("dataset_id") != dataset_id:
        return True

    return False


def _load_current_experiment_plan() -> dict:
    """현재 deployed 모델의 학습 설정을 registry에서 로드합니다.
    없으면 DEFAULT_PLAN 반환."""
    import copy
    try:
        registry = _load_registry()
        deployed = next(
            (m for m in reversed(registry.get("models", [])) if m["status"] == "deployed"),
            None,
        )
        if deployed and deployed.get("experiment_plan"):
            return deployed["experiment_plan"]
    except Exception:
        pass
    return copy.deepcopy(DEFAULT_PLAN)


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"models": []}


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_result(result: OrchestratorResult) -> None:
    print(f"\n  {result.summary()}")
    rp = result.run_plan
    ep = result.experiment_plan
    if rp.skip_stages:
        print(f"  📋 실행 스테이지: {rp.stages}")
        print(f"  ⏭️  스킵 스테이지: {rp.skip_stages}")
    if "ds_development" in result.layers_called:
        print(f"  🔬 실험 설정: target={ep.get('target_column')}, "
              f"models={ep.get('models_to_try')}")
