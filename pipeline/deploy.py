"""
pipeline/deploy.py — 상태 전이 + Feature Flag 업데이트

규칙:
  - 'evaluated_pass' 상태 모델만 배포 가능 (invariant)
  - LLM 에이전트 판단: "deploy" → 즉시 배포, "start_ab_test" → challenger 등록, "reject" → 폐기
  - 배포 시: 신규 → 'deployed', 기존 deployed → 'retired'
  - A/B 테스트 시: 신규 → 'challenger', flags에 challenger 정보 추가
  - Feature Flag는 반드시 이 모듈을 통해서만 변경 (직접 수정 금지)
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path("registry/model_registry.json")
FLAGS_PATH = Path("feature_flags/flags.json")

CHALLENGER_TRAFFIC_WEIGHT = 0.1  # 챌린저 기본 트래픽 비율


def deploy(version: str | None = None, use_agent: bool = True) -> bool:
    """
    'evaluated_pass' 모델을 배포하거나 A/B 테스트 챌린저로 등록합니다.

    Args:
        version: 배포할 버전 (None이면 최신 evaluated_pass 모델)
        use_agent: True면 LLM 에이전트 판단을 거침 (기본값)

    Returns:
        bool: True(성공/challenger 등록) / False(실패/reject)
    """
    registry = _load_registry()
    candidate = _get_candidate(registry, version)

    if candidate is None:
        print("[deploy] ❌ 배포 가능한 'evaluated_pass' 모델이 없습니다.")
        return False

    print(f"\n[deploy] 배포 대상: {candidate['model_name']}  {candidate['version']}  "
          f"(dataset={candidate['dataset_name']})")

    # ── LLM 에이전트 판단 ────────────────────────────────────────
    decision = "deploy"  # fallback
    if use_agent:
        try:
            from agents.pipeline_agent import judge_deployment
            judgment = judge_deployment(candidate["version"])
            decision = judgment["decision"]
            if decision == "reject":
                print(f"  🚫  에이전트 판단: REJECT — 배포 거부")
                return False
        except Exception as e:
            print(f"  ⚠️  에이전트 판단 실패 ({e}) — 자동 배포로 fallback")

    # ── A/B 테스트 챌린저 등록 ───────────────────────────────────
    if decision == "start_ab_test":
        return _register_challenger(registry, candidate)

    # ── 즉시 배포 ────────────────────────────────────────────────
    return _full_deploy(registry, candidate)


def _full_deploy(registry: dict, candidate: dict) -> bool:
    """챔피언으로 즉시 배포합니다."""
    retired = _retire_current(registry, candidate["version"])
    if retired:
        print(f"  이전 champion {retired['version']} → retired")

    candidate["status"] = "deployed"
    _save_registry(registry)
    print(f"  {candidate['version']} → deployed ✅")

    _update_flags(candidate)
    print(f"  flags.json → active_model_version={candidate['version']}")
    return True


def _register_challenger(registry: dict, candidate: dict) -> bool:
    """챌린저로 등록합니다 (트래픽 일부만 받음)."""
    candidate["status"] = "challenger"
    _save_registry(registry)
    print(f"  {candidate['version']} → challenger ✅  (트래픽 {CHALLENGER_TRAFFIC_WEIGHT:.0%})")

    # flags에 challenger 정보 추가
    flags = _load_flags()
    flags["challenger_model_version"] = candidate["version"]
    flags["challenger_traffic_weight"] = CHALLENGER_TRAFFIC_WEIGHT
    _save_flags(flags)
    print(f"  flags.json → challenger_model_version={candidate['version']}, "
          f"traffic={CHALLENGER_TRAFFIC_WEIGHT:.0%}")
    return True


def _get_candidate(registry: dict, version: str | None) -> dict | None:
    """배포할 후보(evaluated_pass 상태)를 반환합니다."""
    candidates = [m for m in registry.get("models", []) if m["status"] == "evaluated_pass"]
    if not candidates:
        return None
    if version:
        matched = [m for m in candidates if m["version"] == version]
        return matched[0] if matched else None
    return candidates[-1]  # 최신 evaluated_pass


def _retire_current(registry: dict, exclude_version: str) -> dict | None:
    """현재 deployed 모델을 retired로 변경하고 반환합니다."""
    for model in registry.get("models", []):
        if model["status"] == "deployed" and model["version"] != exclude_version:
            model["status"] = "retired"
            return model
    return None


def _update_flags(deployed: dict) -> None:
    """Feature Flag를 업데이트합니다. 이 함수를 통해서만 flags.json을 변경합니다."""
    flags = _load_flags()
    flags["active_model_version"] = deployed["version"]
    flags["active_dataset_id"] = deployed["dataset_id"]
    _save_flags(flags)


def load_flags() -> dict:
    return _load_flags()


def _load_flags() -> dict:
    if FLAGS_PATH.exists():
        with open(FLAGS_PATH) as f:
            return json.load(f)
    return {"active_model_version": None, "active_dataset_id": None}


def _save_flags(flags: dict) -> None:
    FLAGS_PATH.parent.mkdir(exist_ok=True)
    with open(FLAGS_PATH, "w") as f:
        json.dump(flags, f, indent=2)


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else None
    success = deploy(version)
    sys.exit(0 if success else 1)
