"""
pipeline/evaluate.py — Quality Gate

규칙:
  - 현재 deployed champion이 있으면: 신규 모델 ROC-AUC > champion ROC-AUC 이어야 통과
  - deployed champion이 없으면 (첫 배포): ROC-AUC >= BASELINE_THRESHOLD 이어야 통과
  - 통과: status → "evaluated_pass"
  - 실패: status → "evaluated_fail", 파이프라인 중단
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from data.loader import load_dataset

REGISTRY_PATH = Path("registry/model_registry.json")
_CONFIG_PATH = Path("config/thresholds.json")


def _load_thresholds() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f).get("evaluate", {})
    return {}


BASELINE_THRESHOLD = _load_thresholds().get("roc_auc_baseline", 0.70)


def evaluate(version: str | None = None) -> bool:
    """
    가장 최근 'trained' 모델을 평가합니다.

    Args:
        version: 특정 버전 지정 (None이면 최신 trained 모델)

    Returns:
        bool: True(통과) / False(실패)
    """
    registry = _load_registry()
    candidate = _get_candidate(registry, version)

    if candidate is None:
        print("[evaluate] ❌ 평가할 'trained' 모델이 없습니다.")
        return False

    print(f"\n[evaluate] 후보 모델: {candidate['model_name']}  {candidate['version']}  "
          f"(dataset={candidate['dataset_name']})")

    # 모델 로드 및 재평가
    roc_auc = _recompute_roc_auc(candidate)
    print(f"  재평가 ROC-AUC: {roc_auc:.4f}")

    # 통과 기준 결정
    champion = _get_deployed_champion(registry, exclude_version=candidate["version"])
    passed, reason = _check_gate(roc_auc, champion)

    if passed:
        print(f"  ✅ PASS — {reason}")
        _update_status(registry, candidate["version"], "evaluated_pass",
                       {"roc_auc": roc_auc})
    else:
        print(f"  ❌ FAIL — {reason}")
        _update_status(registry, candidate["version"], "evaluated_fail",
                       {"roc_auc": roc_auc})

    return passed


def _check_gate(roc_auc: float, champion: dict | None) -> tuple[bool, str]:
    """통과 기준을 계산하고 결과를 반환합니다."""
    if champion is None:
        # 첫 배포: baseline 비교
        threshold = BASELINE_THRESHOLD
        passed = roc_auc >= threshold
        reason = f"첫 배포 기준 {threshold:.2f} {'충족' if passed else '미달'} (현재={roc_auc:.4f})"
    else:
        # 기존 champion 비교
        champion_auc = champion["metrics"]["roc_auc"]
        passed = roc_auc >= champion_auc
        reason = (
            f"champion({champion['version']}) {champion_auc:.4f} 대비 "
            f"{'유지 또는 개선됨' if passed else '성능 하락'} (현재={roc_auc:.4f})"
        )
    return passed, reason


def _recompute_roc_auc(candidate: dict) -> float:
    """저장된 모델을 로드하고 test set에서 ROC-AUC를 재계산합니다."""
    with open(candidate["model_path"], "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    split = load_dataset(candidate["dataset_id"])

    if split.n_classes == 2:
        proba = model.predict_proba(split.X_test)[:, 1]
        return round(float(roc_auc_score(split.y_test, proba)), 4)
    else:
        proba = model.predict_proba(split.X_test)
        y_bin = label_binarize(split.y_test, classes=list(range(split.n_classes)))
        return round(float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")), 4)


def _get_candidate(registry: dict, version: str | None) -> dict | None:
    """평가할 후보 모델(trained 상태)을 반환합니다."""
    trained = [m for m in registry.get("models", []) if m["status"] == "trained"]
    if not trained:
        return None
    if version:
        matched = [m for m in trained if m["version"] == version]
        return matched[0] if matched else None
    # version 미지정: 가장 최근 trained
    return trained[-1]


def _get_deployed_champion(registry: dict, exclude_version: str) -> dict | None:
    """현재 deployed 상태인 챔피언을 반환합니다 (후보 제외)."""
    deployed = [
        m for m in registry.get("models", [])
        if m["status"] == "deployed" and m["version"] != exclude_version
    ]
    return deployed[-1] if deployed else None


def _update_status(registry: dict, version: str, status: str, extra_metrics: dict) -> None:
    for model in registry["models"]:
        if model["version"] == version:
            model["status"] = status
            model["metrics"].update(extra_metrics)
            break
    _save_registry(registry)


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
    success = evaluate(version)
    sys.exit(0 if success else 1)
