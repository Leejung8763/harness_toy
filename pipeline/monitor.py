"""
pipeline/monitor.py — 배포 후 성능 모니터링 + 자동 롤백

규칙:
  - deployed 모델의 성능을 주기적으로 확인 (시뮬레이션)
  - CV_THRESHOLDS 위반 시 사람 개입 없이 자동 롤백
  - 롤백: 현재 모델 → 'rolled_back', flags.json → 이전 버전으로 복원
  - 이전 버전이 없으면 → flags.json 비활성화
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from data.loader import load_dataset
from pipeline.predict import _load_active_model, _load_flags, _load_registry

REGISTRY_PATH = Path("registry/model_registry.json")
FLAGS_PATH = Path("feature_flags/flags.json")

CV_THRESHOLDS = {
    "roc_auc": 0.75,      # 배포 시 기준보다 낮게 설정 (운영 중 허용 하한)
    "error_rate": 0.25,   # 예측 오류율 상한 (credit 데이터셋 기준 ~0.20)
}

MONITOR_ROUNDS = 5


def monitor(rounds: int = MONITOR_ROUNDS, inject_drift: bool = False) -> bool:
    """
    deployed 모델을 N 라운드 동안 모니터링합니다.

    Args:
        rounds: 모니터링 반복 횟수
        inject_drift: True면 드리프트를 인위적으로 주입 (롤백 시나리오 테스트용)

    Returns:
        bool: True(정상 완료) / False(롤백 발생)
    """
    flags = _load_flags()
    version = flags.get("active_model_version")

    if version is None:
        print("[monitor] ❌ 배포된 모델이 없습니다.")
        return False

    print(f"\n[monitor] 모니터링 시작 — {version}  ({rounds} 라운드)")
    if inject_drift:
        print("  ⚠️  드리프트 주입 모드 ON")

    model, entry = _load_active_model()
    split = load_dataset(entry["dataset_id"])

    print(f"\n  {'Round':<8} {'ROC-AUC':>8}  {'ErrorRate':>10}  {'상태':>6}")
    print("  " + "-" * 42)

    for round_num in range(1, rounds + 1):
        metrics = _evaluate_round(model, split, round_num, inject_drift)

        status = "✅"
        violation = _check_thresholds(metrics)
        if violation:
            status = "❌"
            print(f"  {round_num:<8} {metrics['roc_auc']:>8.4f}  {metrics['error_rate']:>10.4f}  {status}")
            print(f"\n  🚨 임계값 위반: {violation}")
            _rollback(version)
            return False

        print(f"  {round_num:<8} {metrics['roc_auc']:>8.4f}  {metrics['error_rate']:>10.4f}  {status}")
        time.sleep(0.3)  # 라운드 간 간격 시뮬레이션

    print(f"\n  ✅ {rounds} 라운드 정상 완료 — {version} 유지")
    return True


def _evaluate_round(
    model: object,
    split,
    round_num: int,
    inject_drift: bool,
) -> dict:
    """한 라운드의 성능을 평가합니다 (실제 데이터 샘플링으로 시뮬레이션)."""
    rng = np.random.default_rng(seed=round_num * 42)

    n = len(split.X_test)
    idx = rng.choice(n, size=min(500, n), replace=False)
    X_sample = split.X_test.iloc[idx]
    y_sample = split.y_test.iloc[idx]

    if inject_drift:
        # 드리프트 시뮬레이션: 노이즈를 점점 강하게 추가
        noise_scale = round_num * 2.0
        X_sample = X_sample + rng.normal(0, noise_scale, X_sample.shape).astype("float32")

    if split.n_classes == 2:
        proba = model.predict_proba(X_sample)[:, 1]
        roc_auc = float(roc_auc_score(y_sample, proba))
    else:
        proba = model.predict_proba(X_sample)
        y_bin = label_binarize(y_sample, classes=list(range(split.n_classes)))
        roc_auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))

    preds = model.predict(X_sample)
    error_rate = float((preds != y_sample.values).mean())

    return {"roc_auc": round(roc_auc, 4), "error_rate": round(error_rate, 4)}


def _check_thresholds(metrics: dict) -> str | None:
    """임계값 위반 항목을 반환합니다. 정상이면 None."""
    if metrics["roc_auc"] < CV_THRESHOLDS["roc_auc"]:
        return (f"roc_auc={metrics['roc_auc']:.4f} < "
                f"threshold={CV_THRESHOLDS['roc_auc']}")
    if metrics["error_rate"] > CV_THRESHOLDS["error_rate"]:
        return (f"error_rate={metrics['error_rate']:.4f} > "
                f"threshold={CV_THRESHOLDS['error_rate']}")
    return None


def _rollback(current_version: str) -> None:
    """현재 모델을 rolled_back으로 변경하고 이전 버전을 복원합니다."""
    registry = _load_registry()

    # 현재 모델 → rolled_back
    for m in registry["models"]:
        if m["version"] == current_version:
            m["status"] = "rolled_back"
            break

    # 이전 retired 모델 중 가장 최근 것을 복원
    retired = [m for m in registry["models"] if m["status"] == "retired"]
    flags = _load_flags()

    if retired:
        prev = retired[-1]
        prev["status"] = "deployed"
        flags["active_model_version"] = prev["version"]
        flags["active_dataset_id"] = prev["dataset_id"]
        print(f"  ↩️  롤백 완료: {current_version} → rolled_back, "
              f"{prev['version']} → deployed")
    else:
        flags["active_model_version"] = None
        flags["active_dataset_id"] = None
        print(f"  ↩️  롤백 완료: {current_version} → rolled_back  "
              f"(복원할 이전 버전 없음, 서비스 중단)")

    _save_registry(registry)
    _save_flags(flags)


def _save_flags(flags: dict) -> None:
    FLAGS_PATH.parent.mkdir(exist_ok=True)
    with open(FLAGS_PATH, "w") as f:
        json.dump(flags, f, indent=2)


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


if __name__ == "__main__":
    import sys

    inject_drift = "--drift" in sys.argv
    rounds = int(next((a for a in sys.argv[1:] if a.isdigit()), MONITOR_ROUNDS))
    success = monitor(rounds=rounds, inject_drift=inject_drift)
    sys.exit(0 if success else 1)
