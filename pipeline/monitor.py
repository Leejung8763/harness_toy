"""
pipeline/monitor.py — 배포 후 성능 모니터링 + 자동 롤백 + Trigger

규칙:
  - deployed 모델의 성능을 주기적으로 확인
  - config/thresholds.json의 임계값 위반 시 자동 롤백
  - 드리프트 감지: PSI(분포 변화) + KS-test(통계 검정) + 성능 지표
  - 롤백: 현재 모델 → 'rolled_back', flags.json → 이전 버전으로 복원
  - 이전 버전이 없으면 → flags.json 비활성화
  - 롤백 후 Automated Pipeline 서버에 재학습 요청 (Trigger)
  - 각 라운드 메트릭은 ml_metadata에 기록
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from data.loader import load_dataset
from ml_metadata import store as metadata_store
from pipeline.predict import _load_active_model, _load_flags, _load_registry

REGISTRY_PATH = Path("registry/model_registry.json")
FLAGS_PATH = Path("feature_flags/flags.json")
PIPELINE_SERVER_URL = "http://localhost:8001"
_CONFIG_PATH = Path("config/thresholds.json")

MONITOR_ROUNDS = 5


def _load_thresholds() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f).get("monitor", {})
    return {}


def _get_cv_thresholds() -> dict:
    t = _load_thresholds()
    return {
        "roc_auc": t.get("roc_auc_min", 0.75),
        "error_rate": t.get("error_rate_max", 0.25),
        "psi": t.get("psi_threshold", 0.20),
        "ks_pvalue": t.get("ks_pvalue_min", 0.05),
    }


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

    thresholds = _get_cv_thresholds()
    print(f"\n[monitor] 모니터링 시작 — {version}  ({rounds} 라운드)")
    print(f"  임계값: ROC-AUC≥{thresholds['roc_auc']}  ErrorRate≤{thresholds['error_rate']}  PSI≤{thresholds['psi']}")
    if inject_drift:
        print("  ⚠️  드리프트 주입 모드 ON")

    model, entry = _load_active_model()
    split = load_dataset(entry["dataset_id"])

    # 베이스라인 분포: 학습 데이터 전체
    baseline_X = split.X_train

    print(f"\n  {'Round':<8} {'ROC-AUC':>8}  {'ErrRate':>8}  {'PSI':>7}  {'KS-p':>7}  {'상태':>4}")
    print("  " + "-" * 58)

    for round_num in range(1, rounds + 1):
        metrics = _evaluate_round(model, split, baseline_X, round_num, inject_drift)

        violation = _check_thresholds(metrics, thresholds)
        status = "❌" if violation else "✅"

        print(
            f"  {round_num:<8} {metrics['roc_auc']:>8.4f}  {metrics['error_rate']:>8.4f}"
            f"  {metrics['psi_mean']:>7.4f}  {metrics['ks_pvalue_min']:>7.4f}  {status}"
        )

        # 라운드 메트릭 기록
        metadata_store.log_monitoring({
            "model_version": version,
            "round": round_num,
            **metrics,
            "violation": violation,
            "inject_drift": inject_drift,
        })

        if violation:
            print(f"\n  🚨 임계값 위반: {violation}")
            # operations_agent로 원인 분석 + 액션 결정
            try:
                from agents.operations_agent import decide_action
                decide_action(
                    model_version=version,
                    feature_names=list(split.X_train.columns),
                    psi_per_feature=metrics.get("psi_per_feature", {}),
                    ks_pvalue_per_feature=metrics.get("ks_pvalue_per_feature", {}),
                    metrics=metrics,
                )
            except Exception as e:
                print(f"  ⚠️  operations_agent 분석 실패: {e}")
            _rollback(version)
            _trigger_retraining(entry["dataset_id"])
            # 배포 결정 회고 업데이트 — 드리프트 감지
            try:
                from agents.eval.decision_logger import update_deploy_outcome
                update_deploy_outcome(version, "drift_detected", monitor_result=metrics)
            except Exception:
                pass
            return False

        time.sleep(0.3)

    print(f"\n  ✅ {rounds} 라운드 정상 완료 — {version} 유지")
    # 배포 결정 회고 업데이트 — 정상 운영
    try:
        from agents.eval.decision_logger import update_deploy_outcome
        update_deploy_outcome(version, "survived")
    except Exception:
        pass
    return True


def _evaluate_round(
    model: object,
    split,
    baseline_X,
    round_num: int,
    inject_drift: bool,
) -> dict:
    """한 라운드의 성능 + 분포 변화를 평가합니다."""
    rng = np.random.default_rng(seed=round_num * 42)

    n = len(split.X_test)
    idx = rng.choice(n, size=min(500, n), replace=False)
    X_sample = split.X_test.iloc[idx].copy()
    y_sample = split.y_test.iloc[idx]

    if inject_drift:
        # 드리프트 시뮬레이션: 라운드마다 노이즈 강도 증가
        noise_scale = round_num * 2.0
        X_sample = X_sample + rng.normal(0, noise_scale, X_sample.shape).astype("float32")

    # ── 성능 지표 ─────────────────────────────────────────────────
    if split.n_classes == 2:
        proba = model.predict_proba(X_sample)[:, 1]
        roc_auc = float(roc_auc_score(y_sample, proba))
    else:
        proba = model.predict_proba(X_sample)
        y_bin = label_binarize(y_sample, classes=list(range(split.n_classes)))
        roc_auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))

    preds = model.predict(X_sample)
    error_rate = float((preds != y_sample.values).mean())

    # ── 분포 변화 감지 ────────────────────────────────────────────
    psi_values, ks_pvalues = [], []
    baseline_sample = baseline_X.sample(min(500, len(baseline_X)), random_state=round_num)

    for col in X_sample.columns:
        psi_values.append(_compute_psi(baseline_sample[col].values, X_sample[col].values))
        _, p = stats.ks_2samp(baseline_sample[col].values, X_sample[col].values)
        ks_pvalues.append(float(p))

    return {
        "roc_auc": round(roc_auc, 4),
        "error_rate": round(error_rate, 4),
        "psi_mean": round(float(np.mean(psi_values)), 4),
        "psi_max": round(float(np.max(psi_values)), 4),
        "ks_pvalue_min": round(float(np.min(ks_pvalues)), 4),
        "psi_per_feature": {col: round(psi, 4) for col, psi in zip(X_sample.columns, psi_values)},
        "ks_pvalue_per_feature": {col: round(p, 4) for col, p in zip(X_sample.columns, ks_pvalues)},
    }


def _compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI(Population Stability Index)를 계산합니다.

    PSI < 0.1  : 변화 없음
    PSI 0.1~0.2: 중간 변화 (모니터링 강화)
    PSI > 0.2  : 유의미한 분포 변화 (드리프트)
    """
    eps = 1e-8
    breakpoints = np.linspace(
        min(baseline.min(), current.min()),
        max(baseline.max(), current.max()),
        bins + 1,
    )
    base_counts = np.histogram(baseline, bins=breakpoints)[0] + eps
    curr_counts = np.histogram(current, bins=breakpoints)[0] + eps
    base_pct = base_counts / base_counts.sum()
    curr_pct = curr_counts / curr_counts.sum()
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def _check_thresholds(metrics: dict, thresholds: dict) -> str | None:
    """임계값 위반 항목을 반환합니다. 정상이면 None."""
    if metrics["roc_auc"] < thresholds["roc_auc"]:
        return f"roc_auc={metrics['roc_auc']:.4f} < threshold={thresholds['roc_auc']}"
    if metrics["error_rate"] > thresholds["error_rate"]:
        return f"error_rate={metrics['error_rate']:.4f} > threshold={thresholds['error_rate']}"
    if metrics["psi_mean"] > thresholds["psi"]:
        return f"psi_mean={metrics['psi_mean']:.4f} > threshold={thresholds['psi']} (분포 변화 감지)"
    if metrics["ks_pvalue_min"] < thresholds["ks_pvalue"]:
        return f"ks_pvalue_min={metrics['ks_pvalue_min']:.4f} < threshold={thresholds['ks_pvalue']} (KS-test 유의)"
    return None


def _trigger_retraining(dataset_id: int) -> None:
    """
    Automated Pipeline 서버에 재학습을 요청합니다. (Operations → Trigger)

    서버가 실행 중이면 HTTP POST, 아니면 경고만 출력합니다.
    """
    try:
        import httpx
        r = httpx.post(
            f"{PIPELINE_SERVER_URL}/pipeline/run",
            json={"dataset_id": dataset_id},
            timeout=5.0,
        )
        if r.status_code == 202:
            run_id = r.json().get("run_id", "?")
            print(f"  🔄 Trigger → Automated Pipeline  run_id={run_id}")
        elif r.status_code == 409:
            print(f"  ⚠️  파이프라인 이미 실행 중 (재학습 요청 스킵)")
        else:
            print(f"  ⚠️  재학습 요청 실패 (HTTP {r.status_code})")
    except Exception:
        print(f"  ⚠️  Automated Pipeline 서버 미연결 — 수동 재실행 필요")
        print(f"       ./ci/start_pipeline_server.sh 후 ./ci/trigger_pipeline.sh")


def _rollback(current_version: str) -> None:
    """이전 버전이 있으면 롤백, 없으면 현재 모델 유지(degraded mode)."""
    registry = _load_registry()

    # 이전 retired 모델 중 가장 최근 것을 복원
    retired = [m for m in registry["models"] if m["status"] == "retired"]
    flags = _load_flags()

    if retired:
        # 이전 모델 존재 → 롤백
        for m in registry["models"]:
            if m["version"] == current_version:
                m["status"] = "rolled_back"
                break
        prev = retired[-1]
        prev["status"] = "deployed"
        flags["active_model_version"] = prev["version"]
        flags["active_dataset_id"] = prev["dataset_id"]
        print(f"  ↩️  롤백 완료: {current_version} → rolled_back, "
              f"{prev['version']} → deployed")
    else:
        # 이전 모델 없음 → 현재 모델 유지(서비스 중단 방지) + 재학습으로 교체 예정
        print(f"  ⚠️  복원할 이전 버전 없음 — {current_version} 유지 (degraded mode)")
        print(f"      재학습 완료 후 새 모델로 교체됩니다.")

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
