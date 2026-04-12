"""
pipeline/ab_test.py — A/B 테스트 실행 및 승격 판단

역할:
  - 챔피언(deployed)과 챌린저(challenger) 모델을 동시에 평가
  - 시뮬레이션 라운드별 ROC-AUC 수집
  - 독립 t-test로 통계적 유의성 검정 → 승격/거부 결정

하네스 역할:
  - 통계 검정은 rule-based (LLM 없음)
  - 결과와 무관하게 champion은 100% 유지, 챌린저만 교체

사용:
  python3 pipeline/ab_test.py  (직접 실행 시 현재 challenger 테스트)
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
from scipy import stats

FLAGS_PATH = Path("feature_flags/flags.json")
REGISTRY_PATH = Path("registry/model_registry.json")

DEFAULT_ROUNDS = 10
SIGNIFICANCE_LEVEL = 0.05          # p-value 임계값
MIN_IMPROVEMENT = 0.001            # 챌린저가 최소 이 이상 좋아야 승격


def run_ab_test(
    challenger_version: str | None = None,
    rounds: int = DEFAULT_ROUNDS,
) -> dict:
    """
    A/B 테스트를 실행하고 승격/거부를 결정합니다.

    Args:
        challenger_version: 테스트할 챌린저 버전 (None이면 flags에서 읽음)
        rounds: 평가 라운드 수

    Returns:
        {
            "winner": "challenger" | "champion" | "inconclusive",
            "promote": bool,
            "p_value": float,
            "champion_mean_auc": float,
            "challenger_mean_auc": float,
            "champion_version": str,
            "challenger_version": str,
            "rounds": int,
        }
    """
    flags = _load_flags()
    challenger_version = challenger_version or flags.get("challenger_model_version")

    if not challenger_version:
        print("[ab_test] ⚠️  챌린저 모델이 없습니다.")
        return {"winner": "champion", "promote": False, "p_value": None,
                "champion_mean_auc": None, "challenger_mean_auc": None,
                "champion_version": flags.get("active_model_version"),
                "challenger_version": None, "rounds": 0}

    champion_version = flags.get("active_model_version")
    champion_model, champion_entry = _load_model(champion_version)
    challenger_model, challenger_entry = _load_model(challenger_version)

    # Feature Store에서 테스트셋 로드
    split = _load_test_split(champion_entry["dataset_id"])

    print(f"\n[ab_test] A/B 테스트 시작")
    print(f"  챔피언: {champion_version}  vs  챌린저: {challenger_version}")
    print(f"  라운드: {rounds}  유의수준: {SIGNIFICANCE_LEVEL}")
    print(f"\n  {'Round':<8} {'Champion AUC':>13} {'Challenger AUC':>15}")
    print("  " + "-" * 40)

    champion_aucs = []
    challenger_aucs = []

    for r in range(1, rounds + 1):
        # 매 라운드마다 테스트셋에서 bootstrap 샘플링
        idx = np.random.choice(len(split["X_test"]), size=len(split["X_test"]), replace=True)
        X_sample = split["X_test"].iloc[idx]
        y_sample = split["y_test"].iloc[idx]

        ch_auc = _compute_roc_auc(champion_model, X_sample, y_sample)
        cr_auc = _compute_roc_auc(challenger_model, X_sample, y_sample)

        champion_aucs.append(ch_auc)
        challenger_aucs.append(cr_auc)
        print(f"  {r:<8} {ch_auc:>13.4f} {cr_auc:>15.4f}")

    result = _decide_winner(
        champion_aucs, challenger_aucs,
        champion_version, challenger_version,
        rounds,
    )

    _print_result(result)
    _log_result(result)
    return result


def promote_challenger(challenger_version: str) -> None:
    """챌린저를 챔피언으로 승격합니다 (registry + flags 업데이트)."""
    registry = _load_registry()

    # 기존 champion → retired
    for m in registry["models"]:
        if m["status"] == "deployed":
            m["status"] = "retired"
            print(f"  {m['version']} → retired")

    # 챌린저 → deployed
    challenger_entry = next(m for m in registry["models"] if m["version"] == challenger_version)
    challenger_entry["status"] = "deployed"
    _save_registry(registry)
    print(f"  {challenger_version} → deployed ✅ (A/B 테스트 승리)")

    # flags 업데이트 — 챌린저 제거, 챔피언 교체
    flags = _load_flags()
    flags["active_model_version"] = challenger_version
    flags.pop("challenger_model_version", None)
    flags.pop("challenger_traffic_weight", None)
    _save_flags(flags)
    print(f"  flags.json → active_model_version={challenger_version}, challenger 제거")


def reject_challenger(challenger_version: str) -> None:
    """챌린저를 거부합니다 (registry + flags에서 제거)."""
    registry = _load_registry()
    for m in registry["models"]:
        if m["version"] == challenger_version:
            m["status"] = "rejected"
            print(f"  {challenger_version} → rejected")
    _save_registry(registry)

    flags = _load_flags()
    flags.pop("challenger_model_version", None)
    flags.pop("challenger_traffic_weight", None)
    _save_flags(flags)
    print(f"  flags.json → challenger 제거, {flags.get('active_model_version')} 유지")


# ── 내부 함수 ─────────────────────────────────────────────────────────────────

def _decide_winner(
    champion_aucs: list[float],
    challenger_aucs: list[float],
    champion_version: str,
    challenger_version: str,
    rounds: int,
) -> dict:
    ch_mean = float(np.mean(champion_aucs))
    cr_mean = float(np.mean(challenger_aucs))

    # Welch's t-test (등분산 가정 없음)
    t_stat, p_value = stats.ttest_ind(challenger_aucs, champion_aucs, equal_var=False)

    challenger_wins = (
        p_value < SIGNIFICANCE_LEVEL        # 통계적으로 유의
        and cr_mean > ch_mean + MIN_IMPROVEMENT  # 최소 개선 폭 충족
    )

    if challenger_wins:
        winner = "challenger"
    elif p_value < SIGNIFICANCE_LEVEL and ch_mean > cr_mean + MIN_IMPROVEMENT:
        winner = "champion"
    else:
        winner = "inconclusive"

    return {
        "winner": winner,
        "promote": challenger_wins,
        "p_value": round(float(p_value), 6),
        "t_stat": round(float(t_stat), 4),
        "champion_mean_auc": round(ch_mean, 4),
        "challenger_mean_auc": round(cr_mean, 4),
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "rounds": rounds,
    }


def _compute_roc_auc(model: object, X: object, y: object) -> float:
    from sklearn.metrics import roc_auc_score
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def _load_test_split(dataset_id: int) -> dict:
    """Feature Store에서 테스트셋을 로드합니다."""
    from feature_store.store import load as fs_load
    split = fs_load(dataset_id)
    return {"X_test": split.X_test, "y_test": split.y_test}


def _load_model(version: str) -> tuple[object, dict]:
    registry = _load_registry()
    entry = next((m for m in registry["models"] if m["version"] == version), None)
    if entry is None:
        raise ValueError(f"registry에 {version}이 없습니다.")
    with open(entry["model_path"], "rb") as f:
        saved = pickle.load(f)
    return saved["model"], entry


def _print_result(result: dict) -> None:
    winner_str = {
        "challenger": f"🏆 챌린저 승리 ({result['challenger_version']})",
        "champion":   f"🛡️  챔피언 유지 ({result['champion_version']})",
        "inconclusive": "🤝 무승부 (챔피언 유지)",
    }[result["winner"]]

    print(f"\n  결과: {winner_str}")
    print(f"  챔피언  평균 AUC: {result['champion_mean_auc']:.4f}")
    print(f"  챌린저  평균 AUC: {result['challenger_mean_auc']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}  (임계값: {SIGNIFICANCE_LEVEL})")
    promote_str = "→ 챌린저 승격" if result["promote"] else "→ 챔피언 유지"
    print(f"  판정: {promote_str}")


def _log_result(result: dict) -> None:
    try:
        from ml_metadata import store as metadata_store
        metadata_store.log_monitoring({
            "stage": "ab_test",
            **result,
        })
    except Exception:
        pass


def _load_flags() -> dict:
    return json.loads(FLAGS_PATH.read_text()) if FLAGS_PATH.exists() else {}


def _save_flags(flags: dict) -> None:
    with open(FLAGS_PATH, "w") as f:
        json.dump(flags, f, indent=2)


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"models": []}


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


if __name__ == "__main__":
    result = run_ab_test()
    if result["promote"]:
        promote_challenger(result["challenger_version"])
    else:
        reject_challenger(result["challenger_version"])
