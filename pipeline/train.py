"""
pipeline/train.py — 다중 모델 학습 + Champion 선정

흐름:
  1. 지정한 dataset_id로 데이터 로드 (data/loader.py 경유, 경계 검증 포함)
  2. 등록된 모든 Challenger 모델을 학습
  3. ROC-AUC 기준으로 Champion 선정
  4. Champion 모델을 models/ 에 저장하고 registry에 기록
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from data.loader import DatasetSplit, load_dataset

MODELS_DIR = Path("models")
REGISTRY_PATH = Path("registry/model_registry.json")

CHALLENGERS: dict[str, object] = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
    "lightgbm": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

CHAMPION_METRIC = "roc_auc"


@dataclass
class ModelResult:
    model_name: str
    roc_auc: float
    fit_time_sec: float


def train(dataset_id: int) -> dict:
    """
    모든 Challenger를 학습하고 Champion을 선정한 뒤 registry에 기록합니다.

    Returns:
        dict: champion 정보 (model_name, roc_auc, model_path, version 포함)
    """
    MODELS_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.parent.mkdir(exist_ok=True)

    split = load_dataset(dataset_id)
    print(f"\n[train] dataset={split.name} | train={len(split.X_train)} | test={len(split.X_test)}")

    results = _train_all(split)
    _print_results(results)

    champion = max(results, key=lambda r: r.roc_auc)
    print(f"\n🏆 Champion: {champion.model_name}  ROC-AUC={champion.roc_auc:.4f}")

    model_path, version = _save_champion(champion.model_name, split)
    entry = _register(dataset_id, split.name, champion, model_path, version)
    return entry


def _train_all(split: DatasetSplit) -> list[ModelResult]:
    results = []
    for name, model in CHALLENGERS.items():
        t0 = time.time()
        model.fit(split.X_train, split.y_train)
        fit_time = time.time() - t0

        auc = _compute_roc_auc(model, split)
        results.append(ModelResult(model_name=name, roc_auc=auc, fit_time_sec=round(fit_time, 2)))
        print(f"  {name:<30} ROC-AUC={auc:.4f}  ({fit_time:.1f}s)")

    return results


def _compute_roc_auc(model: object, split: DatasetSplit) -> float:
    """이진/다중 클래스 모두 처리."""
    if split.n_classes == 2:
        proba = model.predict_proba(split.X_test)[:, 1]
        return round(float(roc_auc_score(split.y_test, proba)), 4)
    else:
        proba = model.predict_proba(split.X_test)
        y_bin = label_binarize(split.y_test, classes=list(range(split.n_classes)))
        return round(float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")), 4)


def _save_champion(model_name: str, split: DatasetSplit) -> tuple[str, str]:
    """Champion 모델을 재학습(전체 데이터)하고 저장."""
    import pandas as pd
    X_full = pd.concat([split.X_train, split.X_test], ignore_index=True)
    y_full = pd.concat([split.y_train, split.y_test], ignore_index=True)

    model = _clone_challenger(model_name)
    model.fit(X_full, y_full)

    version = f"v{_next_version()}"
    model_path = str(MODELS_DIR / f"champion_{split.name}_{version}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "model_name": model_name, "dataset_name": split.name}, f)

    return model_path, version


def _clone_challenger(model_name: str) -> object:
    """CHALLENGERS에서 동일 설정의 새 인스턴스를 반환."""
    from sklearn.base import clone
    return clone(CHALLENGERS[model_name])


def _next_version() -> str:
    registry = _load_registry()
    existing = [e.get("version", "v0") for e in registry.get("models", [])]
    nums = [int(v.lstrip("v")) for v in existing if v.lstrip("v").isdigit()]
    return str(max(nums, default=0) + 1)


def _register(dataset_id: int, dataset_name: str, champion: ModelResult, model_path: str, version: str) -> dict:
    registry = _load_registry()
    entry = {
        "version": version,
        "status": "trained",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "model_name": champion.model_name,
        "metrics": {CHAMPION_METRIC: champion.roc_auc},
        "fit_time_sec": champion.fit_time_sec,
        "model_path": model_path,
        "trained_at": datetime.now().isoformat(),
    }
    registry.setdefault("models", []).append(entry)
    _save_registry(registry)
    return entry


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _print_results(results: list[ModelResult]) -> None:
    print(f"\n{'Model':<30} {'ROC-AUC':>8}  {'Time':>6}")
    print("-" * 48)
    for r in sorted(results, key=lambda x: x.roc_auc, reverse=True):
        print(f"  {r.model_name:<28} {r.roc_auc:>8.4f}  {r.fit_time_sec:>5.1f}s")


if __name__ == "__main__":
    import sys
    dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 44089
    result = train(dataset_id)
    print(f"\nRegistry entry: {json.dumps(result, indent=2)}")
