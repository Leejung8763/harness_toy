"""
pipeline/predict.py — Feature Flag 기반 예측

규칙:
  - 어떤 모델을 쓸지는 feature_flags/flags.json 에서만 읽는다
  - 모델 경로, 버전을 코드에 하드코딩하지 않는다
  - 입력 데이터는 data/loader.py와 동일한 전처리(float32, NaN 처리)를 거쳐야 한다
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FLAGS_PATH = Path("feature_flags/flags.json")
REGISTRY_PATH = Path("registry/model_registry.json")


def predict(X: pd.DataFrame) -> np.ndarray:
    """
    Feature Flag 기반으로 챔피언 또는 챌린저 모델로 예측합니다.
    challenger_traffic_weight 확률로 챌린저 모델을 사용합니다.
    """
    model, _ = _load_active_model()
    X = _validate_input(X)
    return model.predict(X).astype(np.int32)


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    """
    Feature Flag 기반으로 챔피언 또는 챌린저 모델로 확률을 반환합니다.
    """
    model, _ = _load_active_model()
    X = _validate_input(X)
    return model.predict_proba(X)


def get_active_model_info() -> dict:
    """현재 서빙 중인 모델 정보를 반환합니다."""
    flags = _load_flags()
    version = flags.get("active_model_version")
    if version is None:
        return {"status": "no model deployed"}

    registry = _load_registry()
    for m in registry.get("models", []):
        if m["version"] == version:
            return {
                "version": m["version"],
                "model_name": m["model_name"],
                "dataset_name": m["dataset_name"],
                "roc_auc": m["metrics"].get("roc_auc"),
                "model_path": m["model_path"],
            }
    return {"status": "version not found in registry", "version": version}


def _load_active_model() -> tuple[object, dict]:
    """flags.json → A/B 트래픽 분기 → registry → pkl 순서로 모델을 로드합니다."""
    import random
    flags = _load_flags()
    version = flags.get("active_model_version")

    if version is None:
        raise RuntimeError(
            "배포된 모델이 없습니다. pipeline/deploy.py 를 먼저 실행하세요."
        )

    # A/B 트래픽 분기: challenger가 있으면 확률적으로 선택
    challenger = flags.get("challenger_model_version")
    weight = float(flags.get("challenger_traffic_weight", 0.0))
    if challenger and weight > 0 and random.random() < weight:
        version = challenger

    registry = _load_registry()
    entry = next((m for m in registry.get("models", []) if m["version"] == version), None)

    if entry is None:
        raise RuntimeError(f"flags.json의 버전 {version}이 registry에 없습니다.")

    model_path = Path(entry["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    return saved["model"], entry


def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
    """입력 데이터를 float32로 변환하고 NaN을 중앙값으로 대체합니다."""
    X = X.copy()
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(np.float32)


def _load_flags() -> dict:
    if FLAGS_PATH.exists():
        with open(FLAGS_PATH) as f:
            return json.load(f)
    return {"active_model_version": None}


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


if __name__ == "__main__":
    # 현재 deployed 모델 정보 출력 + 샘플 예측 시연
    from data.loader import load_dataset

    info = get_active_model_info()
    print(f"\n[predict] 서빙 중인 모델: {info}")

    flags = _load_flags()
    dataset_id = flags.get("active_dataset_id")
    if dataset_id is None:
        print("active_dataset_id 가 flags.json에 없습니다.")
    else:
        split = load_dataset(dataset_id)
        sample = split.X_test.iloc[:5]

        preds = predict(sample)
        probas = predict_proba(sample)

        print(f"\n샘플 5건 예측 결과:")
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            confidence = max(proba)
            print(f"  [{i}] class={pred}  confidence={confidence:.3f}  proba={proba.round(3)}")
