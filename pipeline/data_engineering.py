"""
pipeline/data_engineering.py — Data Engineering Stage

흐름:
  1. data/loader.py에서 원시 데이터 로드 (경계 검증 포함)
  2. 피처 엔지니어링 적용
       - 분산 0 피처 제거 (train set 기준)
  3. Feature Store에 저장
  4. ML Metadata Store에 실행 기록

규칙:
  - 피처 변환은 train set 기준으로 fit, test set에는 transform만 적용
  - Feature Store에 이미 해당 버전이 있으면 스킵 (멱등성)
  - force=True 로 강제 재실행 가능
"""

from __future__ import annotations

import json
from datetime import datetime

import feature_store.store as fs
import ml_metadata.store as mds
from data.loader import DatasetSplit, load_dataset

FEATURE_VERSION = "v1"


def run_data_engineering(dataset_id: int, force: bool = False) -> dict:
    """
    원시 데이터를 로드하고 피처 엔지니어링 후 Feature Store에 저장합니다.

    Args:
        dataset_id: OpenML 데이터셋 ID
        force: True면 Feature Store에 있어도 재실행

    Returns:
        dict: 실행 메타데이터 (dataset_id, n_features, version 등)
    """
    if not force and fs.exists(dataset_id, FEATURE_VERSION):
        print(f"[data_engineering] ✅ Feature Store 히트 — "
              f"dataset_id={dataset_id} {FEATURE_VERSION} (스킵)")
        meta = fs.list_features()
        cached = next((m for m in meta if m["dataset_id"] == dataset_id), {})
        return {**cached, "skipped": True}

    print(f"[data_engineering] 원시 데이터 로드 — dataset_id={dataset_id}")
    split = load_dataset(dataset_id)
    n_raw = len(split.feature_names)
    print(f"  원시 피처={n_raw}  train={len(split.X_train)}  test={len(split.X_test)}")

    # 피처 엔지니어링
    split = _remove_zero_variance(split)
    n_engineered = len(split.feature_names)
    print(f"  엔지니어링 후 피처={n_engineered}  (분산=0 제거: {n_raw - n_engineered}개)")

    # Feature Store 저장
    store_path = fs.save(split, FEATURE_VERSION)
    print(f"  Feature Store 저장 → {store_path}")

    meta = {
        "dataset_id": dataset_id,
        "dataset_name": split.name,
        "feature_version": FEATURE_VERSION,
        "n_features_raw": n_raw,
        "n_features_engineered": n_engineered,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "skipped": False,
        "created_at": datetime.now().isoformat(),
    }

    # ML Metadata Store 기록
    mds.log_data_engineering(meta)

    return meta


def _remove_zero_variance(split: DatasetSplit) -> DatasetSplit:
    """분산이 0인 피처를 제거합니다 (train set 기준으로 판단)."""
    variances = split.X_train.var()
    keep_cols = variances[variances > 0].index.tolist()
    return DatasetSplit(
        dataset_id=split.dataset_id,
        name=split.name,
        X_train=split.X_train[keep_cols],
        X_test=split.X_test[keep_cols],
        y_train=split.y_train,
        y_test=split.y_test,
        feature_names=keep_cols,
        n_classes=split.n_classes,
    )


if __name__ == "__main__":
    import sys
    dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 44089
    result = run_data_engineering(dataset_id)
    print(f"\n결과: {json.dumps(result, indent=2, default=str)}")
