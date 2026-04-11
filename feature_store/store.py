"""
feature_store/store.py — Feature Store

역할:
  - Data Engineering 결과물(전처리된 피처)을 로컬에 저장·로드
  - 동일 dataset_id의 피처를 재사용해 반복 다운로드/전처리 방지
  - pipeline/train.py는 data/loader.py 대신 Feature Store에서 읽음

저장 형식: Parquet (컬럼 타입 보존, 압축 효율)
경로: feature_store/features/{dataset_id}/{version}/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.loader import DatasetSplit

STORE_DIR = Path("feature_store/features")


def save(split: DatasetSplit, version: str = "v1") -> Path:
    """DatasetSplit을 Feature Store에 저장합니다."""
    store_path = _store_path(split.dataset_id, version)
    store_path.mkdir(parents=True, exist_ok=True)

    split.X_train.to_parquet(store_path / "X_train.parquet", index=False)
    split.X_test.to_parquet(store_path / "X_test.parquet", index=False)
    split.y_train.to_frame().to_parquet(store_path / "y_train.parquet", index=False)
    split.y_test.to_frame().to_parquet(store_path / "y_test.parquet", index=False)

    meta = {
        "dataset_id": split.dataset_id,
        "name": split.name,
        "feature_names": split.feature_names,
        "n_classes": split.n_classes,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "version": version,
        "created_at": datetime.now().isoformat(),
    }
    with open(store_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return store_path


def load(dataset_id: int, version: str = "v1") -> DatasetSplit:
    """Feature Store에서 DatasetSplit을 로드합니다."""
    store_path = _store_path(dataset_id, version)
    if not exists(dataset_id, version):
        raise FileNotFoundError(
            f"Feature Store에 dataset_id={dataset_id} version={version}이 없습니다. "
            "pipeline/data_engineering.py를 먼저 실행하세요."
        )

    with open(store_path / "meta.json") as f:
        meta = json.load(f)

    X_train = pd.read_parquet(store_path / "X_train.parquet")
    X_test = pd.read_parquet(store_path / "X_test.parquet")
    y_col = meta["feature_names"][0] if meta["feature_names"] else 0
    y_train = pd.read_parquet(store_path / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(store_path / "y_test.parquet").iloc[:, 0]

    return DatasetSplit(
        dataset_id=meta["dataset_id"],
        name=meta["name"],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=meta["feature_names"],
        n_classes=meta["n_classes"],
    )


def exists(dataset_id: int, version: str = "v1") -> bool:
    """Feature Store에 해당 데이터셋/버전이 저장되어 있는지 확인합니다."""
    return (_store_path(dataset_id, version) / "meta.json").exists()


def list_features() -> list[dict]:
    """저장된 모든 피처 목록을 반환합니다."""
    if not STORE_DIR.exists():
        return []
    result = []
    for meta_path in sorted(STORE_DIR.glob("*/*/meta.json")):
        with open(meta_path) as f:
            result.append(json.load(f))
    return result


def _store_path(dataset_id: int, version: str) -> Path:
    return STORE_DIR / str(dataset_id) / version
