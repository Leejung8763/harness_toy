"""
data/loader.py — Grinsztajn NeurIPS 2022 Numerical Classification 데이터 로더

경계(Boundary) 규칙:
  - 이 모듈이 시스템으로 데이터가 들어오는 유일한 진입점입니다.
  - 외부 데이터(OpenML)는 여기서 반드시 검증·정제됩니다.
  - pipeline/ 이하 코드는 이 모듈의 반환값만 신뢰합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SUITE_ID = 337  # Grinsztajn 2022 — Numerical Classification


@dataclass
class DatasetInfo:
    dataset_id: int
    name: str
    n_samples: int
    n_features: int
    n_classes: int


@dataclass
class DatasetSplit:
    dataset_id: int
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    n_classes: int


def list_datasets() -> list[DatasetInfo]:
    """Suite 337의 모든 데이터셋 메타데이터를 반환합니다."""
    suite = openml.study.get_suite(SUITE_ID)
    result = []
    for did in suite.data:
        ds = openml.datasets.get_dataset(did, download_data=False)
        result.append(DatasetInfo(
            dataset_id=did,
            name=ds.name,
            n_samples=int(ds.qualities.get("NumberOfInstances", 0)),
            n_features=int(ds.qualities.get("NumberOfFeatures", 0)) - 1,
            n_classes=int(ds.qualities.get("NumberOfClasses", 0)),
        ))
    return result


def load_dataset(
    dataset_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    OpenML에서 데이터셋을 로드하고 경계 검증 후 train/test split을 반환합니다.

    경계 검증 (생략 불가):
      1. NaN → 컬럼별 중앙값으로 대체
      2. X → float32로 통일
      3. y → 0-based 정수 인코딩 (LabelEncoder)
    """
    ds = openml.datasets.get_dataset(dataset_id)
    X, y, _, feature_names = ds.get_data(target=ds.default_target_attribute)

    # --- 경계 검증 시작 ---
    X = _validate_X(X)
    y = _validate_y(y)
    # --- 경계 검증 끝 ---

    n_classes = int(y.nunique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return DatasetSplit(
        dataset_id=dataset_id,
        name=ds.name,
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=list(feature_names),
        n_classes=n_classes,
    )


def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    """NaN 처리 + float32 변환. 수치형 피처 전용."""
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(np.float32)


def _validate_y(y: pd.Series) -> pd.Series:
    """레이블을 0-based 정수로 인코딩."""
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), name=y.name, dtype=np.int32)
