"""
data/loader.py — 유일한 데이터 진입점

외부 데이터(OpenML)가 시스템 내부로 들어오는 경계입니다.
모든 입력 데이터는 반드시 이 모듈을 통해서만 진입합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load(dataset_id: int = 44089) -> tuple[pd.DataFrame, pd.Series]:
    """
    OpenML에서 데이터셋을 로드하고 경계 검증을 수행합니다.

    Returns:
        X: 피처 DataFrame (float32)
        y: 타겟 Series (int32, 0-based)
    """
    import openml

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    X = _validate_X(X)
    y = _validate_y(y)

    return X, y


def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(np.float32)


def _validate_y(y: pd.Series) -> pd.Series:
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), dtype=np.int32)
