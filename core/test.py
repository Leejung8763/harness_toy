"""
core/test.py — 모델 평가 실행

test_agent에 전달할 평가 지표를 계산합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def run(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    홀드아웃 테스트셋으로 최종 평가 지표를 계산합니다.

    Returns:
        dict with keys: roc_auc, f1, accuracy, n_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "roc_auc": float(roc_auc),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "n_test": len(y_test),
    }

    print(f"  [test] roc_auc={metrics['roc_auc']:.4f}, f1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
    return metrics
