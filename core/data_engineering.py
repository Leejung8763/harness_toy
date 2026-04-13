"""
core/data_engineering.py — 데이터 전처리 실행

data_engineering_agent의 판단을 받아 실제 전처리를 수행합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from agents.schemas import DataEngPlan


def run(
    X: pd.DataFrame,
    y: pd.Series,
    plan: DataEngPlan,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    agent 판단에 따라 전처리를 실행합니다.

    Args:
        X: 원본 피처 DataFrame
        y: 타겟 Series
        plan: data_engineering_agent.plan_preprocessing()의 반환값

    Returns:
        X_processed, y (변경 없음)
    """
    X = X.copy()

    exclusions = [c for c in plan.feature_exclusions if c in X.columns]
    if exclusions:
        X = X.drop(columns=exclusions)
        print(f"  [data_eng] dropped features: {exclusions}")

    if plan.impute_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
    elif plan.impute_strategy == "mean":
        X = X.fillna(X.mean(numeric_only=True))
    elif plan.impute_strategy == "drop":
        X = X.dropna()
        y = y.loc[X.index]

    if plan.scaler == "standard":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    elif plan.scaler == "minmax":
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    print(f"  [data_eng] shape={X.shape}, scaler={plan.scaler}, impute={plan.impute_strategy}")
    return X, y
