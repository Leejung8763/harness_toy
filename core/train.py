"""
core/train.py — 학습 실행

train_agent의 판단을 받아 실제 모델 학습을 수행합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    plan: dict,
) -> dict:
    """
    agent 판단에 따라 모델을 학습합니다.

    Args:
        model: core/model_engineering.build()가 반환한 estimator
        X: 전처리된 피처
        y: 타겟
        plan: train_agent.plan_training()의 반환값

    Returns:
        dict with keys: cv_scores, mean_score, std_score, trained_model
    """
    cv_folds = plan.get("cv_folds", 5)
    stratify = plan.get("stratify", True)
    metric = plan.get("primary_metric", "roc_auc")

    scoring = metric
    if metric == "roc_auc" and len(np.unique(y)) > 2:
        scoring = "roc_auc_ovr"

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if stratify else cv_folds

    print(f"  [train] {cv_folds}-fold CV, metric={scoring}, stratify={stratify}")

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mean_score = float(np.mean(cv_scores))
    std_score = float(np.std(cv_scores))

    print(f"  [train] CV {scoring}: {mean_score:.4f} ± {std_score:.4f}")

    # 전체 데이터로 최종 학습
    model.fit(X, y)

    return {
        "cv_scores": cv_scores.tolist(),
        "mean_score": mean_score,
        "std_score": std_score,
        "metric": scoring,
        "trained_model": model,
    }
