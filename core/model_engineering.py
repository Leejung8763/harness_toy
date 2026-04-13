"""
core/model_engineering.py — 모델 인스턴스 생성

model_engineering_agent의 판단을 받아 실제 모델을 생성합니다.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator

from agents.schemas import ModelPlan


def build(plan: ModelPlan) -> BaseEstimator:
    """
    agent 판단에 따라 모델 인스턴스를 생성합니다.

    Args:
        plan: model_engineering_agent.plan_model()의 반환값

    Returns:
        sklearn-compatible estimator
    """
    model_type = plan.model_type
    params = plan.params

    print(f"  [model_eng] type={model_type}, params={params}")

    if model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        allowed = {"learning_rate", "max_iter", "max_depth", "min_samples_leaf", "l2_regularization"}
        return HistGradientBoostingClassifier(**{k: v for k, v in params.items() if k in allowed})

    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            allowed = {"learning_rate", "n_estimators", "max_depth", "num_leaves", "min_child_samples"}
            return LGBMClassifier(verbose=-1, **{k: v for k, v in params.items() if k in allowed})
        except ImportError:
            print("  [model_eng] lightgbm not installed, fallback to HistGradientBoosting")
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier()

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        allowed = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
        return RandomForestClassifier(**{k: v for k, v in params.items() if k in allowed})

    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        allowed = {"C", "max_iter", "solver", "penalty"}
        return LogisticRegression(**{k: v for k, v in params.items() if k in allowed})

    else:
        print(f"  [model_eng] unknown model_type '{model_type}', fallback to HistGradientBoosting")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier()
