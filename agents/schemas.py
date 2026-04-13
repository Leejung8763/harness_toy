"""
agents/schemas.py — 에이전트 입출력 계약 (Pydantic)

각 에이전트의 반환값 타입을 명시합니다.
LLM이 잘못된 값을 반환하면 이 계약이 즉시 차단하고 fallback을 트리거합니다.

이것이 Harness: "구현 방식은 AI에게, 불변 규칙은 코드가 강제"
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── 0. Orchestrator ───────────────────────────────────────────────────────────

class OrchestratorPlan(BaseModel):
    """orchestrator_agent의 반환 계약"""

    goal: str = "ML classification"
    stages: list[Literal["data_eng", "model_eng", "train", "test"]] = Field(
        default=["data_eng", "model_eng", "train", "test"]
    )
    hints: dict[str, str] = Field(default_factory=dict)
    """각 단계에 전달할 힌트. key = 단계명, value = 자연어 힌트"""
    reason: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── 1. Data Engineering ───────────────────────────────────────────────────────

class DataEngPlan(BaseModel):
    """data_engineering_agent의 반환 계약"""

    feature_exclusions: list[str] = Field(default_factory=list)
    scaler: Literal["standard", "minmax", "none"] = "standard"
    impute_strategy: Literal["median", "mean", "drop"] = "median"
    reason: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── 2. Model Engineering ──────────────────────────────────────────────────────

class ModelPlan(BaseModel):
    """model_engineering_agent의 반환 계약"""

    model_type: Literal[
        "hist_gradient_boosting",
        "lightgbm",
        "random_forest",
        "logistic_regression",
    ] = "hist_gradient_boosting"
    params: dict = Field(default_factory=dict)
    reason: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── 3. Train ──────────────────────────────────────────────────────────────────

class TrainPlan(BaseModel):
    """train_agent의 반환 계약"""

    cv_folds: int = Field(default=5, ge=3, le=10)
    stratify: bool = True
    early_stopping: bool = False
    primary_metric: Literal["roc_auc", "f1", "accuracy"] = "roc_auc"
    reason: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── 4. Test ───────────────────────────────────────────────────────────────────

class TestVerdict(BaseModel):
    """test_agent의 반환 계약"""

    verdict: Literal["pass", "fail"] = "fail"
    reason: str = "default"
    suggestions: list[str] = Field(default_factory=list)
    retrain_needed: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
