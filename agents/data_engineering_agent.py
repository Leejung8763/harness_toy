"""
agents/data_engineering_agent.py — 데이터 전처리 전략 결정 에이전트

판단 내용:
  - 제외할 피처 목록 (분산 없음, 타겟 누수 의심 등)
  - 스케일링 방식 (standard / minmax / none)
  - 결측치 처리 방식 (median / mean / drop)

하네스 역할:
  - LLM 반환값은 DataEngPlan(Pydantic)으로 즉시 검증 — 잘못된 값은 경계에서 차단
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 전처리 실행은 core/data_engineering.py가 담당

참고: agents/AGENTS_SPEC.md, agents/schemas.py
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from pydantic import ValidationError

from agents.schemas import DataEngPlan

DEFAULT_PLAN = DataEngPlan(reason="default plan (fallback)")


def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


def _call_llm(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _build_prompt(X: pd.DataFrame, y: pd.Series) -> str:
    stats = X.describe().to_string()
    missing = X.isnull().sum().to_string()
    n_samples, n_features = X.shape
    class_balance = y.value_counts(normalize=True).to_dict()

    return f"""You are a data engineering expert for ML pipelines.

Dataset stats:
- Samples: {n_samples}, Features: {n_features}
- Class balance: {class_balance}
- Missing values per feature:
{missing}
- Feature statistics:
{stats}

Decide the preprocessing strategy. Return JSON with these fields:
- "feature_exclusions": list of column names to drop (e.g., zero-variance, leakage suspects)
- "scaler": one of "standard", "minmax", "none"
- "impute_strategy": one of "median", "mean", "drop"
- "reason": brief explanation
- "confidence": float 0-1

Available columns: {list(X.columns)}
"""


def plan_preprocessing(
    X: pd.DataFrame,
    y: pd.Series,
    use_agent: bool = True,
) -> DataEngPlan:
    """
    데이터 전처리 전략을 결정합니다.

    Returns:
        DataEngPlan (Pydantic) — 하네스가 계약 검증 완료된 판단
    """
    if not use_agent:
        return DataEngPlan(reason="use_agent=False (rule-based)")

    try:
        prompt = _build_prompt(X, y)
        raw = _call_llm(prompt)
        data = json.loads(raw)

        # 실제 존재하는 컬럼만 제외 목록에 포함 (하네스 규칙)
        valid_cols = set(X.columns)
        if "feature_exclusions" in data:
            data["feature_exclusions"] = [c for c in data["feature_exclusions"] if c in valid_cols]

        # ── Pydantic 계약 검증: 잘못된 값은 여기서 즉시 차단 ──
        return DataEngPlan(**data)

    except (ValidationError, Exception) as e:
        print(f"[data_engineering_agent] fallback: {e}")
        return DataEngPlan(reason=f"fallback due to: {e}")
