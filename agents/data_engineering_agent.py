"""
agents/data_engineering_agent.py — 데이터 전처리 전략 결정 에이전트

판단 내용:
  - 제외할 피처 목록 (분산 없음, 타겟 누수 의심 등)
  - 스케일링 방식 (standard / minmax / none)
  - 결측치 처리 방식 (median / mean / drop)

하네스 역할:
  - LLM 실패 시 DEFAULT_PLAN으로 즉시 fallback
  - 실제 전처리 실행은 core/data_engineering.py가 담당

참고: agents/AGENTS_SPEC.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PLAN: dict = {
    "feature_exclusions": [],
    "scaler": "standard",
    "impute_strategy": "median",
    "reason": "default plan (fallback)",
    "confidence": 1.0,
}


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
) -> dict:
    """
    데이터 전처리 전략을 결정합니다.

    Returns:
        dict with keys: feature_exclusions, scaler, impute_strategy, reason, confidence
    """
    if not use_agent:
        return {**DEFAULT_PLAN, "reason": "use_agent=False (rule-based)"}

    try:
        prompt = _build_prompt(X, y)
        raw = _call_llm(prompt)
        plan = json.loads(raw)

        # 유효성 검증
        plan.setdefault("feature_exclusions", [])
        plan["scaler"] = plan.get("scaler", "standard") if plan.get("scaler") in ("standard", "minmax", "none") else "standard"
        plan["impute_strategy"] = plan.get("impute_strategy", "median") if plan.get("impute_strategy") in ("median", "mean", "drop") else "median"
        plan.setdefault("reason", "LLM decision")
        plan.setdefault("confidence", 0.8)

        # 실제 존재하는 컬럼만 제외 목록에 포함
        valid_cols = set(X.columns)
        plan["feature_exclusions"] = [c for c in plan["feature_exclusions"] if c in valid_cols]

        return plan

    except Exception as e:
        print(f"[data_engineering_agent] fallback: {e}")
        return {**DEFAULT_PLAN, "reason": f"fallback due to: {e}"}
