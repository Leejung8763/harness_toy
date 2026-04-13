"""
run_pipeline.py — 4단계 ML 파이프라인 오케스트레이터

Data Engineering → Model Engineering → Train → Test
각 단계에서 에이전트가 판단하고 core/ 코드가 실행합니다.
"""

from __future__ import annotations

import sys

import numpy as np

from agents import data_engineering_agent, model_engineering_agent, test_agent, train_agent
from core import data_engineering, model_engineering, test, train
from data import loader


def run(
    dataset_id: int = 44089,
    use_agent: bool = True,
) -> dict:
    """
    전체 ML 파이프라인을 실행합니다.

    Args:
        dataset_id: OpenML 데이터셋 ID
        use_agent: False면 모든 에이전트를 rule-based fallback으로 실행

    Returns:
        결과 요약 dict
    """
    print(f"\n{'='*50}")
    print(f" ML Pipeline  dataset_id={dataset_id}  use_agent={use_agent}")
    print(f"{'='*50}")

    # ── 1. 데이터 로드 ─────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    X, y = loader.load(dataset_id)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    class_balance = y.value_counts(normalize=True).to_dict()
    print(f"  samples={n_samples}, features={n_features}, classes={n_classes}")

    # ── 2. Data Engineering ────────────────────────────────────────
    print("\n[2/4] Data Engineering...")
    de_plan = data_engineering_agent.plan_preprocessing(X, y, use_agent=use_agent)
    print(f"  agent decision: scaler={de_plan.scaler}, impute={de_plan.impute_strategy}, reason={de_plan.reason}")
    X_processed, y_processed = data_engineering.run(X, y, de_plan)

    # ── 3. Model Engineering ───────────────────────────────────────
    print("\n[3/4] Model Engineering...")
    me_plan = model_engineering_agent.plan_model(
        n_samples=n_samples,
        n_features=X_processed.shape[1],
        n_classes=n_classes,
        class_balance=class_balance,
        preprocessing_plan=de_plan,
        use_agent=use_agent,
    )
    print(f"  agent decision: model={me_plan.model_type}, reason={me_plan.reason}")
    model = model_engineering.build(me_plan)

    # ── 4. Train ───────────────────────────────────────────────────
    print("\n[4a/4] Training...")
    tr_plan = train_agent.plan_training(
        n_samples=n_samples,
        n_classes=n_classes,
        class_balance=class_balance,
        model_plan=me_plan,
        use_agent=use_agent,
    )
    print(f"  agent decision: cv={tr_plan.cv_folds}-fold, stratify={tr_plan.stratify}, reason={tr_plan.reason}")
    train_result = train.run(model, X_processed, y_processed, tr_plan)

    # ── 5. Test ────────────────────────────────────────────────────
    print("\n[4b/4] Testing...")
    metrics = test.run(train_result["trained_model"], X_processed, y_processed)
    verdict = test_agent.judge_results(metrics, me_plan, tr_plan, use_agent=use_agent)
    print(f"  agent verdict: {verdict.verdict.upper()} — {verdict.reason}")
    if verdict.suggestions:
        for s in verdict.suggestions:
            print(f"  💡 {s}")

    # ── 요약 ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    status = "✅ PASS" if verdict.verdict == "pass" else "❌ FAIL"
    print(f" {status}")
    print(f" model    : {me_plan.model_type}")
    print(f" roc_auc  : {metrics['roc_auc']:.4f}")
    print(f" f1       : {metrics['f1']:.4f}")
    print(f" accuracy : {metrics['accuracy']:.4f}")
    print(f" CV mean  : {train_result['mean_score']:.4f} ± {train_result['std_score']:.4f}")
    print(f"{'='*50}\n")

    return {
        "verdict": verdict.verdict,
        "metrics": metrics,
        "train_result": {k: v for k, v in train_result.items() if k != "trained_model"},
        "plans": {
            "data_engineering": de_plan.model_dump(),
            "model_engineering": me_plan.model_dump(),
            "train": tr_plan.model_dump(),
            "test_verdict": verdict.model_dump(),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Pipeline")
    parser.add_argument("--dataset-id", type=int, default=44089)
    parser.add_argument("--no-agent", action="store_true", help="rule-based fallback only")
    args = parser.parse_args()

    result = run(dataset_id=args.dataset_id, use_agent=not args.no_agent)
    sys.exit(0 if result["verdict"] == "pass" else 1)
