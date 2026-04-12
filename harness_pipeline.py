"""
harness_pipeline.py — 전체 파이프라인 오케스트레이터

실행 순서: Data → Train → Evaluate → Deploy → Monitor

규칙:
  - 각 스테이지가 False를 반환하면 즉시 중단
  - 중단 이유는 각 스테이지가 출력
  - 이 파일은 흐름 제어만 담당, 비즈니스 로직은 각 pipeline/ 모듈에 있음
"""

from __future__ import annotations

import argparse
import sys

from pipeline.data_engineering import run_data_engineering
from pipeline.deploy import deploy
from pipeline.evaluate import evaluate
from pipeline.monitor import monitor
from pipeline.train import train

BANNER = """
╔══════════════════════════════════════════════════════╗
║             Harness MLOps Pipeline                   ║
║  Data Eng → Train → Evaluate → Deploy → Monitor     ║
╚══════════════════════════════════════════════════════╝
"""

STAGES = [
    ("Train",    None),
    ("Evaluate", None),
    ("Deploy",   None),
    ("Monitor",  None),
]


def run(
    dataset_id: int = 44089,
    inject_drift: bool = False,
    skip_stages: list[str] | None = None,
) -> bool:
    """
    전체 파이프라인을 순서대로 실행합니다.

    Args:
        dataset_id: 학습에 사용할 OpenML 데이터셋 ID
        inject_drift: True면 Monitor 단계에서 드리프트를 주입
        skip_stages: 건너뛸 스테이지 목록 (None이면 전체 실행)
            예: ["data_eng"] — Feature Store 신선 시 Data Engineering 스킵

    Returns:
        bool: 전 스테이지 성공 시 True
    """
    skip = set(skip_stages or [])
    print(BANNER)
    if skip:
        print(f"  ⏭️  스킵 스테이지: {sorted(skip)}\n")

    # Stage 1 — Data Engineering
    if "data_eng" in skip:
        print("=" * 50)
        print("  STAGE 1 / 5  —  Data Engineering (⏭️  스킵 — Feature Store 재사용)")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  STAGE 1 / 5  —  Data Engineering (Feature Store)")
        print("=" * 50)
        de_result = run_data_engineering(dataset_id)
        if de_result is None:
            _abort("Data Engineering")
            return False

    # Stage 2 — Train
    print("\n" + "=" * 50)
    print("  STAGE 2 / 5  —  Train (Champion Selection)")
    print("=" * 50)
    train_result = train(dataset_id)
    if not train_result:
        _abort("Train")
        return False

    # Stage 3 — Evaluate (Quality Gate)
    print("\n" + "=" * 50)
    print("  STAGE 3 / 5  —  Evaluate (Quality Gate)")
    print("=" * 50)
    if not evaluate():
        _abort("Evaluate")
        return False

    # Stage 4 — Deploy
    print("\n" + "=" * 50)
    print("  STAGE 4 / 5  —  Deploy (CD Stage)")
    print("=" * 50)
    if not deploy():
        _abort("Deploy")
        return False

    # Stage 4.5 — A/B Test (챌린저가 등록된 경우만 실행)
    import json
    from pathlib import Path
    _flags = json.loads(Path("feature_flags/flags.json").read_text()) if Path("feature_flags/flags.json").exists() else {}
    if _flags.get("challenger_model_version"):
        print("\n" + "=" * 50)
        print("  STAGE 4.5   —  A/B Test (Challenger Evaluation)")
        print("=" * 50)
        from pipeline.ab_test import run_ab_test, promote_challenger, reject_challenger
        ab_result = run_ab_test()
        if ab_result["promote"]:
            promote_challenger(ab_result["challenger_version"])
        else:
            reject_challenger(ab_result["challenger_version"])

    # Stage 5 — Monitor
    print("\n" + "=" * 50)
    print("  STAGE 5 / 5  —  Monitor (Continuous Verification)")
    print("=" * 50)
    if not monitor(inject_drift=inject_drift):
        _abort("Monitor")
        return False

    print("\n" + "=" * 50)
    print("  ✅  파이프라인 완료")
    print("=" * 50)
    return True


def _abort(stage: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  🛑  파이프라인 중단 — {stage} 단계 실패")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness MLOps Pipeline")
    parser.add_argument(
        "--dataset-id", type=int, default=44089,
        help="OpenML 데이터셋 ID (기본값: 44089 = credit)"
    )
    parser.add_argument(
        "--drift", action="store_true",
        help="Monitor 단계에서 드리프트를 주입해 롤백 시나리오 시연"
    )
    args = parser.parse_args()

    success = run(dataset_id=args.dataset_id, inject_drift=args.drift)
    sys.exit(0 if success else 1)
