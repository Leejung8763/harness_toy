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

from pipeline.deploy import deploy
from pipeline.evaluate import evaluate
from pipeline.monitor import monitor
from pipeline.train import train

BANNER = """
╔══════════════════════════════════════════╗
║       Harness MLOps Pipeline             ║
║  Data → Train → Evaluate → Deploy → Monitor  ║
╚══════════════════════════════════════════╝
"""

STAGES = [
    ("Train",    None),
    ("Evaluate", None),
    ("Deploy",   None),
    ("Monitor",  None),
]


def run(dataset_id: int = 44089, inject_drift: bool = False) -> bool:
    """
    전체 파이프라인을 순서대로 실행합니다.

    Args:
        dataset_id: 학습에 사용할 OpenML 데이터셋 ID
        inject_drift: True면 Monitor 단계에서 드리프트를 주입

    Returns:
        bool: 전 스테이지 성공 시 True
    """
    print(BANNER)

    # Stage 1 — Train
    print("=" * 50)
    print("  STAGE 1 / 4  —  Train (Champion Selection)")
    print("=" * 50)
    train_result = train(dataset_id)
    if not train_result:
        _abort("Train")
        return False

    # Stage 2 — Evaluate (Quality Gate)
    print("\n" + "=" * 50)
    print("  STAGE 2 / 4  —  Evaluate (Quality Gate)")
    print("=" * 50)
    if not evaluate():
        _abort("Evaluate")
        return False

    # Stage 3 — Deploy
    print("\n" + "=" * 50)
    print("  STAGE 3 / 4  —  Deploy")
    print("=" * 50)
    if not deploy():
        _abort("Deploy")
        return False

    # Stage 4 — Monitor
    print("\n" + "=" * 50)
    print("  STAGE 4 / 4  —  Monitor (Continuous Verification)")
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
