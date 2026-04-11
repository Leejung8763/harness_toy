"""
agents/eval/retrospective.py — 에이전트 결정 회고 분석기

역할:
  - 축적된 deploy_agent 결정 로그를 분석
  - 결정 유형별 분포 + 결과 정확도 계산
  - CLI로 실행하면 요약 리포트 출력

참고: openai/evals의 eval runner + EleutherAI의 benchmark result 패턴

사용법:
  PYTHONPATH=$(pwd) python3 agents/eval/retrospective.py
  PYTHONPATH=$(pwd) python3 agents/eval/retrospective.py --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agents.eval.decision_logger import load_deploy_decisions, load_drift_analyses

# ── 정확도 정의 ───────────────────────────────────────────────────────────────
# deploy 결정 → "survived" 이면 정확한 배포
# hold 결정   → "survived"(배포 안 했는데 모니터 안 망가짐)는 판단 불가
#             → 이후 같은 모델 "drift_detected"면 hold가 결과적으로 옳았을 수 있음
# reject 결정 → 직접 검증 어려움 (모델 미배포)
#
# 실용적 정확도: deploy 결정 중 outcome==survived 비율


def analyze_deploy_decisions(records: list[dict]) -> dict:
    decisions = [r["decision"] for r in records]
    outcomes = [r["outcome"] for r in records if r["outcome"] is not None]

    dist = Counter(decisions)
    outcome_dist = Counter(outcomes)

    deployed = [r for r in records if r["decision"] == "deploy"]
    deployed_with_outcome = [r for r in deployed if r.get("outcome") is not None]
    survived = [r for r in deployed if r.get("outcome") == "survived"]
    drift = [r for r in deployed if r.get("outcome") == "drift_detected"]

    deploy_accuracy = len(survived) / len(deployed_with_outcome) if deployed_with_outcome else None

    held = [r for r in records if r["decision"] == "hold"]
    pending = [r for r in records if r["outcome"] is None]

    return {
        "total_decisions": len(records),
        "decision_distribution": dict(dist),
        "outcome_distribution": dict(outcome_dist),
        "pending_outcome": len(pending),
        "deploy_accuracy": deploy_accuracy,
        "deployed_survived": len(survived),
        "deployed_drift": len(drift),
        "held_count": len(held),
        "avg_confidence": round(
            sum(r["confidence"] for r in records) / len(records), 3
        ) if records else None,
    }


def analyze_drift_analyses(records: list[dict]) -> dict:
    if not records:
        return {"total_analyses": 0}

    all_features: list[str] = []
    for r in records:
        all_features.extend(r.get("affected_features", []))
    feature_freq = Counter(all_features)

    return {
        "total_analyses": len(records),
        "top_affected_features": feature_freq.most_common(5),
        "retrain_triggered_count": sum(1 for r in records if r.get("retrain_triggered")),
    }


def print_report(deploy_records: list[dict], drift_records: list[dict]) -> None:
    deploy_stats = analyze_deploy_decisions(deploy_records)
    drift_stats = analyze_drift_analyses(drift_records)

    print("\n" + "=" * 56)
    print("  🔍  Agent Decision Retrospective Report")
    print("=" * 56)

    # deploy_agent 섹션
    print("\n  [deploy_agent]")
    print(f"  총 결정 수: {deploy_stats['total_decisions']}")
    dist = deploy_stats["decision_distribution"]
    for d, cnt in sorted(dist.items()):
        bar = "█" * cnt
        print(f"    {d:<8} {cnt:>3}  {bar}")

    print(f"\n  결과 업데이트됨 : {deploy_stats['total_decisions'] - deploy_stats['pending_outcome']}")
    print(f"  결과 대기 중   : {deploy_stats['pending_outcome']}")

    if deploy_stats["deploy_accuracy"] is not None:
        acc = deploy_stats["deploy_accuracy"]
        emoji = "✅" if acc >= 0.7 else "⚠️" if acc >= 0.5 else "❌"
        print(f"\n  배포 정확도    : {acc:.0%}  {emoji}")
        print(f"    deploy 후 survived : {deploy_stats['deployed_survived']}")
        print(f"    deploy 후 drift    : {deploy_stats['deployed_drift']}")
    else:
        print("\n  배포 정확도    : 데이터 부족 (outcome 업데이트 필요)")

    if deploy_stats["avg_confidence"] is not None:
        print(f"  평균 신뢰도    : {deploy_stats['avg_confidence']:.1%}")

    # drift_agent 섹션
    print(f"\n  [drift_agent]")
    print(f"  총 분석 수     : {drift_stats['total_analyses']}")
    if drift_stats.get("top_affected_features"):
        print("  자주 영향받은 피처:")
        for feat, cnt in drift_stats["top_affected_features"]:
            print(f"    {feat:<40} {cnt}회")
    print(f"  재학습 트리거  : {drift_stats.get('retrain_triggered_count', 0)}회")

    # 최근 5개 결정
    if deploy_records:
        print(f"\n  최근 deploy 결정 (최대 5개):")
        for r in deploy_records[-5:]:
            outcome = r["outcome"] or "pending"
            ts = r["timestamp"][:16]
            print(f"    {ts}  {r['model_version']:<4}  {r['decision']:<8}  →  {outcome}")

    print("\n" + "=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Decision Retrospective")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 출력")
    args = parser.parse_args()

    deploy_records = load_deploy_decisions()
    drift_records = load_drift_analyses()

    if args.json:
        output = {
            "deploy": analyze_deploy_decisions(deploy_records),
            "drift": analyze_drift_analyses(drift_records),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_report(deploy_records, drift_records)


if __name__ == "__main__":
    main()
