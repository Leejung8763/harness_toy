"""
agents/eval/decision_logger.py — 에이전트 결정 로거

역할:
  - 에이전트가 판단할 때마다 JSONL 파일에 기록
  - 판단 시점에는 outcome=null로 저장
  - monitor.py 완료 후 retrospective.py가 outcome을 채움

로그 형식 (JSONL, 한 줄 = 한 판단):
  {
    "id": "deploy-v2-20260412T024553",
    "timestamp": "2026-04-12T02:45:53",
    "agent": "deploy_agent",
    "decision": "hold",            # deploy / hold / reject
    "model_version": "v2",
    "confidence": 0.9,
    "reason": "성능 동일 ...",
    "context": {roc_auc, champion_roc_auc, ...},
    "outcome": null,               # 나중에 retrospective.py가 채움
    "outcome_updated_at": null
  }

참고: openai/evals 패턴 — 결정을 코드로 기록하고 나중에 평가
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("agents/eval/results")
DEPLOY_LOG = RESULTS_DIR / "deploy_decisions.jsonl"
DRIFT_LOG = RESULTS_DIR / "drift_analyses.jsonl"


def log_deploy_decision(
    model_version: str,
    decision: str,
    confidence: float,
    reason: str,
    context: dict,
) -> str:
    """
    deploy_agent 결정을 기록합니다.

    Returns:
        str: 결정 ID (retrospective 업데이트에 사용)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    decision_id = f"deploy-{model_version}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    record = {
        "id": decision_id,
        "timestamp": datetime.now().isoformat(),
        "agent": "deploy_agent",
        "decision": decision,
        "model_version": model_version,
        "confidence": confidence,
        "reason": reason,
        "context": context,
        "outcome": None,
        "outcome_updated_at": None,
    }

    with open(DEPLOY_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return decision_id


def log_drift_analysis(
    model_version: str,
    affected_features: list[str],
    cause: str,
    recommendation: str,
    metrics: dict,
) -> str:
    """
    drift_agent 분석 결과를 기록합니다.

    Returns:
        str: 분석 ID
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analysis_id = f"drift-{model_version}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    record = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "agent": "drift_agent",
        "model_version": model_version,
        "affected_features": affected_features,
        "cause": cause,
        "recommendation": recommendation,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        "retrain_triggered": True,
    }

    with open(DRIFT_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return analysis_id


def update_deploy_outcome(
    model_version: str,
    outcome: str,
    monitor_result: dict | None = None,
) -> bool:
    """
    deploy_agent 결정의 실제 결과를 업데이트합니다.

    Args:
        model_version: 업데이트할 모델 버전
        outcome: "survived" | "drift_detected" | "manual_rollback"
        monitor_result: 모니터링 지표 (선택)

    Returns:
        bool: 업데이트 성공 여부
    """
    if not DEPLOY_LOG.exists():
        return False

    records = [json.loads(line) for line in DEPLOY_LOG.read_text().splitlines() if line.strip()]
    updated = False

    for rec in records:
        if rec["model_version"] == model_version and rec["outcome"] is None:
            rec["outcome"] = outcome
            rec["outcome_updated_at"] = datetime.now().isoformat()
            if monitor_result:
                rec["monitor_result"] = monitor_result
            updated = True

    if updated:
        with open(DEPLOY_LOG, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return updated


def load_deploy_decisions() -> list[dict]:
    """저장된 모든 deploy 결정을 로드합니다."""
    if not DEPLOY_LOG.exists():
        return []
    return [json.loads(line) for line in DEPLOY_LOG.read_text().splitlines() if line.strip()]


def load_drift_analyses() -> list[dict]:
    """저장된 모든 drift 분석을 로드합니다."""
    if not DRIFT_LOG.exists():
        return []
    return [json.loads(line) for line in DRIFT_LOG.read_text().splitlines() if line.strip()]
