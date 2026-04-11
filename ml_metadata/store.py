"""
ml_metadata/store.py — ML Metadata Store

역할:
  - Data Engineering, Model Training 각 실행의 메타데이터 추적
  - 데이터 리니지(lineage): 어떤 피처 버전으로 어떤 모델이 학습됐는지 기록
  - model_registry.json과의 차이:
      registry = 배포 상태 관리 (trained/deployed/retired)
      ml_metadata = 실험 이력 관리 (무엇을, 어떻게, 결과는)

저장 형식: JSON (ml_metadata/runs.json)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

METADATA_PATH = Path("ml_metadata/runs.json")


def log_data_engineering(meta: dict) -> None:
    """Data Engineering 실행을 기록합니다."""
    _append_run("data_engineering", meta)


def log_training(meta: dict) -> None:
    """모델 학습 실행을 기록합니다."""
    _append_run("training", meta)


def log_monitoring(meta: dict) -> None:
    """모니터링 라운드 결과를 기록합니다."""
    _append_run("monitoring", meta)


def get_runs(stage: str | None = None) -> list[dict]:
    """
    실행 기록을 반환합니다.

    Args:
        stage: 'data_engineering' | 'training' | 'monitoring' | None (전체)
    """
    runs = _load()
    if stage:
        return [r for r in runs if r.get("stage") == stage]
    return runs


def get_latest_run(stage: str) -> dict | None:
    """특정 스테이지의 가장 최근 실행을 반환합니다."""
    runs = get_runs(stage)
    return runs[-1] if runs else None


def _append_run(stage: str, meta: dict) -> None:
    runs = _load()
    entry = {"stage": stage, "logged_at": datetime.now().isoformat(), **meta}
    runs.append(entry)
    _save(runs)


def _load() -> list[dict]:
    METADATA_PATH.parent.mkdir(exist_ok=True)
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return []


def _save(runs: list[dict]) -> None:
    with open(METADATA_PATH, "w") as f:
        json.dump(runs, f, indent=2)
