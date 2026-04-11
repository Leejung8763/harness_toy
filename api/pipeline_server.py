"""
api/pipeline_server.py — Automated Pipeline Service (port 8001)

다이어그램:
  CI/CD Stage      → POST /pipeline/run  (코드 배포 후 파이프라인 실행)
  Operations Trigger → POST /pipeline/run  (드리프트 감지 후 재학습)

역할:
  - Pipeline 실행 요청을 HTTP로 받아 백그라운드에서 실행
  - 동시 실행 방지 (이미 실행 중이면 409 반환)
  - 실행 상태 추적 (idle → running → completed/failed)

엔드포인트:
  POST /pipeline/run     파이프라인 실행 요청 (202 Accepted, 즉시 반환)
  GET  /pipeline/status  현재 실행 상태 확인
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import harness_pipeline

app = FastAPI(
    title="Harness MLOps — Automated Pipeline Service",
    description="Automated Pipeline 실행 API (port 8001)",
    version="0.1.0",
)

# 파이프라인 상태 (단일 실행 보장)
_state: dict = {
    "status": "idle",
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "dataset_id": None,
    "error": None,
}
_lock = threading.Lock()


class PipelineRunRequest(BaseModel):
    dataset_id: int = 44089
    inject_drift: bool = False


class PipelineStatusResponse(BaseModel):
    status: Literal["idle", "running", "completed", "failed"]
    run_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    dataset_id: int | None = None
    error: str | None = None


@app.post("/pipeline/run", status_code=202)
def run_pipeline(req: PipelineRunRequest) -> dict:
    """
    Automated Pipeline을 백그라운드에서 실행합니다.

    - 이미 실행 중이면 409 Conflict
    - 즉시 202 Accepted 반환 후 백그라운드 실행
    - /pipeline/status 로 진행 상태 확인
    """
    with _lock:
        if _state["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="파이프라인이 이미 실행 중입니다. GET /pipeline/status 로 확인하세요.",
            )
        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        _state.update({
            "status": "running",
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "dataset_id": req.dataset_id,
            "error": None,
        })

    thread = threading.Thread(
        target=_execute_pipeline,
        args=(req.dataset_id, req.inject_drift),
        daemon=True,
    )
    thread.start()

    return {"status": "accepted", "run_id": run_id, "dataset_id": req.dataset_id}


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status() -> PipelineStatusResponse:
    """현재 파이프라인 실행 상태를 반환합니다."""
    with _lock:
        return PipelineStatusResponse(**_state)


def _execute_pipeline(dataset_id: int, inject_drift: bool) -> None:
    """백그라운드 스레드에서 파이프라인을 실행합니다."""
    try:
        harness_pipeline.run(dataset_id=dataset_id, inject_drift=inject_drift)
        with _lock:
            _state.update({
                "status": "completed",
                "finished_at": datetime.now().isoformat(),
            })
    except Exception as e:
        with _lock:
            _state.update({
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "error": str(e),
            })
