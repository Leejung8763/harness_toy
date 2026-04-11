"""
api/serve.py — ML Prediction Service (CD Stage: ML Model Serving)

다이어그램:
  Model Registry → CD Stage: ML Model Serving → ML Prediction Service

엔드포인트:
  GET  /health          서버 및 모델 상태 확인
  GET  /model/info      현재 서빙 중인 모델 정보
  POST /predict         클래스 예측 (단건 / 배치)
  POST /predict/proba   클래스 확률 예측

규칙:
  - 어떤 모델을 쓸지는 feature_flags/flags.json 에서만 결정 (pipeline/predict.py 위임)
  - 입력 검증은 Pydantic 스키마로 경계에서 수행
  - 배포된 모델이 없으면 503 반환
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from pipeline.predict import get_active_model_info, predict, predict_proba

app = FastAPI(
    title="Harness MLOps — ML Prediction Service",
    description="Feature Flag 기반 ML 모델 서빙 API",
    version="0.1.0",
)


# ── 요청/응답 스키마 ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """예측 요청. features는 컬럼명→값 매핑의 리스트 (배치 가능)."""
    features: list[dict[str, float]]

    @field_validator("features")
    @classmethod
    def must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("features 리스트가 비어 있습니다.")
        return v


class PredictResponse(BaseModel):
    predictions: list[int]
    model_version: str
    n_samples: int


class PredictProbaResponse(BaseModel):
    probabilities: list[list[float]]
    model_version: str
    n_samples: int


class HealthResponse(BaseModel):
    status: str
    model_deployed: bool
    model_version: str | None


# ── 엔드포인트 ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """서버 및 모델 배포 상태를 확인합니다."""
    info = get_active_model_info()
    deployed = "version" in info
    return HealthResponse(
        status="ok",
        model_deployed=deployed,
        model_version=info.get("version"),
    )


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    """현재 서빙 중인 모델의 상세 정보를 반환합니다."""
    info = get_active_model_info()
    if "version" not in info:
        raise HTTPException(status_code=503, detail="배포된 모델이 없습니다.")
    return info


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """
    클래스를 예측합니다.

    요청 예시:
        {"features": [{"feature1": 1.0, "feature2": 2.0}, ...]}
    """
    info = get_active_model_info()
    if "version" not in info:
        raise HTTPException(status_code=503, detail="배포된 모델이 없습니다.")

    X = pd.DataFrame(req.features)
    preds = predict(X)

    return PredictResponse(
        predictions=preds.tolist(),
        model_version=info["version"],
        n_samples=len(preds),
    )


@app.post("/predict/proba", response_model=PredictProbaResponse)
def predict_proba_endpoint(req: PredictRequest) -> PredictProbaResponse:
    """
    클래스별 확률을 반환합니다.

    응답 예시:
        {"probabilities": [[0.8, 0.2], [0.3, 0.7]], "model_version": "v1", ...}
    """
    info = get_active_model_info()
    if "version" not in info:
        raise HTTPException(status_code=503, detail="배포된 모델이 없습니다.")

    X = pd.DataFrame(req.features)
    probas = predict_proba(X)

    return PredictProbaResponse(
        probabilities=probas.tolist(),
        model_version=info["version"],
        n_samples=len(probas),
    )
