# Train (Champion Selection) 명세

## 개요
5개 Challenger 모델을 동일 데이터로 학습하고 ROC-AUC 기준으로 Champion을 선정한다.

## 공개 API

```python
from pipeline.train import train

train(dataset_id: int) -> dict
```

## 입력
- `dataset_id`: OpenML Suite 337의 dataset ID

## 출력 (dict — registry entry)

```json
{
  "version": "v1",
  "status": "trained",
  "dataset_id": 44089,
  "dataset_name": "credit",
  "model_name": "hist_gradient_boosting",
  "metrics": { "roc_auc": 0.8622 },
  "fit_time_sec": 0.43,
  "model_path": "models/champion_credit_v1.pkl",
  "trained_at": "2026-04-10T..."
}
```

## Challenger 모델 목록

| 모델 | 패키지 |
|------|--------|
| logistic_regression | sklearn |
| random_forest | sklearn |
| hist_gradient_boosting | sklearn |
| xgboost | xgboost |
| lightgbm | lightgbm |

## Champion 선정 기준

- 지표: **ROC-AUC** (이진: 양성 클래스 확률, 다중: OvR macro)
- Champion은 test set에서 가장 높은 ROC-AUC를 기록한 모델
- Champion은 **전체 데이터(train+test)**로 재학습 후 저장

## 사전조건
- `dataset_id`가 OpenML Suite 337에 속할 것
- `registry/`, `models/` 디렉토리가 존재할 것

## 사후조건
- `models/champion_{name}_{version}.pkl` 저장
- `registry/model_registry.json`에 `status: "trained"` 항목 추가
- 버전은 기존 최대 버전 + 1로 자동 증가

## 관련 코드
- 구현: `pipeline/train.py`
- 의존: `data/loader.py` (경계 검증 포함)
- 저장소: `registry/model_registry.json`, `models/`
