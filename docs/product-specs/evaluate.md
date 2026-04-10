# Evaluate (Quality Gate) 명세

## 개요
신규 모델이 기존 champion을 이겨야 배포 가능하다.
이 게이트를 통과하지 않으면 배포 불가 — **파이프라인의 핵심 invariant**.

## 공개 API

```python
from pipeline.evaluate import evaluate

evaluate(version: str | None = None) -> bool
```

## 통과 기준

| 상황 | 기준 |
|------|------|
| 기존 `deployed` champion 없음 (첫 배포) | ROC-AUC ≥ `BASELINE_THRESHOLD` (0.70) |
| 기존 `deployed` champion 있음 | ROC-AUC > champion ROC-AUC |

## 상태 전이

```
trained → evaluated_pass   (통과)
trained → evaluated_fail   (실패, 파이프라인 중단)
```

## 입력
- `version`: 평가할 모델 버전 (None이면 최신 `trained` 모델 자동 선택)

## 출력
- `True`: 통과 → registry `status: "evaluated_pass"` 업데이트
- `False`: 실패 → registry `status: "evaluated_fail"` 업데이트, exit code 1

## 사전조건
- `registry/model_registry.json`에 `status: "trained"` 항목 존재
- 해당 `model_path`에 pkl 파일 존재

## 사후조건
- registry의 해당 버전 `status`가 `evaluated_pass` 또는 `evaluated_fail`로 변경
- `metrics.roc_auc`가 재계산된 값으로 업데이트

## 관련 코드
- 구현: `pipeline/evaluate.py`
- 의존: `data/loader.py`, `registry/model_registry.json`
- 임계값 상수: `BASELINE_THRESHOLD = 0.70`
