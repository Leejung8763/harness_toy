# Monitor 명세

## 개요
deployed 모델을 N 라운드 동안 모니터링하고, 임계값 위반 시 자동 롤백한다.
사람 개입 없이 자동 실행 — **파이프라인의 마지막 안전장치**.

## 공개 API

```python
from pipeline.monitor import monitor

monitor(rounds: int = 5, inject_drift: bool = False) -> bool
```

## CV 임계값 (CV_THRESHOLDS)

| 메트릭 | 임계값 | 위반 시 |
|--------|--------|---------|
| `roc_auc` | ≥ 0.75 | 자동 롤백 |
| `error_rate` | ≤ 0.25 | 자동 롤백 |

> ⚠️ 임계값 변경 시 이 문서와 `pipeline/monitor.py`의 `CV_THRESHOLDS` 모두 업데이트할 것

## 자동 롤백 동작

```
임계값 위반 감지
  → 현재 모델: status = "rolled_back"
  → 이전 retired 모델 있으면: status = "deployed", flags.json 복원
  → 이전 모델 없으면: flags.json active_model_version = null (서비스 중단)
```

## 파라미터
- `rounds`: 모니터링 반복 횟수 (기본 5)
- `inject_drift`: `True`면 노이즈를 점진적으로 추가해 롤백 시나리오 테스트

## 반환값
- `True`: 전 라운드 정상 완료
- `False`: 임계값 위반 → 롤백 발생, exit code 1

## 사전조건
- `flags.json`에 `active_model_version`이 설정되어 있을 것

## 관련 코드
- 구현: `pipeline/monitor.py`
- 임계값: `CV_THRESHOLDS` 상수
- 의존: `pipeline/predict.py`, `data/loader.py`
