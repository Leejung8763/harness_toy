# Predict 명세

## 개요
`flags.json`에서 현재 deployed 모델을 읽어 예측을 수행한다.
모델 버전과 경로를 코드에 하드코딩하지 않는다.

## 공개 API

```python
from pipeline.predict import predict, predict_proba, get_active_model_info

predict(X: pd.DataFrame) -> np.ndarray          # 클래스 예측 (int32, 0-based)
predict_proba(X: pd.DataFrame) -> np.ndarray     # 클래스별 확률 (shape: n_samples × n_classes)
get_active_model_info() -> dict                  # 현재 서빙 중인 모델 메타데이터
```

## 모델 로드 경로

```
flags.json
  → active_model_version
  → registry/model_registry.json (model_path 조회)
  → models/champion_{name}_{version}.pkl
```

## 입력 전처리 규칙 (생략 불가)
- NaN → 컬럼 중앙값으로 대체
- dtype → float32로 변환
- `data/loader.py`의 `_validate_X()`와 동일한 규칙 적용

## 사전조건
- `feature_flags/flags.json`의 `active_model_version`이 설정되어 있을 것
- 해당 버전이 registry에 존재하고 `model_path` pkl 파일이 있을 것

## 에러 케이스
| 상황 | 동작 |
|------|------|
| `active_model_version: null` | `RuntimeError` |
| registry에 버전 없음 | `RuntimeError` |
| pkl 파일 없음 | `FileNotFoundError` |

## 관련 코드
- 구현: `pipeline/predict.py`
- 모델 교체: `pipeline/deploy.py` 실행 → flags.json 자동 업데이트
- 의존: `feature_flags/flags.json`, `registry/model_registry.json`
