# Deploy 명세

## 개요
`evaluated_pass` 모델을 `deployed`로 전환하고 Feature Flag를 업데이트한다.
코드 변경 없이 `flags.json`만으로 서빙 모델을 교체할 수 있다.

## 공개 API

```python
from pipeline.deploy import deploy, load_flags

deploy(version: str | None = None) -> bool
load_flags() -> dict
```

## 상태 전이

```
evaluated_pass → deployed    (신규 모델)
deployed       → retired     (기존 모델, 자동)
```

## 사전조건
- registry에 `status: "evaluated_pass"` 항목이 존재할 것
- **`evaluated_pass` 없이 deploy 호출 시 즉시 False 반환** ← invariant

## 사후조건
- registry: 신규 모델 `status: "deployed"`, 기존 `status: "retired"`
- `feature_flags/flags.json`: `active_model_version`, `active_dataset_id` 업데이트

## Feature Flag 규칙
- `flags.json`은 반드시 이 모듈의 `_update_flags()`를 통해서만 변경
- 직접 파일 수정 금지 — predict.py는 `load_flags()`로만 읽을 것

## flags.json 구조

```json
{
  "active_model_version": "v3",
  "active_dataset_id": 44089
}
```

## 관련 코드
- 구현: `pipeline/deploy.py`
- 의존: `registry/model_registry.json`, `feature_flags/flags.json`
