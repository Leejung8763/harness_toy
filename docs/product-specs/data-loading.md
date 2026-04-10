# Data Loading 명세

## 개요
OpenML Suite 337(Grinsztajn Numerical Classification)에서 데이터를 로드하고 경계 검증 후 반환한다.

## 공개 API

```python
from data.loader import list_datasets, load_dataset

list_datasets() -> list[DatasetInfo]
load_dataset(dataset_id: int, test_size: float = 0.2, random_state: int = 42) -> DatasetSplit
```

## 입력
- `dataset_id`: OpenML Suite 337에 속한 dataset ID (`list_datasets()`로 조회)
- `test_size`: test split 비율 (기본 0.2)
- `random_state`: 재현성 시드 (기본 42)

## 출력 (DatasetSplit)
- `X_train / X_test`: float32, NaN 없음, shape=(n, n_features)
- `y_train / y_test`: int32, 0-based 정수 레이블

## 경계 검증 규칙 (생략 불가)

| 규칙 | 방법 | 위치 |
|------|------|------|
| NaN 제거 | 컬럼별 중앙값으로 대체 | `_validate_X()` |
| X 타입 통일 | float32로 변환 | `_validate_X()` |
| y 인코딩 | LabelEncoder → int32, 0-based | `_validate_y()` |

## 사전조건
- OpenML 접근 가능 (인터넷 연결 또는 로컬 캐시)
- `dataset_id`가 Suite 337에 속할 것

## 사후조건
- `X_train.isna().sum().sum() == 0`
- `X_train.dtypes == float32` (모든 컬럼)
- `y_train.dtype == int32`
- `y_train.min() == 0`
- stratified split 보장

## 관련 코드
- 구현: `data/loader.py`
- 스키마: `docs/generated/db-schema.md`
