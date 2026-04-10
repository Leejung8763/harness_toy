# Generated: Data Schema

> `data/loader.py`가 반환하는 데이터 구조를 문서화합니다.
> 소스: `data/loader.py` — 변경 시 이 파일도 업데이트하세요.

---

## DatasetInfo

```python
@dataclass
class DatasetInfo:
    dataset_id: int    # OpenML dataset ID
    name: str          # 데이터셋 이름
    n_samples: int     # 전체 행 수
    n_features: int    # 피처 수 (타겟 제외)
    n_classes: int     # 클래스 수
```

## DatasetSplit

```python
@dataclass
class DatasetSplit:
    dataset_id: int
    name: str
    X_train: pd.DataFrame   # dtype=float32, NaN 없음
    X_test: pd.DataFrame    # dtype=float32, NaN 없음
    y_train: pd.Series      # dtype=int32, 0-based
    y_test: pd.Series       # dtype=int32, 0-based
    feature_names: list[str]
    n_classes: int
```

---

## Suite 337 — Numerical Classification (16개 데이터셋)

| dataset_id | name | n_samples | n_features | n_classes |
|------------|------|-----------|------------|-----------|
| 44089 | credit | 16,714 | 10 | 2 |
| 44120 | electricity | 38,474 | 7 | 2 |
| 44121 | covertype | 566,602 | 10 | 2 |
| (나머지 13개) | … | … | … | … |

전체 목록: `from data.loader import list_datasets` 로 조회

---

## 마지막 업데이트

- 날짜: 2026-04-10
- 소스: `data/loader.py`, OpenML Suite 337
