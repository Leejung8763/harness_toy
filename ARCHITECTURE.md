# Architecture

> Enforce invariants, not implementations.
> 레이어 구조는 린터와 구조 테스트로 기계적으로 강제합니다.

---

## 레이어 구조

의존성은 **아래 방향으로만** 흐릅니다.

```
Data (data/)
   ↓
Pipeline (pipeline/)
   ↓
Registry / Feature Flags (registry/, feature_flags/)
   ↓
Runtime (harness_pipeline.py)
```

| 레이어 | 경로 | 역할 | 허용 import |
|--------|------|------|-------------|
| Data | `data/loader.py` | 외부 데이터 수집·검증·정제 (경계 진입점) | openml, pandas, numpy, sklearn |
| Pipeline | `pipeline/*.py` | 비즈니스 로직 (train/evaluate/deploy/monitor/predict) | data/, 표준 라이브러리, ML 라이브러리 |
| Registry | `registry/`, `feature_flags/` | 모델 버전 상태 저장, Feature Flag | 파일 I/O만 |
| Runtime | `harness_pipeline.py` | 스테이지 오케스트레이션 | pipeline/ |
| CI/CD | `ci/`, `tests/`, `Makefile` | 빌드·테스트·패키징 | 테스트 대상 모듈만 |

---

## 금지 패턴

```python
# ❌ data/ 레이어에서 pipeline/ import 금지
from pipeline.deploy import deploy  # data/loader.py 안에서 절대 불가

# ❌ flags.json 직접 수정 금지
with open("feature_flags/flags.json", "w") as f:
    json.dump({"active_model_version": "v1"}, f)

# ✅ 반드시 deploy.py를 통해서만 변경
from pipeline.deploy import deploy
deploy()

# ❌ 상태 검증 없이 모델 배포 금지
entry["status"] = "deployed"  # 직접 상태 변경 금지

# ✅ 반드시 evaluated_pass 확인 후 배포
candidates = [m for m in models if m["status"] == "evaluated_pass"]
```

---

## 경계(Boundary) 검증 규칙

외부 데이터가 시스템 내부로 들어오는 지점에서 반드시 유효성을 검증합니다.

```python
# data/loader.py — 유일한 데이터 진입점
def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.fillna(X.median(numeric_only=True))  # NaN → 중앙값
    return X.astype(np.float32)                # float32 통일

def _validate_y(y: pd.Series) -> pd.Series:
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), dtype=np.int32)  # 0-based 정수
```

---

## 모델 상태 전이

```
trained → evaluated_pass / evaluated_fail → deployed → retired / rolled_back
```

- `trained`: `pipeline/train.py`가 설정
- `evaluated_pass / fail`: `pipeline/evaluate.py`가 설정
- `deployed`: `pipeline/deploy.py`가 설정 (evaluated_pass만 가능)
- `retired`: 새 모델 배포 시 기존 deployed 모델에 설정
- `rolled_back`: `pipeline/monitor.py`가 임계값 위반 시 설정

---

## 구조 테스트

```bash
make test   # tests/ 하위 23개 단위 테스트로 핵심 불변 규칙 검증
```
