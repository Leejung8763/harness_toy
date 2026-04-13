# agents/AGENTS_SPEC.md — 에이전트 명세서

> 에이전트를 추가할 때마다 이 파일에 등록하세요.

---

## 에이전트 설계 원칙

1. **에이전트는 판단만 한다** — 실제 파일 변경·모델 저장은 `core/`가 담당
2. **Fallback 필수** — LLM 실패 시 rule-based 로직으로 자동 대체
3. **판단 근거 기록** — 모든 판단은 `reason` 필드에 기록
4. **입력 검증** — 에이전트는 받은 데이터의 유효성을 검증하고 잘못된 값은 기본값으로 대체

---

## 1. data_engineering_agent — 전처리 전략 결정

| 항목 | 내용 |
|------|------|
| **파일** | `agents/data_engineering_agent.py` |
| **호출 위치** | `run_pipeline.py` |
| **입력** | `X: DataFrame`, `y: Series` |
| **출력** | `feature_exclusions`, `scaler`, `impute_strategy`, `reason`, `confidence` |

### 판단 기준
- 분산이 0인 피처 → `feature_exclusions`에 추가
- 클래스 불균형 심하면 → `impute_strategy: median` 유지
- 피처 수 많으면 → `scaler: standard` 권장

### fallback
- LLM 실패 → `scaler: standard`, `impute_strategy: median`, 제외 없음

---

## 2. model_engineering_agent — 모델 선택 결정

| 항목 | 내용 |
|------|------|
| **파일** | `agents/model_engineering_agent.py` |
| **호출 위치** | `run_pipeline.py` |
| **입력** | `n_samples`, `n_features`, `n_classes`, `class_balance`, `preprocessing_plan` |
| **출력** | `model_type`, `params`, `reason`, `confidence` |

### 판단 기준
- 샘플 많고 피처 많으면 → `hist_gradient_boosting` 또는 `lightgbm`
- 해석 가능성 중요하면 → `logistic_regression`
- 중간 규모 → `random_forest`

### 허용 모델
`hist_gradient_boosting` | `lightgbm` | `random_forest` | `logistic_regression`

### fallback
- LLM 실패 → `hist_gradient_boosting` 기본 파라미터

---

## 3. train_agent — 학습 전략 결정

| 항목 | 내용 |
|------|------|
| **파일** | `agents/train_agent.py` |
| **호출 위치** | `run_pipeline.py` |
| **입력** | `n_samples`, `n_classes`, `class_balance`, `model_plan` |
| **출력** | `cv_folds`, `stratify`, `early_stopping`, `primary_metric`, `reason`, `confidence` |

### 판단 기준
- 샘플 < 1000 → `cv_folds: 10`
- 클래스 불균형 > 2:1 → `stratify: true`
- 트리 계열 + `n_estimators` 큼 → `early_stopping: true`

### fallback
- LLM 실패 → 5-fold stratified CV, metric=roc_auc

---

## 4. test_agent — 평가 결과 판단

| 항목 | 내용 |
|------|------|
| **파일** | `agents/test_agent.py` |
| **호출 위치** | `run_pipeline.py` |
| **입력** | `metrics: dict`, `model_plan`, `train_plan` |
| **출력** | `verdict` (pass/fail), `reason`, `suggestions`, `retrain_needed`, `confidence` |

### 판단 기준
- `roc_auc >= 0.70` → `pass`
- `roc_auc < 0.70` → `fail` + 개선 제안
- CV std 높으면 → 불안정 경고

### 권한 범위
- ❌ 모델 재학습 트리거 금지 (run_pipeline.py가 담당)
- ❌ 파일 저장 금지

### fallback
- LLM 실패 → `roc_auc >= 0.70` 기준 rule-based 판정
