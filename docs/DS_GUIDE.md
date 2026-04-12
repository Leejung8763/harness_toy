# DS Development Guide

> 이 문서는 `ds_development_agent`가 실험 설정을 결정할 때 참조하는 가이드입니다.
> 사람이 작성하고, 에이전트는 이 문서를 LLM 컨텍스트로 읽어 판단합니다.

---

## 프로젝트 개요

**도메인:** 신용 리스크 이진 분류 (Give Me Some Credit, Kaggle)  
**목표:** 2년 내 심각한 연체 발생 여부 예측

---

## 타겟 변수

| 항목 | 값 |
|------|-----|
| 기본 타겟 | `SeriousDlqin2yrs` |
| 타입 | 이진 분류 (0: 정상, 1: 연체) |
| 변경 조건 | 데이터셋이 명시적으로 다른 타겟을 포함할 때만 변경 |

> **에이전트 판단 기준:** 컬럼명에 "default", "target", "label", "y" 등이 포함되거나,
> 데이터셋 설명에서 명시적으로 지정된 컬럼을 타겟으로 선택합니다.

---

## 모델 후보

| 모델 키 | 클래스 | 선택 조건 | 우선순위 |
|---------|--------|----------|---------|
| `hist_gradient_boosting` | HistGradientBoostingClassifier | 항상 포함 (기본) | 1 |
| `lightgbm` | LGBMClassifier | 샘플 수 > 10,000 | 2 |
| `xgboost` | XGBClassifier | 샘플 수 > 10,000 | 3 |
| `random_forest` | RandomForestClassifier | 피처 수 < 50 | 4 |
| `logistic_regression` | LogisticRegression | 베이스라인 비교용, 항상 포함 | 5 |

> **에이전트 판단 기준:** 샘플 수·피처 수·클래스 불균형 비율을 고려해
> 위 우선순위에서 실행할 모델을 선택합니다. 최소 2개 이상 선택해야 합니다.

---

## 평가 지표

| 지표 | 조건 | 비고 |
|------|------|------|
| `roc_auc` | 기본 (항상 사용) | Champion 선정 기준 |
| `f1` | 클래스 불균형 비율 > 0.3일 때 참고 | 참고 지표 |
| `accuracy` | 균형 데이터셋에서만 참고 | 참고 지표 |

> **에이전트 판단 기준:** Champion 선정은 `roc_auc` 단일 지표 기준.
> 불균형이 심하면 `f1`도 같이 보고하도록 권고합니다.

---

## 피처 제외 규칙

| 규칙 | 설명 |
|------|------|
| 분산 0 피처 | 자동 제거 (data_engineering.py가 처리) |
| 미래 정보 피처 | 타겟과 시간적으로 동일 시점 이후 데이터 포함 시 수동 제외 |
| ID/인덱스 컬럼 | `id`, `index`, `row_number` 등 예측에 무의미한 컬럼 제외 |
| 고유값 비율 > 95% | 카테고리형 피처 중 거의 모든 값이 유일한 경우 제외 |

> **에이전트 판단 기준:** 위 규칙에 해당하는 컬럼을 `feature_exclusions`에 추가합니다.
> 판단이 불확실하면 제외하지 않고 `reason`에 근거를 명시합니다.

---

## Defaults (코드 fallback과 일치)

에이전트 LLM 호출이 실패할 경우 코드는 아래 값을 사용합니다.

```
target_column:      SeriousDlqin2yrs
models_to_try:      [hist_gradient_boosting, lightgbm, xgboost, random_forest, logistic_regression]
primary_metric:     roc_auc
feature_exclusions: []
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-04-12 | 최초 작성 |
