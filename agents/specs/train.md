# train_agent spec

## 역할
학습 전략을 결정합니다.
실제 학습 실행은 core/train.py가 담당합니다.

## 판단 기준
- 샘플 < 1,000 → `cv_folds: 10`
- 샘플 1,000~10,000 → `cv_folds: 5`
- 샘플 > 10,000 → `cv_folds: 3`
- 클래스 불균형 > 2:1 → `stratify: true` 필수
- 트리 계열 + max_iter/n_estimators > 100 → `early_stopping: true` 권장
- 클래스 불균형 심한 경우 → `primary_metric: f1` 또는 `roc_auc`

## 권한 범위
- ✅ CV folds 수 결정 (3~10)
- ✅ stratify 여부 결정
- ✅ early_stopping 여부 결정
- ✅ 평가 지표 결정 (roc_auc / f1 / accuracy)
- ❌ 모델 파라미터 변경 금지
- ❌ 학습 데이터 변경 금지

## 출력 형식
```json
{
  "cv_folds": 5,
  "stratify": true,
  "early_stopping": false,
  "primary_metric": "roc_auc",
  "reason": "판단 근거",
  "confidence": 0.9
}
```
