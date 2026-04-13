# test_agent spec

## 역할
평가 결과를 해석하고 합격/불합격을 판단합니다.
실제 평가 실행은 core/test.py가 담당합니다.

## 판단 기준
- `roc_auc >= 0.70` → `pass`
- `roc_auc < 0.70` → `fail` + 구체적 개선 제안 필수
- CV std > 0.05 → 불안정 경고 (suggestions에 포함)
- 금융/의료 도메인 → recall 기준 추가 검토
- f1 < 0.60 (불균형 데이터) → `fail` 고려

## 권한 범위
- ✅ pass/fail 판정
- ✅ 개선 제안 작성
- ✅ 재학습 필요 여부 판단
- ❌ 모델 재학습 트리거 금지 (run_pipeline.py가 담당)
- ❌ 파일 저장 금지
- ❌ registry/flags 수정 금지

## 출력 형식
```json
{
  "verdict": "pass",
  "reason": "판단 근거",
  "suggestions": [],
  "retrain_needed": false,
  "confidence": 0.9
}
```
