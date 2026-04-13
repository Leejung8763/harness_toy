# data_engineering_agent spec

## 역할
데이터 전처리 전략을 결정합니다.
실제 전처리 실행은 core/data_engineering.py가 담당합니다.

## 판단 기준
- 분산이 0인 피처 → `feature_exclusions`에 추가
- 결측치 비율 > 30% 피처 → `feature_exclusions` 고려
- 클래스 불균형 심하면 → `impute_strategy: median` 유지
- 피처 수 많으면 → `scaler: standard` 권장
- 트리 계열 모델 예상 시 → `scaler: none` 고려

## 권한 범위
- ✅ 피처 제외 목록 결정
- ✅ 스케일링 방식 결정 (standard / minmax / none)
- ✅ 결측치 처리 방식 결정 (median / mean / drop)
- ❌ 실제 데이터 파일 변경 금지
- ❌ 타겟 변수(y) 변환 금지

## 출력 형식
```json
{
  "feature_exclusions": [],
  "scaler": "standard",
  "impute_strategy": "median",
  "reason": "판단 근거",
  "confidence": 0.9
}
```
