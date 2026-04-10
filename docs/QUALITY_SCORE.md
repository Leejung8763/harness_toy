# Quality Score

> 시스템이 통과해야 하는 품질 기준과 임계값을 정의합니다.
> 임계값 변경 시 이 문서와 구현 코드 **양쪽** 모두 업데이트하세요.

---

## 품질 게이트

| 메트릭 | 임계값 | 위치 | 미통과 시 |
|--------|--------|------|-----------|
| ROC-AUC (첫 배포) | ≥ 0.70 | `pipeline/evaluate.py:BASELINE_THRESHOLD` | 파이프라인 중단 |
| ROC-AUC (재배포) | ≥ 현재 champion | `pipeline/evaluate.py:_check_gate` | 파이프라인 중단 |
| ROC-AUC (운영 중) | ≥ 0.75 | `pipeline/monitor.py:CV_THRESHOLDS` | 자동 롤백 |
| Error Rate (운영 중) | ≤ 0.25 | `pipeline/monitor.py:CV_THRESHOLDS` | 자동 롤백 |

---

## 자동화 검증 (Back-pressure)

> 에이전트가 스스로 검증할 수 있도록 아래 명령을 항상 실행 가능한 상태로 유지합니다.

```bash
# 테스트 전체 실행 (23개 단위 테스트)
make test

# 특정 게이트만 검증
python3 -m pytest tests/test_evaluate.py -v   # Quality Gate
python3 -m pytest tests/test_monitor.py -v    # CV 임계값

# CI/CD Stage 전체 검증
./ci/run.sh
```

---

## 임계값 변경 절차

1. 이 문서의 표를 수정
2. 구현 코드의 상수 수정 (위 "위치" 컬럼 참조)
3. `tests/README.md`의 임계값 참조 표 동시 업데이트
4. 변경 이유를 [exec-plans/tech-debt-tracker.md](exec-plans/tech-debt-tracker.md)에 기록
