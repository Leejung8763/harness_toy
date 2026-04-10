# Tests

> 단위 테스트 명세입니다.
> 테스트는 실제 모델 학습이나 네트워크 호출 없이 순수 로직만 검증합니다.

---

## 실행

```bash
make test              # 전체 테스트
python3 -m pytest tests/ -v         # 상세 출력
python3 -m pytest tests/test_evaluate.py  # 특정 파일만
```

---

## 테스트 파일 목록

| 파일 | 대상 | 검증 내용 |
|------|------|----------|
| `test_evaluate.py` | `pipeline/evaluate.py` | Quality Gate 통과/실패 조건 |
| `test_deploy.py` | `pipeline/deploy.py` | 모델 상태 전이, 후보 선정 |
| `test_monitor.py` | `pipeline/monitor.py` | 임계값 검사, 경계값 |
| `test_predict.py` | `pipeline/predict.py` | 입력 검증, NaN 처리 |

---

## 테스트 설계 원칙

1. **단위 테스트만** — 실제 ML 학습, OpenML 호출, 파일 I/O 없음
2. **순수 로직 검증** — 함수의 입력/출력만 테스트
3. **경계값 포함** — 임계값 정확히 일치하는 케이스 반드시 포함

---

## test_evaluate.py (6개)

| 테스트 | 시나리오 |
|--------|----------|
| `test_first_deploy_pass` | 첫 배포, ROC-AUC ≥ 0.70 → 통과 |
| `test_first_deploy_fail` | 첫 배포, ROC-AUC < 0.70 → 실패 |
| `test_first_deploy_exact_threshold` | ROC-AUC = 0.70 정확히 → 통과 (≥) |
| `test_beats_champion` | 기존 champion 대비 개선 → 통과 |
| `test_equal_to_champion` | 기존 champion과 동점 → 통과 (≥) |
| `test_worse_than_champion` | 기존 champion 대비 하락 → 실패 |

---

## test_deploy.py (7개)

| 테스트 | 시나리오 |
|--------|----------|
| `test_retire_current` | 기존 deployed → retired 전환 |
| `test_retire_current_no_deployed` | deployed 없음 → None 반환 |
| `test_retire_current_excludes_new_version` | 신규 버전은 retire 제외 |
| `test_get_candidate_latest` | 버전 미지정 → 최신 evaluated_pass |
| `test_get_candidate_specific_version` | 특정 버전 지정 → 해당 모델 |
| `test_get_candidate_no_pass` | evaluated_pass 없음 → None |
| `test_get_candidate_version_not_found` | 존재하지 않는 버전 → None |

---

## test_monitor.py (6개)

| 테스트 | 시나리오 |
|--------|----------|
| `test_thresholds_all_ok` | 모든 지표 정상 → None |
| `test_thresholds_low_roc_auc` | ROC-AUC < 0.75 → 위반 |
| `test_thresholds_roc_auc_exact_boundary` | ROC-AUC = 0.75 → 정상 (≥) |
| `test_thresholds_high_error_rate` | error_rate > 0.25 → 위반 |
| `test_thresholds_error_rate_exact_boundary` | error_rate = 0.25 → 정상 (≤) |
| `test_thresholds_both_violated` | 두 지표 모두 위반 → 첫 번째 반환 |

---

## test_predict.py (4개)

| 테스트 | 시나리오 |
|--------|----------|
| `test_validate_input_converts_to_float32` | DataFrame → float32 변환 |
| `test_validate_input_fills_nan_with_median` | NaN → 중앙값 대체 |
| `test_validate_input_does_not_modify_original` | 원본 DataFrame 불변 |
| `test_get_active_model_info_no_flags` | flags.json 없음 → no model deployed |

---

## 임계값 참조

> 구현 코드의 상수와 항상 일치해야 합니다. 변경 시 [docs/QUALITY_SCORE.md](../docs/QUALITY_SCORE.md) 동시 업데이트.

| 상수 | 값 | 위치 |
|------|-----|------|
| `BASELINE_THRESHOLD` | 0.70 | `pipeline/evaluate.py` |
| `CV_THRESHOLDS["roc_auc"]` | 0.75 | `pipeline/monitor.py` |
| `CV_THRESHOLDS["error_rate"]` | 0.25 | `pipeline/monitor.py` |
