# agents/AGENTS_SPEC.md

> 이 파일은 모든 LLM 에이전트의 명세서입니다.
> 에이전트를 추가할 때마다 이 파일에 등록하세요.

---

## 에이전트 설계 원칙

1. **에이전트는 판단만 한다** — 실제 파일 변경(registry, flags.json)은 반드시 pipeline/ 코드가 수행
2. **하네스가 최종 안전망** — LLM 판단이 틀려도 Quality Gate가 막는다
3. **fallback 필수** — 에이전트 호출 실패 시 rule-based 로직으로 자동 대체
4. **판단 근거 기록** — 모든 판단은 ml_metadata에 로깅

---

## 에이전트 목록

### 1. deploy_agent — 배포 판단 에이전트

| 항목 | 내용 |
|------|------|
| **파일** | `agents/deploy_agent.py` |
| **호출 위치** | `pipeline/deploy.py:deploy()` |
| **입력** | 후보 모델 버전, registry 정보, 학습 메타데이터 |
| **출력** | `decision`: deploy / hold / reject |
| **모델** | gpt-4o-mini (GitHub Models API) |

#### 참고 문서
| 문서 | 용도 |
|------|------|
| `config/thresholds.json` | 배포 판단 수치 기준 (ROC-AUC baseline 등) |
| `docs/QUALITY_SCORE.md` | 품질 게이트 기준 및 PSI 해석 |
| `registry/model_registry.json` | 후보/챔피언 모델 상태 및 메트릭 |
| `ml_metadata/runs.json` | 학습 이력 (Feature Store 사용 여부, 샘플 수 등) |

#### 스킬 (할 수 있는 것)
- ✅ registry에서 모델 정보 읽기
- ✅ ml_metadata에서 학습 이력 읽기
- ✅ config/thresholds.json에서 기준값 읽기
- ✅ 배포 여부 판단 및 근거 반환

#### 권한 범위 (할 수 없는 것)
- ❌ registry/model_registry.json 직접 수정 금지
- ❌ feature_flags/flags.json 직접 수정 금지
- ❌ 모델 파일(.pkl) 접근 금지
- ❌ 다른 에이전트 직접 호출 금지 (오케스트레이터를 통해서만)

#### 판단 기준 (프롬프트에 주입되는 규칙)
1. ROC-AUC가 챔피언보다 높거나 같으면 일반적으로 `deploy`
2. 성능 차이 < 0.001이면 변경 비용 대비 효익 낮음 → `hold`
3. ROC-AUC < BASELINE_THRESHOLD(0.70)이면 `reject`
4. 첫 배포(챔피언 없음)는 baseline 충족 시 `deploy`

#### fallback 동작
- LLM API 호출 실패 → 자동 배포(rule-based)로 대체
- 응답 파싱 실패 → `hold` 반환

---

### 2. drift_agent — 드리프트 원인 분석 에이전트

| 항목 | 내용 |
|------|------|
| **파일** | `agents/drift_agent.py` |
| **호출 위치** | `pipeline/monitor.py` 드리프트 감지 후 |
| **입력** | 피처별 PSI, KS-test p-value, 성능 지표 |
| **출력** | 영향 피처 목록, 원인 설명, 재학습 제안 |
| **모델** | gpt-4o-mini (GitHub Models API) |

#### 참고 문서
| 문서 | 용도 |
|------|------|
| `config/thresholds.json` | PSI/KS 임계값 기준 |
| `docs/QUALITY_SCORE.md` | PSI 해석 기준표 |

#### 스킬 (할 수 있는 것)
- ✅ 피처별 PSI/KS 데이터 분석
- ✅ 드리프트 원인 자연어 설명
- ✅ 재학습 방향 제안
- ✅ 분석 결과를 ml_metadata에 기록

#### 권한 범위 (할 수 없는 것)
- ❌ 롤백 결정 금지 (이미 rule-based로 완료된 후 호출됨)
- ❌ 재학습 트리거 금지 (monitor.py가 담당)
- ❌ registry/flags.json 수정 금지

#### fallback 동작
- LLM 호출 실패 → 분석 생략, 롤백/트리거는 정상 진행

---

### 3. pipeline_orchestrator_agent — 파이프라인 오케스트레이터 에이전트

| 항목 | 내용 |
|------|------|
| **파일** | `agents/orchestrator_agent.py` |
| **호출 위치** | `api/pipeline_server.py:_execute_pipeline()` |
| **입력** | trigger_reason, dataset_id, 시스템 상태 (Feature Store / Registry / 최근 실행) |
| **출력** | `RunPlan`: 실행 스테이지 목록 + 스킵 스테이지 + 근거 |
| **모델** | gpt-4o-mini (GitHub Models API) |

#### 참고 문서
| 문서 | 용도 |
|------|------|
| `config/thresholds.json` | 임계값 기준 |
| `registry/model_registry.json` | 현재 배포 모델 상태 |
| `feature_store/features/*/v1/meta.json` | Feature Store 신선도 확인 |
| `ml_metadata/runs.json` | 최근 파이프라인 실행 이력 |

#### 스킬 (할 수 있는 것)
- ✅ Feature Store 신선도 확인 (age_hours 계산)
- ✅ 배포 모델 상태 조회
- ✅ trigger_reason에 따른 스테이지 최적화
- ✅ 스킵 가능 스테이지 결정 및 근거 반환

#### 권한 범위 (할 수 없는 것)
- ❌ 실제 파이프라인 실행 금지 (harness_pipeline.py가 담당)
- ❌ registry/flags.json 수정 금지
- ❌ 다른 에이전트 직접 호출 금지

#### 스테이지 스킵 판단 기준
| trigger_reason | data_eng 스킵 조건 |
|---|---|
| `ci_cd` | 항상 전체 실행 (코드 변경) |
| `drift` | Feature Store age < 24h 이면 스킵 |
| `manual` | Feature Store age < 24h 이면 스킵 |
| `scheduled` | Feature Store age < 24h 이면 스킵 |

#### fallback 동작
- LLM 호출 실패 → rule-based 계획 (Feature Store 신선도만 체크)
- rule-based 실패 → 전체 파이프라인 실행
