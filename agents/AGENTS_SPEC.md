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

### 2. drift_agent — 드리프트 원인 분석 에이전트 *(예정)*

| 항목 | 내용 |
|------|------|
| **파일** | `agents/drift_agent.py` (미구현) |
| **호출 위치** | `pipeline/monitor.py` 드리프트 감지 후 |
| **입력** | PSI/KS-test 결과, 피처별 분포 변화 |
| **출력** | 드리프트 원인 설명, 영향 피처 목록 |

---

### 3. pipeline_orchestrator_agent — 파이프라인 오케스트레이터 에이전트 *(예정)*

| 항목 | 내용 |
|------|------|
| **파일** | `agents/orchestrator_agent.py` (미구현) |
| **호출 위치** | `api/pipeline_server.py` |
| **입력** | 시스템 전체 상태, 트리거 원인 |
| **출력** | 실행할 파이프라인 스테이지 결정 |
