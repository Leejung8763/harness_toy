# Plans

> 로드맵, 진행 중인 작업, 완료된 마일스톤을 추적합니다.

---

## 현재 상태

- [x] **Source Repository** — 로컬 git 초기화 (`git init`)

- [x] **DS Development** — 파이프라인 코드 및 기본 로직
  - `harness_pipeline.py`, `pipeline/` 전체, `data/loader.py`

- [x] **CI/CD Stage** — Build / Test / Package
  - `Makefile` (build, test, package, pipeline, all, clean)
  - `ci/run.sh` (CI/CD Stage — 코드 검증만, ML 파이프라인 제외)
  - `ci/trigger_pipeline.sh` (Automated Pipeline HTTP 트리거)
  - `tests/` (단위 테스트 43개)
  - `tests/README.md` (테스트 명세)

- [x] **Automated Pipeline** — Feature Store + Data Engineering + ML Metadata Store
  - `feature_store/store.py` — 피처 캐시 (Parquet)
  - `pipeline/data_engineering.py` — 피처 엔지니어링 스테이지
  - `ml_metadata/store.py` — 실험 이력·데이터 리니지 추적
  - `pipeline/train.py` — Feature Store 우선 사용 + ML Metadata 기록
  - `registry/model_registry.json` — 모델 버전 및 배포 상태
  - `api/pipeline_server.py` — Automated Pipeline Service (port 8001)
  - `ci/start_pipeline_server.sh` — Pipeline Server 시작 스크립트

- [x] **Operations** — Prediction Service + Trigger 루프
  - `api/serve.py` — ML Prediction Service (port 8000)
  - `ci/start_server.sh` — Prediction Server 시작 스크립트
  - `ci/test_api.py` — 통합 테스트
  - `pipeline/monitor.py` — 드리프트 감지 → 롤백 → `_trigger_retraining()` → POST 8001

---

## 서비스 통신 흐름

```
[CI/CD Stage]
  ci/run.sh → build / test / package
       ↓ (CI 통과 후)
  ci/trigger_pipeline.sh → POST localhost:8001/pipeline/run

[Automated Pipeline Service :8001]
  api/pipeline_server.py → harness_pipeline.py (백그라운드 스레드)
       ↓ (파이프라인 완료)
  새 모델 → deploy.py → feature_flags/flags.json 업데이트

[Operations - Prediction Service :8000]
  api/serve.py → predict.py (Feature Flag 기반 서빙)

[Operations - Monitor]
  pipeline/monitor.py → 드리프트 감지
       ↓ (롤백 후)
  _trigger_retraining() → POST localhost:8001/pipeline/run  (Trigger Loop)
```

---

## 단기 로드맵

- [ ] **Performance Monitoring 고도화** — 운영 메트릭 대시보드 / 알림
- [ ] **A/B Testing** — Feature Flag 기반 모델 트래픽 분산
- [ ] **Data Versioning** — Feature Store 버전 관리 + 롤백

---

## 장기 방향

OpenAI Harness 원칙을 체득하여, 실제 프로덕션 MLOps 시스템 설계에 적용합니다.
AI 에이전트가 최소한의 컨텍스트로 정확하게 작업할 수 있는 코드베이스 구조를 목표로 합니다.

