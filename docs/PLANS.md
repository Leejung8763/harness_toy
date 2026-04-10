# Plans

> 로드맵, 진행 중인 작업, 완료된 마일스톤을 추적합니다.
> 세부 실행 계획은 [exec-plans/active/](exec-plans/active/) 참조.
> 기술 부채는 [exec-plans/tech-debt-tracker.md](exec-plans/tech-debt-tracker.md) 참조.

---

## 현재 상태

- [x] **Source Repository** — 로컬 git 초기화 (`git init`)
- [x] **CI/CD Stage** — Build / Test / Package / Deploy pipeline
  - `Makefile` (build, test, package, pipeline, all, clean)
  - `ci/run.sh` (로컬 CI/CD 실행 스크립트)
  - `tests/` (단위 테스트 23개 — evaluate, deploy, monitor, predict)
- [x] **ML Model Engineering** — Train → Evaluate → Deploy → Monitor
  - `pipeline/train.py` — 5개 Challenger → Champion 선정
  - `pipeline/evaluate.py` — Quality Gate (ROC-AUC)
  - `pipeline/deploy.py` — 상태 전이 + Feature Flag
  - `pipeline/monitor.py` — 성능 모니터링 + 자동 롤백
  - `pipeline/predict.py` — Feature Flag 기반 서빙

---

## 단기 로드맵

- [ ] **Automated Pipeline 영역** (다이어그램 노란 영역)
  - Feature Store 개념 추가
  - Data Engineering 단계 분리
  - ML Metadata Store 연동
- [ ] **ML Prediction Service** — REST API 서빙 (FastAPI 등)
- [ ] **Performance Monitoring** — 운영 메트릭 대시보드/알림

---

## 장기 방향

OpenAI Harness 원칙을 체득하여, 실제 프로덕션 MLOps 시스템 설계에 적용합니다.
AI 에이전트가 최소한의 컨텍스트로 정확하게 작업할 수 있는 코드베이스 구조를 목표로 합니다.
