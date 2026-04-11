# AGENTS.md

> **이 파일은 백과사전이 아닌 목차(table of contents)입니다.**
> 모든 지침을 여기에 넣지 마세요. 규칙은 해당 주제 파일에 두고, 여기서는 링크만 유지합니다.
> 참고: [harness_guide_openai.md](harness_guide_openai.md)

---

## 프로젝트 목적

OpenAI Harness 방식을 실습하기 위한 MLOps 장난감 파이프라인입니다.
MLOps 다이어그램(DS Development → CI/CD → Automated Pipeline → Operations)의
전 사이클을 로컬에서 서비스 분리 구조로 재현합니다.

---

## 서비스 구조 (다이어그램 매핑)

| 다이어그램 영역 | 실행 명령 | 포트 |
|----------------|----------|------|
| DS Development | 코드 작성 단계 (직접 실행 없음) | — |
| CI/CD Stage | `./ci/run.sh` | — |
| Automated Pipeline | `./ci/start_pipeline_server.sh` | 8001 |
| Operations (Serving) | `./ci/start_server.sh` | 8000 |
| Operations (Trigger) | `monitor.py` 내부에서 자동 POST | → 8001 |

---

## 빠른 시작

```bash
# ─── 서버 시작 (각각 별도 터미널) ───────────────────────────────
./ci/start_pipeline_server.sh     # Automated Pipeline (port 8001)
./ci/start_server.sh              # ML Prediction Service (port 8000)

# ─── CI/CD Stage ─────────────────────────────────────────────────
./ci/run.sh                       # build → test → package

# ─── Automated Pipeline 실행 ─────────────────────────────────────
./ci/trigger_pipeline.sh          # POST localhost:8001/pipeline/run
./ci/trigger_pipeline.sh --dataset-id 44120
./ci/trigger_pipeline.sh --drift  # 드리프트 롤백 + Trigger 시나리오

# ─── Operations 테스트 ───────────────────────────────────────────
python3 ci/test_api.py            # 예측 API 통합 테스트
curl http://localhost:8000/health
curl http://localhost:8001/pipeline/status

# ─── Make 타겟 ───────────────────────────────────────────────────
make build      # 의존성 설치 + 문법 검사
make test       # 단위 테스트 (43개)
make package    # wheel 빌드
make pipeline   # ML 파이프라인 직접 실행 (서버 없이)
make all        # build + test + package
```

---

## 프로젝트 구조

```
harness_toy/
├── harness_pipeline.py            # 전체 파이프라인 오케스트레이터
├── Makefile                       # CI/CD 타겟 (build/test/package/pipeline)
│
├── ci/                            # CI/CD + 서버 실행 스크립트
│   ├── run.sh                     # CI/CD Stage (build→test→package)
│   ├── trigger_pipeline.sh        # Automated Pipeline 트리거
│   ├── start_pipeline_server.sh   # Pipeline Server 시작 (port 8001)
│   ├── start_server.sh            # Prediction Server 시작 (port 8000)
│   └── test_api.py                # Prediction API 통합 테스트
│
├── api/                           # 서비스 API 레이어
│   ├── pipeline_server.py         # Automated Pipeline Service (port 8001)
│   └── serve.py                   # ML Prediction Service (port 8000)
│
├── data/
│   └── loader.py                  # 데이터 경계 검증 진입점 (OpenML)
│
├── pipeline/                      # 파이프라인 비즈니스 로직
│   ├── data_engineering.py        # 피처 엔지니어링 → Feature Store 저장
│   ├── train.py                   # Challenger 학습 → Champion 선정
│   ├── evaluate.py                # Quality Gate (ROC-AUC)
│   ├── deploy.py                  # 상태 전이 + Feature Flag 업데이트
│   ├── monitor.py                 # 성능 모니터링 + 롤백 + Trigger
│   └── predict.py                 # Feature Flag 기반 예측
│
├── feature_store/
│   └── store.py                   # 피처 캐시 (Parquet)
│
├── ml_metadata/
│   └── store.py                   # 실험 이력·데이터 리니지 추적
│
├── tests/                         # 단위 테스트 (43개)
│   ├── README.md
│   ├── test_api.py
│   ├── test_deploy.py
│   ├── test_evaluate.py
│   ├── test_feature_store.py
│   ├── test_ml_metadata.py
│   ├── test_monitor.py
│   └── test_predict.py
│
├── registry/
│   └── model_registry.json        # 모델 버전 및 배포 상태
├── feature_flags/
│   └── flags.json                 # 현재 서빙 모델 (deploy.py를 통해서만 변경)
├── models/                        # 학습된 모델 pkl 파일
└── docs/                          # 설계 문서 (아래 색인 참조)
```

---

## 문서 색인

| 문서 | 역할 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 서비스 구조, 레이어 의존성, 금지 패턴 |
| [docs/DESIGN.md](docs/DESIGN.md) | 설계 결정의 배경(why) |
| [docs/PLANS.md](docs/PLANS.md) | 로드맵 및 실행 계획 |
| [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md) | 품질 게이트 임계값 |
| [docs/RELIABILITY.md](docs/RELIABILITY.md) | 장애 대응 및 롤백 전략 |
| [docs/SECURITY.md](docs/SECURITY.md) | 보안 규칙 |
| [tests/README.md](tests/README.md) | 테스트 전략 및 명세 |

---

## 핵심 불변 규칙 (Invariants)

> 구현 방식은 AI에게 맡기되, 아래 경계는 반드시 지켜야 합니다.

1. **데이터 경계**: 외부 데이터는 반드시 `data/loader.py`를 통해서만 시스템에 진입한다.
2. **Feature Flag**: `flags.json`은 반드시 `pipeline/deploy.py`를 통해서만 변경한다.
3. **배포 순서**: `evaluated_pass` 상태인 모델만 deploy 가능하다 (trained → evaluated_pass → deployed).
4. **의존성 방향**: `pipeline/` → `data/` 단방향만 허용. 역방향 import 금지.
5. **서비스 통신**: Operations → Automated Pipeline 재실행은 반드시 HTTP POST로만 요청한다 (직접 함수 호출 금지).

---

## 에이전트 작업 시 주의사항

> 에이전트가 반복 실패하는 패턴이 발견될 때마다 여기에 추가합니다. (Mitchell Hashimoto 패턴)

### [규칙 1] `from data.loader import ...` — ModuleNotFoundError
- **원인**: system `python3`는 `.venv`의 `.pth` 파일을 읽지 않아 프로젝트 루트가 경로에 없음
- **해결**: 직접 실행 시 `PYTHONPATH=$(pwd) python3 pipeline/monitor.py` 처럼 명시
- **ci/ 스크립트**: `export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"` 포함됨
- **Makefile**: `export PYTHONPATH := $(shell pwd)` 포함됨
- **`.pth` 파일 경로 주의**: `.venv/lib/python3.*/site-packages/harness_mlops.pth`가 현재 프로젝트를 가리키는지 확인 (`/Users/leejung/harness_toy` 이어야 함)

### [규칙 2] `.venv` 인터프리터 오류
- **원인**: `.venv`가 다른 경로의 Python을 참조 중 (bad interpreter)
- **해결**: `python3` / `pip3` 직접 사용. Makefile과 ci/run.sh는 system python3 사용

### [규칙 3] `/predict` 422 Unprocessable Entity
- **원인**: `features`는 `list[dict[str, float]]` 형식 필요. `list[float]` 불가
- **올바른 형식**: `{"features": [{"컬럼명": 값, ...}]}`
- **컬럼명 확인**: `registry/model_registry.json`의 deployed 모델 pkl에서 `model.feature_names_in_` 참조

### [규칙 4] `/predict` 500 Internal Server Error
- **원인**: 피처 컬럼명이 모델 학습 시 컬럼명과 불일치 (더미 이름 `f0, f1...` 사용 시 발생)
- **해결**: `ci/test_api.py`의 `_make_sample_features()`가 실제 `feature_names_in_` 사용하도록 수정됨
- **확인**: `python3 ci/test_api.py` 실행 시 모든 테스트 통과 확인

### [규칙 6] `'dict' object has no attribute 'append'` — ml_metadata 파일 형식 오류
- **원인**: `ml_metadata/runs.json`이 `{"next_version": 1, "runs": []}` (dict) 형식으로 남아있음
- **해결**: `echo '[]' > ml_metadata/runs.json` 으로 list 형식으로 교체
- **발생 시점**: `ci/reset.sh` 수정 전에 생성된 파일이 잔존할 때
- **원인**: 이전 서버 프로세스가 종료되지 않은 상태에서 재시작 시도
- **해결**: `lsof -ti:8000 | xargs kill -9` / `lsof -ti:8001 | xargs kill -9`

