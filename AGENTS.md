# AGENTS.md

> **이 파일은 백과사전이 아닌 목차(table of contents)입니다.**
> 모든 지침을 여기에 넣지 마세요. 규칙은 해당 주제 파일에 두고, 여기서는 링크만 유지합니다.
> 참고: [harness_guide_openai.md](harness_guide_openai.md)

---

## 프로젝트 목적

OpenAI Harness 방식을 실습하기 위한 MLOps 장난감 파이프라인입니다.
Data → Train → Evaluate → Deploy → Monitor 전 사이클을 로컬에서 재현하며,
CI/CD Stage(build/test/package)까지 포함한 완전한 ML 시스템 구조를 학습합니다.

---

## 빠른 시작

```bash
# ① CI/CD Stage — 코드 검증 (Build → Test → Package)
./ci/run.sh

# ② Automated Pipeline — ML 파이프라인 실행 (CI/CD 통과 후)
./ci/trigger_pipeline.sh
./ci/trigger_pipeline.sh --drift           # 드리프트 롤백 시나리오
./ci/trigger_pipeline.sh --dataset-id 44120

# 개별 Make 타겟
make build      # 의존성 설치 + 문법 검사
make test       # 단위 테스트 (23개)
make package    # wheel 빌드
make pipeline   # ML 파이프라인 직접 실행
make all        # build + test + package

# 개별 스테이지 실행
python3 pipeline/train.py [dataset_id]
python3 pipeline/evaluate.py
python3 pipeline/deploy.py
python3 pipeline/monitor.py [--drift]
python3 pipeline/predict.py
```

---

## 프로젝트 구조

```
harness_toy/
├── harness_pipeline.py        # 전체 파이프라인 오케스트레이터
├── Makefile                   # CI/CD 타겟 (build/test/package/pipeline)
├── ci/
│   └── run.sh                 # 로컬 CI/CD 실행 스크립트
├── data/
│   └── loader.py              # 데이터 경계 검증 진입점 (OpenML)
├── pipeline/
│   ├── train.py               # Challenger 학습 → Champion 선정
│   ├── evaluate.py            # Quality Gate (ROC-AUC 기준)
│   ├── deploy.py              # 상태 전이 + Feature Flag 업데이트
│   ├── monitor.py             # 배포 후 성능 모니터링 + 자동 롤백
│   └── predict.py             # Feature Flag 기반 예측 서빙
├── tests/
│   ├── test_evaluate.py       # Quality Gate 단위 테스트
│   ├── test_deploy.py         # 상태 전이 단위 테스트
│   ├── test_monitor.py        # 임계값 검사 단위 테스트
│   └── test_predict.py        # 입력 검증 단위 테스트
├── registry/
│   └── model_registry.json    # 모델 버전 및 상태 기록
├── feature_flags/
│   └── flags.json             # 현재 서빙 모델 버전 (deploy.py를 통해서만 변경)
├── models/                    # 학습된 모델 pkl 파일
└── docs/                      # 설계 문서 (아래 색인 참조)
```

---

## 문서 색인

| 문서 | 역할 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 레이어 구조, 의존성 제약, 금지 패턴 |
| [docs/DESIGN.md](docs/DESIGN.md) | 설계 결정의 배경(why) |
| [docs/PLANS.md](docs/PLANS.md) | 로드맵 및 실행 계획 |
| [docs/PRODUCT_SENSE.md](docs/PRODUCT_SENSE.md) | 제품 방향과 판단 기준 |
| [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md) | 품질 게이트 임계값 |
| [docs/RELIABILITY.md](docs/RELIABILITY.md) | 장애 대응 및 롤백 전략 |
| [docs/SECURITY.md](docs/SECURITY.md) | 보안 규칙 |
| [docs/design-docs/](docs/design-docs/) | 상세 설계 문서 |
| [docs/exec-plans/](docs/exec-plans/) | 실행 계획 및 기술 부채 |
| [docs/product-specs/](docs/product-specs/) | 기능 명세 |
| [docs/references/](docs/references/) | LLM 최적화 압축 참조 |

---

## 핵심 불변 규칙 (Invariants)

> 구현 방식은 AI에게 맡기되, 아래 경계는 반드시 지켜야 합니다.

1. **데이터 경계**: 외부 데이터는 반드시 `data/loader.py`를 통해서만 시스템에 진입한다.
2. **Feature Flag**: `flags.json`은 반드시 `pipeline/deploy.py`를 통해서만 변경한다.
3. **배포 순서**: `evaluated_pass` 상태인 모델만 deploy 가능하다 (trained → evaluated_pass → deployed).
4. **의존성 방향**: `pipeline/` → `data/` 단방향만 허용. 역방향 import 금지.

---

## 빠른 시작

```bash
source .venv/bin/activate

# 전체 파이프라인 실행
python harness_pipeline.py

# 드리프트 롤백 시나리오 시연
python harness_pipeline.py --drift

# 다른 데이터셋으로 실행
python harness_pipeline.py --dataset-id 44120

# 개별 스테이지 실행
python pipeline/train.py [dataset_id]
python pipeline/evaluate.py
python pipeline/deploy.py
python pipeline/monitor.py [--drift]
python pipeline/predict.py
```

---

## 에이전트 작업 시 주의사항

> 에이전트가 반복 실패하는 패턴이 발견될 때마다 여기에 추가합니다. (Mitchell Hashimoto 패턴)

### [규칙 1] `from data.loader import ...` — ModuleNotFoundError
- **원인**: Python이 프로젝트 루트를 모듈 경로로 인식하지 못함
- **해결**: `.venv/lib/python3.*/site-packages/harness_mlops.pth`에 프로젝트 루트 경로 등록됨
- **신규 환경 셋업 시**: 프로젝트 루트 경로를 `.pth` 파일에 수동 등록

### [규칙 2] `.venv` 인터프리터 오류
- **원인**: `.venv`가 다른 경로의 Python을 참조 중 (bad interpreter)
- **해결**: `python3` / `pip3` 직접 사용. Makefile과 ci/run.sh는 system python3 사용
