# AGENTS.md

> **이 파일은 백과사전이 아닌 목차(table of contents)입니다.**
> 모든 지침을 여기에 넣지 마세요. 규칙은 해당 주제 파일에 두고, 여기서는 링크만 유지합니다.
> 참고: [harness_guide_openai.md](harness_guide_openai.md)

---

## 프로젝트 목적

> TODO: 이 프로젝트가 왜 존재하는지 한 문단으로 작성하세요.

---

## 빠른 시작

```bash
# TODO: 환경 셋업 및 실행 명령
```

---

## 프로젝트 구조

```
TODO: 디렉토리 트리를 여기에 작성하세요
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

1. TODO: 첫 번째 invariant
2. TODO: 두 번째 invariant

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
- **신규 환경 셋업 시**: `echo "$(pwd)" > .venv/lib/python3.*/site-packages/harness_mlops.pth`
