# OPENAI 지침서

- Give Codex a map, not a 1,000-page instruction manual
  - AGENTS.md를 백과사전이 아닌 목차(table of contents)로 취급
  - 모든 지침을 한 파일에 넣지 않고, 특정 도구 사용법이나 코딩 규칙 등으로 나누어 관리
- Enforce invariants, not implementations
   - 경계(데이터가 시스템 내부로 들어오는)에서 데이터 형태를 파싱하도록 요구하되, 방식은 AI에게 맡김
   - 엄격한 아키텍처 모델(Types → Config → Repo → Service → Runtime → UI)을 커스텀 린터와 구조적 테스트로 기계적으로 강제
- Knowledge that lives in Google Docs, chat threads, or people's heads are not accessible to the system
  - 구글 독스, 슬랙 대화, 혹은 개발자의 머릿속에만 있는 '암묵적 지식'은 에이전트에게 존재하지 않는 데이터와 같음
  - 모든 비즈니스 로직, 의사결정 배경, 도메인 지식을 에이전트가 접근 가능한 코드베이스나 문서(Repo)로 작성
- Technical debt is like a high-interest loan
  - 복잡하고 일관성 없는 코드는 AI에게 극심한 혼란과 추론 비용(Token & Error) 의 급격한 상승을 초래
  - AI-Native 리팩토링은 더 적은 컨텍스트로도 정확한 작업을 수행할 수 있도록 AI의 작업 환경을 최적화하는 필수 작업

# OPENAI HARNESS DIR TREE
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   ├── new-user-onboarding.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   ├── uv-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md