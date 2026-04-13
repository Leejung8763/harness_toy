# agents/AGENTS_SPEC.md — 에이전트 명세 목차

> 각 에이전트의 상세 spec은 `agents/specs/` 에 있습니다.
> 에이전트를 추가할 때 `agents/specs/`에 파일을 추가하고 여기에 링크를 등록하세요.

---

## 에이전트 설계 원칙

1. **에이전트는 판단만 한다** — 실제 파일 변경·모델 저장은 `core/`가 담당
2. **Fallback 필수** — LLM 실패 시 rule-based 로직으로 자동 대체
3. **Spec 참조 필수** — 모든 에이전트는 LLM 호출 시 자신의 spec을 system message로 주입
4. **Pydantic 계약** — 에이전트 출력은 반드시 Pydantic 모델로 검증

---

## 에이전트 목록

| 에이전트 | Spec 파일 | 역할 |
|---------|---------|------|
| orchestrator_agent | [specs/orchestrator.md](specs/orchestrator.md) | 사용자 지시 → 실행 계획 수립 |
| data_engineering_agent | [specs/data_engineering.md](specs/data_engineering.md) | 전처리 전략 결정 |
| model_engineering_agent | [specs/model_engineering.md](specs/model_engineering.md) | 모델 선택 결정 |
| train_agent | [specs/train.md](specs/train.md) | 학습 전략 결정 |
| test_agent | [specs/test.md](specs/test.md) | 평가 결과 판단 |

