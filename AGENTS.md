# AGENTS.md — 목차

> 이 파일은 목차(table of contents)입니다. 규칙은 해당 문서에 두고 여기서는 링크만 유지합니다.

---

## 프로젝트 목적

LLM 에이전트가 각 단계의 핵심 결정을 내리는 ML 파이프라인입니다.
에이전트는 **판단만** 하고, 실제 실행은 `core/` 코드가 담당합니다.

---

## 서비스 구조

```
data/loader.py          ← OpenML 데이터 진입점 (유일)
        │
        ▼
core/data_engineering.py   ← data_engineering_agent 판단 실행
        │
        ▼
core/model_engineering.py  ← model_engineering_agent 판단 실행
        │
        ▼
core/train.py              ← train_agent 판단 실행
        │
        ▼
core/test.py               ← test_agent 판단 실행
```

---

## 에이전트 목록

| 에이전트 | 파일 | 판단 내용 |
|---------|------|---------|
| Data Engineering Agent | `agents/data_engineering_agent.py` | 피처 선택, 전처리 전략 |
| Model Engineering Agent | `agents/model_engineering_agent.py` | 모델 타입, 하이퍼파라미터 후보 |
| Train Agent | `agents/train_agent.py` | CV 전략, early stopping |
| Test Agent | `agents/test_agent.py` | 평가 지표 해석, 합격/불합격 |

자세한 명세: `agents/AGENTS_SPEC.md`

---

## 핵심 불변 규칙

1. **데이터 경계**: 외부 데이터는 반드시 `data/loader.py`를 통해서만 진입
2. **에이전트는 판단만**: 파일 변경·모델 저장은 `core/`가 담당
3. **Fallback 필수**: 모든 에이전트는 LLM 실패 시 rule-based로 자동 대체
4. **의존성 방향**: `core/` → `data/` 단방향만 허용

---

## 에이전트 작업 시 주의사항

### [규칙 1] ModuleNotFoundError
- **해결**: `PYTHONPATH=$(pwd) python run_pipeline.py`

### [규칙 2] GitHub Models API 토큰
- **확인**: `gh auth token` 으로 토큰 발급 확인
- **모델**: `gpt-4o-mini` (base_url: `https://models.inference.ai.azure.com`)
