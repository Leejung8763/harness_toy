# harness_toy — 모델 개발 프로세스 에이전트

LLM 에이전트가 각 단계의 핵심 결정을 내리고, 실제 실행은 `core/` 코드가 담당하는
ML 파이프라인 실습 프로젝트입니다.

## 4개 에이전트 영역

| 영역 | 에이전트 | 실행 코드 |
|------|---------|---------|
| Data Engineering | `agents/data_engineering_agent.py` | `core/data_engineering.py` |
| ML Model Engineering | `agents/model_engineering_agent.py` | `core/model_engineering.py` |
| Train | `agents/train_agent.py` | `core/train.py` |
| Test | `agents/test_agent.py` | `core/test.py` |

## 빠른 시작

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## 설계 원칙
- **에이전트는 판단만** — LLM이 파라미터/전략 결정, 실행은 코드
- **Fallback 필수** — LLM 실패 시 rule-based로 자동 대체
- **데이터 경계** — 외부 데이터는 `data/loader.py`를 통해서만 진입
