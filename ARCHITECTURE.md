# Architecture

> Enforce invariants, not implementations.
> 레이어 구조는 린터와 구조 테스트로 기계적으로 강제합니다.

---

## 레이어 구조

의존성은 **아래 방향으로만** 흐릅니다.

```
Types
  ↓
Config
  ↓
Repo
  ↓
Service
  ↓
Runtime
  ↓
UI
```

> TODO: 프로젝트에 맞게 레이어를 조정하고, 각 레이어가 담당하는 경로를 매핑하세요.

| 레이어 | 경로 | 역할 | 허용 import |
|--------|------|------|-------------|
| Types | `TODO` | 데이터 타입, 스키마 정의 | 외부 라이브러리만 |
| Config | `TODO` | 설정 읽기/쓰기 | Types |
| Repo | `TODO` | 데이터 접근 | Types, Config |
| Service | `TODO` | 비즈니스 로직 | Types, Config, Repo |
| Runtime | `TODO` | 오케스트레이션 | 전부 |

---

## 금지 패턴

```python
# ❌ TODO: 이 프로젝트에서 금지할 패턴을 구체적으로 작성하세요
# ✅ TODO: 대신 허용되는 방식을 함께 작성하세요
```

---

## 경계(Boundary) 검증 규칙

외부 데이터가 시스템 내부로 들어오는 지점에서 반드시 유효성을 검증합니다.
검증 방식은 구현자가 선택하되, 검증 자체는 생략 불가입니다.

```python
# TODO: 이 프로젝트의 경계 지점과 검증 예시를 작성하세요
```

---

## 구조 테스트

> TODO: 아키텍처 규칙을 기계적으로 강제하는 린터/테스트를 여기에 문서화하세요.
