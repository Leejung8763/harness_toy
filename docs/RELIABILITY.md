# Reliability

> 시스템이 이상 상황에서 어떻게 동작하는지 정의합니다.
> 자동 롤백, 장애 시나리오 대응, 복구 절차를 다룹니다.

---

## 드리프트 발생 시 대응 전략

드리프트가 감지되면 **즉각 복구(롤백) + 근본 해결(재학습)** 을 동시에 수행합니다.

```
드리프트 감지 (ROC-AUC < 0.75 또는 ErrorRate > 0.25)
  ├─ 즉시: 현재 모델 → rolled_back
  │        이전 모델(retired) → deployed  (서비스 다운타임 방지)
  └─ 동시: POST localhost:8001/pipeline/run  (재학습 트리거)
                    ↓
           새 모델(v_next) 학습 → 평가 → 배포
```

> **롤백만 하면**: 오래된 모델로 계속 서비스 → 성능 개선 없음
> **재학습만 하면**: 학습 완료 전까지 성능 나쁜 모델이 서빙 → 다운타임
> **둘 다 해야**: 즉각 안정화 + 장기 개선

---

## 자동 복구 상태 전이

```
trained → evaluated_pass → deployed → (드리프트) → rolled_back
                                              ↑              ↓
                                        retired ←── 복원됨   재학습 트리거
```

---

## 장애 시나리오 대응

| 시나리오 | 감지 방법 | 자동 대응 | 수동 개입 필요 |
|----------|-----------|-----------|----------------|
| 모델 성능 저하(드리프트) | ROC-AUC < 0.75 또는 ErrorRate > 0.25 | 롤백 + 재학습 트리거 | 없음 |
| Pipeline Server 미실행 | httpx 연결 실패 | 경고 출력 후 graceful 종료 | `./ci/start_pipeline_server.sh` |
| 복원할 이전 모델 없음 | retired 모델 목록 비어있음 | flags.json 비활성화 | 수동 재학습 필요 |
| 배포된 모델 없음 | flags.json `active_model_version: null` | Prediction API 503 반환 | `./ci/trigger_pipeline.sh` |

---

## 복구 불가 케이스

1. **모든 모델 버전이 rolled_back** — 수동으로 `./ci/trigger_pipeline.sh` 실행
2. **Feature Store 손상** — `feature_store/features/` 디렉토리 삭제 후 재실행 (자동 재생성)
3. **Registry 손상** — `registry/model_registry.json` 삭제 후 재실행 (빈 상태로 재생성)

---

## 현재 드리프트 감지 방식 (시뮬레이션)

> ⚠️ 현재는 `--drift` 플래그로 인위적 노이즈를 주입하는 시뮬레이션입니다.
> B단계(Monitoring 고도화)에서 실제 분포 변화 감지(PSI, KS-test)로 교체 예정.

```python
# pipeline/monitor.py
if inject_drift:
    noise_scale = round_num * 2.0
    X_sample = X_sample + rng.normal(0, noise_scale, X_sample.shape)
```

