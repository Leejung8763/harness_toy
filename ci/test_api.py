#!/usr/bin/env python3
"""
ci/test_api.py — ML Prediction Service 로컬 통합 테스트

사용법:
  python3 ci/test_api.py              # 기본 (localhost:8000)
  python3 ci/test_api.py --port 9000

서버가 먼저 실행 중이어야 합니다:
  ./ci/start_server.sh &
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

BASE_URL = "http://localhost:8000"


def run_tests(base_url: str) -> bool:
    passed = 0
    failed = 0

    print(f"\n ML Prediction Service 통합 테스트 — {base_url}\n")
    print("─" * 50)

    with httpx.Client(base_url=base_url, timeout=10.0) as client:

        # 1. Health check
        _header("GET /health")
        r = client.get("/health")
        _assert_status(r, 200)
        body = r.json()
        _assert("status == ok", body["status"] == "ok")
        print(f"  model_deployed={body['model_deployed']}  version={body['model_version']}")
        passed += 1

        if not body["model_deployed"]:
            print("\n  ⚠️  배포된 모델이 없습니다. ci/trigger_pipeline.sh 를 먼저 실행하세요.")
            return False

        # 2. Model info
        _header("GET /model/info")
        r = client.get("/model/info")
        _assert_status(r, 200)
        info = r.json()
        _assert("version 필드 존재", "version" in info)
        _assert("model_name 필드 존재", "model_name" in info)
        print(f"  {info}")
        passed += 1

        # 3. Predict (샘플 피처로 테스트)
        _header("POST /predict")
        sample = _make_sample_features()
        r = client.post("/predict", json={"features": sample})
        _assert_status(r, 200)
        body = r.json()
        _assert("predictions 반환", len(body["predictions"]) == len(sample))
        _assert("n_samples 일치", body["n_samples"] == len(sample))
        print(f"  predictions={body['predictions']}  version={body['model_version']}")
        passed += 1

        # 4. Predict proba
        _header("POST /predict/proba")
        r = client.post("/predict/proba", json={"features": sample})
        _assert_status(r, 200)
        body = r.json()
        _assert("probabilities 반환", len(body["probabilities"]) == len(sample))
        first_proba = body["probabilities"][0]
        _assert("확률 합 ≈ 1.0", abs(sum(first_proba) - 1.0) < 1e-4)
        print(f"  probabilities[0]={[round(p, 3) for p in first_proba]}")
        passed += 1

        # 5. 빈 features 입력 → 422
        _header("POST /predict (빈 features → 422)")
        r = client.post("/predict", json={"features": []})
        _assert_status(r, 422)
        print(f"  422 Unprocessable Entity ✅")
        passed += 1

    print("\n" + "─" * 50)
    print(f"  결과: {passed} passed / {failed} failed")
    return failed == 0


def _make_sample_features() -> list[dict]:
    """실제 모델이 학습한 피처 수에 맞는 샘플을 생성합니다."""
    import json
    from pathlib import Path
    registry_path = Path("registry/model_registry.json")
    if registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)
        deployed = [m for m in reg.get("models", []) if m["status"] == "deployed"]
        if deployed:
            import pickle
            with open(deployed[-1]["model_path"], "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            n_features = model.n_features_in_
            return [{f"f{i}": float(i) for i in range(n_features)}]
    return [{"f0": 1.0, "f1": 2.0}]


def _header(title: str) -> None:
    print(f"\n  [{title}]")


def _assert_status(r: httpx.Response, expected: int) -> None:
    if r.status_code != expected:
        print(f"  ❌ 상태코드 {r.status_code} (기대: {expected})")
        print(f"     {r.text[:200]}")
        sys.exit(1)
    print(f"  HTTP {r.status_code} ✅")


def _assert(label: str, condition: bool) -> None:
    if not condition:
        print(f"  ❌ FAIL: {label}")
        sys.exit(1)
    print(f"  ✅ {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ok = run_tests(f"http://localhost:{args.port}")
    sys.exit(0 if ok else 1)
