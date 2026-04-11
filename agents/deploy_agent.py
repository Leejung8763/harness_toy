"""
agents/deploy_agent.py — 배포 판단 LLM 에이전트

역할:
  - evaluate.py의 평가 결과를 받아 LLM이 배포 여부를 판단
  - 단순 숫자 비교(ROC-AUC >= champion)를 넘어 맥락을 고려한 판단
  - 판단 근거를 자연어로 설명

하네스 역할:
  - LLM 판단이 틀려도 evaluate.py의 Quality Gate가 최종 안전망
  - LLM이 "배포" 판단해도 BASELINE_THRESHOLD 미달이면 실제 배포 안 됨
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REGISTRY_PATH = Path("registry/model_registry.json")
METADATA_PATH = Path("ml_metadata/runs.json")


def _get_github_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"]).decode().strip()


def _get_llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_github_token(),
    )


def judge_deployment(candidate_version: str) -> dict:
    """
    LLM이 후보 모델의 배포 여부를 판단합니다.

    Args:
        candidate_version: 평가할 모델 버전 (예: "v3")

    Returns:
        {
            "decision": "deploy" | "hold" | "reject",
            "reason": "판단 근거 (한국어)",
            "confidence": 0.0~1.0,
        }
    """
    context = _build_context(candidate_version)
    if context is None:
        return {"decision": "hold", "reason": "모델 정보를 찾을 수 없습니다.", "confidence": 0.0}

    prompt = _build_prompt(context)
    response = _call_llm(prompt)
    result = _parse_response(response)

    _print_judgment(candidate_version, result)
    return result


def _build_context(version: str) -> dict | None:
    """레지스트리와 메타데이터에서 판단에 필요한 맥락을 수집합니다."""
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"models": []}

    candidate = next((m for m in registry["models"] if m["version"] == version), None)
    if candidate is None:
        return None

    champion = next(
        (m for m in reversed(registry["models"]) if m["status"] == "deployed"),
        None,
    )

    # 최근 학습 메타데이터
    runs = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else []
    training_runs = [r for r in runs if r.get("stage") == "training"]
    latest_training = training_runs[-1] if training_runs else {}

    return {
        "candidate": candidate,
        "champion": champion,
        "latest_training": latest_training,
        "total_versions": len(registry["models"]),
    }


def _build_prompt(ctx: dict) -> str:
    candidate = ctx["candidate"]
    champion = ctx["champion"]
    training = ctx["latest_training"]

    champion_info = (
        f"현재 배포 중인 챔피언: {champion['version']} "
        f"(ROC-AUC: {champion.get('metrics', {}).get('roc_auc', 'N/A')})"
        if champion else "현재 배포 중인 모델 없음 (첫 배포)"
    )

    return f"""당신은 MLOps 배포 판단 에이전트입니다.
아래 정보를 바탕으로 신규 모델의 배포 여부를 판단하세요.

## 신규 후보 모델
- 버전: {candidate['version']}
- 모델명: {candidate.get('model_name', 'N/A')}
- 데이터셋: {candidate.get('dataset_name', 'N/A')}
- ROC-AUC: {candidate.get('metrics', {}).get('roc_auc', 'N/A')}
- 상태: {candidate['status']}

## 기존 챔피언
{champion_info}

## 학습 메타데이터
- Feature Store 사용: {training.get('feature_store_hit', 'N/A')}
- 학습 샘플 수: {training.get('n_train_samples', 'N/A')}
- 피처 수: {training.get('n_features', 'N/A')}

## 판단 기준
1. ROC-AUC가 챔피언보다 높거나 같으면 일반적으로 배포
2. 성능 차이가 0.001 미만이면 변경 비용 대비 효익이 낮음
3. 첫 배포는 ROC-AUC >= 0.70이면 배포

## 응답 형식 (JSON만 반환)
{{
  "decision": "deploy" | "hold" | "reject",
  "reason": "한국어로 판단 근거 1~2문장",
  "confidence": 0.0~1.0
}}"""


def _call_llm(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_response(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return {
            "decision": data.get("decision", "hold"),
            "reason": data.get("reason", "판단 불가"),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception:
        return {"decision": "hold", "reason": f"응답 파싱 실패: {raw[:100]}", "confidence": 0.0}


def _print_judgment(version: str, result: dict) -> None:
    icon = {"deploy": "🚀", "hold": "⏸️", "reject": "🚫"}.get(result["decision"], "❓")
    print(f"\n[deploy_agent] {icon} 판단: {result['decision'].upper()}  "
          f"(신뢰도: {result['confidence']:.0%})")
    print(f"  근거: {result['reason']}")


if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else None

    if not version:
        # 최신 evaluated_pass 모델 자동 선택
        registry = json.loads(REGISTRY_PATH.read_text())
        candidates = [m for m in registry["models"] if m["status"] == "evaluated_pass"]
        if not candidates:
            print("판단할 evaluated_pass 모델이 없습니다.")
            sys.exit(1)
        version = candidates[-1]["version"]

    result = judge_deployment(version)
    sys.exit(0 if result["decision"] == "deploy" else 1)
