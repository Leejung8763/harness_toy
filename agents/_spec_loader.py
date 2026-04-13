"""
agents/_spec_loader.py — 에이전트 지침서 로더

agents/specs/{name}.md 에서 각 에이전트의 spec을 읽습니다.
각 에이전트는 LLM 호출 시 자신의 spec을 system message로 주입합니다.

원칙: "Knowledge that lives in docs or people's heads is not accessible to the system"
     → 모든 판단 기준은 반드시 파일에서 읽어야 한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SPECS_DIR = Path(__file__).parent / "specs"

# 에이전트명 → spec 파일명 매핑
_SPEC_FILES = {
    "orchestrator_agent": "orchestrator.md",
    "data_engineering_agent": "data_engineering.md",
    "model_engineering_agent": "model_engineering.md",
    "train_agent": "train.md",
    "test_agent": "test.md",
}


@lru_cache(maxsize=None)
def load(agent_name: str) -> str:
    """
    agents/specs/{name}.md 에서 해당 에이전트의 spec을 읽어 반환합니다.

    Args:
        agent_name: 에이전트 이름 (e.g. "data_engineering_agent")

    Returns:
        spec 파일 전문.
        파일 없으면 빈 문자열 반환 (에이전트 실행 차단 안 함).
    """
    filename = _SPEC_FILES.get(agent_name)
    if not filename:
        return ""

    spec_path = SPECS_DIR / filename
    try:
        return spec_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
