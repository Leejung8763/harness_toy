"""
agents/_spec_loader.py — 에이전트 지침서 로더

AGENTS_SPEC.md에서 에이전트별 섹션을 추출합니다.
각 에이전트는 LLM 호출 시 자신의 spec 섹션을 system message로 주입합니다.

원칙: "Knowledge that lives in docs or people's heads is not accessible to the system"
     → 모든 판단 기준은 반드시 파일에서 읽어야 한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SPEC_PATH = Path(__file__).parent / "AGENTS_SPEC.md"

# AGENTS_SPEC.md 섹션 헤더와 에이전트명 매핑
_SECTION_HEADERS = {
    "data_engineering_agent": "## 1. data_engineering_agent",
    "model_engineering_agent": "## 2. model_engineering_agent",
    "train_agent": "## 3. train_agent",
    "test_agent": "## 4. test_agent",
}


@lru_cache(maxsize=None)
def load(agent_name: str) -> str:
    """
    AGENTS_SPEC.md에서 해당 에이전트의 spec 섹션을 읽어 반환합니다.

    Args:
        agent_name: 에이전트 이름 (e.g. "data_engineering_agent")

    Returns:
        해당 섹션의 전문 문자열.
        파일 없거나 섹션 없으면 빈 문자열 반환 (에이전트 실행 차단 안 함).
    """
    header = _SECTION_HEADERS.get(agent_name)
    if not header:
        return ""

    try:
        content = SPEC_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

    lines = content.splitlines()

    # 해당 섹션 시작 찾기
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header.strip()):
            start = i
            break

    if start is None:
        return ""

    # 다음 동일 레벨(##) 섹션 전까지 추출
    section_lines = []
    for line in lines[start + 1:]:
        if line.startswith("## ") and section_lines:
            break
        section_lines.append(line)

    return header + "\n" + "\n".join(section_lines).strip()
