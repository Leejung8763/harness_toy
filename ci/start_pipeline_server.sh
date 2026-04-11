#!/usr/bin/env bash
# ci/start_pipeline_server.sh — Automated Pipeline Service 로컬 실행 (port 8001)
#
# 다이어그램: Automated Pipeline 영역
#
# 사용법:
#   ./ci/start_pipeline_server.sh           # 포그라운드 실행
#   ./ci/start_pipeline_server.sh --reload  # 개발 모드

set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT=8001
RELOAD=${1:-}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Automated Pipeline Service             ║"
echo "║   http://localhost:${PORT}                  ║"
echo "║   http://localhost:${PORT}/docs             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  POST /pipeline/run    → 파이프라인 실행 요청"
echo "  GET  /pipeline/status → 실행 상태 확인"
echo ""

if [[ "$RELOAD" == "--reload" ]]; then
    python3 -m uvicorn api.pipeline_server:app --host 0.0.0.0 --port "$PORT" --reload
else
    python3 -m uvicorn api.pipeline_server:app --host 0.0.0.0 --port "$PORT"
fi
