#!/usr/bin/env bash
# ci/start_server.sh — ML Prediction Service 로컬 실행 (port 8000)
#
# 다이어그램: CD Stage: ML Model Serving → ML Prediction Service (Operations)
#
# 사용법:
#   ./ci/start_server.sh           # 포그라운드 실행
#   ./ci/start_server.sh --reload  # 코드 변경 시 자동 재시작 (개발 모드)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT=${PORT:-8000}
RELOAD=${1:-}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ML Prediction Service (localhost)      ║"
echo "║   http://localhost:${PORT}                  ║"
echo "║   http://localhost:${PORT}/docs             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if [[ "$RELOAD" == "--reload" ]]; then
    python3 -m uvicorn api.serve:app --host 0.0.0.0 --port "$PORT" --reload
else
    python3 -m uvicorn api.serve:app --host 0.0.0.0 --port "$PORT"
fi
