#!/usr/bin/env bash
# ci/trigger_pipeline.sh — Automated Pipeline 트리거
#                          (다이어그램: CI/CD Stage → Automated Pipeline)
#
# 역할: Pipeline Server(8001)에 HTTP POST로 실행 요청
#       서버가 없으면 직접 실행 (fallback)
#
# 사용법:
#   ./ci/trigger_pipeline.sh                    # 기본 데이터셋 (44089)
#   ./ci/trigger_pipeline.sh --dataset-id 44120
#   ./ci/trigger_pipeline.sh --drift            # 드리프트 시나리오 (직접 실행만)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIPELINE_URL="http://localhost:8001"
DATASET_ID=44089

# --dataset-id 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-id) DATASET_ID="$2"; shift 2 ;;
        --drift)      INJECT_DRIFT=true; shift ;;
        *) shift ;;
    esac
done
INJECT_DRIFT=${INJECT_DRIFT:-false}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║      Automated Pipeline 트리거           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Pipeline Server가 실행 중이면 HTTP POST, 아니면 직접 실행
if curl -sf "$PIPELINE_URL/pipeline/status" > /dev/null 2>&1; then
    echo "  → Pipeline Server 감지 ($PIPELINE_URL)"
    echo "  → POST /pipeline/run  dataset_id=$DATASET_ID"
    echo ""

    RESPONSE=$(curl -s -X POST "$PIPELINE_URL/pipeline/run" \
        -H "Content-Type: application/json" \
        -d "{\"dataset_id\": $DATASET_ID, \"inject_drift\": $INJECT_DRIFT}")
    echo "  응답: $RESPONSE"

    RUN_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id','?'))")
    echo ""
    echo "  실행 상태 확인: curl $PIPELINE_URL/pipeline/status"
    echo "  run_id: $RUN_ID"
else
    echo "  ⚠️  Pipeline Server 미실행 → 직접 실행 (fallback)"
    echo "  💡 서버 시작: ./ci/start_pipeline_server.sh"
    echo ""
    ARGS=""
    [[ "$INJECT_DRIFT" == "true" ]] && ARGS="--drift"
    python3 harness_pipeline.py --dataset-id "$DATASET_ID" $ARGS
fi
