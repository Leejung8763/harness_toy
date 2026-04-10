#!/usr/bin/env bash
# ci/trigger_pipeline.sh — Automated Pipeline 트리거
#                          (다이어그램: CI/CD Stage → Automated Pipeline)
#
# 역할: CI/CD Stage 통과 후 ML 파이프라인을 실행
#       실제 환경에서는 CI/CD 시스템이 자동 호출
#
# 사용법:
#   ./ci/trigger_pipeline.sh                   # 기본 데이터셋
#   ./ci/trigger_pipeline.sh --dataset-id 44120
#   ./ci/trigger_pipeline.sh --drift           # 드리프트 롤백 시나리오

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║      Automated Pipeline 트리거           ║"
echo "║  Train → Evaluate → Deploy → Monitor     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

python3 harness_pipeline.py "$@"
