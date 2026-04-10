#!/usr/bin/env bash
# ci/run.sh — 로컬 CI/CD 파이프라인
# 다이어그램 CI/CD Stage: Build → Test → Package → Deploy(pipeline)
#
# 사용법:
#   ./ci/run.sh              # build + test + package (pipeline 제외)
#   ./ci/run.sh --full       # build + test + package + pipeline (ML 학습 포함)

set -euo pipefail

FULL=false
[[ "${1:-}" == "--full" ]] && FULL=true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         CI/CD Pipeline (local)           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Stage 1: Build
echo "──────────────────────────────────────────"
echo "  STAGE 1 / 4  —  Build"
echo "──────────────────────────────────────────"
make build

# Stage 2: Test
echo ""
echo "──────────────────────────────────────────"
echo "  STAGE 2 / 4  —  Test"
echo "──────────────────────────────────────────"
make test

# Stage 3: Package
echo ""
echo "──────────────────────────────────────────"
echo "  STAGE 3 / 4  —  Package"
echo "──────────────────────────────────────────"
make package

# Stage 4: Deploy pipeline (--full 옵션 시에만 실행)
echo ""
echo "──────────────────────────────────────────"
echo "  STAGE 4 / 4  —  Deploy Pipeline"
echo "──────────────────────────────────────────"
if $FULL; then
    make pipeline
else
    echo "  ⏭  스킵 (ML 학습 포함 실행: ./ci/run.sh --full)"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  CI/CD 완료                          ║"
echo "╚══════════════════════════════════════════╝"
