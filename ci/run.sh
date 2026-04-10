#!/usr/bin/env bash
# ci/run.sh — CI/CD Stage (다이어그램: Source Repository → CI/CD Stage)
#
# 역할: 코드 검증만 담당 (Build → Test → Package)
# ML 파이프라인 실행은 ci/trigger_pipeline.sh 사용
#
# 사용법:
#   ./ci/run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║      CI/CD Stage (코드 검증)             ║"
echo "║  Build → Test → Package                  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Stage 1: Build
echo "──────────────────────────────────────────"
echo "  STAGE 1 / 3  —  Build"
echo "──────────────────────────────────────────"
make build

# Stage 2: Test
echo ""
echo "──────────────────────────────────────────"
echo "  STAGE 2 / 3  —  Test"
echo "──────────────────────────────────────────"
make test

# Stage 3: Package
echo ""
echo "──────────────────────────────────────────"
echo "  STAGE 3 / 3  —  Package"
echo "──────────────────────────────────────────"
make package

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  CI/CD Stage 완료                    ║"
echo "║  ML 파이프라인: ./ci/trigger_pipeline.sh ║"
echo "╚══════════════════════════════════════════╝"
