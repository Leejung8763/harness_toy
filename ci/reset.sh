#!/usr/bin/env bash
# ci/reset.sh — 런타임 상태 초기화
#
# 삭제 대상:
#   - models/         학습된 모델 pkl
#   - feature_store/  캐시된 피처
#   - ml_metadata/    실험 이력
#   - registry/       모델 버전 기록
#   - feature_flags/  배포 상태
#   - build/dist/     빌드 아티팩트 (선택)
#
# 사용법:
#   ./ci/reset.sh           # 런타임 상태만 초기화
#   ./ci/reset.sh --hard    # 빌드 아티팩트까지 포함

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         Harness MLOps — Reset            ║"
echo "╚══════════════════════════════════════════╝"

# ── 확인 프롬프트 ─────────────────────────────────────────────────
read -r -p "  ⚠️  모든 모델과 런타임 데이터를 삭제합니다. 계속할까요? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "  취소됐습니다."
    exit 0
fi

echo ""

# ── 실행 중인 서버 종료 ───────────────────────────────────────────
echo "  [1/5] 실행 중인 서버 종료..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "       port 8000 종료" || echo "       port 8000 미실행"
lsof -ti:8001 | xargs kill -9 2>/dev/null && echo "       port 8001 종료" || echo "       port 8001 미실행"

# ── 모델 파일 삭제 ────────────────────────────────────────────────
echo "  [2/5] 모델 파일 삭제..."
rm -f "$ROOT"/models/*.pkl
echo "       models/ 초기화 완료"

# ── Feature Store 초기화 ─────────────────────────────────────────
echo "  [3/5] Feature Store 초기화..."
rm -rf "$ROOT/feature_store/features/"
echo "       feature_store/features/ 삭제 완료"

# ── 런타임 JSON 초기화 ────────────────────────────────────────────
echo "  [4/5] 런타임 상태 초기화..."

echo '{"models": []}' > "$ROOT/registry/model_registry.json"
echo "       registry/model_registry.json → 초기화"

echo '[]' > "$ROOT/ml_metadata/runs.json"
echo "       ml_metadata/runs.json → 초기화"

echo '{"active_model_version": null, "active_dataset_id": null}' > "$ROOT/feature_flags/flags.json"
echo "       feature_flags/flags.json → 초기화"

# ── 빌드 아티팩트 삭제 (--hard 옵션) ─────────────────────────────
if [[ "${1:-}" == "--hard" ]]; then
    echo "  [5/5] 빌드 아티팩트 삭제 (--hard)..."
    rm -rf "$ROOT/build/" "$ROOT/dist/" "$ROOT"/*.egg-info/
    find "$ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    echo "       build/ dist/ __pycache__ 삭제 완료"
else
    echo "  [5/5] 빌드 아티팩트 유지 (--hard 옵션으로 삭제 가능)"
fi

echo ""
echo "  ✅ 초기화 완료 — v1부터 다시 시작할 준비가 됐습니다."
echo ""
echo "  다음 명령으로 파이프라인을 시작하세요:"
echo "    ./ci/start_pipeline_server.sh   # 터미널 1"
echo "    ./ci/start_server.sh            # 터미널 2"
echo "    ./ci/trigger_pipeline.sh        # 터미널 3"
echo ""
