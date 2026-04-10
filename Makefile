PYTHON := python3
PIP    := pip3

.PHONY: build test package pipeline clean

## build: 의존성 설치 + 문법 검사
build:
	@echo "=== [1/4] BUILD ==="
	$(PIP) install -r requirements.txt -q
	$(PIP) install pytest build -q
	$(PYTHON) -m py_compile harness_pipeline.py
	$(PYTHON) -m py_compile pipeline/train.py pipeline/evaluate.py \
	    pipeline/deploy.py pipeline/monitor.py pipeline/predict.py
	$(PYTHON) -m py_compile data/loader.py
	@echo "✅  BUILD OK"

## test: 단위 테스트 실행
test:
	@echo "=== [2/4] TEST ==="
	$(PYTHON) -m pytest tests/ -v
	@echo "✅  TEST OK"

## package: wheel 패키지 빌드
package:
	@echo "=== [3/4] PACKAGE ==="
	$(PYTHON) -m build --wheel --outdir dist/ . -q
	@ls -lh dist/*.whl
	@echo "✅  PACKAGE OK"

## pipeline: 전체 MLOps 파이프라인 실행
pipeline:
	@echo "=== [4/4] PIPELINE ==="
	$(PYTHON) harness_pipeline.py
	@echo "✅  PIPELINE OK"

## all: build → test → package 순서 실행 (pipeline 제외)
all: build test package

## clean: 빌드 산출물 제거
clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
