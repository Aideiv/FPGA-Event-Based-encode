# Root Makefile — FPGA Event-Based Drone Collision Avoidance
#
# Targets:
#   make help        : Show this help
#   make test        : Run all tests (Python + C++ + FPGA)
#   make test-py     : Run Python pipeline tests
#   make test-arm    : Build & run ARM C++ unit tests
#   make test-fpga   : Build & run FPGA C++ testbench
#   make lint        : Run linters (flake8 for Python)
#   make build       : Build all targets
#   make build-arm   : Build ARM C++ binaries (sim mode)
#   make convert     : Convert pretrained weights → FPGA.pth
#   make clean       : Remove build artifacts
#   make install     : Install Python package
#   make ci          : Run full CI pipeline locally
#
.PHONY: help test test-py test-arm test-fpga lint build build-arm convert clean install ci docker-build docker-test

# Default Python
PYTHON := python3

# Docker toolchain image (Python 3.11 + g++-13 + CPU torch)
IMAGE ?= fpga-event-encode:dev

help:
	@echo "FPGA Event-Based Drone Collision Avoidance"
	@echo ""
	@echo "Targets:"
	@echo "  make test        Run all tests (Python + C++ + FPGA)"
	@echo "  make test-py     Run Python pipeline simulator tests"
	@echo "  make test-arm    Build & run ARM C++ unit tests"
	@echo "  make test-fpga   Build & run FPGA C++ testbench"
	@echo "  make test-hil    Run hardware-in-the-loop simulation"
	@echo "  make lint        Run linting (flake8)"
	@echo "  make build       Build all targets"
	@echo "  make build-arm   Build ARM C++ (simulation mode)"
	@echo "  make convert     Convert pretrained weights → FPGA.pth (d=384→d=128)"
	@echo "  make clean       Remove build artifacts"
	@echo "  make install     Install Python package (pip install -e .)"
	@echo "  make ci          Run full CI pipeline locally"
	@echo "  make docker-build Build the Docker toolchain image"
	@echo "  make docker-test  Run make test inside the Docker image"
	@echo ""

# ── Test Targets ────────────────────────────────────────────────────────────

test: test-py test-arm test-fpga
	@echo ""
	@echo "============================================"
	@echo "  All tests complete"
	@echo "============================================"

test-py:
	@echo "=== Python Pipeline Simulator Tests ==="
	$(PYTHON) test/test_fpga_simulator.py
	@echo "=== Golden Model Equivalence Tests ==="
	$(PYTHON) test/golden_model_test.py

test-arm:
	@echo "=== ARM C++ Unit Tests ==="
	cd arm && $(MAKE) sim
	cd test && g++ -std=c++17 -O2 -I.. -I../arm -o test_arm_cp test_arm_collision_predictor.cpp -lpthread && ./test_arm_cp

test-fpga:
	@echo "=== FPGA C++ Testbench ==="
	cd fpga && g++ -std=c++11 -O2 -I. -D__SIMULATION__ -o testbench testbench.cpp -lm && ./testbench

test-hil:
	@echo "=== Hardware-in-the-Loop Simulation ==="
	$(PYTHON) test/hil_gazebo_bridge.py --demo

# ── Lint ─────────────────────────────────────────────────────────────────────

lint:
	@echo "=== flake8 ==="
	flake8 drone/ models/ train/ test/ --count --show-source --statistics
	@echo "=== Python syntax check ==="
	$(PYTHON) -m py_compile setup.py
	$(PYTHON) -m compileall -q drone/ models/ train/ test/
	@echo "=== clang-format ==="
	clang-format --dry-run --Werror arm/*.h arm/*.cpp fpga/*.h fpga/*.cpp test/*.cpp

# ── Build Targets ────────────────────────────────────────────────────────────

build: build-arm
	@echo "All builds complete"

build-arm:
	cd arm && $(MAKE) sim

convert:
	@echo "=== Converting pretrained weights → FPGA.pth ==="
	$(PYTHON) train/convert_weights_to_fpga.py \
		--input models/models/UNION.pth \
		--output models/models/FPGA.pth

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t $(IMAGE) .

docker-test: docker-build
	docker run --rm -u $(shell id -u):$(shell id -g) -v $(CURDIR):/repo -w /repo $(IMAGE) make test

# ── Install ──────────────────────────────────────────────────────────────────

install:
	pip install --upgrade pip setuptools wheel
	pip install -e .

# ── CI ───────────────────────────────────────────────────────────────────────

ci: lint test build
	@echo ""
	@echo "============================================"
	@echo "  CI pipeline complete"
	@echo "============================================"

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	cd arm && $(MAKE) clean
	cd fpga && rm -f testbench
	cd test && rm -f test_arm_cp test_arm_evasion
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete"