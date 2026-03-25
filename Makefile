SHELL := /bin/bash

# Variables definitions
# -----------------------------------------------------------------------------

PYTHON ?= python3.11
PYTHON_VERSION ?= 3.11.9
VENV_DIR ?= .venv
PYTHON_BIN := $(shell $(PYTHON) -c 'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null)
UV ?= $(PYTHON_BIN) -m uv
VENV_PYTHON := $(VENV_DIR)/bin/python

ifeq ($(TIMEOUT),)
TIMEOUT := 60
endif

ifeq ($(MODEL_PATH),)
MODEL_PATH := ./ml/model/
endif

ifeq ($(MODEL_NAME),)
MODEL_NAME := model.pkl
endif

# Target section and Global definitions
# -----------------------------------------------------------------------------
.PHONY: all bootstrap-uv check-python clean test install run deploy down generate_dot_env venv

all: clean install test run deploy down

bootstrap-uv:
	@$(PYTHON_BIN) -m uv --version >/dev/null 2>&1 || { \
		echo "uv не найден, устанавливаю в базовый интерпретатор $(PYTHON_BIN)"; \
		$(PYTHON_BIN) -m ensurepip --upgrade >/dev/null 2>&1 || true; \
		$(PYTHON_BIN) -m pip install --upgrade pip uv; \
	}

check-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "Python interpreter '$(PYTHON)' not found. Install Python $(PYTHON_VERSION) or pass PYTHON=/path/to/python3.11"; \
		exit 1; \
	}
	@$(PYTHON) -c 'import sys; required = (3, 11, 9); current = sys.version_info[:3]; raise SystemExit(0) if current == required else SystemExit("Expected Python 3.11.9, got {}.{}.{}".format(*current))'


venv: check-python bootstrap-uv
	@test -x $(VENV_PYTHON) || $(UV) venv --python $(PYTHON_BIN) $(VENV_DIR)

test: venv
	$(UV) run --python $(VENV_PYTHON) pytest tests -vv --show-capture=all

metrics:
	radon mi -s src/

install: generate_dot_env venv
	$(UV) pip install --python $(VENV_PYTHON) -e ".[dev]"

run: venv
	PYTHONPATH=src/ $(VENV_PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

deploy: generate_dot_env
	docker-compose build
	docker-compose up -d

down:
	docker-compose down

generate_dot_env:
	@test -f .env || cp .env.example .env

clean:
	@find . -name '*.pyc' -exec rm -rf {} \;
	@find . -name '__pycache__' -exec rm -rf {} \;
	@find . -name 'Thumbs.db' -exec rm -rf {} \;
	@find . -name '*~' -exec rm -rf {} \;
	rm -rf .cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf htmlcov
	rm -rf .tox/
	rm -rf docs/_build
