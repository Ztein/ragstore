IMAGE ?= ragstore:dev

.PHONY: install hooks test lint up down build scan

install:
	uv sync --extra dev

hooks:
	git config core.hooksPath .githooks
	@echo "pre-push hook enabled (runs lint + format + full test suite locally)"

up:
	docker compose up -d weaviate

down:
	docker compose down

test: up
	uv run pytest

lint:
	uv run ruff check src tests

build:
	docker build -t $(IMAGE) .

scan:
	trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 --scanners vuln \
		--trivyignores .trivyignore $(IMAGE)
