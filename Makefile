IMAGE ?= ragstore:dev

.PHONY: install test lint up down build scan

install:
	uv sync --extra dev

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
	trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 --scanners vuln $(IMAGE)
