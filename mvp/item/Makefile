SHELL := /bin/zsh

DC := docker compose
UV := uv run

.PHONY: help up down logs normalize load api spark trino-check check-api check-db check-all init init-pip

help:
	@echo "Targets:"
	@echo "  init        - copy env, create uv env, install deps"
	@echo "  init-pip    - fallback init without uv (venv + pip install)"
	@echo "  up          - start docker services"
	@echo "  down        - stop docker services"
	@echo "  logs        - tail docker logs"
	@echo "  normalize   - normalize source CSV"
	@echo "  load        - load dim_item into Postgres"
	@echo "  api         - run FastAPI service on :8000"
	@echo "  spark       - run Spark job to write Iceberg table"
	@echo "  trino-check - run basic Trino query against Iceberg table"
	@echo "  check-api   - health and sample item API checks"
	@echo "  check-db    - count rows in Postgres dim_item"
	@echo "  check-all   - run DB/API/Trino checks"

init:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed."; \
		echo "Install: brew install uv"; \
		echo "Or use fallback: make init-pip"; \
		exit 1; \
	}
	uv venv
	uv sync

init-pip:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install fastapi uvicorn pydantic psycopg[binary] python-dotenv pytest

up:
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

normalize:
	$(UV) python scripts/normalize_item_csv.py

load:
	$(UV) python scripts/load_item_postgres.py

api:
	$(UV) uvicorn api.main:app --reload --port 8000

spark:
	docker exec -it item-mvp-spark /opt/bitnami/spark/bin/spark-submit \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_item_to_iceberg.py

trino-check:
	docker exec -i item-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_item;"

check-api:
	curl -s http://localhost:8000/health && echo
	curl -s "http://localhost:8000/items?limit=3" && echo

check-db:
	docker exec -i item-mvp-postgres psql -U demand -d demand_mvp -c "SELECT count(*) AS cnt FROM dim_item;"

check-all: check-db check-api trino-check
