# 01 — Platform & Infrastructure

This group covers the foundational platform layer of Demand Studio: infrastructure setup, data architecture, core dimension and fact tables, and cross-cutting platform utilities such as benchmarking and design tooling integration.

## Files

| File | Feature | Summary |
|---|---|---|
| `feature1.md` | Infrastructure & Platform Setup | Docker Compose service cluster (Postgres, MinIO, Spark, Trino, MLflow), environment bootstrap, Makefile targets, and one-time setup steps. |
| `feature2.md` | Internal Data Architecture & Data Contracts | Full data contracts (ERD, table grain, column types, null policy), dual-path forecast loading with execution-lag filtering, and archive integrity rules. |
| `feature3.md` | Dimension Tables | DDL and load pipeline for all five dimension tables: Item, Location, Customer, Time, and DFU; surrogate keys, composite keys, and trigram indexes. |
| `feature4.md` | Fact Tables | DDL and load pipeline for Sales and External Forecast fact tables; monthly grain, model_id scoping, UNIQUE constraints, and materialized aggregate views. |
| `feature26.md` | Postgres vs Trino/Iceberg Benchmarking | Latency comparison API (`GET /bench/compare`) running identical queries against Postgres and Trino, returning per-query min/max/avg/p50/p95 stats and speedup factor. |
| `feature27.md` | Figma MCP Integration | Design-to-code and code-to-design workflow via Figma MCP; component inspection, token extraction, and design sync (not yet started). |
