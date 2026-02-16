"""Generate and store pgvector embeddings for schema metadata.

Reads domain_specs.py, builds text descriptions for each table/column/example,
embeds them via OpenAI text-embedding-3-small, and upserts into chat_embeddings.

Usage:
    uv run python scripts/generate_embeddings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.domain_specs import DOMAIN_SPECS, DomainSpec

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50


def col_type(spec: DomainSpec, col: str) -> str:
    if col in spec.int_fields:
        return "integer"
    if col in spec.float_fields:
        return "numeric"
    if col in spec.date_fields:
        return "date"
    return "text"


DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "item": "product/item master data including brand, category, size, supplier, and UPC information",
    "location": "warehouse and distribution locations with site descriptions and state",
    "customer": "customer accounts with name, city, state, premise code, chain, and channel classification",
    "time": "calendar dimension with dates from 2020-2035 including week/month/quarter/year buckets",
    "dfu": "demand forecast units — the item+group+location combinations used for forecasting, with brand, region, and classification attributes",
    "sales": "monthly sales history with quantities shipped, ordered, and total qty per item/group/location/month",
    "forecast": "monthly statistical forecasts with base forecast and actual demand per item/group/location, supporting lag 0-4 months and multiple model_id values",
}

RELATIONSHIPS = [
    ("relationship", "general", "fact_sales_monthly.dmdunit can be joined to dim_dfu.dmdunit to get item attributes like brand, region, category."),
    ("relationship", "general", "fact_sales_monthly.loc can be joined to dim_dfu.loc to get DFU-level location attributes."),
    ("relationship", "general", "fact_external_forecast_monthly.dmdunit can be joined to dim_dfu.dmdunit to get item attributes."),
    ("relationship", "general", "fact_external_forecast_monthly.loc can be joined to dim_dfu.loc to get location attributes."),
    ("relationship", "general", "dim_dfu.dmdunit typically corresponds to dim_item.item_no for product-level attributes."),
    ("relationship", "general", "fact_sales_monthly.startdate and fact_external_forecast_monthly.startdate are always month-start dates (first day of month)."),
    ("relationship", "general", "fact_external_forecast_monthly.model_id identifies the forecast algorithm. Default is 'external' for source-system forecasts."),
]

EXAMPLE_QUERIES = [
    ("example_query", "sales", "Total sales quantity by item: SELECT dmdunit, SUM(qty) AS total_qty FROM fact_sales_monthly GROUP BY 1 ORDER BY 2 DESC LIMIT 20;"),
    ("example_query", "sales", "Monthly sales trend: SELECT date_trunc('month', startdate)::date AS month, SUM(qty) AS total_qty FROM fact_sales_monthly GROUP BY 1 ORDER BY 1;"),
    ("example_query", "sales", "Sales for a specific item and location: SELECT startdate, qty_shipped, qty_ordered, qty FROM fact_sales_monthly WHERE dmdunit = '100320' AND loc = '1401-BULK' ORDER BY startdate;"),
    ("example_query", "sales", "Top locations by total shipped quantity: SELECT loc, SUM(qty_shipped) AS total_shipped FROM fact_sales_monthly GROUP BY 1 ORDER BY 2 DESC LIMIT 20;"),
    ("example_query", "forecast", "Forecast accuracy for a specific item: SELECT dmdunit, loc, 100 - (100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS accuracy_pct FROM fact_external_forecast_monthly WHERE dmdunit = '100320' GROUP BY 1, 2;"),
    ("example_query", "forecast", "Forecast bias by item: SELECT dmdunit, (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0)) - 1 AS bias FROM fact_external_forecast_monthly GROUP BY 1 ORDER BY 2 DESC LIMIT 20;"),
    ("example_query", "forecast", "Monthly forecast vs actual trend: SELECT date_trunc('month', startdate)::date AS month, SUM(basefcst_pref) AS forecast, SUM(tothist_dmd) AS actual FROM fact_external_forecast_monthly GROUP BY 1 ORDER BY 1;"),
    ("example_query", "forecast", "Forecast accuracy by model: SELECT model_id, 100 - (100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS accuracy_pct FROM fact_external_forecast_monthly GROUP BY 1;"),
    ("example_query", "item", "Search items by brand: SELECT item_no, item_desc, brand_name, category FROM dim_item WHERE brand_name ILIKE '%SMIRNOFF%' LIMIT 20;"),
    ("example_query", "item", "Count items by category: SELECT category, COUNT(*) AS cnt FROM dim_item GROUP BY 1 ORDER BY 2 DESC;"),
    ("example_query", "dfu", "DFUs by region: SELECT region, COUNT(*) AS cnt FROM dim_dfu GROUP BY 1 ORDER BY 2 DESC;"),
    ("example_query", "customer", "Customers by state: SELECT state, COUNT(*) AS cnt FROM dim_customer GROUP BY 1 ORDER BY 2 DESC LIMIT 20;"),
    ("example_query", "location", "All locations: SELECT location_id, site_id, site_desc, state_id FROM dim_location ORDER BY location_id;"),
    ("example_query", "general", "WAPE calculation: SELECT 100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0) AS wape_pct FROM fact_external_forecast_monthly;"),
    ("example_query", "general", "Join sales with DFU attributes: SELECT s.dmdunit, d.brand, d.region, SUM(s.qty) AS total_qty FROM fact_sales_monthly s JOIN dim_dfu d ON s.dmdunit = d.dmdunit AND s.loc = d.loc GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 20;"),
]


def build_texts(spec: DomainSpec) -> list[tuple[str, str, str]]:
    """Return list of (content_type, domain_name, source_text) tuples."""
    rows: list[tuple[str, str, str]] = []
    desc = DOMAIN_DESCRIPTIONS.get(spec.name, spec.name)
    col_list = ", ".join(spec.columns)
    rows.append((
        "table_desc",
        spec.name,
        f"Table {spec.table} stores {desc}. Columns: {col_list}. Primary key: {spec.ck_field}. Business key: {', '.join(spec.key_fields)}.",
    ))
    for col in spec.columns:
        ctype = col_type(spec, col)
        rows.append((
            "column_desc",
            spec.name,
            f"Column {col} ({ctype}) in table {spec.table} ({spec.name} domain).",
        ))
    return rows


def get_conninfo() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5440')} "
        f"dbname={os.getenv('POSTGRES_DB', 'demand_mvp')} "
        f"user={os.getenv('POSTGRES_USER', 'demand')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'demand')}"
    )


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        print("ERROR: Set a valid OPENAI_API_KEY in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Build all text chunks
    all_chunks: list[tuple[str, str, str]] = []
    for spec in DOMAIN_SPECS.values():
        all_chunks.extend(build_texts(spec))
    all_chunks.extend(RELATIONSHIPS)
    all_chunks.extend(EXAMPLE_QUERIES)

    print(f"Embedding {len(all_chunks)} chunks ...")

    # Embed in batches
    texts = [chunk[2] for chunk in all_chunks]
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    # Store in Postgres
    conninfo = get_conninfo()
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chat_embeddings")
            for idx, (content_type, domain_name, source_text) in enumerate(all_chunks):
                vec = embeddings[idx]
                cur.execute(
                    "INSERT INTO chat_embeddings (domain_name, content_type, source_text, embedding) VALUES (%s, %s, %s, %s::vector)",
                    (domain_name, content_type, source_text, str(vec)),
                )
        conn.commit()

    # Create IVFFlat index now that we have data
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS idx_chat_embeddings_vector")
            n_lists = max(1, len(all_chunks) // 10)
            cur.execute(
                f"CREATE INDEX idx_chat_embeddings_vector ON chat_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {n_lists})"
            )
        conn.commit()

    print(f"Stored {len(all_chunks)} embeddings in chat_embeddings.")


if __name__ == "__main__":
    main()
