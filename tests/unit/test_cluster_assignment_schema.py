"""Schema guards for promoted SKU cluster assignment storage."""

from __future__ import annotations

from common.core.paths import PROJECT_ROOT


SQL_DIR = PROJECT_ROOT / "sql"


def test_dim_sku_base_ddl_does_not_create_ml_cluster_column():
    ddl = (SQL_DIR / "005_create_dim_dfu.sql").read_text()

    create_table = ddl.split(");", maxsplit=1)[0]

    assert "ml_cluster TEXT" not in create_table
    assert "idx_dim_sku_ml_cluster" not in ddl


def test_remove_ml_cluster_migration_backfills_then_drops_column():
    ddl = (SQL_DIR / "201_remove_dim_sku_ml_cluster.sql").read_text()

    assert "INSERT INTO sku_cluster_assignment" in ddl
    assert "ALTER TABLE dim_sku DROP COLUMN IF EXISTS ml_cluster" in ddl
    assert "INCLUDE (abc_vol, region, brand_desc, seasonality_profile)" in ddl
