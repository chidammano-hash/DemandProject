"""Durable per-SKU cluster labels — store on the experiment row, read back as a
fallback so a completed experiment stays re-promotable after /tmp is cleared."""

import gzip
import io
from unittest.mock import MagicMock, patch

import pandas as pd

from common.ml.clustering import scenario as scen

CSV = b"sku_ck,cluster_label\nA,high\nB,low\n"


def _conn_cm(conn):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=conn)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _cursor_cm(cur):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cur)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_load_label_bytes_prefers_working_file(tmp_path):
    """When the working file exists it is used directly — no DB hit."""
    p = tmp_path / "cluster_labels.csv"
    p.write_bytes(CSV)
    with patch.object(scen, "get_db_params", side_effect=AssertionError("DB hit")):
        assert scen._load_label_bytes("sc_x", p) == CSV


def test_load_label_bytes_db_fallback(tmp_path):
    """File gone -> decompress the durable copy stored on the experiment row."""
    cur = MagicMock()
    cur.fetchone.return_value = (gzip.compress(CSV),)
    conn = MagicMock()
    conn.cursor.return_value = _cursor_cm(cur)
    with (
        patch.object(scen, "get_db_params", return_value={}),
        patch("psycopg.connect", return_value=_conn_cm(conn)),
    ):
        out = scen._load_label_bytes("sc_x", tmp_path / "gone.csv")
    assert out == CSV
    df = pd.read_csv(io.BytesIO(out))
    assert list(df["cluster_label"]) == ["high", "low"]


def test_load_label_bytes_none_when_no_file_and_no_durable(tmp_path):
    """Legacy experiment (no file, no durable copy) -> None, so promote can 404."""
    cur = MagicMock()
    cur.fetchone.return_value = (None,)
    conn = MagicMock()
    conn.cursor.return_value = _cursor_cm(cur)
    with (
        patch.object(scen, "get_db_params", return_value={}),
        patch("psycopg.connect", return_value=_conn_cm(conn)),
    ):
        assert scen._load_label_bytes("sc_x", tmp_path / "gone.csv") is None


def test_store_durable_labels_missing_file_is_noop(tmp_path):
    """Nothing to store and no DB hit when the labels file doesn't exist."""
    with patch.object(scen, "get_db_params", side_effect=AssertionError("DB hit")):
        scen.store_durable_labels(7, tmp_path / "gone.csv")  # must not raise


def test_store_durable_labels_roundtrip(tmp_path):
    """The stored blob decompresses back to the exact labels CSV."""
    p = tmp_path / "cluster_labels.csv"
    p.write_bytes(CSV)
    conn = MagicMock()
    with (
        patch.object(scen, "get_db_params", return_value={}),
        patch("psycopg.connect", return_value=_conn_cm(conn)),
    ):
        scen.store_durable_labels(7, p)
    sql, params = conn.execute.call_args.args
    assert "cluster_labels_gz" in sql
    gz, exp_id = params
    assert exp_id == 7
    assert gzip.decompress(gz) == CSV
