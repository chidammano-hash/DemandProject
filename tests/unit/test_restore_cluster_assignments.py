"""Unit tests for scripts/ml/restore_cluster_assignments.py.

The repair re-applies the promoted cluster-label assignment onto
dim_sku.ml_cluster after a dim_sku reload wipes it (which otherwise collapses
every per-cluster tree forecast). These tests pin the CSV validation + load
contract; the DB UPDATE is exercised via the live pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.restore_cluster_assignments import load_assignments  # noqa: E402


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "cluster_labels.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_assignments_valid(tmp_path):
    """A well-formed CSV loads with sku_ck + cluster_label as strings."""
    label = "very_high_volume_periodic"
    csv = _write_csv(tmp_path, [
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_id": 5, "cluster_label": label},
        {"sku_ck": "11286_ALL_1401-BULK", "cluster_id": 5, "cluster_label": label},
    ])
    df = load_assignments(csv)
    assert len(df) == 2
    assert df["sku_ck"].dtype == object
    assert df["cluster_label"].dtype == object
    assert set(df["cluster_label"]) == {label}


def test_load_assignments_missing_file(tmp_path):
    """Absent CSV raises FileNotFoundError with a remediation hint."""
    with pytest.raises(FileNotFoundError):
        load_assignments(tmp_path / "does_not_exist.csv")


def test_load_assignments_missing_required_column(tmp_path):
    """A CSV without cluster_label is rejected (would silently no-op otherwise)."""
    csv = _write_csv(tmp_path, [{"sku_ck": "84587_ALL_1401-BULK", "cluster_id": 5}])
    with pytest.raises(ValueError, match="cluster_label"):
        load_assignments(csv)


def test_load_assignments_drops_null_rows(tmp_path):
    """Rows with a null sku_ck or cluster_label are dropped, not written as NULLs."""
    csv = _write_csv(tmp_path, [
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"},
        {"sku_ck": "X_ALL_Y", "cluster_label": None},
        {"sku_ck": None, "cluster_label": "low_vol"},
    ])
    df = load_assignments(csv)
    assert len(df) == 1
    assert df.iloc[0]["sku_ck"] == "84587_ALL_1401-BULK"
