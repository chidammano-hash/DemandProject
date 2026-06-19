"""Unit tests for the Chronos-Bolt fine-tune data prep (spec 32).

Pure functions only — no GPU, no DB, no model load. Validates training-window
shaping, masking, series filtering, the per-series cap, and version resolution.
"""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.ml.finetune_chronos_bolt import (  # noqa: E402 — after importorskip
    _build_windows,
    _existing_versions,
)


def _sales() -> pd.DataFrame:
    rows = []
    for m in range(6):  # series A: 6 months, qty 1..6
        rows.append(("A", pd.Timestamp(2025, 1, 1) + pd.DateOffset(months=m), float(m + 1)))
    for m in range(2):  # series B: 2 months (too short for min_series_months=3)
        rows.append(("B", pd.Timestamp(2025, 1, 1) + pd.DateOffset(months=m), 5.0))
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "qty"])


def test_build_windows_shapes_and_masks():
    w = _build_windows(_sales(), context_length=4, horizon=2,
                       min_series_months=3, stride=1, max_windows_per_series=2)
    assert len(w) > 0
    assert len(w) <= 2  # cap respected (series A only)
    for win in w:
        assert win["context"].shape[0] == 4
        assert win["target"].shape[0] == 2
        assert win["mask"].dtype == torch.bool
        assert win["target_mask"].dtype == torch.bool
        assert bool(win["mask"].any())          # at least one observed context point
        assert bool(win["target_mask"].any())   # at least one real target


def test_build_windows_excludes_short_series():
    # Both series shorter than 10 months -> nothing eligible.
    w = _build_windows(_sales(), context_length=4, horizon=2,
                       min_series_months=10, stride=1, max_windows_per_series=5)
    assert w == []


def test_build_windows_left_pads_short_context():
    # Single 4-month series, context_length 6 -> context left-padded, mask marks pads.
    df = pd.DataFrame(
        [("A", pd.Timestamp(2025, 1, 1) + pd.DateOffset(months=m), float(m + 1)) for m in range(4)],
        columns=["sku_ck", "startdate", "qty"],
    )
    w = _build_windows(df, context_length=6, horizon=2,
                       min_series_months=3, stride=1, max_windows_per_series=5)
    assert w, "expected at least one window"
    win = w[0]
    # leading positions are padding (mask False), trailing are observed (mask True)
    assert win["mask"][0].item() is False
    assert win["mask"][-1].item() is True


def test_existing_versions(tmp_path):
    assert _existing_versions(tmp_path) == 0
    (tmp_path / "v1").mkdir()
    (tmp_path / "v3").mkdir()
    (tmp_path / "scratch").mkdir()  # non-version dir ignored
    assert _existing_versions(tmp_path) == 3
