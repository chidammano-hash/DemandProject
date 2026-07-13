"""Sales delta detection must repair the strict dual-track forecast source."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_unchanged_sales_hash_forces_reload_when_mirror_is_unsynchronized(tmp_path: Path):
    from scripts.etl.load import _do_delta

    raw = tmp_path / "sales.csv"
    raw.write_text("header\n", encoding="utf-8")
    spec = SimpleNamespace(clean_file="staged/sales.csv")
    with (
        patch("scripts.etl.load._resolve_raw_input", return_value=raw),
        patch("scripts.etl.load.file_hash", return_value="a" * 64),
        patch("scripts.etl.load._fetch_last_hash", return_value="a" * 64),
        patch("scripts.etl.load._sales_source_is_synchronized", return_value=False),
        patch("scripts.etl.load._normalize_domain") as normalize,
        patch("common.core.domain_specs.get_spec", return_value=spec),
        patch(
            "scripts.etl.load_dataset_postgres.load_domain",
            return_value={"rows_loaded": 305_139},
        ) as load,
    ):
        result = _do_delta("sales", False)

    normalize.assert_called_once_with("sales", False)
    load.assert_called_once()
    assert result == {
        "status": "success",
        "inserted": 305_139,
        "updated": 0,
        "deleted": 0,
    }


def test_unchanged_sales_hash_skips_when_dual_track_source_is_current(tmp_path: Path):
    from scripts.etl.load import _do_delta

    raw = tmp_path / "sales.csv"
    raw.write_text("header\n", encoding="utf-8")
    with (
        patch("scripts.etl.load._resolve_raw_input", return_value=raw),
        patch("scripts.etl.load.file_hash", return_value="a" * 64),
        patch("scripts.etl.load._fetch_last_hash", return_value="a" * 64),
        patch("scripts.etl.load._sales_source_is_synchronized", return_value=True),
        patch("scripts.etl.load._normalize_domain") as normalize,
    ):
        result = _do_delta("sales", False)

    assert result == "skipped"
    normalize.assert_not_called()
