"""US17e — shared parser for the ETL dispatcher's final JSON stdout line."""
from __future__ import annotations

import pytest

from common.services.etl_job_output import parse_final_json


def test_parses_full_metrics() -> None:
    out = parse_final_json('noise\n{"rows_loaded": 10, "rows_inserted": 7, '
                           '"rows_updated": 3, "rows_deleted": 0}')
    assert out == {"rows_loaded": 10, "rows_inserted": 7,
                   "rows_updated": 3, "rows_deleted": 0, "error": None}


def test_takes_last_json_line() -> None:
    out = parse_final_json('{"rows_loaded": 1}\n{"rows_loaded": 99}')
    assert out["rows_loaded"] == 99


@pytest.mark.parametrize("stdout", ["", "   ", "not json", "[1,2,3]"])
def test_bad_input_returns_defaults(stdout) -> None:
    out = parse_final_json(stdout)
    assert out["rows_loaded"] == 0
    assert out["error"] is None


def test_error_field_surfaced() -> None:
    out = parse_final_json('{"error": "bad partition"}')
    assert out["error"] == "bad partition"
    assert out["rows_loaded"] == 0
