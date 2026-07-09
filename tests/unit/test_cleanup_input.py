"""Tests for location scoping in the raw input cleanup script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("data/input/cleanup_input.py")


def _load_cleanup_module():
    spec = importlib.util.spec_from_file_location("cleanup_input", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_dfu_applies_location_filter(tmp_path):
    cleanup = _load_cleanup_module()
    dfu_path = tmp_path / "dfu.txt"
    dfu_path.write_text(
        "DMDUNIT|DMDGROUP|LOC|U_CLUSTER_ASSIGNMENT\n"
        "item-a|g|1401-BULK|L1_A\n"
        "item-b|g|2201-LOUI|L1_A\n"
        "item-c|g|1401-BULK|L3_A\n"
    )
    cleanup.DFU_FILE = dfu_path

    cleanup.filter_dfu(keep_loc="1401-BULK")

    assert dfu_path.read_text().splitlines() == [
        "DMDUNIT|DMDGROUP|LOC|U_CLUSTER_ASSIGNMENT",
        "item-a|g|1401-BULK|L1_A",
    ]


def test_filter_inventory_applies_location_filter_even_when_dfu_has_other_locations(tmp_path):
    cleanup = _load_cleanup_module()
    dfu_path = tmp_path / "dfu.txt"
    dfu_path.write_text(
        "DMDUNIT|DMDGROUP|LOC\n"
        "item-a|g|1401-BULK\n"
        "item-b|g|2201-LOUI\n"
    )
    inventory_path = tmp_path / "Inventory_Snapshot_2026_07.csv"
    inventory_path.write_text(
        "exec_date,item,loc,qty\n"
        "2026-07-01,item-a,1401-BULK,1\n"
        "2026-07-01,item-b,2201-LOUI,2\n"
    )
    cleanup.DFU_FILE = dfu_path
    cleanup.INV_PATTERN = str(tmp_path / "Inventory_Snapshot_*.csv")

    cleanup.filter_inventory(keep_loc="1401-BULK")

    assert inventory_path.read_text().splitlines() == [
        "exec_date,item,loc,qty",
        "2026-07-01,item-a,1401-BULK,1",
    ]
