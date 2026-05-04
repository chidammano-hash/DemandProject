"""Unit tests for IPfeature11 ABC-XYZ classification pure functions."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inventory.classify_abc_xyz import classify_xyz, compute_abc_xyz_segment


# ---------------------------------------------------------------------------
# classify_xyz
# ---------------------------------------------------------------------------

def test_xyz_x_low_cv():
    assert classify_xyz(0.20) == "X"


def test_xyz_x_boundary():
    assert classify_xyz(0.30) == "X"


def test_xyz_y_moderate():
    assert classify_xyz(0.50) == "Y"


def test_xyz_y_upper_boundary():
    assert classify_xyz(0.80) == "Y"


def test_xyz_z_high_cv():
    assert classify_xyz(0.90) == "Z"


def test_xyz_z_intermittency():
    # Even low cv, high intermittency → Z
    assert classify_xyz(0.10, intermittency_ratio=0.35) == "Z"


def test_xyz_z_intermittency_boundary():
    # Exactly at boundary 0.30 → not Z
    assert classify_xyz(0.10, intermittency_ratio=0.30) == "X"


def test_xyz_none_when_cv_none():
    assert classify_xyz(None) is None


def test_xyz_none_intermittency_with_low_cv():
    assert classify_xyz(0.25, intermittency_ratio=None) == "X"


# ---------------------------------------------------------------------------
# compute_abc_xyz_segment
# ---------------------------------------------------------------------------

def test_segment_ax():
    assert compute_abc_xyz_segment("A", "X") == "AX"


def test_segment_bz():
    assert compute_abc_xyz_segment("B", "Z") == "BZ"


def test_segment_lowercase():
    assert compute_abc_xyz_segment("c", "y") == "CY"


def test_segment_none_abc():
    assert compute_abc_xyz_segment(None, "X") is None


def test_segment_none_xyz():
    assert compute_abc_xyz_segment("A", None) is None


def test_segment_invalid_abc():
    assert compute_abc_xyz_segment("D", "X") is None


def test_segment_all_combinations():
    expected = ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
    for seg in expected:
        abc, xyz = seg[0], seg[1]
        assert compute_abc_xyz_segment(abc, xyz) == seg
