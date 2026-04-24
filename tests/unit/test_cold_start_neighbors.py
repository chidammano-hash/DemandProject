"""Unit tests for common.ml.cold_start_neighbors."""
from __future__ import annotations

import pytest

from common.ml.cold_start_neighbors import (
    DFUMetadata,
    NearestNeighbor,
    build_prompt_prefix,
    nearest_neighbors,
)


def _make_candidates() -> list[DFUMetadata]:
    return [
        DFUMetadata("A1", category="wine", brand="X", abc_class="A", avg_price=12.0),
        DFUMetadata("A2", category="wine", brand="Y", abc_class="B", avg_price=14.0),
        DFUMetadata("B1", category="spirits", brand="Z", abc_class="A", avg_price=25.0),
        DFUMetadata("C1", category="beer", brand="W", abc_class="C", avg_price=4.0),
    ]


def test_nearest_neighbors_returns_k_closest_by_category():
    new = DFUMetadata(
        "NEW", category="wine", brand="X", abc_class="A", avg_price=13.0
    )
    results = nearest_neighbors(new, _make_candidates(), k=2)
    assert len(results) == 2
    # Expect wine DFUs (A1, A2) ranked above spirits/beer since category matches.
    top_ids = {r.dfu_id for r in results}
    assert top_ids <= {"A1", "A2"}
    # Ordered by descending similarity
    assert results[0].similarity >= results[1].similarity


def test_nearest_neighbors_handles_k_larger_than_candidates():
    new = DFUMetadata(
        "NEW", category="beer", brand="W", abc_class="C", avg_price=4.0
    )
    results = nearest_neighbors(new, _make_candidates(), k=100)
    assert len(results) == 4  # only four candidates available


def test_nearest_neighbors_empty_candidates_raises():
    new = DFUMetadata(
        "NEW", category="wine", brand="X", abc_class="A", avg_price=1.0
    )
    with pytest.raises(ValueError):
        nearest_neighbors(new, [], k=3)


def test_nearest_neighbors_zero_k_raises():
    new = DFUMetadata(
        "NEW", category="wine", brand="X", abc_class="A", avg_price=1.0
    )
    with pytest.raises(ValueError):
        nearest_neighbors(new, _make_candidates(), k=0)


def test_build_prompt_prefix_lists_neighbors():
    new = DFUMetadata(
        "NEW", category="wine", brand="X", abc_class="A", avg_price=13.0
    )
    results = nearest_neighbors(new, _make_candidates(), k=3)
    prefix = build_prompt_prefix(results)
    assert "Cold-start neighbors" in prefix
    for r in results:
        assert r.dfu_id in prefix


def test_build_prompt_prefix_empty_list():
    prefix = build_prompt_prefix([])
    assert "Cold-start neighbors" in prefix  # header always present
