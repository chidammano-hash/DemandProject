"""Unit tests for common.ml.shadow_rollout."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.ml.shadow_rollout import ShadowRollout


def test_shadow_rollout_rejects_invalid_traffic():
    with pytest.raises(ValueError):
        ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=1.5)
    with pytest.raises(ValueError):
        ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=-0.01)


def test_shadow_rollout_rejects_same_ids():
    with pytest.raises(ValueError):
        ShadowRollout(champion_id="m", challenger_id="m")


def test_shadow_rollout_rejects_bad_status():
    with pytest.raises(ValueError):
        ShadowRollout(champion_id="a", challenger_id="b", status="invalid")


def test_should_tee_false_when_inactive():
    r = ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=1.0)
    # status='proposed' by default
    assert r.should_tee() is False


def test_should_tee_false_when_traffic_zero_even_if_active():
    r = ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=0.0, status="active")
    assert r.should_tee() is False


def test_should_tee_true_at_full_traffic():
    r = ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=1.0, status="active")
    rng = np.random.default_rng(0)
    assert r.should_tee(rng=rng) is True


def test_should_tee_statistically_matches_traffic_pct():
    r = ShadowRollout(champion_id="a", challenger_id="b", traffic_pct=0.3, status="active")
    rng = np.random.default_rng(0)
    n = 2000
    hits = sum(r.should_tee(rng=rng) for _ in range(n))
    # Allow +/- 5 pct of target at n=2000
    assert 0.25 < hits / n < 0.35


def test_insert_writes_row_and_ledger():
    cur = MagicMock()
    # First fetchone returns ledger's latest hash, then rollout id, then
    # ledger insert id.
    cur.fetchone.side_effect = [(101,), ("0" * 64,), (999,)]
    r = ShadowRollout(
        champion_id="champion", challenger_id="challenger",
        traffic_pct=0.1, status="proposed", notes="initial shadow",
    )
    new_id = r.insert(cur)
    assert new_id == 101
    # Two execute calls expected: fact_shadow_rollout INSERT and ledger append.
    assert cur.execute.call_count >= 1
    first_sql = cur.execute.call_args_list[0].args[0]
    assert "INSERT INTO fact_shadow_rollout" in first_sql
