"""Unit tests for the champion sweep runner's pure logic.

Covers grid expansion (templates × variants, dedup, cap), config hashing,
ranking objectives, the lightweight gate check, segment scoring, and the
recommendation picker. DB-touching paths are exercised by the API tests.
"""

import pytest

from scripts.ml import run_champion_sweep as sweep

# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

def test_config_hash_is_model_order_insensitive():
    a = {"strategy": "rolling", "strategy_params": {"window_months": 6},
         "models": ["a", "b"], "metric": "wape", "lag_mode": "execution", "min_sku_rows": 3}
    b = dict(a, models=["b", "a"])
    assert sweep._config_hash(a) == sweep._config_hash(b)


def test_config_hash_distinguishes_params():
    a = {"strategy": "rolling", "strategy_params": {"window_months": 6},
         "models": ["a"], "metric": "wape", "lag_mode": "execution", "min_sku_rows": 3}
    b = dict(a, strategy_params={"window_months": 3})
    assert sweep._config_hash(a) != sweep._config_hash(b)


# ---------------------------------------------------------------------------
# expand_grid
# ---------------------------------------------------------------------------

_BASE = {"strategy": "rolling", "strategy_params": {"window_months": 6}}
_KW = {
    "base_champion": _BASE,
    "default_models": ["lgbm_cluster", "chronos2_enriched"],
    "default_metric": "wape",
    "default_lag": "execution",
    "default_min_sku": 3,
    "max_candidates": 24,
}


def test_expand_explicit_configs():
    grid = {"configs": [
        {"strategy": "rolling", "strategy_params": {"window_months": 6}},
        {"strategy": "expanding"},
    ]}
    out = sweep.expand_grid(grid, **_KW)
    assert len(out) == 2
    assert {c["strategy"] for c in out} == {"rolling", "expanding"}
    assert all("config_hash" in c for c in out)


def test_expand_dedups_identical_configs():
    grid = {"configs": [{"strategy": "expanding"}, {"strategy": "expanding"}]}
    out = sweep.expand_grid(grid, **_KW)
    assert len(out) == 1


def test_expand_over_cap_raises():
    grid = {"configs": [{"strategy": "expanding", "strategy_params": {"min_prior_months": i}} for i in range(30)]}
    with pytest.raises(ValueError, match="exceeding the max_candidates"):
        sweep.expand_grid(grid, **_KW)


def test_expand_zero_candidates_raises():
    with pytest.raises(ValueError, match="zero candidates"):
        sweep.expand_grid({"configs": []}, **_KW)


def test_explicit_config_inherits_defaults():
    grid = {"configs": [{"strategy": "rolling"}]}
    out = sweep.expand_grid(grid, **_KW)
    assert out[0]["models"] == ["lgbm_cluster", "chronos2_enriched"]
    assert out[0]["metric"] == "wape"
    assert out[0]["lag_mode"] == "execution"
    assert out[0]["min_sku_rows"] == 3


def test_candidate_model_guard_accepts_canonical_models():
    candidates = [{"models": ["lgbm_cluster", "mstl", "chronos2_enriched"]}]
    sweep._validate_candidate_models(
        candidates,
        ["lgbm_cluster", "mstl", "chronos2_enriched"],
    )


def test_candidate_model_guard_rejects_retired_models():
    candidates = [{"models": ["lgbm_cluster", "catboost_cluster"]}]
    with pytest.raises(ValueError, match="catboost_cluster"):
        sweep._validate_candidate_models(candidates, ["lgbm_cluster", "mstl"])


# ---------------------------------------------------------------------------
# objective_score
# ---------------------------------------------------------------------------

_RES = {"champion_accuracy": 88.0, "gap_bps": 120.0, "lag_acc": [90.0, 86.0], "month_acc": [89.0, 87.0]}


def test_objective_accuracy():
    assert sweep.objective_score(_RES, "accuracy", lam=0.5, mu=0.25) == 88.0


def test_objective_gap_to_ceiling_prefers_smaller_gap():
    s_small = sweep.objective_score(dict(_RES, gap_bps=50.0), "gap_to_ceiling", lam=0.5, mu=0.25)
    s_big = sweep.objective_score(dict(_RES, gap_bps=300.0), "gap_to_ceiling", lam=0.5, mu=0.25)
    assert s_small > s_big


def test_objective_robust_penalises_dispersion():
    steady = {"champion_accuracy": 88.0, "gap_bps": 120.0, "lag_acc": [88.0, 88.0], "month_acc": [88.0, 88.0]}
    volatile = {"champion_accuracy": 88.0, "gap_bps": 120.0, "lag_acc": [98.0, 78.0], "month_acc": [98.0, 78.0]}
    assert sweep.objective_score(steady, "robust", lam=0.5, mu=0.25) > sweep.objective_score(volatile, "robust", lam=0.5, mu=0.25)


def test_objective_none_accuracy_returns_none():
    assert sweep.objective_score({"champion_accuracy": None}, "robust", lam=0.5, mu=0.25) is None


# ---------------------------------------------------------------------------
# gate_eligible
# ---------------------------------------------------------------------------

def test_gate_disabled_is_always_eligible():
    assert sweep.gate_eligible(10.0, 90.0, {"enabled": False}) is True


def test_gate_requires_min_improvement():
    gate = {"enabled": True, "min_wape_improvement_pct": 1.0}
    # baseline acc 90 -> wape 10; candidate acc 95 -> wape 5 -> 50% improvement
    assert sweep.gate_eligible(95.0, 90.0, gate) is True
    # candidate acc 90.05 -> wape 9.95 -> 0.5% improvement < 1%
    assert sweep.gate_eligible(90.05, 90.0, gate) is False


def test_gate_no_baseline_is_eligible():
    assert sweep.gate_eligible(70.0, None, {"enabled": True, "min_wape_improvement_pct": 1.0}) is True


# ---------------------------------------------------------------------------
# _pick_recommendation
# ---------------------------------------------------------------------------

def test_recommendation_prefers_gate_eligible_even_if_lower_score():
    options = [
        {"experiment_id": 1, "score": 90.0, "gate_eligible": False},
        {"experiment_id": 2, "score": 80.0, "gate_eligible": True},
    ]
    assert sweep._pick_recommendation(options)["experiment_id"] == 2


def test_recommendation_falls_back_to_top_score_when_none_eligible():
    options = [
        {"experiment_id": 1, "score": 90.0, "gate_eligible": False},
        {"experiment_id": 2, "score": 80.0, "gate_eligible": False},
    ]
    assert sweep._pick_recommendation(options)["experiment_id"] == 1


def test_recommendation_empty_is_none():
    assert sweep._pick_recommendation([]) is None


# ---------------------------------------------------------------------------
# _stdev
# ---------------------------------------------------------------------------

def test_stdev_zero_for_singleton():
    assert sweep._stdev([5.0]) == 0.0


def test_stdev_sample():
    assert round(sweep._stdev([90.0, 80.0, 70.0]), 4) == 10.0
