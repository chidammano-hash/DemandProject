"""Roster-coverage guards for the champion pipeline (loop-3 backtest-all-roster-drift).

Two distinct failure modes are pinned here:

1. **Pre-champion coverage assertion** (durable root-cause guard): every model_id
   in ``get_competing_model_ids()`` MUST have rows in ``fact_external_forecast_monthly``
   before champion selection runs. Otherwise a ``compete: true`` model that was never
   backtested+loaded silently drops out of the competition and the champion is picked
   from a partial field. The assertion fails loud (raises) on the gap.

2. **Makefile / config parity**: every ``compete: true`` id in
   ``forecast_pipeline_config.yaml`` whose backtest is produced sequentially must be
   covered by the ``backtest-all`` chain (directly or transitively), so a clean rebuild
   (``fresh-backtest`` / ``setup-backtest``) produces the full competing roster.
"""

import re
from unittest.mock import MagicMock

import pytest

from scripts.ml.run_champion_selection import assert_competing_models_covered


class TestPreChampionCoverageAssertion:
    """assert_competing_models_covered must fail loud on a missing competing model."""

    def _cursor_with_counts(self, counts: dict[str, int]) -> MagicMock:
        """Build a cursor whose count query returns (model_id, n_rows) tuples."""
        cur = MagicMock()
        cur.fetchall.return_value = list(counts.items())
        return cur

    def test_raises_when_a_competing_model_has_zero_rows(self):
        models = ["lgbm_cluster", "mstl", "nhits"]
        cur = self._cursor_with_counts({"lgbm_cluster": 100, "mstl": 80})
        with pytest.raises(RuntimeError, match="nhits"):
            assert_competing_models_covered(cur, models)

    def test_raises_when_a_competing_model_has_zero_count_row(self):
        models = ["lgbm_cluster", "mstl"]
        # Present in the result set but with a 0 count (e.g. all-NULL load).
        cur = self._cursor_with_counts({"lgbm_cluster": 100, "mstl": 0})
        with pytest.raises(RuntimeError, match="mstl"):
            assert_competing_models_covered(cur, models)

    def test_error_lists_every_missing_model(self):
        models = ["lgbm_cluster", "nbeats", "chronos2_enriched"]
        cur = self._cursor_with_counts({"lgbm_cluster": 100})
        with pytest.raises(RuntimeError) as exc:
            assert_competing_models_covered(cur, models)
        msg = str(exc.value)
        assert "nbeats" in msg
        assert "chronos2_enriched" in msg

    def test_passes_when_all_competing_models_present(self):
        models = ["lgbm_cluster", "mstl", "nhits"]
        cur = self._cursor_with_counts({"lgbm_cluster": 100, "mstl": 80, "nhits": 90})
        # No raise == coverage satisfied (function returns None).
        assert assert_competing_models_covered(cur, models) is None

    def test_query_is_parameterised_and_scoped_to_the_models(self):
        models = ["lgbm_cluster", "mstl"]
        cur = self._cursor_with_counts({"lgbm_cluster": 100, "mstl": 80})
        assert_competing_models_covered(cur, models)
        sql, params = cur.execute.call_args[0]
        # psycopg3 placeholders only — never $1 / f-string value interpolation.
        assert "%s" in sql
        assert "$1" not in sql
        assert list(params) == models


class TestBacktestAllRosterParity:
    """backtest-all must cover every sequentially-produced compete:true model."""

    def _backtest_all_recipe(self) -> str:
        """Return the transitive prerequisite text of the backtest-all target.

        Resolves aggregate prerequisites one level deep.
        """
        from common.core.paths import PROJECT_ROOT

        text = (PROJECT_ROOT / "Makefile").read_text()
        # Grab the backtest-all line and resolve any aggregate prereqs it names.
        m = re.search(r"^backtest-all:(.*)$", text, re.MULTILINE)
        assert m, "backtest-all target not found in Makefile"
        prereqs = m.group(1).split("##")[0].split()

        resolved: list[str] = []
        for p in prereqs:
            resolved.append(p)
            sub = re.search(rf"^{re.escape(p)}:(.*)$", text, re.MULTILINE)
            if sub:
                resolved.extend(sub.group(1).split("##")[0].split())
        return " ".join(resolved)

    def _target_to_model(self) -> dict[str, str]:
        """Map backtest-* prerequisite target names to the model_id they produce."""
        return {
            "backtest-lgbm": "lgbm_cluster",
            "backtest-chronos2e": "chronos2_enriched",
            "backtest-mstl": "mstl",
            "backtest-nhits": "nhits",
            "backtest-nbeats": "nbeats",
        }

    def test_backtest_all_covers_every_competing_model(self):
        from common.core.utils import get_competing_model_ids

        recipe = self._backtest_all_recipe()
        produced = {
            model
            for target, model in self._target_to_model().items()
            if target in recipe.split()
        }
        required = set(get_competing_model_ids())
        missing = required - produced
        assert not missing, (
            f"backtest-all does not produce compete:true models {sorted(missing)} — "
            "a clean rebuild can never select them as champion"
        )

    def test_canonical_five_are_in_backtest_all(self):
        recipe = self._backtest_all_recipe().split()
        for target in (
            "backtest-lgbm",
            "backtest-chronos2e",
            "backtest-mstl",
            "backtest-nhits",
            "backtest-nbeats",
        ):
            assert target in recipe, f"{target} missing from backtest-all chain"
