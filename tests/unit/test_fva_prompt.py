"""Unit tests for common.ai.fva_recommender — prompt construction only.

Existing tests/unit/test_fva_recommender.py already covers Recommendation
Pydantic validation, apply_guardrails, and apply_recommendation. This file
focuses on build_user_prompt: numeric formatting, optional-field handling,
and the metadata "none" fallback.
"""
from __future__ import annotations

import re
from datetime import date

from common.ai.fva_recommender import CustomerHistory, DfuContext, build_user_prompt


def _ctx(**overrides) -> DfuContext:
    base = {
        "item_id": "SKU-1",
        "loc": "LOC-A",
        "forecast_run_month": date(2026, 5, 1),
        "actuals_last_24m": [("2026-03", 100.0), ("2026-04", 110.0)],
        "baseline_forecast": [("2026-06", 120.0), ("2026-07", 130.0)],
        "cluster": "C1",
        "demand_pattern": "smooth",
        "abc_vol": "A",
        "notes": None,
    }
    base.update(overrides)
    return DfuContext(**base)


class TestBuildUserPrompt:
    def test_contains_item_loc_and_t(self):
        prompt = build_user_prompt(_ctx())
        assert "item_id=SKU-1" in prompt
        assert "location=LOC-A" in prompt
        assert "2026-05-01" in prompt

    def test_contains_actuals_and_baseline_strings(self):
        prompt = build_user_prompt(_ctx())
        assert "2026-03=100" in prompt
        assert "2026-04=110" in prompt
        assert "2026-06=120" in prompt
        assert "2026-07=130" in prompt

    def test_numeric_rendering_is_rounded_integer(self):
        # All numeric quantities must appear as `YYYY-MM=NNN` (no decimal point).
        prompt = build_user_prompt(_ctx(
            actuals_last_24m=[("2026-03", 99.4), ("2026-04", 100.7)],
            baseline_forecast=[("2026-06", 120.6), ("2026-07", 130.49)],
        ))
        # 99.4 -> 99, 100.7 -> 101, 120.6 -> 121, 130.49 -> 130 (round-half-to-even via %.0f)
        assert "2026-03=99" in prompt
        assert "2026-04=101" in prompt
        assert "2026-06=121" in prompt
        assert "2026-07=130" in prompt
        # No decimal points in the rendered qty values for any month token.
        qty_tokens = re.findall(r"\d{4}-\d{2}=\d+(?:\.\d+)?", prompt)
        assert qty_tokens, "prompt should contain rendered qty tokens"
        assert all("." not in tok.split("=", 1)[1] for tok in qty_tokens)

    def test_notes_omitted_when_none(self):
        prompt = build_user_prompt(_ctx(notes=None))
        assert "Anomaly/event notes" not in prompt

    def test_notes_rendered_when_present(self):
        prompt = build_user_prompt(_ctx(notes="huge promo on Apr 15"))
        assert "Anomaly/event notes: huge promo on Apr 15" in prompt

    def test_metadata_none_when_all_metadata_fields_missing(self):
        prompt = build_user_prompt(_ctx(
            cluster=None, demand_pattern=None, abc_vol=None,
        ))
        assert "Metadata: none" in prompt
        # And no stray "cluster=" / "pattern=" / "abc=" tokens.
        assert "cluster=" not in prompt
        assert "pattern=" not in prompt
        assert "abc=" not in prompt

    def test_metadata_partial_only_includes_present_fields(self):
        prompt = build_user_prompt(_ctx(
            cluster="C7", demand_pattern=None, abc_vol="B",
        ))
        assert "cluster=C7" in prompt
        assert "abc=B" in prompt
        assert "pattern=" not in prompt
        assert "Metadata: none" not in prompt

    def test_actuals_length_in_label(self):
        prompt = build_user_prompt(_ctx(
            actuals_last_24m=[(f"2025-{m:02d}", 100.0) for m in range(1, 13)],
        ))
        assert "Actuals (last 12 months ending at T)" in prompt

    def test_baseline_horizon_in_label(self):
        prompt = build_user_prompt(_ctx(
            baseline_forecast=[("2026-06", 1.0), ("2026-07", 2.0), ("2026-08", 3.0)],
        ))
        assert "T+1..T+3" in prompt

    def test_trailing_instruction_present(self):
        prompt = build_user_prompt(_ctx())
        # The user message must end with the action prompt so the model
        # always knows when input ends.
        assert prompt.strip().endswith("Return the JSON recommendation now.")


class TestTopCustomersInPrompt:
    """v1.1.0 — top-customer history rendered for the LLM."""

    def _customers(self) -> list[CustomerHistory]:
        return [
            CustomerHistory(
                customer_no="50592",
                customer_name="STMLS LLC",
                monthly=[("2025-05", 12.0), ("2025-06", 10.0), ("2025-07", 8.0),
                         ("2025-08", 6.0),  ("2025-09", 4.0),  ("2025-10", 2.0)],
            ),
            CustomerHistory(
                customer_no="68002",
                customer_name="WELCOME TO THE FARM",
                monthly=[("2025-05", 3.0), ("2025-06", 2.0), ("2025-07", 1.0)],
            ),
        ]

    def test_top_customers_section_renders_with_names_and_share(self):
        prompt = build_user_prompt(_ctx(top_customers=self._customers()))
        assert "Top customers" in prompt
        assert "50592" in prompt
        assert "STMLS LLC" in prompt
        assert "68002" in prompt
        assert "WELCOME TO THE FARM" in prompt
        # Per-month values appear in the line for each customer.
        assert "2025-05=12" in prompt
        assert "2025-10=2" in prompt
        # Share-of-top-set percentage is rendered.
        assert "% of top set" in prompt

    def test_no_top_customers_field_keeps_section_absent(self):
        """An older context with no customer info renders no customers line —
        the LLM sees the same prompt as in v1.0.0 for that DFU."""
        prompt = build_user_prompt(_ctx(top_customers=None))
        assert "Top customers" not in prompt

    def test_empty_customer_list_treated_as_absent(self):
        prompt = build_user_prompt(_ctx(top_customers=[]))
        assert "Top customers" not in prompt
