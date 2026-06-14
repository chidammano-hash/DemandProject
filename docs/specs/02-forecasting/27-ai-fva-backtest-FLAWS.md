# AI FVA Backtest — Known Flaws & Open Concerns

**Status:** Review notes — 2026-06-14
**Reviews:** [27-ai-fva-backtest.md](27-ai-fva-backtest.md)
**Implementation:** [common/ai/fva_recommender.py](../../../common/ai/fva_recommender.py) · [api/routers/forecasting/fva.py](../../../api/routers/forecasting/fva.py)

This document is a critical review of the AI Planner FVA backtest (spec 02-27). It exists so the
known weaknesses are visible to anyone selling, extending, or trusting the headline FVA number.
**It does not claim the feature is wrong to build — it claims the feature must be read as a
measurement/falsification harness, not as a forecasting improvement.**

---

## 1. The improvement thesis is conditional, not given

We already run tuned ML + statistical champion models. An LLM **cannot** beat those at pattern
extrapolation (trend, seasonality, intermittency) — the champion already extracts everything
learnable from history. The *only* legitimate value channel for an AI override is **exogenous
information the model never saw**: a known future promotion, a price change, a customer ramp/loss,
a one-time event, a new-product analog. This is the classic judgmental-override / FVA premise.

**Risk:** the AI context ([`DfuContext`](../../../common/ai/fva_recommender.py#L72-L105)) is almost
entirely backward-looking — past actuals, the baseline, and customer history. The one forward-looking
field is `notes` (anomaly/customer events), which is free-text and optional. **If `notes` is usually
empty, the AI is re-reading the same history the champion already fit, and there is no theoretical
reason it would beat the model — it would mostly add noise.**

**Required negative control:** run the backtest with `notes` stripped (history + baseline only) and
confirm FVA collapses to ~0. If lift appears even with context stripped, that is leakage or noise,
**not** value. This control should gate any positive headline number.

## 2. Unaddressed leakage vector — the LLM's parametric memory

Spec §4.3 and the leakage test (spec line 531) only sanitize the **context payload**. They do nothing
about hindsight the model carries **in its weights**. A current model (Opus knowledge cutoff Jan 2026)
asked to make a recommendation for a historical month may have effectively memorized known macro
events and product/customer outcomes for that period. No context-filtering test catches this.

**Consequence:** the metered-API ("authoritative") run is structurally advantaged over a local model
with no such memory, inflating the head-to-head and the headline lift. This must be named in spec §10
Risks. Mitigations to consider: prefer backtest windows that predate the model's training data only
where that is impossible (it usually isn't), report local-vs-API gap as a *suspect* signal rather than
a quality signal, and treat the negative control in §1 as the real arbiter.

## 3. `SHIFT_TIMING` is mislabeled (correctness bug)

Spec §4.2 (line 129) defines `SHIFT_TIMING` as moving demand between months
(`ai_forecast[from] -= delta; ai_forecast[to] += delta`). The apply rule
([`apply_recommendation`](../../../common/ai/fva_recommender.py#L316-L323)) treats it **identically to
`REPLACE`** — it overwrites per-month quantities and does not conserve/shift volume. The code comment
admits this. A recommendation type that is advertised in the taxonomy and **rolled up separately in
reporting** (`mv_ai_fva_by_recommendation_type`) does not do what its name and spec say.

**Fix before trusting per-recommendation-type rollups:** either implement true demand shifting or
rename the code to reflect REPLACE semantics.

## 4. `pct_change` schema vs. guardrail mismatch

The Pydantic field allows `-100..+500`
([`Recommendation.pct_change`](../../../common/ai/fva_recommender.py#L43)), while the prompt says
"typical -50..+50" and [`apply_guardrails`](../../../common/ai/fva_recommender.py#L229-L272) clips at
±50. Not a bug (guardrails clip), but a +500% ceiling on a forecast override is a footgun: a wild value
is accepted and logged before clipping. Tighten the schema bound or document why it is wide.

## 5. FVA waterfall integration SQL — pointless self-join

[`fva.py:111-118`](../../../api/routers/forecasting/fva.py#L111-L118) joins `mv_ai_fva_overall` to
itself (`m` and `o`) on the same `run_id`, then `SUM(o.n_dfus)` with a `GROUP BY`. If that MV is one
row per run (spec §4.6 grain), the self-join and SUM are redundant — `n_dfus` reads directly. Harmless
but confusing; reads like a leftover from a different grain. Simplify.

## 6. Reproducibility is overclaimed

Spec Goal 7 and §10 promise reproducibility. `temperature=0` does **not** make LLMs deterministic
across batching/hardware/provider/model-version, and "±5% rationale difference" (spec line 532) is not
a measurable acceptance criterion. Acceptable for MVP, but do not sell "reproducible" as a hard
guarantee — pin model + prompt version and report the run as *replayable from stored context*, not
*bit-identical*.

---

## Is any of this what other software does?

- **FVA / forecast waterfall, measured by accuracy lift over a baseline:** yes — textbook (Gilliland/SAS;
  standard in SAP IBP, Blue Yonder, o9, Kinaxis, John Galt). The waterfall in
  [`STAGE_DEFS`](../../../api/routers/forecasting/fva.py#L12-L18) is the canonical pattern, and the
  literature's blunt finding — overrides frequently *destroy* value — is exactly why you measure them.
- **A chat LLM emitting the numeric override:** no — not mainstream. Vendor "GenAI" is copilots
  (narration, anomaly explanation, scenario Q&A), not the LLM producing the forecast number. Credible
  "AI forecasting" uses **time-series foundation models** (TimeGPT, Chronos, TimesFM, Moirai) trained on
  time series — a different thing from prompting a chat model with "scale up by X%." Evidence that chat
  LLMs improve point-forecast accuracy over good statistical/ML baselines is weak-to-negative. This part
  is novel/fringe — defensible as an experiment, not vouched for by prior art.

## Bottom line

Build and keep it as a **measurement and falsification tool**, not a forecasting upgrade. Be prepared
for the honest result to be "no measurable lift," especially on the local model and especially if event
context is thin. Before trusting any positive number: run the §1 negative control, address/disclose the
§2 parametric-memory leakage, and fix the §3 `SHIFT_TIMING` mislabel.
