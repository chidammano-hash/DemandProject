# 32 — Fine-Tuned Foundation Forecasting Models

**Status:** Implemented (code) — full fine-tune run + causal-holdout gate pending
**Date:** 2026-06-19
**Related:** 31-algorithm-improvement-backlog.md, 03-backtest-framework.md, 07-champion-selection.md, 24-candidate-forecast-promotion.md, 19-forecast-pipeline-config.md

> ## Implementation status (2026-06-19)
> **Landed & smoke-verified end-to-end** (fine-tune → checkpoint → reload via dispatcher → predict, loss dropping over steps):
> - Config `algorithms.chronos_bolt_ft` (+ `fine_tune` block) — `config/forecasting/forecast_pipeline_config.yaml`.
> - Fine-tune script `scripts/ml/finetune_chronos_bolt.py` (causal `--train-cutoff`, HF `Trainer` on `ChronosBoltModelForForecasting`, versioned checkpoint + `training_metadata.json`).
> - Dispatcher `_run_chronos_bolt_ft` + `_resolve_ft_checkpoint` (numeric-versioned/flat/base-fallback) in `common/ml/expert_panel/foundation_models.py`; registered in `_FOUNDATION_DISPATCH`.
> - Backtest wrapper `scripts/ml/run_backtest_chronos_bolt_ft.py`.
> - **Production gap closed** — `_generate_finetuned_bolt_production()` (batched real-model path, grouped by last month) in `scripts/forecasting/generate_production_forecasts.py`; guarded (no-ops to the formula when no checkpoint).
> - Make targets `finetune-chronos-bolt` / `backtest-bolt-ft(-full)`; `accelerate` added to the `foundation` extra; unit tests `tests/unit/test_finetune_chronos_bolt.py` + `test_foundation_ft_dispatch.py` (9 tests).
>
> **Phase-0 gate (passed):** chronos-forecasting 2.2.2 + torch/transformers/accelerate present; `ChronosBoltModelForForecasting` is HF-`Trainer`-trainable; `amazon/chronos-bolt-base` cached. Noise-floor confirmed the targets are model-fixable (`very_low_volume_steady` floor 67% vs realized ~15-29%; `medium_volume_moderate` 40% vs ~6-16%); `medium_volume_seasonal_volatile` (floor 0.4%) is intrinsic noise → excluded.
>
> **Remaining (operator-run, compute-heavy):** a real fine-tune (`make finetune-chronos-bolt ARGS="--train-cutoff <cutoff>"`, GPU recommended) → `make backtest-bolt-ft` scored on the **post-cutoff holdout only** → gate vs incumbent → flip `compete/forecast: true` + add to `champion.models` on promotion.

---

## 1. Problem

Forecast accuracy is stuck at a ~74–75% plateau (volume-weighted WAPE; oracle ceiling ~86.5%). The
diagnostic + experimental work in [31-algorithm-improvement-backlog.md](31-algorithm-improvement-backlog.md)
ruled out every *cheap* lever with evidence: champion-strategy swaps (tournament), static per-cluster
routing (causal holdout), customer/OOS enrichment (refuted), and horizon fixes (flat lag curves). The
only lever left that can **move the ceiling** is genuinely better base models on the hard segments.

Two facts make foundation-model fine-tuning the highest-leverage single bet:

1. **The foundation models already win the hard segments — zero-shot.** Per-cluster, at execution lag,
   on existing backtests: `nhits`/`nbeats`/`chronos`/`bolt_hierarchical` beat the per-cluster trees on
   every low/medium-volume cluster, often massively (e.g. `very_low_volume_steady`: a foundation model
   hit ~85% where the per-cluster `lgbm` hit ~1%). Per-cluster trees **starve** on sparse series;
   pretrained models that cross-learn across the whole portfolio do not.

2. **The foundation models are 100% untapped.** Every entry in `forecast_pipeline_config.yaml`
   (`chronos`, `chronos_bolt`, `chronos2`, `chronos2_enriched`, `bolt_hierarchical`) runs **zero-shot**
   (`tune: false`) — pretrained weights, never adapted to this company's demand. Fine-tuning on the
   41M-row demand history is the canonical way to raise their ceiling, especially for the low-volume
   and high-CV clusters where the marginal value of in-domain adaptation is largest.

> **Audit finding that reshapes scope (confirm in Phase 0):** `scripts/forecasting/generate_production_forecasts.py`
> does **not** call the real foundation models for the production (forward) forecast — it uses a
> heuristic seasonal+recency **formula fallback** for any `model_id` in the chronos/bolt family. So
> today's foundation models only influence **backtest** and the **expert panel**, not the forward
> production forecast. Fine-tuning is worthless in production unless we also wire the real model into
> production inference. **Closing that gap is a first-class deliverable of this spec, not an aside.**

---

## 2. Goals / Non-goals

**Goals**
- Add a **fine-tuned Chronos-Bolt variant** (`chronos_bolt_ft`) trained on the company's demand history,
  integrated as a normal roster algorithm (backtest → champion pool → production), gated identically.
- Demonstrate a **ceiling lift on the hard clusters** (low-volume + high-CV) vs the zero-shot parent,
  large enough that adding it to the champion pool raises portfolio accuracy past the promote gate.
- **Close the production-inference gap** so the real (fine-tuned) foundation model produces the forward
  forecast, not a formula.

**Non-goals**
- No new champion-selection math (the champion already blends/selects; this only improves a candidate).
- Not fine-tuning every foundation model at once — Bolt first; generalize later (§9, Phase 3).
- Not chasing the high-CV segment if Phase 0 shows it's noise-floored (that routes to quantile/SLA work,
  backlog item 5 — out of scope here).

---

## 3. Design principles

1. **Reuse the foundation-backtest harness, don't reinvent it.** A fine-tuned model is just another
   `FoundationModelSpec` with a different dispatcher key and a checkpoint to load. Timeframe generation,
   per-DFU series assembly, checkpointing, post-processing, and accuracy all come from
   `common/ml/foundation_backtest.py` unchanged.
2. **Fine-tune the underlying HF model with the library recipe, then load the checkpoint with the
   existing pipeline.** Chronos-Bolt fine-tunes via `chronos-forecasting`'s training entry point on the
   underlying T5; the result is a checkpoint dir that `ChronosBoltPipeline.from_pretrained(<dir>)` loads
   exactly like the hub weights. **Inference code does not change** — only *where the weights come from*.
3. **Training is a separate, explicit step — never inside `model_registry`.** Tree `.fit()` goes through
   `model_registry.fit_model`; foundation FT is a standalone script writing a checkpoint. `_FoundationStub`
   stays inference-only. This keeps the "all tree `.fit()` via model_registry" rule intact.
4. **Causal training.** Fine-tune on a TRAIN window only; evaluate on a later holdout. No leakage —
   mirror the backtest framework's timeframe/embargo discipline (`03-backtest-framework.md`).
5. **Gate like any model.** The FT variant earns promotion only via the existing
   `champion.promote_gate` (≥1% WAPE improvement, ≥80% coverage) and the candidate→production path
   (`24-candidate-forecast-promotion.md`). No special-casing.
6. **Checkpoints are versioned artifacts.** Fine-tuned weights live under `data/models/<model_id>/` with
   a version + training metadata; promotion records the exact checkpoint used.

---

## 4. Architecture & data flow

```
                fine-tune (TRAIN window)                 evaluate (HOLDOUT)              promote
demand history ───────────────────────────► checkpoint ──────────────────► backtest ──────────► production
(fact_external/                              data/models/                    foundation_backtest   candidate→
 sales monthly)   scripts/ml/finetune_       chronos_bolt_ft/v{N}/          (existing harness)     production
                  chronos_bolt.py            + training_metadata.json        + champion pool        + REAL model
                  (chronos-forecasting                                       gate vs incumbent      inference
                   training recipe)                                                                 (gap closed)
```

Stages:
1. **Data prep** — assemble per-series training data (sku_ck → monthly qty, TRAIN window only) in the
   format the chronos training recipe expects (long/arrow dataset). Reuse `load_backtest_data()`.
2. **Fine-tune** — `scripts/ml/finetune_chronos_bolt.py` runs the library training loop (base =
   `amazon/chronos-bolt-{small|base}`), writes a checkpoint + `training_metadata.json` to
   `data/models/chronos_bolt_ft/v{N}/`. Device via existing `_resolve_device()` (MPS/CUDA/CPU,
   `DEMAND_GPU`).
3. **Backtest** — `scripts/ml/run_backtest_chronos_bolt_ft.py` registers a `FoundationModelSpec`
   (dispatcher key `chronos_bolt_ft`) whose loader points the pipeline at the checkpoint dir; everything
   else flows through the standard foundation backtest. Falls back to the zero-shot base if no checkpoint
   exists (so the roster never hard-fails).
4. **Evaluate & gate** — accuracy computed by the standard post-processing; compared to the incumbent
   champion via the promote gate, **focused on the hard clusters** (success = ceiling moves there).
5. **Production inference (gap closed)** — `generate_production_forecasts.py` calls the real
   `chronos_bolt_ft` pipeline for the forward forecast when the model is in the production/champion set,
   replacing the formula fallback for this variant.

---

## 5. Config schema

New roster entry (mirrors the existing foundation schema; adds a `fine_tune` block):

```yaml
algorithms:
  chronos_bolt_ft:
    type: foundation                  # same inference type; FT is a weights source, not a new type
    enabled: true
    tune: false                       # FT != hyperparameter tuning (Optuna)
    backtest: true
    compete: true                     # joins the champion pool once it beats its parent
    forecast: true
    expert: false
    output_dir: data/backtest/chronos_bolt_ft
    params:
      base_model: amazon/chronos-bolt-base   # parent checkpoint (hub id)
      checkpoint_dir: data/models/chronos_bolt_ft   # versioned FT weights (vN subdir resolved at load)
      device: auto
      batch_size: 512
      prediction_length: 6
      num_workers: 1
      fine_tune:
        enabled: true
        context_length: 36            # months of history fed during training
        learning_rate: 1.0e-4
        num_steps: 2000               # or epochs; tuned in Phase 1
        warmup_ratio: 0.1
        min_series_months: 6          # exclude ultra-short series from training
        train_cutoff: causal          # train window only; holdout reserved for eval
```

Roster wiring: add `chronos_bolt_ft` to `champion.models` once it clears the gate; it is picked up by
`get_competing_model_ids()` / `get_forecastable_model_ids()` automatically via `enabled`+`compete`/`forecast`.

---

## 6. New components

| Component | Path | Role |
|---|---|---|
| Fine-tune script | `scripts/ml/finetune_chronos_bolt.py` | data prep + library training loop → checkpoint + metadata |
| Backtest wrapper | `scripts/ml/run_backtest_chronos_bolt_ft.py` | `FoundationModelSpec` for the FT variant (loads checkpoint) |
| Dispatcher entry | `common/ml/expert_panel/foundation_models.py` | `_run_chronos_bolt_ft()` = Bolt inference pointed at `checkpoint_dir` (else base) + add to dispatch table |
| Checkpoint convention | `data/models/chronos_bolt_ft/v{N}/` | versioned weights + `training_metadata.json` (base, window, steps, val loss) |
| Production inference | `scripts/forecasting/generate_production_forecasts.py` | call the real FT pipeline for this `model_id` (close the formula-fallback gap) |
| Make targets | `Makefile` | `finetune-chronos-bolt`, `backtest-chronos-bolt-ft` |
| Deps | `pyproject.toml` | `foundation` extra: training deps (`accelerate`, `transformers`, and chronos training extras) |

Note: the loader change is small — `_run_chronos_bolt_ft()` is `_run_chronos_bolt()` with the pipeline
built from `checkpoint_dir` instead of the hub id. No change to the prediction path.

---

## 7. Evaluation & gating

- **Causal split:** train on the first N months, hold out the last M (reuse the framework's
  timeframe/embargo split, as in the routing validation: train ≤ cutoff → holdout > cutoff).
- **Primary success metric — ceiling movement on the hard clusters:** `chronos_bolt_ft` accuracy vs its
  zero-shot parent (`chronos_bolt`) on `very_low_volume_steady`, `low_volume_periodic`,
  `medium_volume_*volatile/moderate`, measured via `agg_accuracy_by_dfu` + the
  `/forecast/accuracy/decomposition` endpoint (spec 31's diagnostic layer is the instrument).
- **Promotion metric — portfolio:** add `chronos_bolt_ft` to the champion pool, re-run champion
  selection, and check the champion's accuracy clears the gate (≥1% rel WAPE improvement, ≥80%
  coverage) vs the current incumbent. Only then promote.
- **Guardrail:** report per-segment so a gain on low-volume isn't masking a regression on the healthy
  high-volume core.

---

## 8. Dependencies & infra

- **Library:** `chronos-forecasting` is already a `foundation` optional dep; fine-tuning needs its
  training extras + `accelerate`/`transformers` (add to the `foundation` group in `pyproject.toml`).
- **Compute:** Bolt-small/base fine-tunes are modest. Device resolution already exists
  (`_resolve_device`, `DEMAND_GPU`, MPS/CUDA/CPU). CUDA strongly preferred for training; MPS/CPU viable
  for small Bolt at higher wall-clock. **Phase 0 measures actual runtime** to decide local vs GPU box.
- **Storage:** checkpoints in `data/models/chronos_bolt_ft/` (gitignored `data/`); add to
  `clean-artifacts`/`fresh-*` housekeeping in the `Makefile`.

---

## 9. Rollout phases

- **Phase 0 — feasibility spike (cheap, gating).** (a) Confirm `chronos-forecasting` exposes a usable
  fine-tune recipe for Bolt and that a checkpoint reloads via `ChronosBoltPipeline.from_pretrained`.
  (b) Confirm/close the production formula-fallback finding. (c) Measure fine-tune runtime on this
  hardware. (d) Run backlog **item 0** (irreducible-noise floor) to confirm the low-volume segments are
  model-fixable. Decision gate: proceed only if FT is supported and the segments are fixable.
- **Phase 1 — single variant, backtest only.** Implement `finetune_chronos_bolt.py` +
  `run_backtest_chronos_bolt_ft.py` + dispatcher entry + config. Fine-tune on the causal train window,
  backtest the holdout, measure hard-cluster ceiling movement vs the zero-shot parent.
- **Phase 2 — production integration + promotion.** Close the production-inference gap for the variant,
  add it to the champion pool, re-run selection, gate, promote via the standard path.
- **Phase 3 — generalize.** Apply the same recipe to `chronos2`; explore per-segment fine-tunes (a Bolt
  fine-tuned on low-volume series only) if Phase 1 shows segment-specific gains.

---

## 10. Testing (per repo rules)

- **Unit** (`tests/unit/test_finetune_chronos_bolt.py`): data-prep shaping (series → training format,
  TRAIN-window cutoff, `min_series_months` filter); checkpoint-dir/version resolution; config parsing
  of the `fine_tune` block. Mock the training loop (no real GPU in CI).
- **Unit** (`tests/unit/test_foundation_dispatch.py` extension): `_run_chronos_bolt_ft` loads from
  `checkpoint_dir` when present and falls back to base when absent.
- **Integration**: a tiny synthetic backtest (few DFUs, few months) through
  `run_backtest_chronos_bolt_ft.py` producing predictions + metadata, asserting the output schema
  matches the other foundation backtests.
- **No new API endpoints** in Phase 1–2 (reuse `/backtest-management/*` + champion endpoints), so no new
  API tests beyond confirming the variant appears in the roster responses.
- Run `make test-all` after each phase.

---

## 11. Success criteria

1. `chronos_bolt_ft` beats zero-shot `chronos_bolt` on the hard clusters by a clear margin on the causal
   holdout (target: +5pt accuracy on `low_volume_periodic`-class clusters; low-volume is the
   highest-confidence win).
2. Adding it to the champion pool lifts portfolio accuracy enough to **clear the promote gate** vs the
   incumbent (≥1% rel WAPE, ≥80% coverage).
3. Production forward forecasts for the variant come from the **real model**, not the formula fallback.
4. If (1) fails on the high-CV clusters, that's an accepted negative result → those route to backlog
   item 5 (quantile/SLA), and we keep the low-volume win.

---

## 12. Open questions

- Does the installed `chronos-forecasting` version expose Bolt fine-tuning directly, or do we fine-tune
  the underlying T5 via `transformers` Trainer? (Phase 0)
- Fine-tune **one global** Bolt on all series, or **segment-specific** Bolts (e.g. a low-volume-only
  fine-tune)? Phase 1 global first; Phase 3 tests segment-specific.
- Is the production formula fallback intentional (latency/throughput at 24-month horizon)? If so,
  production must call the real model only for promoted foundation variants, and we measure the
  throughput cost. (Phase 0/2)
- Checkpoint reproducibility: pin base-model revision + training seed in `training_metadata.json`.
