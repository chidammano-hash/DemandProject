-- 193_add_source_mix.sql
-- Per-DFU-month blend composition for blended champions.
--
-- The champion selection picks a model (or blends several) per DFU per month.
-- `source_model_id` (sql/041) records the single winning model; for BLENDED
-- champions (ensemble / learned_blend / shrinkage / etc.) that's just the
-- "ensemble" label. `source_mix` records the actual mix — a JSON array of
-- {"model": <id>, "weight": <0-1>} — so the Item Analysis chart can show the
-- per-month champion composition, e.g. "champion (40% NBEATS, 35% LGBM)".
-- NULL for single-model picks (implies 100% of source_model_id).
--
-- Idempotent: IF NOT EXISTS.

ALTER TABLE fact_external_forecast_monthly
    ADD COLUMN IF NOT EXISTS source_mix JSONB;

COMMENT ON COLUMN fact_external_forecast_monthly.source_mix IS
    'Blend composition for blended champions: JSON array of {model, weight}. NULL = single-model pick.';
