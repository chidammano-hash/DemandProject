-- 194_create_dfu_naive_scale_view.sql
-- Per-DFU in-sample seasonal-naive MAE — the MASE denominator (scale q).
--
-- MASE (common/services/metrics.py compute_mase) scores an eval-window forecast
-- RELATIVE to the in-sample seasonal-naive forecast, so it is scale-free and fair
-- to structurally-hard long-tail SKUs whose tiny base makes WAPE% brutal:
--
--     MASE = mean(|forecast - actual|) over eval window  /  q
--     q    = mean(|x_t - x_{t-m}|) over the IN-SAMPLE series (m = 1 or 12)
--
-- To surface per-DFU MASE in the accuracy-decomposition layer we need the scale q
-- precomputed per DFU. This MV produces it, one row per (item_id, customer_group,
-- loc), leakage-safely. (The consuming endpoint — F-03b — arrives separately.)
--
-- LEAKAGE RULE (the whole ballgame): fact_sales_monthly (2023-06..2026-05) OVERLAPS
-- the eval/backtest window in fact_external_forecast_monthly (2025-06..2026-05). The
-- scale MUST be computed STRICTLY from history BEFORE the eval window — otherwise the
-- denominator borrows information from the period being scored and MASE is optimistic.
-- So the in-sample series is `startdate < per-DFU eval cutoff`, where the cutoff is
-- that DFU's earliest backtested target month (MIN startdate over its rows in
-- fact_external_forecast_monthly with tothist_dmd & basefcst_pref NOT NULL).
--
-- DENSIFICATION (required for intermittent demand): ~24% of (DFU, month) demand
-- pairs have NO sales row — zero-demand months are not materialized. The naive
-- baseline for intermittent SKUs must COUNT those zeros, so across each DFU's
-- in-sample span [first in-sample month .. month-before-cutoff] we generate_series
-- every month, LEFT JOIN actuals, COALESCE(qty, 0), THEN take the naive diffs.
--
-- In-sample actuals source = fact_sales_monthly.qty (matches the forecast fact's
-- actual tothist_dmd 98.6%; qty_shipped only ~80%). DFU grain = the full 3-key
-- (item_id, customer_group, loc) — match on ALL THREE keys (the dim_sku fan-out trap).
--
-- Refreshed alongside agg_accuracy_by_dfu / agg_accuracy_by_dim after backtest-load
-- (same cadence) via `make refresh-accuracy-mvs` / `accuracy-slice-refresh` /
-- `refresh-mvs-tiered`. Ordered AFTER its source tables (fact_sales_monthly +
-- fact_external_forecast_monthly are tier-1 facts; this MV joins both).

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_dfu_naive_scale AS
WITH eval_cutoff AS (
  -- Per-DFU cutoff = earliest backtested target month. Everything < this is in-sample.
  SELECT
    item_id,
    customer_group,
    loc,
    MIN(date_trunc('month', startdate)::date) AS eval_min_month
  FROM fact_external_forecast_monthly
  WHERE tothist_dmd IS NOT NULL
    AND basefcst_pref IS NOT NULL
  GROUP BY item_id, customer_group, loc
),
insample AS (
  -- In-sample monthly actuals, STRICTLY before the cutoff (leakage filter). Summed
  -- per DFU-month because fact_sales_monthly can carry multiple rows per month.
  SELECT
    s.item_id,
    s.customer_group,
    s.loc,
    date_trunc('month', s.startdate)::date     AS month,
    SUM(s.qty)::double precision               AS qty
  FROM fact_sales_monthly s
  JOIN eval_cutoff c
    ON s.item_id = c.item_id
   AND s.customer_group = c.customer_group
   AND s.loc = c.loc
  WHERE date_trunc('month', s.startdate)::date < c.eval_min_month
  GROUP BY s.item_id, s.customer_group, s.loc, date_trunc('month', s.startdate)::date
),
bounds AS (
  -- In-sample span endpoints per DFU (drives densification). Carries the eval cutoff
  -- so it flows through to the final row without a re-join to eval_cutoff.
  SELECT
    i.item_id,
    i.customer_group,
    i.loc,
    c.eval_min_month,
    MIN(i.month) AS first_month,
    MAX(i.month) AS last_month
  FROM insample i
  JOIN eval_cutoff c
    ON i.item_id = c.item_id
   AND i.customer_group = c.customer_group
   AND i.loc = c.loc
  GROUP BY i.item_id, i.customer_group, i.loc, c.eval_min_month
),
months AS (
  -- Densify: every month in [first_month .. last_month], including zero-demand gaps.
  -- eval_min_month rides along so the leakage cutoff lands on the final row directly.
  SELECT
    b.item_id,
    b.customer_group,
    b.loc,
    b.eval_min_month,
    gs::date AS month
  FROM bounds b
  CROSS JOIN LATERAL generate_series(b.first_month, b.last_month, interval '1 month') AS gs
),
dense AS (
  -- COALESCE missing months to 0 so the naive baseline counts intermittent zeros.
  SELECT
    m.item_id,
    m.customer_group,
    m.loc,
    m.eval_min_month,
    m.month,
    COALESCE(i.qty, 0)::double precision AS qty
  FROM months m
  LEFT JOIN insample i
    ON m.item_id = i.item_id
   AND m.customer_group = i.customer_group
   AND m.loc = i.loc
   AND m.month = i.month
),
diffs AS (
  -- Naive absolute diffs: m=1 (random-walk) and m=12 (annual seasonal). LAG is NULL
  -- for the first m months of each DFU, which AVG ignores below.
  SELECT
    item_id,
    customer_group,
    loc,
    eval_min_month,
    month,
    ABS(qty - LAG(qty, 1)  OVER w) AS d1,
    ABS(qty - LAG(qty, 12) OVER w) AS d12
  FROM dense
  WINDOW w AS (PARTITION BY item_id, customer_group, loc ORDER BY month)
)
SELECT
  item_id,
  customer_group,
  loc,
  COUNT(*)::int                AS n_insample_months,  -- densified month count
  MIN(month)                   AS insample_min_month,
  MAX(month)                   AS insample_max_month,
  MIN(eval_min_month)          AS eval_min_month,      -- the leakage cutoff (constant per DFU)
  AVG(d1)::double precision    AS scale_m1,            -- NULL when < 2 months
  AVG(d12)::double precision   AS scale_m12            -- NULL/sparse when < 13 months
FROM diffs
GROUP BY item_id, customer_group, loc
WITH NO DATA;

-- Unique index backs REFRESH MATERIALIZED VIEW CONCURRENTLY. The DFU 3-key is unique
-- (one row per item_id × customer_group × loc). Plain (non-CONCURRENTLY) build is safe
-- here because the MV is created WITH NO DATA.
CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_dfu_naive_scale
  ON agg_dfu_naive_scale (item_id, customer_group, loc);
