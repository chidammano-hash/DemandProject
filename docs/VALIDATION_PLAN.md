# Supply Chain Command Center — SKU-Level Validation Plan

## Context
Validate every functionality end-to-end by picking specific SKUs and asserting exact values at every step. Each test has a SQL assertion block that must pass. If any assertion fails, it becomes a ticket.

**Convention:** We pick SKUs after data loads by querying the DB. Every step below shows the "pick SKU" query and the assertion queries.

---

## Table of Contents

- [Step 1: Data Load & Row Counts](#step-1-data-load--row-counts)
- [Step 2: Explorer & Domain API](#step-2-explorer--domain-api-pick-1-sku-per-domain)
- [Step 3: Clustering](#step-3-clustering-assert-cluster-assignments-for-5-skus)
- [Step 4: LGBM Backtest](#step-4-lgbm-backtest-deep-validation-of-10-skus)
- [Step 5: CatBoost + XGBoost Backtests](#step-5-catboost--xgboost-backtests-same-10-skus)
- [Step 6: Champion Selection](#step-6-champion-selection-assert-correct-winner-for-10-skus)
- [Step 7: Accuracy & SHAP API](#step-7-accuracy--shap-api-assert-api-returns-for-10-skus)
- [Step 8: Safety Stock](#step-8-safety-stock-assert-formulas-for-5-skus)
- [Step 9: EOQ](#step-9-eoq-assert-formulas-for-same-5-skus)
- [Step 10: Replenishment Policies](#step-10-replenishment-policies-assert-policy-assignments-for-5-skus)
- [Step 11: Production Forecast & CI](#step-11-production-forecast--ci-assert-for-5-skus)
- [Step 12: Inventory Projection](#step-12-inventory-projection-assert-for-3-skus)
- [Step 13: Planned Orders](#step-13-planned-orders-assert-for-3-skus)
- [Step 14: Replenishment Plan](#step-14-replenishment-plan-assert-for-3-skus)
- [Step 15: Exceptions & Storyboard](#step-15-exceptions--storyboard-assert-for-3-skus)
- [Step 16: Fill Rate & Service Level](#step-16-fill-rate--service-level-assert-for-3-skus)
- [Step 17: Bias Corrections](#step-17-bias-corrections-assert-for-3-skus)
- [Step 18: S&OP Cycle](#step-18-sop-cycle-assert-stage-machine)
- [Step 19: Control Tower KPIs](#step-19-control-tower-kpis-assert-aggregation)
- [Step 20: Full Test Suite](#step-20-full-test-suite)
- [Step 21: Demand Variability](#step-21-demand-variability-assert-for-5-skus)
- [Gap Ticket Template](#gap-ticket-template)

---

## Step 1: Data Load & Row Counts

### 1.1 Normalize + Load
```bash
make normalize-all && make load-all
```

### 1.2 Assert: All Tables Populated
```sql
-- ASSERT: Every table has rows
DO $$
DECLARE
  r RECORD;
BEGIN
  FOR r IN
    SELECT unnest(ARRAY[
      'dim_item','dim_location','dim_customer','dim_time','dim_sku',
      'fact_sales_monthly','fact_external_forecast_monthly','fact_inventory_snapshot'
    ]) AS tbl
  LOOP
    EXECUTE format('SELECT count(*) FROM %I', r.tbl) INTO r;
    ASSERT r.count > 0, format('Table %s is empty!', r.tbl);
  END LOOP;
END $$;
```

### 1.3 Assert: Data Integrity Constraints
```sql
-- ASSERT: Sales are TYPE=1 only
SELECT COUNT(*) AS bad_rows FROM fact_sales_monthly WHERE type != 1;
-- Expected: 0

-- ASSERT: All sales dates are month-start
SELECT COUNT(*) AS bad_dates FROM fact_sales_monthly
WHERE startdate != date_trunc('month', startdate)::date;
-- Expected: 0

-- ASSERT: Forecast lags 0-4 only
SELECT COUNT(*) AS bad_lags FROM fact_external_forecast_monthly
WHERE lag NOT BETWEEN 0 AND 4;
-- Expected: 0

-- ASSERT: Forecast dates are month-start
SELECT COUNT(*) AS bad_fcst FROM fact_external_forecast_monthly
WHERE fcstdate != date_trunc('month', fcstdate)::date
   OR startdate != date_trunc('month', startdate)::date;
-- Expected: 0

-- ASSERT: Lag matches date arithmetic
SELECT COUNT(*) AS lag_mismatch FROM fact_external_forecast_monthly
WHERE lag != (
  (EXTRACT(YEAR FROM startdate) - EXTRACT(YEAR FROM fcstdate))::int * 12
  + (EXTRACT(MONTH FROM startdate) - EXTRACT(MONTH FROM fcstdate))::int
);
-- Expected: 0

-- ASSERT: No duplicate forecast keys
SELECT forecast_ck, model_id, COUNT(*) FROM fact_external_forecast_monthly
GROUP BY 1,2 HAVING COUNT(*) > 1;
-- Expected: 0 rows

-- ASSERT: No duplicate sales keys
SELECT sales_ck, COUNT(*) FROM fact_sales_monthly
GROUP BY 1 HAVING COUNT(*) > 1;
-- Expected: 0 rows
```

---

## Step 2: Explorer & Domain API (Pick 1 SKU per domain)

### 2.1 Pick SKUs
```sql
-- Pick a sample item
SELECT item_id, item_desc, brand_name FROM dim_item LIMIT 1;
-- Example result: item_id='100320', item_desc='BRAND X 750ML', brand_name='BRAND X'

-- Pick a sample DFU
SELECT item_id, customer_group, loc, execution_lag, abc_vol
FROM dim_sku WHERE execution_lag IS NOT NULL LIMIT 1;
-- Example: item_id='100320', customer_group='ALL', loc='1401-BULK', execution_lag=2
```

### 2.2 Assert: API returns correct data for that SKU
```bash
# ASSERT: Domain page returns our item
curl -s "http://localhost:8000/domains/item?q=100320&limit=1" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['total'] >= 1, 'Item not found'
row = d['rows'][0]
assert row['item_id'] == '100320', f'Wrong item: {row[\"item_id\"]}'
print('PASS: Item 100320 found via /domains/item')
"

# ASSERT: Suggest endpoint returns our item
curl -s "http://localhost:8000/domains/item/suggest?field=item_id&q=100320" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert '100320' in d['values'], 'Suggest did not return 100320'
print('PASS: Suggest returns 100320')
"

# ASSERT: Sales data exists for our DFU
curl -s "http://localhost:8000/domains/sales?limit=5&filters=%7B%22item_id%22:%22100320%22%7D" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['total'] > 0, 'No sales for item 100320'
row = d['rows'][0]
assert row['type'] == 1, f'Wrong type: {row[\"type\"]}'
print(f'PASS: {d[\"total\"]} sales rows for 100320')
"
```

---

## Step 3: Clustering (Assert cluster assignments for 5 SKUs)

### 3.1 Run
```bash
make cluster-all && make seasonality-all
```

### 3.2 Pick & Assert
```sql
-- Pick 5 SKUs across volume tiers
SELECT item_id, loc, ml_cluster, abc_vol, seasonality_profile,
       demand_mean, demand_cv, variability_class
FROM dim_sku
WHERE ml_cluster IS NOT NULL AND abc_vol IS NOT NULL
ORDER BY demand_mean DESC
LIMIT 5;
-- Save these 5 as SKU_C1..SKU_C5
```

### 3.3 Assert: Clustering Invariants
```sql
-- ASSERT: All DFUs have ml_cluster assigned
SELECT COUNT(*) AS unassigned FROM dim_sku WHERE ml_cluster IS NULL;
-- Document actual count; ideally 0

-- ASSERT: Cluster count in valid range [9, 18]
SELECT COUNT(DISTINCT ml_cluster) AS k FROM dim_sku WHERE ml_cluster IS NOT NULL;
-- Expected: 9 <= k <= 18

-- ASSERT: No cluster has < 2% of total DFUs
WITH totals AS (
  SELECT COUNT(*) AS total FROM dim_sku WHERE ml_cluster IS NOT NULL
),
clusters AS (
  SELECT ml_cluster, COUNT(*) AS n FROM dim_sku WHERE ml_cluster IS NOT NULL GROUP BY 1
)
SELECT c.ml_cluster, c.n, ROUND(100.0 * c.n / t.total, 2) AS pct
FROM clusters c, totals t
WHERE 100.0 * c.n / t.total < 2.0;
-- Expected: 0 rows (no cluster below 2%)

-- ASSERT: ABC classification complete (A/B/C for all with abc_vol)
SELECT abc_vol, COUNT(*) FROM dim_sku WHERE abc_vol IS NOT NULL GROUP BY 1 ORDER BY 1;
-- Expected: A, B, C only (no nulls or others)

-- ASSERT: Seasonality profiles valid
SELECT seasonality_profile, COUNT(*) FROM dim_sku
WHERE seasonality_profile IS NOT NULL GROUP BY 1;
-- Expected: values in (high, medium, low, none, insufficient_history)

-- ASSERT: For each of 5 SKUs, verify cluster label is meaningful
-- For SKU_C1 (highest demand_mean):
SELECT item_id, loc, ml_cluster, demand_mean, demand_cv
FROM dim_sku WHERE item_id = '<SKU_C1_ITEM>' AND loc = '<SKU_C1_LOC>';
-- ASSERT: ml_cluster contains 'high_volume' substring (high demand → high volume cluster)
```

---

## Step 4: LGBM Backtest (Deep validation of 10 SKUs)

### 4.1 Run
```bash
make backtest-lgbm && make backtest-load-lgbm
```

### 4.2 Pick 10 SKUs with varying execution lags
```sql
SELECT DISTINCT ON (d.execution_lag)
  d.item_id, d.customer_group, d.loc, d.execution_lag, d.ml_cluster, d.abc_vol
FROM dim_sku d
JOIN fact_sales_monthly s ON s.item_id = d.item_id AND s.loc = d.loc
WHERE d.execution_lag BETWEEN 0 AND 4
GROUP BY d.item_id, d.customer_group, d.loc, d.execution_lag, d.ml_cluster, d.abc_vol
HAVING COUNT(DISTINCT s.startdate) >= 24  -- at least 24 months history
ORDER BY d.execution_lag, COUNT(DISTINCT s.startdate) DESC;
-- Returns 5 rows (one per exec lag 0,1,2,3,4)
-- Pick 2 per exec lag = 10 SKUs total (add LIMIT 2 per group)
```

### 4.3 Assert for EACH of the 10 SKUs

Replace `<ITEM>`, `<CG>`, `<LOC>`, `<EXEC_LAG>` with actual values.

```sql
-- ═══════════════════════════════════════════════
-- ASSERTION BLOCK: Per-SKU Backtest Validation
-- Run this for EACH of 10 SKUs
-- ═══════════════════════════════════════════════

-- A1: Predictions exist at execution lag
SELECT COUNT(*) AS pred_count
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster';
-- ASSERT: pred_count > 0

-- A2: All predictions use correct lag = execution_lag
SELECT COUNT(*) AS wrong_lag
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
  AND lag != <EXEC_LAG>;
-- ASSERT: wrong_lag = 0

-- A3: fcstdate = startdate - execution_lag months
SELECT COUNT(*) AS date_mismatch
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
  AND fcstdate != (startdate - INTERVAL '<EXEC_LAG> months')::date;
-- ASSERT: date_mismatch = 0

-- A4: No negative forecasts
SELECT COUNT(*) AS negatives
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
  AND basefcst_pref < 0;
-- ASSERT: negatives = 0

-- A5: Actuals match fact_sales_monthly
SELECT COUNT(*) AS actual_mismatch
FROM fact_external_forecast_monthly f
LEFT JOIN fact_sales_monthly s
  ON f.item_id = s.item_id AND f.loc = s.loc AND f.startdate = s.startdate
WHERE f.item_id = '<ITEM>' AND f.loc = '<LOC>'
  AND f.model_id = 'lgbm_cluster'
  AND f.tothist_dmd IS NOT NULL
  AND s.qty IS NOT NULL
  AND ABS(f.tothist_dmd - s.qty) > 0.01;
-- ASSERT: actual_mismatch = 0

-- A6: Archive has all 5 lags (0-4)
SELECT lag, COUNT(*) FROM backtest_lag_archive
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
GROUP BY 1 ORDER BY 1;
-- ASSERT: 5 rows (lag 0,1,2,3,4), each with count > 0

-- A7: Accuracy at execution lag (compute WAPE)
SELECT
  100 - (100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
         / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS accuracy_pct,
  (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0)) - 1 AS bias
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
  AND tothist_dmd IS NOT NULL AND tothist_dmd != 0;
-- ASSERT: accuracy_pct IS NOT NULL (has valid data)
-- RECORD: accuracy_pct and bias for comparison in Step 6

-- A8: Lag curve — accuracy should generally degrade with lag
SELECT lag,
  100 - (100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
         / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS accuracy_pct
FROM backtest_lag_archive
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'lgbm_cluster'
  AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
GROUP BY 1 ORDER BY 1;
-- ASSERT: lag_0_accuracy >= lag_4_accuracy (generally, may not be strict)
-- RECORD: full lag curve for each SKU
```

---

## Step 5: CatBoost + XGBoost Backtests (Same 10 SKUs)

### 5.1 Run
```bash
make backtest-catboost && make backtest-xgboost && make backtest-load-all
```

### 5.2 Assert for EACH of 10 SKUs: Repeat A1-A8 with `model_id = 'catboost_cluster'` and `model_id = 'xgboost_cluster'`

### 5.3 Assert: Cross-model comparison
```sql
-- For each SKU: compare all 3 models + external
SELECT model_id,
  COUNT(*) AS rows,
  100 - (100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
         / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS accuracy_pct,
  (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0)) - 1 AS bias
FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
GROUP BY 1 ORDER BY 3 DESC;
-- RECORD: Which model wins per SKU
-- ASSERT: All 3 ML models + 'external' present
```

---

## Step 6: Champion Selection (Assert correct winner for 10 SKUs)

### 6.1 Run
```bash
make champion-all
```

### 6.2 Assert: Champion rows exist
```sql
-- ASSERT: Champion model exists
SELECT COUNT(*) FROM fact_external_forecast_monthly WHERE model_id = 'champion';
-- ASSERT: count > 0

-- ASSERT: source_model_id populated
SELECT COUNT(*) AS missing_source FROM fact_external_forecast_monthly
WHERE model_id = 'champion' AND source_model_id IS NULL;
-- ASSERT: missing_source = 0 (or document count)
```

### 6.3 Assert for EACH of 10 SKUs
```sql
-- C1: Champion accuracy >= worst individual model
WITH models AS (
  SELECT model_id,
    100 - (100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
           / NULLIF(ABS(SUM(tothist_dmd)), 0)) AS acc
  FROM fact_external_forecast_monthly
  WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
    AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
  GROUP BY 1
)
SELECT * FROM models ORDER BY acc DESC;
-- ASSERT: 'champion' accuracy >= MIN(lgbm, catboost, xgboost) accuracy

-- C2: Champion's source_model_id is a valid model
SELECT DISTINCT source_model_id FROM fact_external_forecast_monthly
WHERE item_id = '<ITEM>' AND customer_group = '<CG>' AND loc = '<LOC>'
  AND model_id = 'champion';
-- ASSERT: source_model_id IN ('lgbm_cluster', 'catboost_cluster', 'xgboost_cluster')

-- C3: Champion predictions count = execution-lag predictions of source model
SELECT
  (SELECT COUNT(*) FROM fact_external_forecast_monthly
   WHERE item_id='<ITEM>' AND loc='<LOC>' AND model_id='champion') AS champ_n,
  (SELECT COUNT(*) FROM fact_external_forecast_monthly
   WHERE item_id='<ITEM>' AND loc='<LOC>'
     AND model_id=(SELECT DISTINCT source_model_id FROM fact_external_forecast_monthly
                   WHERE item_id='<ITEM>' AND loc='<LOC>' AND model_id='champion' LIMIT 1)
  ) AS source_n;
-- ASSERT: champ_n approximately equals source_n (champion copies source predictions)
```

---

## Step 7: Accuracy & SHAP API (Assert API returns for 10 SKUs)

### 7.1 Assert: Accuracy slice endpoint
```bash
# ASSERT: Accuracy by model_id returns all models
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=model_id" | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = [b['bucket'] for b in d['buckets']]
for m in ['lgbm_cluster','catboost_cluster','xgboost_cluster','champion','external']:
    assert m in models, f'Model {m} missing from accuracy slice'
print(f'PASS: All models in accuracy slice ({len(d[\"buckets\"])} buckets)')
"

# ASSERT: Lag curve returns lags 0-4
curl -s "http://localhost:8000/forecast/accuracy/lag-curve" | python3 -c "
import sys, json
d = json.load(sys.stdin)
lags = [x['lag'] for x in d['by_lag']]
assert sorted(lags) == [0,1,2,3,4], f'Missing lags: {lags}'
print('PASS: Lag curve has lags 0-4')
"
```

### 7.2 Assert: SHAP for specific SKU
```bash
# ASSERT: SHAP summary exists for lgbm
curl -s "http://localhost:8000/forecast/shap/lgbm_cluster/summary" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert len(d['features']) > 0, 'No SHAP features'
assert d['features'][0]['mean_abs_shap'] > 0, 'SHAP values are zero'
print(f'PASS: SHAP has {len(d[\"features\"])} features, top={d[\"features\"][0][\"feature\"]}')
"
```

---

## Step 8: Safety Stock (Assert formulas for 5 SKUs)

### 8.1 Run
```bash
make setup-inv-planning
```

### 8.2 Pick 5 SKUs with known demand stats
```sql
SELECT d.item_id, d.loc, d.abc_vol,
  d.demand_mean AS demand_mean_monthly,
  d.demand_std AS demand_std_monthly,
  COALESCE(lt.lead_time_mean_days, 14) AS lt_mean_days,
  COALESCE(lt.lead_time_std_days, COALESCE(lt.lead_time_mean_days, 14) * 0.20) AS lt_std_days
FROM dim_sku d
LEFT JOIN dim_item_lead_time_profile lt ON d.item_id = lt.item_id AND d.loc = lt.loc
WHERE d.demand_mean > 0 AND d.abc_vol IS NOT NULL
ORDER BY d.demand_mean DESC
LIMIT 5;
-- Save as SKU_SS1..SKU_SS5 with their demand_mean, demand_std, lt_mean, lt_std, abc_vol
```

### 8.3 Assert: Formula verification for EACH SKU

```sql
-- ═══════════════════════════════════════════════
-- ASSERTION BLOCK: Safety Stock Formula Verification
-- Constants: DAYS_PER_MONTH = 30.44
-- Z lookup: A→2.054 (98%), B→1.645 (95%), C→1.282 (90%)
-- ═══════════════════════════════════════════════

SELECT
  t.item_id, t.loc,
  t.ss_demand_only,
  t.ss_lt_only,
  t.ss_combined,
  t.reorder_point,

  -- Manual formula recomputation:
  -- sigma_D_daily = demand_std / sqrt(30.44)
  -- D_avg_daily = demand_mean / 30.44
  -- SS_demand = Z * sqrt(lt_mean * (sigma_D_daily)^2)
  -- SS_lt = Z * D_avg_daily * lt_std
  -- SS_combined = Z * sqrt(lt_mean * (sigma_D_daily)^2 + (D_avg_daily)^2 * lt_std^2)
  -- ROP = D_avg_daily * lt_mean + SS_combined

  -- Use actual Z from the table (derive from abc_vol → service_level → z_table)
  CASE t.abc_vol WHEN 'A' THEN 2.054 WHEN 'B' THEN 1.645 ELSE 1.282 END AS expected_z,

  CASE t.abc_vol WHEN 'A' THEN 2.054 WHEN 'B' THEN 1.645 ELSE 1.282 END
    * SQRT(t.lt_mean_days * POWER(t.demand_std_monthly / SQRT(30.44), 2))
    AS expected_ss_demand,

  CASE t.abc_vol WHEN 'A' THEN 2.054 WHEN 'B' THEN 1.645 ELSE 1.282 END
    * (t.demand_mean_monthly / 30.44) * t.lt_std_days
    AS expected_ss_lt,

  CASE t.abc_vol WHEN 'A' THEN 2.054 WHEN 'B' THEN 1.645 ELSE 1.282 END
    * SQRT(
        t.lt_mean_days * POWER(t.demand_std_monthly / SQRT(30.44), 2)
        + POWER(t.demand_mean_monthly / 30.44, 2) * POWER(t.lt_std_days, 2)
      )
    AS expected_ss_combined,

  (t.demand_mean_monthly / 30.44) * t.lt_mean_days
    + CASE t.abc_vol WHEN 'A' THEN 2.054 WHEN 'B' THEN 1.645 ELSE 1.282 END
      * SQRT(
          t.lt_mean_days * POWER(t.demand_std_monthly / SQRT(30.44), 2)
          + POWER(t.demand_mean_monthly / 30.44, 2) * POWER(t.lt_std_days, 2)
        )
    AS expected_rop

FROM fact_safety_stock_targets t
WHERE t.item_id = '<ITEM>' AND t.loc = '<LOC>';

-- ASSERT for each SKU:
-- ABS(ss_demand_only - expected_ss_demand) < 1.0
-- ABS(ss_lt_only - expected_ss_lt) < 1.0
-- ABS(ss_combined - expected_ss_combined) < 1.0
-- ABS(reorder_point - expected_rop) < 1.0

-- ASSERT: Guard rails respected
SELECT item_id, loc, ss_combined,
  demand_mean_monthly / 30.44 * 3 AS min_ss,    -- 3 days supply
  demand_mean_monthly / 30.44 * 120 AS max_ss   -- 120 days supply
FROM fact_safety_stock_targets
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: ss_combined BETWEEN min_ss AND max_ss (or ss_combined=0 if zero demand)
```

---

## Step 9: EOQ (Assert formulas for same 5 SKUs)

```sql
-- ═══════════════════════════════════════════════
-- ASSERTION BLOCK: EOQ Formula Verification
-- EOQ = sqrt(2 * D * S / (H * C))
-- Defaults: S=50, H=0.25, C=1.00, MOQ=1
-- Cap: 6 months of demand
-- ═══════════════════════════════════════════════

SELECT
  e.item_id, e.loc,
  e.annual_demand,
  e.ordering_cost,
  e.holding_cost_pct,
  e.unit_cost,
  e.moq,
  e.eoq,
  e.effective_eoq,

  -- Manual: EOQ = sqrt(2 * D * S / (H * C))
  SQRT(2.0 * e.annual_demand * e.ordering_cost
       / NULLIF(e.holding_cost_pct * e.unit_cost, 0)) AS expected_eoq,

  -- effective = max(eoq, moq) capped at 6 months supply
  LEAST(
    GREATEST(
      SQRT(2.0 * e.annual_demand * e.ordering_cost
           / NULLIF(e.holding_cost_pct * e.unit_cost, 0)),
      e.moq
    ),
    e.demand_mean_monthly * 6  -- 6-month cap
  ) AS expected_effective_eoq,

  -- Costs
  e.annual_holding_cost,
  e.annual_order_cost,
  e.total_annual_cost,
  e.holding_cost_pct * e.unit_cost * (e.effective_eoq / 2.0) AS expected_holding,
  e.ordering_cost * (e.annual_demand / NULLIF(e.effective_eoq, 0)) AS expected_ordering

FROM fact_eoq_targets e
WHERE e.item_id = '<ITEM>' AND e.loc = '<LOC>';

-- ASSERT:
-- ABS(eoq - expected_eoq) < 1.0
-- effective_eoq >= moq
-- effective_eoq <= demand_mean_monthly * 6
-- ABS(total_annual_cost - (expected_holding + expected_ordering)) < 1.0
```

---

## Step 10: Replenishment Policies (Assert policy assignments for 5 SKUs)

```sql
-- ASSERT: Policy assignments follow ABC + variability rules
SELECT
  d.item_id, d.loc, d.abc_vol, d.variability_class,
  a.policy_id, p.policy_type, p.segment, p.service_level
FROM dim_sku d
JOIN fact_dfu_policy_assignment a ON d.item_id = a.item_id AND d.loc = a.loc
JOIN dim_replenishment_policy p ON a.policy_id = p.policy_id
WHERE d.item_id = '<ITEM>' AND d.loc = '<LOC>';

-- ASSERT per SKU:
-- IF abc_vol = 'A' AND variability_class != 'lumpy': policy_type = 'continuous_rop'
-- IF abc_vol = 'B' AND variability_class != 'lumpy': policy_type = 'periodic_review'
-- IF abc_vol = 'C' AND variability_class != 'lumpy': policy_type = 'min_max'
-- IF variability_class = 'lumpy': policy_type = 'manual' (overrides ABC)
```

---

## Step 11: Production Forecast & CI (Assert for 5 SKUs)

### 11.1 Run
```bash
make setup-demand-planning
```

### 11.2 Assert for each SKU
```sql
-- ASSERT: Production forecast exists with 18-month horizon
SELECT item_id, loc, COUNT(*) AS months,
  MIN(forecast_month) AS first_month, MAX(forecast_month) AS last_month
FROM fact_production_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_production_forecast)
GROUP BY 1,2;
-- ASSERT: months = 18
-- ASSERT: first_month = planning_date + 1 month

-- ASSERT: CI bands valid (lower < point < upper, all >= 0)
SELECT COUNT(*) AS bad_ci FROM fact_production_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_production_forecast)
  AND (forecast_qty < 0
    OR forecast_qty_lower > forecast_qty
    OR forecast_qty_upper < forecast_qty
    OR forecast_qty_lower < 0);
-- ASSERT: bad_ci = 0

-- ASSERT: CI width grows with horizon (sqrt scaling)
SELECT horizon_months,
  forecast_qty_upper - forecast_qty_lower AS ci_width
FROM fact_production_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_production_forecast)
ORDER BY horizon_months;
-- ASSERT: ci_width at horizon=4 / ci_width at horizon=1 ≈ sqrt(4)/sqrt(1) = 2.0 (±30%)

-- ASSERT: Uses champion model
SELECT DISTINCT model_id FROM fact_production_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_production_forecast);
-- ASSERT: model_id = source_model_id of champion for this SKU
```

---

## Step 12: Inventory Projection (Assert for 3 SKUs)

```sql
-- ASSERT: 3 scenarios present, 90-day horizon
SELECT scenario, COUNT(DISTINCT projection_date) AS days,
  MIN(projection_date) AS start_dt, MAX(projection_date) AS end_dt
FROM fact_inventory_projection
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
GROUP BY 1;
-- ASSERT: 3 rows (no_order, with_open_po, with_planned_orders)
-- ASSERT: days ≈ 90 for each

-- ASSERT: no_order scenario has monotonically decreasing qty
SELECT COUNT(*) AS non_decreasing FROM (
  SELECT projection_date, projected_qty,
    LAG(projected_qty) OVER (ORDER BY projection_date) AS prev_qty
  FROM fact_inventory_projection
  WHERE item_id = '<ITEM>' AND loc = '<LOC>' AND scenario = 'no_order'
) sub WHERE projected_qty > prev_qty;
-- ASSERT: non_decreasing = 0 (no receipts → only consumption)

-- ASSERT: with_open_po >= no_order at every date
SELECT COUNT(*) AS violations FROM (
  SELECT n.projection_date, n.projected_qty AS no_order_qty, p.projected_qty AS po_qty
  FROM fact_inventory_projection n
  JOIN fact_inventory_projection p
    ON n.item_id = p.item_id AND n.loc = p.loc AND n.projection_date = p.projection_date
  WHERE n.scenario = 'no_order' AND p.scenario = 'with_open_po'
    AND n.item_id = '<ITEM>' AND n.loc = '<LOC>'
) sub WHERE po_qty < no_order_qty;
-- ASSERT: violations = 0
```

---

## Step 13: Planned Orders (Assert for 3 SKUs)

```sql
-- ASSERT: Planned orders triggered when qty <= ROP
SELECT o.item_id, o.loc, o.trigger_date, o.trigger_reason,
  o.recommended_qty, o.moq, o.reorder_point, o.safety_stock,
  o.current_qty_on_hand, o.confidence_score, o.status
FROM fact_planned_orders o
WHERE o.item_id = '<ITEM>' AND o.loc = '<LOC>'
ORDER BY o.trigger_date;
-- ASSERT: status = 'proposed'
-- ASSERT: recommended_qty >= moq (MOQ floor)
-- ASSERT: recommended_qty % moq = 0 (rounded to MOQ multiple)
-- ASSERT: confidence_score BETWEEN 0.5 AND 1.0
-- ASSERT: expected_receipt_date = order_by_date + lead_time_days

-- ASSERT: Max 3 orders per SKU
SELECT item_id, loc, COUNT(*) AS n FROM fact_planned_orders
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
GROUP BY 1,2;
-- ASSERT: n <= 3
```

---

## Step 14: Replenishment Plan (Assert for 3 SKUs)

```sql
-- ASSERT: 12-month forward plan exists
SELECT COUNT(*) AS months FROM fact_replenishment_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_replenishment_plan);
-- ASSERT: months = 12

-- ASSERT: Policy dispatch matches assignment
SELECT rp.plan_month, rp.policy_type,
  rp.reorder_point, rp.order_qty, rp.order_up_to_level, rp.is_jit,
  pa.policy_id
FROM fact_replenishment_plan rp
JOIN fact_dfu_policy_assignment pa ON rp.item_id = pa.item_id AND rp.loc = pa.loc
JOIN dim_replenishment_policy p ON pa.policy_id = p.policy_id
WHERE rp.item_id = '<ITEM>' AND rp.loc = '<LOC>'
  AND rp.plan_version = (SELECT MAX(plan_version) FROM fact_replenishment_plan)
LIMIT 3;
-- ASSERT: rp.policy_type = p.policy_type
-- IF continuous_rop: reorder_point > 0 AND order_qty > 0
-- IF periodic_review: order_up_to_level > 0
-- IF min_max: reorder_point > 0 AND order_up_to_level > 0
-- IF manual: is_jit = TRUE

-- ASSERT: SS from CI spread formula
SELECT plan_month, sigma_method,
  forecast_qty_upper, forecast_qty_lower,
  sigma_demand_monthly,
  -- Manual: sigma = (upper - lower) / (2 * 1.282)
  (forecast_qty_upper - forecast_qty_lower) / (2 * 1.282) AS expected_sigma
FROM fact_replenishment_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND sigma_method = 'ci_spread'
  AND plan_version = (SELECT MAX(plan_version) FROM fact_replenishment_plan)
LIMIT 3;
-- ASSERT: ABS(sigma_demand_monthly - expected_sigma) < 1.0
```

---

## Step 15: Exceptions & Storyboard (Assert for 3 SKUs)

```sql
-- ASSERT: Exception types are valid
SELECT DISTINCT exception_type FROM fact_replenishment_exceptions;
-- ASSERT: All values in ('stockout','below_ss','below_rop','excess','zero_velocity','intramonth')

-- ASSERT: Severity is valid
SELECT DISTINCT severity FROM fact_replenishment_exceptions;
-- ASSERT: All values in ('critical','high','medium','low')

-- For a specific below_ss exception:
SELECT e.item_id, e.loc, e.exception_type, e.severity,
  e.current_qty_on_hand, e.ss_combined, e.reorder_point,
  e.recommended_order_qty
FROM fact_replenishment_exceptions e
WHERE e.item_id = '<ITEM>' AND e.loc = '<LOC>'
  AND e.exception_type = 'below_ss';
-- ASSERT: current_qty_on_hand < ss_combined (that's WHY it's below_ss)
-- ASSERT: recommended_order_qty > 0

-- ASSERT: Storyboard exception queue has valid entries
SELECT exception_type, COUNT(*), AVG(severity) AS avg_severity
FROM exception_queue
WHERE status = 'open'
GROUP BY 1;
-- ASSERT: exception_type IN ('forecast_bias','stockout_risk','accuracy_drop','excess_risk','model_drift','new_item')
```

---

## Step 16: Fill Rate & Service Level (Assert for 3 SKUs)

```sql
-- ASSERT: Fill rate computed correctly
SELECT item_id, loc, month_start,
  total_ordered, total_shipped,
  fill_rate,
  total_shipped::numeric / NULLIF(total_ordered, 0) AS expected_fill_rate,
  shortage_qty,
  GREATEST(total_ordered - total_shipped, 0) AS expected_shortage
FROM mv_fill_rate_monthly
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY month_start DESC LIMIT 3;
-- ASSERT: ABS(fill_rate - expected_fill_rate) < 0.001
-- ASSERT: ABS(shortage_qty - expected_shortage) < 0.01
-- ASSERT: fill_rate BETWEEN 0 AND 1
```

---

## Step 17: Bias Corrections (Assert for 3 SKUs)

```sql
-- ASSERT: Correction factor within guard rails [0.70, 1.30]
SELECT item_id, loc, plan_month,
  rolling_bias_3m, correction_factor_raw, correction_factor,
  correction_was_clipped, flagged_for_review
FROM fact_bias_corrections
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY plan_month DESC LIMIT 3;
-- ASSERT: correction_factor BETWEEN 0.70 AND 1.30
-- ASSERT: IF correction_was_clipped THEN correction_factor IN (0.70, 1.30)
-- ASSERT: IF ABS(correction_factor - 1.0) > 0.20 THEN flagged_for_review = TRUE

-- ASSERT: Raw correction formula
-- raw = 1 / (1 + rolling_bias_3m)
SELECT item_id, loc, plan_month,
  rolling_bias_3m,
  correction_factor_raw,
  1.0 / (1.0 + rolling_bias_3m) AS expected_raw
FROM fact_bias_corrections
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND rolling_bias_3m IS NOT NULL
LIMIT 3;
-- ASSERT: ABS(correction_factor_raw - expected_raw) < 0.001
```

---

## Step 18: S&OP Cycle (Assert stage machine)

```sql
-- ASSERT: Current cycle exists
SELECT cycle_id, cycle_month, status FROM fact_sop_cycles
ORDER BY cycle_month DESC LIMIT 1;
-- ASSERT: status IN ('demand_review','supply_review','pre_sop','executive_sop','approved','closed')

-- ASSERT: Stage progression is valid (no skipping)
-- approved_at should be NULL if status hasn't reached 'approved'
SELECT cycle_id, cycle_month, status,
  demand_review_at, supply_review_at, pre_sop_at, executive_sop_at, approved_at
FROM fact_sop_cycles
ORDER BY cycle_month DESC LIMIT 1;
-- ASSERT: IF status='demand_review' THEN supply_review_at IS NULL
-- ASSERT: IF status='approved' THEN all timestamp columns NOT NULL
```

---

## Step 19: Control Tower KPIs (Assert aggregation)

```bash
# ASSERT: Control tower returns all 5 sections
curl -s "http://localhost:8000/control-tower/kpis" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for key in ['health','exceptions','fill_rate','demand_signals','intramonth']:
    assert key in d, f'Missing section: {key}'
assert d['health']['total_dfus'] > 0, 'No DFUs in health'
assert 0 <= d['fill_rate'].get('portfolio_fill_rate_3m', 0) <= 1, 'Invalid fill rate'
print(f'PASS: Control tower KPIs valid. {d[\"health\"][\"total_dfus\"]} DFUs tracked.')
"
```

---

## Step 20: Full Test Suite

```bash
# Backend: 3042 tests
~/.local/bin/uv run pytest tests/ -q

# Frontend: 874 tests
cd frontend && PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot
```

---

## Gap Ticket Template

If ANY assertion above fails, create a ticket in this format:

```markdown
## TICKET: [STEP]-[ASSERTION_ID]
**Step:** [Step number and name]
**SKU:** item_id=X, loc=Y
**Assertion:** [Which assert failed]
**Expected:** [What the formula/rule says]
**Actual:** [What the DB returned]
**Root Cause:** [Code file:line or SQL file causing the issue]
**Fix:** [Concrete change needed]
```

---

## Step 21: Demand Variability (Assert for 5 SKUs)

### 21.1 Run
```bash
~/.local/bin/uv run python scripts/compute_demand_variability.py
```

### 21.2 Pick 5 SKUs across variability classes
```sql
SELECT item_id, loc, demand_cv, demand_mad, demand_mean, demand_std,
       intermittency_ratio, variability_class
FROM dim_sku
WHERE demand_cv IS NOT NULL AND variability_class IS NOT NULL
ORDER BY demand_cv DESC
LIMIT 5;
```

### 21.3 Assert: Formula & Classification
```sql
-- ASSERT: CV = std / mean (24-month window)
SELECT item_id, loc, demand_cv, demand_std, demand_mean,
  demand_std / NULLIF(demand_mean, 0) AS expected_cv
FROM dim_sku
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: ABS(demand_cv - expected_cv) < 0.001

-- ASSERT: Variability class follows CV thresholds
-- smooth: CV < 0.30, erratic: 0.30 <= CV < 0.80, lumpy: CV >= 0.80 AND intermittency > 0.30, intermittent: intermittency > 0.30
SELECT item_id, loc, demand_cv, intermittency_ratio, variability_class,
  CASE
    WHEN intermittency_ratio > 0.30 AND demand_cv >= 0.80 THEN 'lumpy'
    WHEN intermittency_ratio > 0.30 THEN 'intermittent'
    WHEN demand_cv < 0.30 THEN 'smooth'
    WHEN demand_cv < 0.80 THEN 'erratic'
    ELSE 'lumpy'
  END AS expected_class
FROM dim_sku
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: variability_class = expected_class

-- ASSERT: MAD = mean(|x - mean|)
-- ASSERT: All stats columns NOT NULL for SKUs with >= 3 months history
SELECT COUNT(*) AS missing_stats FROM dim_sku
WHERE demand_mean IS NOT NULL AND demand_cv IS NULL;
-- ASSERT: missing_stats = 0

-- ASSERT: Skewness and kurtosis populated
SELECT item_id, loc, demand_skewness, demand_kurtosis
FROM dim_sku WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: Both NOT NULL
```

---

## Step 22: Lead Time Variability (Assert for 3 SKUs)

### 22.1 Run
```bash
~/.local/bin/uv run python scripts/compute_lead_time_variability.py
```

### 22.2 Pick 3 SKUs with lead time profiles
```sql
SELECT item_id, loc, lead_time_mean_days, lead_time_std_days,
       lead_time_cv, lead_time_class, observation_count
FROM dim_item_lead_time_profile
WHERE observation_count >= 3
ORDER BY lead_time_cv DESC
LIMIT 3;
```

### 22.3 Assert
```sql
-- ASSERT: CV = std / mean
SELECT item_id, loc, lead_time_cv, lead_time_std_days, lead_time_mean_days,
  lead_time_std_days / NULLIF(lead_time_mean_days, 0) AS expected_cv
FROM dim_item_lead_time_profile
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: ABS(lead_time_cv - expected_cv) < 0.001

-- ASSERT: Lead time class follows thresholds (stable < 0.15, moderate < 0.40, volatile >= 0.40)
SELECT item_id, loc, lead_time_cv, lead_time_class,
  CASE
    WHEN lead_time_cv < 0.15 THEN 'stable'
    WHEN lead_time_cv < 0.40 THEN 'moderate'
    ELSE 'volatile'
  END AS expected_class
FROM dim_item_lead_time_profile
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: lead_time_class = expected_class

-- ASSERT: Minimum observations enforced
SELECT COUNT(*) AS bad_obs FROM dim_item_lead_time_profile
WHERE observation_count < 3;
-- ASSERT: bad_obs = 0 (or these have NULL classification)
```

---

## Step 23: Blended Forecast (Assert for 3 SKUs)

### 23.1 Run
```bash
~/.local/bin/uv run python scripts/compute_blended_forecast.py
```

### 23.2 Pick 3 SKUs with blended data
```sql
SELECT item_id, loc, blend_month,
       statistical_qty, sensing_qty, blended_qty,
       alpha_weight, outlier_capped
FROM fact_blended_forecast
ORDER BY blend_month DESC
LIMIT 3;
```

### 23.3 Assert
```sql
-- ASSERT: Blended = alpha * sensing + (1-alpha) * statistical
SELECT item_id, loc, blend_month,
  statistical_qty, sensing_qty, blended_qty, alpha_weight,
  alpha_weight * sensing_qty + (1 - alpha_weight) * statistical_qty AS expected_blended
FROM fact_blended_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY blend_month DESC LIMIT 3;
-- ASSERT: ABS(blended_qty - expected_blended) < 0.01

-- ASSERT: Alpha decays over 4-week horizon (week 1 = highest alpha)
SELECT item_id, loc, blend_month, sensing_week, alpha_weight
FROM fact_blended_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY blend_month, sensing_week;
-- ASSERT: alpha_weight monotonically decreasing across weeks

-- ASSERT: Outlier capping at 3σ
SELECT COUNT(*) AS uncapped_outliers FROM fact_blended_forecast
WHERE outlier_capped = TRUE
  AND ABS(sensing_qty - statistical_qty) < 3 * COALESCE(sensing_std, 1);
-- ASSERT: uncapped_outliers = 0 (only flagged when truly > 3σ)

-- ASSERT: blended_qty >= 0 (no negative forecasts)
SELECT COUNT(*) AS negatives FROM fact_blended_forecast WHERE blended_qty < 0;
-- ASSERT: negatives = 0
```

---

## Step 24: Consensus Plan (Assert for 3 SKUs)

### 24.1 Run
```bash
~/.local/bin/uv run python scripts/generate_consensus_plan.py
```

### 24.2 Pick 3 SKUs
```sql
SELECT item_id, loc, plan_month, statistical_qty, override_qty,
       consensus_qty, override_type, override_source
FROM fact_consensus_plan
WHERE override_qty IS NOT NULL
ORDER BY plan_month DESC LIMIT 3;
```

### 24.3 Assert
```sql
-- ASSERT: Hard override replaces entirely
SELECT item_id, loc, plan_month, consensus_qty, override_qty, override_type
FROM fact_consensus_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>' AND override_type = 'hard';
-- ASSERT: consensus_qty = override_qty

-- ASSERT: Soft override applies multiplier within bounds [0.10, 5.00]
SELECT item_id, loc, plan_month, statistical_qty, override_multiplier, consensus_qty,
  statistical_qty * override_multiplier AS expected_consensus
FROM fact_consensus_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>' AND override_type = 'soft';
-- ASSERT: ABS(consensus_qty - expected_consensus) < 0.01
-- ASSERT: override_multiplier BETWEEN 0.10 AND 5.00

-- ASSERT: No override → consensus = statistical
SELECT COUNT(*) AS bad FROM fact_consensus_plan
WHERE override_qty IS NULL AND override_type IS NULL
  AND ABS(consensus_qty - statistical_qty) > 0.01;
-- ASSERT: bad = 0

-- ASSERT: Type-priority conflict resolution (hard > soft > none)
SELECT item_id, loc, plan_month, override_type, override_source
FROM fact_consensus_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY plan_month;
-- ASSERT: If multiple overrides exist for same DFU+month, 'hard' wins
```

---

## Step 25: Demand Signals (Assert for 3 SKUs)

### 25.1 Run
```bash
~/.local/bin/uv run python scripts/compute_demand_signals.py
```

### 25.2 Pick 3 SKUs
```sql
SELECT item_id, loc, signal_date, mtd_actual, mtd_expected,
       signal_status, alert_level, pct_deviation
FROM fact_demand_signals
ORDER BY signal_date DESC LIMIT 3;
```

### 25.3 Assert
```sql
-- ASSERT: pct_deviation = (mtd_actual - mtd_expected) / NULLIF(mtd_expected, 0)
SELECT item_id, loc, signal_date,
  mtd_actual, mtd_expected, pct_deviation,
  (mtd_actual - mtd_expected) / NULLIF(mtd_expected, 0) AS expected_deviation
FROM fact_demand_signals
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY signal_date DESC LIMIT 3;
-- ASSERT: ABS(pct_deviation - expected_deviation) < 0.001

-- ASSERT: Signal status mapping (above/below/on_plan)
SELECT item_id, loc, pct_deviation, signal_status,
  CASE
    WHEN pct_deviation > 0.10 THEN 'above_plan'
    WHEN pct_deviation < -0.10 THEN 'below_plan'
    ELSE 'on_plan'
  END AS expected_status
FROM fact_demand_signals
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: signal_status = expected_status

-- ASSERT: Alert level mapping (urgent if |dev| > 25%, watch if > 10%)
SELECT item_id, loc, pct_deviation, alert_level,
  CASE
    WHEN ABS(pct_deviation) > 0.25 THEN 'urgent'
    WHEN ABS(pct_deviation) > 0.10 THEN 'watch'
    ELSE 'none'
  END AS expected_alert
FROM fact_demand_signals
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: alert_level = expected_alert
```

---

## Step 26: Echelon Planning (Assert for 3 SKUs)

### 26.1 Run
```bash
~/.local/bin/uv run python scripts/compute_echelon_targets.py
```

### 26.2 Pick 3 multi-location items
```sql
SELECT item_id, echelon_level, COUNT(DISTINCT loc) AS locations,
       pooled_demand_std, cascade_risk_score
FROM fact_echelon_targets
WHERE echelon_level = 'central'
GROUP BY 1,2,4,5
HAVING COUNT(DISTINCT loc) >= 2
LIMIT 3;
```

### 26.3 Assert
```sql
-- ASSERT: Pooled σ = sqrt(Σ σ²) across child locations
SELECT e.item_id, e.echelon_level, e.pooled_demand_std,
  SQRT(SUM(POWER(d.demand_std, 2))) AS expected_pooled_std
FROM fact_echelon_targets e
JOIN dim_sku d ON e.item_id = d.item_id
WHERE e.item_id = '<ITEM>' AND e.echelon_level = 'central'
GROUP BY e.item_id, e.echelon_level, e.pooled_demand_std;
-- ASSERT: ABS(pooled_demand_std - expected_pooled_std) < 1.0

-- ASSERT: Echelon SS < sum of individual SS (risk pooling benefit)
SELECT e.item_id, e.echelon_ss,
  SUM(ss.ss_combined) AS sum_individual_ss
FROM fact_echelon_targets e
JOIN fact_safety_stock_targets ss ON e.item_id = ss.item_id
WHERE e.item_id = '<ITEM>' AND e.echelon_level = 'central'
GROUP BY e.item_id, e.echelon_ss;
-- ASSERT: echelon_ss <= sum_individual_ss (risk pooling reduces total SS)

-- ASSERT: Cascade risk score between 0 and 1
SELECT COUNT(*) AS bad_scores FROM fact_echelon_targets
WHERE cascade_risk_score < 0 OR cascade_risk_score > 1;
-- ASSERT: bad_scores = 0
```

---

## Step 27: Rebalancing (Assert for 3 SKU-pairs)

### 27.1 Run
```bash
~/.local/bin/uv run python scripts/compute_rebalancing.py
```

### 27.2 Pick 3 transfer recommendations
```sql
SELECT item_id, from_loc, to_loc, transfer_qty,
       excess_at_source, deficit_at_dest,
       transfer_cost, stockout_cost_avoided, net_benefit
FROM fact_rebalancing_plan
WHERE net_benefit > 0
ORDER BY net_benefit DESC LIMIT 3;
```

### 27.3 Assert
```sql
-- ASSERT: Net benefit = stockout_avoided + carrying_saved - transfer_cost
SELECT item_id, from_loc, to_loc,
  net_benefit,
  stockout_cost_avoided + carrying_cost_saved - transfer_cost AS expected_benefit
FROM fact_rebalancing_plan
WHERE item_id = '<ITEM>' AND from_loc = '<FROM>' AND to_loc = '<TO>';
-- ASSERT: ABS(net_benefit - expected_benefit) < 0.01

-- ASSERT: Transfer qty <= excess at source
SELECT item_id, transfer_qty, excess_at_source
FROM fact_rebalancing_plan
WHERE item_id = '<ITEM>' AND from_loc = '<FROM>';
-- ASSERT: transfer_qty <= excess_at_source

-- ASSERT: Transfer qty <= deficit at destination
SELECT item_id, transfer_qty, deficit_at_dest
FROM fact_rebalancing_plan
WHERE item_id = '<ITEM>' AND to_loc = '<TO>';
-- ASSERT: transfer_qty <= deficit_at_dest

-- ASSERT: Only profitable transfers recommended
SELECT COUNT(*) AS unprofitable FROM fact_rebalancing_plan
WHERE net_benefit <= 0;
-- ASSERT: unprofitable = 0
```

---

## Step 28: Investment Plan (Assert for 3 SKUs)

### 28.1 Run
```bash
~/.local/bin/uv run python scripts/compute_investment_plan.py
```

### 28.2 Pick 3 SKUs
```sql
SELECT item_id, loc, ss_investment_value,
       current_service_level, target_service_level,
       marginal_roi, incremental_csl, incremental_investment
FROM fact_investment_plan
ORDER BY marginal_roi DESC LIMIT 3;
```

### 28.3 Assert
```sql
-- ASSERT: SS investment = ss_combined * unit_cost
SELECT ip.item_id, ip.loc, ip.ss_investment_value,
  ss.ss_combined * COALESCE(e.unit_cost, 1.0) AS expected_investment
FROM fact_investment_plan ip
JOIN fact_safety_stock_targets ss ON ip.item_id = ss.item_id AND ip.loc = ss.loc
JOIN fact_eoq_targets e ON ip.item_id = e.item_id AND ip.loc = e.loc
WHERE ip.item_id = '<ITEM>' AND ip.loc = '<LOC>';
-- ASSERT: ABS(ss_investment_value - expected_investment) < 1.0

-- ASSERT: Marginal ROI = CSL increment / investment increment
SELECT item_id, loc, marginal_roi, incremental_csl, incremental_investment,
  incremental_csl / NULLIF(incremental_investment, 0) AS expected_roi
FROM fact_investment_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: ABS(marginal_roi - expected_roi) < 0.0001

-- ASSERT: Service level between 0 and 1
SELECT COUNT(*) AS bad_sl FROM fact_investment_plan
WHERE current_service_level < 0 OR current_service_level > 1
   OR target_service_level < 0 OR target_service_level > 1;
-- ASSERT: bad_sl = 0
```

---

## Step 29: Service Level Actuals (Assert for 3 SKUs)

### 29.1 Run
```bash
~/.local/bin/uv run python scripts/compute_service_level_actuals.py
```

### 29.2 Pick 3 SKUs
```sql
SELECT item_id, loc, month_start, actual_service_level,
       target_service_level, gap, consecutive_misses
FROM fact_service_level_tracking
ORDER BY month_start DESC LIMIT 3;
```

### 29.3 Assert
```sql
-- ASSERT: Gap = actual - target
SELECT item_id, loc, month_start,
  actual_service_level, target_service_level, gap,
  actual_service_level - target_service_level AS expected_gap
FROM fact_service_level_tracking
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY month_start DESC LIMIT 3;
-- ASSERT: ABS(gap - expected_gap) < 0.001

-- ASSERT: Consecutive misses increments when gap < 0
SELECT item_id, loc, month_start, gap, consecutive_misses,
  LAG(consecutive_misses) OVER (PARTITION BY item_id, loc ORDER BY month_start) AS prev_misses
FROM fact_service_level_tracking
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY month_start;
-- ASSERT: IF gap < 0 THEN consecutive_misses = prev_misses + 1
-- ASSERT: IF gap >= 0 THEN consecutive_misses = 0

-- ASSERT: Service level between 0 and 1
SELECT COUNT(*) AS bad FROM fact_service_level_tracking
WHERE actual_service_level < 0 OR actual_service_level > 1;
-- ASSERT: bad = 0
```

---

## Step 30: Quantile Forecasts (Assert for 3 SKUs)

### 30.1 Run
```bash
~/.local/bin/uv run python scripts/generate_quantile_forecasts.py
```

### 30.2 Pick 3 SKUs
```sql
SELECT item_id, loc, forecast_month, p10, p50, p90,
       sigma_forecast
FROM fact_quantile_forecast
ORDER BY forecast_month DESC LIMIT 3;
```

### 30.3 Assert
```sql
-- ASSERT: P10 < P50 < P90 (monotonic quantiles)
SELECT COUNT(*) AS bad_order FROM fact_quantile_forecast
WHERE p10 > p50 OR p50 > p90;
-- ASSERT: bad_order = 0

-- ASSERT: σ_forecast = (P90 - P10) / 2.5632
SELECT item_id, loc, forecast_month,
  p10, p90, sigma_forecast,
  (p90 - p10) / 2.5632 AS expected_sigma
FROM fact_quantile_forecast
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY forecast_month DESC LIMIT 3;
-- ASSERT: ABS(sigma_forecast - expected_sigma) < 0.1

-- ASSERT: All quantiles >= 0
SELECT COUNT(*) AS negatives FROM fact_quantile_forecast
WHERE p10 < 0 OR p50 < 0 OR p90 < 0;
-- ASSERT: negatives = 0

-- ASSERT: P50 ≈ point forecast (champion)
SELECT q.item_id, q.loc, q.forecast_month, q.p50,
  pf.forecast_qty AS champion_qty,
  ABS(q.p50 - pf.forecast_qty) / NULLIF(pf.forecast_qty, 0) AS pct_diff
FROM fact_quantile_forecast q
JOIN fact_production_forecast pf ON q.item_id = pf.item_id AND q.loc = pf.loc
  AND q.forecast_month = pf.forecast_month
WHERE q.item_id = '<ITEM>' AND q.loc = '<LOC>'
LIMIT 3;
-- ASSERT: pct_diff < 0.15 (within 15% of champion point forecast)
```

---

## Step 31: Financial Plan (Assert for 3 SKUs)

### 31.1 Run
```bash
~/.local/bin/uv run python scripts/compute_financial_plan.py
```

### 31.2 Pick 3 SKUs
```sql
SELECT item_id, loc, plan_month,
       inventory_value, carrying_cost_monthly,
       excess_value, days_of_supply
FROM fact_financial_plan
ORDER BY plan_month DESC LIMIT 3;
```

### 31.3 Assert
```sql
-- ASSERT: Carrying cost = value × 0.25 / 12 (annual rate / 12)
SELECT item_id, loc, plan_month,
  inventory_value, carrying_cost_monthly,
  inventory_value * 0.25 / 12.0 AS expected_carrying
FROM fact_financial_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY plan_month DESC LIMIT 3;
-- ASSERT: ABS(carrying_cost_monthly - expected_carrying) < 0.01

-- ASSERT: Excess = on_hand beyond 180-day DOS threshold
SELECT item_id, loc, plan_month,
  days_of_supply, excess_value,
  CASE WHEN days_of_supply > 180 THEN inventory_value * (1 - 180.0/NULLIF(days_of_supply,0))
       ELSE 0 END AS expected_excess
FROM fact_financial_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
ORDER BY plan_month DESC LIMIT 3;
-- ASSERT: ABS(excess_value - expected_excess) < 1.0

-- ASSERT: DOS = on_hand / daily_demand
SELECT item_id, loc, days_of_supply, inventory_qty, daily_demand_rate,
  inventory_qty / NULLIF(daily_demand_rate, 0) AS expected_dos
FROM fact_financial_plan
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
LIMIT 3;
-- ASSERT: ABS(days_of_supply - expected_dos) < 0.1
```

---

## Step 32: Event Adjustments (Assert for 3 events)

### 32.1 Run
```bash
~/.local/bin/uv run python scripts/apply_event_adjustments.py
```

### 32.2 Pick 3 events
```sql
SELECT event_id, item_id, loc, event_month,
       base_qty, multiplier, additive, adjusted_qty,
       override_type
FROM fact_event_adjustments
WHERE multiplier != 1.0 OR additive != 0
ORDER BY event_month DESC LIMIT 3;
```

### 32.3 Assert
```sql
-- ASSERT: adjusted_qty = base_qty × multiplier + additive (for non-hard overrides)
SELECT event_id, item_id, loc,
  base_qty, multiplier, additive, adjusted_qty, override_type,
  base_qty * multiplier + additive AS expected_adjusted
FROM fact_event_adjustments
WHERE item_id = '<ITEM>' AND loc = '<LOC>' AND override_type != 'hard';
-- ASSERT: ABS(adjusted_qty - expected_adjusted) < 0.01

-- ASSERT: Hard override replaces entirely
SELECT event_id, adjusted_qty, override_qty
FROM fact_event_adjustments
WHERE override_type = 'hard' AND item_id = '<ITEM>';
-- ASSERT: adjusted_qty = override_qty

-- ASSERT: Multiplier within bounds [0, 5.0]
SELECT COUNT(*) AS bad_mult FROM fact_event_adjustments
WHERE multiplier < 0 OR multiplier > 5.0;
-- ASSERT: bad_mult = 0

-- ASSERT: adjusted_qty >= 0
SELECT COUNT(*) AS negatives FROM fact_event_adjustments
WHERE adjusted_qty < 0;
-- ASSERT: negatives = 0
```

---

## Step 33: Storyboard Exception Detection (Assert types & severity)

### 33.1 Run
```bash
~/.local/bin/uv run python scripts/generate_storyboard_exceptions.py
```

### 33.2 Assert: All 6 exception types
```sql
-- ASSERT: Valid exception types
SELECT DISTINCT exception_type FROM exception_queue;
-- ASSERT: All in ('forecast_bias','stockout_risk','accuracy_drop','excess_risk','model_drift','new_item')

-- ASSERT: Severity formula = 0.4×financial + 0.4×rule + 0.2×urgency
SELECT eq.exception_id, eq.exception_type, eq.severity,
  eq.financial_impact_score, eq.rule_score, eq.urgency_score,
  0.4 * eq.financial_impact_score + 0.4 * eq.rule_score + 0.2 * eq.urgency_score AS expected_severity
FROM exception_queue eq
WHERE eq.status = 'open'
LIMIT 5;
-- ASSERT: ABS(severity - expected_severity) < 0.01

-- ASSERT: Dedup window = 7 days (no duplicates within 7 days for same DFU+type)
SELECT item_id, loc, exception_type, COUNT(*) AS n,
  MIN(created_at), MAX(created_at),
  MAX(created_at) - MIN(created_at) AS span
FROM exception_queue
WHERE status = 'open'
GROUP BY 1,2,3
HAVING COUNT(*) > 1
  AND MAX(created_at) - MIN(created_at) < INTERVAL '7 days';
-- ASSERT: 0 rows (no duplicates within 7-day window)

-- ASSERT: supporting_data is valid JSONB
SELECT COUNT(*) AS bad_json FROM exception_queue
WHERE supporting_data IS NULL OR supporting_data::text = '{}';
-- ASSERT: bad_json = 0 (every exception has supporting evidence)
```

---

## Step 34: Supply Chain Scenarios (Assert for 2 scenarios)

### 34.1 Run
```bash
~/.local/bin/uv run python scripts/run_supply_chain_scenario.py --scenario supplier_delay
```

### 34.2 Pick scenarios
```sql
SELECT scenario_id, scenario_type, item_id, loc,
       impact_start, impact_end, delay_days, capacity_reduction_pct,
       projected_stockout_date, revenue_at_risk
FROM fact_supply_scenarios
ORDER BY created_at DESC LIMIT 2;
```

### 34.3 Assert
```sql
-- ASSERT: Valid scenario types
SELECT DISTINCT scenario_type FROM fact_supply_scenarios;
-- ASSERT: All in ('supplier_delay','capacity_constraint','demand_shock','transport_disruption','quality_hold')

-- ASSERT: Impact dates valid
SELECT COUNT(*) AS bad_dates FROM fact_supply_scenarios
WHERE impact_start > impact_end OR impact_start IS NULL;
-- ASSERT: bad_dates = 0

-- ASSERT: Revenue at risk >= 0
SELECT COUNT(*) AS bad_rev FROM fact_supply_scenarios
WHERE revenue_at_risk < 0;
-- ASSERT: bad_rev = 0

-- ASSERT: Projected stockout date is within impact window (if exists)
SELECT scenario_id, impact_start, impact_end, projected_stockout_date
FROM fact_supply_scenarios
WHERE projected_stockout_date IS NOT NULL
  AND (projected_stockout_date < impact_start OR projected_stockout_date > impact_end + INTERVAL '90 days');
-- ASSERT: 0 rows
```

---

## Step 35: Lead Time Learning (Assert for 3 SKUs)

### 35.1 Run
```bash
~/.local/bin/uv run python scripts/update_lead_time_actuals.py
```

### 35.2 Pick 3 SKUs with PO receipts
```sql
SELECT lt.item_id, lt.loc, lt.lead_time_mean_days, lt.lead_time_std_days,
       lt.observation_count, lt.last_updated
FROM dim_item_lead_time_profile lt
WHERE lt.observation_count >= 5
ORDER BY lt.last_updated DESC LIMIT 3;
```

### 35.3 Assert
```sql
-- ASSERT: Lead time mean matches PO receipt data
SELECT lt.item_id, lt.loc, lt.lead_time_mean_days,
  AVG(r.receipt_date - r.order_date) AS expected_mean_days
FROM dim_item_lead_time_profile lt
JOIN fact_po_receipts r ON lt.item_id = r.item_id AND lt.loc = r.loc
WHERE lt.item_id = '<ITEM>' AND lt.loc = '<LOC>'
GROUP BY lt.item_id, lt.loc, lt.lead_time_mean_days;
-- ASSERT: ABS(lead_time_mean_days - expected_mean_days) < 1.0

-- ASSERT: Observation count matches PO receipt count
SELECT lt.observation_count,
  COUNT(*) AS actual_receipts
FROM dim_item_lead_time_profile lt
JOIN fact_po_receipts r ON lt.item_id = r.item_id AND lt.loc = r.loc
WHERE lt.item_id = '<ITEM>' AND lt.loc = '<LOC>'
GROUP BY lt.observation_count;
-- ASSERT: observation_count = actual_receipts (within 12-month window)

-- ASSERT: Degradation alert if CV increased
SELECT item_id, loc, lead_time_cv,
  LAG(lead_time_cv) OVER (PARTITION BY item_id, loc ORDER BY last_updated) AS prev_cv
FROM dim_item_lead_time_profile
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- RECORD: Check if degradation was flagged when CV increased
```

---

## Step 36: Open POs & Procurement Workflow (Assert for 3 POs)

### 36.1 Run
```bash
~/.local/bin/uv run python scripts/load_open_pos.py
~/.local/bin/uv run python scripts/release_planned_orders.py
```

### 36.2 Pick 3 POs
```sql
SELECT po_id, item_id, loc, supplier_id, po_qty,
       order_date, expected_delivery_date, status
FROM fact_open_purchase_orders
ORDER BY order_date DESC LIMIT 3;
```

### 36.3 Assert
```sql
-- ASSERT: PO status is valid
SELECT DISTINCT status FROM fact_open_purchase_orders;
-- ASSERT: All in ('open','partially_received','closed','cancelled')

-- ASSERT: Expected delivery >= order date
SELECT COUNT(*) AS bad_dates FROM fact_open_purchase_orders
WHERE expected_delivery_date < order_date;
-- ASSERT: bad_dates = 0

-- ASSERT: PO qty > 0
SELECT COUNT(*) AS bad_qty FROM fact_open_purchase_orders
WHERE po_qty <= 0;
-- ASSERT: bad_qty = 0

-- ASSERT: Procurement workflow stages are valid
SELECT DISTINCT workflow_status FROM fact_procurement_workflow;
-- ASSERT: All in ('proposed','planner_approved','buyer_released','po_sent')

-- ASSERT: Stage transitions are sequential (no skipping)
SELECT pw.order_id, pw.workflow_status,
  pw.proposed_at, pw.planner_approved_at, pw.buyer_released_at, pw.po_sent_at
FROM fact_procurement_workflow pw
LIMIT 3;
-- ASSERT: IF workflow_status='planner_approved' THEN proposed_at IS NOT NULL
-- ASSERT: IF workflow_status='po_sent' THEN all prior timestamps NOT NULL
```

---

## Step 37: Seasonality Detection (Assert for 5 SKUs)

### 37.1 Run
```bash
~/.local/bin/uv run python scripts/detect_seasonality.py
~/.local/bin/uv run python scripts/update_seasonality_profiles.py
```

### 37.2 Pick 5 SKUs across seasonality profiles
```sql
SELECT item_id, loc, seasonality_profile, seasonality_cv,
       yoy_correlation, acf_lag12
FROM dim_sku
WHERE seasonality_profile IS NOT NULL
ORDER BY seasonality_cv DESC LIMIT 5;
```

### 37.3 Assert
```sql
-- ASSERT: Profile follows CV thresholds (from config: low=0.15, medium=0.35, high=0.70)
SELECT item_id, loc, seasonality_cv, seasonality_profile,
  CASE
    WHEN seasonality_cv >= 0.70 THEN 'high'
    WHEN seasonality_cv >= 0.35 THEN 'medium'
    WHEN seasonality_cv >= 0.15 THEN 'low'
    ELSE 'none'
  END AS expected_profile_raw
FROM dim_sku
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- NOTE: Final profile requires confirmation gate (YoY corr >= 0.40 OR ACF lag-12 >= 0.30)

-- ASSERT: Confirmation gate applied
SELECT item_id, loc, seasonality_cv, seasonality_profile,
  yoy_correlation, acf_lag12,
  CASE
    WHEN seasonality_cv < 0.15 THEN 'none'
    WHEN yoy_correlation >= 0.40 OR acf_lag12 >= 0.30 THEN
      CASE WHEN seasonality_cv >= 0.70 THEN 'high'
           WHEN seasonality_cv >= 0.35 THEN 'medium'
           ELSE 'low' END
    ELSE 'none'  -- Failed confirmation → none
  END AS expected_profile
FROM dim_sku
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: seasonality_profile = expected_profile

-- ASSERT: Valid profile values only
SELECT DISTINCT seasonality_profile FROM dim_sku WHERE seasonality_profile IS NOT NULL;
-- ASSERT: All in ('high','medium','low','none','insufficient_history')
```

---

## Step 38: ABC-XYZ Classification (Assert for 5 SKUs)

### 38.1 Run
```bash
~/.local/bin/uv run python scripts/classify_abc_xyz.py
```

### 38.2 Pick 5 SKUs across the 9-cell matrix
```sql
SELECT item_id, loc, abc_vol, xyz_variability,
       demand_mean, demand_cv, revenue_rank_pct
FROM dim_sku
WHERE abc_vol IS NOT NULL AND xyz_variability IS NOT NULL
ORDER BY demand_mean DESC LIMIT 5;
```

### 38.3 Assert
```sql
-- ASSERT: ABC follows cumulative revenue %
-- A = top 80%, B = next 15%, C = bottom 5%
SELECT abc_vol, COUNT(*) AS n,
  MIN(revenue_rank_pct), MAX(revenue_rank_pct)
FROM dim_sku WHERE abc_vol IS NOT NULL
GROUP BY 1 ORDER BY 1;
-- ASSERT: A max_pct <= 0.80, B min_pct > 0.80 AND max_pct <= 0.95, C min_pct > 0.95

-- ASSERT: XYZ follows CV thresholds
-- X: CV < 0.50, Y: 0.50 <= CV < 1.00, Z: CV >= 1.00
SELECT item_id, loc, demand_cv, xyz_variability,
  CASE
    WHEN demand_cv < 0.50 THEN 'X'
    WHEN demand_cv < 1.00 THEN 'Y'
    ELSE 'Z'
  END AS expected_xyz
FROM dim_sku
WHERE item_id = '<ITEM>' AND loc = '<LOC>';
-- ASSERT: xyz_variability = expected_xyz

-- ASSERT: All 9 cells represented (AX, AY, AZ, BX, BY, BZ, CX, CY, CZ)
SELECT abc_vol || xyz_variability AS cell, COUNT(*)
FROM dim_sku
WHERE abc_vol IS NOT NULL AND xyz_variability IS NOT NULL
GROUP BY 1 ORDER BY 1;
-- ASSERT: 9 rows (all cells populated, or document missing cells)

-- ASSERT: No invalid values
SELECT COUNT(*) AS bad FROM dim_sku
WHERE (abc_vol NOT IN ('A','B','C') AND abc_vol IS NOT NULL)
   OR (xyz_variability NOT IN ('X','Y','Z') AND xyz_variability IS NOT NULL);
-- ASSERT: bad = 0
```

---

## Step 39: Safety Stock Simulation (Assert Monte Carlo for 3 SKUs)

### 39.1 Run
```bash
~/.local/bin/uv run python scripts/run_ss_simulation.py
```

### 39.2 Pick 3 SKUs
```sql
SELECT item_id, loc, simulation_id,
       ss_level, simulated_service_level, stockout_probability,
       avg_inventory, total_cost, iterations
FROM fact_ss_simulation_results
ORDER BY simulation_id DESC LIMIT 3;
```

### 39.3 Assert
```sql
-- ASSERT: 10K iterations per simulation
SELECT simulation_id, iterations FROM fact_ss_simulation_results
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
LIMIT 1;
-- ASSERT: iterations = 10000

-- ASSERT: Service level monotonically increases with SS level
SELECT ss_level, simulated_service_level
FROM fact_ss_simulation_results
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
  AND simulation_id = (SELECT MAX(simulation_id) FROM fact_ss_simulation_results
                       WHERE item_id = '<ITEM>' AND loc = '<LOC>')
ORDER BY ss_level;
-- ASSERT: simulated_service_level is monotonically non-decreasing

-- ASSERT: Stockout probability = 1 - service_level
SELECT ss_level, simulated_service_level, stockout_probability,
  1.0 - simulated_service_level AS expected_stockout_prob
FROM fact_ss_simulation_results
WHERE item_id = '<ITEM>' AND loc = '<LOC>'
LIMIT 3;
-- ASSERT: ABS(stockout_probability - expected_stockout_prob) < 0.001

-- ASSERT: Service level between 0 and 1
SELECT COUNT(*) AS bad FROM fact_ss_simulation_results
WHERE simulated_service_level < 0 OR simulated_service_level > 1;
-- ASSERT: bad = 0
```

---

## Step 40: Data Quality Engine (Assert DQ checks & fixes)

### 40.1 Run
```bash
~/.local/bin/uv run python scripts/populate_dq_checks.py
~/.local/bin/uv run python scripts/fix_dq_issues.py
```

### 40.2 Assert
```sql
-- ASSERT: DQ checks populated for all domains
SELECT domain, COUNT(*) AS checks, SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END) AS passed
FROM fact_dq_checks
GROUP BY 1 ORDER BY 1;
-- ASSERT: Each domain has at least 1 check

-- ASSERT: Valid check statuses
SELECT DISTINCT status FROM fact_dq_checks;
-- ASSERT: All in ('pass','fail','warn','skip')

-- ASSERT: Failed checks have fix recommendations
SELECT check_id, domain, check_name, status, fix_recommendation
FROM fact_dq_checks WHERE status = 'fail' LIMIT 5;
-- ASSERT: fix_recommendation IS NOT NULL for all failed checks

-- ASSERT: Fixes applied correctly (original preserved in fact_sales_original)
SELECT COUNT(*) AS original_rows FROM fact_sales_original;
SELECT COUNT(*) AS current_rows FROM fact_sales_monthly;
-- ASSERT: original_rows >= current_rows (fixes may remove bad rows)

-- ASSERT: No orphan DFUs (dim_sku without sales)
SELECT COUNT(*) AS orphans FROM dim_sku d
LEFT JOIN fact_sales_monthly s ON d.item_id = s.item_id AND d.loc = s.loc
WHERE s.item_id IS NULL AND d.demand_mean > 0;
-- ASSERT: orphans = 0 (or document count)
```

---

## Summary: Assertion Count per Step

| Step | What | # SKUs | # Assertions |
|------|------|--------|-------------|
| 1 | Data Load | - | 7 |
| 2 | Explorer API | 1 | 3 |
| 3 | Clustering | 5 | 6 |
| 4 | LGBM Backtest | 10 | 80 (8×10) |
| 5 | CatBoost/XGB | 10 | 80 (8×10) |
| 6 | Champion | 10 | 30 (3×10) |
| 7 | Accuracy API | - | 3 |
| 8 | Safety Stock | 5 | 20 (4×5) |
| 9 | EOQ | 5 | 20 (4×5) |
| 10 | Policies | 5 | 5 |
| 11 | Prod Forecast | 5 | 20 (4×5) |
| 12 | Projection | 3 | 9 (3×3) |
| 13 | Planned Orders | 3 | 15 (5×3) |
| 14 | Repl Plan | 3 | 12 (4×3) |
| 15 | Exceptions | 3 | 6 |
| 16 | Fill Rate | 3 | 9 (3×3) |
| 17 | Bias Corr | 3 | 6 (2×3) |
| 18 | S&OP | 1 | 3 |
| 19 | Control Tower | - | 3 |
| 20 | Test Suite | - | 3014 |
| 21 | Demand Variability | 5 | 20 (4×5) |
| 22 | Lead Time Variability | 3 | 9 (3×3) |
| 23 | Blended Forecast | 3 | 12 (4×3) |
| 24 | Consensus Plan | 3 | 12 (4×3) |
| 25 | Demand Signals | 3 | 9 (3×3) |
| 26 | Echelon Planning | 3 | 9 (3×3) |
| 27 | Rebalancing | 3 | 12 (4×3) |
| 28 | Investment Plan | 3 | 9 (3×3) |
| 29 | Service Level | 3 | 9 (3×3) |
| 30 | Quantile Forecasts | 3 | 12 (4×3) |
| 31 | Financial Plan | 3 | 9 (3×3) |
| 32 | Event Adjustments | 3 | 12 (4×3) |
| 33 | Storyboard Exceptions | - | 4 |
| 34 | Supply Scenarios | 2 | 8 (4×2) |
| 35 | Lead Time Learning | 3 | 9 (3×3) |
| 36 | Open POs & Procurement | 3 | 15 (5×3) |
| 37 | Seasonality | 5 | 15 (3×5) |
| 38 | ABC-XYZ | 5 | 20 (4×5) |
| 39 | SS Simulation | 3 | 12 (4×3) |
| 40 | DQ Engine | - | 5 |
| **TOTAL** | | **~145 SKUs** | **~3642 assertions** |
