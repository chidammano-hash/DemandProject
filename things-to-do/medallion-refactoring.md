# Medallion Pipeline — Refactoring Opportunities

## 1. Critical Bugs (fix before refactoring)

| # | Issue | File | Lines |
|---|-------|------|-------|
| **B1** | Column mismatch: query uses `created_at` but table has `quarantined_at` | `api/routers/medallion.py` | 202, 219 |
| **B2** | Error handling returns tuple `{"error": ...}, 404` instead of raising `HTTPException` | `api/routers/medallion.py` | 72, 239 |

---

## 2. Code Duplication (High Impact)

| # | Issue | File | Lines |
|---|-------|------|-------|
| **D1** | 5 helper functions duplicated identically between `common/medallion.py` and `load_dataset_postgres.py`: `_elapsed()`, `qident()`, `typed_expr()`, `business_key_expr()`, `NULL_SQL` | Both files | ~50 lines each |
| **D2** | Quarantine workflow repeated 3x (fetch bad rows -> INSERT quarantine -> UPDATE status) | `common/medallion.py` | 317-342, 359-385, 518-548 |
| **D3** | Imputation logic nearly identical for numeric (median) and categorical (mode) paths | `common/medallion.py` | 570-639 |
| **D4** | Percentile computation (`percentile_cont(...)`) repeated in 4 places | `common/medallion.py` | 573, 608, 655-656, 716 |
| **D5** | MV refresh list hardcoded in multiple scripts | `scripts/load_dataset_postgres.py` | 620-626 |
| **D6** | `complete_batch()` and `fail_batch()` nearly identical — differ only in SET clause | `common/medallion.py` | 94-113 |

**Recommendation**: Extract shared helpers to `common/sql_helpers.py`. Extract `_quarantine_rows()`, `_impute_with_audit()`, `_get_percentiles()` helpers. Consolidate batch status into single `_update_batch_status()`.

---

## 3. Functions Too Long

| # | Function | File | Lines | Size |
|---|----------|------|-------|------|
| **L1** | `main()` | `scripts/load_dataset_postgres.py` | 337-642 | **305 lines** |
| **L2** | `run_silver_dq_gate()` | `common/medallion.py` | 291-406 | **115 lines** |
| **L3** | `promote_to_silver()` | `common/medallion.py` | 179-284 | **106 lines** |
| **L4** | `_fix_completeness_with_audit()` | `common/medallion.py` | 553-641 | **89 lines** |

**Recommendation**: Extract `_run_legacy_load()` from `main()`. Break DQ gate into `_check_completeness()` and `_check_range()`. Split completeness fix into `_impute_numeric()` and `_impute_categorical()`.

---

## 4. Magic Strings / Numbers

| # | Value | File | Lines | Suggested Constant |
|---|-------|------|-------|--------------------|
| **M1** | `1.5` (IQR multiplier) | `common/medallion.py` | 669-670 | `IQR_OUTLIER_MULTIPLIER` |
| **M2** | `730` (lead time max days) | `common/medallion.py` | 718, 728 | `LEAD_TIME_MAX_DAYS` |
| **M3** | `7` (global median fallback) | `common/medallion.py` | 722 | `LEAD_TIME_DEFAULT_DAYS` |
| **M4** | `1024 * 1024` (hash chunk size) | `common/medallion.py` | 120 | `HASH_CHUNK_SIZE` |
| **M5** | `'external'` (model_id literal, 4 occurrences) | `scripts/load_dataset_postgres.py` | 182, 541, 550 | `EXTERNAL_MODEL_ID` |
| **M6** | `'512MB'` / `'1GB'` (pg session settings) | `scripts/load_dataset_postgres.py` | 487-489 | Config YAML |
| **M7** | `0.5` / `0.25` / `0.75` percentiles | `common/medallion.py` | 573, 608, 655-656 | `PERCENTILE_*` constants |

---

## 5. Structural Issues

| # | Issue | Impact |
|---|-------|--------|
| **S1** | Two completely separate code paths in `load_dataset_postgres.py` (medallion vs legacy) sharing zero functions | High maintenance burden |
| **S2** | Forecast-specific logic scattered via `is_forecast` checks in 5+ places | Should use strategy pattern or separate function |
| **S3** | Index drop/recreate lifecycle spread across 4 functions (~50 lines) | Should be an `IndexLifecycleManager` class |
| **S4** | Progress reporting boilerplate repeated for every phase (~8 instances of `print + t0 + elapsed`) | Should be a context manager |
| **S5** | SQL building interleaved with business logic in `promote_to_silver()` and `run_silver_dq_gate()` | Separate SQL builders from executors |

---

## 6. DDL Gaps

| # | Issue | File |
|---|-------|------|
| **DDL1** | Missing UNIQUE constraints on all 8 silver table `*_ck` fields — dedup relies solely on Python code | `sql/082_create_silver_tables.sql` |
| **DDL2** | Datetime column naming inconsistency: `quarantined_at` (quarantine) vs `created_at` (lineage) | 083 vs 085 SQL files |

---

## 7. Error Handling Gaps

| # | Issue | File | Lines |
|---|-------|------|-------|
| **E1** | Legacy load has no try/except/fail_batch — partial failures leave DB in indeterminate state | `scripts/load_dataset_postgres.py` | 483-629 |
| **E2** | `file_hash()` has no error handling for IO/permission errors | `common/medallion.py` | 116-125 |
| **E3** | CSV COPY has no try/except to report which row/column failed | `common/medallion.py` | 156-159 |
| **E4** | `typed_expr()` silently returns untyped column if field not in spec | `common/medallion.py` | 45-63 |
| **E5** | `_resolve_forecast_execution_lag()` silently defaults unmatched DFUs to lag 0 with no warning threshold | `scripts/load_dataset_postgres.py` | 126-170 |

---

## 8. SQL Safety

| # | Issue | File | Lines |
|---|-------|------|-------|
| **SQL1** | `col_min`/`col_max` from config interpolated directly into WHERE clauses without parameterization | `common/medallion.py` | 509, 511 |
| **SQL2** | `batch_id` hardcoded via f-string in SQL instead of `%s` param | `common/medallion.py` | 234, 264 |

---

## 9. Test Coverage Gaps (37+ missing tests)

| Category | Count | Key Gaps |
|----------|-------|----------|
| Untested internal functions | 6 | `_fix_range_with_audit`, `_fix_completeness_with_audit`, `_fix_outliers_with_audit`, `_fix_lead_time_with_audit`, `_elapsed`, `_config` |
| Untested edge cases | 18+ | Empty bronze table, 0% pass rate, all-quarantined batch, IQR=0, empty file hash, config missing |
| Missing error paths | 8+ | Config validation, sales original table missing, gate failure then gold called |
| Missing integration tests | 5 | Full bronze->silver->gold flow, sales dual-track, multi-check gate, fix cascade, batch failure recovery |

---

## 10. Recommended File Structure (post-refactor)

```
common/
  sql_helpers.py      # NEW: _elapsed, qident, typed_expr, business_key_expr, NULL_SQL
  medallion.py        # SIMPLIFIED: uses sql_helpers, shorter functions
scripts/
  load_dataset_postgres.py  # SLIMMED: dispatch only (~200 lines)
```

---

**Total: 2 bugs, 6 duplication areas, 4 oversized functions, 7 magic values, 5 structural issues, 2 DDL gaps, 5 error handling gaps, 2 SQL safety items, 37+ test gaps.**
