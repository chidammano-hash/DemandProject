## common/ml + engines + ai — Refactor Opportunities

_Scope: `common/ml/` (+ subpackages), `common/engines/`, `common/ai/`. Read-only audit — no code changed._

### Quick wins
- Dead: `dq_engine.py:25-26` `_reset_config()` defined, never called → delete.
- Dead: `comparison.py:591` `missing_keys` assigned, never used → delete.
- ~~Dead/orphaned: `exception_engine.py:225-517` four `detect_*` pure functions are never called~~ **CORRECTION (verified):** the four `detect_*` functions ARE the unit-tested public API (~50 call sites in `test_exception_engine.py` + `test_exception_financial_impact.py`) — NOT dead. Only *production* `run_exception_detection` reimplements them inline. The real fix is rewiring orchestration to call the pure detectors and deleting the INLINE copies (see #1) — do NOT delete the pure functions. (`generate_headline` at 566-601 IS used — keep.)
- Magic number `0.74` rule-score cap repeated at `exception_engine.py:278,421,485,690,774,892,965` → one config key.
- Hardcoded `margin_assumption = 0.30` at `exception_engine.py:652,742,825` → config key.
- `dl_models.py:91` `params.get("learning_rate", 0.001)` — only Python-side ML default left; move to `forecast_pipeline_config.yaml`.
- Duplicated `_tools_to_openai()` at `ai_planner.py:530-542` and `tuning_advisor.py:382-394` (identical) → move to `llm_client.py`.
- `ai_planner.py:996` / `tuning_advisor.py:678` `json.loads(json.dumps(block.input))` redundant double-encode → drop.

### Ranked opportunities

1. **exception_engine.py — detection rule logic duplicated pure-vs-inline (4×)**
   - Files: `common/engines/exception_engine.py`
   - Problem: Each detector exists twice — pure function and re-inlined in `run_exception_detection`: forecast_bias (`225-314` vs `646-729`), stockout_risk (`317-385` vs `734-815`), accuracy_drop (`388-456` vs `820-927`), excess_risk (`459-517` vs `932-1000`). Same scoring math in two places; the pure versions are dead.
   - Proposed change: Orchestration calls the pure detectors; delete inline copies. Split into `detection_rules.py`, `financial_impact.py` (`136-219`), `exception_scoring.py`, `exception_orchestration.py`.
   - Impact: High (kills ~320 dup lines) · Effort: M · Risk: M (live path — needs test parity)

2. **ai_planner.py / tuning_advisor.py — duplicated agentic loop + raw client instantiation**
   - Files: `common/ai/ai_planner.py`, `tuning_advisor.py`, `llm_client.py`
   - Problem: OpenAI loops (`ai_planner.py:853-943` vs `tuning_advisor.py:524-620`) and Anthropic loops (`ai_planner.py:946-1032` vs `tuning_advisor.py:622-707`) are ~80% identical. Both instantiate `OpenAI()`/`anthropic.Anthropic()` directly (`ai_planner.py:809-814`, `tuning_advisor.py:478-485`), bypassing `llm_client.py`'s retry/timeout wrapper that `fva_recommender.py:218` uses correctly.
   - Proposed change: Extract `AgenticLoopRunner` in `llm_client.py` with `run_openai_loop`/`run_anthropic_loop(messages, tools, dispatch_fn, budget)`; both agents supply only tool dispatch. Route client creation through `LLMClient`.
   - Impact: High (~200 dup lines, centralized retry) · Effort: H · Risk: M

3. **ai_planner.py — split 1149-line god-module**
   - Problem: Tool data-access (`77-410`), audit/insert (`413-523`), tool schema + 65-line `_SYSTEM_PROMPT` (`545-786`), and the agent class (`793-1149`) all in one file; 13 repeated `with pool.connection()` blocks.
   - Proposed change: `ai_planner_tools.py`, `ai_planner_schema.py`, `ai_planner_agent.py`. Move `MAX_TURNS=40`/`TOKEN_BUDGET=100_000` (`36-37`) + model/temp defaults (`805-807`) to config.
   - Impact: High · Effort: M · Risk: Low

4. **tree_models.py — direct estimator instantiation bypasses model_registry**
   - Files: `common/ml/expert_panel/tree_models.py`, `model_registry.py`
   - Problem: `_MODEL_LIB` (`30-32`) + `importlib.import_module` + `getattr` (`197-198`) construct `LGBMRegressor`/`CatBoostRegressor`/`XGBRegressor` directly, then `_extract_model_params` (`39-71`) hardcodes per-model defaults (`54-69`). Violates "all tree instantiation goes through `model_registry.build_model`" + "no Python-side hyperparameter defaults". (It does call `fit_model` at `280`.)
   - Proposed change: Replace importlib build with `model_registry.build_model(model_id, params)`; move `setdefault` defaults into `forecast_pipeline_config.yaml`. Delete `_MODEL_LIB`.
   - Impact: High · Effort: M · Risk: M (confirm `build_model` exposes best-iteration handling tree_models relies on)

5. **backtest_framework.py — split 1776-line module + extract giant orchestrator**
   - Problem: `run_tree_backtest` (`1128-1777`) is 649 lines / 12 params with a 231-line SHAP-retrain block (`1352-1583`) duplicating safety-WAPE logic between per-cluster (`1372-1487`) and global (`1520-1556`) paths, plus an 87-line recursive loop (`1586-1673`).
   - Proposed change: Split into `cluster_profiles.py` (`49-214`), `data_loading.py` (`265-397`), `postprocessing.py` (`639-745`), `metrics.py` (`947-1064`), `backtest_runner.py`. Extract `_apply_shap_selection_and_retrain`, `_run_recursive_inference`, `_prepare_train_predict_split`; group the 12 params into a config dataclass.
   - Impact: High · Effort: H · Risk: M (core forecast path — extract behind unchanged tests)

6. **backtest_framework.py — repeated forecast_ck / WAPE-with-actuals helpers**
   - Problem: `forecast_ck` build duplicated (`479-485`, `548-554`, `735-741`); actuals-join + WAPE inline 6× (`988-990`, `1049-1056`, `1534-1543`, `1547-1556`); date formatting 4×; DFU-key fallback (`sku_ck`/`dfu_ck`) 3×.
   - Proposed change: Extract `build_forecast_ck()`, `compute_wape_with_actuals()`, `format_dates()`, `get_dfu_key()`.
   - Impact: Med-High · Effort: L · Risk: Low

7. **dq_engine.py — SQL identifier interpolation + unused dispatch registry**
   - Problem: Table/column names f-string-interpolated into SQL (`38-40`, `54`, `62-89`, `242-243`, `339-369`) — violates the `psycopg.sql.Identifier` rule. `CHECK_FUNCTIONS` registry (`473-486`) exists but `_run_single` (`722-796`) dispatches via 13 if/elif branches instead.
   - Proposed change: Wrap identifiers with `psycopg.sql.Identifier`. Make `_run_single` and `_flatten_checks` (`513-720`) data-driven off `CHECK_FUNCTIONS` + a param-mapping table.
   - Impact: High (security rule + simplification) · Effort: M · Risk: M

8. **foundation_models.py — split 1216-line file; foundation loaders straddle the rule**
   - Files: `common/ml/expert_panel/foundation_models.py`, `foundation_backtest.py`
   - Problem: 8 model runners in one file with inline pipeline instantiation (`118-120`, `389`, `548`). Rule says foundation loaders live ONLY in `foundation_backtest.py`. `_run_chronos2_enriched` (`530-759`) is 230 lines / 6-deep nesting.
   - Proposed change: One module per family under `foundation_models/`; keep loader/cache logic in/near `foundation_backtest.py`; lazy-import via `_FOUNDATION_DISPATCH`.
   - Impact: Med-High · Effort: H · Risk: M (optional deps + device handling)

9. **foundation_models.py — copy-pasted forecast post-processing across runners**
   - Problem: NaN/Inf sanitize (`206-208`, `325-327`, `461`, `737`), non-negative clip (`212`, `330`, `459`, `735`, `873`), repeat/tile result-frame build (`217-222`, `332-337`, `463-468`, `739-744`) duplicated in every runner. ~23 scattered hardcoded `params.get(default)`.
   - Proposed change: `_sanitize_forecasts()`, `_clip_nonnegative()`, `_build_forecast_frame(...)`; move per-model defaults into YAML.
   - Impact: Med · Effort: L-M · Risk: Low

10. **comparison.py — split + collapse repeated metric/lift/baseline blocks**
    - Problem: `compare_all` (`694-878`) does 7× `_restrict()`, 6× accuracy calls, 5× near-identical lift blocks (`810-839`) ignoring the existing `_safe_diff()` (`958`). `compute_portfolio_predictions` (`522-687`) has 6-deep fallback nesting. Per-segment metrics implemented 3 ways.
    - Proposed change: Split `metrics.py`/`baselines.py`/`portfolio.py`; drive baselines off a `BASELINE_REGISTRY`; route all lifts through `_safe_diff`; extract fallback-priority helpers.
    - Impact: Med · Effort: M · Risk: Low

11. **champion/ — blend-result dict construction duplicated across 5 strategy files**
    - Files: `common/ml/champion/{blend,routing,basic,bandit,meta}.py`
    - Problem: The `{"basefcst_pref": blended, "tothist_dmd": actual, ...}` row build is repeated ~40× (e.g. `blend.py:161,328,409,525,693,803,899,976,1067`); the literal `"basefcst_pref"` is used instead of `FORECAST_QTY_COL`. `helpers.py` blend primitives aren't reused everywhere.
    - Proposed change: A single `make_blend_row(blended, actual, **meta)` in `champion/helpers.py` keyed off `FORECAST_QTY_COL`.
    - Impact: Med · Effort: L-M · Risk: Low

12. **exception_engine.py — hardcoded thresholds & repeated config coercion**
    - Problem: `_SEVERITY_BANDS_DEFAULT`/`_RESPONSE_HOURS_DEFAULT` (`33-40`) + per-detector fallback constants hardcoded; `float(cfg.get(key, default))` repeated 90+×; late `load_config` under a bare except (`1060-1063`).
    - Proposed change: Move thresholds to `config/.../exception_engine.yaml`; build a typed config object once; narrow the except.
    - Impact: Med · Effort: M · Risk: Low

13. **fva_recommender / ai schema — duplicated Recommendation vs CreateInsightInput, weak validators**
    - Problem: Two overlapping insight schemas (`fva_recommender.py:38-53` strict; `ai_planner.py:44-70` loose); `summary_must_contain_metrics` only checks for one digit; `create_insight` (`449-523`) swallows validation via bare except (`490`) and returns `-1`.
    - Proposed change: Factor a shared base schema; tighten the validator; let `create_insight` raise/return typed result.
    - Impact: Med · Effort: L · Risk: Low (callers check return — confirm the `-1` contract first)

14. **dl_models.py — last Python-side ML hyperparameter default** (`learning_rate` at `:91`) → move to YAML. · Impact: Low-Med · Effort: Low · Risk: Low

**Note:** No bare `pd.read_sql` found in scope; no leftover clustering/sku_features shims — those refactors appear cleanly completed. The `"basefcst_pref"` literal debt concentrates in `champion/blend.py` (#11).
