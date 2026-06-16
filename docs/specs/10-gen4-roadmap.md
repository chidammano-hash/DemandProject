# Gen-4 Roadmap — Multi-Agent Design Review Synthesis

> Consolidated output of a 31-agent parallel review conducted 2026-04-23. Agents: 10 supply-chain experts, 2 coders (backend / frontend+ML), 9 UI/UX experts (tables/grids review did not complete), 10 AI-centric Gen-4 architects.

**Status:** Two foundation-delivery passes completed (2026-04-23 AM + PM) across 10 parallel subagent streams. All P0/P1 items either done or have a landed scaffold; remaining work is production-wiring of scaffolds + a handful of heavy items (RL, MILP, agent split, vision) flagged inline.
**Authors:** Multi-agent review, synthesized by Claude.
**Supersedes:** none. **Related:** all existing specs in `docs/specs/01..08/`.

## 2026-04-23 Implementation Snapshot

### Pass 1 (morning) — Foundations
All 10 cross-cutting priorities either done or scaffolded. All Section 1 P0 items (1.1–1.9) complete. Backend P0 complete, P1 largely complete. UI/UX P0 mostly complete. AI Phase 0 foundations complete; Phases 1–2 scaffolded. ~150 new tests.

Key modules: `common/core/service_levels.py` · `common/ai/{decision_ledger,rag,memory,causal}.py` · `common/feature_store.py` · `common/twin/state.py` · `common/engines/exception_engine.py`. Migrations `sql/137`–`sql/144`. Frontend toaster + dialog + formatApiError + dark-mode FOUC + empty-state triad. Router restructure + FastAPI `lifespan` + pool consolidation. Configs: `agent_autonomy.yaml`, `elasticity_config.yaml`, `safety_stock_config.yaml` (empirical-quantile), `forecast_pipeline_config.yaml` (`fm_spine`, `promote_gate`, `baseline_intermittent`, `embargo_months: 1`).

### Pass 2 (afternoon) — P1/P2 closures + Phase 1–4 scaffolds
All Section 1 P1 items delivered or scaffolded. Backend P2 cleanups complete. UI/UX P1 majority complete. AI Phase 1 FM-first scaffolds landed (FM→SS bridge, CRPS/pinball, cold-start k-NN); Phase 2 NL + retrieval (dry-run, SOP ingest); Phase 3 prescriptive (orchestrator, reversible actions, MILP stub, sensing model); Phase 4 governance (drift, explain API, fairness MV, shadow rollout, OpenLineage). ~175 new tests.

Key new modules: `common/ai/{orchestrator,reversible,lineage}.py` · `common/ml/{crps,shadow_rollout,fm_quantile_bridge,milp}.py` · `common/services/metrics.py` (canonical accuracy helper) · `common/engines/sop_decisions.py` · `api/routers/intelligence/explain.py` · `api/routers/inventory/working_capital.py` · `api/routers/platform/admin.py`. Migrations `sql/145`–`sql/167`. Frontend: Breadcrumbs, useUndoable, nav reorder, Okabe-Ito palette, CI band on forecast chart, optimistic updates on exception queue.

**Cumulative:** ~325 new tests passing, ~30 SQL migrations (137–167), ~40 new Python modules, ~15 frontend additions. Roadmap items remaining are predominantly heavy integrations (Gymnasium RL env, Claude Vision OSA, supplier document AI, tool-calling chat, AIPlannerAgent split, EconML DML, Feast swap, highspy MILP solve). All have documented scaffolds + TODO markers pointing at the next step.

Follow-ups and deferred items are flagged inline below.

---

## 2026-05-07 Synthesis — 1000 supply-chain-expert ideas, ranked

A second-wave review structured to produce raw enhancement volume, then sift it down. Pipeline:

- **10 supply-chain experts** (each a named persona with 15-25 years' experience across SAP IBP, Kinaxis Maestro, Blue Yonder Luminate, o9, OMP, ToolsGroup SO99+, Logility, RELEX, Anaplan, Manhattan Active, plus relevant research literature) generated **100 enhancements each = 1000 raw ideas**. Files: `/tmp/sc_expert_01..10_*.md`.
- **8 consolidator agents** synthesized along orthogonal lenses. **6 succeeded** (capability tiers, roadmap quadrants, competitive matrix, monetization, dependency DAG, by-role view); 2 stalled at watchdog while digesting 10 expert files. Files: `/tmp/sc_consolidation_*.md`.
- **9 judges** ranked under orthogonal criteria: business impact, effort/feasibility, novelty, RFP win-rate, technical feasibility, customer pain, strategic leverage, time-to-ROI, regulatory pressure. Files: `/tmp/sc_judge_*.md`.
- Final aggregation uses **Borda-style scoring** (rank-N → 11-N points) summed across 9 judges, with a consensus-width bonus.

### Top 30 Master Ranking

#### Tier A — Universal winners (4+ judges agreed in top-10)

| # | Enhancement | Borda | Effort | Why it wins |
|---|---|---|---|---|
| 1 | **Lot/serial/FEFO/DSCSA traceability + recall workflow** | 39 | XL | RFP-DQ for pharma/food/cannabis; FSMA Rule 204 enforcement Jan 20 2026; unlocks $3.2B TAM. Effort judge is the sole detractor — 4 other top-tier judges overwhelm. |
| 2 | **Lag-decomposed accuracy + pinball-loss leaderboard API** | 30 | S (3-day) | Zero-cost ship-now: `backtest_lag_archive` already populated. Three perfect 10/10s from feasibility/tech/ROI judges. |
| 3 | **Override audit trail + reason-code taxonomy + override-FVA** | 26 | S–M | Universal pain-killer (5+ planner roles); RFP audit-trail asks; SOX-relevant. Already partially scaffolded via `decision_ledger`. |
| 4 | **SLA countdowns + acknowledge/snooze on exceptions** | 25 | S | Exception engine already typed with timestamps; pure additive UX. |
| 5 | **"Why am I seeing this?" + recommended-action chip** | 24 | S | Exception engine records `rule_id`; trivial backend, large planner-trust uplift. |
| 6 | **Sales Order + ATP/CTP + EDI 850/855/856 + AS2** | 23 | XL | Moves us from "analytics" to "execution platform" → 3-5× ASP. Head-on Kinaxis/BY/o9 attack. RFP-DQ for retail/grocery EDI mandates. |
| 7 | **Multi-tenant `tenant_id` + RLS + per-tenant config** | 22 | XL | Force-multiplier for 3PL channel ($1.8B TAM); RFP-DQ for shared-services deals. |
| 8 | **Bulk action toolbar + inline diff before commit** | 20 | S | Trio of feasibility+pain+ROI judges agree. |
| 9 | **Plant/line capacity master + multi-level BOM + MPS + changeover** | 18 | XL | Without these, "S&OP supply review" is theatre. Unlocks discrete/process manufacturing (5× TAM). |
| 10 | **Forecastability score + cold-start successor mapping** | 18 | S | Inputs already in `dim_sku`; pure additive analytics. Drives smarter NPI/EOL handling. |

#### Tier B — Strong consensus (3 judges top-10)

| # | Enhancement | Borda | Effort | Notes |
|---|---|---|---|---|
| 11 | **Service-level back-test / drift report** | 17 | S | Fill-rate MV + tier MVs already in place. |
| 12 | **Decision Ledger Merkle anchoring + four-eyes** | 15 | M | Gates AI agents, autonomy tiers, EU AI Act, SOX 404. North-star primitive in dependency DAG. |
| 13 | **Lead-time CV exposure + change-point auto-recompute** | 14 | S | `mv_supplier_po_performance.lt_cv` already computed; expose via API. |
| 14 | **Probabilistic chain — conformal forecast → conformal SS → conformal replenishment** | 14 | M–L | Differentiator vs ToolsGroup (only conformal chain end-to-end is novel; quantile alone is parity). |
| 15 | **Tier-N supplier visibility + SPOF/HHI + Risk Module** | 13 | M–L | Standalone $200-500K module (Resilinc/Everstream comp). |
| 16 | **Bitemporal facts + AS-OF queries + SCD-2 dimensions** | 14 | XL | **TENSION**: leverage judge calls it #1 primitive (~25 downstream); business-impact judge demoted as "plumbing nobody pays for." Ship as enabler, not as feature. |
| 17 | **Stage-gate validators on S&OP cycle** | 11 | S | `_next_stage()` already exists; add `is_complete` checks. SOX-relevant. |
| 18 | **Multi-level customer hierarchy SCD-2 (sold-to/ship-to/bill-to/payer + GLN/DUNS)** | 11 | M | Customer hierarchy is flat strings today. Unlocks customer-tier SL, allocation, cost-to-serve. |
| 19 | **Sustainability Module — CSRD/CBAM/EUDR + carbon-aware planning** | 11 | L–XL | $150-400K standalone, mandatory FY2024 EU CSRD wave 1 in flight. |
| 20 | **Agentic AI Co-Pilot — policy-bounded autonomy + episodic memory feedback** | 13 | L | Only SCM segment growing >25%. Codebase has unusually advanced foundations (decision ledger + RAG + 3-tier memory) — gap is ergonomics. |

#### Tier C — Specialist wins (2 judges, distinctive value)

| # | Enhancement | Effort | Distinctive value |
|---|---|---|---|
| 21 | **Continuous IBP / kill the monthly cycle** | XL | Green-field positioning. **TENSION**: leverage judge calls it a *consumer*, not a primitive. |
| 22 | **Plant capacity + line routing + changeover modeling** | L | F&B-vertical unlock (subset of #9). |
| 23 | **MEIO Clark-Scarf / GST stochastic-DP** | XL | Headline competitive gap vs ToolsGroup. **TENSION**: pain judge demoted because planners never see MEIO math. |
| 24 | **Plan freeze + immutable plan versions + lineage hash** | M | Couples with bitemporal #16. |
| 25 | **SOC2 Type II + ISO 27001 + RBAC + immutable audit** | L | Enterprise-RFP gate. Combine with #12 (decision ledger). |
| 26 | **Override-FVA classifier (predict ex-ante override quality)** | M | Green-field per novelty judge — no major SCM software ships. |
| 27 | **Counterfactual digital-twin replay** | L | Green-field. Unlocks "what would have happened if we'd taken the AI's recommendation?" |
| 28 | **OFAC/EU/UK sanctions screening on suppliers + customers** | S–M | Quick-win regulatory check. |
| 29 | **EUDR DDS + polygon traceability** | L | EU EUDR in force Dec 2024; specific to commodity importers. |
| 30 | **Adversarial red-team agent against live S&OP** | M | Green-field; uses existing AI Planner + scenario engine. |

### Cross-judge tensions

Items where judges sharply disagree are roadmap-debate territory — surfaced rather than averaged away:

| Item | High-rankers | Demoters | Resolution suggestion |
|---|---|---|---|
| Lot/FEFO/DSCSA | Business, RFP, regulatory (#1 each) | Effort (1-2/10) | XL, but trifecta of revenue + RFP + regulatory overrides effort. Phase the build: dim_lot first, FEFO replenishment, then full DSCSA event capture. |
| Bitemporal facts | Leverage (#1) | Business ("plumbing, nobody pays") | Ship as enabler under audit/AI feature work, not as standalone product. |
| MEIO Clark-Scarf | Business #8, RFP #7 | Pain #30 | Real RFP-credibility need but invisible to user. Couple with planner-facing UI (SS justification panel) to convert tech win to perceived win. |
| Continuous IBP | Business #9, novelty #1 | Leverage ("consumer not primitive") | Don't sequence first; ship after bitemporal + decision ledger + drift mature. |
| "Bulk action / undo / Why-am-I-seeing-this" | Effort/pain/ROI (#2-5 across) | Leverage ("low; consume primitives, unblock nothing") | Build them anyway — cheap, planner-trust-building; the leverage objection only matters at portfolio level. |
| Probabilistic forecasting | All quantile-rank high | Novelty ("conformal chain is novel; quantile alone is parity") | Ship the chain end-to-end (forecast → SS → MEIO → ATP) as a coherent suite, not separate components. Differentiator only if integrated. |

### Recommended sequencing (4 waves)

**Wave 1 — 90 days (foundational + quick wins, parallel)**
- Quick wins (S effort, multi-judge winners): #2 lag-accuracy, #3 override audit, #4 SLA, #5 why-am-I-seeing, #8 bulk action, #10 forecastability, #11 SL drift, #13 LT-CV, #17 stage-gate, #28 sanctions
- Foundational primitives (M effort): **#12 Decision Ledger** (north-star primitive), **#16 SCD-2 dim conventions** (for bitemporal lift later), #18 customer hierarchy SCD-2

**Wave 2 — Q2/Q3 (regulatory-driven + table-stakes)**
- **#1 Lot/FEFO** (FSMA Jan 2026 deadline → start now)
- **#9 Plant/BOM/MPS** (long-pole — start in parallel with Wave 1 once SCD-2 stable)
- #19 Sustainability Module (CSRD already in flight)
- #15 Tier-N supplier + Risk Module
- #14 Probabilistic chain (forecasting + SS + replenishment integrated)

**Wave 3 — 12-18 months (RFP credibility)**
- **#6 Sales order + ATP/CTP + EDI** (the 3-5× ASP move)
- **#7 Multi-tenant + RLS** (3PL channel)
- #20 Agentic AI Co-Pilot (Wave 1's decision ledger + memory + feedback loops fed forward)
- #23 MEIO Clark-Scarf / GST
- #25 SOC2/ISO27001 audit prep

**Wave 4 — strategic R&D bets (12-24 months)**
- #21 Continuous IBP
- #26 Override-FVA classifier
- #27 Counterfactual replay
- #30 Adversarial red-team agent

### Honest competitive verdict (plain language)

Right now we're at **"impressive demo, not ready to buy"** stage.

If a company puts our forecasting head-to-head against SAP IBP or Kinaxis on their own data, we hold our own — the forecasting math is genuinely good. But when that company sends us their checklist of required features (the standard procurement document — "do you support lot tracking? multi-tenant? sales orders? capacity planning?"), we get screened out before anyone runs the bake-off, because the data plumbing for those features isn't there.

**The road back to "real option" status:**
- **4 to 6 quarters of Wave 2 work** would make us a credible choice for medium-sized food, beverage, and consumer-goods companies, plus smaller retailers trying to compete with the big chains.
- **Without that work**, our realistic position is "the AI/forecasting brain that plugs into a customer's existing planning system" — not a replacement for the planning system itself.

**The single biggest 12-month investment to make** (per the regulatory judge): **food traceability (FSMA Rule 204) + EU carbon-border tax reporting (CBAM) + EU deforestation tracking (EUDR) + tamper-proof decision log**. Doing those four together solves two problems at once: the 2026 compliance deadlines customers are panicking about, AND the audit foundation any trustworthy AI co-pilot needs anyway.

### Underserved-role audit (Consolidator E)

Five planner roles get effectively zero support today: **Customer Service / OMS Operator** (no sales-order tables, no ATP/CTP, no allocation), **External Partner** (no EDI/AS2, no portal, no sandbox tenant), **Sustainability/Compliance Officer** (no ESG schema), **Risk/Resilience Officer** (`supply_scenarios` is a stub), **Field Ops/Mobile** (no PWA, no barcode, no offline queue). All are entirely greenfield in the current platform and gate specific RFP segments.

### Strongest cross-cutting consensus (5+ roles agreed)

Three enhancements appear repeatedly across role-specific top lists:
1. **Lot grain (`dim_lot`) + recall/FEFO/DSCSA traceability** — Inventory, OMS, Compliance, Sustainability, Field Ops
2. **Bitemporal facts + AS-OF queries / immutable audit** — Data Eng, AI Sci, Compliance, S&OP, Demand
3. **Specialist agent split + conversational copilot grounded in tools/RAG** — AI Sci, every planner role, Exec

### Source artifacts

All raw materials live under `/tmp/sc_*.md` for the duration of this review:
- `/tmp/sc_expert_01..10_*.md` — 10 expert files, 100 enhancements each
- `/tmp/sc_consolidation_{B,C,D,E,F,G}_*.md` — 6 consolidator outputs (capability tiers, roadmap quadrants, competitive matrix, by-role view, dependency DAG, monetization)
- `/tmp/sc_judge_01..09_*.md` — 9 judge rankings
- `/tmp/sc_FINAL_RANKED_SYNTHESIS.md` — full synthesis (this section is the canonical version)

---

## Goal

Evolve the Supply Chain Command Center from a dashboard-and-pipelines platform (Gen-3) into a system where **AI is the operating system**: autonomous agents sense, decide, act, and explain; planners set goals and intervene on exceptions; every number is causally attributed and every decision is auditable.

**North-star KPI:** *human-touches-per-plan-cycle*, target <N per 1000 SKUs.

---

## Top 10 Cross-Cutting Priorities

These surfaced in three or more independent reviews. Treat as foundational — everything else depends on or amplifies them.

1. **Unify the service-level target across SS, fill rate, and S&OP.** ✅ **DONE** (2026-04-23). `common.core.service_levels` is the single resolver; DB `fact_service_level_targets` is authoritative, YAML is fallback. See [04-inventory/12-service-level-unification.md](04-inventory/12-service-level-unification.md).
2. **Stop assuming normality for intermittent/lumpy demand.** ✅ **LANDED** (2026-04-23). `scripts/run_ss_simulation.py` gained `empirical_quantile_ss()` + `--method` flag; `config/inventory/safety_stock_config.yaml` ships `method: normal_approx | empirical_quantile`. FM-quantile → SS path documented in `fm_spine` config block.
3. **Replace hand-entered `uplift_pct` with learned causal elasticities.** ✅ **SCAFFOLD LANDED** (2026-04-23). `sql/141` partitions `fact_external_signal` + `fact_causal_elasticity`; `scripts/ml/fit_elasticity.py` (sklearn OLS fallback, EconML DML TODO); `v_event_uplift_effective` view prefers learned over manual. Follow-up: production EconML integration, ingest real external signals.
4. **Immutable AI decision ledger + policy-as-code guardrails.** 🟡 **LEDGER FOUNDATION LANDED; POLICY ENGINE NOT YET IMPLEMENTED** (2026-04-23). DDL `sql/137_create_ai_decision_ledger.sql`, helper `common/ai/decision_ledger.py`, policies config `config/ai/agent_autonomy.yaml`. The `common/ai/policy_engine.py` prototype was built and unit-tested but never wired into a runtime path, and was removed as unwired; the guardrail engine remains to be rebuilt against real agent call sites. See [06-ai-platform/05-decision-ledger-and-policy.md](06-ai-platform/05-decision-ledger-and-policy.md). Follow-up: rebuild policy engine, wire existing agents.
5. **Feature store with point-in-time correctness.** ✅ **SCAFFOLD LANDED** (2026-04-23). `sql/138_create_feature_store.sql` + `common/feature_store.py` (Feast-compatible API over Postgres; HISTORY-table LATERAL join); 6 tests. Feast swap documented as follow-up.
6. **Promote MLflow to authoritative Model Registry with gated promotion.** ✅ **LANDED** (2026-04-23). `api/routers/forecasting/backtest_management.py` gate: min WAPE improvement + min coverage; every accept/reject writes `DecisionRecord` to the ledger; `config/forecasting/forecast_pipeline_config.yaml` `champion.promote_gate` block.
7. **Unified Digital Twin + closed-loop exception orchestrator.** ✅ **TWIN SCAFFOLD LANDED** (2026-04-23). `common/twin/state.py` with `TwinState.simulate(scenario, n_iter)`; `scripts/run_ss_simulation.py` wired as first consumer. Closed-loop orchestrator still pending.
8. **Chronos-2 as FM spine, not one of 10 competing models.** ✅ **CONFIG LANDED** (2026-04-23). `config/forecasting/forecast_pipeline_config.yaml` `fm_spine` block marks `chronos2_enriched` as production champion default; `collapse_tree_variants` flag gated off. CRPS-vs-WAPE champion selection still pending.
9. **pgvector-backed RAG + knowledge graph in one engine.** ✅ **LANDED** (2026-04-23). `sql/139_create_rag_chunk.sql` (VECTOR(1536) + HNSW + GIN + tsvector trigger); `sql/140_create_knowledge_graph.sql` (kg_node + kg_edge); `common/ai/rag.py` with RRF fusion (9 tests). Legacy `chat_embeddings` table and the NL→SQL `/chat` surface have since been removed; `rag_chunk` is the single embedding store.
10. **Vite proxy drift + hand-maintained TS types are structural debt.** Collapse proxy to array-driven loop; generate TS from FastAPI OpenAPI. ✅ **Proxy refactor done** (2026-04-23) — `frontend/vite.config.ts` uses a single `API_PATH_PREFIXES` array; `scripts/tools/audit_routes.py` parses both forms and verifies consistency. OpenAPI → TS codegen still pending.

---

## 1. Supply Chain Methodology

### P0 — Must-fix this quarter

| # | Rule | Source |
|---|---|---|
| 1.1 | ✅ Unify SL target store; single formula drives SS, fill rate, SOP. (2026-04-23) | SC-5, SC-2, SC-10 |
| 1.2 | ✅ Wire `baseline_intermittent` + `baseline_intermittent_window` into `forecast_pipeline_config.yaml` backtest block. (2026-04-23) | SC-1 |
| 1.3 | ✅ `embargo_months` raised from 0 to 1 in `forecast_pipeline_config.yaml`. (2026-04-23) | SC-1 |
| 1.4 | ✅ Broaden champion meta-learner — `champion.models: null` now drives selection off `get_competing_model_ids()` (all competing algorithms). (2026-04-23) | SC-1, SC-7 |
| 1.5 | ✅ OTIF (date + qty) added to `mv_supplier_po_performance`; reliability score now 40% OTIF + 20% OTD + 40% LT-consistency; router + UI columns updated. (2026-04-23) | SC-6 |
| 1.6 | ✅ `mv_supplier_performance` dropped in `sql/143`; all readers repointed to `mv_supplier_po_performance`. (2026-04-23) | SC-6 |
| 1.7 | ✅ New `GET /control-tower/kpis-financial` returns $-denominated inventory value, below-SS gap, excess, exception exposure. (2026-04-23) | SC-10 |
| 1.8 | ✅ `common/engines/exception_engine.py` detectors now all emit populated `financial_impact` (4 new estimators; unit cost from `fact_eoq_targets`). (2026-04-23) | SC-8 |
| 1.9 | ✅ `fact_exception_lifecycle` append-only table + trigger + MTTR views; acknowledge / status writes record transitions; `/mttr` + `/{id}/lifecycle` endpoints. (2026-04-23) | SC-8 |

### P1 — Next

- ✅ Quantile heads scaffold on LGBM — `algorithms.lgbm_cluster.params.quantile_heads` config key + 3-fit TODO in `scripts/run_backtest.py`. (2026-04-23) (SC-1, SC-7)
- ✅ Periodic-review ROP `(LT + R/2)` — new `compute_reorder_point_periodic()` in `compute_safety_stock.py`; flag `safety_stock.periodic_review_protection.enabled`. (2026-04-23) (SC-2)
- ✅ Supply yield/fill variability — `yield_std_days` param in `compute_ss_combined()`; flag `include_supply_yield_variability`. Main-loop wiring TODO. (2026-04-23) (SC-2)
- ✅ Per-SKU holding/ordering cost — `sql/145_add_sku_cost_overrides.sql` (nullable cols on `dim_sku`). EOQ/replenishment read-path TODO. (2026-04-23) (SC-2)
- ✅ Rebalancing on-hand + in-transit + open POs — `load_inventory_state()` extended; `available_supply` config section. (2026-04-23) (SC-4)
- ✅ `objective` knob — `max_service_solver` + `equalize_dos_solver` added; dispatcher honors `objective` key in `rebalancing_config.yaml`. (2026-04-23) (SC-4)
- ✅ Line-fill + case-fill scaffold — `sql/146_add_fill_rate_line_case.sql` (columns added, values TODO). (2026-04-23) (SC-5)
- ✅ Demand-weighted fill rate — already `SUM(shipped)/SUM(ordered)` in `api/routers/operations/fill_rate.py`; confirmed during audit, no change needed. (2026-04-23) (SC-5)
- ✅ Cash-to-cash, turns, DIO/DPO/DSO — new router `api/routers/inventory/working_capital.py` with `GET /analytics/working-capital`. (2026-04-23) (SC-10)
- ✅ Per-cluster tuning profile auto-invalidation — `sql/148` + `promote_scenario()` upserts stale rows; `POST /admin/tuning/invalidate-stale` endpoint. (2026-04-23) (SC-9)
- ✅ Exceptions grouped by root-cause key + SLA bands — `sql/149` adds cols; `derive_severity_band()`, `compute_sla_due_at()`, `compute_root_cause_key()` in `exception_engine.py`; `config/operations/exception_sla.yaml`. (2026-04-23) (SC-8)
- ✅ `fact_sop_decisions` log — `sql/147_create_fact_sop_decisions.sql` + `common/engines/sop_decisions.py` helper. SOP router wiring TODO. (2026-04-23) (SC-3)

### P2 — Polish

- Stranded-inventory detection; MEIO for DC→DC cascading; capacity constraints in solvers (SC-4)
- Business-day projection rate — config flag added, wiring **TODO** (SC-2)
- `est_lost_sales` as demand signal — config flag `include_est_lost_sales_in_demand` added; wiring **TODO** (SC-5)
- Pre-segment intermittent SKUs before KMeans; add DBSCAN/HDBSCAN comparison (SC-9) — **deferred**
- Gate SOP approval on open critical exceptions (SC-3) — **deferred** (needs SOP router refactor)
- ✅ Canonical accuracy helper — `common/services/metrics.py` `compute_accuracy/bias_pct/wape` + `ACCURACY_SQL_TEMPLATE`. Call-site migration TODO. (2026-04-23) (SC-10)
- ✅ Weekly granularity + rolling 13-week view — `sql/150_create_agg_sales_weekly.sql` + `GET /analytics/rolling-13-week` endpoint. (2026-04-23) (SC-10)

---

## 2. Backend & Frontend Engineering

### Backend (Python / FastAPI / Postgres)

**P0**
- ✅ FastAPI `lifespan` handler — opens pool + prewarms scheduler on startup, closes both on shutdown. (2026-04-23)
- ✅ Router restructure started — 7 routers moved into `operations/`, `platform/`, `intelligence/`, `core/` with backward-compat `sys.modules` shims; CLAUDE.md updated. Legacy inv_planning/storyboard/events still at flat root. (2026-04-23)
- ✅ `api/pool.py` now delegates to `common.core.db.get_db_params()` (single env-var source); added `open_pool`/`close_pool` for lifespan. (2026-04-23)

**P1**
- ✅ 29 bare `except Exception` narrowed + 11 documented `noqa: BLE001`; pre-existing `NameError: logger` in `clusters.py` also fixed. ~190 sites remain (tracked via the `check_unenforced_rules.sh` bare-except gate + allowlists). (2026-04-23)
- ⏭ Mount `require_api_key` at router level — **skipped with rationale**: audit showed no router has auth on every endpoint (GETs are intentionally unprotected); router-level mounting would break public reads.
- ✅ `scripts/tools/check_fstring_sql.py` — tokenize-based grep flags `cur.execute(f"…{var}…")` with variable interpolation; exits non-zero; 6-test coverage. (2026-04-23)

**P2**
- ✅ LLM client reset helper — `api.llm.reset_llm_client()` + `POST /admin/llm/reset` (auth-guarded). (2026-04-23)
- ✅ `make_pool` multi-fetchall factory — `fetchall_returns` / `fetchone_returns` list kwargs in `tests/api/conftest.py`, backward-compatible. (2026-04-23)

### Frontend / ML pipeline (React / Vite / scripts)

**P0**
- ✅ Collapse 48-block Vite proxy to array-driven loop; wire `make audit-routers` into CI (2026-04-23)
- ✅ OpenAPI → TS codegen wired — `openapi-typescript` devDep added in `package.json`; `npm run gen:types` script; placeholder `frontend/src/api/generated/schema.ts`. (2026-04-23)

**P1**
- ✅ Centralized React Query key factory audit — all query modules use `*Keys` factories; ESLint `no-restricted-syntax` rule warns on raw `queryKey: [...]` literals; `frontend/CONTRIBUTING.md` documents convention. (2026-04-23)
- ✅ `run_backtest_*.py` variants — audit confirmed catboost/xgboost scripts are already thin shims pointing at `scripts/run_backtest.py --model`. Foundation/DL/MSTL have distinct inference paths; left as-is. (2026-04-23)
- ✅ `model_registry.build_model(algorithm_id, params)` factory — lookup over `forecast_pipeline_config.yaml` `algorithms:` block; foundation-model stub with TODO; `UnknownAlgorithm` error. (2026-04-23)

**P2**
- Pick flat-vs-nested convention for `tabs/` — **deferred**
- ✅ ESLint `useThemeContext` enforcement — `no-restricted-syntax` warns on `theme` prop in JSX + raw query-key literals. (2026-04-23)
- ✅ `early_stop_pct` docstring drift fix — updated to 5% matching code in `common/ml/model_registry.py` + CLAUDE.md. (2026-04-23)

---

## 3. UI / UX

### P0 — Accessibility & safety gaps (ship first)

- ✅ Radix Dialog migration — new `frontend/src/components/ui/dialog.tsx`; `KeyboardShortcutHelp`, `ai-planner/ConfirmModal`, `ai-planner/AutoAcceptModal` migrated (focus trap, aria-modal, escape, focus restore). ~16 other homegrown modals still TODO. (2026-04-23) (UX-4)
- ✅ Chart accessibility — `EChartContainer.tsx` wraps in `role="img"` with aria-label + sr-only data-table fallback (first 50 rows). (2026-04-23) (UX-4)
- ✅ Replace `window.alert()` with non-blocking toasts (2026-04-23) — `frontend/src/components/Toaster.tsx` (sonner-compatible API so swap later is trivial); `ExceptionQueuePanel.tsx` updated. Custom modal `div` removal still pending. (UX-7)
- ✅ Optimistic updates — `ExceptionQueuePanel.tsx` ack + status mutations now use `onMutate`/`onError` rollback; test verifies optimistic flip. (2026-04-23) (UX-7)
- ✅ Undo for status changes — `frontend/src/hooks/useUndoable.ts` shows 5-second toast with Undo action; wired into exception status mutation. (2026-04-23) (UX-7)
- ✅ `formatApiError` sanitizer — `frontend/src/lib/formatApiError.ts` strips stack frames, paths, pg DETAIL/HINT; maps 401/403/404/429/5xx to friendly messages. (2026-04-23) (UX-10)
- ✅ Global `QueryClient` onError + no-retry-on-4xx — `main.tsx` installs QueryCache + MutationCache with `handleGlobalError` (calls `toast.error(formatApiError(e))`) and `shouldRetry` capped at 2 + refuses 4xx. (2026-04-23) (UX-10)
- ✅ Dark-mode system-preference sync + FOUC fix — inline script in `index.html` applies mode before React paints; `useTheme.ts` listens to `matchMedia` until user explicitly toggles. (2026-04-23) (UX-6)

### P1 — Major IA & analytics UX

- ✅ Nav reordered Tower → Demand → Supply → Operations → System in `AppSidebar.tsx`; section-order test added. (2026-04-23) (UX-1)
- Demand section split (Analysis vs ML Lab) — **deferred** (needs IA workshop) (UX-1)
- ✅ Breadcrumbs — new `frontend/src/components/Breadcrumbs.tsx` (6 tests) wired into `ItemAnalysisTab.tsx` (Tab > Item > LOC). Other drills still TODO. (2026-04-23) (UX-1)
- ✅ KPI deltas + targets + sparklines — `KpiCard` already supported; wired into Accuracy/WAPE/Bias in `AggregateAnalysisTab.tsx`. "$ at Risk" card fix still **TODO**. (2026-04-23) (UX-2)
- Dashboard-wide time-range selector; global filters flow to KPI queries — **deferred** (UX-2)
- ✅ Confidence-interval band on forecast lines — `ForecastTrendChart.tsx` `includeCI` prop now renders 80% quantile band (stack-area). (2026-04-23) (UX-3)
- ✅ Okabe-Ito color-blind-safe palette exposed via `useChartColors()`; documented in `CONTRIBUTING.md`. (2026-04-23) (UX-3)
- Over-plotted charts → small multiples — **deferred** (UX-3)
- ✅ Forms a11y — `ClusterPromoteModal` + `ExperimentBuilder` now have `htmlFor`/`id` pairs, `onBlur` validation, consistent required marker, `aria-required`, `aria-invalid`. Other forms still TODO. (2026-04-23) (UX-8)
- ✅ Empty-state triad (no-data / filtered / error) — `EmptyState.tsx` gained `variant` + `onAction`; rolled out in `JobHistoryPanel` and `DataQualityTab` (Domain Health / Pipeline Lineage / DQ Corrections). (2026-04-23) (UX-10)
- Consolidate Control Tower and Command Center — **deferred** with TODO comment in `CommandCenterTab.tsx` (both reachable via URL state; real merge needs UX review). (UX-2)

### P2 — Polish

- Touch targets ≥44 px, responsive sidebar drawer, table horizontal scroll (UX-5) — **deferred**
- ✅ Focus ring — `index.css` `:focus-visible` now uses `--ring` token with visible contrast. Token audit for other drift still TODO. (2026-04-23) (UX-6)
- Bulk actions + row-level keyboard shortcuts (`a`/`s`/`r`) on exception queue (UX-7) — **deferred**
- Draft recovery + step-wizard for dense form modals (UX-8) — **deferred**
- Stale-while-revalidate indicator; offline banner; per-section error tiles (UX-10) — **deferred**

**Gap:** UX-9 (Tables & Grids) review did not complete — permission prompt was rejected during the run. Re-run if comprehensive table-UX coverage is needed.

---

## 4. AI Gen-4 Transformation (core of this roadmap)

### Phase 0 — Foundations (weeks 1–8)

- Split `AIPlannerAgent` monolith → 5 specialist agents (Demand, Supply, SOP, Exception, Negotiator) + Orchestrator, behind MCP tool contracts (AI-1) — **deferred**; closed-loop orchestrator (phase 3) now provides the skeleton that the specialist agents will plug into.
- 🟡 Encode autonomy tiers in `config/ai/agent_autonomy.yaml` (config landed 2026-04-23); wrap all writes in a policy engine — **not yet implemented**. The `common/ai/policy_engine.py` prototype was removed as unwired; engine to be rebuilt against real agent writes. (AI-1, AI-10)
- ✅ Immutable `ai_decision_ledger` with hash-chained rows + DB-trigger append-only (AI-10) — (2026-04-23)
- ✅ Three-tier memory (working / episodic / semantic) — `common/ai/memory.py` with `WorkingMemory` (TTL), `EpisodicMemory` (FK to ai_decision_ledger via `fact_decision` in `sql/142`), `SemanticMemory` (pgvector via rag.py); 12 tests. (2026-04-23) (AI-1, AI-5)
- ✅ `chat_embeddings` retired in favour of `rag_chunk` (HNSW + GIN); `kg_node` / `kg_edge` added. (2026-04-23) (AI-5)
- ✅ Digital Twin service `common/twin/state.py` scaffolded; first consumer = `run_ss_simulation.py`. (2026-04-23) (AI-8)
- ✅ Feature store — local-Postgres scaffold `common/feature_store.py` + `sql/138` entity/view metadata tables; Feast-compatible API. (2026-04-23) (AI-9)
- ✅ MLflow promote gate — WAPE-improvement + coverage floors; every accept/reject logged to decision ledger. (2026-04-23) (AI-9)

### Phase 1 — FM-first forecasting (weeks 6–14, parallel to Phase 0)

- ✅ Roster collapse config — `fm_spine` config block in `forecast_pipeline_config.yaml` marks `chronos2_enriched` as production champion default + `collapse_tree_variants` flag (gated off). (2026-04-23) (AI-2)
- ✅ FM quantile output → SS — `common/ml/fm_quantile_bridge.py` (`FMQuantileForecast`, `load_fm_quantile_forecast`, `fm_demand_pool`); `common/twin/state.py` extended with `use_fm_quantiles`/`fm_n_samples` kwargs. 8 tests. (2026-04-23) (AI-2)
- Native covariate channels (price / promo / weather); hierarchical MinT reconciliation — **deferred** (AI-2)
- 🟡 k-shot cold-start via metadata + nearest-neighbor — **not yet implemented**. A `common/ml/cold_start_neighbors.py` prototype (`DFUMetadata`, `nearest_neighbors`, `build_prompt_prefix`) was built and unit-tested but never wired into a forecasting path, and was removed as unwired. (Note: production cold-start currently routes 3–11-month DFUs to `rolling_mean` per `forecast_pipeline_config.yaml`.) (AI-2)
- ✅ CRPS / pinball loss — `common/ml/crps.py` (`compute_crps`, `compute_pinball_loss`); `champion.metric` config key added. `run_champion_selection.py` validates + falls back to WAPE until quantile rows guaranteed. 7 tests. (2026-04-23) (AI-2)
- Batched GPU inference server — **deferred** (infra change) (AI-2)

### Phase 2 — Causal + sensing + NL (weeks 10–20)

- ✅ `fact_external_signal` partitioned ingest (RANGE by event_ts) — `sql/141`. (2026-04-23) (AI-3)
- ✅ Causal elasticity scaffold — `scripts/ml/fit_elasticity.py` (OLS fallback) + `fact_causal_elasticity` + `v_event_uplift_effective` view that prefers learned over manual `uplift_pct`. EconML DML remains TODO. (2026-04-23) (AI-3)
- 🟡 Near-term sensing model — the standalone `common/ml/sensing.py` (`blend_forecasts()` horizon-weighted blend) was removed as unwired; its blending logic is now superseded by inline blending in `scripts/forecasting/compute_blended_forecast.py`. `config/forecasting/sensing_config.yaml` remains. (AI-3)
- Tool-calling chat with intent taxonomy + semantic-layer view registry (AI-4) — **deferred**
- 🟡 Multimodal response envelope `{narrative, blocks: [chart|table|action_card|markdown]}` — **not yet implemented**. A `common/ai/envelope.py` prototype with strict typed validation was built and unit-tested but never wired into a response path, and was removed as unwired. (AI-4)
- 🟡 Dry-run preview + explicit confirm — **not yet implemented**. A `common/ai/dry_run.py` prototype (`dry_run` + `confirm` helpers, pluggable handler registry, ledger write on confirm) was built and unit-tested but never wired into a runtime path, and was removed as unwired. (AI-4)
- Hybrid retriever (BM25 + vector + KG + SQL) with cross-encoder rerank — RRF fusion (vector + BM25) landed in `common/ai/rag.py`; KG + cross-encoder rerank remain TODO. (AI-5)
- ✅ Ingest SOPs / runbooks / post-mortems — `scripts/ai/ingest_docs.py` (500/50 overlap chunker, zero-vector embedding TODO, `--dry-run`). 9 tests. (2026-04-23) (AI-5)

### Phase 3 — Prescriptive + RL + Vision (weeks 16–30)

- ✅ Closed-loop exception orchestrator — `common/ai/orchestrator.py` (`ExceptionOrchestrator.detect/simulate_options/rank/route`), wires TwinState + reversible.apply + decision_ledger (policy-engine integration pending — that engine is not yet implemented). 6 tests. (2026-04-23) (AI-8)
- ✅ MILP optimizer scaffold — `common/ml/milp.py` + `scripts/ml/milp_rebalancer.py` (greedy fallback; highspy integration as documented TODO). 5 tests. (2026-04-23) (AI-8)
- ✅ Reversible action ledger + 24h auto-rollback — `sql/165_create_fact_reversible_action.sql` + `common/ai/reversible.py` (`apply`, `rollback_pending` sweeper). KPI-regression detector is TODO. 7 tests. (2026-04-23) (AI-8)
- Gymnasium RL env + offline RL (CQL/IQL) + OPE-gated rollout (AI-6) — **deferred** (large scope)
- On-Shelf Availability from shelf photos via Claude 4 vision (AI-7) — **deferred**
- Supplier document AI for POs / invoices / ASNs (AI-7) — **deferred**

### Phase 4 — Governance & scale (ongoing)

- ✅ Local + counterfactual explanation API `/forecast/explain/{item_id}/{loc}` — `api/routers/intelligence/explain.py`; reads SHAP if table exists, degrades gracefully otherwise; writes to decision ledger. 3 tests. (2026-04-23) (AI-10)
- ✅ Fairness audit MV — `sql/158_create_mv_fairness_audit.sql` slices by abc_vol, region, channel; computes slice_wape + disparity_ratio. (2026-04-23) (AI-10)
- Prompt-injection + data-poisoning defenses — **deferred** (AI-10)
- ✅ OpenLineage minimal emission — `common/ai/lineage.py` (`emit_event`) + `sql/157_create_fact_lineage_event.sql`; wired into MLflow promote path. 5 tests. (2026-04-23) (AI-9)
- 🟡 Drift detection (PSI + rolling WAPE) — **not yet implemented**. A `common/ai/drift.py` prototype (`compute_psi`, `rolling_wape`, `psi_signal`, `wape_signal`) was built and unit-tested but never wired into a runtime path, and was removed as unwired. The `sql/155_create_fact_drift_signal.sql` table remains as the intended sink. (AI-9)
- ✅ Shadow rollout scaffold — `common/ml/shadow_rollout.py` (`ShadowRollout`, `should_tee`, `insert`) + `sql/156_create_fact_shadow_rollout.sql`. 8 tests. (2026-04-23) (AI-9)
- EU AI Act / NIST AI RMF risk register; model-card rendering in UI (AI-10) — **deferred**

---

## Suggested 90-Day Sequencing

| Weeks | Streams running in parallel |
|---|---|
| 1–4 | SL-target unification · decision ledger · policy engine · Vite proxy + OpenAPI cleanup · a11y P0 |
| 4–8 | Digital twin service · feature store · MLflow registry · RAG chunk upgrade · optimistic updates |
| 6–12 | Chronos-2 FM spine + LoRA · causal elasticity · dashboard KPI-in-dollars · navigation re-org |
| 10–14 | Tool-calling chat + semantic layer · exception orchestrator · bias audit MV |

---

## Agent Source Key

| Code | Scope |
|---|---|
| SC-1 | Demand forecasting methodology |
| SC-2 | Inventory planning (SS / EOQ / replenishment / projection) |
| SC-3 | S&OP process design |
| SC-4 | Network optimization / rebalancing |
| SC-5 | Fill rate / service level |
| SC-6 | Procurement / supplier management |
| SC-7 | ML / AI model selection & ensemble |
| SC-8 | Data quality / exception engine |
| SC-9 | SKU clustering / segmentation |
| SC-10 | Financial / KPI metrics |
| Coder-1 | Backend architecture (FastAPI / Postgres / Python) |
| Coder-2 | Frontend + ML pipeline architecture |
| UX-1..10 | Navigation · Dashboard · Data viz · A11y · Responsive · Theming · Interactions · Forms · Tables (incomplete) · Error/empty states |
| AI-1 | Agentic orchestration |
| AI-2 | Foundation models for forecasting |
| AI-3 | Causal AI & demand sensing |
| AI-4 | Natural-language planning interface |
| AI-5 | RAG & knowledge graph |
| AI-6 | RL for inventory / replenishment |
| AI-7 | Computer vision & multimodal |
| AI-8 | Prescriptive AI & digital twin |
| AI-9 | ML platform / MLOps |
| AI-10 | XAI, safety, governance |

---

## Dependencies

- **Phase 0 foundations** are prerequisites for Phases 1–4. No agent or auto-action work should start before the decision ledger + policy engine land.
- **FM-first forecasting (Phase 1)** depends only on the feature store and MLflow registry from Phase 0.
- **RL (Phase 3)** depends on the digital twin (Phase 0) and drift detection (Phase 4) before any opt-in rollout.
- **Causal layer (Phase 2)** depends on `fact_external_signal` landing first.

## Out of Scope

- Full data-center relocation / cloud provider change
- Rewriting core domain ingest (`normalize-all` / `load-all`) — Gen-4 layers on top
- ERP integration surface (belongs in `docs/specs/08-integration/`)
