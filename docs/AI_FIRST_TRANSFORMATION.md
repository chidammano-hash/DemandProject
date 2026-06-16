# AI-First Transformation — Strategy & Vision

> Synthesized from a 10-lens parallel review of the Supply Chain Command Center codebase (2026-04-28). Each finding cites the file where the pattern lives so the work is actionable, not aspirational.

---

## TL;DR — The Core Insight

**The platform has already paid for the substrate of an AI-first product but ships a classical CRUD experience on top of it.**

The codebase contains: `decision_ledger`, `lineage`, `causal`, `tuning_advisor`, `champion_strategies`, `external_signals`, `annotations`, `audit_log`, `pgvector RAG`, `notification_engine`, `outcome tracking`. (Earlier `policy_engine` and `dry_run` prototypes were built but never wired in, and have been removed; rebuilding them as wired primitives is part of the work below.) None of these primitives are surfaced in the UI. The product is one composition layer away from being unrecognizable — in the best way.

The transformation is not "add AI to the app." It is "stop hiding the AI we already built, retire the legacy CRUD shell wrapped around it, and let the agents drive."

---

## Part 1 — The Old World: Patterns to Retire

### 1.1 — Mental Model Legacy

| Old assumption | Where it shows up | Why it dies |
|---|---|---|
| Demand is a single number | `fact_production_forecast` writes one qty per `(item, loc, month)` | Demand is a distribution. Point forecasts lie about confidence. |
| The month is the planning unit | `cron: 0 6 2 * *` in `forecast_pipeline_config.yaml` | Real shocks (port closure, viral SKU, weather) move in hours. 30-day cadence is malpractice. |
| Planners triage exceptions | `ExceptionQueuePanel.tsx` (750 LOC), `_VALID_EXCEPTION_TYPES` in `inv_planning_exceptions.py` | Exceptions arrive *with* proposed resolutions. Humans approve, they don't triage. |
| The dashboard is where work happens | `ControlTowerTab.tsx`, `CommandCenterTab.tsx` (798 LOC) | Work happens in an Inbox, not a wall of charts. Dashboards are the artifact of pre-agent thinking. |
| ML is a feature you click | "AI Planner" tab as a single sidebar entry | ML is the substrate. The UI is the interface to a fleet of agents, not a button labeled "AI". |
| ABC/XYZ classes are a planning primitive | `dim_sku.abc_vol`, `inv_planning_abc_xyz.py`, `inv_planning_policy.py` `segment` | A 1950s storage heuristic. With per-SKU ML, bucketing is information loss. |
| Static service levels per class | `abc_xyz_service_level` config | Service level is *derived* from margin × stockout cost × customer tier. Stop storing it. |
| Periodic review is a real thing | `_VALID_POLICY_TYPES = {"periodic_review", ...}` | Calendars are pre-streaming. With live snapshots, planning is event-driven. |

### 1.2 — Architecture Legacy

- **Nightly batch ETL.** `scripts/etl/run_pipeline.py` + `Makefile pipeline-full` assume a 24-hour cycle. Latency from event to forecast is *days*.
- **Manual MV refresh chains.** `etl_config.yaml mv_refresh:` and `refresh-mvs-tiered` orchestrate ~15 MVs by hand. Brittle, blocking, all-or-nothing.
- **Hand-coded per-source schemas.** `DomainSpec` + `sql/NNN_create_*.sql` per dataset. No schema registry, no inference, every new source is a Make-target ceremony.
- **`api/llm.py` is a singleton chat client, not an agent runtime.** No tool registry, no turn loop, no token/cost ledger, no per-agent identity. Every caller re-implements (`MAX_TURNS=40`, `TOKEN_BUDGET=100_000` in `ai_planner.py`).
- **Lineage exists in DDL only.** `sql/157_create_fact_lineage_event.sql` is OpenLineage-shaped but unwired. The killer feature is paid for and unshipped.
- **No external-signal ingestion.** `sql/067` and `sql/141` define landing tables; zero rows flow.
- **No CDC from upstream ERP/WMS.** Inputs are CSV dumps in `data/input/`. A 4-hour shipment delay can't reach a planner before the next nightly batch.
- **Champion selection runs offline.** `make champion-all` is a shell command. No agent is in the loop.

### 1.3 — UX Legacy

- **Sidebar with 16 items + 21 tab files.** `frontend/src/tabs/` is a memory test. Default entry should be one screen, not a tab matrix.
- **`InvPlanningTab.tsx`'s 33-panel mega-tab.** With 5 `VIEW_PRESETS` ("Daily Essentials", "Weekly Review", ...) — a confession that nobody can find anything.
- **`SettingsTab.tsx`'s 3-pane field grid** for ~37 YAML files. The classic 30-field configuration screen antipattern.
- **`JobsTab.tsx` polling every 2-10s.** An ops-engineer view shoved at planners.
- **KPI tile rows everywhere.** `KpiCard.tsx` wallpapered across `ControlTowerTab.tsx`, `CommandCenterTab.tsx`. Decontextualized numbers, no spatial meaning.
- **HTML `<table>` heatmap** in `inv-planning/NetworkHeatmapPanel.tsx` (lines 138-179). A network problem visualized with no network.
- **Filter panels left of grid.** `GlobalFilterBar.tsx` everywhere. Filter-first assumes the user knows what to look at — they don't.
- **Modal dialogs / right-rail "investigation panels"** in `StoryboardTab.tsx`. 2014 ServiceNow pattern. Context switching breaks flow.
- **Status badges with no temporal context.** `getSeverityConfig("critical").badge` tells you "critical" but not "for how long" or "trending toward critical."
- **`useJobNotification` toast spam.** Every job submit/cancel/finish fires a notification. Planners learn to dismiss without reading.
- **Single-user dashboards.** No presence, no cursors, no threads, no @mention surfacing in-app. Planning is fundamentally collaborative; the UI pretends it isn't.

### 1.4 — ML/Forecasting Legacy

- **Once-a-month batch champion regeneration.** Cron `0 6 2 * *` in `forecast_pipeline_config.yaml`. The whole 5-stage pipeline (`cluster → backtest → load → champion → forecast`) is offline batch.
- **Champion-by-historical-WAPE-only.** `champion_strategies.py` registers 17 strategies — every one ranks on `cum_abs_err / cum_actual`. No signal from forecast *uncertainty*, residual autocorrelation, or downstream cost.
- **Foundation models stubbed.** `model_registry.py` lines 304-326 return `_FoundationStub` for Chronos/Bolt/N-BEATS. The `chronos2_enriched` declaration exists, the loader is a TODO.
- **Single-point forecasts in production.** `fact_production_forecast` writes one number. `fact_quantile_forecast` is parallel and ignored downstream.
- **Static feature lists.** `shap_selector.py` runs once per backtest, then frozen for weeks.
- **Exogenous signals as post-hoc adjustments.** `apply_event_adjustments.py` patches a finished forecast with multipliers. Promo/weather/macro never enter the model as features.
- **Cluster strategy hard-coded as `per_cluster`.** Every algorithm in YAML uses the same strategy.
- **SHAP as the only explainability.** Opaque to non-ML users.
- **No causal model for promo/event lift.** Multipliers, not uplift estimation.

### 1.5 — Process Legacy (inferred from what's missing)

- **Email exports + screenshots-into-Slack.** `notification_engine.py` only knows external channels (slack/teams/email/pagerduty). No in-app inbox channel. Conversations evaporate.
- **No decision threads on stage advance.** `sop.py` `_AdvanceRequest.notes: str` accepts `approved_by` as a free-text field. Anyone with the API key can type any name.
- **No subscriber model.** Forecasts have no `owner_user_id`. Nobody is paged when "their" SKU breaks.
- **No version history with named saves.** `model_promotion_log` is the only audit trail; doesn't extend to plans, scenarios, S&OP cycles.
- **No record of *why* a number changed.** Override queues mutate qty without comment capture.
- **Excel as the real planning surface.** Inferred from the absence of any feature that would obsolete it.

---

## Part 2 — The New World: Patterns to Adopt

### 2.1 — Six Concrete Agents (replacing the "AI Planner" tab)

Each agent uses a `policy_engine.evaluate()` (to be rebuilt — the prototype was removed as unwired) + the existing `decision_ledger.append_decision()` + `reversible.apply()` triad. Tier defaults from `config/ai/agent_autonomy.yaml`.

| Agent | Trigger | Tools | Escalation |
|---|---|---|---|
| **ExceptionResolver** | APScheduler 15-min + on `fact_replenishment_exceptions` insert | `simulate_options` (twin), `get_supplier_lead_times`, `get_alternate_locs`, `apply_reversible` | severity=critical OR `pct_change > guardrail` → human inbox; else auto-apply |
| **ChampionGuardian** | Nightly + on backtest completion | Read `mv_control_tower_kpis`, `fact_candidate_forecast`, run `champion_strategies.simulate`, call promote endpoint | WAPE drift >5pp or new model wins by <2pp → human approval |
| **TuneOrchestrator** | When `cluster_tuning_profile.stale=true` | `tuning_advisor.suggest_search_space`, kick `make tune-lgbm-clusters`, watch MLflow | Budget overrun >2x baseline → pause |
| **DataQualitySentinel** | On each `make load-all` completion | `dq_engine` checks, drift detection (to be rebuilt — prototype removed), schema diff | P0 schema break → block; soft drift → quarantine partition |
| **SOPNarrator** | Monthly + on-demand | Read `mv_network_balance`, `mv_supplier_performance`, generate exec memo | `advisory` permanently — humans always edit |
| **PolicyDriftAuditor** | Weekly | Read `ai_decision_ledger`, compare actual vs predicted by a dry-run preview layer (to be rebuilt — prototype removed), propose `agent_autonomy.yaml` edits | Every change human-approval (the meta-agent) |

Underneath: replace `api/llm.py` with `common/ai/agent_runtime.py` — a single tool-loop runner (turn cap, token ledger, retry, provider failover) that all six agents inherit.

### 2.2 — Decision Class Registry & Autonomy Levels (L0–L5)

A new table `fact_decision_class` maps `(decision_type, scope)` → autonomy level, read by every write endpoint before mutating state.

| Level | Behavior | Example classes |
|---|---|---|
| **L0** | Human-only, no agent suggestion | Supplier change, contract terms |
| **L1** | Advisory: agent suggests, human always commits | Default for new classes |
| **L2** | Suggest-and-confirm: pre-baked one-click | PO placement >$X |
| **L3** | Auto-execute with revert window | SS recompute under $10k |
| **L4** | Auto-execute, post-hoc audit | Rebalancing under threshold |
| **L5** | Fully autonomous | Reserved for future |

**Trust graduation.** Each class auto-promotes one level every 30 days if (a) revert-rate <5%, (b) realized vs estimated impact correlation >0.7, (c) zero P0 incidents. Within a year the platform earns its way from co-pilot to autopilot, *per class*, with audit trail.

### 2.3 — Probabilistic, Continuous, Self-Explaining Forecasts

- **Foundation-model-first with tree fallback.** Promote `chronos2_enriched` to default; route only sparse/idiosyncratic clusters to LGBM. Replace `_FoundationStub` with real loader supporting zero-shot + LoRA per cluster.
- **Distributions, not points.** Collapse `fact_production_forecast` and `fact_quantile_forecast` into a single distributional table (`p05, p25, p50, p75, p95` + sample paths). Downstream consumers (`compute_safety_stock.py`, exception engine) take a distribution.
- **CRPS as primary champion metric, alongside WAPE.** `common/ml/crps.py` already exists.
- **Continuous online retraining triggered by drift.** `scripts/ml/detect_drift.py` exists; wire into APScheduler as daily trigger calling partial-fit when CUSUM on residuals breaks threshold. Replace monthly cron with drift-event topic.
- **Causal AI for promo/event lift.** Replace `apply_event_adjustments.py` with causal forest / DML estimating lift conditional on baseline + price + competitor state, feed *uplift* as covariate.
- **LLM-driven feature discovery loop.** LLM reads SHAP report after each backtest, proposes new transformations, auto-generates pandas code, A/B tests in `champion-experiments`.
- **Cost-aware champion.** Augment `champion_strategies.py` with `cost_aware` strategy weighting each error by holding cost vs stockout cost. The "best" model minimizes inventory $$, not abstract accuracy.
- **Hierarchical reconciliation by default.** `run_backtest_bolt_hierarchical.py` exists in isolation — make MinT post-step on every forecast.
- **Shadow mode mandatory.** Every new champion runs N weeks against live; auto-promote only when CRPS + business-cost both improve.

### 2.4 — Inventory Planning, Reimagined

- **RL-driven (s, S) policies per SKU.** Train contextual bandit / DQN: state = (on-hand, in-transit, forecast distribution, lead-time CDF, margin, days-to-event), action = order qty, reward = -(holding + stockout × lost-margin + expedite). Replace `compute_safety_stock.py` outputs with learned policy table; King formula only as cold-start prior for SKUs <12 months history.
- **Multi-echelon stochastic optimization (MESO).** Replace per-loc SS with Graves-Willems Guaranteed-Service Model across `dim_transfer_lane` + supplier nodes. One service-level promise to the customer, optimized stock placement upstream. Data exists.
- **Dynamic per-SKU service levels driven by economics.** `SL* = p / (p + h)` where `p = unit_margin + brand_penalty + customer_tier_weight`. Pull margin from `customer_analytics.py`. Kill `abc_xyz_service_level` as a stored column.
- **Auto-resolved exceptions.** Each `fact_inventory_exceptions` row arrives with proposed action + simulated outcome + confidence. If `confidence > 0.85` AND `financial_impact < auto_approve_cap`, agent fires PO/transfer with `actor='agent'`. Humans see the long tail.
- **Digital twin simulation before commit.** Every recommended order from `compute_replenishment_plan.py` runs through Monte Carlo over 8 weeks against forecast quantiles. UI shows P10/P50/P90 outcome before commit. No order leaves without stated stockout probability.
- **Quantile-native planning.** Pipe P10/P50/P90 directly into policy. `sigma_D_daily / sqrt(30.44)` is a lossy round-trip — kill it.
- **Continuous review, event-driven.** Drop `review_cycle_days` from `PolicyCreateBody`. Trigger replenishment on signal change, not calendars.
- **Causal root-cause attribution.** `getRootCauseExplanation` switch in `ExceptionQueuePanel.tsx:76` becomes SHAP-style attribution: "forecast 70%, lead-time slip 20%, supplier OTIF 10%."

### 2.5 — Real-Time Data Fabric

- **Event-sourced ingestion via Debezium + Kafka.** Replace `load_*` scripts with connectors reading ERP/WMS WAL changes; consumers upsert into `fact_*` tables and emit `fact_lineage_event` rows automatically.
- **Schema registry with auto-DomainSpec.** Drive `domain_specs.py` from a Confluent/Iceberg registry — new sources self-onboard.
- **External-signal agents.** Wire ingestors into `fact_external_signal`: NOAA weather, MarineTraffic port congestion, USTR tariff RSS, GDELT/news, Reddit + Google Trends. Hook into existing `v_event_uplift_effective` and `fact_causal_elasticity` so signals drive forecast uplift.
- **LLM extraction from supplier comms.** Worker watches mailbox/SFTP/PDF dropbox, calls Claude with structured-output schemas, writes `eta_change`/`cost_change` rows; cite-back source PDF in `raw_payload`.
- **Embedding-based entity resolution.** Replace brittle string joins with pgvector matching on item descriptions and customer names — store in `rag_chunk` / `dim_*_embedding`.
- **Streaming anomaly gate.** Insert pre-`INSERT` step flagging 5σ volume jumps, null-rate spikes, unseen categoricals → write to `fact_drift_signal` and quarantine.
- **OpenLineage emitter.** Decorate `profiled_section()` with `lineage.emit(START/COMPLETE, inputs, outputs)`. Populate `fact_lineage_event` automatically. Expose via `/lineage/*`.

### 2.6 — Multiplayer Planning

- **WebSocket presence per artifact.** `/ws/presence?resource_type=sop_cycle&resource_id=42`; avatar stack on `SopTab` header and per `InvPlanningTab` panel.
- **Threads anchored to a resource pin.** Extend `fact_annotation` to pin `(panel_id, sku_id, month)` or `(chart_id, dataPoint)`. Right-rail thread sidebar.
- **In-app `@mention` inbox.** Add `inapp` channel to `CHANNEL_SENDERS` in `notification_engine.py` writing to `fact_user_inbox`; bell icon on shell.
- **Decision threads on stage advance.** Replace `_AdvanceRequest.notes: str` with required `decision_thread_id`; LLM summarizes thread into `approval_rationale`.
- **Owners + subscribers per forecast/SKU.** New `fact_subscription(user_id, resource_type, resource_id, role)`. `fact_production_forecast` rows gain `owner_user_id`.
- **Named saves + shareable URLs.** `fact_planning_snapshot(snapshot_id, label, created_by, parent_id)` with `?snapshot=abc123` deep-links.
- **AI scribe.** Stream `fact_audit_log` deltas during a cycle into LLM, append "What changed and why" markdown block to `sop/cycles/{id}` response.

---

## Part 3 — Next-Gen UI Vision

### 3.1 — The New Home Screen: Morning Briefing

Replace the 16-item sidebar landing on `InvPlanningTab.tsx` with a single conversational feed:

> **Good morning, Mano.** Overnight: 7 SKUs went red, 2 POs slipped, $1.2M cash at risk. Here's what to do first.
>
> **1.** ExceptionResolver wants to expedite 3 SKUs ($42k). Approve / Modify / Deny / Always-allow-like-this. *(simulated stockout drops 38% → 4%)*
>
> **2.** ChampionGuardian flagged Chronos2 beating LGBM by 2.3pp WAPE on the seasonal_high cluster. Promote? *(shadow mode 14 days, $11k EV)*
>
> **3.** Hurricane forming near Savannah port — 47 inbound POs at risk. Reroute draft ready.

Each item is a card. Default action is highlighted. The 33 panels become on-demand attachments, not navigation.

### 3.2 — Replacement Map (specific files → specific reimaginings)

| Current | Replace with |
|---|---|
| `ControlTowerTab.tsx` (5-zone wall of charts) | Single living-network map (ECharts `graph` + geo): nodes = locations sized by inventory, edges = animated PO/transfer flows colored by lead-time risk. Pulse intensity = velocity. KPIs become hover-state details. 12-week scrub bar at bottom. |
| `inv-planning/ExceptionQueuePanel.tsx` (table of 750 LOC) | Inbox of resolution cards. Each card: tiny on-hand vs ROP gauge, 90-day sparkline, projected stockout countdown, one-tap "simulate the fix" inline (no modal). |
| `NetworkHeatmapPanel.tsx` (HTML `<table>`) | 3D inventory positioning (deck.gl `ColumnLayer`): X = location, Y = category, Z = DOS, color = service risk. Tilt + orbit reveals ridges of excess and valleys of shortage. |
| `StoryboardTab.tsx` (left-list/right-detail) | Film-strip timeline: each frame = one exception's projected next 12 weeks, autoplaying. Click to pause and act. |
| _(no current chat surface — removed)_ | Persistent bottom command bar (Linear-style) on every route, with `/sku`, `/customer`, `/forecast`, `/jobs` slash-mentions resolving to typed entities. |
| `SettingsTab.tsx` (37-YAML field grid) | Conversational config: "Loosen safety stock for B-class items in Texas" → diff preview → commit. |
| `JobsTab.tsx` (polling KPI cards) | Slack-like activity stream. Jobs are messages with reactions ("retry", "cancel", "schedule like this"). Pipelines are threaded replies. |
| `KpiCard.tsx` rows | Vital-signs cards: single sparkline + delta + tiny semantic icon. No standalone numeric tiles — every number lives next to its 12-month spark. |
| `FVATab.tsx` bar charts | Sankey decomposition: demand split into channel/customer/SKU bands, thickness = volume, color = bias direction. |
| `EventCalendarPanel.tsx` (timeline) | GitHub-style 52-week heatmap. Click a cell to scrub the network. |
| Per-tab date pickers | Global temporal scrubber at bottom — re-projects every panel to that point in time. |

### 3.3 — Universal Affordances on Every View

- **`Why?` drawer.** Right-click any number → opens lineage chain to source data + agent decisions that touched it. `lineage.py` table exists; surface it.
- **`What if?` slider.** Right-click any number → counterfactual ("what if lead time +2 weeks?"). Reuses causal-chain backend.
- **Data-freshness chip.** Pull `max(loaded_at)` per source table; render "Sales: 4m ago • Forecast: 2h ago • Weather: live" beside every chart.
- **Inline micro-charts in narrative text.** "Fill rate dropped to 92% [▁▂▃▅▇▆▃] over 14 days at DC-12 [▲]" — Tufte sparkbars woven into AI-planner explanations.
- **Subscribe-to-this toggle** on any SKU/customer/forecast row.
- **Cursors + presence** on every shared dashboard.
- **Cmd-K palette** spanning tabs, SKUs, customers, jobs, configs, questions. Sidebar collapses to a 3-icon rail (Home, Inbox, History).

### 3.4 — Generative UI

The terminal vision: **the home screen rebuilds itself per planner per day.** An agent reads the planner's role, yesterday's actions, overnight exception deltas, upcoming S&OP milestones, and assembles three action cards, one chart that matters today, one decision pending. The 21 tabs become a fallback library the agent draws from. The product stops being "an app with tabs" and becomes "a planner whose UI happens to be in your browser."

---

## Part 4 — Ten Wow-Factor Moonshots

Each is technically credible (3 engineers, 6 months), specific to supply chain, and compounds with use. They build on substrate the codebase already ships.

### 1. **Counterfactual Cinema**
Netflix-style replay where any past stockout, missed forecast, or expedite is rerun under the assumption that one decision had been different. Side-by-side animated timeline of actual vs counterfactual P&L, OTIF, DOS.
*Tech:* `common/ai/causal.py` + `inv_planning_simulation` + deterministic replay seeded from `decision_ledger`.
*Risk:* Counterfactual validity depends on causal graph quality.

### 2. **Planner Twin**
Personalized agent shadows each planner for 2 weeks, learns override patterns from `decision_ledger`, then proposes pre-baked overrides each Monday with confidence scores ("you'd have changed these 14 — accept all?").
*Tech:* Per-user behavior embeddings + LLM rationale generator + `reversible.py` for safe auto-apply.
*Risk:* Habit-locking — bad planner habits get amplified.

### 3. **Supply Chain Black Box**
Aircraft flight recorder for the supply chain. Every forecast, override, exception, supplier tick captured as queryable event stream you can rewind on a global scrubber. "Take me to 3pm last Tuesday" — full system state restored.
*Tech:* Event-sourced log on `decision_ledger` + Postgres logical replication snapshots.
*Risk:* Storage cost; replay determinism for MV-derived state.

### 4. **Forecast Confessional**
Every Friday the champion model "confesses" its largest misses in plain English, names features that betrayed it, proposes its own retraining plan. Planner approves with one click.
*Tech:* SHAP delta vs realized actuals + LLM templated narrative + auto-PR into tuning queue.
*Risk:* LLM hallucinating spurious feature blame on intermittent series.

### 5. **The Negotiation Room**
Multi-agent simulation where Buyer-AI, Supplier-AI (built from supplier's historical behavior in `mv_supplier_performance`), and CFO-AI debate a PO before it's sent. Output: recommended terms, BATNA, transcript.
*Tech:* `common/ai/orchestrator.py` + supplier persona from MV + lead-time variance.
*Risk:* Synthetic supplier persona drift — must be grounded.

### 6. **Glass Network**
3D WebGL map where every SKU-location-customer flow is a glowing edge whose thickness pulses with real demand. Anomalies light up red and "ring." Bloomberg Terminal meets SimCity.
*Tech:* deck.gl + WebSocket stream of MV deltas + edge-bundling layout.
*Risk:* Visualizing 198M inventory snapshots without melting browsers — needs aggressive aggregation.

### 7. **Pre-Mortem Bot**
Before any plan commits, an agent writes the post-mortem that hasn't happened yet: "Here's how this plan most likely fails by week 6, ranked by probability." Inverts the planning workflow — failure modes surface before plans ship.
*Tech:* Monte Carlo over `inv_planning_simulation` + LLM narrator over top-k failure trajectories.
*Risk:* Planner desensitization if every plan gets a doomsday scroll.

### 8. **Demand DNA**
Every SKU gets a 1-page genetic profile — seasonality fingerprint, intermittency, customer concentration, substitution kin, lifecycle stage — auto-generated and embedding-searchable. "Find me 12 SKUs that behave like SKU-9821 before it crashed."
*Tech:* `common/ml/sku_features/` + pgvector over feature embeddings. Already 80% there.
*Risk:* Embedding staleness — needs nightly re-embed.

### 9. **Live Whisper Stream**
CNBC-style ticker fed by an agent scanning X/Reddit/news/weather/port AIS, surfacing only signals tied to your specific portfolio. *"Hurricane forming near Savannah port — 47 of your inbound POs at risk."*
*Tech:* Streaming RAG over external feeds + `external_signals.py` + entity-linker against `dim_item`.
*Risk:* Signal-to-noise — false positives erode trust within a week.

### 10. **The Auditor**
Standing AI agent that continuously re-derives every dashboard KPI from raw facts and flags any drift between "what the dashboard says" and "what the data says." Self-policing trust layer.
*Tech:* Shadow query layer + SQL diff engine + a query-tracking layer (an earlier `query_tracker.py` prototype was removed as unwired; this would be rebuilt).
*Risk:* Compute overhead — must sample, not exhaust.

### Bonus: **Conversational Planning Replay**
Planner asks: "Show me what would have happened in March if ExceptionResolver had been on autonomous tier." System replays the month against `decision_ledger` + a dry-run twin (the preview layer to be rebuilt — prototype removed), scoring counterfactual P&L, fill-rate, stock-outs vs actual. Then offers: "Promote autonomous tier under these guardrails?" with one-click commit. Only possible because every decision is hash-chained, reversible, and twin-simulatable — infrastructure already half-built.

---

## Part 5 — Sequenced Roadmap

### Phase 1 (Weeks 0-6) — Wire What Already Exists

The cheapest, highest-impact moves. Pure surfacing.

1. **Surface `decision_ledger` as a UI tab.** Filterable, with `prior_state → new_state` diff view.
2. **Wire `ExceptionOrchestrator` to `inv_planning_exceptions.py`.** Exceptions arrive with proposed actions today.
3. **Build a global command bar (Linear-style).** New surface, new backend — replaces the removed NL→SQL `ChatPanel.tsx`.
4. **Add `inapp` channel to `notification_engine.py`** + bell icon. Stop losing conversations to Slack.
5. **Replace one `KpiCard` row with vital-signs cards.** Prove the pattern.
6. **Wire OpenLineage emitter** into `profiled_section()`. Populate `fact_lineage_event`. Add `/lineage/*` endpoint.
7. **Replace `_FoundationStub`** in `model_registry.py` with real Chronos2 loader.

### Phase 2 (Weeks 6-16) — Build Agent Runtime

Refactoring + first agents.

1. **`common/ai/agent_runtime.py`** — single tool-loop runner replacing `api/llm.py` ad-hoc clients.
2. **Decision class registry** + autonomy level resolver in every write endpoint.
3. **Ship ExceptionResolver, ChampionGuardian** as L1-default agents.
4. **Dry-run-by-default** for all agent writes (`commit_token` pattern).
5. **Auto-revert window** + scheduled sweep of L3+ decisions.
6. **Probabilistic forecast collapse** — single distributional table, downstream consumers updated.
7. **Trust-score loop** — extend `ai_recommendation_outcomes` with realized-impact comparison, feed back to autonomy resolver.

### Phase 3 (Weeks 16-30) — Next-Gen UI

The visible transformation.

1. **Morning Briefing home screen** replacing `InvPlanningTab.tsx` landing.
2. **Glass Network** replacing `ControlTowerTab.tsx`.
3. **Exception Inbox** replacing `ExceptionQueuePanel.tsx`.
4. **Cmd-K palette** + sidebar collapse.
5. **WebSocket presence + thread sidebar.**
6. **Conversational config** replacing `SettingsTab.tsx`.
7. **`Why?` and `What if?` drawers** universal.

### Phase 4 (Weeks 30-52) — Moonshots

Pick 2-3 of the ten. Recommended starting set:

1. **Counterfactual Cinema** — highest demo value, leverages existing causal/ledger primitives.
2. **Demand DNA** — 80% already built in `sku_features/`, immediate planner utility.
3. **Live Whisper Stream** — visible external signal integration, builds trust in the data fabric.

---

## Part 6 — Definition of Done for "AI-First"

The transformation is complete when:

- [ ] A planner can spend a workday without opening the sidebar.
- [ ] Every number on screen is right-clickable for lineage and counterfactual.
- [ ] No exception arrives without a proposed action and simulated outcome.
- [ ] Every write to a fact table is logged with `decided_by ∈ {human:<email>, agent:<name>}` + rationale.
- [ ] Every L3+ agent decision has an inverse SQL row and a revert window.
- [ ] Forecasts are distributions everywhere; the word "point forecast" is unsaid.
- [ ] Champion promotions happen without human intervention >70% of the time.
- [ ] The global command bar can answer "what would have happened if..." questions.
- [ ] No external signal source requires a Make target — they all flow as events.
- [ ] No planner needs to screenshot a chart into Slack.
- [ ] The AI Planner tab no longer exists — because the AI is everywhere.

---

*Document generated 2026-04-28 from a 10-agent parallel review. Each finding cites the file where the pattern lives. Implementation starts with Phase 1 — wire what we already paid for.*
