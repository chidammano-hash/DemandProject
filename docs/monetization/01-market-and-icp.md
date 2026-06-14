# Market & ICP — Supply Chain Command Center

> **Audience:** founding team, GTM advisors, prospective design-partner customers.
> **Status:** v1, 2026-05-13. To be revisited after the first three design-partner conversations.
> **Related:** [CLAUDE.md](../../CLAUDE.md), [PLATFORM_GUIDE.md](../PLATFORM_GUIDE.md), [docs/specs/README.md](../specs/README.md), [10-gen4-roadmap.md](../specs/10-gen4-roadmap.md).

---

## 1. Executive Snapshot

**Supply Chain Command Center (SCCC)** is a unified, AI-native demand and inventory planning platform built originally inside a top-three U.S. beverage-alcohol distributor to plan ~317K DFUs and ~300K active SKUs across ~30K open POs for 41 demand planners + 40 replenishment analysts. It compresses what today is split across Blue Yonder Demand, ToolsGroup SO99+, an inventory module, and an S&OP system into one platform: **12-algorithm demand forecasting** (LightGBM/CatBoost/XGBoost, three customer-enriched tree variants, four Amazon Chronos foundation models, MSTL, N-HiTS, N-BEATS, baselines — see [PLATFORM_GUIDE.md](../PLATFORM_GUIDE.md)), per-cluster champion/challenger selection ([common/ml/champion/](../../common/ml/champion/)), 15-sub-feature/34-panel inventory planning ([api/routers/inventory/](../../api/routers/inventory/)), an exception-driven action feed ([exception_engine.py](../../common/engines/exception_engine.py)), an LLM planning agent that writes ranked insights to the database ([ai_planner.py](../../common/ai/ai_planner.py)), and a full S&OP cycle with control tower ([api/routers/operations/](../../api/routers/operations/)). Postgres + FastAPI + React, Docker-deployable, built and operated by a single data-leader-plus-Claude-Code workflow — implying structurally lower build, host, and maintenance cost than tier-1 incumbents.

**Positioning statement (one sentence):**

> *Supply Chain Command Center is the AI-native demand and inventory planning platform for mid-market distributors and B2B wholesalers (US$500M–US$5B revenue, 50,000–500,000 SKUs) who have outgrown spreadsheets and Netstock but cannot justify a US$2–8M Blue Yonder, Kinaxis, or o9 deployment.*

---

## 2. Ideal Customer Profile (ICP)

### 2.1 Primary ICP — Mid-market North American B2B distributors and wholesalers (US$500M–US$5B revenue)

| Attribute | Value |
|---|---|
| **Industry verticals** (in priority order) | Beverage alcohol (beer/wine/spirits distribution), foodservice & broadline food distribution, industrial/MRO distribution, electrical & plumbing wholesale, auto parts & aftermarket, consumer health/beauty distribution. |
| **Revenue band** | US$500M – US$5B annual revenue. |
| **SKU count** | 30,000 – 500,000 active SKUs. |
| **Locations** | 5 – 100 distribution centers / branches. |
| **Customers shipped to** | 5,000 – 250,000 (B2B accounts, not consumers). |
| **Demand planners** | 5 – 75 FTEs in the planning function. |
| **Replenishment analysts / buyers** | 5 – 75 FTEs. |
| **Current planning stack** | One of: (a) **Excel + JDA Manugistics / legacy Demand Solutions / Logility v6**, (b) ToolsGroup SO99+ on a 7-10 year contract that's up for renewal, (c) **SAP APO** they're being forced off (mainstream maintenance ends 2027), (d) **Blue Yonder Demand** that's underused because of TCO. |
| **ERP** | SAP ECC/S4, Oracle EBS/Fusion, Microsoft Dynamics 365, NetSuite, Infor M3/CSI, or industry-specific (Encompass, Provision, ECRS). |
| **Annual planning-system spend (combined license + services)** | US$300K – US$1.5M today. SCCC target ASP US$200K – US$700K all-in. |
| **Buying committee** | VP/Director Supply Chain (economic buyer), Head of Demand Planning (champion), Head of Inventory/Replenishment (champion), CIO/IT (technical gate), CFO (working-capital ROI gate). |

**Why these companies?** Five operational pains the platform solves head-on:

1. **Long tail of intermittent SKUs** breaking the statistical models in Demand Solutions / Logility / SAP APO. Per-cluster routing ([common/ml/champion/](../../common/ml/champion/)) sends >70%-zero clusters to rolling-mean; foundation models ([common/ml/foundation_backtest.py](../../common/ml/foundation_backtest.py)) handle cold-start.
2. **Stockout-biased history.** Bolt Hierarchical ([docs/specs/02-forecasting/20-bolt-hierarchical.md](../specs/02-forecasting/20-bolt-hierarchical.md)) reconstructs true unconstrained demand from customer-grain orders — a problem nearly every B2B distributor has and almost no incumbent solves natively.
3. **Planner overload.** ~7,500 SKUs per planner. The unified action feed ([inv_planning_insights.py](../../api/routers/inventory/inv_planning_insights.py)) + LLM planning agent ([ai_planner.py](../../common/ai/ai_planner.py)) are designed for this scale.
4. **No budget for tier-1.** Blue Yonder/Kinaxis/o9 land at US$1.5–5M Y1 plus US$1M+/yr SI. Mid-market can't fund the integrator team.
5. **Quarterly working-capital pressure.** Capital-investment, cash-flow-timeline, and rebalancing panels ([inv_planning_investment.py](../../api/routers/inventory/inv_planning_investment.py), [inv_planning_rebalancing.py](../../api/routers/inventory/inv_planning_rebalancing.py)) target the CFO conversation directly.

### 2.2 Secondary ICP A — Beverage alcohol distribution (vertical wedge, US$1B+ revenue)

The platform was built inside this vertical, which means the data model already embeds beverage-alcohol artifacts: brand/category/market/channel filters in the global filter bar ([frontend/src/tabs/](../../frontend/src/tabs/)), customer-channel mix analytics ([api/routers/intelligence/customer_analytics/](../../api/routers/intelligence/customer_analytics/)), supplier-pour patterns (sourcing fact ~1.05M rows), and the three-tier (supplier → distributor → retailer/on-premise) demand semantics. Universe: ~50 control-state and open-state distributors in the U.S. (RNDC, Breakthru, Republic National, Empire Merchants, Johnson Brothers, Glazer's Family of Companies, Martignetti, Allied Beverage, etc.) plus ~150 mid-tier independents. **TAM ~$80M–$150M ARR.** The vertical-SaaS narrative ("modern Encompass/Provision replacement for planning") is a credible RFP wedge here even at premium pricing.

### 2.3 Secondary ICP B — North American 3PLs offering managed inventory services (3PL channel)

3PLs increasingly offer "managed inventory" or VMI as a paid service to their shippers. They need a multi-tenant planning platform they can stand up per-customer in days, not months. Today they hack this with Excel templates. The platform's clean container deployment, single-tenant Postgres, and the multi-tenant `tenant_id`/RLS work item already on the Gen-4 roadmap (Wave 3, item #7, [10-gen4-roadmap.md](../specs/10-gen4-roadmap.md)) make this a credible 12-month positioning. Universe: ~50 mid-to-large 3PLs in NA with managed-inventory practices (DHL Supply Chain, Kuehne+Nagel CL, GEODIS, NFI, Saddle Creek, etc.). **TAM ~$120M–$200M ARR.** Caveat: requires the multi-tenancy + RBAC hardening on the roadmap (Wave 3, items #7 and #25) before this is a sellable motion.

---

## 3. TAM / SAM / SOM

All figures are **bands, not point estimates**, derived from public analyst ranges (Gartner Magic Quadrant SCP/SCEP, IDC, ARC Advisory, Lokad benchmarks, vendor 10-Ks for the publics). Treat as Fermi-style sizing for Series A / seed conversations, not as a banker's deck.

### 3.1 North America first

| Tier | Universe | Logic | Range |
|---|---|---|---|
| **TAM** (total addressable supply chain planning software in NA) | All NA companies with US$100M+ revenue and physical inventory: ~22,000 firms. Average planning-software spend at this size band = US$50K – US$1M/yr. | 22,000 × ~US$200K average ≈ US$4.4B. Cross-checks: Gartner sizes the global "supply chain planning" software market at US$7–9B in 2024 with NA ≈ 50%. | **US$4 – 5B ARR** |
| **SAM** (mid-market B2B distributors, wholesalers, foodservice — fits the ICP) | ~3,500 firms in NA with US$500M – US$5B revenue and >30K SKUs. Average willing spend US$300K – US$700K. | 3,500 × ~US$400K ≈ US$1.4B. Cross-check: Logility, ToolsGroup, and Netstock combined NA mid-market planning ARR runs ~US$300–500M; assume that's 25–35% penetrated. | **US$1.2 – 1.6B ARR** |
| **SOM** (5-year capture, single-product motion) | 1.5–3% of SAM in 5 years assuming a competent founder-led GTM, two named verticals, ~30 logos by Year 5. | 1.5–3% × US$1.4B = **US$20 – 45M ARR**. | **US$20 – 45M ARR by Y5** |

### 3.2 Global extension (Years 3-7)

| Tier | Range | Notes |
|---|---|---|
| **Global TAM** (planning software, all geos, all industries) | **US$8 – 12B ARR** | Gartner SCP MQ Q1 2025 + IDC SCM Software Tracker 2024 ranges. |
| **Global SAM** (same ICP filter, EU + UK + ANZ + LATAM + select APAC) | **US$2.5 – 3.5B ARR** | EU adds ~US$700M (EU CSRD/CBAM regulations actually accelerate purchase intent — see [10-gen4-roadmap.md](../specs/10-gen4-roadmap.md) Tier B item #19). |
| **Global SOM** (Y7) | **US$60 – 120M ARR** | Conservative; assumes one EMEA hub by Year 4. |

### 3.3 Vertical wedge — beverage alcohol distribution (NA only)

~200 distributors, ASP US$400K–US$1.2M (consolidated industry → larger players, vertical depth justifies premium): **TAM ~US$80M – US$150M ARR.** A 25–40% capture in this niche over 5 years (~US$25M – US$60M ARR) is a credible aim and creates the proof-points needed to attack adjacent verticals.

---

## 4. Competitive Landscape

| # | Competitor | Sweet spot | Weakness | Where SCCC wins | Where SCCC loses |
|---|---|---|---|---|---|
| 1 | **Blue Yonder (Luminate Demand & Fulfillment)** | F500 retail/CPG, broad SCM suite incl. transportation/warehouse. | TCO US$2–8M Y1, 12–24 mo deployments, dated UX, integrator-dependent. | Mid-market (100% of SAM); time-to-value (weeks vs quarters); planner UX; per-DFU explainability via SHAP ([api/routers/forecasting/shap.py](../../api/routers/forecasting/shap.py)). | Multi-echelon network optimization, transportation, S&OE depth, global enterprise references. |
| 2 | **o9 Solutions** | F500 CPG, hi-tech, automotive — graph-based "Enterprise Knowledge Graph." | Premium pricing US$3–10M, heavy Indian-SI engagement model, narrative-heavy. | Same as Blue Yonder; AI-agent + LLM features ([common/ai/](../../common/ai/)) ship today, not on a roadmap deck. | Knowledge-graph depth, attach-rate marketing, brand. |
| 3 | **Kinaxis (Maestro / RapidResponse)** | F500 hi-tech, pharma, automotive — concurrent-planning engine. | High TCO, batch-window-thinking re-architected as concurrent — still feels like an MRP descendant; weak on AI/forecasting, strong on supply response. | AI-native forecasting; mid-market price; 12-algorithm competition that Maestro lacks. | Concurrent supply-response engine, manufacturing/MPS depth, capable-to-promise. |
| 4 | **Anaplan** | Mid-to-large enterprise S&OP / IBP, financial planning — connected planning. | Hub-spoke modeling is slow at high cardinality; not a forecasting engine; calculated columns choke at SKU-level. | True ML forecasting (Anaplan has none native); SKU-level scale (Anaplan struggles >100K dimensionality); inventory math out of the box. | C-level "connected planning" narrative, FP&A integration, partner ecosystem. |
| 5 | **RELEX** | Grocery, food-retail, fashion — replenishment optimization at store level. | Vertical-locked retail; weak outside grocery; commercial structure assumes large scale. | Distributor verticals (B2B) where RELEX is weak; lower TCO; broader S&OP scope. | Grocery retail proper (lose every time); fresh/perishable depth; promotion modeling. |
| 6 | **ToolsGroup (SO99+)** | Mid-market inventory optimization — "demand sensing" + probabilistic. | Aging UX, poor planner-trust workflows, weak on LLM/AI agents, customers are restless at renewal. | Modern UX; explainability; champion/challenger across 12 algorithms vs SO99+ single proprietary engine; LLM planner agent; the same probabilistic chain ([common/ml/crps.py](../../common/ml/crps.py), [common/ml/cold_start_neighbors.py](../../common/ml/cold_start_neighbors.py)) plus a UI that mid-market planners actually use. | 30-year track record, MEIO Clark-Scarf depth, deeper math literature. |
| 7 | **Logility (Demand Optimization, Voyager)** | Long-tail mid-market manufacturing/distribution; Aptean now owns. | Stagnant product velocity post-acquisition; legacy UX; aggressive price-then-discount; minimal AI. | Product velocity (we ship monthly); LLM/AI; explainability; cleaner codebase. | Long-running customer references, big channel/SI footprint. |
| 8 | **SAP IBP** | SAP-native shops; integrated with S/4HANA. | Notoriously slow planning runs, deep SAP-consultant requirement, datasets locked behind SAP Datasphere/HANA, slow innovation. | Anyone running SAP S/4 + frustrated with IBP performance/UX (large segment); price; AI features. | SAP-mandated shops where IBP is the political answer; SAP partnership strength. |
| 9 | **Oracle SCP / Demantra** | Oracle-native shops; suite play. | Demantra is end-of-innovation; new "Oracle Fusion SCM Planning" is greenfield rebuild and incomplete. | Oracle shops looking outside the suite; faster product velocity; modern AI. | Pure Oracle shops; suite discounting. |
| 10 | **Netstock / NETSTOCK** | SMB (US$10M – US$200M revenue), QuickBooks/NetSuite-integrated inventory optimization. | Hits a ceiling at ~30K SKUs and ~5 planners; thin demand-planning depth; limited customization. | Above the Netstock ceiling — exactly the "upgraded from Netstock" customer is our sweet spot; depth in forecasting + S&OP. | Below US$200M revenue (we're overkill); pre-built ERP connectors. |
| 11 | **Slimstock / GAINSystems / Onebeat** | EU mid-market inventory; MEIO niche; retail-allocation niche. | Narrow scope; aging products; thin AI. | Broader unified scope; modern AI/UX. | Depth in their specific niche or geography. |

**Tier-1 conclusion:** SCCC does not compete with Blue Yonder/Kinaxis/o9/SAP IBP head-on for F500 deals — yet. It competes for the **mid-market customers those vendors over-sell to and under-serve**, and for the **renewal cycles of ToolsGroup, Logility, and SAP APO refugees**.

---

## 5. Differentiators

Five evidence-backed differentiators, ranked by defensibility:

### 5.1 Twelve-algorithm champion/challenger with per-cluster routing

Most platforms ship one engine (ToolsGroup), two or three (Blue Yonder, Logility), or "bring your own model" with no orchestration (Anaplan). SCCC ships **12 algorithms governed by one config** ([config/forecasting/forecast_pipeline_config.yaml](../../config/forecasting/forecast_pipeline_config.yaml)) — three tree models, three customer-enriched tree variants (34 customer-derived features), four Amazon Chronos foundation models, MSTL, N-HiTS + N-BEATS deep-learning, plus baselines — competing per DFU, per timeframe, with **8 champion-selection strategies** ([common/ml/champion/](../../common/ml/champion/), 31 strategies across 9 modules) including a meta-learner. Per-cluster routing sends intermittent (>70% zero) demand to a rolling-mean baseline, sparse clusters skip SHAP, foundation models handle cold-start. **No publicly-marketed competitor at our price point ships this matrix.**

### 5.2 SHAP-based per-cluster explainability

Per-cluster SHAP feature selection ([common/ml/shap_selector.py](../../common/ml/shap_selector.py), [api/routers/forecasting/shap.py](../../api/routers/forecasting/shap.py)) with stratified sampling for sparse clusters, a per-DFU on-demand SHAP endpoint, and a protected-features list that survives all filter stages. UI panels in Accuracy and Item Analysis tabs. Addresses the #1 RFP question — *"explain this forecast"* — that Blue Yonder/o9 answer with keynote charts and SCCC answers with 4 endpoints.

### 5.3 LLM planning agent + AI tuning advisor — proactive, not chatbot

Two AI features shipping in SCCC that competitors carry only on roadmap decks:

1. **AI Planning Agent** ([common/ai/ai_planner.py](../../common/ai/ai_planner.py)) — Claude / GPT proactive exception work-queue. 10 tools (9 SQL lookups + `create_insight`), circuit-breaker guards (40 turns, 100K tokens per scan), structured insight cards with severity, financial impact, causal-chain reasoning. Async portfolio scans write to `ai_insights`; observability via `ai_call_log`. **Batch agent, not chatbot** — produces a ranked queue planners work.
2. **AI Tuning Advisor** ([common/ai/tuning_advisor.py](../../common/ai/tuning_advisor.py)) — agentic chat in the LGBM Tuning tab; reviews prior runs, recommends parameter changes via structured cards, and (with confirmation) launches new backtests.

Gen-4 roadmap Wave 1 ([10-gen4-roadmap.md](../specs/10-gen4-roadmap.md)) extends this with decision ledger (Merkle-anchored), policy engine, RAG memory, and reversible-action orchestrator already scaffolded in [common/ai/](../../common/ai/) — **~18 months of head-start** on the agentic-SC narrative tier-1s now sell on slides.

### 5.4 Bolt Hierarchical — bottom-up customer-grain forecasting that corrects stockout bias (**green-field**)

Almost every B2B distributor's history is biased: recorded "sales" = demand − stockouts, so tree models under-forecast frequent stockouts and trigger more stockouts. **Bolt Hierarchical** ([docs/specs/02-forecasting/20-bolt-hierarchical.md](../specs/02-forecasting/20-bolt-hierarchical.md)) runs Chronos Bolt at the customer × item × location grain using true demand from `fact_customer_demand_monthly`, aggregates bottom-up to item × location, reconciles with top-down Bolt (weighted average → MinTrace shrinkage), and maps to DFU grain for champion competition. **No mid-market competitor solves this; tier-1s either ignore it or solve only top-down.**

### 5.5 Unified forecast + inventory + S&OP + AI in one product, sub-US$1M ASP

A mid-market distributor today needs 3–4 products to match SCCC's scope: demand planning (ToolsGroup/Logility, ~US$200–500K/yr), inventory optimization (Netstock/Slimstock/SO99+, ~US$150–400K/yr), S&OP (Anaplan/Board/spreadsheets, ~US$200–600K/yr), AI overlay (bespoke or none). Combined ~US$700K–US$2M/yr plus US$300K–US$1M integration. SCCC at US$200–700K all-in is a **50–70% TCO reduction**. The unified data model ([sql/](../../sql/) — 88 DDL files, 81 tables) is the moat: no cross-product integration tax.

---

## 6. Vulnerabilities (where SCCC is weak vs incumbents)

Honest list — these gate certain deals today and frame the build-vs-defer roadmap. Most are tracked on the Gen-4 roadmap ([10-gen4-roadmap.md](../specs/10-gen4-roadmap.md)).

| # | Gap | Severity for primary ICP | Severity for enterprise | Roadmap status |
|---|---|---|---|---|
| 1 | **No multi-tenancy.** Single-tenant Postgres only. | Low (SaaS-per-customer is fine for primary ICP). | High (3PLs, MSPs need RLS). | Wave 3 item #7 — XL effort. |
| 2 | **No enterprise SSO / SAML / SCIM.** RBAC exists ([api/routers/platform/auth_router.py](../../api/routers/platform/auth_router.py)) but only JWT. | Medium. | Critical (table-stakes RFP gate). | Wave 3 item #25 — combine with SOC2 Type II. |
| 3 | **No SOC2 Type II / ISO 27001 / HIPAA.** | Medium. | Critical. | Wave 3 item #25. |
| 4 | **No transportation, no warehouse, no network optimization.** Pure planning, no execution. | Acceptable (out of scope, partner-friendly). | Critical (Blue Yonder / Manhattan / Oracle ship the suite). | Out of scope by design — keep partnership posture. |
| 5 | **No production scheduling, MPS, BOM, capacity, changeover.** | Low (distributors don't manufacture). | High (manufacturing customers need this). | Wave 2 item #9 — XL, gates discrete/process manufacturing TAM. |
| 6 | **No EDI 850/855/856, no AS2, no order management / ATP / CTP.** "Analytics, not execution." | Medium-high — RFP-DQ for retail/grocery. | Critical. | Wave 3 item #6 — XL, the 3-5× ASP move. |
| 7 | **No lot/serial/FEFO traceability (no DSCSA).** | Critical for pharma / food / cannabis (FSMA Rule 204 enforcement Jan 2026). | Critical. | Wave 2 item #1 — top-ranked Borda 39 across 9 judges. |
| 8 | **No native MEIO Clark-Scarf / GST.** Two-echelon SS exists ([api/routers/operations/echelon_planning.py](../../api/routers/operations/echelon_planning.py)) but not full stochastic-DP. | Medium. | High (RFP credibility vs ToolsGroup). | Wave 3 item #23 — XL. |
| 9 | **B2B distribution bias in the data model and UX.** Brand/category/channel/market filters are a beverage-alcohol shape. | Strength for primary ICP, weakness elsewhere. | High for true vertical-mismatch. | Generalize labels; mostly a content/branding fix. |
| 10 | **Single-instance Postgres scale.** Async pool + read-replica routing ([api/pool.py](../../api/pool.py)) is in pilot; pg-queue scaffold ([common/services/pg_queue.py](../../common/services/pg_queue.py)) for long jobs landed. Tested at ~198M rows / ~112K DFUs. | Acceptable. | Will require sharding above ~1M DFUs. | Active — Pass-2 perf sweep landed sql/170-185. |
| 11 | **No global / multi-currency / multi-timezone / FX.** Time dim is calendar-only. | Acceptable. | Critical for global enterprise. | Future — gate global-enterprise expansion on this. |
| 12 | **No formal customer success org, no implementation methodology, no certified partners.** Built and operated by one data leader. | High (mid-market customers expect a runbook). | Critical. | GTM, not product — gate the first 3 deals on white-glove design-partnership. |
| 13 | **No mobile, no supplier-facing portal.** Supplier KPIs exist ([inv_planning_supplier.py](../../api/routers/inventory/inv_planning_supplier.py)) but no external UI. | Low–Medium. | High. | Future. |
| 14 | **Limited public benchmarks / case studies.** Internal SGWS use is unsold publicly; no external WAPE benchmarks vs ToolsGroup/Blue Yonder. | High (RFP-credibility). | Critical. | GTM — fix with first 3 design partners. |

---

## 7. Positioning Options

Three distinct positioning plays, each with a different ICP, message, win-thesis, and required investment.

### 7.1 Option A — "Modern alternative to ToolsGroup / Logility for mid-market distributors"

| Dimension | Detail |
|---|---|
| **Target ICP** | Primary ICP exactly (US$500M – US$5B distributors), pre-qualified by ToolsGroup / Logility / Demand Solutions / SAP APO incumbency at renewal year 5–10. |
| **Message** | *"Half the TCO, twice the algorithms, an AI agent that triages your exception queue every morning — and you can stand it up in 90 days, not 18 months."* |
| **Why it could win** | Differentiators 5.1 + 5.5 (12-algorithm + unified TCO) directly address the specific pain that drives renewal-defection. Pipe is large (~3,500 SAM firms in NA); a non-trivial fraction renews ToolsGroup/Logility every year. |
| **What it requires** | (a) 3 reference customers with public ROI numbers (working capital reduction + WAPE delta vs incumbent). (b) An "RFP-pack": 30 stock answers to the 30 standard questions ToolsGroup-incumbent RFPs ask. (c) A 90-day implementation methodology with a partner or in-house CS. (d) SOC2 Type II within 18 months. |
| **Risk** | Mid-market RFPs are won on references and risk, not features. Without 3 named logos, every deal is a custom POC. |

### 7.2 Option B — "AI-native demand planner copilot — turn your 10 planners into 30"

| Dimension | Detail |
|---|---|
| **Target ICP** | Same primary ICP, but enter via the *VP Supply Chain* at companies that explicitly want to "do something with AI" and have a 6-month proof-of-value budget (US$50K–US$150K). |
| **Message** | *"Your planners spend 60% of their time chasing exceptions. Our AI agent — built on Claude/GPT — pre-triages your portfolio every morning, ranks the top 200 SKUs that need attention, explains why, and suggests an action. Your planners do 3× the work in the same hours."* |
| **Why it could win** | Differentiators 5.3 (LLM agent) + 5.2 (SHAP explainability) hit a *narrative* a CIO/COO is actively looking to fund right now. Decision-cycle is faster (innovation budget, not core-systems renewal). Lower deal size, faster path to first 5 logos. |
| **What it requires** | (a) Sharper "agent productized" packaging — today the agent ships as part of the platform; needs to be a top-of-fold story. (b) Land-and-expand motion: AI copilot lands at US$50–150K, expands to full platform at US$300–600K. (c) Decision-ledger + policy-engine maturity (Wave 1 item #12, [10-gen4-roadmap.md](../specs/10-gen4-roadmap.md)) to satisfy the "can we trust the agent?" question. (d) Real before/after planner-productivity metrics from the SGWS deployment. |
| **Risk** | "AI copilot" is a crowded narrative — every BY/o9/Kinaxis/Anaplan deck has an agent slide. Differentiation has to be *demonstrable* (live demo on prospect data), not narrative. |

### 7.3 Option C — "Vertical SaaS for beverage alcohol distribution"

| Dimension | Detail |
|---|---|
| **Target ICP** | Secondary ICP A — beverage alcohol distributors only. ~50 control/open-state distributors plus ~150 mid-tier independents in the US. |
| **Message** | *"Built inside [SGWS]. Knows beverage-alcohol — three-tier system, brand/market/channel mix, supplier pours, control-state nuances, allocation, vintage. You can be on it in 60 days because we already speak your language."* |
| **Why it could win** | Vertical-SaaS plays consistently outperform horizontal plays at Series A — narrower ICP, sharper messaging, references compound. The platform's data shape is *already* beverage-alcohol-flavored. Insider credibility via the founder. Premium ASP justified (US$400K – US$1.2M). |
| **What it requires** | (a) Founder has to lean into the "former SGWS data leader" narrative publicly. (b) NDA-clean version of the SGWS use case as a co-developed reference. (c) Beverage-alcohol-specific modules: allocation engine, VAP (value-added-package) modeling, depletion forecasting (sell-through to retailer), three-tier compliance reporting. (d) Vertical-conference presence (Wine Industry Tech, NABCA, WSWA). |
| **Risk** | Vertical lock — TAM is bounded at US$80–150M ARR. Hard to pivot if vertical contracts (consolidation in beverage alcohol is real — Glazer's + Southern → SGWS, RNDC + Young's → RNDC, etc.). Some distributors will refuse to use software touched by a competitor (SGWS connection). |

---

## 8. Recommended Primary Positioning

**Recommendation: Option A — "Modern alternative to ToolsGroup / Logility for mid-market distributors" — anchored by Option C as a vertical wedge in Year 1.**

Rationale:

1. **Market sizing** — Option A addresses a US$1.2–1.6B SAM. Option B is a feature in everyone's deck and gets commoditized into the larger platforms. Option C is a credible US$80–150M wedge.
2. **Sales motion fit** — Mid-market planning RFPs are won on **TCO + risk + references**, not narrative. SCCC's actual product strength is exactly that: a unified, lower-TCO, modern alternative. Option A maps the message to the buying motion.
3. **Founder fit** — A career data leader with deep beverage-alcohol context can credibly walk into a beverage-alcohol distributor's CIO office and run a 60-day pilot. That's Option C as a *wedge*, not as the main product positioning. Use the first three logos in beverage alcohol, then expand laterally to foodservice, broadline distribution, and industrial wholesale (the verticals with the most architectural similarity).
4. **AI is the supporting actor, not the headline** — The LLM planning agent (5.3) and per-DFU explainability (5.2) become the *proof points* inside the Option-A pitch, not the lead message. This avoids commoditization while still capturing the AI-budget zeitgeist when it comes up.

**12-month execution sequence:**

1. **Q1–Q2:** Land 2 design partners in beverage-alcohol distribution at US$100–200K. Co-develop public ROI (WAPE delta, working-capital release, planner-productivity). Close gating product gaps (SOC2 readiness, implementation methodology, RBAC hardening).
2. **Q3:** First 3 logos public. Ship the "modern alternative to ToolsGroup/Logility" RFP-pack and 90-day implementation methodology. Recruit one mid-tier distribution-specialist SI.
3. **Q4:** Expand to foodservice, industrial, electrical, plumbing distributors. Target 5 net-new logos. ARR run-rate aim: US$1.5–3M.
4. **Year 2:** Wave 3 (multi-tenant + SOC2) unlocks 3PL channel + enterprise pilots. EU exploration when CSRD/CBAM pull is qualified (roadmap item #19).

**The single biggest commercial risk to manage:** the platform is currently *one operator's deployment*. Until there are 3 public reference customers with ROI numbers, every prospect conversation devolves into a custom POC. The first 6 months of GTM should be exclusively about converting design-partner deployments into public proof. Everything else (multi-tenancy, SOC2, EDI, MEIO) is a second-order priority that becomes urgent only after the references exist.

---

## Appendix — Sources & Logic for Sizing

- **Gartner MQ for SCP Solutions** (Q1 2025) and **IDC SCM Software Tracker** (2024) — converge on US$7–12B global SCP software market, NA ≈ 50% share.
- **ARC Advisory SCP Market 2024** + public **10-Ks** (Kinaxis 2024 ~US$483M / ~30% growth, Manhattan SCM segment, Aptean/Logility commentary) — used to cross-check SAM.
- **Lokad** market posts and **John Galt** blog — mid-market vendor pricing benchmarks.
- **US Census 2022 Wholesale Trade** (NAICS 423/424) — ~22,000 firms US$100M+, ~3,500 in the US$500M–US$5B band.
- **WSWA + NABCA** directories — beverage-alcohol vertical universe.
- ASP figures are anecdotal ranges; triangulate against design-partner conversations before fundraising-deck use.
