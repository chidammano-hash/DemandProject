# Pricing & Packaging — Supply Chain Command Center

> Audience: CEO, Head of Sales, Head of Finance.
> Purpose: agree the commercial model before we open a single sales conversation.
> Decision asked: approve the recommended hybrid pricing model, the three-edition lineup, and the published list prices below.

This document covers what we sell, how we charge for it, and what a deal looks like at three customer sizes. It is grounded in the modules actually built in this repository — see [docs/specs/README.md](../specs/README.md) for the full feature index and [docs/ARCHITECTURE.md](../ARCHITECTURE.md) for scale numbers.

---

## 1. Pricing philosophy

There are three honest options for a forecasting/inventory platform:

| Approach | What it anchors to | When it wins |
|---|---|---|
| **Cost-plus** | Our infra + labor + target margin | Never — kills enterprise pricing; floors us at ~$30k ACV |
| **Competitor-anchored** | List prices of o9, Blue Yonder, RELEX | Useful as a *ceiling check*, but those vendors price on legacy assumptions; copying them caps our growth |
| **Value-based** | A share of measurable customer P&L impact (inventory release, OOS recovery, planner productivity) | Wins when we can compute customer value in a discovery call and defend it in procurement |

**Recommendation: value-based, with a competitor sanity check.** SC buyers — VPs of Supply Chain and CFOs — evaluate ROI in absolute dollars, not seat counts. The product was built inside a multi-billion-dollar distributor against a real P&L: 41 demand planners, 40 replenishment analysts, 317K DFUs, 300K SKUs. Price as if we are claiming a slice of measurable value, not renting a dashboard. Cost-plus is a non-starter — hosting + ops for a mid-market tenant is < $80K/yr, which would floor us at a fraction of fair value.

---

## 2. Value drivers — five P&L levers a CFO can verify

Every sales motion must lead with one of these five formulas. They are the only numbers that survive procurement.

| # | Driver | Formula a CFO can audit | Where the product earns it |
|---|---|---|---|
| **V1** | Forecast accuracy lift | `(WAPE_before − WAPE_after) × Annual_COGS × 0.5` (each WAPE point ≈ 0.5% of COGS in carrying + lost margin) | [02-forecasting/](../specs/02-forecasting/) — 12-algorithm roster (LGBM/CatBoost/XGBoost + Chronos T5/Bolt/Bolt-Hierarchical/Chronos2/2E + MSTL + N-HiTS + N-BEATS + baselines), per-cluster SHAP, champion meta-learner. Internal benchmark: LGBM moved from 59% to 68% accuracy. |
| **V2** | Inventory reduction (working capital release) | `(DOS_before − DOS_after) / 365 × Annual_COGS × WACC` | [04-inventory/03-safety-stock.md](../specs/04-inventory/03-safety-stock.md), [04-inventory/04-replenishment.md](../specs/04-inventory/04-replenishment.md), [04-inventory/08-investment.md](../specs/04-inventory/08-investment.md), [04-inventory/11-rebalancing.md](../specs/04-inventory/11-rebalancing.md). Z-score safety stock + Monte Carlo + efficient-frontier budget allocator + cross-location rebalancing. |
| **V3** | OOS revenue protection | `OOS_units_avoided × ASP × Gross_margin%` | [04-inventory/05-exception-queue.md](../specs/04-inventory/05-exception-queue.md), [04-inventory/06-analytics.md](../specs/04-inventory/06-analytics.md) (intramonth stockout detection, fill-rate analytics, demand signals). |
| **V4** | Planner productivity | `Planners × Hours_saved_per_week × 50 × Loaded_rate` | [06-ai-platform/01-ai-planning-agent.md](../specs/06-ai-platform/01-ai-planning-agent.md) (Claude-driven exception triage), [06-ai-platform/03-control-tower.md](../specs/06-ai-platform/03-control-tower.md), unified Action Feed. Internal benchmark: 41 demand planners + 40 replenishment analysts now run with the action feed surfacing pre-ranked work — 6–10 hours/week recovered per planner is conservative. |
| **V5** | Planning-stack consolidation | `Σ(retired tools' ACV) − our_ACV` (typical retire list: standalone forecasting tool, separate inventory optimizer, S&OP spreadsheet sprawl, BI license overage) | One platform covers forecasting + inventory + S&OP + customer analytics + AI agent + control tower. |

**Order of pitch.** Lead with V2 (inventory release) for CFOs — biggest, fastest to prove, least disputable. Lead with V4 (planner productivity) for VPs of Supply Chain. Use V1 as the technical proof point that V2 is real. V3 and V5 close the deal.

---

## 3. Pricing models considered

| # | Model | Pros | Cons | Who uses it |
|---|---|---|---|---|
| **M1** | **Per-SKU / per-DFU** (e.g., $0.05–$0.50 / SKU / month) | Scales with customer size; easy to defend ("you have more SKUs, you get more value"); aligns infra cost | Becomes hostile at scale (>1M SKUs); customers game it by aggregating SKUs upstream; punishes long-tail catalogs | RELEX, ToolsGroup, Netstock |
| **M2** | **Per-user / per-seat** | Familiar to procurement; predictable ACV; easy to model | Doesn't capture value of automation (the AI agent literally replaces seats); incentivizes the customer to under-license | Anaplan, some Kinaxis editions |
| **M3** | **Per-module / a-la-carte** | High deal flexibility; clean upsell path | Hard to package; sales conversations get stuck on line items; champion fatigue | Blue Yonder (legacy JDA pricing), o9 |
| **M4** | **Tiered by customer revenue** (e.g., 0.02–0.08% of revenue) | Anchors directly to customer scale and ability to pay; very simple | Customers resist sharing revenue; opaque on what they're getting | Some boutique consulting-ware (rare in pure SaaS) |
| **M5** | **Pure consumption** (per forecast run, per AI insight, per API call) | Aligns with cloud cost; grows with usage | Unpredictable bills kill enterprise procurement; finance hates it | Snowflake, OpenAI — *not* SC software |
| **M6** | **Hybrid: platform fee + scale meter + add-on modules** | Platform fee gives revenue floor; scale meter captures fairness; add-ons drive expansion | More complex to quote; needs a deal desk | o9 (effectively), modern SaaS (Datadog, MongoDB Atlas) |

---

## 4. Recommended model — **Hybrid (M6)**

**Pricing formula:**

```
Annual Contract Value = Edition Platform Fee
                      + (Active SKUs / 10,000) × per-10K-SKU rate
                      + Σ Add-On Module Fees
                      + Implementation (one-time)
```

**Why this wins:** the platform fee anchors the deal floor and tells sales which conversation to have. The per-10K-SKU meter scales fairly without the nickel-and-dime feel of per-SKU. Add-ons drive expansion without re-opening the master contract. Customers can model their bill on one page. Procurement signs.

### Worked examples (annual list, before discount)

| Customer profile | Edition | Platform fee | SKU meter | Typical add-ons | **Year-1 ACV** |
|---|---|---|---|---|---|
| **Small** — $50M revenue, 10K SKUs, 5 planners, 1 distribution center | Starter | $48,000 | 1 × $4,000 = $4,000 | none | **$52,000** |
| **Mid** — $500M revenue, 75K SKUs, 20 planners, 5 DCs | Professional | $180,000 | 7.5 × $3,500 = $26,250 | AI Copilot ($45,000) + Customer Analytics ($30,000) | **$281,250** |
| **Large** — $5B revenue, 500K SKUs, 80 planners, 25 DCs | Enterprise | $480,000 | 50 × $2,800 = $140,000 | AI Copilot ($120,000) + Foundation Models ($90,000) + Executive S&OP ($75,000) + Customer Analytics ($60,000) | **$965,000** |

SKU-meter rate decays with scale (volume discount baked into the rack rate, not negotiated case-by-case). Platform fees are the "what you are buying" anchor; the meter is the "how big you are" anchor.

**Cross-check vs. value drivers at the Mid customer** ($300M COGS): V2 = $329K/yr (5 days of inventory at 8% WACC). V4 = $450K/yr (20 planners × 6 hrs/wk × $75 loaded). V1 = $450K/yr (3 WAPE points × 0.5%). Combined defensible value ≈ **$1.2M** against $281K ACV = **4.3× ROI in year one**. That is the deal we sell.

---

## 5. Packaging — three editions

Every edition gives the customer the same data foundation ([01-foundation/](../specs/01-foundation/)) and the same React UI ([07-user-experience/02-ui-architecture.md](../specs/07-user-experience/02-ui-architecture.md)). Editions differ on which **modules** are turned on and which **add-ons** are eligible.

### Starter — $48,000 / yr platform + SKU meter
**Target ICP:** $25M–$150M revenue distributors / mid-market CPG / regional manufacturers. ≤ 25K SKUs. ≤ 10 planners. Single warehouse or 2–3 DCs.

**Included modules:**
- Demand Forecasting — tree models only (LGBM, CatBoost, XGBoost) — [02-forecasting/04-tree-models.md](../specs/02-forecasting/04-tree-models.md)
- Champion selection — basic strategies (expanding, rolling, decay) — [02-forecasting/07-champion-selection.md](../specs/02-forecasting/07-champion-selection.md)
- Production forecast generation — [02-forecasting/08-production-forecast.md](../specs/02-forecasting/08-production-forecast.md)
- Inventory Planning core — Safety Stock, EOQ, Replenishment policies, Exception queue, Fill rate — [04-inventory/03-safety-stock.md](../specs/04-inventory/03-safety-stock.md), [04-inventory/04-replenishment.md](../specs/04-inventory/04-replenishment.md), [04-inventory/05-exception-queue.md](../specs/04-inventory/05-exception-queue.md), [04-inventory/06-analytics.md](../specs/04-inventory/06-analytics.md)
- ABC-XYZ segmentation — [04-inventory/07-abc-xyz-supplier.md](../specs/04-inventory/07-abc-xyz-supplier.md)
- Control Tower (read-only) — [06-ai-platform/03-control-tower.md](../specs/06-ai-platform/03-control-tower.md)
- Standard Data Quality engine — [01-foundation/03-data-quality.md](../specs/01-foundation/03-data-quality.md)
- Email + Slack notifications — [08-integration/04-notifications.md](../specs/08-integration/04-notifications.md)

**Excluded:** Foundation models, AI Planning Agent, Customer Analytics, S&OP cycle, Multi-Echelon, Scenario Planning, Webhooks, Read-replica.

**Upgrade trigger to Professional:** customer crosses 25K SKUs, hires a third planner, or asks for "an AI assistant."

---

### Professional — $180,000 / yr platform + SKU meter
**Target ICP:** $150M–$1.5B revenue. 25K–150K SKUs. 10–40 planners. Multi-DC. Has a named S&OP process. This is the **center of gravity** — most of the pipeline lands here.

**Everything in Starter, plus:**
- Champion advanced strategies — meta-learner, hybrid_warmup, adaptive_ensemble, ensemble_rolling — [02-forecasting/07-champion-selection.md](../specs/02-forecasting/07-champion-selection.md)
- Per-cluster SHAP feature selection + per-cluster tuning profiles — [02-forecasting/23-lgbm-accuracy-tuning.md](../specs/02-forecasting/23-lgbm-accuracy-tuning.md)
- Unified Model Tuning Studio — [02-forecasting/11-unified-model-tuning-v2.md](../specs/02-forecasting/11-unified-model-tuning-v2.md)
- SKU Clustering & Experimentation Studio — [03-demand-intelligence/01-sku-clustering.md](../specs/03-demand-intelligence/01-sku-clustering.md)
- Champion Experimentation Studio — [03-demand-intelligence/05-champion-experimentation-studio.md](../specs/03-demand-intelligence/05-champion-experimentation-studio.md)
- Forecast CI bands + bias correction — [02-forecasting/09-bias-correction.md](../specs/02-forecasting/09-bias-correction.md), [02-forecasting/10-forecast-ci-bands.md](../specs/02-forecasting/10-forecast-ci-bands.md)
- Multi-Echelon safety stock + Inventory Rebalancing — [04-inventory/09-multi-echelon.md](../specs/04-inventory/09-multi-echelon.md), [04-inventory/11-rebalancing.md](../specs/04-inventory/11-rebalancing.md)
- Capital Investment Optimizer (efficient frontier) — [04-inventory/08-investment.md](../specs/04-inventory/08-investment.md)
- S&OP Cycle (six-stage) — [05-operations/01-sop-cycle.md](../specs/05-operations/01-sop-cycle.md)
- Event Calendar + Financial Planning — [05-operations/02-financial-planning.md](../specs/05-operations/02-financial-planning.md), [05-operations/03-event-calendar.md](../specs/05-operations/03-event-calendar.md)
- Storyboard with causal chains — [06-ai-platform/04-storyboard.md](../specs/06-ai-platform/04-storyboard.md)
- FVA tracking — [08-integration/07-fva.md](../specs/08-integration/07-fva.md)
- Webhooks + Teams + PagerDuty — [08-integration/10-webhooks.md](../specs/08-integration/10-webhooks.md)
- Standard support (8×5, 8h response)

**Excluded:** Foundation-model forecasting (Chronos T5/Bolt/2/2E + Bolt-Hierarchical), AI Planning Agent + Market Intel, Customer Analytics, Scenario Planning, white-label, premium support.

**Upgrade trigger to Enterprise:** customer requests foundation-model forecasting, hires its first analytics-ops team, asks for a customer-level forecast view, or requires 24×7 support.

---

### Enterprise — $480,000 / yr platform + SKU meter
**Target ICP:** $1B+ revenue. 150K+ SKUs. 40+ planners. Multi-region. Has a Chief Supply Chain Officer. Wants the AI agent in production, not as a demo.

**Everything in Professional, plus:**
- Foundation Model forecasting — Chronos T5, Chronos Bolt, Chronos 2, Chronos 2 Enriched, Bolt Hierarchical with customer-level reconciliation — [02-forecasting/18-chronos-foundation-models.md](../specs/02-forecasting/18-chronos-foundation-models.md), [02-forecasting/20-bolt-hierarchical.md](../specs/02-forecasting/20-bolt-hierarchical.md)
- Customer-Enriched tree models (34 customer features) — [02-forecasting/21-customer-enriched-features.md](../specs/02-forecasting/21-customer-enriched-features.md)
- Deep learning + statistical models (N-HiTS, N-BEATS, MSTL)
- AI Planning Agent (Claude-powered, full tool-use loop) — [06-ai-platform/01-ai-planning-agent.md](../specs/06-ai-platform/01-ai-planning-agent.md)
- Market Intelligence (Google + GPT-4o briefings) — [06-ai-platform/02-market-intel.md](../specs/06-ai-platform/02-market-intel.md)
- Decision Ledger & Policy — [06-ai-platform/05-decision-ledger-and-policy.md](../specs/06-ai-platform/05-decision-ledger-and-policy.md)
- Scenario Planning (disruption simulation) — [05-operations/04-scenario-planning.md](../specs/05-operations/04-scenario-planning.md)
- Customer Analytics (demand-aware customer map, OOS hotspots, channel mix, concentration) — [03-demand-intelligence/07-customer-analytics.md](../specs/03-demand-intelligence/07-customer-analytics.md), [03-demand-intelligence/06-demand-history-workbench.md](../specs/03-demand-intelligence/06-demand-history-workbench.md)
- External signals integration (weather, economic indicators, POS) — [08-integration/06-external-signals.md](../specs/08-integration/06-external-signals.md)
- Read-replica routing for analytics workloads — see [docs/RUNBOOK.md](../RUNBOOK.md) "Read Replica Deployment"
- Async pool + Redis cache + pg-queue worker for >250K SKU scale
- API Governance + per-role rate limiting — [08-integration/09-api-governance.md](../specs/08-integration/09-api-governance.md)
- RBAC with full audit log — [08-integration/02-rbac.md](../specs/08-integration/02-rbac.md)
- Premium support (24×7, 1h Sev-1, named CSM, quarterly executive review)

---

### Edition comparison

| Capability | Starter | Professional | Enterprise |
|---|---|---|---|
| Tree forecasting (LGBM/CatBoost/XGBoost) | ✓ | ✓ | ✓ |
| Per-cluster SHAP + tuning profiles | — | ✓ | ✓ |
| Foundation models (Chronos / Bolt-Hierarchical) | — | — | ✓ |
| Deep learning + statistical (N-HiTS / N-BEATS / MSTL) | — | — | ✓ |
| Customer-enriched features | — | — | ✓ |
| Inventory core (SS / EOQ / policies / exceptions) | ✓ | ✓ | ✓ |
| Multi-echelon + rebalancing + budget optimizer | — | ✓ | ✓ |
| S&OP cycle + event calendar + financial planning | — | ✓ | ✓ |
| Scenario planning | — | — | ✓ |
| Customer Analytics | — | — | ✓ |
| AI Planning Agent + Market Intel | — | — | ✓ |
| Webhooks + external signals | — | Webhooks only | ✓ |
| Read-replica + async pool | — | — | ✓ |
| Support SLA | 8×5 / next BD | 8×5 / 8h | 24×7 / 1h |
| Max SKUs (soft cap) | 25K | 150K | unlimited |
| Max users | 10 | 40 | unlimited |

---

## 6. Add-ons

Add-ons are sold to Professional and (where noted) Starter customers. They are the engine of expansion revenue. List prices below; all are annual.

| SKU | What it is | Eligible editions | List price |
|---|---|---|---|
| **AI Copilot Pack** | AI Planning Agent + Market Intel + Storyboard causal chains + AI Tuning Chat | Professional, Enterprise (included) | $45K (Pro) — adds usage cap of 10K Claude calls/mo |
| **Foundation Model Forecasting** | Chronos T5 / Bolt / 2 / 2E + Bolt Hierarchical | Professional | $90K |
| **Executive S&OP** | Six-stage S&OP cycle + scenario planning + executive review pack | Starter, Professional | $30K (Starter) / $75K (Pro) |
| **Customer Analytics** | Demand-aware customer map, demand history workbench, customer-enriched forecasting | Professional | $30K–$60K (tiered by customer count) |
| **External Signals** | POS + weather + economic indicators ingestion | Professional, Enterprise | $24K |
| **White-Label / Embedded** | Strip our branding, embed under partner shell, partner-managed tenancy | Professional, Enterprise | 25% uplift on platform fee, $50K minimum |
| **Premium Support** | 24×7, 1h Sev-1, named CSM, quarterly business review | Starter, Professional | 22% of platform fee |
| **Sandbox Tenant** | Non-prod copy of production for tuning experiments | All | $18K |
| **Read-Replica Deployment** | Analytics workload isolation | Professional | $24K |
| **Implementation — see Section 9** | One-time professional services | All | $40K–$350K |

---

## 7. Competitive pricing benchmarks

Sources: G2, Gartner Magic Quadrant Critical Capabilities reports (2024–2025), publicly disclosed deal sizes, win/loss interviews, vendor partner program docs. Where ranges are wide, the figure represents typical mid-market deal size, not enterprise outliers.

| Vendor | Typical pricing model | Mid-market ACV (75K SKUs / 20 planners) | Enterprise ACV (500K+ SKUs) | Notes |
|---|---|---|---|---|
| **Blue Yonder Luminate** | Per-module, per-user, heavily negotiated | $500K–$1.2M | $2M–$8M | 6–12 month implementation, services often equal license |
| **o9 Solutions** | Platform fee + per-user + scope | $600K–$1.5M | $2.5M–$10M+ | "Enterprise knowledge graph" positioning; smallest deals rarely under $500K |
| **Kinaxis Maestro / RapidResponse** | Per-named-user (~$5K–$15K/seat/yr) + platform | $400K–$900K | $1.5M–$5M | Concurrent planning differentiator |
| **RELEX Solutions** | Per-store / per-SKU + platform | $350K–$800K | $1M–$3M | Strong in retail / grocery |
| **ToolsGroup SO99+** | Per-SKU tier + module | $200K–$500K | $700K–$2M | Demand sensing strength |
| **Anaplan** | Per-user (~$2K–$8K/user/yr) + workspace | $250K–$700K | $1M–$4M | More planning-platform than purpose-built SC |
| **Netstock** | Per-user, low-end | $30K–$120K | n/a (doesn't play here) | SMB / Acumatica/NetSuite ecosystem |
| **Streamline / GMDH** | Per-user, low-end | $40K–$150K | n/a | SMB |

**Where we slot in:** Starter ($52K) competes with Netstock/Streamline/GMDH on price and wins on forecasting depth + AI agent. Professional ($281K mid) undercuts ToolsGroup by 15–30%, RELEX/Kinaxis by 50–70%, with more breadth (S&OP + clustering + tuning studios in one platform). Enterprise ($965K mid) sits at one-third to one-half the price of o9 / Blue Yonder for comparable scope — pitch as "the AI-native re-platform of supply chain at the price of a legacy point tool."

We are deliberately priced below the legacy enterprise vendors. We have no incumbency, no analyst awards, no reference list — discount-vs-incumbent is our wedge for the first 18–24 months. After 10 Enterprise references, raise the rack 20–30%.

---

## 8. Discounting & deal structure

| Lever | Standard policy |
|---|---|
| **Term discounts** | 1-yr: 0% | 2-yr (paid annually): 8% | 3-yr (paid annually): 15% | 3-yr prepaid: 22% |
| **Multi-year price lock** | Year-2 and Year-3 capped at CPI or 5%, whichever is lower (only on multi-year) |
| **Ramp deals** | Year-1 at 60% of run-rate, Year-2 at 100%, Year-3 at 100% — only for Enterprise; requires 3-yr commit |
| **Deal-desk discount bands** | AE may offer up to 10%. Sales Manager up to 20%. VP Sales up to 30%. CEO sign-off > 30%. Logo-trophy deals (named target accounts) get a one-time CEO waiver. |
| **Add-on discounts** | Add-ons attached at signing get 15% off list. Add-ons added mid-contract are at full list (this incentivizes the bundled close). |
| **MSA terms** | NET-30 standard. NET-60 only above $500K ACV with finance approval. NET-90 declined. |
| **Auto-renewal** | 12-month auto-renew with 60-day notice. Annual uplift 7% absent multi-year lock. |
| **Cancellation** | No mid-term cancellation. Cause-based termination: 30-day cure period. |
| **Pilot / POC** | 60-day paid pilot only — $25K, fully creditable to ACV at signing. No free POCs above Starter scope. |
| **Most-favored-nation** | Decline. Offer a "rate review at renewal" clause instead. |

**Sales-quoting guardrails:** never discount the platform fee below 30% off list. Never discount the SKU meter — it is the fairness mechanism, discounting it means giving away scale. Concentrate discounts on add-ons and implementation, which are easier to defend.

Industry context: 25–40% off list is normal in SC software. 20% on a 3-yr commit is the expected deal. 35%+ needs a real reason — logo, reference, or displacement of a named competitor.

---

## 9. Implementation & professional services

We sell implementation as fixed-fee tiers, not T&M. Reason: T&M scares procurement and creates an open-ended liability. Fixed-fee forces our delivery org to scope properly.

| Tier | Scope | Duration | Fixed fee |
|---|---|---|---|
| **Quick-Start** (Starter) | Standard CSV ingestion of sales + forecast + inventory; default ABC-XYZ + safety stock policies; 1 day of planner training | 4 weeks | **$40,000** |
| **Standard** (Professional) | Up to 3 source-system integrations (ERP + WMS + planning tool); per-cluster tuning of one tree model; S&OP cycle setup; 3 days of planner + admin training; 1 month hypercare | 8–12 weeks | **$120,000** |
| **Advanced** (Enterprise) | Up to 6 source-system integrations; foundation-model warm-up; AI agent persona tuning; multi-echelon configuration; customer-analytics setup; full data-quality remediation; 5 days training; 3 months hypercare | 16–24 weeks | **$280,000** |
| **Full Transform** (Enterprise + displacement) | Replaces 2+ legacy systems (e.g., retire Blue Yonder Demand + a separate inventory tool); change management; integrated cutover plan; legacy data migration | 6–9 months | **$350,000–$600,000** (custom-quoted, but always fixed-fee) |

**Margin target: 45–55% gross margin.** Requires standardized integration playbooks for the top 6 ERPs (SAP, Oracle, NetSuite, JDE, MS D365, Infor); delivery pod of 1 senior consultant + 2 engineers + 1 part-time data scientist; day-rate floor $2,200 external / $1,100 internal cost.

**Services-to-license ratio target: 0.4×–0.7×** at signing. Below 0.4× we under-scope; above 0.7× we look like consulting-ware, which depresses our SaaS multiple.

**Annual training subscription** ($24K): refresher + new-hire onboarding + quarterly webinars. Attaches to ~40% of Pro/Enterprise customers in Year 2. ~80% margin, reduces churn.

---

## 10. Land-and-expand motion

**The land:** sell **Professional** with the Inventory module front-and-center. The opening sales conversation is "we will release X days of working capital from your inventory in 90 days" (V2). Buyer = VP of Supply Chain or CFO. Sponsor = head of inventory planning. Proof = 60-day paid pilot on one product category.

**Why not lead with forecasting?** Every SC vendor claims AI forecasting. Inventory release is a number on a balance sheet the buyer cares about *this quarter*. Forecasting is the proof, not the headline.

**Why not lead with the AI agent?** It is the magic demo moment, not a contract line item. Lead with it and procurement spends 6 weeks on AI-governance questions before signing. Sell it as a Pro add-on after the inventory case is made.

**Why not Starter as the land?** $50K ACVs don't pay for an enterprise sales motion. Starter exists for (a) channel/partner-led mid-market and (b) a price anchor that makes Professional look reasonable.

**Expansion path (the "land curve"):**

| Quarter | Move | Typical ARR delta |
|---|---|---|
| Q0 | Land Professional + Standard Implementation | +$250K ARR |
| Q1 | 60-day hypercare; quarterly business review #1 surfaces gaps | — |
| Q2 | Attach AI Copilot Pack (after agent demo at QBR #2) | +$45K ARR |
| Q3 | Attach Customer Analytics (when customer-level forecast question surfaces) | +$45K ARR |
| Q4 | Attach Executive S&OP at the next planning-cycle pain point | +$75K ARR |
| Y2 | Renewal at 100% with 7% uplift; option to upgrade to Enterprise | +$200K ARR (edition jump) + uplift |
| Y2 Q3 | Attach Foundation Models when customer wants accuracy gains beyond tree-model ceiling | +$90K ARR |

**Net-revenue-retention target: 130%** by Year 2 on accounts that survive Year 1. That is the number the board should be tracking, not just new logo ACV. The whole pricing architecture above — Professional as the land, dense add-on shelf, edition-jump Enterprise — is engineered to hit this NRR.

**What sales should *not* do:** don't lead with "we have 12 algorithms" — lead with "we will release $X of inventory in 90 days." Don't quote per-SKU in the first call — always quote bundled ACV. Don't promise foundation models in a Starter deal — the upgrade has to feel like an upgrade. Don't give AI Copilot away to close — it is the highest-margin add-on; protect it.

---

## Decision summary

1. **Approve the hybrid pricing model**: edition platform fee + per-10K-SKU meter + add-ons.
2. **Approve the three editions**: Starter $48K, Professional $180K, Enterprise $480K (platform fee only, before SKU meter and add-ons).
3. **Approve the lead-value driver**: V2 (inventory working-capital release) as the headline of every Professional and Enterprise sales conversation.
4. **Approve the discount bands**: AE 10%, SM 20%, VP 30%, CEO above. Never discount the SKU meter.
5. **Approve the services model**: fixed-fee, four tiers, 45–55% gross margin target.

Once approved, this becomes the only pricing reference for sales, deal desk, finance, and the customer-success org. Any deal that deviates routes through the deal desk for sign-off.
