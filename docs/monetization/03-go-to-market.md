# Go-to-Market Plan — Supply Chain Command Center

> Operating plan for commercializing Supply Chain Command Center (SCCC). Companion to [01-market-and-icp.md](01-market-and-icp.md) (positioning + ICP) and [02-pricing-and-packaging.md](02-pricing-and-packaging.md) (pricing). Written for the founding team and the board.

---

## 1. GTM Philosophy: Sales-Led, Product-Assisted

We are building a **sales-led** motion with **product-assisted trial mechanics**. We are not building a PLG (product-led growth) business, and we should resist any temptation to pretend otherwise.

### Why sales-led wins for this category

Three structural factors decide the motion in supply chain planning software:

| Factor | Reality at SCCC ACVs | Implication |
|---|---|---|
| **Average Contract Value** | Target $250K–$1.2M ACV (per [02-pricing-and-packaging.md](02-pricing-and-packaging.md)) | Below $50K ACV PLG works; above $150K ACV the buyer requires a salesperson, a statement of work, and a security review — full stop. |
| **Complexity of sale** | 6–10 person buying committee, ERP/EDW integration, change-management impact on 50–500 planners | A self-serve trial cannot resolve "who owns the demand plan" or "how does this replace JDA/SAP IBP". |
| **ICP buying behavior** | Supply chain leaders buy via RFP, reference calls, analyst reports, peer networks (CSCMP, IBF, ASCM) | They expect a named AE, a Solutions Engineer, a customer success owner, and a roadmap commitment. PLG signals the opposite of enterprise readiness. |

The codebase strengthens this thesis. The product spans **80 mounted routers across 6 domains** (`api/routers/{forecasting,inventory,operations,intelligence,platform,core}/`), **88 SQL migrations**, **40 frontend query modules**, **22 React tabs**, and a 4–6 hour `make setup-all` data pipeline ([docs/PLATFORM_GUIDE.md](../PLATFORM_GUIDE.md)). That is not a 14-day-trial product. The proof that it is differentiated — Chronos foundation models, per-cluster SHAP, champion meta-learner across 8 strategies, the Bolt Hierarchical bottom-up reconciliation in [docs/specs/02-forecasting/20-bolt-hierarchical.md](../specs/02-forecasting/20-bolt-hierarchical.md) — requires a Solutions Engineer in the room to articulate.

### Where product-assist creates leverage

Sales-led does not mean "no product mechanics". Four investments compress the sales cycle:

- **Snapshot demo tenant** pre-loaded with synthetic beverage-alcohol data (~50K SKUs, 2 years, 24 customers); AE spins up in 60 seconds.
- **POV self-serve loader** (`make load-prospect`) that ingests a prospect CSV extract into a sandboxed tenant for SE-led baseline + lift in 7 days. Built on the `DomainSpec` registry and `make normalize-all` / `make load-all` chain.
- **Accuracy "proof report" PDF** auto-generated from `agg_accuracy_lag_archive` and the FVA ladder ([docs/specs/08-integration/07-fva.md](../specs/08-integration/07-fva.md)) — the AE leave-behind.
- **AI Planner insight teaser**: the [AI Planning Agent](../specs/06-ai-platform/01-ai-planning-agent.md) writes 5 sample cards against the prospect's data — a tangible "Monday morning" moment.

Steady-state pipeline mix: 40% outbound, 30% partner, 20% inbound, 10% expansion.

---

## 2. The Sales Motion: End-to-End Playbook

### 2.1 Stage definitions and conversion targets

| # | Stage | Owner | Exit criteria | Conversion target |
|---|---|---|---|---|
| 0 | **Target list / fit** | RevOps + AE | Account scored against ICP fit grid (revenue, SKU count, vertical, planner count, tech stack signals) | n/a |
| 1 | **Cold prospecting** | SDR | Discovery call booked with VP Supply Chain or Director Demand Planning | 3% of touched accounts |
| 2 | **Discovery** | AE + SE | Confirmed pain (forecast accuracy gap, planner inefficiency, stockouts), economic owner identified, MEDDPICC fields ≥60% complete | 60% to demo |
| 3 | **Technical demo** | SE-led, AE present | Champions identified, 2nd meeting booked with broader team, technical Q&A list < 5 open items | 50% to POV |
| 4 | **Mutual close plan + POV scoping** | AE | Signed POV SOW or LOI, data extract committed, success metrics agreed in writing | 70% to POV start |
| 5 | **POV execution (30/60/90)** | CSM-lead + SE | POV success criteria met (Section 5), executive readout completed | 60% to negotiation |
| 6 | **Procurement / negotiation** | AE + Legal | Pricing approved, MSA + DPA + Order Form red-lined, security review passed | 70% to close |
| 7 | **Closed-won** | AE → CSM | Signed contract, kickoff scheduled within 14 days | n/a |

**Blended cycle length**: 6–9 months for new-logo enterprise. Faster (4–5 months) for distributors already running on a flat ERP that lacks a planning module; slower (9–14 months) for accounts replacing an entrenched Blue Yonder, o9, or SAP IBP install.

### 2.2 Qualification framework: MEDDPICC, not BANT

BANT is too thin for a $500K ACV planning sale. We adopt **MEDDPICC** with explicit tracking inside Salesforce as required fields by stage:

| Letter | Definition | Where it lives in our world |
|---|---|---|
| **M**etrics | Quantified pain ($M tied to forecast error, stockouts, working capital, planner FTEs) | Pulled from prospect's S&OP scorecard or estimated from the ROI calculator in [02-pricing-and-packaging.md](02-pricing-and-packaging.md) |
| **E**conomic Buyer | Person who can sign the order form | Usually CFO for >$500K, COO/CSO for $250K–$500K |
| **D**ecision Criteria | Documented selection criteria | Often an RFP scorecard; we ghost-write criteria favorable to our strengths |
| **D**ecision Process | Steps from POV to signature | Mapped on a mutual close plan (Google Sheet shared with champion) |
| **P**aper Process | Procurement, legal, security review | Discovered by month 2; never surprises us at month 8 |
| **I**dentify Pain | Named pain by persona, in their words | Captured in discovery; recited back in every internal review |
| **C**hampion | Internal advocate willing to sell on our behalf | Usually Director Demand Planning or Director Replenishment |
| **C**ompetition | Incumbent system + alternative we are evaluated against | Drives battlecard usage |

Deals advance through stage gates only when MEDDPICC fields cross thresholds. RevOps audits weekly. Stuck deals get a "MEDDPICC clinic" with the VP Sales — no closing on hope.

### 2.3 Demo flow (90 minutes, SE-led)

| Min | Segment | Purpose |
|---|---|---|
| 5 | Discovery recap in their vocabulary | Earn the room |
| 15 | Control Tower "Monday view" ([06-ai-platform/03-control-tower.md](../specs/06-ai-platform/03-control-tower.md)) | Anchor on planner productivity |
| 20 | Forecasting depth: champion-vs-incumbent, per-cluster SHAP, Chronos foundation models, Bolt Hierarchical stockout-bias correction | Technical wow moment |
| 20 | Inventory Action Feed + AI Planning Agent with $-impact insights | Replaces 6 spreadsheets |
| 15 | S&OP cycle, Storyboard, FVA | Proves full operating loop, not a forecasting toy |
| 10 | Q&A + next-step commitment (POV scope, data extract, sponsor) | Advance the deal |
| 5 | Leave-behind: ROI snapshot, architecture 1-pager, POV SOW | Make it easy to say yes |

### 2.4 POV / pilot structure

See Section 5 — this is the heart of the motion.

### 2.5 Procurement and close

Common landmines and how we prevent them:

| Landmine | Prevention |
|---|---|
| Security review surprises (SOC2 Type II, pen test, DPA) | Push trust portal + SOC2 evidence into the prospect's hands by end of discovery; pre-engage their CISO in month 2. |
| Procurement re-opens pricing post-POV | Mutual close plan locks pricing range in writing at POV kickoff; CFO is briefed by the AE before procurement engages. |
| Legal stalls on MSA red-lines | Pre-negotiated fallback positions on liability cap (1× ACV), data residency, and IP ownership. Preferred-paper deals only above $750K ACV. |
| Single-tenant deployment scope creep | Standard deployment SOW with a fixed-fee implementation tier; anything custom is a separate Statement of Work with a Professional Services rate card. |

---

## 3. Buyer Personas and Buying Committee

A real enterprise SCCC deal has 6–10 people in the room. Map and name every one:

| Persona | What they care about | Top objections | How we neutralize |
|---|---|---|---|
| **VP / SVP Supply Chain** (economic owner) | Network-wide service level, working capital, cost-to-serve, board narrative | "Will this disrupt my team during peak season?" "How does this differ from the o9 demo I saw?" | Phased rollout plan, named CSM, beverage-alcohol customer reference, side-by-side feature comp vs o9/Blue Yonder/Kinaxis. |
| **Director Demand Planning** (champion #1) | Forecast accuracy, planner productivity, FVA, defensibility of the number to Sales/Marketing | "How is this different from Chronos in a notebook?" "Will my planners trust an AI exception queue?" | Live champion-vs-incumbent demo, [Champion Experimentation Studio](../specs/03-demand-intelligence/05-champion-experimentation-studio.md), explainability via per-cluster SHAP, FVA scoreboard. |
| **Director Replenishment / Inventory** (champion #2) | Fill rate, days of supply, exception aging, transfer/PO efficiency | "Does it actually generate orders or just dashboards?" "How does it integrate with my ERP for receipts?" | Action Feed + Replenishment Plan + Rebalancing demo; ERP integration architecture from [docs/specs/08-integration/01-integration-architecture.md](../specs/08-integration/01-integration-architecture.md). |
| **CIO / CTO / VP IT** (technical gatekeeper) | Architecture fit, security, total cost of ownership, vendor risk | "Why not Snowflake + dbt + a notebook?" "Are you SOC2?" "What is your runtime footprint?" | Single-tenant Postgres deployment story, architecture diagrams, SOC2 roadmap, openness on tech stack (Python/FastAPI/React/Postgres — no exotic dependencies). |
| **CFO / Finance partner** (signature for >$500K) | ROI payback period, total cost (license + implementation + runtime), risk-adjusted business case | "Show me the ROI in <12 months." "What if we cancel after year 1?" | Quantified business case from the value drivers in [02-pricing-and-packaging.md](02-pricing-and-packaging.md); month-by-month payback schedule; out-clause negotiated only with auto-renew protection. |
| **Head of S&OP / Demand-Supply Lead** | Cycle discipline, executive alignment, scenario planning | "Can it run our 6-stage S&OP in our calendar?" | [S&OP Cycle](../specs/05-operations/01-sop-cycle.md) walkthrough, Scenario Planning module. |
| **End-user planners (3–5 in the room)** | "Will this make my Monday morning better or worse?" | "I already have my Excel model." | Hands-on demo time, planner-friendly UI tour, retention of analyst work via SQL Runner + CSV export. |
| **Procurement** | Risk, paper terms, renewals leverage | n/a (objections are commercial, not technical) | Standard MSA, fixed pricing tiers, no custom legal until $1M+. |

---

## 4. Channel Strategy

### 4.1 Channel mix by year

| Channel | Year 1 | Year 2 | Year 3 | Rationale |
|---|---|---|---|---|
| **Direct (founder + AE)** | 90% | 65% | 50% | Highest control, fastest learning loop, only practical motion for first 10 logos. |
| **SI / consulting partners** (Bristlecone, Genpact, Tredence, mid-market boutiques first; Deloitte/Accenture/EY in year 2) | 5% | 20% | 30% | SIs bring deals + implementation revenue; we keep license, they keep services. Critical for scale. |
| **Tech alliances** (AWS, Snowflake, Databricks marketplaces) | 5% | 10% | 12% | Marketplace co-sell unlocks committed-spend budgets, accelerates procurement. |
| **Industry/vertical resellers** (e.g. WSWA-affiliated tech consultancies, beverage-alcohol-focused SIs) | 0% | 5% | 5% | Fast-track in beverage-alcohol; not a primary channel. |
| **OEM / embedded** (forecasting engine inside an ERP/WMS partner) | 0% | 0% | 3% | Optionality — not a Year 1–2 priority. |

### 4.2 Partner program design

- **Year 1** — no formal program. Recruit 2–3 boutique SIs (Bristlecone, Genpact, Tredence, LatentView, Mu Sigma) hungry for AI-native logos in exchange for first-mover implementation rights. Skip Big 4 — they need proof, not pitches.
- **Year 2** — tiered program (Registered → Silver → Gold) with deal registration, 15–25% margin, certified training. Add Deloitte/Accenture once we have 8+ logos and a Gartner mention.
- **Year 3** — AWS + Snowflake marketplace listings; co-sell with their account teams. Single-tenant Postgres deploys cleanly into customer VPCs.

We will **not** chase ISV-distribution (SAP/Oracle/Microsoft embedding) in Y1–2 — a 24-month dance that consumes engineering and produces nothing for direct ARR.

---

## 5. POV / Pilot Framework: 30-60-90 Day Proof

The POV is the single most important sales artifact. A bad POV kills more deals than bad pricing. Standardize ruthlessly.

### 5.1 POV principles

- **Paid POV** ($25K–$75K, credited 100% against year-1 license at signature). Free POVs attract tire-kickers and underweight the customer's effort. Paid POVs bring an executive sponsor and a real data extract.
- **Time-boxed**: 90 days from data-extract handover to executive readout. Hard stop.
- **Success criteria in writing**, signed by the economic buyer, before any data moves.
- **One ICP-fit slice**: 1 distribution center or 1 product category, not "everything everywhere". Scope creep kills POVs.
- **CSM-led, SE-supported**: the same humans who will run the post-sale account run the POV. No throw-it-over-the-wall.

### 5.2 The 30-60-90 framework

| Phase | Days | Goal | Deliverables | Go/No-Go gate |
|---|---|---|---|---|
| **Phase 1: Ingest & Baseline** | 0–30 | Land customer data, mirror current planning state | Data extract loaded into single-tenant POV instance via `make load-prospect`; baseline accuracy report (incumbent forecast WAPE, bias by ABC-XYZ class); inventory health snapshot | Data quality ≥95% (DQ engine score); baseline metrics agreed as the comparison anchor |
| **Phase 2: Lift Demonstration** | 31–60 | Run SCCC champion vs. incumbent; show inventory + exception value | Champion-vs-incumbent accuracy delta (target: ≥3 pts WAPE improvement on slow movers, ≥1 pt on fast movers); AI Planner insights for top 50 exceptions with $ impact; safety stock optimization showing working-capital release | Forecast accuracy ≥ baseline + 1 pt blended; ≥10 high-confidence AI insights validated by planner team |
| **Phase 3: Operational Proof** | 61–90 | Prove the platform fits a real planning week; build the business case | One full S&OP cycle simulated; planner UAT with ≥3 planners using the UI for 2 weeks; Storyboard exception-resolution time vs. current state; CFO-ready ROI model | Planner NPS ≥40; documented hours saved per week per planner; signed executive readout |

### 5.3 Success metrics (tied to value drivers in [02-pricing-and-packaging.md](02-pricing-and-packaging.md))

| Value driver | POV target | Measurement |
|---|---|---|
| Forecast accuracy lift | +1 pt WAPE blended, +3 pts on slow movers | `agg_accuracy_lag_archive` champion vs. incumbent |
| Inventory release | 5–10% reduction in safety stock at constant service level | SS optimizer output × unit cost |
| Stockout reduction | 10–20% reduction in lost-sales events | Intramonth stockout detection |
| Planner productivity | 4–8 hours/week saved per planner | UAT time-tracking |
| Decision quality | ≥80% of AI insights actioned within 7 days | `ai_recommendation_outcomes` table |

### 5.4 Anti-patterns

Letting the customer "evaluate" without written success criteria; promising 6 verticals of features in one POV; SE silence >3 business days; skipping the executive readout (no CFO-friendly deck = procurement stall).

---

## 6. Launch Plan: 0–90, 90–180, 180–365

### 6.1 Days 0–90: Foundation

| Workstream | Deliverable |
|---|---|
| **Design Partner Program** | Recruit 3–5 design partners at 50–70% discount in exchange for: (a) named reference, (b) joint case study, (c) quarterly product feedback session, (d) logo rights. Target: founder's network at SGWS, Boeing Distribution, McKesson, Daimler, plus 1 net-new beverage-alcohol distributor. |
| **Sales infrastructure** | Salesforce CRM, MEDDPICC fields, Gong, Outreach/Salesloft, ZoomInfo, LinkedIn Sales Navigator. Stage definitions and conversion dashboards live by day 60. |
| **Sales collateral** | Pitch deck (V1), demo script + demo tenant, 1-pager, ROI calculator (Google Sheet first, app later), security trust page, MSA + Order Form templates. |
| **Analyst pre-briefs** | Informal briefings with Gartner (Tim Payne, Pia Orup Lund), IDC (Simon Ellis), Forrester. Goal: get on the radar, not on a quadrant — that takes 12+ months. |
| **First conference presence** | NRF (January), Gartner SCP Symposium (May), IBF Best Practices (June). Booth at one, attend others. |

### 6.2 Days 90–180: First Wins

| Workstream | Deliverable |
|---|---|
| **Convert design partners** | 2 of 3 design-partner POVs to paid deals; first 2 case studies drafted (anonymized acceptable for now). |
| **First commercial logos** | 2–3 net-new logos at full ACV via founder-led outbound; cycle length will be longer (8–10 months) than steady state. |
| **Hire AE #1 + SE #1** | See Section 10. AE shadows founder for 30 days, takes own pipeline by day 90. |
| **Content engine launch** | Weekly technical blog (founder + ML engineer): "Stockout-bias correction with hierarchical reconciliation", "Per-cluster SHAP for sparse demand", "When foundation models beat tree models". Goal: become the technical authority planners email each other. |
| **Webinar #1** | "The AI Planning Agent vs. the Planning Chatbot" — co-host with a design-partner customer. |
| **Conference activation** | Booth + speaking slot at Gartner SCP Symposium (May) + WSWA Convention (April) — beverage-alcohol-anchored ICP. |

### 6.3 Days 180–365: Scale

| Workstream | Deliverable |
|---|---|
| **6–10 net-new logos** | Total 8–13 logos by year-end; ARR target per [02-pricing-and-packaging.md](02-pricing-and-packaging.md). |
| **Analyst inclusion** | Submit briefing for Gartner MQ for SCP, IDC MarketScape for Demand Planning, Forrester Wave for S&OP. Realistic timeline for first inclusion: 18–24 months. |
| **Hire SDR #1, CSM #1, second AE** | See Section 10. |
| **Partner program launch** | Formal program with 3–5 SI partners; deal registration in Salesforce. |
| **Conference cadence** | NRF, Gartner SCP, Manifest, NACDS, WSWA, IBF, ISM World, CSCMP EDGE — 6–8 events with founder/CRO + 2 booth events. |
| **First customer advisory board** | 5–8 customers, in-person, 1.5 days, founder-hosted. Roadmap input + reference cultivation. |

---

## 7. Marketing & Demand Generation

### 7.1 Content engine (the cornerstone)

Planners read deeply technical content when it is genuine. Our differentiator is technical depth — make it visible.

- **Weekly technical blog** (founder + ML engineer): forecasting research applied to real planning ([15-expert-panel](../specs/02-forecasting/15-expert-panel-algorithm-selection.md), [20-bolt-hierarchical](../specs/02-forecasting/20-bolt-hierarchical.md), [23-lgbm-accuracy-tuning](../specs/02-forecasting/23-lgbm-accuracy-tuning.md)).
- **Bi-weekly planner thought leadership**: pieces for Directors of Demand Planning / Replenishment — "How we cut intramonth stockouts 18% with an exception queue", "FVA scoreboards that don't get gamed".
- **Podcast (monthly, year 2)**: founder interviews supply chain leaders.
- **SEO + LLM-discoverability**: own "demand forecasting accuracy", "champion model selection", "AI exception management supply chain"; structured FAQ + glossary so Claude/ChatGPT cite us.

### 7.2 Other channels

- **Webinars** — bi-weekly Y1, weekly Y2; customer co-hosts most credible, practitioner deep-dives + analyst guest features round out the calendar.
- **ABM (top-200 accounts)** via 6sense/Demandbase: personalized landing pages, printed direct mail (forecast-accuracy benchmark leave-behind), 3-touch LinkedIn nurture before outbound, founder 1:1 outbound for tier-1.
- **Paid**: LinkedIn-heavy (>70%), Google Search on high-intent keywords, Gartner Peer Insights once reviews accumulate. No display, no programmatic.
- **Community + associations**: sponsor APICS/ASCM regional chapters, CSCMP EDGE booth + exec dinner, ISM World, IBF; host an invite-only planner Slack ourselves — becomes a moat.

---

## 8. Vertical Strategy

We win by going **deep before wide**. Three verticals in priority order:

### 8.1 Vertical 1 (Year 1–2): Beverage-Alcohol Distribution

**Why first:** the founder spent 20+ years at SGWS — the largest US distributor — and owns the network, vocabulary, and credibility. Beverage-alcohol distribution has unique demand patterns (seasonality, allocation/quota dynamics, customer-level demand variance, three-tier regulatory complexity) that the platform has been engineered for ([fact_customer_demand_monthly](../specs/01-foundation/07-customer-demand-fact.md), Bolt Hierarchical, customer-enriched features). Total addressable: ~25 enterprise distributors in the US + ~80 mid-market, plus the global market (Pernod, Diageo, AB-InBev distribution arms).

**Wedge logos**: SGWS (warm), RNDC, Breakthru, Johnson Brothers, Empire Merchants. Win 3 of these and the rest follow.

**Conferences**: WSWA Convention (April), NABCA Annual Conference (March).

### 8.2 Vertical 2 (Year 2–3): Broader Food & Beverage Distribution

**Why second:** the closest adjacency — same channel mix complexity, same distribution math, same planner persona. Sysco, US Foods, Performance Food Group, Reinhart, KeHE, UNFI. Slightly larger ACVs, slightly longer cycles, but the platform translates 1:1.

**Conferences**: IFDA (International Foodservice Distributors Association), NACDS for retail-adjacent.

### 8.3 Vertical 3 (Year 3+): Industrial Distribution

**Why third:** large ACVs ($800K–$2M), high SKU counts (Boeing Distribution, McKesson, Grainger, Fastenal, MSC Industrial). Founder's network at Boeing Distribution + McKesson is a wedge. Buying cycles are longer (12–18 months), but customer LTV is higher.

**Conferences**: NAW (National Association of Wholesaler-Distributors), MDM Industrial Distribution Summit.

### 8.4 Watchlist (do not chase yet)

- **CPG manufacturers** — large TAM but dominated by entrenched incumbents (Blue Yonder, o9). Selectively pursue only with a referral.
- **Pharma distribution** — high-value but heavily regulated; defer until SOC2 + HIPAA + DSCSA readiness.
- **Retail (apparel, electronics, grocery)** — different planning math (size/color, store-level), would force material product investment. Defer.

---

## 9. Reference Customer Strategy: First 3–5 Logos

### 9.1 Land plan

| Account | Source | Strategy | Discount | Status |
|---|---|---|---|---|
| **SGWS (Southern Glazer's)** | Founder's home company | Anchor reference; structured commercial deal post-internal acceptance | 60% (design partner) | Existing internal deployment is the proof case — convert to formal commercial agreement |
| **Boeing Distribution Services** | Founder's network | Industrial-distribution beachhead | 60% | Warm-intro outbound by founder, Q1 |
| **McKesson** | Founder's network | Healthcare-adjacent distribution; serves Vertical 3 wedge | 50% | Warm-intro outbound, Q2 |
| **Daimler Trucks N.A. parts distribution** | Founder's network | Industrial after-market parts distribution | 50% | Warm-intro outbound, Q2 |
| **1 net-new beverage-alcohol distributor** (RNDC, Breakthru, Empire, or Johnson Bros) | Cold outbound + warm intro from founder's WSWA network | Vertical 1 expansion outside SGWS | 50% | Q3 |

### 9.2 Design-partner contract terms

- 50–70% discount on year-1 ACV
- Year 2 onward: returns to standard pricing (with an early-renewal incentive)
- In exchange: (a) named reference (logo + customer call), (b) joint case study (anonymized OK to start), (c) quarterly product council seat, (d) keynote/co-presentation at one conference per year, (e) 30-day SLA on quote responses for analyst/prospect inquiries.

### 9.3 Reference operations

- **Reference manager hired by month 9** — protects customer time, books reference calls, rotates the load
- **Tiered reference program**: Tier 1 (full reference call), Tier 2 (logo + quote), Tier 3 (private analyst-only reference)
- **Always-on case study pipeline**: 1 published case study per quarter post-month-12

---

## 10. Sales Team Build

Hiring tied to ARR milestones, not calendar. Hiring early kills capital efficiency; hiring late strangles deals.

| Role | Hire trigger | Why now |
|---|---|---|
| **Founder** | Day 0 — sells the first 3–5 logos personally | No one else can. Founder credibility, technical depth, network. |
| **Solutions Engineer #1** | $0 ARR (immediately) | Founder cannot scale demos. SE is the highest-leverage early hire. Must be deeply technical (Python, ML, SQL fluent) and customer-fluent. |
| **Account Executive #1** | $1M ARR or 3 closed-won logos | Founder hands off net-new outbound. AE shadows founder for 30 days, then runs full cycle on tier-2 accounts while founder retains tier-1. |
| **Customer Success Manager #1** | 5 paying customers | At 5 customers, retention risk + expansion opportunity exceed founder bandwidth. CSM also leads POVs (transition from SE-led). |
| **Sales Development Rep #1** | $2M ARR | Pipeline coverage gap appears around $2M ARR run-rate. Outbound + inbound qualification. Pair 1:1 with AE. |
| **Account Executive #2** | $3–4M ARR | Coverage requires 2nd AE. Vertical specialization begins (AE #1 = beverage-alcohol, AE #2 = industrial). |
| **Solutions Engineer #2** | $4M ARR | Demo + POV bandwidth. SE-to-AE ratio target: 1:2 in early stage, 1:3 at scale. |
| **VP Sales / CRO** | $5–7M ARR or 4 quota-carrying reps | Founder transitions out of day-to-day deal coaching. Hired from a $50–200M ARR planning/SCM software vendor — not from a $1B+ company. |
| **Marketing #1 (Demand Gen lead)** | $2M ARR | Until then, founder + content engine. At $2M, paid + ABM + events need a dedicated owner. |
| **Reference Manager / Customer Marketing** | 8–10 paying customers | Reference load exceeds CSM capacity; case studies, advisory board, conference customer-speakers all need ownership. |

**Avoid**: CRO before $3M ARR (kills founder learning loop), SDRs before AE has a repeatable playbook, junior AEs (this category requires 8+ years SCM software selling).

---

## 11. KPIs and Scorecard

### 11.1 Leading indicators (weekly)

| Metric | Target | Owner |
|---|---|---|
| New qualified discovery calls / week | 15 (Y1), 30 (Y2), 50 (Y3) | SDR + AE |
| Pipeline coverage (open pipeline ÷ remaining quota) | 4× | AE + RevOps |
| Demo → POV conversion | ≥50% | AE + SE |
| POV → close conversion | ≥60% | CSM + AE |
| Average sales cycle (discovery → close) | <9 months blended | RevOps |
| Multi-thread depth (named contacts per active opp) | ≥4 by stage 3, ≥6 by stage 5 | AE |
| MEDDPICC completeness by stage | 60% by stage 3, 90% by stage 5 | AE |

### 11.2 Lagging indicators (monthly / quarterly)

| Metric | Year 1 target | Year 2 target | Year 3 target |
|---|---|---|---|
| Net-new ARR | $1.5–2.5M | $5–8M | $12–18M |
| ACV | $400K avg | $500K avg | $600K avg |
| Net Revenue Retention (NRR) | n/a (too early) | ≥110% | ≥120% |
| Gross Revenue Retention | ≥90% | ≥92% | ≥93% |
| Logo retention | ≥95% | ≥95% | ≥95% |
| Gross margin | ≥75% | ≥78% | ≥80% |
| CAC payback period | <24 months | <18 months | <14 months |
| Magic Number | n/a | ≥0.7 | ≥1.0 |
| Sales cycle length | 9 months | 7 months | 6 months |

### 11.3 Operational rhythm

- **Weekly forecast call** — AEs commit by stage; commits are rolled into the board forecast
- **Bi-weekly pipeline review** — every open deal >$200K reviewed; MEDDPICC gaps surfaced
- **Monthly deal post-mortems** — every won and lost deal in the prior month, root-caused
- **Quarterly QBR** — vertical performance, channel mix, content engine ROI, customer references
- **Annual planning** — September for next-year targets, hiring plan, segment expansion

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Founder bandwidth becomes the bottleneck | High | High | Hire SE #1 immediately; structured handoff to AE #1 by month 12 |
| Big Three (Blue Yonder, o9, Kinaxis) match feature messaging | Medium | Medium | Lead with depth (foundation models, AI exception queue, single-tenant deployment, faster TTV); never compete on RFP feature checklists |
| Single-tenant deployment scales operationally | Medium | High | Standardize via Docker Compose + IaC; cap deployment scope; charge for custom integrations separately |
| Long sales cycle starves cash | Medium | High | Paid POVs ($25K–$75K) generate revenue during the cycle; design partners pay reduced but real ARR |
| ICP drift (chasing CPG, retail, pharma early) | High | High | Vertical discipline enforced at deal review; QBR scores vertical fit |
| Analyst inclusion takes 18–24 months | High | Low (early), Medium (year 2) | Active analyst relations from month 1; informal briefings before formal MQ submission |

---

## 13. Summary

We will execute a **sales-led, product-assisted** motion targeting beverage-alcohol distribution first, broader food & beverage second, industrial distribution third. The founder closes the first 3–5 logos personally — leveraging SGWS, Boeing, McKesson, and Daimler relationships — at design-partner discounts in exchange for references and case studies. The first hire is a Solutions Engineer; AE #1 follows at $1M ARR; SDR and CSM follow at $2M ARR. Pricing and ROI math live in [02-pricing-and-packaging.md](02-pricing-and-packaging.md); ICP and positioning in [01-market-and-icp.md](01-market-and-icp.md). The single most important operational discipline is the 30-60-90 paid POV with written success criteria — POV quality drives every downstream KPI on the scorecard.
