# 05 — Financials & Business Plan

**Document owner:** Founder / acting CFO
**Audience:** Founder, prospective board, prospective investors
**Companion docs (planned):** 01 Market & ICP, 02 Pricing & Packaging, 03 Go-to-Market, 04 Product Readiness
**Last review:** 2026-05-13
**Model unit:** USD, fiscal year = calendar year, Y1 = first full year after commercial launch

This is the operating model the founder/board will use to run the business monthly. It is built to be defensible to a Series A partner: every line is traceable to an assumption block. The numbers are plausible mid-case estimates for a vertical SaaS in supply chain planning sold to mid-market and lower-enterprise distributors and manufacturers.

---

## 1. Operating Thesis

**Five-year vision (if it works).** Supply Chain Command Center becomes the planning system of record for ~80–120 mid-market distributors and manufacturers in CPG, beverage-alcohol, food, industrial distribution, and aftermarket parts — companies in the $250M–$3B revenue band that have outgrown spreadsheets and Microsoft D365 / NetSuite planning, are too small to justify a $5–15M Blue Yonder, Kinaxis, or o9 deployment, and that get nothing useful from RELEX (CPG-grocery focus) or ToolsGroup (mid-market but slow product velocity). At year 5 the business is doing roughly **$28–35M ARR**, growing 60–80% Y/Y, gross margin in the high 70s, NRR around 120%, and burning $4–8M annually against a Series B/C balance sheet of ~$25M. The wedge is forecasting accuracy (the multi-model ensemble + foundation-model layer + per-cluster tuning) and the time-to-value of the integrated inventory + S&OP + control tower stack — competitors charge for those as separate modules.

**Year 10 scenarios.** (a) **Strategic acquisition** — most likely outcome — by a tier-1 ERP (SAP, Oracle, Microsoft, Infor) or by a planning consolidator (Blue Yonder/Panasonic, Kinaxis, e2open) at 6–10x ARR, target $200–500M outcome. (b) **IPO-track** — only if the company crosses ~$100M ARR with a credible $250M+ TAM-capture story and clean Rule-of-40 — possible but requires the AI-copilot layer and multi-vertical expansion to compound. (c) **Lifestyle / profitable SaaS** — at $20–30M ARR with 80–100 customers, founder controls equity, distributes $5–10M of free cash flow annually. **Failure modes:** (1) ICP too narrow → 6+ month sales cycles drag CAC payback past 30 months; (2) implementation services overrun → margin compression below 65% kills the SaaS multiple; (3) a foundation-model + LLM-native competitor (well-funded YC/Series-A startup) leapfrogs the differentiator before ARR compounds; (4) IP / non-compete dispute with the founder’s current employer; (5) a tier-1 ERP ships “good-enough” planning bundled at zero incremental price, compressing the wedge.

---

## 2. Revenue Model Summary

**Pricing model (assumed from doc 02):** annual SaaS subscription, three tiers + services.

| Tier | Target customer | List ACV | Discount norms | Includes |
|---|---|---|---|---|
| **Starter** | $250M–$500M revenue, single division, ≤ 50K SKUs, 1 planner team | **$50K** | 0–10% | Forecasting + Inventory Planning + basic S&OP. 5 named users. |
| **Professional** | $500M–$2B revenue, multi-DC, 50K–500K SKUs, 5–25 planners | **$250K** | 10–25% | Adds Control Tower, Customer Analytics, AI Planning Agent, multi-echelon, scenario planning. 25 named users. Modules priced per cluster of SKUs above 250K. |
| **Enterprise** | $2B+ revenue, multi-division, 500K+ SKUs, 25+ planners | **$1M+** (typical $1.0–2.5M) | 15–30% | All modules + custom data adapters + dedicated CSM + SLA + private deployment option. |

**Services (one-time at deal start):**
- Starter: ~$25K (5–7 weeks remote implementation).
- Professional: ~$100K (10–14 weeks, hybrid).
- Enterprise: ~$250–400K (16–24 weeks, includes data integration).

**Assumption — Blended ACV ramp:**

| Metric | Y1 | Y2 | Y3 |
|---|---|---|---|
| % of new logos in Starter | 60% | 35% | 20% |
| % of new logos in Professional | 35% | 55% | 60% |
| % of new logos in Enterprise | 5% | 10% | 20% |
| **Blended new-logo ACV** | **$130K** | **$240K** | **$360K** |
| Services attach (% of new ACV) | 50% | 40% | 35% |
| Services revenue per new logo | $65K | $96K | $126K |

**Assumption:** services revenue is delivered ~70% in the deal year, ~30% next quarter; recognized ratably. Services gross margin is **30%** (delivery cost is 70% of revenue) — this is the single biggest pull on consolidated gross margin. By Y3 the goal is to drive services attach down via a partner channel (regional SIs taking the implementation work).

---

## 3. Customer Acquisition Cost (CAC)

**Assumption — sales motion:** sales-led, not PLG. SC planning is bought top-down by VP Supply Chain / VP Operations / CFO. POC-driven cycles of 3–6 months for Professional, 6–12 months for Enterprise. Outbound + warm intros from founder’s network is the Y1 channel; events + content + a small SDR team are added Y2.

**Industry benchmarks (mid-market enterprise SC SaaS):**
- Healthy CAC payback: **12–24 months** (vs 5–7 months for SMB PLG, vs 24–36 months for tier-1 ERP-adjacent).
- Healthy LTV/CAC: **3.0–5.0x**.
- Magic number: **0.7–1.2** acceptable for enterprise SaaS at this stage; <0.5 is a red flag.

**CAC build (Year 2 illustrative — first “real” GTM year):**

| Item | Y2 spend | Notes |
|---|---|---|
| AE fully-loaded ($220K base + $200K OTE @50% var + 25% benefits/tax) | $620K × 2 = $1.24M | 2 AEs by mid-Y2 |
| SDR fully-loaded ($75K base + $40K var + 25% load) | $145K × 2 = $290K | 2 SDRs |
| SE / solutions consultant | $260K × 1 = $260K | 1 SE supports both AEs |
| Marketing lead + programs | $200K + $150K = $350K | Events, content, paid |
| Travel, T&E, demo infra | $150K | |
| Sales tooling (Outreach, Gong, Salesforce) | $80K | |
| **Total S&M** | **$2.37M** | |
| **New logos closed Y2** | **18** (mix-weighted) | See ramp in §6 |
| **New ARR Y2** | **$4.32M** | 18 logos × $240K blended |
| **Loaded CAC per logo** | **$132K** | |
| **CAC payback (months)** | **15.5 months** | $132K / ($240K × 0.78 GM / 12) |

This sits inside the 12–24 month healthy band. Magic number for Y2: $4.32M new ARR / $2.37M prior-period S&M ≈ **1.04** — strong for mid-market SaaS. Y1 numbers are deliberately worse because we are building the design-partner book; the model recovers in Y2.

---

## 4. Lifetime Value (LTV)

**Assumption stack:**
- **Subscription gross margin:** 82% steady state. Hosting (~$15K/customer/year for an avg-sized Professional tenant on multi-tenant Postgres + Redis + GPU inference for foundation models), customer support staff, OpenAI/Anthropic API draw (LLM agent + AI Planning Agent — this is real cost, modeled at ~$3K/customer/year), pgvector/Postgres infra, MLflow.
- **Blended gross margin (sub + services):** 75% Y1, 76% Y2, 78% Y3 (services drag eases as attach rate falls).
- **Gross retention (GRR):** 90%. SC software is sticky — switching costs are high (data integration, planner muscle memory, S&OP cycle integration). Logos that churn typically do so because of acquisition or change of CIO, not dissatisfaction.
- **Net revenue retention (NRR):** 120% steady state. Drivers: (a) module upsell (a Starter customer adding Customer Analytics + AI Agent doubles ACV); (b) SKU growth (most pricing is partly SKU-tiered); (c) seat expansion (planner team grows after value is proven); (d) M&A by the customer pulling new divisions onto the platform.
- **Implied annual customer ARR growth:** 1.20 × (subscription) compounded.
- **Implied avg lifetime (geometric):** 1 / (1 − GRR) = **10 years gross**, but realistic working assumption is **7 years** (logos that double in size still count as the same logo for retention).

**LTV calculation (Professional tier, year-3 cohort):**

```
ACV_year_1 = $240K
Annual subscription GP at 82% GM = $197K
Cohort lifetime cash flow (NRR 1.20, GRR 0.90, 7-year horizon, discounted at 15%):
  Σ (197K × 1.20^t × 0.90^t) / (1.15^t) for t = 0..6 ≈ $1.42M
LTV (gross-margin-weighted, 7-year, 15% discount) ≈ $1.4M
```

**LTV / CAC (Y2):** $1.4M / $132K ≈ **10.6x.** That looks too high — and it would be if every cohort behaved this way. Realistic blended steady state, after accounting for Starter mix, lower Starter NRR, and CAC inflation in Y3+, is **LTV/CAC ≈ 4–6x** by Y3. Still healthy.

---

## 5. Unit Economics Targets

| KPI | Y1 | Y2 | Y3 | Target by Y4 | Notes |
|---|---|---|---|---|---|
| Subscription gross margin | 78% | 80% | 82% | 82% | Hosting + LLM costs amortize as customer base grows |
| Blended gross margin (sub + services) | 70% | 74% | 78% | 80% | Services drop from 35% to 15% of revenue |
| GRR (logo retention by ARR) | n/a | 92% | 90% | 90% | Y1 has no churnable cohort |
| NRR | n/a | 110% | 120% | 120% | Cohort upsell takes ~2 years to materialize |
| CAC payback (months) | 24+ | 15.5 | 14 | 12 | |
| LTV / CAC | 2.5x | 6x | 5x | 5x | Realistic blended |
| Magic number | 0.4 | 1.0 | 1.1 | 1.0 | Trailing 4Q new ARR ÷ prior-Q S&M × 4 |
| Burn multiple (cash burn / net new ARR) | 5.0x | 1.5x | 0.9x | 0.6x | This is the single most-watched investor metric |
| Rule of 40 | n/a | n/a | 40+ | 50+ | Growth % + EBITDA % |

---

## 6. Three-Year P&L Projection

**Assumption — recommended capital scenario (b): angel + small seed of $2.5M, see §9.** All P&L lines below assume this funding.

### 6.1 Customer & ARR Ramp

| Period | Paid design partners (cumulative) | Full-price new logos (period) | Logos cumulative | New ARR (period) | Expansion ARR | Churn ARR | **Ending ARR** |
|---|---|---|---|---|---|---|---|
| Y1 Q1 | 1 | 0 | 1 | $50K | $0 | $0 | $50K |
| Y1 Q2 | 2 | 0 | 2 | $50K | $0 | $0 | $100K |
| Y1 Q3 | 3 | 1 | 4 | $180K | $0 | $0 | $280K |
| Y1 Q4 | 3 | 1 | 5 | $130K | $0 | $0 | $410K |
| **Y1 total** | — | **2** | **5** | **$410K** | **$0** | **$0** | **$410K** |
| Y2 H1 | — | 7 | 12 | $1.68M | $30K | $20K | $2.10M |
| Y2 H2 | — | 11 | 23 | $2.64M | $80K | $40K | $4.78M |
| **Y2 total** | — | **18** | **23** | **$4.32M** | **$110K** | **$60K** | **$4.78M** |
| Y3 H1 | — | 15 | 38 | $5.40M | $400K | $200K | $10.38M |
| Y3 H2 | — | 18 | 56 | $6.48M | $700K | $400K | $17.16M |
| **Y3 total** | — | **33** | **56** | **$11.88M** | **$1.10M** | **$600K** | **$17.16M** |

**Assumption — design partner mechanics:** founder closes 3 paid design partners ($50K each, 12-month term, full-price renewal) by end of Y1 from existing network. Of those 3, 2 convert to full-price Professional ($240K) at renewal. First 2 “real” logos close Y1 H2 from outbound. By mid-Y2, AE #1 and AE #2 are productive (AE quota target $1.0M/year ramping to $2.0M by Y3). Y3 ramps to 6 quota-carrying AEs.

### 6.2 P&L by Quarter (Y1) and Half (Y2–Y3) — $ thousands

| Line item | Y1 Q1 | Y1 Q2 | Y1 Q3 | Y1 Q4 | **Y1** | Y2 H1 | Y2 H2 | **Y2** | Y3 H1 | Y3 H2 | **Y3** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Revenue** | | | | | | | | | | | |
| Subscription revenue (recognized) | 13 | 19 | 50 | 88 | **170** | 730 | 1,720 | **2,450** | 3,930 | 6,890 | **10,820** |
| Services revenue (recognized) | 0 | 25 | 75 | 95 | **195** | 480 | 760 | **1,240** | 980 | 1,180 | **2,160** |
| **Total revenue** | **13** | **44** | **125** | **183** | **365** | **1,210** | **2,480** | **3,690** | **4,910** | **8,070** | **12,980** |
| **COGS** | | | | | | | | | | | |
| Hosting + LLM + 3rd party | 5 | 8 | 15 | 25 | 53 | 90 | 200 | 290 | 450 | 800 | 1,250 |
| Customer support + CSM | 0 | 30 | 50 | 60 | 140 | 180 | 240 | 420 | 360 | 520 | 880 |
| Services delivery cost | 0 | 18 | 53 | 67 | 138 | 336 | 532 | 868 | 686 | 826 | 1,512 |
| **Total COGS** | **5** | **56** | **118** | **152** | **331** | **606** | **972** | **1,578** | **1,496** | **2,146** | **3,642** |
| **Gross profit** | 8 | (12) | 7 | 31 | **34** | 604 | 1,508 | **2,112** | 3,414 | 5,924 | **9,338** |
| Gross margin % | 62% | n/a | 6% | 17% | **9%** | 50% | 61% | **57%** | 70% | 73% | **72%** |
| **Operating expense** | | | | | | | | | | | |
| R&D (eng + ML + product) | 165 | 165 | 250 | 320 | 900 | 740 | 880 | 1,620 | 1,260 | 1,580 | 2,840 |
| S&M (sales + marketing) | 80 | 110 | 240 | 300 | 730 | 1,000 | 1,370 | 2,370 | 1,980 | 2,790 | 4,770 |
| G&A (founder + finance + legal + ops) | 80 | 80 | 110 | 130 | 400 | 280 | 360 | 640 | 460 | 580 | 1,040 |
| **Total opex** | **325** | **355** | **600** | **750** | **2,030** | **2,020** | **2,610** | **4,630** | **3,700** | **4,950** | **8,650** |
| **EBITDA** | (317) | (367) | (593) | (719) | **(1,996)** | (1,416) | (1,102) | **(2,518)** | (286) | 974 | **688** |
| EBITDA margin % | n/m | n/m | n/m | n/m | **(547%)** | n/m | n/m | **(68%)** | (6%) | 12% | **5%** |
| **Cash burn** (≈ EBITDA + working capital) | (340) | (390) | (620) | (760) | **(2,110)** | (1,500) | (1,150) | **(2,650)** | (350) | 900 | **550** |
| **Cash balance** (start: $2,500K seed @ Y1 Q1) | 2,160 | 1,770 | 1,150 | 390 | 390 | 1,890 (after Series A $4.0M raise Y2 Q1) | 740 | 740 | 6,390 (after Series A $6M raise Y3 Q1) | 7,290 | 7,290 |

**Notes on the model:**
- Y1 Q1 starting cash assumes the $2.5M seed closed at month 1.
- Y2 Q1 raise ($4M “late seed extension” or small Series A) is needed because Y1 ends with only $390K cash and Y2 H1 burn is $1.5M. **This is the model’s tightest pinch.** See §9.
- Y3 Q1 raise (~$6M, true Series A) extends runway into Y4 and funds AE expansion.
- Subscription revenue lags ARR by ~3 months on average (ratable recognition; deals close mid-quarter).
- Services revenue is recognized over the implementation period, conservatively 3–4 months.
- COGS line "hosting + LLM" assumes ~$15K/customer/year by Y3, which is conservative for the foundation-model GPU inference burden — could be 1.5x worse if foundation models become THE primary champion across all DFUs (see §11 sensitivity).
- R&D includes infra + DevOps + ML experimentation cost (data labeling, GPU training time).

### 6.3 ARR Bridge Summary

| | Y1 end | Y2 end | Y3 end |
|---|---:|---:|---:|
| Beginning ARR | $0 | $410K | $4.78M |
| + New ARR | $410K | $4.32M | $11.88M |
| + Expansion ARR | $0 | $110K | $1.10M |
| − Churn ARR | $0 | $60K | $600K |
| **Ending ARR** | **$410K** | **$4.78M** | **$17.16M** |
| Implied NRR (Y/Y) | n/a | 112% | 121% |

**ARR-end-of-Y2 ≈ $4.78M** is the headline number to defend in any investor conversation. It is achieved with 23 customers and a $2.5M seed + $4M extension. This is the operating plan.

---

## 7. Hiring Plan

Tied to ARR milestones, not calendar. Comp ranges are US-remote market 2026.

| Hire # | Role | Trigger | Target start | Base | OTE / total comp | Fully-loaded (1.25x) |
|---:|---|---|---|---:|---:|---:|
| 0 | Founder / CEO | Day 0 | Y1 Q1 | $150K | $150K | $190K |
| 1 | Senior full-stack engineer (Python/React) | $0 ARR — needed to ship | Y1 Q1 | $200K | $200K + $40K eq vest | $250K |
| 2 | ML engineer (forecasting + LLM ops) | $0 ARR — protect the wedge | Y1 Q2 | $220K | $220K | $275K |
| 3 | AE #1 (founder-replicating seller) | $400K ARR signed | Y1 Q4 | $180K | $360K (50/50) | $450K |
| 4 | Customer success / professional services lead | 5 paying customers | Y2 Q1 | $180K | $200K | $250K |
| 5 | SE / solutions consultant | AE #1 carrying 4+ open opps | Y2 Q1 | $220K | $260K | $325K |
| 6 | Engineer #2 (backend/data) | $1M ARR | Y2 Q2 | $190K | $190K | $240K |
| 7 | AE #2 | $1.5M ARR & AE #1 ≥ 80% quota | Y2 Q3 | $180K | $360K | $450K |
| 8 | SDR #1 | AE #2 hired | Y2 Q3 | $75K | $115K | $145K |
| 9 | Marketing lead | $2M ARR | Y2 Q3 | $180K | $200K | $250K |
| 10 | VP Engineering | $4M ARR (= end of Y2) | Y3 Q1 | $260K | $300K + 1.0% eq | $375K |
| 11 | VP Sales | $5M ARR | Y3 Q1 | $250K | $500K + 0.75% eq | $620K |
| 12–14 | AE #3, #4, #5 | VP Sales onboard | Y3 Q1–Q2 | $180K | $360K each | $450K each |
| 15 | SDR #2 | AE #3 hired | Y3 Q1 | $75K | $115K | $145K |
| 16 | Engineer #3 + ML #2 | $5M ARR | Y3 Q1 | $200K | $200K each | $250K each |
| 17 | CSM #2 | 25 customers | Y3 Q2 | $160K | $190K | $240K |
| 18 | Implementation consultant #2 | services revenue ≥ $1.5M run-rate | Y3 Q2 | $150K | $170K | $215K |
| 19 | VP Customer Success | 35 customers | Y3 Q3 | $230K | $280K + 0.5% eq | $350K |
| 20 | Demand gen marketing manager | $8M ARR | Y3 Q3 | $150K | $170K | $215K |

**Headcount trajectory:** Y1 end = 4 (founder + 2 eng + AE #1). Y2 end = 11. Y3 end = 21. Year-3 R&D headcount ratio is **38% of total**, S&M is **48% of total** — both inside vertical SaaS norms.

**Key principle:** the platform is 70 routers / 88 SQL files / 90 spec files of accumulated functional depth — this is the founder’s moat. The first 5 hires must protect that moat, not dilute it. The CTO question is deferred until after Series A; until then, the founder is the de facto CTO with engineer #1 as architectural deputy.

---

## 8. Capital Required — Three Scenarios

### Scenario A — Bootstrap / consulting-funded
- **Capital source:** founder personal savings $200–300K + 2–3 paid design partners at $50K each = $350–450K total cash. Founder keeps day job for 12 months OR consults part-time at $300/hr to cover personal burn.
- **Team Y1:** founder + 1 part-time eng. No AE. No marketing.
- **ARR target by Y2 end:** $1.0–1.5M (10–15 logos, mostly Starter via warm intros from beverage-alcohol distributor network).
- **Pros:** Zero dilution. Founder owns 100%. Forces ruthless prioritization.
- **Cons:** 24-month delay vs scenario (b)/(c) on real GTM. By Y3 a well-funded foundation-model-native competitor likely owns the segment narrative. Founder burnout risk is severe. No ability to handle a sudden enterprise opportunity.
- **When this works:** Only if the founder is genuinely indifferent to outcome size and is happy at $20–30M ARR / $5M FCF in year 7.

### Scenario B — Angel + small seed ($2.5M) — RECOMMENDED
- **Capital source:** $500K from angels / SC industry operators (VPs at large distributors, retired SCM execs from McKesson/Boeing/Daimler) + $2.0M from a vertical-SaaS focused seed fund (e.g. Ridge, Bowery, Costanoa, Foundation, Zetta — all of which have done SC or vertical AI deals).
- **Team Y1:** founder + senior FS eng + ML eng + AE #1 (Q4). 4 people.
- **Team Y2 (post $4M extension Q1):** 11 people.
- **ARR target by Y2 end:** $4.78M.
- **Dilution:** ~22% to investors at seed (post-money $11M, pre-money $8.5M) + ~18% to investors at Series A. Founder ends Y3 at ~50–55%.
- **Pros:** Realistic enterprise GTM ramp. Cash to survive a bad quarter. Industry angels open doors. Series A is achievable from $4–5M ARR base.
- **Cons:** Capital pressure means hiring AE before Y1 product is fully battle-tested. Founder must spend ~30% of time fundraising in Y2.

### Scenario C — Institutional seed ($6M)
- **Capital source:** institutional seed + small Series A blend ($6–8M total in 12 months). Lead from a tier-1 vertical SaaS investor (ICONIQ, Bessemer, Insight, Felicis, Battery, M12, Microsoft VC).
- **Team Y1:** founder + 4 hires + AE + SE. 6–7 people by Y1 end.
- **Team Y2:** 16–18 people. Pre-built field org.
- **ARR target by Y2 end:** $6–8M (more AEs ramping earlier — but also more S&M waste in Y1 if product is not yet repeatable).
- **Dilution:** ~30% at seed (post-money $20M) + ~20% at Series A. Founder ends Y3 at ~40%.
- **Pros:** Series A is teed up. Bigger swing on outcome. Bench strength to handle a big enterprise deal.
- **Cons:** **Premature scaling risk is extreme.** The platform has ~3K backend tests and is single-tenant — if you hire 4 AEs before solving multi-tenancy and SOC 2, half the new logos will stall in security review and AE quota attainment will be ~30%, killing CAC and the Series A narrative.

### Recommendation: Scenario B
Scenario B is the right path. Three reasons:

1. **The product is functionally deep but operationally early.** Single-tenant, no SOC 2, no SSO, no enterprise-grade audit trail. Hiring 4 AEs before solving these (Scenario C) burns CAC. Hiring zero AEs (Scenario A) wastes 24 months of accumulated product lead.
2. **The founder’s industry credibility maps to mid-market design partners.** $2.5M is enough to pay for a senior eng + ML + AE for 18 months. That converts founder credibility into 5–10 paying customers, which converts paying customers into a Series A story.
3. **It preserves optionality.** From a $4–5M ARR base at Y2, both an institutional Series A *and* a strategic acquihire by a tier-1 ERP are live options. Scenario A locks out the strategic option. Scenario C locks out the bootstrap-to-profitability option.

---

## 9. Funding Milestones & Comparables

| Round | When | Amount | Valuation (post) | Trigger / proof points | Comparables |
|---|---|---:|---:|---|---|
| **Pre-seed angel** | Y1 Q1 | $0.5M | $5M | Founder + working product (already exists) + 2 LOI paid design partners | Anaplan’s 2008 angel; ToolsGroup’s 1990s self-funded period; many YC 2024 SC AI seeds at $2M post |
| **Seed (priced)** | Y1 Q1 (or rolled into pre-seed) | $2.0M | $10M | Working platform + 1 paid design partner signed | Coupa Series A 2008 ($7.5M @ $20M post); Kinaxis early rounds (1990s); RELEX seed 2005 |
| **Seed extension / small A** | Y2 Q1 | $4M | $25–30M | $410K ARR + 5 customers + first non-network logo | Vertical SaaS 2024 norms (Tive $54M Series B, Project44 — but earlier comps: 2018 Flexport seed extension) |
| **Series A** | Y3 Q1 | $6M (could go to $10M depending on ARR) | $60–80M | $4.78M ARR + 23 customers + 110%+ NRR + repeatable AE motion | Coupa Series B 2010 ($7.5M @ $43M); Anaplan Series A 2010 ($10.5M); o9 2017 first round; recent: Pando.ai 2023 Series A ($30M, but already at $10M ARR) |
| **Series B** | Y4 Q3 | $20–30M | $200–300M | $20M ARR, NRR 120%, GRR 90%, magic 1.0+, burn multiple < 1.5 | Kinaxis IPO trajectory (2014 IPO at $135M revenue); Anaplan (2018 IPO @ $240M revenue); o9 unicorn 2020 |

**Implication:** the operating model in §6 is built to be precisely the right shape at end-of-Y2 (~$5M ARR, 100%+ NRR, magic ≈ 1.0, burn multiple ≈ 1.5x) to raise a clean Series A from the standard SC SaaS investor set.

---

## 10. Risks & Sensitivities

Five sensitivity scenarios. Each shows the impact on Y3 ending ARR, Y3 EBITDA, and cumulative cash needed by end of Y3.

| Scenario | Mechanism | Y3 ARR Δ | Y3 EBITDA Δ | Cum. cash needed by Y3 end | Mitigation |
|---|---|---:|---:|---:|---|
| **Base case** | As §6 | $17.2M | +$0.7M | $12.5M raised | — |
| **Slow sales cycle (1.5x)** | Avg cycle 5mo → 7.5mo. New logos Y2/Y3 −33%. | $12.0M (−30%) | −$2.6M | $15M (need extra $2.5M Series A) | Tighten ICP, lean into channel partners |
| **GTM cost 1.5x** | All S&M costs +50% (more events, paid, AEs ramp slower). | $17.2M (no Δ) | −$3.5M | $16M (need $3.5M more) | Cap AE hiring; defer marketing lead 6 months |
| **Churn 2x (10% → 20% gross)** | Logos churn at $1M ARR cohort | $14.5M (−16%) | −$1.5M | $14M | Triple CSM headcount; renewal-management discipline |
| **Foundation-model commoditization** | LLM hosting cost flat but accuracy edge erodes; perceived differentiation drops; ASP −15%, win rate −20% | $11.8M (−31%) | −$3.2M | $16M | Double down on inventory + S&OP + control tower modules; AI agent UX as moat |
| **IP / non-compete dispute with current employer** | 6-month delay + $400K legal + 1-year non-solicit on top 5 prospects | $10M (−42%) | −$3M (legal + delay) | $17M (need to extend runway 6mo) | Pre-negotiate IP carve-out before launch; clean repo provenance audit; founder steps back temporarily if needed |
| **Combined "bad case" (slow sales + churn 2x + GTM 1.3x)** | All three at once | $7.5M (−56%) | −$5M | $20M (Series A must be $10M+) | Cut burn 30%; pause AE hiring; emergency bridge |

The single most important sensitivity is the **IP/non-compete risk**. Before any external capital is taken, a clean IP audit and (ideally) written carve-out from the founder’s current employer is non-negotiable. Investors will bake this into diligence.

The next most important is **services margin compression**. If services attach stays at 50% with 30% margin, blended GM stays in the low 70s instead of climbing to 78–80%. This kills the SaaS multiple at Series A. Mitigation: invest early (Y2 Q4) in a partner-led implementation playbook so Y3+ services revenue is mostly partner-fulfilled.

---

## 11. Founder Economics & Equity

**Starting point:** Founder owns 100% pre-funding.

**Post-funding cap table (Scenario B trajectory):**

| Round | Pre-money | Post-money | New investor % | Founder % | Option pool % | Investor cumulative % |
|---|---:|---:|---:|---:|---:|---:|
| Day 0 | — | — | 0% | 100% | 0% | 0% |
| Pre-seed/seed combo ($2.5M @ $10M post) | $7.5M | $10M | 25% (incl. 10% top-up option pool) | 75% (after pool) → **65%** | 10% | 25% |
| Seed extension ($4M @ $28M post) | $24M | $28M | 14% (no pool top-up) | **56%** | 10% | 39% |
| Series A ($6M @ $60M post) | $54M | $60M | 10% (+ 5% pool top-up) | **48%** | 12% | 49% |
| Series B ($25M @ $250M post) | $225M | $250M | 10% (+ 5% pool top-up) | **41%** | 14% | 59% |

**Founder ends Series A at ~48%, ends Series B at ~41%.** This is **inside the typical band of 35–50% for a single-founder SC SaaS at Series B**. (Anaplan founder Michael Gould was at ~25% by IPO; Coupa founders were lower; bootstrapped Kinaxis founders kept more.)

**Vesting / cliff:**
- Founder shares: 4-year vest, 1-year cliff, single-trigger acceleration on 100% acquisition (negotiate at seed). Already-built IP can be assigned in via a founder-friendly IP assignment with full vesting at incorporation, but investors will push back — common compromise is 25% pre-vested, 75% on standard 4-year vest.
- Option pool: standard 4-year vest, 1-year cliff for all hires. VP-level may negotiate accelerated vest on involuntary termination after change-of-control (double-trigger).

**Co-founder / CTO equity (if hired pre-seed):** typical 15–25% pre-funding for a true co-founder CTO with 4-year vest. If hired post-seed as VP Eng, 1.0–2.0% with full 4-year vest. The model in §7 assumes the founder operates as CTO through Series A (no co-founder), then hires a VP Engineering at 1.0% in Y3 Q1.

**Founder cash compensation:** $150K Y1 (under-market on purpose), $200K Y2, $275K Y3 (post-Series A, board-approved).

---

## 12. Key Financial KPIs to Instrument from Day 1

These are the lines on every monthly board deck. Build the data plumbing for them in month 1, before there is anything to measure.

| KPI | Definition | Y1 target | Y2 target | Y3 target |
|---|---|---|---|---|
| **ARR** | Annualized contracted recurring revenue at month-end | $410K | $4.8M | $17M |
| **NRR** | (Start ARR + Expansion − Contraction − Churn) / Start ARR, T12M | n/m | 110% | 120% |
| **GRR** | (Start ARR − Contraction − Churn) / Start ARR, T12M | n/m | 92% | 90% |
| **Magic number** | (Net new ARR × 4) / prior-Q S&M spend | 0.4 | 1.0 | 1.1 |
| **CAC payback** | Loaded CAC / (Subscription GP per month) | 24mo | 16mo | 14mo |
| **Subscription gross margin** | (Subscription rev − hosting − support) / Subscription rev | 78% | 80% | 82% |
| **Burn multiple** | Net cash burn / Net new ARR, T12M | 5.0x | 1.5x | 0.9x |
| **Rule of 40** | YoY growth % + EBITDA margin % | n/a | n/a | 40+ |
| **Pipeline coverage** | Open opps × probability / next-Q quota | 3.0x | 3.0x | 4.0x |
| **AE quota attainment (avg)** | Closed ARR / quota | 70% | 75% | 80% |
| **Logos at end-of-period** | Distinct paying customers | 5 | 23 | 56 |
| **ACV trend** | Avg new-logo ACV, T6M | $130K | $240K | $360K |
| **Time-to-first-value** | Contract sign → first production forecast | 14 weeks | 10 weeks | 6 weeks |
| **LLM/AI cost per customer** | Anthropic + OpenAI spend / customers | $4K | $3K | $3K |
| **Hosting cost per customer** | Postgres + Redis + GPU / customers | $20K | $15K | $12K |

The two non-obvious ones — **time-to-first-value** and **LLM cost per customer** — matter because they are the leading indicators of (a) services margin and (b) gross margin compression as foundation-model usage grows. Wire these into the product itself (the FVA framework and `ai_call_log` table already exist — instrument them for finance, not just for ML diagnostics).

---

## 13. 12-Month Operating Plan (Scenario B — Recommended Path)

| Month | Build | GTM | Hiring | Fundraise |
|---|---|---|---|---|
| **M1** | Multi-tenancy spike (single-DB-per-tenant); SOC 2 Type 1 kickoff | Founder closes pre-seed angel ($500K) | Founder full-time | Pre-seed wires |
| **M2** | Tenant isolation in API + DB; SSO scaffolding (SAML); design partner #1 contract signed | Pitch seed deck to 15 vertical SaaS funds | Hire #1: Senior FS engineer | Term sheet conversations |
| **M3** | Design partner #1 onboarded (legacy beverage distributor); customer-specific data adapter pattern | Design partner #1 paying $50K; design partner #2 LOI | Hire #2: ML engineer | Seed term sheet selected |
| **M4** | Multi-tenant complete for forecasting + inventory modules; first SOC 2 audit window | Design partner #2 onboarding; founder + #1 doing customer-success work | — | Seed close ($2M) — **$2.5M total in bank** |
| **M5** | AI Planning Agent hardening (rate limits, cost caps); production deployment runbook | Design partner #3 contract; first cold-outbound activity | — | Board #1 |
| **M6** | Feature freeze for v1.0 GA; SOC 2 audit fieldwork | First non-network sales conversation; demo environment polished | — | — |
| **M7** | v1.0 GA shipped; Postgres scale tests at 5x current data volume | First non-network paid POC starts ($25K POC fee, credit to ACV) | Hire #3: AE #1 (October–November) | Series A pitch deck draft |
| **M8** | Performance burn-down on top 10 slow endpoints; usage analytics | AE #1 ramping; founder closes design partner #3 | AE #1 onboard | — |
| **M9** | Champion strategy v2 (foundation model competition); customer-enriched models default-on | First non-network logo CLOSED — $180K Professional. Total ARR: $230K | — | — |
| **M10** | Tenant-level usage telemetry → billing event stream | AE #1 closes 1 deal ($120K Starter); design partner #1 renews at full price ($230K) | — | Sketch Series A milestone tree |
| **M11** | Compliance: SOC 2 Type 1 report received; HIPAA-readiness gap analysis (for pharma/medical distributors) | AE #1 working 6 open opportunities; Q4 push | — | Begin late-seed extension conversations |
| **M12** | Y2 product roadmap finalized; planning v2 spec (forecast at customer-DC grain in production) | **Y1 close: $410K ARR, 5 logos, 1 lost POC.** Magic ≈ 0.4 (acceptable for Y1). | — | Late-seed extension term sheet (target $4M @ $25–30M post) |

**Y1 board-meeting cadence:** monthly for first 6 months, then every 6 weeks. Standing slides: ARR walk, pipeline coverage, burn vs plan, hiring vs plan, NPS / CSAT from design partners, top 3 product risks, top 3 GTM risks.

---

## Appendix A — Cost-of-Goods Detail

Per-customer COGS, steady state Y3:

| Component | Per Starter | Per Professional | Per Enterprise |
|---|---:|---:|---:|
| Postgres (multi-tenant pool) | $2K | $5K | $20K |
| Redis | $0.5K | $1K | $3K |
| Foundation model GPU inference (T4/A10G or replace with managed Bedrock) | $1K | $4K | $15K |
| LLM API (Anthropic/OpenAI for AI Planning Agent + Tuning Chat) | $1K | $3K | $8K |
| Observability (Datadog/Grafana Cloud) | $0.5K | $1K | $4K |
| Backup, storage, S3 | $0.5K | $1K | $3K |
| Customer support FTE allocation | $3K | $8K | $25K |
| **Total COGS / customer** | **$8.5K** | **$23K** | **$78K** |
| % of ACV | 17% | 9% | 8% |
| Implied subscription GM | 83% | 91% | 92% |

The model in §6 uses a more conservative blended subscription GM of 80–82% to account for service-tier mix and the fact that early customers will run on over-provisioned infrastructure during the multi-tenancy maturation period.

---

## Appendix B — What This Model Does NOT Cover

- **International expansion** — model is 100% North America for Y1–Y3. EMEA push begins at $25M ARR (Y4+).
- **Channel / SI partner economics** — assumed zero partner-sourced ARR through Y3. A regional SI program (e.g. with a beverage / CPG-focused boutique) could add 10–20% to Y3 new ARR if launched in Y2 H2.
- **M&A** — no acquired ARR or technology tuck-ins modeled.
- **Open-source / community edition** — not part of the recommended go-to-market. If pursued, would shift R&D mix and add a developer-relations cost line.
- **Regulated-vertical premium pricing** (pharma distribution, defense aftermarket) — could lift Enterprise ACV by 30–50% but adds compliance cost (HIPAA, ITAR) — not modeled.

---

**End of document.**

Next reviews: monthly during fundraise, quarterly thereafter. Owner re-baselines the operating plan after seed close, after Series A close, and before Series B kickoff.
