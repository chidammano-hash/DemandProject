# Monetization Plan — Supply Chain Command Center

A consolidated business model and execution plan for commercializing the Supply Chain Command Center (SCCC) platform as a vertical-SaaS product for mid-market B2B distributors.

This README is the executive synthesis. The five companion documents go deep on each area — read them in the order numbered.

## The Five Working Documents

| # | Document | Author angle | Words |
|---|---|---|---|
| 01 | [Market & ICP](01-market-and-icp.md) | SaaS GTM strategist | 4,054 |
| 02 | [Pricing & Packaging](02-pricing-and-packaging.md) | SaaS pricing strategist | 3,618 |
| 03 | [Go-to-Market](03-go-to-market.md) | CRO / head of GTM | 4,541 |
| 04 | [Product Readiness](04-product-readiness.md) | Fractional CTO / commercial-readiness audit | 6,047 |
| 05 | [Financials & Business Plan](05-financials-and-business-plan.md) | SaaS CFO / venture operator | 6,108 |

---

## The Business in One Page

**What we sell.** A unified supply chain planning platform — demand forecasting (multi-model ML + foundation models, per-cluster strategies, SHAP explainability, champion/challenger), inventory planning (safety stock, fill rate, action feed), S&OP, control tower, customer analytics, and an AI/LLM planner copilot — sold as SaaS to mid-market North American B2B distributors.

**Why we win.** Tier-1 platforms (Blue Yonder, o9, Kinaxis) are too expensive and too slow to deploy for mid-market distributors. Tier-2 incumbents (ToolsGroup, Logility, SAP APO, Demand Solutions) have aging tech, weak AI, and clunky UX. SCCC is a modern, AI-native, pre-integrated unified platform priced 30–60% below the legacy enterprise vendors. The wedge is **beverage-alcohol distribution** — the founder has 20 years of operator credibility there and the product was already engineered for three-tier-distribution complexity.

**Who buys it.** Primary ICP: North American B2B distributors / wholesalers, US$500M–$5B revenue, 30K–500K SKUs, 5–75 planners, currently on ToolsGroup / Logility / SAP APO / Demand Solutions / spreadsheets. Secondary: beverage-alcohol distributors $1B+ (vertical wedge), broader F&B distribution, industrial distribution.

**How we price it.** Hybrid: edition platform fee + per-10K-SKU meter + add-on modules. Three editions: **Starter $48K** (≤25K SKUs), **Professional $180K** (the center of gravity, $150M–$1.5B customers), **Enterprise $480K** (>$1B customers). Blended new-logo ACV: ~$240K. Lead value driver in every sales call: **inventory working-capital release** — a balance-sheet number the CFO can verify this quarter.

**How we sell it.** Sales-led with product-assist (snapshot demo tenant, paid 60-day POV, accuracy-proof PDF auto-generated from POV data). Buying committee: VP Supply Chain (champion), Director Demand Planning, Director Replenishment, CIO/CTO, CFO. POV → close in 4–6 months. Average sales cycle 6–9 months for Pro, 9–12 months for Enterprise.

**Where the money goes.** **Recommended capital path: Scenario B — $2.5M angel/seed → $4M extension at Y2 Q1 → $6M Series A at Y3 Q1 (~$12.5M cumulative).** Hires sequenced to ARR triggers. Y2 ARR target: **$4.78M** with ~112% NRR, magic number ≈ 1.0 — the right shape to raise a clean Series A from the standard SC-SaaS investor set.

**The single biggest risk.** **IP / non-compete with the founder's current employer (SGWS).** This was built while employed. Until a written assignment-back or carve-out exists, no investor conversation, no customer conversation, no design partner is safe. Estimated 6-month delay and $400K legal exposure if unresolved before launch. Resolve this before anything else.

---

## What Needs to Be Done to Monetize

The full prioritized list. Items are sequenced — later items depend on earlier ones being resolved.

### Phase 0: Legal & Foundational (Months 0–3, before any commercial activity)

| # | Item | Owner | Effort | Why blocking |
|---|---|---|---|---|
| 0.1 | **Resolve IP ownership with SGWS.** Negotiate a written assignment-back, license-back, or clean carve-out for SCCC. Engage employment counsel + IP counsel. Audit the codebase for SGWS-specific identifiers, branding, ticket links, employee names, customer data — purge before any external code share. | Founder + IP attorney | M | #1 risk per [05](05-financials-and-business-plan.md). Without this, every downstream step is at risk. |
| 0.2 | **Form the company.** Delaware C-Corp (required for institutional capital). 83(b) election. Cap table set up (Carta or Pulley). Stock plan. Founder vesting with reverse cliff if appropriate. | Founder + corp counsel | S | Required before any contracts, hires, or fundraise. |
| 0.3 | **Sanitize the codebase.** Per [04](04-product-readiness.md): remove `admin123` seed, fix anonymous-as-admin auth fallback in `common/auth.py:166-169` to fail-closed, purge SGWS Jira/Confluence references, scrub demo data with real customer/product names. | Founder + 1 engineer | S | Cannot demo the product to outsiders until this is done. |
| 0.4 | **Brand & legal artifacts.** Company name, domain, marks, trademark search. ToS, Privacy Policy, MSA template, DPA template, Sub-processor list, AUP. | Founder + brand designer + corp counsel | M | Required before signing the first design partner. |
| 0.5 | **Personal financial runway.** Founder bridge for 12–18 months ($200–500K) if going Scenario B angel route, or close $2.5M seed first if institutional money funds the salary from day one. | Founder | — | Determines hiring timing in Phase 1. |

### Phase 1: Productize (Months 3–9, in parallel with first design partner conversations)

| # | Item | Owner | Effort | Why required |
|---|---|---|---|---|
| 1.1 | **Multi-tenancy.** Recommended: silo'd Postgres per tenant + per-request pool routing. Adds tenant routing layer in `api/core.py` / `common/core/db.py`. Per-tenant ML model artifacts, per-tenant data, per-tenant config. | 1 senior eng | L (4–6 eng-months) | Selling to >1 customer is impossible without it. P0. |
| 1.2 | **Per-tenant API keys + SSO (SAML/OIDC).** Replace single global `API_KEY` env var with per-tenant key management. Add WorkOS or Frontegg for SSO + SCIM provisioning. RBAC roles per module. | 1 mid eng | M (1.5–2 eng-months) | Enterprise security review will fail without these. P0. |
| 1.3 | **First ERP/cloud connector.** Pick one for the bev-alc / F&B beachhead — likely SAP S/4 + a Snowflake connector. Per-customer config & schema-mapping UI. | 1 mid eng + founder for domain mapping | L (3 eng-months for first; 1 month each for follow-ons) | No enterprise buyer ingests via CSV upload. P0 for first paid customer. |
| 1.4 | **Stripe + metering + invoicing.** Subscription billing wired to the per-10K-SKU meter from [02](02-pricing-and-packaging.md). Customer billing portal. Tax via Stripe Tax. | 1 eng | M (1–1.5 eng-months) | Required before second paid customer. |
| 1.5 | **Admin / customer-success tooling.** Internal UI to provision tenants, impersonate (with audit log), see usage, see model performance per tenant, manage entitlements. | 1 eng | M (1.5–2 eng-months) | Required for >3 tenants. |
| 1.6 | **Observability + on-call.** Datadog or Grafana Cloud, Sentry, status page (Statuspage.io), PagerDuty. Backup + RPO/RTO targets. | 1 eng | S–M | 99.9% SLA promises require this in place day one. |
| 1.7 | **SOC 2 Type I engagement.** Pick GRC tool (Vanta or Drata). Pen test (NCC, Bishop Fox, or Cobalt). 4–6 months to Type I, then 6 months observation to Type II. | Founder + part-time GRC consultant | M ($30–60K + Vanta $7–15K/yr) | Tier-1 prospects will not move past procurement without at least Type I in progress. |

### Phase 2: First Money (Months 6–12)

| # | Item | Owner | Effort | Why required |
|---|---|---|---|---|
| 2.1 | **3–5 design partners.** 50–70% discount, 12-month commit, written reference + named ROI as part of the contract. Lead from founder's network: SGWS adjacency (with IP cleanup done), Boeing Distribution alumni network, McKesson alumni network, Daimler alumni network, plus 1–2 bev-alc distributors via WSWA / NACDS. | Founder | — | Without 3 referenceable logos, every prospect conversation devolves into a custom POC per [01](01-market-and-icp.md). |
| 2.2 | **First Solutions Engineer hire.** Per [03](03-go-to-market.md): SE #1 is the first hire after the founder, before AE #1. The founder can sell; nobody else can demo the product depth. Target: senior pre-sales engineer from ToolsGroup / Logility / o9 / Blue Yonder. | Founder | $200–250K fully loaded | Hire happens at design-partner #2. |
| 2.3 | **Standardized POV motion.** 60-day paid POV ($25–50K, credited to first-year ACV). Data ingest in week 1, baseline accuracy in week 3, lift demonstration in week 6, go/no-go in week 8. Auto-generated accuracy-proof PDF tied to the WAPE/working-capital deltas. | Founder + SE #1 | M | Per [03](03-go-to-market.md), this is the conversion engine. |
| 2.4 | **Reference assets.** Case study per design partner with WAPE delta, fill-rate delta, working-capital release in $. Quote from the customer's VP Supply Chain. Public if possible, private-redacted if not. | Founder + marketing contractor | S each | Lead generation for AE #1 hire depends on these. |

### Phase 3: GTM Engine (Months 9–18)

| # | Item | Owner | Effort | Why required |
|---|---|---|---|---|
| 3.1 | **AE #1 hire.** Quota: $1.2M ACV in year one. Background: senior AE from a tier-2 SC vendor. OTE $250–320K, 50/50 base/variable. Hire trigger: 2 paid customers, $400K cumulative ACV, SOC 2 Type I awarded. | Founder | $300K+ fully loaded | Founder cannot scale past 5 customers personally. |
| 3.2 | **CSM #1 hire.** Trigger: 3 paying customers. Background: from a tier-2 SC vendor. Owns NRR target (target 110–130%). | Founder | $180–220K fully loaded | NRR is half the Series A story. |
| 3.3 | **Marketing / demand-gen lead.** Trigger: $1.5M ARR. Owns SEO, planner-focused thought leadership, ABM for top-200 accounts, LinkedIn paid, conference presence (Gartner SCP Symposium, NRF, Manifest, NACDS, WSWA). | Founder | $200K | AE #1 cannot prospect to quota without a pipeline machine. |
| 3.4 | **Analyst briefings.** Gartner SCP team, IDC, Forrester. Goal: appear in the next Gartner Magic Quadrant for Supply Chain Planning Solutions or Forrester Wave (typically a 12–18 month inclusion cycle). | Founder | $30–60K incl. analyst contracts | Tier-1 prospects RFP shortlist by analyst inclusion. |
| 3.5 | **First two additional connectors.** Pick based on first-customer telemetry: most likely Oracle, NetSuite, or Microsoft Dynamics + Databricks. | 1 eng | M each | Each unlocks a slice of the addressable pipeline. |
| 3.6 | **SOC 2 Type II.** 6 months observation after Type I. Required for any deal >$200K ACV. | GRC consultant | $50–80K incremental | Annual recertification thereafter. |

### Phase 4: Series A Prep (Months 18–24)

| # | Item | Owner | Effort | Why required |
|---|---|---|---|---|
| 4.1 | **Hit Y2 ARR ≈ $4.78M with NRR ≥ 110%, GRR ≥ 90%, magic number ≈ 1.0.** Per [05](05-financials-and-business-plan.md). | All | — | This is the Series A bar in vertical SaaS. |
| 4.2 | **VP Engineering hire.** Trigger: ~$3M ARR. Founder steps back from day-to-day code, owns roadmap and customer escalation. | Founder | $350K+ | Required before Series A diligence. |
| 4.3 | **VP Sales hire.** Trigger: $4M ARR + 2 quota-carrying AEs. Builds the AE bench from 2 → 6. | Founder | $400K+ OTE | Required for the Series A growth narrative. |
| 4.4 | **Cap table + 409A + diligence pack.** Updated 409A every 12 months and at fundraise. Data room: financials, MRR cohorts, sales metrics, customer references, ARR bridge, SOC 2 reports, IP assignment chain. | Founder + corp counsel | M | Series A diligence will fail in 2 weeks without this. |

---

## Cross-Cutting "Must Resolve" Decisions

These are decision points the founder owns. They determine the rest of the plan.

1. **Capital path.** Scenario A (bootstrap, founder keeps day job 12–18 months), Scenario B (recommended: $2.5M angel/seed → $4M extension → $6M Series A), Scenario C ($6M institutional seed). Each has a different hiring sequence and risk profile. See [05 §9](05-financials-and-business-plan.md).
2. **SGWS exit timing.** Day-job overlap with the founder is feasible only in Scenario A. In Scenario B/C, founder needs to be full-time within 60 days of close. SGWS departure timing is also entangled with the IP cleanup negotiation — leverage exists while employed.
3. **Vertical-first vs horizontal-first.** Recommended in [01](01-market-and-icp.md) and [03](03-go-to-market.md): bev-alc vertical wedge in Year 1, broaden in Year 2. The contrary view: skip the vertical wedge and go straight at mid-market F&B distribution. The vertical play has higher conversion but smaller TAM. Founder's call.
4. **Single-tenant hosted vs SaaS-only.** Some tier-1 mid-market distributors will demand single-tenant deployment in their VPC. Per [04 §11](04-product-readiness.md), recommended sequence: SaaS-first, single-tenant hosted by Y2 (Terraform-installable), on-prem only by exception (Y3+). Saying "SaaS-only" loses ~25% of qualified pipeline early.
5. **Hiring SE #1 vs AE #1 first.** Per [03](03-go-to-market.md), SE first. This is contrary to the standard SaaS playbook (AE first). Defendable because the founder can sell, but cannot scale demos.

---

## What I Would Do First (Recommended 90-Day Plan)

If the founder asked "what do I do in the next 90 days," this is the answer:

**Days 0–30**
- Engage IP/employment counsel. Open formal conversation with SGWS legal about IP carve-out. (Item 0.1)
- File Delaware C-Corp, 83(b), open a business bank account. Set up Carta. (Item 0.2)
- Pick a name, secure domain + trademark search. (Item 0.4 partial)
- Have one engineer (contractor or trusted ex-colleague) start Item 0.3 codebase sanitization in parallel.

**Days 30–60**
- Draft term sheet for 3 design partners — even before the product is multi-tenant — at 70% discount, 12-month commit, written reference clause. Begin warm intro conversations from SGWS / Boeing / McKesson / Daimler networks (without violating non-compete — surface only after IP cleanup and counsel signoff).
- Begin angel/seed conversations. Target list: SC-savvy operator angels (former Blue Yonder, o9, Kinaxis, RELEX, ToolsGroup execs); vertical-SaaS focused seed funds (Bowery, Costanoa, Resolute, Forerunner for B2B vertical, Felicis).
- Draft the MSA, DPA, ToS templates. (Item 0.4 finish)

**Days 60–90**
- Close angel/seed round of $2.0–2.5M (Scenario B). Lead with operator angels for credibility, top up with one institutional check.
- Hire first engineer (full-time, equity-heavy). Start Item 1.1 multi-tenancy work.
- Sign first 1–2 design partners on a verbal commit, contract pending IP cleanup completion.
- Engage Vanta + GRC consultant. Begin SOC 2 Type I work. (Item 1.7 starts)

By month 3–4, IP cleanup should be done or near-done. By month 6, multi-tenancy should be live and first paid POV running. By month 9, first paid customer signed. By month 12, 3–5 paying customers, $600K–$1M ARR, SE #1 hired, ready to hire AE #1.

---

## Realistic Outcomes

Per [05](05-financials-and-business-plan.md):

| Milestone | Target | Date (Scenario B) |
|---|---|---|
| First paid customer | 1 | Month 9–12 |
| ARR end of Year 1 | $1.2M | Month 12 |
| ARR end of Year 2 | $4.78M | Month 24 |
| ARR end of Year 3 | ~$12M | Month 36 |
| Series A close | $20M at $80–120M post | Month 30 |
| Cumulative cash burn through Y3 | ~$10M | — |
| Exit horizon | $50–100M ARR, strategic acquisition or growth-equity recap | Year 6–8 |

This is not a billion-dollar IPO trajectory. It is a credible vertical-SaaS business that, executed well, sells for **$300–600M to a Blue Yonder, Coupa, Manhattan Associates, or PE roll-up** in years 6–8.

---

## Honest Risks

1. **IP / non-compete (P0).** Resolve before anything else. See [05 §11](05-financials-and-business-plan.md).
2. **Founder bandwidth.** Doing this part-time while at SGWS works only in Scenario A. Realistically a 6-month bridge maximum.
3. **No reference customers.** Per [01](01-market-and-icp.md), no public ROI numbers exist yet. Until 3 logos with named WAPE/working-capital deltas are landed, the sales motion stalls. This is the highest-leverage Year-1 priority.
4. **Foundation-model commoditization.** Chronos, TimesFM, etc. are getting commoditized. The product's defensibility is **not** the algorithms — it is the unified scope (forecasting + inventory + S&OP + agentic), the operator-grade UX, and the per-cluster + champion/challenger framework. Sales messaging must not lead with "we have foundation models."
5. **Services-margin compression.** [05](05-financials-and-business-plan.md) flags that 30% delivery margin at 50% attach holds blended GM in the low 70s. Mitigation: invest in self-serve ingest and config (Items 1.3 + 1.5) to reduce services attach over time.

---

## Where to Start Reading

- New to the plan: read this README, then [01 Market & ICP](01-market-and-icp.md), then [05 Financials](05-financials-and-business-plan.md) to ground the numbers.
- Engineering-minded: jump to [04 Product Readiness](04-product-readiness.md) — it is the most concrete punch list with specific files cited.
- Sales-minded: read [02 Pricing](02-pricing-and-packaging.md) and [03 GTM](03-go-to-market.md) back-to-back.
- Investor-pitch prep: README + 05 + 01.
