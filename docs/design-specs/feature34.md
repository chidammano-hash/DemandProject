# Feature 34: Inventory Planning Module — World-Class Design Specification

## Executive Summary

The Inventory Planning Module is a unified inventory optimization system for Demand Studio that transforms demand forecasts into optimal inventory decisions. It covers safety stock optimization, replenishment automation, ABC-XYZ classification, what-if simulation, supplier performance tracking, and seasonal buildup planning — designed to compete with SAP IBP, Blue Yonder, Kinaxis, and o9 Solutions.

The module leverages existing Demand Studio infrastructure: champion model forecasts (Feature 15), DFU clustering (Feature 7), seasonality detection (Feature 30), backtest accuracy data (Features 8–9, 12–13, 21, 24–25), and the what-if scenario architecture (Feature 29). It introduces 10 new PostgreSQL tables, 30+ REST API endpoints, a multi-method safety stock engine, LP/gradient/simulation optimization solvers, Monte Carlo simulation, and a React UI with 8 sub-panels.

**Key differentiators over commercial vendors:**

- **Native champion model integration**: Safety stock computed directly from the best-of-models forecast, not a separate demand planning system.
- **Empirical safety stock from backtests**: Forecast error distributions drawn from the backtest archive — no distributional assumptions required.
- **4-method safety stock engine**: Parametric (Normal), Empirical (backtest quantiles), Conformal Prediction (MAPIE), Bootstrap (intermittent demand).
- **11 what-if scenario types** with sub-second single-item response and vectorized batch solvers.
- **Open-source ML/optimization stack**: scikit-learn, PuLP/HiGHS, DEAP, Optuna, MAPIE — no proprietary solver licenses.

**Expected business impact:**

| Metric | Improvement |
|--------|------------|
| Inventory investment reduction | 10–30% |
| Service level improvement | 2–5 percentage points |
| Stockout reduction | 20–50% |
| Excess/obsolete reduction | 15–25% |
| Payback period | 6–12 months |

---

## Table of Contents

- **Part I: Theoretical Foundation**
  - [1. Classical Inventory Models](#1-classical-inventory-models)
    - [1.1 EOQ (Economic Order Quantity)](#11-eoq-economic-order-quantity)
    - [1.2 Newsvendor Model](#12-newsvendor-model)
    - [1.3 Continuous Review Policies](#13-continuous-review-policies)
    - [1.4 Periodic Review Policies](#14-periodic-review-policies)
    - [1.5 Dynamic Lot Sizing](#15-dynamic-lot-sizing)
  - [2. Safety Stock Models](#2-safety-stock-models)
    - [2.1 Standard Safety Stock](#21-standard-safety-stock)
    - [2.2 Service Level Types](#22-service-level-types)
    - [2.3 Intermittent Demand](#23-intermittent-demand)
    - [2.4 Safety Stock Optimization Under Constraints](#24-safety-stock-optimization-under-constraints)
  - [3. Multi-Echelon Inventory Optimization (MEIO)](#3-multi-echelon-inventory-optimization-meio)
  - [4. ABC-XYZ Classification](#4-abc-xyz-classification)
    - [4.1 ABC (Value Analysis)](#41-abc-value-analysis)
    - [4.2 XYZ (Variability Analysis)](#42-xyz-variability-analysis)
    - [4.3 Nine-Cell Matrix Policy Recommendations](#43-nine-cell-matrix-policy-recommendations)
  - [5. Key Inventory Metrics (KPIs)](#5-key-inventory-metrics-kpis)
- **Part II: Data Architecture**
  - [6. Data Requirements — What Already Exists](#6-data-requirements--what-already-exists)
  - [7. New Data Required](#7-new-data-required)
  - [8. Data Volume Estimates](#8-data-volume-estimates)
  - [9. Integration Architecture](#9-integration-architecture)
- **Part III: ML/AI Models & Optimization**
  - [10. Probabilistic Forecasting for Inventory](#10-probabilistic-forecasting-for-inventory)
  - [11. Safety Stock Engine](#11-safety-stock-engine)
  - [12. Optimization Solvers](#12-optimization-solvers)
  - [13. Monte Carlo Simulation Engine](#13-monte-carlo-simulation-engine)
  - [14. Anomaly Detection](#14-anomaly-detection)
- **Part IV: What-If Scenarios**
  - [15. Service Level What-If](#15-service-level-what-if)
  - [16. Lead Time What-If](#16-lead-time-what-if)
  - [17. Demand Variability What-If](#17-demand-variability-what-if)
  - [18. Supply Disruption What-If](#18-supply-disruption-what-if)
  - [19. Budget Constraint What-If](#19-budget-constraint-what-if)
  - [20. Network/Sourcing What-If](#20-networksourcing-what-if)
  - [21. ABC Reclassification What-If](#21-abc-reclassification-what-if)
  - [22. Seasonal Buildup What-If](#22-seasonal-buildup-what-if)
  - [23. Promotion Impact What-If](#23-promotion-impact-what-if)
  - [24. MOQ/Order Multiple What-If](#24-moqorder-multiple-what-if)
  - [25. Scenario Management](#25-scenario-management)
- **Part V: User Interface**
  - [26. Navigation Architecture](#26-navigation-architecture)
  - [27. Dashboard / Overview Panel](#27-dashboard--overview-panel)
  - [28. Inventory Position View](#28-inventory-position-view)
  - [29. Item Detail / DFU Drill-Down](#29-item-detail--dfu-drill-down)
  - [30. Safety Stock Optimizer Panel](#30-safety-stock-optimizer-panel)
  - [31. Replenishment Recommendations Panel](#31-replenishment-recommendations-panel)
  - [32. ABC-XYZ Classification Panel](#32-abc-xyz-classification-panel)
  - [33. What-If Scenario Panel](#33-what-if-scenario-panel)
  - [34. Inventory Projection Chart](#34-inventory-projection-chart)
  - [35. Supplier Performance Panel](#35-supplier-performance-panel)
  - [36. Alerts & Exceptions Panel](#36-alerts--exceptions-panel)
- **Part VI: Competitive Positioning**
  - [37. Vendor Comparison](#37-vendor-comparison)
  - [38. Industry Benchmarks](#38-industry-benchmarks)
  - [39. Expected Business Impact](#39-expected-business-impact)
- **Part VII: Technical Architecture**
  - [40. File Structure](#40-file-structure)
  - [41. API Endpoints](#41-api-endpoints)
  - [42. Python Dependencies (New)](#42-python-dependencies-new)
  - [43. Config File (inventory_config.yaml)](#43-config-file-inventory_configyaml)
- **Part VIII: Implementation Roadmap**
  - [44. Phased Implementation](#44-phased-implementation)
  - [45. Makefile Targets](#45-makefile-targets)
  - [46. Testing Strategy](#46-testing-strategy)
- **Part IX: Appendices**
  - [Appendix A: Mathematical Reference](#appendix-a-mathematical-reference)
  - [Appendix B: Solver Decision Matrix](#appendix-b-solver-decision-matrix)
  - [Appendix C: Integration with Existing Features](#appendix-c-integration-with-existing-features)
  - [Appendix D: Glossary](#appendix-d-glossary)

---

# Part I: Theoretical Foundation

---

## 1. Classical Inventory Models

This section presents the mathematical foundations underlying every computation in the Inventory Planning Module. All formulas are implemented in `common/inventory_engine.py` and are referenced by their section number in code comments for traceability.

### 1.1 EOQ (Economic Order Quantity)

The Economic Order Quantity model determines the optimal order size that minimizes the sum of ordering costs and holding costs under steady deterministic demand.

**Assumptions:** Constant demand rate D, fixed ordering cost K per order, linear holding cost h per unit per year, instantaneous replenishment, no shortages allowed.

**Total Relevant Cost (TRC):**

```
TRC(Q) = (D / Q) × K + (Q / 2) × h
```

Where:
- `D` = annual demand (units/year)
- `Q` = order quantity (units)
- `K` = fixed ordering cost ($/order)
- `h` = holding cost per unit per year ($/unit/year), typically `h = unit_cost × holding_rate`

**Optimal Order Quantity:**

Setting dTRC/dQ = 0:

```
Q* = sqrt(2 × D × K / h)
```

**Derived Quantities:**

```
Optimal number of orders per year:   N* = D / Q*
Optimal cycle time:                  T* = Q* / D  (years)
Minimum total relevant cost:         TRC* = sqrt(2 × D × K × h)
Average cycle inventory:             I_avg = Q* / 2
Annual holding cost:                 HC = (Q* / 2) × h
Annual ordering cost:                OC = (D / Q*) × K
```

Note: At the optimum, HC = OC (holding cost equals ordering cost).

**Extension — EOQ with Quantity Discounts:**

When supplier offers price breaks at quantity thresholds q_1 < q_2 < ... < q_m with unit costs c_1 > c_2 > ... > c_m:

```
For each price break j:
  h_j = c_j × holding_rate
  Q*_j = sqrt(2 × D × K / h_j)
  If Q*_j < q_j, set Q_j = q_j (round up to qualify for discount)
  TRC_j = D × c_j + (D / Q_j) × K + (Q_j / 2) × h_j
Select j with minimum TRC_j
```

**Extension — EOQ with Planned Backorders:**

When backorders are permitted at penalty cost b per unit per year:

```
Q* = sqrt(2 × D × K / h) × sqrt((h + b) / b)
S* = Q* × h / (h + b)
```

Where S* is the maximum backorder level. The total cost decreases because the system deliberately allows some stockouts to reduce holding costs.

**Extension — EOQ with Perishability:**

For items with shelf life T_shelf, the order quantity is constrained:

```
Q* = min(sqrt(2 × D × K / h), D × T_shelf)
```

This prevents ordering more than can be consumed within the shelf life.

### 1.2 Newsvendor Model

The Newsvendor (single-period) model applies to products with a single selling season — leftover stock has reduced salvage value, and unmet demand is lost.

**Setup:**
- `D` = random demand with CDF F(x) and PDF f(x)
- `c` = unit cost
- `p` = selling price
- `v` = salvage value (v < c)
- `Cu` = underage cost (cost of being one unit short) = `p - c` (lost profit)
- `Co` = overage cost (cost of having one unit too many) = `c - v` (loss on salvage)

**Critical Ratio:**

```
CR = Cu / (Cu + Co) = (p - c) / (p - v)
```

**Optimal Order Quantity:**

```
Q* = F^(-1)(CR) = F^(-1)(Cu / (Cu + Co))
```

Where F^(-1) is the inverse CDF (quantile function) of the demand distribution.

**For Normal demand D ~ N(mu, sigma^2):**

```
Q* = mu + z(CR) × sigma
```

Where z(CR) = Phi^(-1)(CR) is the standard normal z-value corresponding to the critical ratio.

**Expected Profit:**

```
E[Profit] = (p - c) × mu - (p - v) × E[max(Q - D, 0)]
```

**Expected Leftover (Overage):**

```
E[max(Q - D, 0)] = (Q - mu) × Phi((Q - mu)/sigma) + sigma × phi((Q - mu)/sigma)
```

**Expected Stockout (Underage):**

```
E[max(D - Q, 0)] = (mu - Q) × (1 - Phi((Q - mu)/sigma)) + sigma × phi((Q - mu)/sigma)
```

**Application in Inventory Planning:** The newsvendor model applies to seasonal pre-builds, promotional inventory, and end-of-life planning — any scenario with a single decision point and uncertain demand.

### 1.3 Continuous Review Policies

In continuous review systems, inventory position is monitored at all times (or effectively in real-time via system updates). An order is triggered when inventory position crosses a threshold.

**Policy (s, Q) — Fixed-Quantity Policy:**

- When inventory position drops to or below reorder point `s`, place an order of fixed quantity `Q`.
- `s` = reorder point = expected demand during lead time + safety stock
- `Q` = order quantity (typically EOQ)

```
s = d_bar × L + SS
Q = Q* (EOQ or constrained variant)
SS = z(alpha) × sigma_D × sqrt(L)
```

Where:
- `d_bar` = average demand per period
- `L` = lead time in periods
- `SS` = safety stock
- `alpha` = cycle service level (Type 1)
- `sigma_D` = standard deviation of demand per period

**Policy (s, S) — Order-Up-To Policy:**

- When inventory position drops to or below `s`, order enough to bring position up to `S`.
- Order quantity varies: `Q = S - IP` where IP is inventory position at time of order.
- Preferred when demand is lumpy or ordering cost is negligible relative to review cost.

```
s = d_bar × L + SS
S = s + EOQ  (approximation)
```

More precisely, S is set to minimize expected cost per cycle. The (s, S) policy is optimal for the general single-item problem with fixed ordering costs (Scarf, 1960).

**Demand During Lead Time (DDLT) Distribution:**

For safety stock computation, we need the distribution of demand during lead time:

```
If demand per period ~ N(mu_d, sigma_d^2) and L is constant:
  DDLT ~ N(mu_d × L, sigma_d^2 × L)

If lead time is also random with mean L_bar and std sigma_L:
  DDLT ~ N(mu_d × L_bar, L_bar × sigma_d^2 + mu_d^2 × sigma_L^2)
```

The second formula accounts for both demand uncertainty and lead time uncertainty — critical for supplier-dependent items.

### 1.4 Periodic Review Policies

In periodic review systems, inventory is checked at fixed intervals (e.g., weekly, monthly). Orders can only be placed at review epochs.

**Policy (R, S) — Fixed-Interval Order-Up-To:**

- Every R periods, observe inventory position and order up to level S.
- Protection period = R + L (must cover both the review interval and lead time).

```
S = d_bar × (R + L) + SS
SS = z(alpha) × sigma_D × sqrt(R + L)
```

Key insight: periodic review requires MORE safety stock than continuous review because the protection period is R + L instead of just L.

**Policy (R, s, S) — Can-Order Policy:**

- Every R periods, check inventory position.
- If position <= s, order up to S.
- If position > s, do not order.
- Reduces unnecessary small orders when inventory is still adequate.

```
s = d_bar × (R + L) + SS_s
S = s + EOQ  (approximation)
```

**Choosing Review Period R:**

- Trade-off: shorter R reduces safety stock but increases review/ordering costs.
- Practical: align with supplier delivery schedules, warehouse receiving capacity, or purchasing team workflow.

```
R_optimal = sqrt(2 × K / (D × h))  (analogous to EOQ cycle time)
```

**Comparison: Continuous vs Periodic Review:**

| Attribute | Continuous (s, Q) | Periodic (R, S) |
|-----------|-------------------|------------------|
| Safety stock | z × sigma × sqrt(L) | z × sigma × sqrt(R + L) |
| System cost | Lower SS | Higher SS |
| Operational cost | Real-time tracking needed | Simpler scheduling |
| Best for | High-value (A) items | Low-value (C) items |
| Order coordination | Difficult | Natural batching |

### 1.5 Dynamic Lot Sizing

When demand varies over a finite planning horizon of T periods, the classic EOQ formula (which assumes constant demand) is suboptimal. Dynamic lot sizing methods determine optimal order quantities for each period.

**Wagner-Whitin Algorithm (Optimal):**

The Wagner-Whitin (1958) algorithm finds the globally optimal solution via dynamic programming:

```
Let d_t = demand in period t (t = 1, ..., T)
Let K = fixed ordering cost
Let h = holding cost per unit per period
Let f(t) = minimum cost to satisfy demand through period t

f(0) = 0
f(t) = min over j in {1, ..., t} of:
  f(j-1) + K + h × sum_{i=j}^{t} (i - j) × d_i

Complexity: O(T^2) time, O(T) space
```

The algorithm exploits the zero-inventory ordering property: it is never optimal to order when ending inventory is positive. This reduces the solution space dramatically.

**Silver-Meal Heuristic:**

A greedy approximation that minimizes cost per period over the horizon covered by each order:

```
Start at period j = 1
For each candidate order spanning periods j to t:
  C(j, t) = (K + h × sum_{i=j}^{t} (i - j) × d_i) / (t - j + 1)
Continue extending t while C(j, t) decreases
When C increases, place order covering periods j to t-1
Set j = t, repeat
```

Silver-Meal typically achieves within 1–2% of Wagner-Whitin optimal cost.

**Least Unit Cost (LUC) Heuristic:**

Similar to Silver-Meal but minimizes cost per unit instead of cost per period:

```
C_unit(j, t) = (K + h × sum_{i=j}^{t} (i - j) × d_i) / sum_{i=j}^{t} d_i
```

**Part Period Balancing (PPB):**

Orders sized so that holding cost approximately equals ordering cost in each cycle — a discrete analogue of the EOQ principle.

**Power-of-Two Policies:**

For multi-item coordination (e.g., items sharing a supplier or transport), constrain order intervals to powers of two (1, 2, 4, 8, ... periods). This guarantees order intervals are nested, enabling joint replenishment. The cost penalty vs. unconstrained optimal is at most 6%.

---

## 2. Safety Stock Models

Safety stock is the buffer inventory held to protect against uncertainty in demand and supply. It is the single most impactful parameter in inventory planning — setting it too low causes stockouts; setting it too high ties up working capital.

### 2.1 Standard Safety Stock

**Demand Uncertainty Only (constant lead time):**

```
SS = z(CSL) × sigma_D × sqrt(L)
```

Where:
- `z(CSL)` = inverse standard normal at cycle service level (e.g., z(0.95) = 1.645, z(0.99) = 2.326)
- `sigma_D` = standard deviation of demand per period
- `L` = lead time in periods (constant)

**Demand + Lead Time Uncertainty:**

When both demand and lead time are uncertain (the common real-world case):

```
SS = z(CSL) × sqrt(L_bar × sigma_D^2 + d_bar^2 × sigma_LT^2)
```

Where:
- `L_bar` = mean lead time
- `sigma_D` = demand standard deviation per period
- `d_bar` = mean demand per period
- `sigma_LT` = lead time standard deviation (in periods)

This is the combined uncertainty formula. The two terms under the square root represent:
1. `L_bar × sigma_D^2` — demand variability amplified over lead time
2. `d_bar^2 × sigma_LT^2` — lead time variability amplified by demand rate

**Forecast-Error-Based Safety Stock:**

When using forecasts (as in Demand Studio with champion models), safety stock should buffer against forecast error, not raw demand variability:

```
SS = z(CSL) × sigma_FE × sqrt(L)
```

Where `sigma_FE` is the standard deviation of forecast errors. This is typically SMALLER than `sigma_D` because the forecast captures predictable patterns (trend, seasonality), leaving only the unpredictable residual.

**Source of sigma_FE in Demand Studio:** Computed from `backtest_lag_archive` — the historical forecast errors from the champion model at the relevant lag.

### 2.2 Service Level Types

Three fundamentally different ways to define "service level" — choosing the wrong one leads to incorrect safety stock.

**Type 1: Cycle Service Level (CSL / alpha)**

Definition: The probability that no stockout occurs during a replenishment cycle.

```
P(DDLT <= ROP) >= alpha
ROP = d_bar × L + z(alpha) × sigma_DDLT
```

Properties:
- Easiest to compute (just the z-value lookup)
- Does NOT consider the magnitude of shortages — a cycle with 1 unit short counts the same as 1000 units short
- Overestimates required safety stock for high-demand items
- Typical values: 90%–99%

**Type 2: Fill Rate (beta)**

Definition: The fraction of demand filled from on-hand stock (no backorder).

```
beta = 1 - E[units short per cycle] / Q
beta = 1 - (sigma_DDLT / Q) × L(z)
```

Where L(z) is the standard normal loss function:

```
L(z) = phi(z) - z × (1 - Phi(z))
```

And phi(z) is the standard normal PDF, Phi(z) is the standard normal CDF.

Solving for z given target beta:

```
L(z) = (1 - beta) × Q / sigma_DDLT
```

This requires iterative solution (Newton's method or table lookup) because L(z) has no closed-form inverse.

Properties:
- More meaningful for business: "we fill 98% of customer orders from stock"
- Accounts for shortage magnitude
- Requires LESS safety stock than Type 1 for the same numeric target (e.g., beta=0.98 requires less SS than alpha=0.98)
- Preferred metric for most supply chain applications

**Type 3: Ready Rate (gamma)**

Definition: The fraction of time that inventory is positive (i.e., item is "in stock").

```
gamma = P(on-hand > 0) at a random point in time
```

Properties:
- Relevant for retail shelf availability
- Harder to compute analytically
- Related to but not equal to fill rate

**Conversion Between Service Levels:**

For a given safety stock level, the three service levels differ:

```
alpha < gamma < beta  (typically, for the same SS)
```

This is why specifying "95% service level" without stating the type is ambiguous and dangerous. The Inventory Planning Module always requires explicit type specification.

### 2.3 Intermittent Demand

Many items (spare parts, slow movers, long-tail SKUs) exhibit intermittent demand: long periods of zero demand punctuated by occasional non-zero orders. Standard Normal-based safety stock fails for these items.

**Syntetos-Boylan Classification:**

Classify items into four categories based on two metrics:

```
ADI = Average Demand Interval = mean(inter-demand periods)
CV^2 = (sigma_demand_size / mean_demand_size)^2
```

| | CV^2 < 0.49 | CV^2 >= 0.49 |
|---|---|---|
| **ADI < 1.32** | Smooth | Erratic |
| **ADI >= 1.32** | Intermittent | Lumpy |

Recommended forecasting method per category:
- **Smooth:** Standard exponential smoothing or ARIMA
- **Erratic:** Simple Moving Average (demand size varies but timing is regular)
- **Intermittent:** Croston's method (demand is sporadic but sizes are similar)
- **Lumpy:** TSB (Teunter-Syntetos-Babai) method (both sporadic and variable)

**Croston's Method:**

Separate exponential smoothing on two components:
1. Non-zero demand sizes: `z_t = alpha × d_t + (1 - alpha) × z_{t-1}` (updated only when d_t > 0)
2. Inter-demand intervals: `p_t = alpha × q_t + (1 - alpha) × p_{t-1}` (where q_t = periods since last non-zero demand)

Forecast per period: `f_t = z_t / p_t`

**TSB Method (Teunter-Syntetos-Babai):**

Improved Croston that incorporates obsolescence risk:

```
z_t = alpha × d_t + (1 - alpha) × z_{t-1}   (when d_t > 0)
P_t = beta × 1{d_t > 0} + (1 - beta) × P_{t-1}
f_t = P_t × z_t
```

Where P_t is the estimated demand probability per period (decays toward zero for obsolescent items).

**Safety Stock for Intermittent Demand:**

Normal approximation fails because demand is not normally distributed. Options:

1. **Empirical quantile from bootstrap:** Resample demand history, compute DDLT distribution directly.
2. **Compound Poisson model:** Demand = Poisson(arrivals) × Gamma(sizes). Use numerical convolution for DDLT.
3. **Negative Binomial approximation:** When demand is overdispersed relative to Poisson.

The Inventory Planning Module uses bootstrap simulation for all items classified as Intermittent or Lumpy by the Syntetos-Boylan scheme.

### 2.4 Safety Stock Optimization Under Constraints

In practice, safety stock cannot be set item-by-item in isolation. There are portfolio-level constraints (budget, warehouse space) and business rules (minimum service levels by category).

**Lagrangian Relaxation — Budget-Constrained SS Optimization:**

```
Maximize: sum_i SL_i(SS_i)
Subject to: sum_i SS_i × c_i <= Budget
            SS_i >= 0 for all i
```

The Lagrangian:

```
L(SS, lambda) = sum_i SL_i(SS_i) - lambda × (sum_i SS_i × c_i - Budget)
```

First-order condition for each item:

```
dSL_i / dSS_i = lambda × c_i
```

For Type 1 service level with Normal demand:

```
dSL_i / dSS_i = phi(z_i) / sigma_DDLT_i
```

Algorithm:
1. Binary search on lambda
2. For each lambda, solve per-item: `SS_i = sigma_DDLT_i × Phi^(-1)(target)` where target balances marginal benefit vs. marginal cost
3. Check budget constraint
4. Adjust lambda and repeat

Converges in O(log(1/epsilon) × n) time.

**Gradient-Based Total Cost Minimization:**

```
Minimize: sum_i [HC_i(SS_i) + SC_i(SS_i)]
Where:
  HC_i = SS_i × c_i × holding_rate   (holding cost)
  SC_i = shortage_cost_i × E[shortage_i(SS_i)]   (shortage cost)
```

Gradient:

```
dTotalCost/dSS_i = c_i × holding_rate - shortage_cost_i × dE[shortage]/dSS_i
```

At optimum: marginal holding cost = marginal shortage cost reduction.

**Multi-Objective Optimization:**

When service level and inventory cost are both objectives (not convertible to a single metric), use Pareto optimization:

```
Minimize: [TotalCost(SS), -TotalServiceLevel(SS)]
Subject to: SS_i >= 0, sum_i SS_i × c_i <= Budget_max
```

Solved via NSGA-II (Non-dominated Sorting Genetic Algorithm II) to produce a Pareto frontier. The planner selects their preferred operating point on the frontier.

---

## 3. Multi-Echelon Inventory Optimization (MEIO)

Multi-echelon systems have multiple stocking points in a supply chain (e.g., central warehouse -> regional DCs -> retail stores). Optimizing each location independently leads to excessive safety stock due to the bullwhip effect.

### 3.1 Clark-Scarf Serial Systems

For a serial supply chain with N stages (1 = retailer, N = most upstream):

```
Echelon inventory at stage j = on-hand at j + on-hand at all downstream stages + in-transit to j and all downstream stages
```

The Clark-Scarf decomposition (1960) shows that the optimal policy for a serial system decomposes into N independent newsvendor problems, one per echelon. Each echelon computes its base-stock level independently using echelon costs.

**Echelon holding cost:**

```
h_j^e = h_j - h_{j+1}  (local holding cost minus next upstream holding cost)
```

This ensures no double-counting of holding costs.

### 3.2 Graves-Willems Guaranteed Service Model (GSM)

The Guaranteed Service Model (Graves & Willems, 2000) is the industry standard for MEIO:

- Each stage quotes a guaranteed service time (GST) to its downstream customer.
- Each stage receives guaranteed service from its upstream supplier.
- Net replenishment time = inbound service time + processing time - outbound service time.
- Safety stock at each stage covers demand uncertainty during net replenishment time.

```
SS_j = z(alpha_j) × sigma_D × sqrt(max(0, SI_j + T_j - S_j))
```

Where:
- `SI_j` = inbound service time (quoted by supplier)
- `T_j` = processing time at stage j
- `S_j` = outbound service time (quoted to customer)

Optimization: minimize total safety stock investment by choosing service times at each stage:

```
Minimize: sum_j c_j × SS_j(SI_j, S_j)
Subject to: S_j <= SI_downstream(j)  (service time feasibility)
            0 <= S_j <= S_max
```

This is a concave minimization problem solvable by dynamic programming on spanning trees of the supply chain network.

### 3.3 Bullwhip Effect Quantification

The bullwhip effect amplifies demand variability upstream:

```
Var(Order_upstream) / Var(Demand_customer) = 1 + 2L/p + 2L^2/p^2
```

Where L = lead time, p = forecast smoothing periods. This formula (Lee et al., 1997) shows that longer lead times and shorter forecast windows amplify variability.

MEIO reduces the bullwhip effect by coordinating inventory policies across echelons. Phase 2 of the Inventory Planning Module will implement the Graves-Willems GSM for Demand Studio's multi-location network.

---

## 4. ABC-XYZ Classification

### 4.1 ABC (Value Analysis)

ABC classification segments items by their contribution to total revenue (or total COGS), following the Pareto principle:

```
For each item i:
  annual_revenue_i = sum(qty_shipped_i × unit_price_i) over trailing 12 months
Sort items by annual_revenue descending
Compute cumulative percentage of total revenue
```

| Class | Revenue Share | Typical Item Share | Management Priority |
|-------|--------------|-------------------|-------------------|
| **A** | Top 80% | ~20% of items | Highest — tight control, frequent review, demand-driven |
| **B** | Next 15% (80–95%) | ~30% of items | Standard — regular review, automated policies |
| **C** | Bottom 5% (95–100%) | ~50% of items | Economy — infrequent review, simple rules |

**Default thresholds (configurable in `inventory_config.yaml`):**

```
A: cumulative revenue <= 80%
B: 80% < cumulative revenue <= 95%
C: cumulative revenue > 95%
```

### 4.2 XYZ (Variability Analysis)

XYZ classification segments items by demand predictability, measured by the Coefficient of Variation (CV):

```
CV_i = sigma_demand_i / mean_demand_i
```

Computed over trailing 12 months of monthly demand.

| Class | CV Range | Demand Pattern | Forecasting Difficulty |
|-------|----------|---------------|----------------------|
| **X** | CV < 0.5 | Stable, predictable | Easy — standard methods work well |
| **Y** | 0.5 <= CV < 1.0 | Some variability, trend or seasonality | Moderate — robust methods needed |
| **Z** | CV >= 1.0 | Highly variable, intermittent, lumpy | Hard — bootstrap/Croston needed |

### 4.3 Nine-Cell Matrix Policy Recommendations

Combining ABC and XYZ yields a 3x3 matrix with distinct inventory policies per cell. This is the core segmentation strategy for the Inventory Planning Module.

**AX — High Value, Stable Demand**

- **Target Service Level:** 98–99% (Type 2 fill rate)
- **Review Policy:** Continuous (s, Q) — real-time monitoring justified by high value
- **Safety Stock Method:** Parametric Normal — demand is predictable, Normal assumption valid
- **SS Formula:** `SS = z(0.99) × sigma_FE × sqrt(L)` using champion model forecast errors
- **Automation Level:** Fully automated with manual override alerts
- **Replenishment:** EOQ-based, daily PO generation
- **Rationale:** High revenue impact demands high service; stable demand means parametric methods are reliable

**AY — High Value, Moderate Variability**

- **Target Service Level:** 97–98% (Type 2 fill rate)
- **Review Policy:** Continuous (s, Q) — high value justifies real-time tracking
- **Safety Stock Method:** Empirical (backtest quantiles) — some variability makes empirical more robust
- **SS Formula:** `SS = quantile(backtest_errors, 0.98) × sqrt(L)`
- **Automation Level:** Semi-automated with planner review of exceptions
- **Replenishment:** EOQ with seasonal adjustments
- **Rationale:** Variability requires data-driven SS; high value warrants close attention

**AZ — High Value, Highly Variable**

- **Target Service Level:** 95–97% (Type 2 fill rate)
- **Review Policy:** Continuous (s, S) — variable order sizes adapt to demand lumpiness
- **Safety Stock Method:** Bootstrap simulation — Normal assumptions fail
- **SS Formula:** Bootstrap DDLT, take 97th percentile minus mean
- **Automation Level:** Manual review required — planner judgment critical
- **Replenishment:** Min/max with safety stock buffer
- **Rationale:** High value but unpredictable — accept slightly lower service level; use robust methods

**BX — Medium Value, Stable Demand**

- **Target Service Level:** 95–97% (Type 2 fill rate)
- **Review Policy:** Periodic (R, S) — weekly review sufficient
- **Safety Stock Method:** Parametric Normal
- **SS Formula:** `SS = z(0.95) × sigma_FE × sqrt(R + L)`
- **Automation Level:** Fully automated
- **Replenishment:** Order-up-to with weekly cycle
- **Rationale:** Moderate value, predictable — standard automation works well

**BY — Medium Value, Moderate Variability**

- **Target Service Level:** 94–96% (Type 2 fill rate)
- **Review Policy:** Periodic (R, S) — weekly review
- **Safety Stock Method:** Empirical (backtest quantiles)
- **SS Formula:** `SS = quantile(backtest_errors, 0.95) × sqrt(R + L)`
- **Automation Level:** Automated with exception reporting
- **Replenishment:** Order-up-to with demand-driven adjustments
- **Rationale:** Balanced approach — empirical methods handle moderate variability

**BZ — Medium Value, Highly Variable**

- **Target Service Level:** 90–94% (Type 2 fill rate)
- **Review Policy:** Periodic (R, s, S) — order only when needed
- **Safety Stock Method:** Bootstrap simulation
- **SS Formula:** Bootstrap DDLT, take 94th percentile minus mean
- **Automation Level:** Semi-automated, periodic planner review
- **Replenishment:** Min/max with flexible order quantities
- **Rationale:** Moderate value, high variability — balance cost against service; avoid over-investment

**CX — Low Value, Stable Demand**

- **Target Service Level:** 90–93% (Type 1 cycle service level)
- **Review Policy:** Periodic (R, S) — monthly review sufficient
- **Safety Stock Method:** Parametric Normal with simplified formula
- **SS Formula:** `SS = z(0.90) × sigma_D × sqrt(R + L)` (use raw demand variance, not forecast error — forecasting effort not justified)
- **Automation Level:** Fully automated, no manual intervention
- **Replenishment:** Fixed-interval order-up-to, monthly
- **Rationale:** Low value, predictable — minimize management effort; simple rules suffice

**CY — Low Value, Moderate Variability**

- **Target Service Level:** 85–90% (Type 1 cycle service level)
- **Review Policy:** Periodic (R, S) — monthly review
- **Safety Stock Method:** Parametric Normal with demand-based variance
- **SS Formula:** `SS = z(0.85) × sigma_D × sqrt(R + L)`
- **Automation Level:** Fully automated, exception-only alerts
- **Replenishment:** Monthly order-up-to
- **Rationale:** Low value, some variability — accept lower service level; cost of excess > cost of shortage

**CZ — Low Value, Highly Variable (Intermittent / Lumpy)**

- **Target Service Level:** 80–85% (Type 1 cycle service level)
- **Review Policy:** Periodic (R, s, S) — monthly, order only when depleted
- **Safety Stock Method:** Croston/TSB for intermittent items; bootstrap for lumpy items
- **SS Formula:** Croston forecast × safety factor, or bootstrap quantile
- **Automation Level:** Fully automated with obsolescence review
- **Replenishment:** Order only when stock hits zero or near-zero
- **Rationale:** Low value, unpredictable — minimize investment; accept stockouts; review for obsolescence regularly

**Summary Matrix:**

| | **X (CV < 0.5)** | **Y (0.5 <= CV < 1.0)** | **Z (CV >= 1.0)** |
|---|---|---|---|
| **A (top 80% rev)** | SL: 98-99%, Continuous (s,Q), Normal, Auto | SL: 97-98%, Continuous (s,Q), Empirical, Semi-auto | SL: 95-97%, Continuous (s,S), Bootstrap, Manual |
| **B (next 15% rev)** | SL: 95-97%, Periodic (R,S), Normal, Auto | SL: 94-96%, Periodic (R,S), Empirical, Auto+exceptions | SL: 90-94%, Periodic (R,s,S), Bootstrap, Semi-auto |
| **C (bottom 5% rev)** | SL: 90-93%, Periodic (R,S), Normal simple, Full auto | SL: 85-90%, Periodic (R,S), Normal simple, Full auto | SL: 80-85%, Periodic (R,s,S), Croston/TSB, Auto+obsolescence |

---

## 5. Key Inventory Metrics (KPIs)

All KPI formulas are implemented in `common/inventory_engine.py` and surfaced via `GET /inventory/dashboard`.

### 5.1 Days of Supply (DOS)

```
DOS = Qty_On_Hand / Avg_Daily_Demand
```

Where `Avg_Daily_Demand = sum(qty_shipped, trailing 90 days) / 90`.

Interpretation:
- DOS > target → potential excess
- DOS < SS_days → stockout risk
- DOS = 0 → stockout

### 5.2 Inventory Turns

```
Inventory_Turns = Annual_COGS / Avg_Inventory_Value
```

Where:
- `Annual_COGS = sum(qty_shipped × unit_cost)` over trailing 12 months
- `Avg_Inventory_Value = mean(qty_on_hand × unit_cost)` over trailing 12 months (from monthly snapshots)

Higher turns indicate better inventory productivity. Turns vary dramatically by industry (see Section 38).

### 5.3 Fill Rate (Order Fill Rate)

```
Fill_Rate = sum(qty_shipped) / sum(qty_ordered) × 100
```

Over a time window (typically trailing 3 months). Measures the fraction of customer demand satisfied from available stock.

### 5.4 GMROI (Gross Margin Return on Inventory Investment)

```
GMROI = Gross_Margin / Avg_Inventory_Cost
Gross_Margin = Revenue - COGS
```

GMROI > 1 means each dollar of inventory generates more than $1 of gross margin. Target varies by industry: retail 2–4, wholesale 1.5–3.

### 5.5 Carrying Cost (Total Inventory Holding Cost)

```
Carrying_Cost = Avg_Inventory_Value × Carrying_Rate

Carrying_Rate = Capital_Cost + Storage_Cost + Insurance + Obsolescence
               (typically 15-30% per year)
```

Breakdown:
- Capital cost: 8–15% (cost of capital / opportunity cost)
- Storage cost: 2–5% (warehouse space, utilities, labor)
- Insurance: 1–3%
- Obsolescence/shrinkage: 2–10% (industry-dependent)

### 5.6 Stockout Frequency

```
Stockout_Frequency = count(days where qty_on_hand = 0) / total_days × 100
```

Per item or aggregated across portfolio. Can also be measured as count of stockout events (transitions from positive to zero on-hand).

### 5.7 Excess and Obsolescence Rate

```
Excess_Qty = max(0, qty_on_hand - max_demand_12m)
Excess_Value = Excess_Qty × unit_cost
Excess_Rate = Excess_Value / Total_Inventory_Value × 100

Obsolete: items with zero demand for > 12 months and qty_on_hand > 0
Obsolete_Value = sum(qty_on_hand × unit_cost) for obsolete items
```

### 5.8 Perfect Order Rate

```
Perfect_Order_Rate = Orders_Delivered_Complete_OnTime_Undamaged / Total_Orders × 100
```

Composite metric combining fill rate, on-time delivery, and quality.

### 5.9 Forward Coverage (Weeks of Supply)

```
Forward_Coverage_Weeks = (qty_on_hand + qty_in_transit + qty_on_order) / weekly_demand_forecast
```

Uses forward-looking forecast demand (from champion model) rather than historical average. More useful than DOS for seasonal items.

### 5.10 Inventory Health Score (Composite)

A single 0–100 score combining multiple metrics:

```
Health_Score = w1 × normalize(DOS, target_DOS)
            + w2 × normalize(Fill_Rate, target_FR)
            + w3 × normalize(Turns, target_Turns)
            + w4 × (1 - Excess_Rate / max_excess)
            + w5 × (1 - Stockout_Frequency / max_stockout)
```

Default weights: w1=0.25, w2=0.25, w3=0.20, w4=0.15, w5=0.15.

Traffic light mapping:
- Green (Healthy): Score >= 80
- Yellow (Watch): 60 <= Score < 80
- Red (Critical): Score < 60

---

# Part II: Data Architecture

---

## 6. Data Requirements — What Already Exists

The Inventory Planning Module builds on the existing Demand Studio data model. This section maps existing tables to inventory planning needs and identifies gaps.

### 6.1 Existing Tables Used by Inventory Planning

**dim_item — Item Master**

| Column | Inventory Use |
|--------|--------------|
| item_no | Primary key for inventory position |
| description | Display in grids and alerts |
| brand | Filtering/segmentation |
| category | ABC classification grouping |
| department | Reporting hierarchy |
| class_ | Sub-category segmentation |
| unit_cost | Inventory valuation, holding cost computation |
| unit_price | Revenue calculation for ABC, GMROI |

**dim_location — Location Master**

| Column | Inventory Use |
|--------|--------------|
| loc | Primary key for inventory position |
| loc_name | Display label |
| state | Regional aggregation |
| zone | Warehouse zone for storage allocation |

**dim_dfu — Demand Forecast Unit Master**

| Column | Inventory Use |
|--------|--------------|
| dfu_ck | Composite key linking item + location + customer_group |
| execution_lag | Lag between forecast and execution (affects lead time) |
| total_lt | Total lead time (direct input to safety stock) |
| cluster_assignment | Policy segmentation by demand cluster |
| seasonality_profile | Seasonal buildup planning (none/low/medium/high) |
| seasonality_strength | Magnitude of seasonal effect |
| peak_month | Month of peak demand — pre-build timing |
| trough_month | Month of trough demand — destocking timing |

**fact_sales_monthly — Demand History**

| Column | Inventory Use |
|--------|--------------|
| item_no, loc, customer_group, month | Grain for demand computation |
| qty_shipped | Actual demand (primary measure) |
| qty_ordered | Customer order quantity (for fill rate computation) |
| qty | Generic quantity measure |
| type | Filter: TYPE=1 for sales |

Usage: Trailing 12–24 months of `qty_shipped` for demand statistics (mean, std, CV), ABC revenue calculation, XYZ variability analysis, and intermittent demand classification.

**fact_external_forecast_monthly — Multi-Model Forecasts**

| Column | Inventory Use |
|--------|--------------|
| item_no, loc, forecast_date, month | Forecast grain |
| model_id | Model identifier (champion, lgbm_global, etc.) |
| forecast_qty | Point forecast |
| actual_qty | Actual demand (for error computation) |
| lag | Forecast horizon (0–4 months) |

Usage: Champion model forecasts (`model_id='champion'`) provide the expected demand input for safety stock computation. Forecast errors (forecast_qty - actual_qty) from the champion model drive empirical safety stock.

**backtest_lag_archive — Forecast Error History**

| Column | Inventory Use |
|--------|--------------|
| forecast_ck | DFU composite key |
| model_id | Model that produced the forecast |
| lag | Forecast horizon |
| forecast_qty, actual_qty | For error distribution |
| timeframe | Backtest timeframe for traceability |

Usage: The full distribution of forecast errors from the champion model at each lag — this is the empirical basis for safety stock. Instead of assuming Normal errors, the Inventory Planning Module can use the actual quantiles of historical forecast errors to set safety stock.

### 6.2 Existing Materialized Views

**agg_sales_monthly** — Pre-aggregated sales for O(1) KPI queries (demand statistics)

**agg_forecast_monthly** — Pre-aggregated forecasts for accuracy KPIs

**agg_accuracy_by_dim** — Accuracy sliced by dimensions including seasonality_profile

**agg_dfu_coverage** — DFU coverage counts

These views provide fast access to demand and forecast data needed by the inventory dashboard.

### 6.3 Existing Config Used

**config/clustering_config.yaml** — Cluster labels map to inventory policy segments

**config/seasonality_config.yaml** — Seasonality thresholds determine seasonal buildup planning

**config/model_competition.yaml** — Identifies the champion model whose forecasts drive inventory planning

---

## 7. New Data Required

### 7.1 Inventory Position Snapshots

**Table: `fact_inventory_snapshot`**

Daily snapshots of inventory position per item-location. This is the core operational data for the inventory module.

```sql
CREATE TABLE fact_inventory_snapshot (
    item_no         VARCHAR(50)    NOT NULL,
    loc             VARCHAR(50)    NOT NULL,
    snapshot_date   DATE           NOT NULL,
    qty_on_hand     NUMERIC(15,2)  NOT NULL DEFAULT 0,
    qty_in_transit  NUMERIC(15,2)  NOT NULL DEFAULT 0,
    qty_on_order    NUMERIC(15,2)  NOT NULL DEFAULT 0,
    qty_allocated   NUMERIC(15,2)  NOT NULL DEFAULT 0,
    qty_available   NUMERIC(15,2)  GENERATED ALWAYS AS (qty_on_hand - qty_allocated) STORED,
    qty_backorder   NUMERIC(15,2)  NOT NULL DEFAULT 0,
    unit_cost       NUMERIC(12,4),
    inventory_value NUMERIC(14,2)  GENERATED ALWAYS AS (qty_on_hand * unit_cost) STORED,
    load_ts         TIMESTAMPTZ    NOT NULL DEFAULT now(),

    PRIMARY KEY (item_no, loc, snapshot_date)
) PARTITION BY RANGE (snapshot_date);
```

Partitioning: Monthly partitions, retain 24 months. Auto-create partitions via `pg_partman` or DDL script.

Indexes:
```sql
CREATE INDEX idx_inv_snapshot_item ON fact_inventory_snapshot (item_no);
CREATE INDEX idx_inv_snapshot_loc ON fact_inventory_snapshot (loc);
CREATE INDEX idx_inv_snapshot_date ON fact_inventory_snapshot (snapshot_date DESC);
```

### 7.2 Purchase Orders

**Table: `fact_purchase_orders`**

Tracks purchase orders for receipt planning, supplier performance, and pipeline inventory computation.

```sql
CREATE TABLE fact_purchase_orders (
    po_number           VARCHAR(50)    NOT NULL,
    po_line             INTEGER        NOT NULL,
    item_no             VARCHAR(50)    NOT NULL,
    loc                 VARCHAR(50)    NOT NULL,
    supplier_id         VARCHAR(50)    NOT NULL,
    qty_ordered         NUMERIC(12,2)  NOT NULL,
    qty_received        NUMERIC(12,2)  NOT NULL DEFAULT 0,
    qty_outstanding     NUMERIC(12,2)  GENERATED ALWAYS AS (qty_ordered - qty_received) STORED,
    order_date          DATE           NOT NULL,
    promised_date       DATE           NOT NULL,
    actual_receipt_date DATE,
    unit_cost           NUMERIC(12,4),
    po_value            NUMERIC(14,2)  GENERATED ALWAYS AS (qty_ordered * unit_cost) STORED,
    status              VARCHAR(20)    NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'partial', 'received', 'cancelled')),
    lead_time_actual    INTEGER        GENERATED ALWAYS AS (
                            CASE WHEN actual_receipt_date IS NOT NULL
                                 THEN actual_receipt_date - order_date
                                 ELSE NULL END
                        ) STORED,
    lead_time_variance  INTEGER        GENERATED ALWAYS AS (
                            CASE WHEN actual_receipt_date IS NOT NULL
                                 THEN actual_receipt_date - promised_date
                                 ELSE NULL END
                        ) STORED,
    load_ts             TIMESTAMPTZ    NOT NULL DEFAULT now(),

    PRIMARY KEY (po_number, po_line)
);
```

Indexes:
```sql
CREATE INDEX idx_po_item ON fact_purchase_orders (item_no);
CREATE INDEX idx_po_supplier ON fact_purchase_orders (supplier_id);
CREATE INDEX idx_po_status ON fact_purchase_orders (status) WHERE status IN ('open', 'partial');
CREATE INDEX idx_po_promised ON fact_purchase_orders (promised_date) WHERE status IN ('open', 'partial');
```

### 7.3 Supplier Master

**Table: `dim_supplier`**

```sql
CREATE TABLE dim_supplier (
    supplier_sk             BIGSERIAL       PRIMARY KEY,
    supplier_id             VARCHAR(50)     NOT NULL UNIQUE,
    supplier_name           VARCHAR(200)    NOT NULL,
    country                 VARCHAR(100),
    region                  VARCHAR(100),
    avg_lead_time_days      NUMERIC(8,2),
    lead_time_std_days      NUMERIC(8,2),
    on_time_delivery_pct    NUMERIC(5,2),
    quality_rating_pct      NUMERIC(5,2),
    moq                     NUMERIC(12,2)   DEFAULT 1,
    order_multiple          NUMERIC(12,2)   DEFAULT 1,
    payment_terms_days      INTEGER,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    load_ts                 TIMESTAMPTZ     NOT NULL DEFAULT now(),
    modified_ts             TIMESTAMPTZ     NOT NULL DEFAULT now()
);
```

### 7.4 Supplier-Item Mapping

**Table: `map_supplier_item`**

Maps which suppliers can provide which items, with supplier-specific lead times and costs. Supports multi-sourcing.

```sql
CREATE TABLE map_supplier_item (
    supplier_id     VARCHAR(50)     NOT NULL REFERENCES dim_supplier(supplier_id),
    item_no         VARCHAR(50)     NOT NULL,
    lead_time_days  INTEGER         NOT NULL,
    unit_cost       NUMERIC(12,4)   NOT NULL,
    moq             NUMERIC(12,2)   DEFAULT 1,
    order_multiple  NUMERIC(12,2)   DEFAULT 1,
    is_primary      BOOLEAN         NOT NULL DEFAULT false,
    effective_date  DATE            NOT NULL DEFAULT CURRENT_DATE,
    expiry_date     DATE,
    load_ts         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    PRIMARY KEY (supplier_id, item_no, effective_date)
);

CREATE INDEX idx_supplier_item_item ON map_supplier_item (item_no);
CREATE INDEX idx_supplier_item_primary ON map_supplier_item (item_no, is_primary) WHERE is_primary = true;
```

### 7.5 Inventory Cost Configuration

**Table: `dim_inventory_cost`**

Cost parameters for inventory optimization. Can be set at item-location level, item level, category level, or global default (resolved by specificity).

```sql
CREATE TABLE dim_inventory_cost (
    item_no             VARCHAR(50),
    loc                 VARCHAR(50),
    holding_cost_pct    NUMERIC(5,4)    NOT NULL DEFAULT 0.2000,
    ordering_cost       NUMERIC(10,2)   NOT NULL DEFAULT 50.00,
    shortage_cost       NUMERIC(10,2)   NOT NULL DEFAULT 100.00,
    obsolescence_rate   NUMERIC(5,4)    NOT NULL DEFAULT 0.0500,
    capital_cost_rate   NUMERIC(5,4)    NOT NULL DEFAULT 0.1000,
    storage_cost_rate   NUMERIC(5,4)    NOT NULL DEFAULT 0.0300,
    insurance_rate      NUMERIC(5,4)    NOT NULL DEFAULT 0.0100,
    is_default          BOOLEAN         NOT NULL DEFAULT false,
    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT now(),

    UNIQUE (item_no, loc)
);

-- Global default row
INSERT INTO dim_inventory_cost (item_no, loc, is_default)
VALUES (NULL, NULL, true);
```

### 7.6 Inventory Policy Configuration

**Table: `dim_inventory_policy`**

Per-DFU inventory policy configuration, either auto-assigned by ABC-XYZ classification or manually overridden by planners.

```sql
CREATE TABLE dim_inventory_policy (
    dfu_ck                  TEXT            PRIMARY KEY,
    target_service_level    NUMERIC(5,4)    NOT NULL DEFAULT 0.9500,
    service_level_type      VARCHAR(10)     NOT NULL DEFAULT 'fill_rate'
                            CHECK (service_level_type IN ('csl', 'fill_rate', 'ready_rate')),
    review_period_days      INTEGER         NOT NULL DEFAULT 7,
    policy_type             VARCHAR(20)     NOT NULL DEFAULT 'rQ'
                            CHECK (policy_type IN ('rQ', 'sS', 'RS', 'RsS', 'base_stock')),
    ss_method               VARCHAR(30)     NOT NULL DEFAULT 'normal'
                            CHECK (ss_method IN ('normal', 'empirical', 'conformal', 'bootstrap', 'croston', 'tsb')),
    abc_class               CHAR(1)         CHECK (abc_class IN ('A', 'B', 'C')),
    xyz_class               CHAR(1)         CHECK (xyz_class IN ('X', 'Y', 'Z')),
    demand_class            VARCHAR(20)     CHECK (demand_class IN ('smooth', 'erratic', 'intermittent', 'lumpy')),
    is_manual_override      BOOLEAN         NOT NULL DEFAULT false,
    override_reason         TEXT,
    override_by             VARCHAR(100),
    override_ts             TIMESTAMPTZ,
    load_ts                 TIMESTAMPTZ     NOT NULL DEFAULT now(),
    modified_ts             TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_policy_abc ON dim_inventory_policy (abc_class);
CREATE INDEX idx_policy_xyz ON dim_inventory_policy (xyz_class);
CREATE INDEX idx_policy_override ON dim_inventory_policy (is_manual_override) WHERE is_manual_override = true;
```

### 7.7 Inventory Planning Output

**Table: `fact_inventory_plan`**

The computed output of the safety stock engine — one row per DFU per planning run.

```sql
CREATE TABLE fact_inventory_plan (
    dfu_ck              TEXT            NOT NULL,
    plan_date           DATE            NOT NULL,
    plan_version        INTEGER         NOT NULL DEFAULT 1,

    -- Safety stock and reorder parameters
    safety_stock        NUMERIC(12,2)   NOT NULL,
    reorder_point       NUMERIC(12,2)   NOT NULL,
    order_quantity      NUMERIC(12,2)   NOT NULL,
    order_up_to         NUMERIC(12,2),
    min_order_qty       NUMERIC(12,2)   DEFAULT 1,

    -- Demand during lead time statistics
    forecast_mean       NUMERIC(12,2)   NOT NULL,
    forecast_std        NUMERIC(12,2)   NOT NULL,
    lead_time_mean      NUMERIC(8,2)    NOT NULL,
    lead_time_std       NUMERIC(8,2)    DEFAULT 0,

    -- Position and coverage
    days_of_supply      NUMERIC(8,2),
    forward_coverage_wk NUMERIC(8,2),
    current_on_hand     NUMERIC(12,2),
    current_position    NUMERIC(12,2),

    -- Method and model metadata
    champion_model      TEXT,
    ss_method_used      TEXT            NOT NULL,
    policy_type         TEXT            NOT NULL,
    target_sl           NUMERIC(5,4)    NOT NULL,
    achieved_sl         NUMERIC(5,4),

    -- Cost outputs
    total_cost_annual   NUMERIC(14,2),
    holding_cost_annual NUMERIC(14,2),
    ordering_cost_annual NUMERIC(14,2),
    shortage_cost_annual NUMERIC(14,2),
    ss_investment       NUMERIC(14,2),

    -- Classification
    abc_class           CHAR(1),
    xyz_class           CHAR(1),

    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT now(),

    PRIMARY KEY (dfu_ck, plan_date, plan_version)
);

CREATE INDEX idx_plan_date ON fact_inventory_plan (plan_date DESC);
CREATE INDEX idx_plan_abc ON fact_inventory_plan (abc_class);
```

### 7.8 Promotion Calendar

**Table: `dim_promotion`**

Tracks planned promotions that will temporarily lift demand — critical for avoiding stockouts during promo periods.

```sql
CREATE TABLE dim_promotion (
    promo_id        VARCHAR(50)     PRIMARY KEY,
    item_no         VARCHAR(50)     NOT NULL,
    loc             VARCHAR(50),
    start_date      DATE            NOT NULL,
    end_date        DATE            NOT NULL,
    lift_factor     NUMERIC(5,2)    NOT NULL DEFAULT 1.00,
    promo_type      VARCHAR(50),
    description     TEXT,
    status          VARCHAR(20)     NOT NULL DEFAULT 'planned'
                    CHECK (status IN ('planned', 'active', 'completed', 'cancelled')),
    load_ts         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CHECK (end_date >= start_date),
    CHECK (lift_factor >= 0)
);

CREATE INDEX idx_promo_item ON dim_promotion (item_no);
CREATE INDEX idx_promo_dates ON dim_promotion (start_date, end_date);
CREATE INDEX idx_promo_active ON dim_promotion (status) WHERE status IN ('planned', 'active');
```

### 7.9 Inventory Alerts / Exception Log

**Table: `fact_inventory_alert`**

Stores generated alerts and exceptions with recommended actions and resolution tracking.

```sql
CREATE TABLE fact_inventory_alert (
    alert_id            BIGSERIAL       PRIMARY KEY,
    dfu_ck              TEXT            NOT NULL,
    item_no             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    alert_date          DATE            NOT NULL DEFAULT CURRENT_DATE,
    alert_type          VARCHAR(50)     NOT NULL
                        CHECK (alert_type IN (
                            'stockout_risk', 'stockout_active',
                            'excess_stock', 'obsolescence_risk',
                            'lead_time_spike', 'demand_anomaly',
                            'budget_breach', 'service_level_miss',
                            'supplier_issue', 'expiry_risk'
                        )),
    severity            VARCHAR(20)     NOT NULL
                        CHECK (severity IN ('critical', 'warning', 'info')),
    message             TEXT            NOT NULL,
    recommended_action  TEXT,
    metric_value        NUMERIC(14,2),
    threshold_value     NUMERIC(14,2),
    status              VARCHAR(20)     NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'acknowledged', 'in_progress', 'resolved', 'dismissed')),
    resolved_by         VARCHAR(100),
    resolved_ts         TIMESTAMPTZ,
    resolution_note     TEXT,
    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT now()
) PARTITION BY RANGE (alert_date);

CREATE INDEX idx_alert_status ON fact_inventory_alert (status) WHERE status IN ('open', 'acknowledged', 'in_progress');
CREATE INDEX idx_alert_severity ON fact_inventory_alert (severity, status);
CREATE INDEX idx_alert_dfu ON fact_inventory_alert (dfu_ck);
CREATE INDEX idx_alert_type ON fact_inventory_alert (alert_type);
```

### 7.10 Inventory Scenario Results

**Table: `fact_inventory_scenario`**

Stores what-if scenario parameters and results. Follows the pattern established by the What-If Clustering Scenarios (Feature 29).

```sql
CREATE TABLE fact_inventory_scenario (
    scenario_id     VARCHAR(50)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    scenario_name   VARCHAR(200),
    scenario_type   VARCHAR(50)     NOT NULL
                    CHECK (scenario_type IN (
                        'service_level', 'lead_time', 'demand_variability',
                        'supply_disruption', 'budget_constraint', 'network_sourcing',
                        'abc_reclassify', 'seasonal_buildup', 'promotion_impact',
                        'moq_adjustment', 'custom'
                    )),
    params          JSONB           NOT NULL,
    results         JSONB,
    baseline_kpis   JSONB,
    scenario_kpis   JSONB,
    status          VARCHAR(20)     NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'promoted')),
    created_by      VARCHAR(100),
    created_ts      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    completed_ts    TIMESTAMPTZ,
    runtime_seconds NUMERIC(8,2),
    error_message   TEXT
);

CREATE INDEX idx_scenario_status ON fact_inventory_scenario (status);
CREATE INDEX idx_scenario_type ON fact_inventory_scenario (scenario_type);
```

---

## 8. Data Volume Estimates

Based on a mid-size deployment with 10,000 items across 100 locations:

| Table | Estimated Rows | Growth Rate | Retention | Storage |
|-------|---------------|-------------|-----------|---------|
| fact_inventory_snapshot | 1M/day (1M item-loc pairs) | Daily append | 24 months | ~500 GB |
| fact_purchase_orders | 500K active, 5M historical | 50K/month | Indefinite | ~2 GB |
| dim_supplier | 500–2,000 | ~10/month | Indefinite | < 1 MB |
| map_supplier_item | 50,000 | ~500/month | Indefinite | < 50 MB |
| dim_inventory_cost | 1M (item-loc) + defaults | Rare changes | Indefinite | < 100 MB |
| dim_inventory_policy | 1M (per DFU) | Recomputed weekly | Current only | < 100 MB |
| fact_inventory_plan | 1M (per DFU per run) | Weekly recompute | 12 months | ~5 GB |
| dim_promotion | 10K active | 1K/month | 24 months | < 50 MB |
| fact_inventory_alert | 10K/day | Daily generation | 6 months | ~2 GB |
| fact_inventory_scenario | 1K scenarios | 10/week | 12 months | < 100 MB |

**Total incremental storage:** ~510 GB (dominated by daily inventory snapshots).

**Partitioning strategy:**
- `fact_inventory_snapshot`: Monthly range partitions on `snapshot_date`. Auto-drop partitions older than 24 months.
- `fact_inventory_alert`: Monthly range partitions on `alert_date`. Auto-drop partitions older than 6 months.
- All other tables: No partitioning needed at this scale.

**Materialized views (created in `sql/018_create_inventory_views.sql`):**

```sql
-- Latest inventory position (most recent snapshot per item-loc)
CREATE MATERIALIZED VIEW mv_inventory_current AS
SELECT DISTINCT ON (item_no, loc)
    item_no, loc, snapshot_date, qty_on_hand, qty_in_transit,
    qty_on_order, qty_allocated, qty_available, qty_backorder,
    unit_cost, inventory_value
FROM fact_inventory_snapshot
ORDER BY item_no, loc, snapshot_date DESC;

-- Inventory KPI aggregates
CREATE MATERIALIZED VIEW mv_inventory_kpis AS
SELECT
    i.category,
    i.department,
    l.state,
    p.abc_class,
    p.xyz_class,
    count(*) AS item_count,
    sum(c.qty_on_hand * c.unit_cost) AS total_inventory_value,
    avg(c.qty_on_hand / NULLIF(d.avg_daily_demand, 0)) AS avg_dos,
    -- ... additional aggregations
FROM mv_inventory_current c
JOIN dim_item i ON c.item_no = i.item_no
JOIN dim_location l ON c.loc = l.loc
LEFT JOIN dim_inventory_policy p ON ...
LEFT JOIN (demand stats subquery) d ON ...
GROUP BY GROUPING SETS (
    (i.category), (i.department), (l.state),
    (p.abc_class), (p.xyz_class),
    (p.abc_class, p.xyz_class),
    ()
);

-- Supplier performance scorecard
CREATE MATERIALIZED VIEW mv_supplier_scorecard AS
SELECT
    s.supplier_id,
    s.supplier_name,
    count(*) AS total_pos,
    avg(po.lead_time_actual) AS avg_lead_time,
    stddev(po.lead_time_actual) AS std_lead_time,
    count(*) FILTER (WHERE po.actual_receipt_date <= po.promised_date)::float
        / NULLIF(count(*) FILTER (WHERE po.actual_receipt_date IS NOT NULL), 0) * 100
        AS on_time_delivery_pct,
    avg(po.lead_time_variance) AS avg_late_days
FROM dim_supplier s
JOIN fact_purchase_orders po ON s.supplier_id = po.supplier_id
WHERE po.status IN ('received', 'partial')
GROUP BY s.supplier_id, s.supplier_name;
```

---

## 9. Integration Architecture

### 9.1 ERP Integration (SAP / Oracle / D365 / NetSuite)

The Inventory Planning Module requires data feeds from the organization's ERP system. Three integration patterns are supported:

**Pattern 1: Batch File Extract (CSV/Parquet)**

```
ERP → Scheduled Export → SFTP/S3 → Demand Studio ETL → PostgreSQL
```

- Inventory snapshots: Daily extract at end-of-day
- Purchase orders: Daily delta extract (new/changed POs)
- Supplier master: Weekly full extract
- Use existing `normalize_dataset_csv.py` and `load_dataset_postgres.py` patterns

**Pattern 2: Change Data Capture (CDC)**

```
ERP Database → Debezium → Kafka → Demand Studio Consumer → PostgreSQL
```

- Near-real-time updates for PO status changes, receipt confirmations
- Preferred for time-sensitive replenishment workflows
- Phase 2 implementation

**Pattern 3: REST API Integration**

```
ERP REST API → Demand Studio Sync Service → PostgreSQL
```

- Polling-based: scheduled API calls every N minutes
- Webhook-based: ERP pushes events to Demand Studio endpoint
- Suitable for cloud ERPs (NetSuite, D365 Business Central)

### 9.2 WMS (Warehouse Management System) Integration

```
WMS → Receipt Confirmations, Inventory Adjustments → Demand Studio
```

- On-hand quantity adjustments (cycle counts, damage, returns)
- Receipt confirmations (PO status → "received")
- File-based (EDI 846, CSV) or REST API

### 9.3 Internal Integration (Existing Demand Studio Modules)

| Source Module | Data Flow | Target Use |
|--------------|-----------|------------|
| Champion Model Selection (F15) | `model_id='champion'` forecasts | Demand input for SS, ROP, EOQ |
| Backtest Archive (F8–F13) | `backtest_lag_archive` error distributions | Empirical safety stock quantiles |
| DFU Clustering (F7) | `dim_dfu.cluster_assignment` | Policy segmentation by demand pattern |
| Seasonality Detection (F30) | `dim_dfu.seasonality_profile`, `peak_month`, `trough_month` | Seasonal buildup timing, time-varying SS |
| Seasonality Filtering (F32) | Accuracy views with seasonality dimension | Filter inventory KPIs by seasonal profile |
| What-If Scenarios (F29) | Architecture pattern (background thread, concurrency guard) | Inventory scenario runner |
| Market Intelligence (F18) | Supply market context | Supplier evaluation enrichment |

**Data Flow Diagram:**

```
                    ┌─────────────────────────┐
                    │     External Systems     │
                    │  ERP / WMS / Suppliers   │
                    └─────────┬───────────────┘
                              │ (batch CSV / CDC / API)
                              ▼
                    ┌─────────────────────────┐
                    │   Ingestion Layer        │
                    │  normalize + load scripts│
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
    ┌─────────────┐  ┌──────────────┐   ┌──────────────────┐
    │ Inventory   │  │ Purchase     │   │ Supplier         │
    │ Snapshots   │  │ Orders       │   │ Master + Mapping │
    └──────┬──────┘  └──────┬───────┘   └──────┬───────────┘
           │                │                   │
           └────────────────┼───────────────────┘
                            ▼
                  ┌─────────────────────┐
                  │  Inventory Engine   │ ← Champion Forecasts
                  │  (common/)          │ ← Backtest Errors
                  │  SS / ROP / EOQ     │ ← Cluster Assignments
                  │  ABC-XYZ            │ ← Seasonality Profiles
                  │  Optimization       │
                  └─────────┬───────────┘
                            │
              ┌─────────────┼─────────────────┐
              ▼             ▼                 ▼
    ┌──────────────┐ ┌─────────────┐  ┌────────────────┐
    │ Inventory    │ │ Alerts      │  │ Scenario       │
    │ Plan Output  │ │ / Exceptions│  │ Results        │
    └──────┬───────┘ └──────┬──────┘  └──────┬─────────┘
           │                │                 │
           └────────────────┼─────────────────┘
                            ▼
                  ┌─────────────────────┐
                  │   FastAPI Endpoints  │
                  │  /inventory/*        │
                  └─────────┬───────────┘
                            ▼
                  ┌─────────────────────┐
                  │   React UI          │
                  │   InventoryTab      │
                  │   8 sub-panels      │
                  └─────────────────────┘
```

---

# Part III: ML/AI Models & Optimization

---

## 10. Probabilistic Forecasting for Inventory

Traditional point forecasts (single number) are insufficient for inventory planning — we need the full distribution of future demand to set appropriate safety stock. The Inventory Planning Module implements four approaches to probabilistic demand estimation, each suited to different data availability and item characteristics.

### 10.1 Conformal Prediction Intervals (MAPIE)

Conformal prediction provides distribution-free prediction intervals with guaranteed coverage probability, regardless of the underlying model's assumptions.

**Method:** Apply MAPIE (Model Agnostic Prediction Intervals Estimator) as a wrapper around the champion model:

```python
from mapie.regression import MapieRegressor

# Wrap champion model with conformal prediction
mapie = MapieRegressor(champion_model, method="plus", cv=5)
mapie.fit(X_train, y_train)
y_pred, y_intervals = mapie.predict(X_test, alpha=0.05)  # 95% interval
```

**Safety stock from conformal intervals:**

```
SS = (upper_bound_95 - point_forecast) × sqrt(L / forecast_horizon)
```

The width of the conformal interval directly measures forecast uncertainty — wider intervals mean more safety stock.

**Advantages:**
- Distribution-free: no normality assumption
- Finite-sample coverage guarantee
- Works with any champion model (LGBM, CatBoost, XGBoost, etc.)
- Adapts to heteroscedastic demand (wider intervals for less predictable items)

### 10.2 Backtest-Error-Based Empirical Safety Stock

Unique to Demand Studio: leverage the rich backtest archive to compute empirical forecast error distributions.

**Method:**

```python
# Pull champion model forecast errors from backtest_lag_archive
errors = SELECT (forecast_qty - actual_qty)
         FROM backtest_lag_archive
         WHERE model_id = 'champion' AND lag = target_lag
         GROUP BY forecast_ck

# Per-DFU error distribution
for each dfu:
    error_dist = errors[dfu]
    # Safety stock = quantile of errors at target service level
    SS = np.percentile(error_dist, (1 - alpha) * 100) * sqrt(L)
```

**Advantages:**
- No distributional assumptions
- Captures the actual error characteristics of the champion model at each lag
- Accounts for model-specific biases (persistent over/under-forecasting)
- Per-DFU calibration — not a one-size-fits-all approach

**Implementation detail:** For DFUs with insufficient backtest history (< 6 timeframes), fall back to cluster-level error distribution (pool errors from all DFUs in the same cluster).

### 10.3 Quantile Regression Forests

Train a random forest to predict quantiles of the demand distribution directly:

```python
from sklearn.ensemble import RandomForestRegressor

# Train forest on demand data
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10)
rf.fit(X_train, y_train)

# Extract quantiles from leaf node distributions
def predict_quantile(rf, X, quantile):
    leaf_ids = rf.apply(X)
    predictions = []
    for i in range(X.shape[0]):
        leaf_values = []
        for tree_idx in range(rf.n_estimators):
            leaf_id = leaf_ids[i, tree_idx]
            leaf_values.extend(tree_leaf_values[tree_idx][leaf_id])
        predictions.append(np.percentile(leaf_values, quantile * 100))
    return np.array(predictions)

demand_q95 = predict_quantile(rf, X_future, 0.95)
SS = demand_q95 - demand_point_forecast
```

### 10.4 DeepAR Sample Paths

For DFUs with sufficient history (>= 24 months), use DeepAR to generate N sample paths of future demand:

```
Sample paths: d_1^(1), ..., d_L^(1)
              d_1^(2), ..., d_L^(2)
              ...
              d_1^(N), ..., d_L^(N)

DDLT per path: DDLT^(k) = sum_{t=1}^{L} d_t^(k)
SS = quantile(DDLT^(1), ..., DDLT^(N), 1-alpha) - mean(DDLT)
```

This naturally captures serial correlation in demand (demand tomorrow depends on demand today).

---

## 11. Safety Stock Engine

### 11.1 Engine Architecture

The safety stock engine (`common/inventory_engine.py`) implements four methods, automatically selected per DFU based on ABC-XYZ classification and data availability:

```
┌─────────────────────────────────────────────────────────┐
│                   Safety Stock Engine                    │
│                                                         │
│  Input:                                                 │
│    - DFU key                                           │
│    - Demand history (fact_sales_monthly)                │
│    - Champion forecast (fact_external_forecast_monthly)  │
│    - Forecast errors (backtest_lag_archive)              │
│    - Lead time (dim_dfu.total_lt + supplier LT history) │
│    - Policy config (dim_inventory_policy)               │
│    - Cost params (dim_inventory_cost)                   │
│                                                         │
│  Method Selection:                                      │
│    ABC-XYZ → default method (from 9-cell matrix)       │
│    Override: manual selection via dim_inventory_policy   │
│                                                         │
│  Methods:                                               │
│    1. Parametric (Normal)   → AX, BX, CX, CY items    │
│    2. Empirical (Backtest)  → AY, BY items             │
│    3. Conformal Prediction  → items with champion model │
│    4. Bootstrap Simulation  → AZ, BZ, CZ items         │
│                                                         │
│  Output:                                                │
│    - safety_stock (units)                              │
│    - reorder_point (units)                             │
│    - order_quantity (units)                            │
│    - order_up_to (units, for s-S policies)             │
│    - achieved_service_level (from simulation)          │
│    - total_annual_cost                                 │
└─────────────────────────────────────────────────────────┘
```

### 11.2 Method 1: Parametric (Normal Distribution)

For items with stable, predictable demand (X class):

```python
def compute_ss_normal(
    demand_mean: float,      # mean demand per period
    demand_std: float,       # std dev of demand per period (or forecast error std)
    lead_time_mean: float,   # mean lead time in periods
    lead_time_std: float,    # std dev of lead time
    service_level: float,    # target (0-1)
    sl_type: str,           # 'csl' or 'fill_rate'
    order_qty: float,       # for fill rate calculation
) -> dict:
    z = norm.ppf(service_level) if sl_type == 'csl' else _solve_fill_rate_z(...)

    # Combined demand + lead time uncertainty
    sigma_ddlt = sqrt(lead_time_mean * demand_std**2 + demand_mean**2 * lead_time_std**2)

    ss = z * sigma_ddlt
    rop = demand_mean * lead_time_mean + ss

    return {"safety_stock": ss, "reorder_point": rop, "sigma_ddlt": sigma_ddlt}
```

### 11.3 Method 2: Empirical (Backtest Quantiles)

For items with moderate variability (Y class) — uses actual forecast errors from the backtest archive:

```python
def compute_ss_empirical(
    forecast_errors: np.ndarray,   # from backtest_lag_archive
    lead_time_mean: float,
    service_level: float,
    min_observations: int = 6,
) -> dict:
    if len(forecast_errors) < min_observations:
        raise InsufficientDataError("Need >= 6 backtest timeframes")

    # Scale errors to lead time horizon
    # errors are per-period; DDLT error scales by sqrt(L)
    scaled_errors = forecast_errors * sqrt(lead_time_mean)

    # Safety stock = quantile of error distribution
    ss = np.percentile(scaled_errors, service_level * 100)

    # If errors are biased (model consistently under-forecasts), SS captures that
    return {"safety_stock": max(0, ss), "error_mean": np.mean(forecast_errors),
            "error_std": np.std(forecast_errors)}
```

### 11.4 Method 3: Conformal Prediction

For items with a trained champion model:

```python
def compute_ss_conformal(
    champion_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_future: pd.DataFrame,
    lead_time_mean: float,
    alpha: float = 0.05,    # 95% prediction interval
) -> dict:
    mapie = MapieRegressor(champion_model, method="plus", cv=5)
    mapie.fit(X_train, y_train)
    y_pred, y_intervals = mapie.predict(X_future, alpha=alpha)

    # SS from interval width
    interval_width = y_intervals[:, 1, 0] - y_pred  # upper - point
    ss = np.mean(interval_width) * sqrt(lead_time_mean)

    return {"safety_stock": ss, "prediction_interval": y_intervals}
```

### 11.5 Method 4: Bootstrap Simulation

For items with highly variable or intermittent demand (Z class):

```python
def compute_ss_bootstrap(
    demand_history: np.ndarray,
    lead_time_samples: np.ndarray,  # historical lead times
    service_level: float,
    n_simulations: int = 10000,
) -> dict:
    ddlt_samples = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Sample lead time
        lt = np.random.choice(lead_time_samples)
        lt_periods = max(1, round(lt))

        # Bootstrap demand over lead time
        demand_sample = np.random.choice(demand_history, size=lt_periods, replace=True)
        ddlt_samples[i] = np.sum(demand_sample)

    ss = np.percentile(ddlt_samples, service_level * 100) - np.mean(ddlt_samples)

    return {
        "safety_stock": max(0, ss),
        "ddlt_mean": np.mean(ddlt_samples),
        "ddlt_std": np.std(ddlt_samples),
        "ddlt_p50": np.median(ddlt_samples),
        "ddlt_p95": np.percentile(ddlt_samples, 95),
        "ddlt_p99": np.percentile(ddlt_samples, 99),
    }
```

### 11.6 EOQ Computation

```python
def compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
    moq: float = 1,
    order_multiple: float = 1,
) -> float:
    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return moq

    eoq_raw = sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)

    # Apply MOQ constraint
    eoq = max(eoq_raw, moq)

    # Round up to order multiple
    if order_multiple > 1:
        eoq = ceil(eoq / order_multiple) * order_multiple

    return eoq
```

### 11.7 Fill Rate Computation (Type 2 Service Level)

```python
def compute_fill_rate(z: float, sigma_ddlt: float, Q: float) -> float:
    """Compute fill rate (Type 2 service level) given z-value."""
    loss = _normal_loss_function(z)
    expected_shortage = sigma_ddlt * loss
    fill_rate = 1 - expected_shortage / Q
    return max(0, min(1, fill_rate))

def _normal_loss_function(z: float) -> float:
    """L(z) = phi(z) - z * (1 - Phi(z))"""
    return norm.pdf(z) - z * (1 - norm.cdf(z))

def solve_z_for_fill_rate(target_beta: float, sigma_ddlt: float, Q: float) -> float:
    """Find z such that fill_rate(z) = target_beta. Newton's method."""
    target_loss = (1 - target_beta) * Q / sigma_ddlt

    # Newton's method to invert loss function
    z = 0.0  # initial guess
    for _ in range(50):
        loss = _normal_loss_function(z)
        loss_deriv = -(1 - norm.cdf(z))  # dL/dz
        z = z - (loss - target_loss) / loss_deriv
        if abs(loss - target_loss) < 1e-10:
            break

    return z
```

### 11.8 Computation Flow (Full Pipeline)

```
For each DFU in portfolio:
    1. Load demand history (fact_sales_monthly, trailing 24 months)
    2. Load champion forecast (fact_external_forecast_monthly, model_id='champion')
    3. Load forecast errors (backtest_lag_archive, champion model, target lag)
    4. Load lead time data (dim_dfu.total_lt + supplier history from fact_purchase_orders)
    5. Load policy config (dim_inventory_policy: target SL, method, review period)
    6. Load cost params (dim_inventory_cost: holding, ordering, shortage costs)
    7. Classify demand pattern (Syntetos-Boylan: smooth/erratic/intermittent/lumpy)
    8. Select SS method (from policy config or ABC-XYZ default)
    9. Compute safety stock (using selected method)
   10. Compute EOQ (with MOQ and order multiple constraints)
   11. Compute reorder point: ROP = forecast_mean × lead_time + SS
   12. Compute order-up-to level: S = ROP + EOQ (for s-S policies)
   13. Compute achieved service level (via fill rate formula or simulation)
   14. Compute total annual cost: holding + ordering + shortage
   15. Write to fact_inventory_plan
   16. Generate alerts if: SS changed > 20%, DOS < threshold, stockout risk
```

Vectorized implementation processes all DFUs in batches of 1000 using NumPy arrays.

---

## 12. Optimization Solvers

### 12.1 Real-Time Solvers (< 1 second, in API request path)

Used for single-item computations in the UI (e.g., user changes service level slider for one item).

**Single-Item SS/ROP/EOQ:**

```python
# NumPy vectorized — sub-millisecond per item
ss = norm.ppf(service_level) * sigma_ddlt
rop = demand_mean * lead_time + ss
eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
```

**Service Level to z-value conversion:**

```python
z = scipy.stats.norm.ppf(service_level)  # < 0.1ms
```

**Fill rate calculation:**

```python
fill_rate = 1 - sigma_ddlt * normal_loss(z) / Q  # < 0.1ms
```

### 12.2 Near-Real-Time Solvers (1–30 seconds, async)

Used for category-level optimizations and what-if scenario evaluations.

**Category-Level SS Optimization (Gradient-Based):**

Minimize total cost across all items in a category, subject to service level constraints:

```python
from scipy.optimize import minimize

def total_cost(ss_vector, demands, costs, service_levels):
    holding = np.sum(ss_vector * costs['holding'])
    shortage = np.sum(
        costs['shortage'] * demands * (1 - compute_fill_rates(ss_vector, ...))
    )
    return holding + shortage

result = minimize(
    total_cost,
    x0=initial_ss,
    method='L-BFGS-B',
    bounds=[(0, None)] * n_items,
    constraints=[{'type': 'ineq', 'fun': service_level_constraint}]
)
```

**LP Budget Allocation (PuLP + HiGHS):**

Maximize total fill rate subject to budget constraint:

```python
import pulp

prob = pulp.LpProblem("SS_Budget_Allocation", pulp.LpMaximize)

# Decision variables: safety stock per item (continuous)
ss = [pulp.LpVariable(f"ss_{i}", lowBound=0) for i in range(n_items)]

# Objective: maximize weighted service level
prob += pulp.lpSum([weight[i] * service_level_approx(ss[i]) for i in range(n_items)])

# Budget constraint
prob += pulp.lpSum([ss[i] * unit_cost[i] for i in range(n_items)]) <= budget

# Minimum service level per ABC class
for i in range(n_items):
    prob += service_level_approx(ss[i]) >= min_sl[abc_class[i]]

prob.solve(pulp.HiGHS_CMD(msg=0, timeLimit=30))
```

### 12.3 Batch Solvers (1–60 minutes, background job)

Used for portfolio-wide optimization runs (weekly batch job).

**Portfolio-Wide SS Optimization (Lagrangian Relaxation):**

```python
def lagrangian_ss_optimization(
    items: pd.DataFrame,     # item data with demand stats, costs
    budget: float,           # total SS investment budget
    epsilon: float = 0.01,   # convergence tolerance
) -> pd.DataFrame:
    """
    Binary search on Lagrange multiplier lambda.
    For each lambda, compute optimal per-item SS.
    Adjust lambda until budget constraint is satisfied.
    """
    lambda_lo, lambda_hi = 0.0, 1000.0

    for iteration in range(100):
        lam = (lambda_lo + lambda_hi) / 2

        # Per-item optimal SS given lambda
        # At optimum: marginal service value = lambda * marginal holding cost
        # phi(z_i) / sigma_i = lambda * c_i * h_rate
        # z_i = norm.ppf(1 - lambda * c_i * h_rate * sigma_i / phi(z_i))  [iterative]
        ss_vector = compute_ss_for_lambda(items, lam)

        total_investment = np.sum(ss_vector * items['unit_cost'])

        if abs(total_investment - budget) / budget < epsilon:
            break
        elif total_investment > budget:
            lambda_lo = lam  # increase penalty → reduce SS
        else:
            lambda_hi = lam  # decrease penalty → increase SS

    return ss_vector
```

**Monte Carlo Policy Evaluation (see Section 13):**

Simulates inventory operations over 12–24 month horizon, computing realized service level, total cost, and stockout frequency for each policy configuration.

**Multi-Objective Optimization (NSGA-II via DEAP):**

Simultaneously minimizes total cost and maximizes total service level:

```python
from deap import base, creator, tools, algorithms

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # min cost, max SL
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    ss_vector = np.array(individual)
    total_cost = compute_total_cost(ss_vector)
    total_sl = compute_portfolio_service_level(ss_vector)
    return (total_cost, total_sl)

# NSGA-II parameters
toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selNSGA2)
# ... register crossover, mutation

pop, logbook = algorithms.eaMuPlusLambda(
    pop, toolbox, mu=100, lambda_=200,
    cxpb=0.7, mutpb=0.3, ngen=200
)

# Extract Pareto frontier
pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
```

### 12.4 Solver Selection Logic

The engine auto-selects the appropriate solver based on problem characteristics:

```python
def select_solver(
    n_items: int,
    has_constraints: bool,
    constraint_types: list,  # ['budget', 'moq', 'service_level', 'space']
    time_budget_seconds: float,
) -> str:
    if n_items == 1:
        return "analytical"        # < 1ms
    elif n_items <= 100 and not has_constraints:
        return "vectorized_numpy"  # < 100ms
    elif 'moq' in constraint_types or 'integer' in constraint_types:
        return "mip_pulp_highs"    # MIP for integer constraints
    elif has_constraints and n_items <= 50000:
        return "lp_pulp_highs"     # LP for continuous with constraints
    elif time_budget_seconds >= 60 and n_items <= 10000:
        return "nsga2_deap"        # multi-objective Pareto
    elif n_items <= 100000:
        return "lagrangian"        # scalable budget-constrained
    else:
        return "heuristic_greedy"  # largest portfolio sizes
```

---

## 13. Monte Carlo Simulation Engine

The Monte Carlo engine evaluates inventory policies by simulating day-by-day inventory operations over a planning horizon.

### 13.1 Simulation Architecture

```python
def simulate_inventory_policy(
    demand_distribution: str,    # 'normal', 'poisson', 'negbin', 'bootstrap'
    demand_params: dict,         # mean, std, or historical data for bootstrap
    lead_time_distribution: str, # 'constant', 'normal', 'empirical'
    lead_time_params: dict,
    policy_type: str,            # 'rQ', 'sS', 'RS'
    policy_params: dict,         # s, Q, S, R as applicable
    horizon_days: int = 365,
    n_replications: int = 10000,
    initial_inventory: float = None,
) -> SimulationResult:
    """
    Vectorized Monte Carlo simulation.
    Runs n_replications in parallel using NumPy broadcasting.
    """
    # Pre-generate all random variates (vectorized)
    if demand_distribution == 'normal':
        demands = np.random.normal(
            demand_params['mean'], demand_params['std'],
            size=(n_replications, horizon_days)
        )
        demands = np.maximum(demands, 0)  # demand >= 0
    elif demand_distribution == 'poisson':
        demands = np.random.poisson(
            demand_params['mean'], size=(n_replications, horizon_days)
        )
    elif demand_distribution == 'bootstrap':
        demands = np.random.choice(
            demand_params['history'],
            size=(n_replications, horizon_days),
            replace=True
        )

    # Pre-generate lead times
    if lead_time_distribution == 'constant':
        lead_times = np.full(
            (n_replications, horizon_days), lead_time_params['mean']
        )
    elif lead_time_distribution == 'normal':
        lead_times = np.random.normal(
            lead_time_params['mean'], lead_time_params['std'],
            size=(n_replications, horizon_days)
        ).clip(min=1).astype(int)

    # Simulate day-by-day (vectorized across replications)
    on_hand = np.full(n_replications, initial_inventory or policy_params['s'] + policy_params.get('Q', 0))
    total_demand = np.zeros(n_replications)
    total_filled = np.zeros(n_replications)
    total_orders = np.zeros(n_replications)
    stockout_days = np.zeros(n_replications)
    holding_units = np.zeros(n_replications)

    # Pending orders queue (simplified: track arrival day and quantity)
    # ... (day-by-day simulation loop)

    return SimulationResult(
        fill_rate=np.mean(total_filled / total_demand),
        fill_rate_ci=_confidence_interval(total_filled / total_demand, 0.95),
        avg_inventory=np.mean(holding_units / horizon_days),
        stockout_frequency=np.mean(stockout_days / horizon_days),
        total_cost=np.mean(total_cost_per_rep),
        total_cost_ci=_confidence_interval(total_cost_per_rep, 0.95),
        n_orders=np.mean(total_orders),
    )
```

### 13.2 Performance Targets

| Scenario | Items | Replications | Target Time |
|----------|-------|-------------|-------------|
| Single-item deep simulation | 1 | 100,000 | < 1s |
| Category evaluation | 100 | 10,000 | < 10s |
| Portfolio screening | 10,000 | 1,000 | < 30s |
| Full portfolio deep simulation | 10,000 | 10,000 | < 5 min |
| Scenario comparison (2 policies) | 10,000 | 10,000 each | < 10 min |

Achieved via:
- NumPy vectorized operations (no Python loops over replications)
- Pre-generated random variates
- Batch processing (1000 items at a time to manage memory)
- Optional: Numba JIT compilation for the day-by-day loop

### 13.3 Confidence Interval Computation

For 10,000 replications, the 95% confidence interval width for fill rate:

```
CI_width = 2 × 1.96 × std(fill_rate_per_rep) / sqrt(n_replications)
```

For fill rate ~ 0.95 with std ~ 0.03: CI_width = 2 × 1.96 × 0.03 / 100 = 0.0012 (0.12 percentage points). This is sufficiently precise for decision-making.

---

## 14. Anomaly Detection

### 14.1 Demand Outlier Detection

Before computing safety stock, filter demand outliers to prevent inflated safety stock:

```python
from sklearn.ensemble import IsolationForest

def detect_demand_anomalies(
    demand_history: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Flag anomalous demand observations using Isolation Forest on residuals.
    Residuals = actual - rolling median (removes trend/seasonality).
    """
    demand_history['rolling_median'] = demand_history['qty'].rolling(6, center=True).median()
    demand_history['residual'] = demand_history['qty'] - demand_history['rolling_median']

    iso = IsolationForest(contamination=contamination, random_state=42)
    demand_history['is_anomaly'] = iso.fit_predict(demand_history[['residual']]) == -1

    return demand_history
```

Anomalous observations are excluded from demand statistics (mean, std, CV) but retained in the dataset with a flag. Planners can review and override.

### 14.2 Lead Time Spike Detection

```python
def detect_lead_time_spikes(
    recent_lt: float,           # most recent actual lead time
    historical_mean: float,
    historical_std: float,
    threshold_z: float = 2.5,
) -> dict:
    z_score = (recent_lt - historical_mean) / max(historical_std, 0.1)
    is_spike = z_score > threshold_z

    return {
        "is_spike": is_spike,
        "z_score": z_score,
        "recent_lt": recent_lt,
        "expected_lt": historical_mean,
        "deviation_days": recent_lt - historical_mean,
    }
```

If a lead time spike is detected, generate a `lead_time_spike` alert and optionally increase safety stock temporarily.

### 14.3 Inventory Discrepancy Detection

Compare actual on-hand against expected on-hand (based on receipts minus shipments):

```python
expected_on_hand = previous_on_hand + receipts - shipments - adjustments
actual_on_hand = current_snapshot.qty_on_hand
discrepancy = actual_on_hand - expected_on_hand

if abs(discrepancy) > threshold:
    generate_alert('inventory_discrepancy', severity='warning', ...)
```

---

# Part IV: What-If Scenarios

The Inventory Planning Module supports 11 distinct what-if scenario types — more than any commercial competitor. Each scenario follows the same execution pattern established by the What-If Clustering Scenarios (Feature 29): async background execution, concurrency guard, results stored in `fact_inventory_scenario`, and UI-driven promote-to-production.

---

## 15. Service Level What-If

**Purpose:** Evaluate the cost of increasing or decreasing service level targets.

**Input Parameters:**
- `target_service_level`: float (0.80 to 0.999)
- `scope`: "all" | "abc_class" | "category" | "item_list"
- `scope_filter`: filter value (e.g., "A", "Grocery", [list of items])

**Computation:**

```python
def scenario_service_level(target_sl: float, scope: str, scope_filter):
    baseline = load_current_plan(scope, scope_filter)
    scenario = recompute_safety_stock(
        items=baseline,
        target_service_level=target_sl,
        # all other params unchanged
    )
    return {
        "baseline_sl": np.mean(baseline['achieved_sl']),
        "scenario_sl": target_sl,
        "baseline_ss_value": np.sum(baseline['ss_investment']),
        "scenario_ss_value": np.sum(scenario['ss_investment']),
        "delta_investment": scenario_ss_value - baseline_ss_value,
        "delta_pct": (scenario_ss_value - baseline_ss_value) / baseline_ss_value * 100,
        "items_changed": np.sum(scenario['safety_stock'] != baseline['safety_stock']),
        "cost_of_service_curve": _build_cost_curve(baseline, np.arange(0.80, 1.00, 0.01)),
    }
```

**Visualization:**
- Scatter plot: service level % (x-axis) vs. total inventory investment $ (y-axis) — shows the cost-of-service curve
- Marginal cost curve: d(Investment)/d(SL) — shows diminishing returns at high service levels
- Delta table: items with largest absolute change in SS

## 16. Lead Time What-If

**Purpose:** Evaluate impact of lead time changes (supplier improvement or disruption).

**Input Parameters:**
- `lead_time_delta_days`: integer (+/- days)
- `scope`: "all" | "supplier" | "item_list"
- `scope_filter`: supplier_id or item list

**Computation:**

```python
def scenario_lead_time(lt_delta: int, scope, scope_filter):
    baseline = load_current_plan(scope, scope_filter)
    adjusted_lt = baseline['lead_time_mean'] + lt_delta

    scenario = recompute_safety_stock(
        items=baseline,
        lead_time_override=adjusted_lt,
    )

    return {
        "baseline_lt_avg": np.mean(baseline['lead_time_mean']),
        "scenario_lt_avg": np.mean(adjusted_lt),
        "baseline_ss_total": np.sum(baseline['safety_stock']),
        "scenario_ss_total": np.sum(scenario['safety_stock']),
        "ss_delta_units": scenario_ss - baseline_ss,
        "ss_delta_value": (scenario_ss - baseline_ss) * baseline['unit_cost'],
        "pipeline_inventory_delta": lt_delta * np.sum(baseline['forecast_mean']),
    }
```

**Visualization:**
- Grouped bar chart: baseline vs. scenario SS per category
- Waterfall chart: decomposition of inventory investment change

## 17. Demand Variability What-If

**Purpose:** Stress-test safety stock against demand volatility changes.

**Input Parameters:**
- `variability_multiplier`: float (0.5 to 3.0) — multiplies demand std dev
- `scope`: "all" | "category" | "cluster"

**Computation:**

```python
def scenario_demand_variability(multiplier: float, scope, scope_filter):
    baseline = load_current_plan(scope, scope_filter)

    scenario = recompute_safety_stock(
        items=baseline,
        demand_std_override=baseline['forecast_std'] * multiplier,
    )

    return {
        "multiplier": multiplier,
        "baseline_avg_cv": np.mean(baseline['forecast_std'] / baseline['forecast_mean']),
        "scenario_avg_cv": np.mean(baseline['forecast_std'] * multiplier / baseline['forecast_mean']),
        "baseline_ss_value": np.sum(baseline['ss_investment']),
        "scenario_ss_value": np.sum(scenario['ss_investment']),
        "sensitivity_curve": _build_sensitivity_curve(baseline, np.arange(0.5, 3.0, 0.1)),
    }
```

## 18. Supply Disruption What-If

**Purpose:** Model the impact of a supplier disruption on inventory and service levels.

**Input Parameters:**
- `supplier_id`: string
- `disruption_duration_days`: integer
- `disruption_start_date`: date (optional, default = today)
- `alternative_supplier_id`: string (optional)

**Computation:**

```python
def scenario_supply_disruption(supplier_id, duration_days, alt_supplier_id=None):
    affected_items = get_items_by_supplier(supplier_id)

    # Effective lead time = current LT + disruption duration
    disrupted_lt = affected_items['lead_time_mean'] + duration_days

    # If alternative supplier, use their lead time after disruption ends
    if alt_supplier_id:
        alt_lt = get_supplier_lead_time(alt_supplier_id, affected_items)
        effective_lt = np.minimum(disrupted_lt, alt_lt)
    else:
        effective_lt = disrupted_lt

    scenario = recompute_safety_stock(items=affected_items, lead_time_override=effective_lt)
    buffer_needed = scenario['safety_stock'] - affected_items['current_safety_stock']

    return {
        "affected_items": len(affected_items),
        "affected_revenue_pct": affected_items['annual_revenue'].sum() / total_revenue * 100,
        "buffer_stock_needed": buffer_needed.sum(),
        "buffer_investment": (buffer_needed * affected_items['unit_cost']).sum(),
        "estimated_stockout_items": np.sum(affected_items['current_position'] < scenario['reorder_point']),
        "days_until_stockout": _compute_days_until_stockout(affected_items),
        "alternative_sourcing": alt_supplier_id is not None,
    }
```

## 19. Budget Constraint What-If

**Purpose:** Find optimal SS allocation given a fixed budget ceiling.

**Input Parameters:**
- `budget_ceiling`: float ($)
- `optimization_objective`: "fill_rate" | "weighted_fill_rate" | "stockout_count"
- `min_service_levels`: dict (per ABC class)

**Computation:**

Uses Lagrangian relaxation (Section 12.3) to solve:

```
Maximize: sum_i weight_i × fill_rate_i(SS_i)
Subject to: sum_i SS_i × unit_cost_i <= budget_ceiling
            fill_rate_i(SS_i) >= min_SL[abc_class_i]   for all i
```

**Output:**
- Per-item SS allocation (optimized)
- Achieved fill rate per item and overall
- Pareto frontier: budget vs. fill rate curve
- Shadow price of budget constraint (marginal fill rate per additional $1)

**Visualization:**
- Efficient frontier scatter: budget (x) vs. portfolio fill rate (y)
- Tornado sensitivity diagram: which items/categories are most sensitive to budget changes
- Budget allocation waterfall: how the budget is distributed across ABC classes

## 20. Network/Sourcing What-If

**Purpose:** Evaluate inventory savings from network changes (consolidation, dual sourcing).

### 20.1 Risk Pooling (Location Consolidation)

**Input Parameters:**
- `consolidation_group`: list of locations to consolidate
- `target_location`: the consolidated location

**Computation:**

```
Current total SS = sum(SS_i) for i in consolidation group
Pooled demand std = sqrt(sum(sigma_i^2) + 2 * sum_{i<j} rho_ij * sigma_i * sigma_j)
```

If demands are independent (rho = 0):

```
Pooled sigma = sqrt(sum(sigma_i^2))
SS_pooled = z × sqrt(sum(sigma_i^2)) × sqrt(L)

Risk pooling savings = 1 - SS_pooled / sum(SS_i)
For n identical locations: savings = 1 - 1/sqrt(n)  (e.g., 4 locations → 50% savings)
```

### 20.2 Dual Sourcing

**Input Parameters:**
- `primary_supplier_id`, `secondary_supplier_id`
- `split_ratio`: fraction to primary (e.g., 0.7)

**Computation:**

```
Effective lead time = weighted average of supplier lead times
LT_eff = split × LT_primary + (1 - split) × LT_secondary
LT_std_eff = sqrt(split^2 × LT_std_primary^2 + (1-split)^2 × LT_std_secondary^2)

SS_dual < SS_single (because effective LT is shorter and less variable)
```

## 21. ABC Reclassification What-If

**Purpose:** Evaluate impact of changing ABC/XYZ threshold boundaries.

**Input Parameters:**
- `a_threshold`: float (default 0.80)
- `b_threshold`: float (default 0.95)
- `x_cv_threshold`: float (default 0.50)
- `y_cv_threshold`: float (default 1.00)

**Computation:**

```python
def scenario_abc_reclassify(a_pct, b_pct, x_cv, y_cv):
    # Reclassify all items
    new_classes = classify_abc_xyz(items, a_pct, b_pct, x_cv, y_cv)

    # Apply new policy recommendations per 9-cell matrix
    new_policies = assign_policies_from_matrix(new_classes)

    # Recompute SS with new policies
    scenario = recompute_safety_stock(items, policies=new_policies)

    # Migration matrix: count items moving between cells
    migration = compute_migration_matrix(current_classes, new_classes)

    return {
        "migration_matrix": migration,   # 9x9 matrix of item counts
        "class_counts": new_classes.value_counts(),
        "baseline_investment": current_plan['ss_investment'].sum(),
        "scenario_investment": scenario['ss_investment'].sum(),
        "delta_investment": ...,
        "items_reclassified": np.sum(new_classes != current_classes),
    }
```

## 22. Seasonal Buildup What-If

**Purpose:** Plan pre-season inventory buildup for seasonal items.

**Integration:** Uses existing seasonality profiles from Feature 30 (`dim_dfu.seasonality_profile`, `peak_month`, `trough_month`).

**Input Parameters:**
- `target_peak_service_level`: float (usually higher than normal SL for peak season)
- `buildup_start_months_before_peak`: integer (1–4)
- `buildup_rate`: "linear" | "front_loaded" | "back_loaded"

**Computation:**

```python
def scenario_seasonal_buildup(target_peak_sl, start_months, rate):
    seasonal_items = get_items_by_seasonality('medium', 'high')

    for item in seasonal_items:
        peak = item['peak_month']
        trough = item['trough_month']
        peak_demand = item['forecast_mean'] * item['peak_trough_ratio']

        # Safety stock during peak uses higher SL target
        ss_peak = compute_ss(demand=peak_demand, sl=target_peak_sl, ...)

        # Build schedule: ramp up SS from normal to peak level
        buildup_schedule = _generate_buildup_schedule(
            normal_ss=item['safety_stock'],
            peak_ss=ss_peak,
            start_month=peak - start_months,
            peak_month=peak,
            rate=rate,
        )

    return {
        "seasonal_items_count": len(seasonal_items),
        "peak_ss_investment": ...,
        "incremental_investment": peak_ss - normal_ss total,
        "buildup_schedules": buildup_schedule_per_item,
        "timing_recommendation": f"Start buildup {start_months} months before peak",
    }
```

## 23. Promotion Impact What-If

**Purpose:** Pre-position inventory for planned promotions.

**Input Parameters:**
- `promo_id`: reference to `dim_promotion`
- `lift_factor_override`: optional override of expected lift

**Computation:**

```python
def scenario_promotion(promo_id, lift_factor_override=None):
    promo = load_promotion(promo_id)
    lift = lift_factor_override or promo['lift_factor']

    # Expected demand during promo
    baseline_demand = forecast_during_period(promo['start_date'], promo['end_date'])
    promo_demand = baseline_demand * lift

    # Additional inventory needed
    additional_qty = promo_demand - baseline_demand
    pre_position_date = promo['start_date'] - timedelta(days=lead_time)

    return {
        "baseline_demand": baseline_demand,
        "expected_promo_demand": promo_demand,
        "additional_inventory_needed": additional_qty,
        "pre_position_date": pre_position_date,
        "investment": additional_qty * unit_cost,
    }
```

## 24. MOQ/Order Multiple What-If

**Purpose:** Evaluate impact of changing minimum order quantities or order multiples.

**Input Parameters:**
- `scope`: "supplier" | "item_list"
- `moq_override`: new MOQ value
- `order_multiple_override`: new order multiple

**Computation:**

```
If MOQ > EOQ: effective order quantity = MOQ (increases cycle stock)
Cycle stock increase = (MOQ - EOQ) / 2
Additional holding cost = cycle_stock_increase × unit_cost × holding_rate
```

## 25. Scenario Management

### 25.1 Execution Pattern

All scenarios follow the same async execution pattern as What-If Clustering (Feature 29):

```python
import threading

_scenario_lock = threading.Lock()

@router.post("/inventory/scenario")
async def run_scenario(request: ScenarioRequest):
    if not _scenario_lock.acquire(blocking=False):
        raise HTTPException(409, "Another scenario is running")

    scenario_id = str(uuid.uuid4())
    # Insert pending row
    await insert_scenario(scenario_id, request.scenario_type, request.params)

    # Run in background thread
    thread = threading.Thread(
        target=_execute_scenario,
        args=(scenario_id, request),
    )
    thread.start()

    return {"scenario_id": scenario_id, "status": "running"}
```

### 25.2 Save/Load and Versioning

- Each scenario is persisted in `fact_inventory_scenario` with full parameters and results.
- Scenarios can be named and tagged for comparison.
- Version history: same scenario type can be re-run with different parameters.

### 25.3 A/B Comparison

Compare any two scenarios (or scenario vs. baseline):

```
GET /inventory/scenario/compare?baseline_id=xxx&scenario_id=yyy
```

Returns:
- Per-item comparison table: SS baseline, SS scenario, delta, impact
- Aggregate KPI comparison: total investment, fill rate, turns, DOS
- Category breakdown

### 25.4 Promote to Production

Apply scenario recommendations as the new operational plan:

```
POST /inventory/scenario/{id}/promote
```

This:
1. Updates `dim_inventory_policy` with scenario's recommended policies
2. Updates `fact_inventory_plan` with scenario's recommended SS/ROP/EOQ
3. Marks scenario status as "promoted"
4. Logs the action for audit trail

---

# Part V: User Interface

---

## 26. Navigation Architecture

### Top-Level Tab Integration

A new "Inventory" tab is added to the existing tab bar:

```
[ Explorer | Accuracy | DFU Analysis | Clusters | Market Intel | Chat | Inventory ]
```

- Keyboard shortcut: `7` switches to Inventory tab (consistent with existing 1–6 mappings)
- `VALID_TABS` array in `useUrlState.ts` extended to include `"inventory"`
- Lazy-loaded via `React.lazy(() => import('./tabs/InventoryTab'))` with Suspense fallback

### Sub-Navigation Within Inventory Tab

The Inventory tab uses a secondary horizontal tab bar (pill-style) below the main tab bar:

```
[ Overview | Position | Replenishment | Safety Stock | Classification | Scenarios | Suppliers | Alerts ]
```

- Sub-tab state stored in URL: `?tab=inventory&sub=safety-stock`
- Keyboard shortcuts: Ctrl+1 through Ctrl+8 switch sub-tabs
- Each sub-tab is a separate component in `frontend/src/tabs/inventory/`

### Persistent Global Filters

A filter bar persists across all sub-tabs:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Category: [All ▼]  Location: [All ▼]  ABC: [All ▼]  Status: [All ▼]│
│ Date Range: [2026-01-01] to [2026-02-24]   [Apply] [Reset]          │
└─────────────────────────────────────────────────────────────────────┘
```

Filters are managed via React context and applied to all TanStack Query keys for automatic cache invalidation.

---

## 27. Dashboard / Overview Panel

### KPI Cards (Top Row)

9 KPI cards in a 3x3 responsive grid, each with:
- Metric name and value (large font)
- Traffic light indicator (green/yellow/red dot)
- Trend arrow (up/down/flat) with percentage change vs. prior month
- Sparkline chart (trailing 12 months)

| Card | Metric | Green | Yellow | Red |
|------|--------|-------|--------|-----|
| 1 | Total Inventory Value | Within 10% of target | 10-20% over target | >20% over target |
| 2 | Average DOS | Within target range | ±20% of target | >30% deviation |
| 3 | Inventory Turns | >= target | 80-100% of target | < 80% of target |
| 4 | Fill Rate | >= 97% | 93-97% | < 93% |
| 5 | Stockout Count | 0-2% items | 2-5% items | > 5% items |
| 6 | Excess Value | < 5% of total | 5-10% of total | > 10% of total |
| 7 | Open PO Value | Within budget | ±15% of budget | > 15% over budget |
| 8 | GMROI | >= 2.0 | 1.5-2.0 | < 1.5 |
| 9 | Active Alerts | 0 critical | 1-3 critical | > 3 critical |

### Dashboard Charts (Bottom Section)

```
┌────────────────────────────────┬──────────────────────────────────┐
│   DOS Histogram                │   Service Level vs Target        │
│   (bar chart, 0-120+ days)     │   (bullet chart by ABC class)    │
├────────────────────────────────┼──────────────────────────────────┤
│   Inventory Turns Trend        │   Fill Rate Trend                │
│   (line chart, 12 months)      │   (line chart, 12 months)        │
├────────────────────────────────┼──────────────────────────────────┤
│   Carrying Cost Trend          │   Stock Health Donut             │
│   (area chart, 12 months)      │   (Healthy/Watch/Critical/SO)    │
└────────────────────────────────┴──────────────────────────────────┘
```

All charts use ECharts (via `EChartContainer.tsx`) for canvas-based rendering, supporting 10K+ data points.

Click-through: clicking any chart element navigates to the relevant sub-tab with appropriate filters applied.

---

## 28. Inventory Position View

### Data Grid

Virtualized TanStack Table (TanStack Table + TanStack Virtual) supporting 10K+ rows:

**Columns:**
| Column | Type | Width | Sort | Filter |
|--------|------|-------|------|--------|
| Item No | Text | 100px | Yes | Trigram search |
| Description | Text | 200px | No | Trigram search |
| Location | Text | 80px | Yes | Exact |
| On-Hand | Number | 90px | Yes | Range |
| In-Transit | Number | 90px | Yes | Range |
| Allocated | Number | 90px | Yes | Range |
| Available | Number (computed) | 90px | Yes | Range |
| ROP | Number | 80px | Yes | Range |
| Safety Stock | Number | 90px | Yes | Range |
| DOS | Number | 70px | Yes | Range |
| Status | Badge | 100px | Yes | Dropdown |
| ABC | Badge | 50px | Yes | Dropdown |
| XYZ | Badge | 50px | Yes | Dropdown |
| Trend | Sparkline | 80px | No | No |

**Status Badges:**

| Status | Color | Condition |
|--------|-------|-----------|
| Overstock | Blue (#3b82f6) | DOS > 2 × target_DOS |
| Healthy | Green (#22c55e) | target_DOS × 0.5 <= DOS <= target_DOS × 1.5 |
| Low | Yellow (#eab308) | SS_days < DOS < target_DOS × 0.5 |
| Stockout | Red (#ef4444) | qty_on_hand = 0 |
| Dead | Gray (#6b7280) | No demand for 12+ months, qty_on_hand > 0 |

**Inline Sparklines:** 12-month inventory trend rendered as a tiny line chart in each row using a lightweight canvas-based renderer (not ECharts — too heavy for per-row rendering).

**Row Click:** Opens the Item Detail drawer (Section 29).

**Toolbar:**
- Export CSV (via papaparse, existing `export.ts` utility)
- Column visibility toggle
- Density toggle (compact/comfortable)
- Refresh button

---

## 29. Item Detail / DFU Drill-Down

A slide-over drawer (50% viewport width on desktop, full width on mobile) appears when clicking a row in the Position grid.

### Header Section

```
┌──────────────────────────────────────────────────────┐
│ Item: 100320 — Industrial Widget                     │
│ Location: 1401-BULK  │  Category: Hardware           │
│ ABC: A  │  XYZ: X  │  Cluster: high_volume_steady   │
│ Seasonality: medium  │  Champion: lgbm_global        │
└──────────────────────────────────────────────────────┘
```

### 6 KPI Mini-Cards

```
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ On-Hand │   DOS   │   SS    │   ROP   │Fill Rate│  Turns  │
│  2,450  │  38 d   │   820   │  1,640  │  98.2%  │  10.4   │
│  +120   │  +3 d   │  -40    │  -80    │  +0.3%  │  +0.8   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### Demand Chart

Dual-axis chart showing:
- Left axis: quantity (units)
- Right axis: accuracy %
- Series: historical sales (bar), champion forecast (line), confidence interval (shaded area), actual demand (dots for periods with actuals)
- Horizon: 12 months history + 6 months forecast

### Inventory Projection Chart

12-month forward projection:
- On-hand inventory level (step-line, accounts for discrete receipts)
- Safety stock threshold (dashed red line)
- Reorder point (dashed amber line)
- Planned receipt markers (green triangles)
- Estimated stockout date marker (red diamond, if applicable)

### Supply Timeline

Gantt-style visualization of open POs:
```
PO-12345 ████████████░░░░░░░░ (70% received, 30% outstanding)
PO-12389 ░░░░░░████████████░░ (in transit, ETA: Mar 15)
PO-12401 ░░░░░░░░░░░░░░░░░░░░ (ordered, not shipped, ETA: Apr 02)
```

### Policy Panel

```
┌──────────────────────────┬──────────────────────────┐
│      Current Policy      │   Recommended Policy     │
├──────────────────────────┼──────────────────────────┤
│ Service Level: 98%       │ Service Level: 98%       │
│ Policy: (s, Q)           │ Policy: (s, Q)           │
│ SS Method: Normal        │ SS Method: Empirical     │ ← change recommended
│ Safety Stock: 860        │ Safety Stock: 820        │ ← lower by 40
│ Reorder Point: 1,720     │ Reorder Point: 1,640     │
│ Order Qty (EOQ): 1,200   │ Order Qty (EOQ): 1,200   │
│                          │                          │
│                          │ [Apply Recommendation]   │
│                          │ [Override Manually]       │
└──────────────────────────┴──────────────────────────┘
```

---

## 30. Safety Stock Optimizer Panel

### Controls

- **Service level slider**: 80%–99.9% with tick marks at 90, 95, 98, 99
- **Method selector**: dropdown (Normal / Empirical / Conformal / Bootstrap / Auto)
- **Scope selector**: All items / By ABC class / By category / Custom filter
- **[Optimize] button**: triggers computation

### Summary Cards

After optimization:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Total SS $   │ Items Changed│ Avg SS Change│ Investment Δ │
│ $12.4M       │ 3,247        │ -8.2%        │ -$1.1M       │
│ (-8.1% Δ)    │ (of 10,000)  │              │ (savings)    │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Recommendation Grid

TanStack Table with columns:
- Item No, Description, Location
- Current SS, Recommended SS, Delta, Delta %
- Current SL, Achieved SL
- Impact $ (delta × unit cost)
- Method Used
- [Expand] button → full calculation breakdown

### Batch Actions

- [Apply All Recommendations] — bulk update to `dim_inventory_policy` and `fact_inventory_plan`
- [Apply Selected] — apply only checked rows
- [Export to CSV] — download recommendations
- [Undo Last Apply] — revert to previous plan version

### Manual Override

Click any row → edit modal with:
- Safety stock override value
- Reason code dropdown (Promotional, Supplier Risk, Obsolescence, Executive Decision, Other)
- Free-text notes
- Sets `is_manual_override = true` in `dim_inventory_policy`

---

## 31. Replenishment Recommendations Panel

### Priority-Ranked Recommendations

Cards grouped by priority:

```
🔴 CRITICAL (5 items) — Stockout or will stockout within lead time
🟡 URGENT (23 items) — Below reorder point, need immediate order
🟢 NORMAL (147 items) — Approaching reorder point, scheduled order
🔵 PLANNED (312 items) — Upcoming review cycle orders
```

### Recommendation Grid

| Column | Description |
|--------|-------------|
| Priority | Critical / Urgent / Normal / Planned |
| Item No | Item identifier |
| Location | Location code |
| Available | Current available stock |
| ROP | Reorder point |
| Suggested Qty | Recommended order quantity |
| Supplier | Primary supplier name |
| Lead Time | Expected lead time (days) |
| Est. Delivery | Estimated delivery date |
| Investment | Suggested qty × unit cost |
| [Actions] | Create PO / Skip / Defer |

### Batch PO Creation

1. Select items → [Create POs]
2. System groups selected items by supplier
3. Generates one PO per supplier with line items
4. Shows PO preview with totals
5. [Confirm] creates PO records in `fact_purchase_orders`

### Calendar View Toggle

Toggle between grid view and calendar view:
- Calendar shows expected delivery dates
- Color-coded by priority
- Drag-and-drop to adjust timing

---

## 32. ABC-XYZ Classification Panel

### 3x3 Heat Map Matrix

Interactive 3x3 grid visualization:

```
         X (Stable)    Y (Variable)    Z (Unpredictable)
      ┌──────────────┬──────────────┬──────────────┐
  A   │    AX        │    AY        │    AZ        │
(High)│  847 items   │  412 items   │   89 items   │
      │  $45.2M      │  $18.7M      │   $4.1M      │
      │  SL: 98-99%  │  SL: 97-98%  │  SL: 95-97%  │
      ├──────────────┼──────────────┼──────────────┤
  B   │    BX        │    BY        │    BZ        │
(Med) │  1,204 items │  892 items   │  213 items   │
      │  $8.3M       │  $5.1M       │   $1.2M      │
      │  SL: 95-97%  │  SL: 94-96%  │  SL: 90-94%  │
      ├──────────────┼──────────────┼──────────────┤
  C   │    CX        │    CY        │    CZ        │
(Low) │  2,156 items │  2,789 items │  1,398 items │
      │  $1.8M       │  $1.2M       │   $0.4M      │
      │  SL: 90-93%  │  SL: 85-90%  │  SL: 80-85%  │
      └──────────────┴──────────────┴──────────────┘
```

Cell color intensity based on inventory value. Click any cell to filter the position grid to that segment.

### Pareto Chart

Dual-axis chart:
- Bar chart: items sorted by revenue (descending)
- Line chart: cumulative revenue % (0–100%)
- Vertical threshold lines at A/B and B/C boundaries
- Color-coded bars by ABC class

### Interactive Threshold Sliders

```
ABC Thresholds:          XYZ Thresholds:
A: [======|====] 80%     X: [====|======] CV 0.50
B: [=========|=] 95%     Y: [========|==] CV 1.00
```

Dragging sliders immediately updates:
- Item counts per class
- Heat map values
- Policy implications summary

### Policy Assignment

Table showing current vs. recommended policy per cell:

| Cell | Items | Current SL | Recommended SL | Current Method | Recommended Method | Policy |
|------|-------|-----------|---------------|---------------|-------------------|--------|
| AX | 847 | 97.5% | 98.5% | Normal | Normal | (s,Q) |
| AY | 412 | 96.0% | 97.5% | Normal | Empirical | (s,Q) |
| ... | ... | ... | ... | ... | ... | ... |

---

## 33. What-If Scenario Panel

### Layout

Expandable panel (reuses the ClustersTab What-If panel pattern from Feature 29):

```
┌─────────────────────────────────────────────────────┐
│ ▼ What-If Scenarios                         [Run ▶] │
├─────────────────────────────────────────────────────┤
│ Scenario Type: [Service Level ▼]                     │
│                                                     │
│ ┌─── Parameters ──────────────────────────────────┐ │
│ │ Target Service Level: [====|====] 97.0%         │ │
│ │ Scope: [All Items ▼]                            │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─── Results ─────────────────────────────────────┐ │
│ │ ┌───────────┬───────────┬───────────┐           │ │
│ │ │ Baseline  │ Scenario  │ Delta     │           │ │
│ │ │ SS: $13.5M│ SS: $14.8M│ +$1.3M   │           │ │
│ │ │ SL: 95.2% │ SL: 97.0% │ +1.8 pp  │           │ │
│ │ │ Turns: 8.4│ Turns: 7.9│ -0.5     │           │ │
│ │ └───────────┴───────────┴───────────┘           │ │
│ │                                                 │ │
│ │ [Cost-of-Service Curve]  [Tornado Diagram]      │ │
│ │ [Detail Table]           [Export]                │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ [Save Scenario] [Compare to...] [Promote to Prod]   │
└─────────────────────────────────────────────────────┘
```

### Scenario Type Selector

Dropdown with all 11 types:
1. Service Level Change
2. Lead Time Change
3. Demand Variability Change
4. Supply Disruption
5. Budget Constraint
6. Network Consolidation
7. Dual Sourcing
8. ABC Reclassification
9. Seasonal Buildup
10. Promotion Impact
11. MOQ Adjustment

Each type shows different parameter inputs.

### Visualization: Cost-of-Service Efficient Frontier

ECharts scatter chart:
- X-axis: Service Level % (80–100%)
- Y-axis: Total SS Investment ($)
- Current operating point (large blue dot)
- Efficient frontier curve (smooth line through Pareto-optimal points)
- Scenario point (large orange dot)
- Infeasible region shading

### Visualization: Tornado Sensitivity Diagram

Horizontal bar chart showing sensitivity of total cost to each parameter:
- Lead Time (+1 day / -1 day) → SS impact
- Service Level (+1% / -1%) → SS impact
- Demand Variability (+10% / -10%) → SS impact
- Ordering Cost (+$10 / -$10) → total cost impact
- Holding Rate (+2% / -2%) → total cost impact

Bars extend left (decrease) and right (increase) from center line.

---

## 34. Inventory Projection Chart

### 12-Month Forward Projection

Stacked area chart:

```
Qty
│  ┌──────────────────── Order-Up-To Level
│  │   ╱╲    ╱╲    ╱╲
│  │  ╱  ╲  ╱  ╲  ╱  ╲    On-Hand + In-Transit (stacked area)
│  │ ╱    ╲╱    ╲╱    ╲
│  ├─────────────────────── Reorder Point
│  │
│  ├─────────────────────── Safety Stock Level
│  │
│  └──────────────────────── Zero
└──────────────────────────────────────────── Time
     Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec  Jan  Feb
```

Layers:
- Light blue area: on-hand inventory
- Blue area (stacked above): in-transit inventory
- Dotted blue area (stacked above): planned orders
- Red dashed line: safety stock threshold
- Orange dashed line: reorder point
- Green triangles: expected receipt dates
- Red line: demand forecast
- Shaded red region: forecast confidence interval

### Gap Analysis

Auto-detect and highlight shortfall periods where projected inventory falls below safety stock:

```
⚠ SHORTFALL DETECTED: Projected inventory below safety stock from Jun 2026 to Aug 2026
   Shortfall peak: -340 units in Jul 2026
   Recommended action: Advance PO-12401 by 2 weeks or place emergency order of 500 units
```

---

## 35. Supplier Performance Panel

### Scorecard Grid

TanStack Table:

| Column | Description |
|--------|-------------|
| Supplier | Supplier name + ID |
| Items Supplied | Count of active items |
| Avg Lead Time | Mean actual lead time (days) |
| LT Std Dev | Lead time variability |
| OTD % | On-time delivery percentage |
| Quality % | Percent of POs without quality issues |
| Composite Grade | A/B/C/D grade (weighted score) |
| Impact on SS | Total SS attributable to this supplier's LT variability |

### Lead Time Distribution

Per-supplier histogram:
- X-axis: lead time (days)
- Y-axis: frequency (PO count)
- Vertical line: promised lead time
- Color: green (on-time), red (late)
- Distribution statistics: mean, std, p50, p95

### Impact Analysis

Interactive calculator:

```
If [Supplier ABC Corp] improves OTD from [82%] to [95%]:
  → Lead time std decreases from [5.2 days] to [2.8 days]
  → Safety stock reduction: 1,247 units across 89 items
  → Annual savings: $186,400 in holding cost
```

---

## 36. Alerts & Exceptions Panel

### Alert Cards

Priority-colored cards:

```
🔴 CRITICAL — 3 active
┌─────────────────────────────────────────────────────────┐
│ STOCKOUT RISK: Item 100320 at Loc 1401-BULK             │
│ Current on-hand: 142 units | ROP: 1,640 | No open POs  │
│ Days until stockout: ~3 days at current demand rate     │
│                                                         │
│ [Create Emergency PO] [Adjust SS] [View Detail]         │
└─────────────────────────────────────────────────────────┘

🟡 WARNING — 12 active
┌─────────────────────────────────────────────────────────┐
│ EXCESS STOCK: Item 200155 at Loc 2301-RETAIL            │
│ DOS: 187 days (target: 45) | Excess value: $23,400      │
│ No demand in last 3 months | Demand trend: declining     │
│                                                         │
│ [Plan Markdown] [Transfer to Alt Location] [Review]      │
└─────────────────────────────────────────────────────────┘
```

### Alert Types

| Alert Type | Trigger Condition | Severity | Recommended Action |
|-----------|-------------------|----------|-------------------|
| stockout_risk | DOS < lead_time_days AND no open PO | Critical | Create Emergency PO |
| stockout_active | qty_on_hand = 0 AND demand > 0 | Critical | Expedite PO / Transfer |
| excess_stock | DOS > 2 × target AND declining demand | Warning | Markdown / Transfer |
| obsolescence_risk | No demand 12+ months, on-hand > 0 | Warning | Write-off / Liquidate |
| lead_time_spike | Actual LT > mean + 2.5 × std | Warning | Adjust SS temporarily |
| demand_anomaly | Demand > mean + 3 × std | Info | Review forecast |
| budget_breach | Projected inventory cost > budget | Warning | Optimize / Defer orders |
| service_level_miss | Achieved SL < target - 2% | Warning | Increase SS |
| supplier_issue | OTD < 80% trailing 3 months | Warning | Supplier review / Alt source |
| expiry_risk | On-hand > remaining shelf life demand | Warning | Accelerate sales / Transfer |

### Alert Configuration

Planners can customize alert thresholds:

```
┌─── Alert Configuration ────────────────────────────────┐
│ Stockout Risk: DOS threshold [7] days                   │
│ Excess Stock: DOS threshold [90] days                   │
│ Lead Time Spike: z-score threshold [2.5]                │
│ Demand Anomaly: z-score threshold [3.0]                 │
│ Service Level Miss: tolerance [-2.0] pp                 │
│ Email notifications: [ON/OFF]                           │
│ [Save Configuration]                                    │
└─────────────────────────────────────────────────────────┘
```

---

# Part VI: Competitive Positioning

---

## 37. Vendor Comparison

| Capability | Feature 34 | SAP IBP | Blue Yonder | Kinaxis | o9 Solutions | ToolsGroup | Lokad |
|------------|-----------|---------|-------------|---------|-------------|------------|-------|
| **Safety Stock Optimization** | 4 methods (Normal, Empirical, Conformal, Bootstrap) | GSM-based parametric | ML-based probabilistic | Parametric + simulation | AI/ML-based | Probabilistic (native) | Quantile-based |
| **Multi-Echelon (MEIO)** | Phase 2 (Graves-Willems) | Yes (advanced) | Yes (advanced) | Limited (2 echelons) | Yes (graph-based) | Yes (native) | Yes |
| **Probabilistic Forecasting** | Conformal + empirical + DeepAR | Limited (parametric) | Yes (ML pipelines) | Limited (scenarios) | Yes (AI-native) | Yes (native probabilistic) | Yes (quantile grids) |
| **What-If Scenarios** | 11 types, sub-second single-item | 3-4 types, batch | 5-6 types | Yes (core strength, in-memory) | Yes (AI-driven) | Limited (3-4 types) | Programmatic scenarios |
| **Monte Carlo Simulation** | Yes (vectorized NumPy, 10K runs) | No (analytical only) | Limited | Yes | Yes | No | Yes (custom) |
| **ABC-XYZ Auto-Policy** | 9-cell matrix with auto-assignment | Yes | Yes | Yes | Yes | Yes | Programmable |
| **Champion Model Integration** | Native (uses Demand Studio champion) | No (separate DP module) | No (separate DP module) | No (external forecast input) | Partial (native DP) | No (external forecast) | Yes (native) |
| **Backtest-Based SS** | Native (uses backtest archive) | No | No | No | No | No | Yes (custom) |
| **Real-Time What-If** | < 1s single-item | No (batch, minutes) | No (batch, minutes) | Yes (in-memory, seconds) | Yes (graph-compute) | No (batch) | No (batch) |
| **Open Source Stack** | Yes (Python, PuLP, DEAP, scikit-learn) | No (proprietary) | No (proprietary) | No (proprietary) | No (proprietary) | No (proprietary) | Envision DSL |
| **Intermittent Demand** | Croston, TSB, Bootstrap | Basic | Croston | Basic | ML-based | Yes (native) | Quantile-based |
| **Supplier Performance** | Integrated scorecard + impact analysis | SAP Ariba integration | Yes | Yes | Yes | Limited | External |
| **Seasonal Buildup** | Native (uses Feature 30 profiles) | Yes | Yes | Yes | Yes | Yes | Programmable |
| **Deployment** | On-prem/Docker/Cloud | Cloud (BTP) | Cloud | Cloud/On-prem | Cloud | Cloud/On-prem | Cloud |
| **Licensing** | Open source | $$$$$ | $$$$ | $$$$ | $$$$ | $$$ | $$$$ |

### Key Differentiators vs. Competition

1. **Native forecast integration**: No other vendor natively uses the same system's champion model forecasts and backtest error distributions. Commercial tools require a separate demand planning module or manual forecast input.

2. **4-method safety stock**: Most competitors use 1–2 methods. Feature 34 auto-selects the optimal method per item based on ABC-XYZ classification.

3. **11 what-if scenarios**: Broadest scenario coverage of any platform, with sub-second response for single-item scenarios.

4. **Open source**: No proprietary solver licenses. PuLP + HiGHS provides LP/MIP solving equivalent to commercial solvers for inventory-scale problems.

5. **Integrated analytics**: Clustering, seasonality, backtesting, and inventory planning in a single platform — no integration overhead.

---

## 38. Industry Benchmarks

Target KPIs by industry vertical, used for default configuration and performance comparison:

| Industry | Target Fill Rate | Target Turns | Target DOS | Typical Carrying Cost | Typical SS as % of Inventory |
|----------|-----------------|--------------|------------|----------------------|------------------------------|
| Retail (General) | 95–98% | 8–12 | 30–45 days | 20–25% | 15–25% |
| CPG / FMCG | 97–99% | 10–15 | 25–35 days | 18–22% | 10–20% |
| Manufacturing (Discrete) | 93–97% | 4–8 | 45–90 days | 15–20% | 20–30% |
| Manufacturing (Process) | 95–98% | 6–10 | 35–60 days | 15–20% | 15–25% |
| Pharmaceutical | 99%+ | 3–6 | 60–120 days | 25–35% | 25–40% |
| Electronics / High-Tech | 95–98% | 6–12 | 30–60 days | 20–30% | 15–25% |
| Automotive | 97–99% | 8–15 | 25–45 days | 15–20% | 10–20% |
| Aerospace & Defense | 95–99% | 2–4 | 90–180 days | 20–30% | 30–50% |
| Wholesale / Distribution | 96–99% | 8–14 | 25–45 days | 18–25% | 12–20% |
| E-commerce | 97–99% | 12–20 | 18–30 days | 22–28% | 10–18% |

These benchmarks are used to:
1. Set default service level targets in `inventory_config.yaml`
2. Calibrate KPI traffic light thresholds
3. Provide context in the dashboard ("Your turns of 8.4 are in the 60th percentile for Retail")

---

## 39. Expected Business Impact

Based on published case studies from similar implementations (McKinsey, Gartner, APICS/ASCM):

| Impact Area | Conservative | Typical | Aggressive |
|-------------|-------------|---------|------------|
| Inventory investment reduction | 10% | 20% | 30% |
| Service level improvement | +1 pp | +3 pp | +5 pp |
| Stockout reduction | 20% | 35% | 50% |
| Excess/obsolete reduction | 15% | 25% | 40% |
| Planner productivity improvement | 20% | 40% | 60% |
| Ordering cost reduction | 10% | 20% | 30% |
| Carrying cost reduction | 10% | 20% | 30% |

**ROI calculation for a $100M inventory portfolio:**

```
Annual carrying cost (25%):           $25.0M
20% inventory reduction:              -$5.0M inventory → -$1.25M/yr carrying cost
Stockout reduction (35% fewer):       +$0.5M/yr in recovered revenue
Planner productivity (40%):           +$0.3M/yr in labor savings
Ordering cost reduction (20%):        +$0.1M/yr

Total annual benefit:                 $2.15M/yr
Implementation cost:                  $0.5M–$1.0M (internal team)
Payback period:                       3–6 months
3-year ROI:                           500–1,000%
```

---

# Part VII: Technical Architecture

---

## 40. File Structure

### Backend

```
mvp/demand/
├── api/
│   └── routers/
│       └── inventory.py                        # 30+ REST API endpoints
│           # Dashboard endpoints
│           # Position endpoints
│           # Safety stock endpoints
│           # Replenishment endpoints
│           # Classification endpoints
│           # Scenario endpoints
│           # Supplier endpoints
│           # Alert endpoints
│
├── common/
│   ├── inventory_engine.py                     # Core computation engine
│   │   # compute_safety_stock_normal()
│   │   # compute_safety_stock_empirical()
│   │   # compute_safety_stock_conformal()
│   │   # compute_safety_stock_bootstrap()
│   │   # compute_eoq()
│   │   # compute_reorder_point()
│   │   # compute_fill_rate()
│   │   # solve_z_for_fill_rate()
│   │   # compute_days_of_supply()
│   │   # compute_inventory_turns()
│   │   # compute_forward_coverage()
│   │   # compute_carrying_cost()
│   │   # compute_health_score()
│   │   # normal_loss_function()
│   │
│   ├── inventory_optimizer.py                  # Optimization solvers
│   │   # optimize_ss_lagrangian()
│   │   # optimize_ss_gradient()
│   │   # optimize_ss_lp()
│   │   # optimize_ss_mip()
│   │   # optimize_ss_nsga2()
│   │   # select_solver()
│   │   # simulate_inventory_policy()
│   │
│   ├── inventory_classifier.py                 # ABC-XYZ classification
│   │   # classify_abc()
│   │   # classify_xyz()
│   │   # classify_abc_xyz()
│   │   # classify_demand_pattern()             # Syntetos-Boylan
│   │   # assign_policies_from_matrix()
│   │   # compute_abc_pareto()
│   │
│   └── inventory_alerts.py                     # Alert generation
│       # detect_stockout_risk()
│       # detect_excess_stock()
│       # detect_lead_time_spikes()
│       # detect_demand_anomalies()
│       # generate_alerts()
│       # process_alert_action()
│
├── scripts/
│   ├── compute_inventory_params.py             # Batch safety stock computation CLI
│   │   # --mode all|incremental
│   │   # --scope all|category|abc_class
│   │   # Writes to fact_inventory_plan
│   │
│   ├── run_inventory_scenario.py               # What-if scenario runner
│   │   # run_scenario()
│   │   # promote_scenario()
│   │   # get_scenario_result()
│   │
│   ├── classify_inventory.py                   # ABC-XYZ classification CLI
│   │   # Classifies all DFUs
│   │   # Writes to dim_inventory_policy
│   │
│   ├── generate_inventory_alerts.py            # Alert generation batch job
│   │   # Scans inventory position
│   │   # Generates alerts in fact_inventory_alert
│   │
│   └── load_inventory_snapshots.py             # Inventory snapshot ingestion
│       # CSV/Parquet → fact_inventory_snapshot
│       # Follows normalize + load pattern
│
├── config/
│   └── inventory_config.yaml                   # All configuration parameters
│
├── sql/
│   ├── 017_create_inventory_tables.sql         # DDL for 10 new tables
│   ├── 018_create_inventory_views.sql          # Materialized views + indexes
│   └── 019_inventory_seed_data.sql             # Default config rows
│
└── tests/
    ├── api/
    │   └── test_inventory.py                   # API endpoint tests (30+ tests)
    │
    └── unit/
        ├── test_inventory_engine.py            # SS formula tests (40+ tests)
        ├── test_inventory_optimizer.py          # Optimizer tests (20+ tests)
        ├── test_inventory_classifier.py         # ABC-XYZ tests (15+ tests)
        └── test_inventory_alerts.py             # Alert logic tests (15+ tests)
```

### Frontend

```
mvp/demand/frontend/src/
├── tabs/
│   ├── InventoryTab.tsx                        # Main tab with sub-navigation router
│   │   # Sub-tab state management
│   │   # Global filter context provider
│   │   # Lazy loading of sub-panels
│   │
│   ├── inventory/                              # Sub-tab components
│   │   ├── OverviewPanel.tsx                   # Dashboard KPIs + charts
│   │   │   # 9 KPI cards with traffic lights
│   │   │   # DOS histogram
│   │   │   # Service level vs target bullet chart
│   │   │   # Inventory turns trend
│   │   │   # Fill rate trend
│   │   │   # Carrying cost trend
│   │   │   # Stock health donut chart
│   │   │
│   │   ├── PositionPanel.tsx                   # Inventory position grid
│   │   │   # Virtualized TanStack Table
│   │   │   # Status badges, sparklines
│   │   │   # Row click → Item Detail drawer
│   │   │
│   │   ├── ItemDetailDrawer.tsx                # Slide-over DFU detail
│   │   │   # Header metadata
│   │   │   # 6 KPI mini-cards
│   │   │   # Demand chart (ECharts)
│   │   │   # Inventory projection chart
│   │   │   # Supply timeline (Gantt)
│   │   │   # Policy panel (current vs recommended)
│   │   │
│   │   ├── ReplenishmentPanel.tsx              # Replenishment recommendations
│   │   │   # Priority cards
│   │   │   # Recommendation grid
│   │   │   # Batch PO creation flow
│   │   │   # Calendar view toggle
│   │   │
│   │   ├── SafetyStockPanel.tsx                # SS optimizer
│   │   │   # Service level slider
│   │   │   # Method selector
│   │   │   # Summary cards
│   │   │   # Recommendation grid
│   │   │   # Manual override modal
│   │   │
│   │   ├── ClassificationPanel.tsx             # ABC-XYZ matrix
│   │   │   # 3x3 heat map
│   │   │   # Pareto chart
│   │   │   # Interactive threshold sliders
│   │   │   # Policy assignment table
│   │   │
│   │   ├── ScenarioPanel.tsx                   # What-if scenarios
│   │   │   # Scenario type selector
│   │   │   # Parameter inputs (per type)
│   │   │   # Results comparison table
│   │   │   # Cost-of-service frontier chart
│   │   │   # Tornado sensitivity diagram
│   │   │   # Save/load/promote actions
│   │   │
│   │   ├── SupplierPanel.tsx                   # Supplier performance
│   │   │   # Scorecard grid
│   │   │   # Lead time distribution histogram
│   │   │   # Impact analysis calculator
│   │   │
│   │   └── AlertsPanel.tsx                     # Alerts & exceptions
│   │       # Priority-colored cards
│   │       # One-click actions
│   │       # Alert configuration modal
│   │
│   └── __tests__/
│       ├── InventoryTab.test.tsx                # Main tab smoke test
│       ├── OverviewPanel.test.tsx               # Dashboard tests
│       ├── PositionPanel.test.tsx               # Grid rendering tests
│       ├── SafetyStockPanel.test.tsx            # SS optimizer tests
│       ├── ClassificationPanel.test.tsx         # ABC-XYZ matrix tests
│       └── ScenarioPanel.test.tsx               # What-if scenario tests
│
├── api/
│   └── queries.ts                              # Extended with inventory query keys + fetch functions
│       # fetchInventoryDashboard()
│       # fetchInventoryPosition()
│       # fetchItemDetail()
│       # fetchSafetyStockRecommendations()
│       # fetchReplenishmentSuggestions()
│       # fetchClassificationMatrix()
│       # fetchSupplierScorecard()
│       # fetchInventoryAlerts()
│       # runInventoryScenario()
│       # promoteScenario()
│
└── types/
    └── index.ts                                # Extended with inventory types
        # InventoryPosition
        # InventoryPlan
        # SafetyStockRecommendation
        # ReplenishmentSuggestion
        # ABCXYZClassification
        # InventoryScenario
        # SupplierScorecard
        # InventoryAlert
        # InventoryKPIs
```

---

## 41. API Endpoints

### Dashboard Endpoints

```
GET  /inventory/dashboard
  Query params: category, location, abc_class, date_from, date_to
  Response: { kpis: InventoryKPIs, charts: DashboardCharts }

GET  /inventory/dashboard/dos-distribution
  Query params: category, location, abc_class
  Response: { buckets: [{ range: "0-15", count: 234 }, ...] }

GET  /inventory/dashboard/stock-health
  Response: { healthy: 6234, watch: 2100, critical: 890, stockout: 156, dead: 620 }
```

### Position Endpoints

```
GET  /inventory/position
  Query params: category, location, abc_class, xyz_class, status,
                search, sort_by, sort_dir, offset, limit
  Response: { rows: InventoryPosition[], total: number, approx: boolean }

GET  /inventory/position/{dfu_ck}
  Response: InventoryPositionDetail (full item detail with history)

GET  /inventory/position/{dfu_ck}/projection
  Query params: horizon_months (default 12)
  Response: { dates: [], on_hand: [], in_transit: [], planned: [],
              ss_line: [], rop_line: [], forecast: [], receipts: [] }

GET  /inventory/position/{dfu_ck}/demand-history
  Query params: months (default 24)
  Response: { months: [], qty_shipped: [], forecast: [], confidence_upper: [], confidence_lower: [] }
```

### Safety Stock Endpoints

```
GET  /inventory/safety-stock/recommendations
  Query params: scope, scope_filter, method, target_sl, sort_by, offset, limit
  Response: { recommendations: SSRecommendation[], summary: SSSummary }

GET  /inventory/safety-stock/recommendations/{dfu_ck}
  Response: SSRecommendationDetail (with full calculation breakdown)

POST /inventory/safety-stock/optimize
  Body: { target_service_level: float, scope: str, scope_filter: str,
          method: str, budget_ceiling: float? }
  Response: { job_id: str, status: "running" }

GET  /inventory/safety-stock/optimize/{job_id}
  Response: OptimizationResult

POST /inventory/safety-stock/apply
  Body: { dfu_cks: string[], plan_version: int }
  Response: { applied: int, failed: int, errors: [] }

POST /inventory/safety-stock/override
  Body: { dfu_ck: str, safety_stock: float, reason_code: str, notes: str }
  Response: { success: bool }
```

### Replenishment Endpoints

```
GET  /inventory/replenishment/suggestions
  Query params: priority, supplier_id, sort_by, offset, limit
  Response: { suggestions: ReplenishmentSuggestion[], summary: { critical: int, urgent: int, ... } }

POST /inventory/replenishment/create-po
  Body: { items: [{ item_no, loc, qty, supplier_id }] }
  Response: { po_numbers: string[], total_value: float }

GET  /inventory/replenishment/calendar
  Query params: date_from, date_to
  Response: { events: [{ date, item_no, qty, type: "receipt"|"order" }] }
```

### Classification Endpoints

```
GET  /inventory/classification/matrix
  Query params: a_threshold, b_threshold, x_cv_threshold, y_cv_threshold
  Response: { matrix: ABCXYZMatrix, pareto: ParetoData, totals: ClassTotals }

POST /inventory/classification/reclassify
  Body: { a_threshold: float, b_threshold: float, x_cv: float, y_cv: float }
  Response: { items_reclassified: int, migration_matrix: int[][] }

GET  /inventory/classification/policy-map
  Response: { cells: [{ abc: "A", xyz: "X", policy: PolicyConfig, item_count: int }] }
```

### Scenario Endpoints

```
POST /inventory/scenario
  Body: ScenarioRequest (type, params, name)
  Response: { scenario_id: str, status: "running" }

GET  /inventory/scenario/{id}
  Response: ScenarioResult

GET  /inventory/scenario/{id}/details
  Response: ScenarioDetailResult (per-item breakdown)

POST /inventory/scenario/{id}/promote
  Response: { promoted: bool, items_updated: int }

GET  /inventory/scenarios
  Query params: type, status, limit
  Response: { scenarios: ScenarioSummary[] }

GET  /inventory/scenario/compare
  Query params: baseline_id, scenario_id
  Response: ComparisonResult

DELETE /inventory/scenario/{id}
  Response: { deleted: bool }
```

### Supplier Endpoints

```
GET  /inventory/suppliers
  Query params: sort_by, sort_dir, offset, limit
  Response: { suppliers: SupplierScorecard[] }

GET  /inventory/suppliers/{supplier_id}
  Response: SupplierDetail

GET  /inventory/suppliers/{supplier_id}/lead-times
  Response: { histogram: [{ days: int, count: int }], stats: LTStats }

GET  /inventory/suppliers/{supplier_id}/items
  Response: { items: SupplierItem[] }

GET  /inventory/suppliers/{supplier_id}/impact
  Query params: target_otd
  Response: { ss_reduction: float, cost_savings: float, items_affected: int }
```

### Alert Endpoints

```
GET  /inventory/alerts
  Query params: severity, alert_type, status, dfu_ck, offset, limit
  Response: { alerts: InventoryAlert[], summary: { critical: int, warning: int, info: int } }

GET  /inventory/alerts/{alert_id}
  Response: InventoryAlertDetail

POST /inventory/alerts/{alert_id}/action
  Body: { action: "acknowledge"|"resolve"|"dismiss", notes: str }
  Response: { success: bool }

GET  /inventory/alerts/config
  Response: AlertConfig

PUT  /inventory/alerts/config
  Body: AlertConfig
  Response: { success: bool }

GET  /inventory/alerts/summary
  Response: { by_type: {}, by_severity: {}, trend_7d: [] }
```

---

## 42. Python Dependencies (New)

Added to `pyproject.toml`:

```toml
[project.dependencies]
# Existing dependencies...

# Inventory Planning Module (new)
pulp = ">=2.8"              # LP/MIP modeling with HiGHS solver (open-source, no license)
cvxpy = ">=1.4"             # Convex optimization modeling (optional, for advanced formulations)
deap = ">=1.4"              # NSGA-II multi-objective evolutionary optimization
optuna = ">=3.5"            # Bayesian hyperparameter optimization for policy tuning
mapie = ">=0.9"             # Conformal prediction intervals (model-agnostic)

# Optional (Phase 4)
# simpy = ">=4.1"           # Discrete event simulation for complex supply chain modeling
```

**Dependency rationale:**

| Package | Why Needed | Size | License |
|---------|-----------|------|---------|
| PuLP | LP/MIP solver interface; HiGHS included | 15 MB | BSD |
| cvxpy | Convex optimization for advanced formulations | 30 MB | Apache 2.0 |
| DEAP | NSGA-II Pareto optimization | 5 MB | LGPL |
| Optuna | Bayesian policy optimization | 20 MB | MIT |
| MAPIE | Conformal prediction intervals | 10 MB | BSD |

Total new dependency footprint: ~80 MB. All open-source, no proprietary licenses.

---

## 43. Config File (inventory_config.yaml)

```yaml
# Inventory Planning Module Configuration
# Path: mvp/demand/config/inventory_config.yaml

inventory:
  # ──────────────── Service Level Targets ────────────────
  service_level_targets:
    A: 0.98        # Type 2 fill rate for A-class items
    B: 0.95        # Type 2 fill rate for B-class items
    C: 0.85        # Type 1 cycle service level for C-class items
    default: 0.95  # Default if not classified

  service_level_type:
    A: "fill_rate"
    B: "fill_rate"
    C: "csl"

  # ──────────────── ABC Classification ────────────────
  abc_thresholds:
    a_pct: 0.80    # Top 80% of revenue = A class
    b_pct: 0.95    # Next 15% (80-95%) = B class
                   # Bottom 5% (95-100%) = C class

  # ──────────────── XYZ Classification ────────────────
  xyz_thresholds:
    x_cv: 0.50     # CV < 0.50 = X class (stable)
    y_cv: 1.00     # 0.50 <= CV < 1.00 = Y class (variable)
                   # CV >= 1.00 = Z class (unpredictable)

  # ──────────────── Cost Defaults ────────────────
  holding_cost_pct_default: 0.20       # 20% of item value per year
  ordering_cost_default: 50.00         # $50 per order
  shortage_cost_multiplier: 2.0        # shortage_cost = multiplier × unit_cost
  capital_cost_rate: 0.10              # 10% cost of capital
  storage_cost_rate: 0.03              # 3% warehouse cost
  insurance_rate: 0.01                 # 1% insurance
  obsolescence_rate_default: 0.05      # 5% obsolescence

  # ──────────────── Review Periods ────────────────
  review_periods:
    A: 1           # Daily review for A items
    B: 7           # Weekly review for B items
    C: 30          # Monthly review for C items

  # ──────────────── Safety Stock Methods ────────────────
  ss_method_defaults:
    AX: "normal"
    AY: "empirical"
    AZ: "bootstrap"
    BX: "normal"
    BY: "empirical"
    BZ: "bootstrap"
    CX: "normal"
    CY: "normal"
    CZ: "croston"

  # ──────────────── Policy Defaults ────────────────
  policy_defaults:
    AX: "rQ"       # Continuous review, fixed quantity
    AY: "rQ"
    AZ: "sS"       # Continuous review, order-up-to
    BX: "RS"       # Periodic review, order-up-to
    BY: "RS"
    BZ: "RsS"      # Periodic review, can-order
    CX: "RS"
    CY: "RS"
    CZ: "RsS"

  # ──────────────── Intermittent Demand ────────────────
  intermittent:
    adi_threshold: 1.32      # Syntetos-Boylan ADI boundary
    cv2_threshold: 0.49      # Syntetos-Boylan CV^2 boundary
    min_observations: 12     # Minimum months of data for classification
    croston_alpha: 0.15      # Smoothing parameter for Croston's method

  # ──────────────── Simulation ────────────────
  simulation:
    n_replications: 10000    # Monte Carlo replications
    confidence_level: 0.95   # Confidence interval level
    horizon_days: 365        # Simulation horizon
    batch_size: 1000         # Items per batch (memory management)
    seed: 42                 # Random seed for reproducibility

  # ──────────────── Optimization Solver ────────────────
  solver:
    default: "highs"         # LP/MIP solver (PuLP backend)
    time_limit_seconds: 300  # Max solver runtime
    gap_tolerance: 0.01      # MIP optimality gap tolerance (1%)
    nsga2:
      population_size: 100
      n_generations: 200
      crossover_prob: 0.7
      mutation_prob: 0.3
    lagrangian:
      max_iterations: 100
      convergence_epsilon: 0.01

  # ──────────────── Alert Thresholds ────────────────
  alerts:
    stockout_risk_dos_days: 7          # Alert if DOS < this many days
    excess_stock_dos_multiplier: 2.0   # Alert if DOS > target × multiplier
    lead_time_spike_z_score: 2.5       # Alert if LT z-score > threshold
    demand_anomaly_z_score: 3.0        # Alert if demand z-score > threshold
    service_level_miss_tolerance: 0.02 # Alert if SL < target - tolerance
    obsolescence_months: 12            # Alert if no demand for this many months

  # ──────────────── Projection ────────────────
  projection:
    horizon_months: 12       # Forward projection horizon
    receipt_lead_time_buffer: 3  # Days buffer on receipt estimates

  # ──────────────── Seasonal Buildup ────────────────
  seasonal:
    buildup_start_months_before_peak: 2  # Start building 2 months before peak
    peak_service_level_uplift: 0.02      # +2pp service level during peak
    buildup_rate: "linear"               # linear | front_loaded | back_loaded
```

---

# Part VIII: Implementation Roadmap

---

## 44. Phased Implementation

### Phase 1: Foundation (Weeks 1–4) — Highest ROI

**Goal:** Core data model, safety stock engine, and basic UI with dashboard and position view.

**Backend:**
- DDL: Create 10 new tables (`sql/017_create_inventory_tables.sql`)
- DDL: Create materialized views (`sql/018_create_inventory_views.sql`)
- DDL: Seed default config rows (`sql/019_inventory_seed_data.sql`)
- `common/inventory_engine.py`: All 4 SS methods, EOQ, ROP, fill rate, loss function, DOS, turns, health score
- `common/inventory_classifier.py`: ABC-XYZ classification, Syntetos-Boylan demand pattern classification
- `scripts/load_inventory_snapshots.py`: CSV → `fact_inventory_snapshot` ingestion
- `scripts/compute_inventory_params.py`: Batch SS/ROP/EOQ computation → `fact_inventory_plan`
- `scripts/classify_inventory.py`: ABC-XYZ classification → `dim_inventory_policy`
- `config/inventory_config.yaml`: Full configuration file
- API router: `api/routers/inventory.py` with:
  - `GET /inventory/dashboard` (KPIs + chart data)
  - `GET /inventory/dashboard/dos-distribution`
  - `GET /inventory/dashboard/stock-health`
  - `GET /inventory/position` (paginated grid)
  - `GET /inventory/position/{dfu_ck}` (single item detail)
  - `GET /inventory/position/{dfu_ck}/projection` (12-month projection)
  - `GET /inventory/safety-stock/recommendations` (all items)
  - `GET /inventory/classification/matrix` (ABC-XYZ matrix)

**Frontend:**
- `InventoryTab.tsx`: Main tab with sub-navigation
- `OverviewPanel.tsx`: 9 KPI cards + 6 dashboard charts
- `PositionPanel.tsx`: Virtualized grid with status badges

**Integration:**
- Champion forecasts from `fact_external_forecast_monthly` → demand input for SS
- Backtest errors from `backtest_lag_archive` → empirical SS method
- Cluster assignments from `dim_dfu` → policy segmentation
- Seasonality profiles from `dim_dfu` → seasonal awareness

**Tests:**
- `tests/unit/test_inventory_engine.py`: 40+ unit tests for all formulas
- `tests/unit/test_inventory_classifier.py`: 15+ tests for ABC-XYZ
- `tests/api/test_inventory.py`: API tests for all Phase 1 endpoints
- `frontend/src/tabs/__tests__/InventoryTab.test.tsx`: Smoke test

**Deliverable:** Planners can view inventory position, dashboard KPIs, and safety stock recommendations. ABC-XYZ classification is operational. Champion model forecasts drive safety stock computation.

### Phase 2: Optimization & Classification (Weeks 5–8)

**Goal:** Advanced optimization, full classification UI, item detail drawer, and safety stock optimizer panel.

**Backend:**
- `common/inventory_optimizer.py`: Gradient-based optimizer, LP budget allocation (PuLP + HiGHS)
- API endpoints:
  - `POST /inventory/safety-stock/optimize` (run optimization)
  - `POST /inventory/safety-stock/apply` (apply recommendations)
  - `POST /inventory/safety-stock/override` (manual override)
  - `POST /inventory/classification/reclassify` (update thresholds)
  - `GET /inventory/classification/policy-map`
- What-if scenarios (first 3 types):
  - Service Level What-If
  - Lead Time What-If
  - Demand Variability What-If
  - `POST /inventory/scenario`
  - `GET /inventory/scenario/{id}`

**Frontend:**
- `SafetyStockPanel.tsx`: Service level slider, method selector, recommendation grid, manual override modal
- `ClassificationPanel.tsx`: 3x3 heat map, Pareto chart, threshold sliders
- `ItemDetailDrawer.tsx`: Slide-over with KPIs, demand chart, projection chart, policy panel
- `ScenarioPanel.tsx`: Initial version with 3 scenario types

**Tests:**
- `tests/unit/test_inventory_optimizer.py`: 20+ optimizer tests
- API tests for all Phase 2 endpoints
- Frontend component tests for new panels

**Deliverable:** Planners can optimize safety stock with constraints, adjust ABC-XYZ thresholds, drill into item details, and run basic what-if scenarios.

### Phase 3: Replenishment & Suppliers (Weeks 9–12)

**Goal:** Replenishment recommendations, supplier performance tracking, and alert management.

**Backend:**
- Supplier data ingestion scripts
- PO data ingestion scripts
- API endpoints:
  - `GET /inventory/replenishment/suggestions`
  - `POST /inventory/replenishment/create-po`
  - `GET /inventory/suppliers` (scorecards)
  - `GET /inventory/suppliers/{id}/lead-times`
  - `GET /inventory/suppliers/{id}/impact`
  - `GET /inventory/alerts`
  - `POST /inventory/alerts/{id}/action`
  - `GET /inventory/alerts/config`
  - `PUT /inventory/alerts/config`
- `common/inventory_alerts.py`: Alert detection logic
- `scripts/generate_inventory_alerts.py`: Batch alert generation

**Frontend:**
- `ReplenishmentPanel.tsx`: Priority cards, recommendation grid, batch PO creation
- `SupplierPanel.tsx`: Scorecard grid, LT distribution histogram, impact calculator
- `AlertsPanel.tsx`: Priority-colored cards, one-click actions, configuration modal

**Tests:**
- Alert generation unit tests
- Replenishment logic unit tests
- API tests for all Phase 3 endpoints
- Frontend component tests

**Deliverable:** Full operational workflow: monitor alerts, review replenishment suggestions, create POs, track supplier performance.

### Phase 4: Advanced Optimization (Weeks 13–16)

**Goal:** Monte Carlo simulation, all 11 what-if scenarios, multi-objective optimization, seasonal buildup.

**Backend:**
- Monte Carlo simulation engine (vectorized NumPy)
- NSGA-II multi-objective optimization (DEAP)
- Bayesian policy optimization (Optuna)
- All remaining what-if scenario types:
  - Supply Disruption What-If
  - Budget Constraint What-If
  - Network/Sourcing What-If
  - ABC Reclassification What-If
  - Seasonal Buildup What-If
  - Promotion Impact What-If
  - MOQ/Order Multiple What-If
- `POST /inventory/scenario/{id}/promote`
- `GET /inventory/scenario/compare`

**Frontend:**
- Cost-of-service efficient frontier chart
- Tornado sensitivity diagram
- Seasonal buildup planning calendar
- Promotion impact visualization
- Scenario save/load/compare/promote workflow

**Tests:**
- Monte Carlo simulation tests (statistical convergence)
- NSGA-II Pareto optimality tests
- All scenario type API tests
- End-to-end integration tests

**Deliverable:** World-class inventory optimization with all 11 what-if scenarios, Monte Carlo simulation, and Pareto optimization. Feature-complete module.

---

## 45. Makefile Targets

```makefile
# ──────────────── Inventory Schema ────────────────
inventory-schema:                ## Apply inventory DDL (10 tables + indexes)
	$(PSQL) -f sql/017_create_inventory_tables.sql
	$(PSQL) -f sql/018_create_inventory_views.sql
	$(PSQL) -f sql/019_inventory_seed_data.sql

inventory-views:                 ## Refresh inventory materialized views
	$(PSQL) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_current;"
	$(PSQL) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_kpis;"
	$(PSQL) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_supplier_scorecard;"

# ──────────────── Inventory Data Ingestion ────────────────
inventory-load-snapshots:        ## Load inventory snapshots from CSV
	$(UV_RUN) python scripts/load_inventory_snapshots.py

inventory-load-pos:              ## Load purchase orders from CSV
	$(UV_RUN) python scripts/load_purchase_orders.py

inventory-load-suppliers:        ## Load supplier master + item map from CSV
	$(UV_RUN) python scripts/load_suppliers.py

# ──────────────── Inventory Computation ────────────────
inventory-classify:              ## Run ABC-XYZ classification
	$(UV_RUN) python scripts/classify_inventory.py

inventory-optimize:              ## Run batch safety stock optimization
	$(UV_RUN) python scripts/compute_inventory_params.py --mode all

inventory-optimize-incremental:  ## Incremental SS optimization (changed items only)
	$(UV_RUN) python scripts/compute_inventory_params.py --mode incremental

inventory-simulate:              ## Run Monte Carlo simulation (all items)
	$(UV_RUN) python scripts/compute_inventory_params.py --mode simulate --replications 10000

inventory-alerts:                ## Generate inventory alerts
	$(UV_RUN) python scripts/generate_inventory_alerts.py

# ──────────────── Inventory Pipeline ────────────────
inventory-all: inventory-load-snapshots inventory-classify inventory-optimize inventory-alerts inventory-views  ## Full inventory pipeline

# ──────────────── Inventory Testing ────────────────
test-inventory:                  ## Run all inventory tests (backend)
	$(UV_RUN) pytest tests/unit/test_inventory_engine.py tests/unit/test_inventory_optimizer.py tests/unit/test_inventory_classifier.py tests/unit/test_inventory_alerts.py tests/api/test_inventory.py -v

test-inventory-unit:             ## Run inventory unit tests only
	$(UV_RUN) pytest tests/unit/test_inventory_engine.py tests/unit/test_inventory_optimizer.py tests/unit/test_inventory_classifier.py tests/unit/test_inventory_alerts.py -v

test-inventory-api:              ## Run inventory API tests only
	$(UV_RUN) pytest tests/api/test_inventory.py -v
```

---

## 46. Testing Strategy

### Backend Unit Tests

**`tests/unit/test_inventory_engine.py`** (~40 tests):

```python
# Safety Stock Formulas
test_ss_normal_demand_only                   # Basic z × sigma × sqrt(L)
test_ss_normal_demand_and_lt_uncertainty     # Combined uncertainty formula
test_ss_normal_high_service_level            # z(0.99) = 2.326
test_ss_normal_zero_lt_std                   # Degenerates to demand-only formula
test_ss_empirical_basic                      # Quantile of forecast errors
test_ss_empirical_insufficient_data          # Raises error if < 6 observations
test_ss_empirical_biased_errors              # Captures systematic bias
test_ss_bootstrap_basic                      # Monte Carlo DDLT simulation
test_ss_bootstrap_intermittent               # Works with many zeros
test_ss_bootstrap_reproducible               # Same seed → same result

# EOQ
test_eoq_basic                              # Q* = sqrt(2DK/h)
test_eoq_with_moq                           # Max(EOQ, MOQ)
test_eoq_with_order_multiple                # Round up to multiple
test_eoq_zero_demand                        # Returns MOQ

# Fill Rate / Service Level
test_fill_rate_calculation                  # 1 - sigma*L(z)/Q
test_fill_rate_high_q                       # Large Q → high fill rate
test_normal_loss_function                   # L(z) = phi(z) - z*(1-Phi(z))
test_solve_z_for_fill_rate                  # Newton's method convergence
test_solve_z_for_fill_rate_edge_cases       # beta=0.999, beta=0.5

# KPI Calculations
test_days_of_supply                         # OH / daily demand
test_inventory_turns                        # COGS / avg inventory
test_forward_coverage                       # (OH + IT + OO) / weekly forecast
test_carrying_cost                          # Avg inventory × carrying rate
test_health_score_green                     # Score >= 80
test_health_score_red                       # Score < 60
test_health_score_weights                   # Weights sum to 1
```

**`tests/unit/test_inventory_optimizer.py`** (~20 tests):

```python
test_lagrangian_basic                       # Budget-constrained optimization
test_lagrangian_tight_budget                # Budget forces low SS
test_lagrangian_loose_budget                # Budget non-binding
test_gradient_optimizer_convergence         # Total cost decreases
test_lp_budget_allocation                   # PuLP + HiGHS solution
test_lp_min_service_level_constraint        # Per-class min SL respected
test_mip_with_moq                           # Integer constraints
test_nsga2_pareto_dominance                 # No solution dominates another
test_solver_selection_single_item           # Returns "analytical"
test_solver_selection_large_portfolio       # Returns "lagrangian"
test_monte_carlo_convergence                # Fill rate CI width < 1%
test_monte_carlo_reproducibility            # Same seed → same result
```

**`tests/unit/test_inventory_classifier.py`** (~15 tests):

```python
test_abc_classification_basic               # Pareto principle
test_abc_threshold_80_95                    # Default thresholds
test_abc_custom_thresholds                  # Non-default thresholds
test_xyz_classification_basic               # CV-based classification
test_abc_xyz_combined                       # 9-cell matrix assignment
test_syntetos_boylan_smooth                 # ADI < 1.32, CV^2 < 0.49
test_syntetos_boylan_intermittent           # ADI >= 1.32, CV^2 < 0.49
test_syntetos_boylan_erratic                # ADI < 1.32, CV^2 >= 0.49
test_syntetos_boylan_lumpy                  # ADI >= 1.32, CV^2 >= 0.49
test_policy_assignment_ax                   # AX → continuous, normal, 98%
test_policy_assignment_cz                   # CZ → periodic, Croston, 80-85%
```

### Backend API Tests

**`tests/api/test_inventory.py`** (~30 tests):

Using `httpx.AsyncClient(transport=ASGITransport(app))` with mocked DB pool:

```python
# Dashboard
test_dashboard_kpis                         # Returns all 9 KPIs
test_dashboard_with_category_filter         # Category filter applied
test_dos_distribution_histogram             # Returns histogram buckets

# Position
test_position_list_paginated                # Offset/limit pagination
test_position_list_with_filters             # ABC, status, location filters
test_position_detail                        # Single DFU detail
test_position_projection                    # 12-month projection array

# Safety Stock
test_ss_recommendations_list                # Returns recommendations
test_ss_optimize_async                      # Returns job_id, status=running
test_ss_apply_batch                         # Updates plan + policy
test_ss_manual_override                     # Sets is_manual_override=true

# Classification
test_classification_matrix                  # 3x3 matrix with counts
test_reclassify                            # Updates thresholds, returns migration

# Scenarios
test_create_scenario                        # Returns scenario_id
test_get_scenario_result                    # Returns completed result
test_compare_scenarios                      # Baseline vs scenario comparison
test_promote_scenario                       # Updates policy + plan
test_scenario_concurrency_guard             # 409 if scenario already running

# Suppliers
test_supplier_scorecard                     # Returns supplier list with metrics
test_supplier_lead_time_distribution        # Returns histogram

# Alerts
test_alerts_list                            # Returns active alerts
test_alert_action_acknowledge               # Status → acknowledged
test_alert_config_get                       # Returns thresholds
test_alert_config_update                    # Updates thresholds
```

### Frontend Tests

**`frontend/src/tabs/__tests__/InventoryTab.test.tsx`**:

```typescript
test('renders Inventory tab with sub-navigation')
test('switches between sub-tabs')
test('applies global filters')
test('shows loading state while data fetches')
```

**Additional frontend test files** (6 total for sub-panels):

Each sub-panel has a smoke test file verifying:
- Component renders without errors
- Loading state shown during data fetch
- Data displayed after fetch resolves
- User interactions trigger expected callbacks

### Test Execution

```bash
# All inventory tests (backend + frontend)
make test-inventory && cd frontend && npx vitest run src/tabs/__tests__/InventoryTab.test.tsx

# Full test suite (all 350+ tests)
make test-all
```

Expected test counts after Phase 4:
- Backend: 189 existing + 105 inventory = 294 tests
- Frontend: 108 existing + 24 inventory = 132 tests
- Total: 426 tests

---

# Part IX: Appendices

---

## Appendix A: Mathematical Reference

Quick reference of all formulas used in the Inventory Planning Module.

### Safety Stock Formulas

| Formula | Expression | Use Case |
|---------|-----------|----------|
| SS (demand uncertainty only) | `SS = z × sigma_D × sqrt(L)` | Constant lead time, known demand variability |
| SS (demand + LT uncertainty) | `SS = z × sqrt(L_bar × sigma_D^2 + d_bar^2 × sigma_LT^2)` | Variable lead time |
| SS (forecast-error-based) | `SS = z × sigma_FE × sqrt(L)` | When using champion model forecast |
| SS (empirical) | `SS = quantile(errors, 1-alpha) × sqrt(L)` | From backtest error distribution |
| SS (bootstrap) | `SS = percentile(DDLT_samples, 1-alpha) - mean(DDLT_samples)` | Intermittent/lumpy demand |
| SS (periodic review) | `SS = z × sigma_D × sqrt(R + L)` | Periodic review policies |

### Order Quantity Formulas

| Formula | Expression | Use Case |
|---------|-----------|----------|
| EOQ | `Q* = sqrt(2 × D × K / h)` | Basic economic order quantity |
| EOQ with MOQ | `Q = max(Q*, MOQ)` | Supplier minimum order |
| EOQ with multiple | `Q = ceil(Q* / M) × M` | Order multiple constraint |
| EOQ with backorders | `Q* = sqrt(2DK/h) × sqrt((h+b)/b)` | Planned backorders allowed |

### Reorder Point and Order-Up-To

| Formula | Expression | Use Case |
|---------|-----------|----------|
| Reorder Point | `ROP = d_bar × L + SS` | Continuous review |
| Order-Up-To (periodic) | `S = d_bar × (R + L) + SS` | Periodic review |
| Order-Up-To (s,S) | `S = s + EOQ` | Continuous (s,S) policy |

### Service Level Formulas

| Formula | Expression | Use Case |
|---------|-----------|----------|
| Cycle Service Level | `CSL = Phi(z) = Phi((ROP - d_bar × L) / sigma_DDLT)` | Type 1 |
| Fill Rate | `beta = 1 - (sigma_DDLT × L(z)) / Q` | Type 2 |
| Normal Loss Function | `L(z) = phi(z) - z × (1 - Phi(z))` | Fill rate computation |
| z from fill rate | `L(z) = (1 - beta) × Q / sigma_DDLT` (solve iteratively) | Inverse fill rate |
| Critical Ratio (Newsvendor) | `CR = Cu / (Cu + Co)` | Single-period |

### KPI Formulas

| KPI | Formula |
|-----|---------|
| Days of Supply | `DOS = qty_on_hand / avg_daily_demand` |
| Inventory Turns | `annual_COGS / avg_inventory_value` |
| Fill Rate | `qty_shipped / qty_ordered × 100` |
| GMROI | `gross_margin / avg_inventory_cost` |
| Carrying Cost | `avg_inventory_value × carrying_rate` |
| Forward Coverage | `(OH + IT + OO) / weekly_forecast_demand` |
| Health Score | `sum(w_i × normalize(metric_i, target_i))` |

### Cost Formulas

| Cost | Formula |
|------|---------|
| Total Relevant Cost | `TRC = (D/Q) × K + (Q/2) × h` |
| Annual Holding Cost | `(Q/2 + SS) × h` |
| Annual Ordering Cost | `(D/Q) × K` |
| Expected Shortage Cost | `(D/Q) × sigma_DDLT × L(z) × shortage_cost` |
| Total Annual Cost | `holding + ordering + shortage` |
| Carrying Rate | `capital + storage + insurance + obsolescence` |

---

## Appendix B: Solver Decision Matrix

| Problem | Size | Solver | Library | Expected Time | Phase |
|---------|------|--------|---------|---------------|-------|
| Single-item SS/ROP/EOQ | 1 item | Analytical formula | scipy.stats, numpy | < 1 ms | 1 |
| Batch SS (no constraints) | 10K items | Vectorized NumPy | numpy | < 100 ms | 1 |
| Category-level SS optimization | 100-1K items | Gradient (L-BFGS-B) | scipy.optimize | 1-5 s | 2 |
| Budget-constrained SS (LP) | 10K items | Linear Programming | pulp + HiGHS | < 5 s | 2 |
| SS with MOQ constraints (MIP) | 10K items | Mixed Integer Programming | pulp + HiGHS | < 30 s | 4 |
| Multi-objective (cost vs SL) | 10K items | NSGA-II | DEAP | 2-5 min | 4 |
| Monte Carlo simulation | 10K items × 10K runs | Vectorized simulation | numpy | 30-60 s | 4 |
| Policy optimization | 100 items | Bayesian optimization | Optuna | 5-10 min | 4 |
| Portfolio-wide Lagrangian | 100K items | Lagrangian relaxation | numpy + scipy | 1-5 min | 2 |
| Conformal prediction | 10K items | MAPIE wrapper | mapie | 10-30 s | 2 |

---

## Appendix C: Integration with Existing Features

| Existing Feature | Feature # | Integration Point | Data Flow Direction | Phase |
|-----------------|-----------|-------------------|--------------------|----|
| Champion Model Selection | F15 | Champion forecast as demand input | F15 → F34 | 1 |
| Backtest Framework (LGBM) | F9 | Forecast error distribution for empirical SS | F9 → F34 | 1 |
| Backtest Framework (CatBoost) | F12 | Forecast error distribution for empirical SS | F12 → F34 | 1 |
| Backtest Framework (XGBoost) | F13 | Forecast error distribution for empirical SS | F13 → F34 | 1 |
| Backtest Framework (Prophet) | F21 | Forecast error distribution for empirical SS | F21 → F34 | 1 |
| Backtest Framework (StatsForecast) | F24 | Forecast error distribution for empirical SS | F24 → F34 | 1 |
| DFU Clustering | F7 | Cluster labels for policy segmentation | F7 → F34 | 1 |
| Seasonality Detection | F30 | Seasonal profiles for buildup planning | F30 → F34 | 1 |
| Seasonality Filtering | F32 | Filter inventory metrics by seasonal profile | F32 → F34 | 1 |
| What-If Clustering Scenarios | F29 | Architecture pattern (async, concurrency, promote) | F29 → F34 (pattern) | 2 |
| Market Intelligence | F18 | Supply market context for supplier evaluation | F18 → F34 | 3 |
| Data Explorer | F16 | Inventory tables exposed in generic explorer | F34 → F16 | 1 |
| Multi-Model Forecast Support | F6 | Multi-model forecasts for confidence intervals | F6 → F34 | 2 |
| Backtest All-Lags Archive | F10 | Error distributions at multiple forecast horizons | F10 → F34 | 1 |

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **ABC Classification** | Segmentation of items by revenue contribution (A=top 80%, B=next 15%, C=bottom 5%) |
| **XYZ Classification** | Segmentation of items by demand variability (X=stable CV<0.5, Y=variable 0.5-1.0, Z=unpredictable CV>1.0) |
| **Carrying Cost** | Annual cost of holding inventory, expressed as a percentage of inventory value |
| **Champion Model** | The best-performing forecast model per DFU, selected by WAPE from model competition |
| **Conformal Prediction** | Distribution-free method for constructing prediction intervals with guaranteed coverage |
| **Croston's Method** | Forecasting method for intermittent demand that separately smooths demand size and interval |
| **CSL (Cycle Service Level)** | Probability of no stockout during a single replenishment cycle (Type 1 service level) |
| **CV (Coefficient of Variation)** | Standard deviation divided by mean; measures relative variability |
| **DDLT** | Demand During Lead Time — the total demand expected from order placement to receipt |
| **DFU** | Demand Forecast Unit — the unique combination of item + location + customer group |
| **DOS (Days of Supply)** | On-hand inventory divided by average daily demand |
| **EOQ** | Economic Order Quantity — the order size that minimizes total ordering + holding costs |
| **Fill Rate** | Fraction of demand filled from on-hand stock (Type 2 service level) |
| **GMROI** | Gross Margin Return on Inventory Investment |
| **GSM** | Guaranteed Service Model — the Graves-Willems method for multi-echelon optimization |
| **HiGHS** | Open-source LP/MIP solver used via PuLP |
| **Intermittent Demand** | Demand pattern with frequent zero-demand periods (ADI >= 1.32) |
| **Lagrangian Relaxation** | Optimization technique that relaxes constraints into the objective function |
| **Loss Function L(z)** | Standard normal loss function: L(z) = phi(z) - z*(1-Phi(z)); used for fill rate computation |
| **MEIO** | Multi-Echelon Inventory Optimization |
| **MOQ** | Minimum Order Quantity — the smallest quantity a supplier will accept |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II — multi-objective evolutionary optimizer |
| **Newsvendor** | Single-period inventory model for perishable/seasonal items |
| **PuLP** | Python Linear Programming library for LP/MIP modeling |
| **ROP (Reorder Point)** | Inventory position level at which a replenishment order is triggered |
| **Safety Stock** | Buffer inventory held to protect against demand and supply uncertainty |
| **Syntetos-Boylan** | Classification scheme for intermittent demand (smooth/erratic/intermittent/lumpy) |
| **TSB Method** | Teunter-Syntetos-Babai method — improved Croston's with obsolescence decay |
| **Wagner-Whitin** | Optimal dynamic lot sizing algorithm using dynamic programming |
| **WAPE** | Weighted Absolute Percentage Error — the metric used for champion model selection |

---

## Appendix E: References

1. Silver, E.A., Pyke, D.F., & Thomas, D.J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
2. Chopra, S., & Meindl, P. (2019). *Supply Chain Management: Strategy, Planning, and Operation* (7th ed.). Pearson.
3. Graves, S.C., & Willems, S.P. (2000). "Optimizing Strategic Safety Stock Placement in Supply Chains." *Manufacturing & Service Operations Management*, 2(1), 68-83.
4. Syntetos, A.A., & Boylan, J.E. (2005). "The Accuracy of Intermittent Demand Estimates." *International Journal of Forecasting*, 21(2), 303-314.
5. Wagner, H.M., & Whitin, T.M. (1958). "Dynamic Version of the Economic Lot Size Model." *Management Science*, 5(1), 89-96.
6. Lee, H.L., Padmanabhan, V., & Whang, S. (1997). "The Bullwhip Effect in Supply Chains." *Sloan Management Review*, 38(3), 93-102.
7. Scarf, H. (1960). "The Optimality of (s, S) Policies in the Dynamic Inventory Problem." *Mathematical Methods in the Social Sciences*, 196-202.
8. Clark, A.J., & Scarf, H. (1960). "Optimal Policies for a Multi-Echelon Inventory Problem." *Management Science*, 6(4), 475-490.
9. Croston, J.D. (1972). "Forecasting and Stock Control for Intermittent Demands." *Operational Research Quarterly*, 23(3), 289-303.
10. Teunter, R.H., Syntetos, A.A., & Babai, M.Z. (2011). "Intermittent Demand: Linking Forecasting to Inventory Obsolescence." *European Journal of Operational Research*, 214(3), 606-615.

---

*Feature 34: Inventory Planning Module — World-Class Design Specification*
*Demand Studio — February 2026*
*Version 1.0*

---

## Implementation Status (MVP — Phase 1)

The MVP implements a simplified version of this specification. Key differences from spec:

### What was implemented:
- **DDL**: `sql/017_create_fact_inventory_snapshot.sql` — column names differ from spec (`item_no` not `item`, `snapshot_date` not `exec_date`, `lead_time_days` as NUMERIC not INTEGER, `qty_on_hand`/`qty_on_hand_on_order`/`qty_on_order`/`mtd_sales`)
- **Normalize script**: Dedicated `scripts/normalize_inventory_csv.py` for merging 14 monthly CSVs (not generic normalizer)
- **Domain spec**: `INVENTORY_SPEC` in `common/domain_specs.py` with `source_columns` mapping
- **API endpoints** (4 inline routes in `api/main.py`): `/inventory/position`, `/inventory/kpis`, `/inventory/trend`, `/inventory/item-detail`
- **Materialized view**: `agg_inventory_monthly` with LAG() CTE for daily sales derivation, EOM snapshots, proper monthly sales (MAX not SUM)
- **Frontend**: `InventoryTab.tsx` with KPI cards (severity color-coded), trend chart, position table, item detail drill-down
- **Tests**: 22 tests in `tests/api/test_inventory.py`
- **Makefile targets**: `normalize-inventory`, `load-inventory`, `refresh-agg-inventory`, `db-apply-inventory`, `inventory-pipeline`
- **Inventory backtest** (Feature 37): 4 additional endpoints (`/inventory-backtest/*`) and `InvBacktestTab.tsx`

### What remains from the full spec:
- Phases 2-4 (safety stock optimization, ABC-XYZ classification engine, ML replenishment, S&OP integration)
- Separate inventory router module (currently inline)
- Advanced supply chain KPIs beyond DOS/WOC/Turns/LT Coverage


---

## Examples

### Example: Inventory KPI endpoint

```bash
curl -s "http://localhost:8000/inventory/kpis?item_no=100320&loc=1401-BULK" | jq .
# {
#   "dos": 45.2,           "woc": 6.4,    "turns": 8.1,
#   "avg_on_hand": 4320,   "avg_daily_sales": 96.0,
#   "lt_coverage_days": 38, "service_level_pct": 97.3,
#   "eom_on_hand": 3980,   "eom_on_order": 2100
# }
```

### Example: Inventory position (paginated)

```bash
curl -s "http://localhost:8000/inventory/position?item_no=100320&limit=3" | jq '.rows[0]'
# {
#   "item_no": "100320", "loc": "1401-BULK",
#   "snapshot_date": "2026-01-31",
#   "qty_on_hand": 3980, "qty_on_order": 2100,
#   "mtd_sales": 788, "lead_time_days": 42
# }
```

### Example: Load 190M-row inventory dataset

```bash
make inventory-pipeline
# Step 1: normalize-inventory → merge 14 monthly CSVs into single clean CSV
#   Reads: datafiles/Inventory_Snapshot_2024_12.csv ... Inventory_Snapshot_2026_01.csv
#   Derives: qty_on_order = qty_on_hand_on_order - qty_on_hand
#   Output: data/inventory_snapshot_clean.csv (~190M rows)
# Step 2: load-inventory → PostgreSQL COPY → fact_inventory_snapshot
# Step 3: refresh agg_inventory_monthly materialized view
```

### Example: DOS and WOC thresholds for KPI color-coding

| KPI             | Green (OK)   | Yellow (Warning) | Red (Critical) |
|-----------------|-------------|-----------------|----------------|
| Days of Supply  | DOS ≥ 30    | 14 ≤ DOS < 30   | DOS < 14       |
| Weeks of Cover  | WOC ≥ 4.3   | 2.0 ≤ WOC < 4.3 | WOC < 2.0      |
| Inventory Turns | Turns ≥ 6   | 3 ≤ Turns < 6   | Turns < 3      |

### Example: Inventory trend endpoint

```bash
curl -s "http://localhost:8000/inventory/trend?item_no=100320&loc=1401-BULK&months=6" | jq '.rows[0]'
# {
#   "month_start": "2025-08-01",
#   "eom_qty_on_hand": 4120,
#   "eom_qty_on_hand_on_order": 6220,
#   "monthly_sales": 788,
#   "avg_daily_sls": 25.4,
#   "dos": 162.2,
#   "latest_lead_time_days": 42,
#   "snapshot_days": 31
# }
```

### Example: Inventory item-detail endpoint

```bash
curl -s "http://localhost:8000/inventory/item-detail?item_no=100320&loc=1401-BULK" | jq .
# {
#   "item_no": "100320",
#   "loc": "1401-BULK",
#   "item_desc": "CABERNET SAUV 750ML",
#   "brand": "Acme Wines",
#   "category": "Red Wine",
#   "cluster_assignment": "high_volume_steady",
#   "seasonality_profile": "non_seasonal",
#   "latest_snapshot": {
#     "snapshot_date": "2026-01-31",
#     "qty_on_hand": 3980,
#     "qty_on_order": 2100,
#     "lead_time_days": 42,
#     "mtd_sales": 788
#   },
#   "kpis": {
#     "dos": 45.2,
#     "woc": 6.4,
#     "turns": 8.1,
#     "lt_coverage_days": 38
#   }
# }
```
