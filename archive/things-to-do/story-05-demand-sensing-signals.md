# Story 05 — Real-Time Demand Sensing & External Signals

## Problem
The system has stub endpoints for external signals (`fact_external_signal` table exists, `demand_signals_external_config.yaml` exists) and basic velocity-based demand signals, but no real POS/EDI integration, weather impact, economic indicator feeds, or promotional lift modeling. Short-horizon demand adjustments are manual.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `pos_daily_sales.csv` | CSV | item × store × day | item_no, store_id, sale_date, qty_sold, revenue, promo_flag, markdown_pct | POS/EDI |
| **F2** | `weather_actuals.csv` | CSV | location × day | location_id, date, temp_high_f, temp_low_f, precipitation_in, snow_in, weather_event (heat_wave|storm|freeze|none) | Weather API |
| **F3** | `economic_indicators.csv` | CSV | indicator × month | indicator_name (cpi|unemployment|consumer_confidence|housing_starts|gdp_growth), region, period_month, value, yoy_change_pct | BLS/FRED |
| **F4** | `promotional_calendar.csv` | CSV | promotion | promo_id, promo_name, promo_type (tpr|bogo|display|ad_feature), item_no, loc, start_date, end_date, expected_lift_pct, actual_lift_pct, discount_pct, ad_spend | Marketing/Trade |
| **F5** | `social_sentiment.csv` | CSV | brand × week | brand_name, week_start, sentiment_score (-1 to 1), mention_count, trending_flag, source (twitter|reddit|news) | Social Listening |
| **F6** | `competitor_pricing.csv` | CSV | item × competitor × week | item_no, competitor_name, week_start, competitor_price, our_price, price_gap_pct | Price Intelligence |

## Incremental Implementation

### Phase 1: POS Daily Sales Integration (1 sprint)
- `sql/086_create_pos_daily.sql` — `fact_pos_daily_sales` (daily grain, ~10M rows/month)
- `scripts/load_pos_daily.py` — incremental daily load with dedup
- Daily-to-monthly rollup view for comparison with `fact_sales_monthly`
- Intra-month demand signal enhancement: replace velocity proxy with actual POS data
- API: `GET /demand-sensing/pos/daily-trend`, `/pos/vs-forecast`

### Phase 2: Weather & Economic Signals (1 sprint)
- `sql/087_create_external_signals.sql` — extend existing `fact_external_signal` with typed columns
- `scripts/fetch_weather_data.py` — daily pull from OpenWeatherMap or NOAA API
- `scripts/fetch_economic_indicators.py` — monthly pull from FRED API
- Correlation engine: weather_event × category × region → demand impact coefficient
- API: `GET /signals/weather-impact`, `/signals/economic-outlook`
- Panel: External signals dashboard with correlation matrix

### Phase 3: Promotional Lift Modeling (1 sprint)
- `sql/088_create_promotions.sql` — `dim_promotion`, `fact_promo_performance`
- `scripts/compute_promo_lift.py` — baseline decomposition + lift estimation
- Integration with demand plan: inject promotional uplift into consensus forecast
- Integration with `fact_event_calendar` (already exists) — extend with promotional data
- API: `GET /promotions/calendar`, `/promotions/lift-analysis`, `/promotions/cannibalization`
- Panel: Promotional effectiveness dashboard, cannibalization analysis

### Phase 4: ML Demand Sensing Model (1 sprint)
- `scripts/train_demand_sensing_model.py` — gradient boosted model using POS + weather + promotions
- Short-horizon (1-4 week) override of statistical forecast when signals are strong
- Auto-adjust `alpha_weight` in blended demand based on signal quality
- Automated sensing → blend → replan pipeline

## Dependencies
- POS data requires EDI integration or retailer portal API (850/852 EDI transactions)
- Weather API: OpenWeatherMap (free tier) or NOAA
- FRED API: free, rate-limited
- `fact_blended_demand_plan` already supports `sensing_signal_qty` — extend with signal source attribution
- `fact_event_calendar` already exists — promotional data extends it

## Business Value
- 15-30% improvement in short-horizon forecast accuracy (industry benchmark)
- Weather-driven demand adjustments for seasonal/outdoor categories
- Promotional lift prediction reduces over/under-stocking during events
- Faster response to demand shifts (daily vs monthly signal)
