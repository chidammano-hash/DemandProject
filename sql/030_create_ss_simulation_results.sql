-- IPfeature10: Safety Stock Monte Carlo Simulation
-- Table: fact_ss_simulation_results
-- Stores per-item-loc simulation results with service level curves.

CREATE TABLE IF NOT EXISTS fact_ss_simulation_results (
    sim_sk                  BIGSERIAL PRIMARY KEY,
    sim_run_id              TEXT NOT NULL,
    item_id                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    simulation_date         DATE NOT NULL,
    n_simulations           INTEGER NOT NULL,
    -- Distribution params used
    demand_distribution     TEXT,              -- 'empirical' | 'normal'
    demand_mean             NUMERIC(15,4),
    demand_std              NUMERIC(15,4),
    lt_distribution         TEXT,              -- 'empirical' | 'constant'
    lt_mean_days            NUMERIC(10,2),
    lt_std_days             NUMERIC(10,2),
    -- Results: JSONB array [{ss_qty, csl}]
    results_by_ss_level     JSONB NOT NULL,
    -- Recommendations
    target_csl              NUMERIC(6,4),
    recommended_ss          NUMERIC(15,4),     -- minimum SS achieving target_csl
    recommended_ss_days     NUMERIC(10,2),     -- recommended_ss / avg_daily_demand
    -- Comparison with analytical formula
    analytical_ss           NUMERIC(15,4),     -- from fact_safety_stock_targets.ss_combined
    sim_vs_analytical_pct   NUMERIC(10,2),     -- (sim - analytical) / analytical × 100
    -- Metadata
    run_duration_secs       NUMERIC(8,2),
    load_ts                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ss_sim_run_item
    ON fact_ss_simulation_results (sim_run_id, item_id, loc);
CREATE INDEX IF NOT EXISTS idx_ss_sim_item_loc
    ON fact_ss_simulation_results (item_id, loc, simulation_date DESC);
CREATE INDEX IF NOT EXISTS idx_ss_sim_divergence
    ON fact_ss_simulation_results (sim_vs_analytical_pct)
    WHERE ABS(sim_vs_analytical_pct) > 20;
