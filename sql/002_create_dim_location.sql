CREATE TABLE IF NOT EXISTS dim_location (
  location_sk BIGSERIAL PRIMARY KEY,
  location_ck TEXT UNIQUE NOT NULL,
  location_id TEXT NOT NULL,
  site_id TEXT,
  site_desc TEXT,
  state_id TEXT,
  primary_demand_location TEXT,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_location_site ON dim_location (site_id);
CREATE INDEX IF NOT EXISTS idx_dim_location_state ON dim_location (state_id);
CREATE INDEX IF NOT EXISTS idx_dim_location_primary ON dim_location (primary_demand_location);
