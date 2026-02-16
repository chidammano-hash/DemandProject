CREATE TABLE IF NOT EXISTS dim_time (
  time_sk BIGSERIAL PRIMARY KEY,
  time_ck TEXT UNIQUE NOT NULL,
  date_key DATE NOT NULL,
  day_name TEXT NOT NULL,
  day_of_week INTEGER NOT NULL,
  day_of_month INTEGER NOT NULL,
  day_of_year INTEGER NOT NULL,
  iso_week_year INTEGER NOT NULL,
  iso_week INTEGER NOT NULL,
  week_start_date DATE NOT NULL,
  week_end_date DATE NOT NULL,
  month_number INTEGER NOT NULL,
  month_name TEXT NOT NULL,
  month_start_date DATE NOT NULL,
  month_end_date DATE NOT NULL,
  quarter_number INTEGER NOT NULL,
  quarter_label TEXT NOT NULL,
  quarter_start_date DATE NOT NULL,
  quarter_end_date DATE NOT NULL,
  year_number INTEGER NOT NULL,
  year_start_date DATE NOT NULL,
  year_end_date DATE NOT NULL,
  week_bucket TEXT NOT NULL,
  month_bucket TEXT NOT NULL,
  quarter_bucket TEXT NOT NULL,
  year_bucket TEXT NOT NULL,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_time_date ON dim_time (date_key);
CREATE INDEX IF NOT EXISTS idx_dim_time_week_bucket ON dim_time (week_bucket);
CREATE INDEX IF NOT EXISTS idx_dim_time_month_bucket ON dim_time (month_bucket);
CREATE INDEX IF NOT EXISTS idx_dim_time_quarter_bucket ON dim_time (quarter_bucket);
CREATE INDEX IF NOT EXISTS idx_dim_time_year_number ON dim_time (year_number);
