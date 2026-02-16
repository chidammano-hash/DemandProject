CREATE TABLE IF NOT EXISTS dim_customer (
  customer_sk BIGSERIAL PRIMARY KEY,
  customer_ck TEXT UNIQUE NOT NULL,
  site TEXT NOT NULL,
  customer_no TEXT NOT NULL,
  customer_name TEXT,
  city TEXT,
  state TEXT,
  zip TEXT,
  premise_code TEXT,
  status TEXT,
  license_name TEXT,
  store_type_desc TEXT,
  chain_type_desc TEXT,
  state_chain_name TEXT,
  corp_chain_name TEXT,
  rpt_channel_desc TEXT,
  rpt_sub_channel_desc TEXT,
  rpt_ship_type_desc TEXT,
  customer_acct_type_desc TEXT,
  delivery_freq_code TEXT,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_customer_site ON dim_customer (site);
CREATE INDEX IF NOT EXISTS idx_dim_customer_no ON dim_customer (customer_no);
CREATE INDEX IF NOT EXISTS idx_dim_customer_state ON dim_customer (state);
CREATE INDEX IF NOT EXISTS idx_dim_customer_status ON dim_customer (status);
CREATE INDEX IF NOT EXISTS idx_dim_customer_channel ON dim_customer (rpt_channel_desc);
