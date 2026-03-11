-- Inventory Rebalancing: Transfer Network Topology
-- Defines which locations can ship to which, at what cost, with what constraints.

CREATE TABLE IF NOT EXISTS dim_transfer_lane (
    lane_sk                    BIGSERIAL PRIMARY KEY,
    lane_id                    TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    source_loc                 TEXT NOT NULL,
    dest_loc                   TEXT NOT NULL,
    transfer_mode              TEXT NOT NULL DEFAULT 'truck'
                               CHECK (transfer_mode IN ('truck','rail','air','parcel')),
    -- Cost model ($/unit)
    cost_per_unit              NUMERIC(10,4) NOT NULL,
    handling_cost              NUMERIC(10,4) DEFAULT 0,
    freight_cost               NUMERIC(10,4) DEFAULT 0,
    receiving_cost             NUMERIC(10,4) DEFAULT 0,
    fixed_cost_per_shipment    NUMERIC(10,2) DEFAULT 0,
    -- Lead time
    transfer_lt_days           INTEGER NOT NULL DEFAULT 3,
    -- Constraints
    min_transfer_qty           INTEGER DEFAULT 1,
    max_transfer_qty           INTEGER,
    batch_size                 INTEGER DEFAULT 1,
    max_shipments_per_week     INTEGER DEFAULT 5,
    max_receiving_units_per_period INTEGER,
    -- Status
    is_active                  BOOLEAN DEFAULT TRUE,
    effective_from             DATE,
    effective_to               DATE,
    -- Metadata
    load_ts                    TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_loc, dest_loc, transfer_mode)
);

CREATE INDEX IF NOT EXISTS idx_lane_source ON dim_transfer_lane (source_loc) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_lane_dest   ON dim_transfer_lane (dest_loc) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_lane_pair   ON dim_transfer_lane (source_loc, dest_loc);
