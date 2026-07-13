"""Shared constants for API tests.

Centralizes commonly duplicated magic values (Z-scores, model IDs,
timestamps, item/location identifiers, segments, statuses, pagination
defaults) so that tests can import them instead of re-declaring.

Usage:
    from tests.api.test_constants import MODEL_LGBM, ITEM_1, DEFAULT_LIMIT
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Z-score values (service level → Z)
# ---------------------------------------------------------------------------

Z_90 = 1.282       # 90% service level
Z_95 = 1.645       # 95% service level
Z_975 = 1.96       # 97.5% service level
Z_99 = 2.33        # 99% service level
Z_995 = 2.576      # 99.5% service level

Z_SCORES = {
    0.90: Z_90,
    0.95: Z_95,
    0.975: Z_975,
    0.99: Z_99,
    0.995: Z_995,
}


# ---------------------------------------------------------------------------
# Common timestamps
# ---------------------------------------------------------------------------

TS_2025_01_01 = "2025-01-01"
TS_2025_01_15 = "2025-01-15T10:00:00"
TS_2025_02_01 = "2025-02-01"
TS_2025_03_01 = "2025-03-01"
TS_2025_04_01 = "2025-04-01"
TS_2025_06_01 = "2025-06-01"
TS_2025_06_30 = "2025-06-30"

TS_2026_03_20_10 = "2026-03-20T10:00:00"
TS_2026_03_20_11 = "2026-03-20T11:00:00"
TS_2026_03_22_10 = "2026-03-22T10:00:00"
TS_2026_03_22_11 = "2026-03-22T11:00:00"
TS_2026_03_23_08 = "2026-03-23T08:00:00"
TS_2026_03_25_10 = "2026-03-25T10:00:00+00:00"
TS_2026_03_25_10_05 = "2026-03-25T10:00:05+00:00"
TS_2026_03_25_10_05_COMPLETED = "2026-03-25T10:05:00+00:00"


# ---------------------------------------------------------------------------
# Model IDs (backtest / champion / tuning)
# ---------------------------------------------------------------------------

MODEL_LGBM = "lgbm_cluster"
MODEL_NHITS = "nhits"
MODEL_NBEATS = "nbeats"
MODEL_MSTL = "mstl"
MODEL_CHRONOS2_ENRICHED = "chronos2_enriched"
MODEL_EXTERNAL = "external"

FORECAST_MODELS = [
    MODEL_LGBM,
    MODEL_NHITS,
    MODEL_NBEATS,
    MODEL_MSTL,
    MODEL_CHRONOS2_ENRICHED,
]

# Model IDs as JSON string (for champion experiments row)
FORECAST_MODELS_JSON = (
    '["lgbm_cluster", "nhits", "nbeats", "mstl", "chronos2_enriched"]'
)


# ---------------------------------------------------------------------------
# ABC / XYZ segments
# ---------------------------------------------------------------------------

ABC_A = "A"
ABC_B = "B"
ABC_C = "C"
ABC_ALL = [ABC_A, ABC_B, ABC_C]

XYZ_X = "X"
XYZ_Y = "Y"
XYZ_Z = "Z"
XYZ_ALL = [XYZ_X, XYZ_Y, XYZ_Z]

# Combined segments
SEG_AX = "AX"
SEG_BZ = "BZ"


# ---------------------------------------------------------------------------
# Common item IDs used in tests
# ---------------------------------------------------------------------------

ITEM_1 = "ITEM001"
ITEM_2 = "ITEM002"
ITEM_100320 = "100320"
ITEM_100321 = "100321"

# Alternate item IDs used across test files
ITEM_I001 = "I001"
ITEM_ITEM1 = "ITEM1"
ITEM_ITEM2 = "ITEM2"


# ---------------------------------------------------------------------------
# Common location IDs used in tests
# ---------------------------------------------------------------------------

LOC_1 = "LOC1"
LOC_2 = "LOC2"
LOC_BULK = "1401-BULK"
LOC_001 = "LOC001"


# ---------------------------------------------------------------------------
# Common customer groups
# ---------------------------------------------------------------------------

CUSTOMER_GRP1 = "GRP1"


# ---------------------------------------------------------------------------
# Status values (used across jobs, experiments, exceptions)
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# Exception-specific statuses
STATUS_OPEN = "open"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_RESOLVED = "resolved"

# All job statuses
JOB_STATUSES = [STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED]


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

ALL_SEVERITIES = [SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW]


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------

EXCEPTION_BELOW_ROP = "below_rop"
EXCEPTION_BELOW_ROP_CRITICAL = "below_rop_critical"
EXCEPTION_BELOW_SS = "below_ss"
EXCEPTION_STOCKOUT = "stockout"
EXCEPTION_EXCESS = "excess"
EXCEPTION_ZERO_VELOCITY = "zero_velocity"

ALL_EXCEPTION_TYPES = [
    EXCEPTION_BELOW_ROP, EXCEPTION_BELOW_ROP_CRITICAL,
    EXCEPTION_BELOW_SS, EXCEPTION_STOCKOUT,
    EXCEPTION_EXCESS, EXCEPTION_ZERO_VELOCITY,
]


# ---------------------------------------------------------------------------
# Insight types
# ---------------------------------------------------------------------------

INSIGHT_STOCKOUT_RISK = "stockout_risk"
INSIGHT_EXCESS_INVENTORY = "excess_inventory"


# ---------------------------------------------------------------------------
# Default pagination parameters
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 50
DEFAULT_OFFSET = 0
MAX_LIMIT = 1000

# Common pagination combos
PAGINATION_SMALL = {"limit": 10, "offset": 0}
PAGINATION_DEFAULT = {"limit": DEFAULT_LIMIT, "offset": DEFAULT_OFFSET}
PAGINATION_WITH_OFFSET = {"limit": 10, "offset": 20}


# ---------------------------------------------------------------------------
# Champion experiment strategies
# ---------------------------------------------------------------------------

STRATEGY_EXPANDING = "expanding"
STRATEGY_ROLLING = "rolling"
STRATEGY_DECAY = "decay"
STRATEGY_ENSEMBLE = "ensemble"
STRATEGY_META_LEARNER = "meta_learner"

ALL_STRATEGIES = [
    STRATEGY_EXPANDING, STRATEGY_ROLLING, STRATEGY_DECAY,
    STRATEGY_ENSEMBLE, STRATEGY_META_LEARNER,
]


# ---------------------------------------------------------------------------
# Champion experiment metrics
# ---------------------------------------------------------------------------

METRIC_ACCURACY_PCT = "accuracy_pct"
METRIC_WAPE = "wape"


# ---------------------------------------------------------------------------
# Tuning experiment timeframes
# ---------------------------------------------------------------------------

TIMEFRAME_A = "A"
TIMEFRAME_B = "B"
TIMEFRAME_C = "C"


# ---------------------------------------------------------------------------
# Common accuracy / error values used in mocks
# ---------------------------------------------------------------------------

ACCURACY_BASELINE = 69.34
WAPE_BASELINE = 30.66
BIAS_BASELINE = -0.0132
N_PREDICTIONS_DEFAULT = 2725140
N_DFUS_DEFAULT = 50602

ACCURACY_IMPROVED = 72.5
WAPE_IMPROVED = 27.5
BIAS_IMPROVED = 0.032


# ---------------------------------------------------------------------------
# Cluster experiment defaults
# ---------------------------------------------------------------------------

SCENARIO_ID_DEFAULT = "sc_20260320_100000_a1b2"
TEMPLATE_PRODUCTION_BASELINE = "production_baseline"
TEMPLATE_EXPANDING_CONSERVATIVE = "expanding_conservative"
OPTIMAL_K_DEFAULT = 8
SILHOUETTE_SCORE_DEFAULT = 0.342


# ---------------------------------------------------------------------------
# Test base URL (used with httpx.AsyncClient)
# ---------------------------------------------------------------------------

TEST_BASE_URL = "http://test"
