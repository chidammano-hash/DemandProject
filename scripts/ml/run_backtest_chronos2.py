"""
Run Chronos 2 foundation model backtest on ALL DFUs.

Chronos 2 is Amazon's latest generation time-series foundation model.
Returns 21 quantile forecasts with built-in batching. Zero-shot, no feature engineering.

Produces two CSVs under data/backtest/chronos2/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
  - backtest_metadata.json            (accuracy stats)
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.foundation_backtest import (
    FoundationModelSpec,
    build_argparser,
    run_foundation_backtest,
)

logger = logging.getLogger(__name__)

MODEL_ID = "chronos2"
CONFIG_KEY = "chronos2"
DISPATCHER_KEY = "chronos2"


def _extract_params(algo_cfg: dict) -> dict:
    return {
        "device": algo_cfg.get("device", "auto"),
        "batch_size": algo_cfg.get("batch_size", 1024),
        "prediction_length": algo_cfg.get("prediction_length", 6),
    }


def _log_config(
    model_id: str,
    n_timeframes: int,
    embargo_months: int,
    params: dict,
    n_workers: int,
) -> str:
    return (
        f"Chronos 2 backtest config: model_id={model_id}, n_timeframes={n_timeframes}, "
        f"embargo_months={embargo_months}, batch_size={params['batch_size']}, "
        f"workers={n_workers}"
    )


spec = FoundationModelSpec(
    model_id=MODEL_ID,
    config_key=CONFIG_KEY,
    dispatcher_key=DISPATCHER_KEY,
    display_name="Chronos 2",
    extract_params=_extract_params,
    model_params_key="chronos2_params",
    extra_metadata={
        "params_source": "forecast_pipeline_config",
        "model_type": "foundation_model",
        "architecture": "chronos2",
        "zero_shot": True,
    },
    log_config_summary=_log_config,
    supports_parallel=True,
    profiler_prefix="c2",
    tmp_dir_prefix="chronos2_bt_",
)


def main() -> None:
    parser = build_argparser(
        "Run Chronos 2 foundation model backtest (zero-shot, all DFUs)",
        supports_parallel=True,
    )
    args = parser.parse_args()
    run_foundation_backtest(spec, args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
