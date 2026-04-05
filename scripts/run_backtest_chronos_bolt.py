"""
Run Chronos Bolt (v2) foundation model backtest on ALL DFUs.

Chronos Bolt uses a native encoder architecture — up to 250x faster than
Chronos-T5 Large with comparable accuracy. Zero-shot, no feature engineering.

Produces two CSVs under data/backtest/chronos_bolt/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
  - backtest_metadata.json            (accuracy stats)
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.foundation_backtest import (
    FoundationModelSpec,
    build_argparser,
    run_foundation_backtest,
)

logger = logging.getLogger(__name__)

MODEL_ID = "chronos_bolt"
CONFIG_KEY = "chronos_bolt"
DISPATCHER_KEY = "chronos_bolt"


def _extract_params(algo_cfg: dict) -> dict:
    return {
        "model_size": algo_cfg.get("model_size", "base"),
        "device": algo_cfg.get("device", "auto"),
        "batch_size": algo_cfg.get("batch_size", 1024),
        "num_samples": algo_cfg.get("num_samples", 12),
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
        f"Chronos Bolt backtest config: model_id={model_id}, n_timeframes={n_timeframes}, "
        f"embargo_months={embargo_months}, model_size={params['model_size']}, "
        f"batch_size={params['batch_size']}, workers={n_workers}"
    )


spec = FoundationModelSpec(
    model_id=MODEL_ID,
    config_key=CONFIG_KEY,
    dispatcher_key=DISPATCHER_KEY,
    display_name="Chronos Bolt",
    extract_params=_extract_params,
    model_params_key="chronos_bolt_params",
    extra_metadata={
        "params_source": "forecast_pipeline_config",
        "model_type": "foundation_model",
        "architecture": "chronos_bolt_v2",
        "zero_shot": True,
    },
    log_config_summary=_log_config,
    supports_parallel=True,
    profiler_prefix="bolt",
    tmp_dir_prefix="bolt_bt_",
)


def main() -> None:
    parser = build_argparser(
        "Run Chronos Bolt (v2) foundation model backtest (zero-shot, all DFUs)",
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
