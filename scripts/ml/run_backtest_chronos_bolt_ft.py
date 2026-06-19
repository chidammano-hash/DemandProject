"""Run the FINE-TUNED Chronos-Bolt backtest on all DFUs (spec 32).

Identical harness to run_backtest_chronos_bolt.py, but the dispatcher loads weights
from the local fine-tuned checkpoint (data/models/chronos_bolt_ft/v{N}/) produced by
scripts/ml/finetune_chronos_bolt.py, falling back to the zero-shot base if absent.

CAUSALITY: the fine-tune was trained on months < its train_cutoff (see the checkpoint's
training_metadata.json). Score this model ONLY on the post-cutoff holdout — the latest
backtest timeframe(s) / month_from >= cutoff — earlier timeframes are in-sample.

Outputs under data/backtest/chronos_bolt_ft/:
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

from common.ml.foundation_backtest import (  # noqa: E402 — after sys.path bootstrap
    FoundationModelSpec,
    build_argparser,
    run_foundation_backtest,
)

logger = logging.getLogger(__name__)

MODEL_ID = "chronos_bolt_ft"
CONFIG_KEY = "chronos_bolt_ft"
DISPATCHER_KEY = "chronos_bolt_ft"


def _extract_params(algo_cfg: dict) -> dict:
    return {
        "base_model": algo_cfg.get("base_model", "amazon/chronos-bolt-base"),
        "checkpoint_dir": algo_cfg.get("checkpoint_dir", "data/models/chronos_bolt_ft"),
        "model_size": algo_cfg.get("model_size", "base"),
        "device": algo_cfg.get("device", "auto"),
        "batch_size": algo_cfg.get("batch_size", 512),
        "num_samples": algo_cfg.get("num_samples", 12),
        "prediction_length": algo_cfg.get("prediction_length", 6),
    }


def _log_config(model_id: str, n_timeframes: int, embargo_months: int, params: dict, n_workers: int) -> str:
    return (
        f"Chronos Bolt FT backtest config: model_id={model_id}, n_timeframes={n_timeframes}, "
        f"embargo_months={embargo_months}, checkpoint_dir={params['checkpoint_dir']}, "
        f"batch_size={params['batch_size']}, workers={n_workers}"
    )


spec = FoundationModelSpec(
    model_id=MODEL_ID,
    config_key=CONFIG_KEY,
    dispatcher_key=DISPATCHER_KEY,
    display_name="Chronos Bolt (fine-tuned)",
    extract_params=_extract_params,
    model_params_key="chronos_bolt_ft_params",
    extra_metadata={
        "params_source": "forecast_pipeline_config",
        "model_type": "foundation_model",
        "architecture": "chronos_bolt_v2_finetuned",
        "zero_shot": False,
    },
    log_config_summary=_log_config,
    supports_parallel=True,
    profiler_prefix="bolt_ft",
    tmp_dir_prefix="bolt_ft_bt_",
)


def main() -> None:
    parser = build_argparser(
        "Run fine-tuned Chronos-Bolt backtest (spec 32, all DFUs)",
        supports_parallel=True,
    )
    args = parser.parse_args()
    run_foundation_backtest(spec, args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
