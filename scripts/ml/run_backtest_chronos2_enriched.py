"""
Run Chronos 2 Enriched backtest — Chronos 2 with covariate features.

Unlike the zero-shot Chronos 2 backtest, this version passes:
  - past_covariates: lag/rolling/croston features + categoricals
  - future_covariates: calendar/fourier features (known for any future date)

Produces CSVs under data/backtest/chronos2_enriched/.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.expert_panel.foundation_models import run_foundation_models
from common.ml.feature_engineering import build_feature_matrix, mask_future_sales
from common.ml.foundation_backtest import (
    FoundationModelSpec,
    PerTimeframeContext,
    PreTimeframeContext,
    build_argparser,
    run_foundation_backtest,
)
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

MODEL_ID = "chronos2_enriched"
CONFIG_KEY = "chronos2_enriched"
DISPATCHER_KEY = "chronos2_enriched"


def _extract_params(algo_cfg: dict) -> dict:
    return {
        "device": algo_cfg.get("device", "auto"),
        "batch_size": algo_cfg.get("batch_size", 512),
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
        f"Chronos 2 Enriched config: model_id={model_id}, n_timeframes={n_timeframes}, "
        f"embargo_months={embargo_months}, batch_size={params['batch_size']}"
    )


def _pre_timeframe_hook(ctx: PreTimeframeContext) -> pd.DataFrame:
    """Build the feature matrix once before iterating through timeframes."""
    logger.info("Step 3: Building feature matrix (one-time)...")
    with profiled_section("build_features"):
        full_grid = build_feature_matrix(
            ctx.sales_df, ctx.dfu_attrs, ctx.item_attrs, ctx.all_months,
            cat_dtype="str", customer_features=ctx.customer_features,
        )
    logger.info("Feature matrix: %s", full_grid.shape)
    return full_grid


def _per_timeframe_hook(ctx: PerTimeframeContext, full_grid: pd.DataFrame) -> pd.DataFrame:
    """Mask features and run enriched inference for a single timeframe."""
    train_end = ctx.tf["train_end"]
    label = ctx.label

    logger.info(
        "  Train: %s rows (%s DFUs), Predict months: %d, Features: pending",
        f"{len(ctx.train_sales):,}",
        f"{ctx.train_sales['sku_ck'].nunique():,}",
        len(ctx.predict_months),
    )

    with profiled_section(f"mask_tf_{label}"):
        masked_grid = mask_future_sales(full_grid, train_end)

    train_grid = masked_grid[masked_grid["startdate"] <= train_end].copy()

    logger.info(
        "  Train: %s rows (%s DFUs), Predict months: %d, Features: %d cols",
        f"{len(ctx.train_sales):,}",
        f"{ctx.train_sales['sku_ck'].nunique():,}",
        len(ctx.predict_months),
        train_grid.shape[1],
    )

    with profiled_section(f"c2e_tf_{label}"):
        preds = run_foundation_models(
            ctx.train_sales[["sku_ck", "startdate", "qty"]],
            ctx.predict_months,
            {ctx.dispatcher_key: ctx.model_params},
            feature_grid=train_grid,
        )

    return preds


spec = FoundationModelSpec(
    model_id=MODEL_ID,
    config_key=CONFIG_KEY,
    dispatcher_key=DISPATCHER_KEY,
    display_name="Chronos 2 Enriched",
    extract_params=_extract_params,
    model_params_key="chronos2_enriched_params",
    extra_metadata={
        "params_source": "forecast_pipeline_config",
        "model_type": "foundation_model",
        "architecture": "chronos2_enriched",
        "zero_shot": False,
        "covariates": {
            "past_numeric": [
                "qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_6", "qty_lag_12",
                "rolling_mean_3m", "rolling_mean_6m", "rolling_mean_12m",
                "mom_growth", "demand_accel", "volatility_ratio",
                "croston_demand_size", "croston_demand_interval", "croston_probability",
            ],
            "past_categorical": ["brand", "region", "abc_vol"],
            "future": [
                "month", "quarter", "is_quarter_end", "is_year_end", "days_in_month",
                "fourier_sin_12", "fourier_cos_12", "fourier_sin_6", "fourier_cos_6",
                "fourier_sin_4", "fourier_cos_4", "fourier_sin_3", "fourier_cos_3",
            ],
        },
    },
    log_config_summary=_log_config,
    supports_parallel=False,  # enriched needs feature grid, no parallel workers
    include_item_attrs=False,
    include_customer_features=True,
    pre_timeframe_hook=_pre_timeframe_hook,
    per_timeframe_hook=_per_timeframe_hook,
    profiler_prefix="c2e",
)


def main() -> None:
    parser = build_argparser(
        "Run Chronos 2 Enriched backtest (with covariates, all DFUs)",
        supports_parallel=False,
    )
    args = parser.parse_args()
    run_foundation_backtest(spec, args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
