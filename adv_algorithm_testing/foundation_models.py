"""Foundation models for Advanced Expert Panel.

Models: Chronos-2, TimesFM, Moirai, TimeGPT, Lag-Llama.
Each supports zero-shot forecasting — no training required.
Gracefully skips models whose libraries are not installed.
"""

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _check_chronos() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def _check_timesfm() -> bool:
    try:
        import timesfm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_nixtla() -> bool:
    try:
        import nixtla  # noqa: F401
        return True
    except ImportError:
        return False


def _check_moirai() -> bool:
    try:
        import uni2ts  # noqa: F401
        return True
    except ImportError:
        return False


def _check_lag_llama() -> bool:
    try:
        import lag_llama  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to a concrete device string (mps / cuda / cpu)."""
    if device_setting != "auto":
        return device_setting
    gpu_env = os.environ.get("DEMAND_GPU", "auto").lower()
    if gpu_env == "off":
        return "cpu"
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Chronos
# ---------------------------------------------------------------------------

def _run_chronos(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Amazon Chronos foundation model (zero-shot)."""
    if not _check_chronos():
        logger.info("chronos-forecasting not installed; skipping Chronos")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    from chronos import ChronosPipeline
    import torch

    model_size = params.get("model_size", "small")
    model_name = f"amazon/chronos-t5-{model_size}"
    device_setting = params.get("device", "auto")
    prediction_length = len(predict_months)
    num_samples = params.get("num_samples", 20)

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos: using Apple MPS GPU")

    logger.info("Chronos: loading %s on %s...", model_name, device)
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    sku_cks = sales_df["sku_ck"].unique()
    all_results: list[dict] = []

    # Batch by DFU — Chronos handles batches of context tensors
    grouped = sales_df.groupby("sku_ck", sort=False)
    batch_size = params.get("batch_size", 32)
    sku_list = list(sku_cks)

    for batch_start in range(0, len(sku_list), batch_size):
        batch_skus = sku_list[batch_start:batch_start + batch_size]
        contexts = []
        valid_skus = []

        for sku_ck in batch_skus:
            group = grouped.get_group(sku_ck).sort_values("startdate")
            values = group["qty"].values.astype(np.float32)
            if len(values) < 3:
                continue
            contexts.append(torch.tensor(values))
            valid_skus.append(sku_ck)

        if not contexts:
            continue

        try:
            forecasts = pipeline.predict(
                contexts,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            # forecasts shape: (n_series, num_samples, prediction_length)
            median_forecasts = np.median(forecasts.numpy(), axis=1)

            for i, sku_ck in enumerate(valid_skus):
                for j, month in enumerate(predict_months):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(median_forecasts[i, j]), 0.0),
                        "algorithm_id": "chronos",
                    })
        except (RuntimeError, ValueError) as exc:
            logger.warning("Chronos batch failed: %s", exc)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                "Chronos: %d/%d DFUs processed",
                min(batch_start + batch_size, len(sku_list)),
                len(sku_list),
            )

    result = pd.DataFrame(all_results)
    logger.info("Chronos: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# TimesFM
# ---------------------------------------------------------------------------

def _run_timesfm(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Google TimesFM foundation model (zero-shot)."""
    if not _check_timesfm():
        logger.info("timesfm not installed; skipping TimesFM")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import timesfm

    prediction_length = len(predict_months)
    context_length = params.get("context_length", 512)

    logger.info("TimesFM: loading model (context=%d)...", context_length)
    tfm = timesfm.TimesFm(
        context_len=context_length,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    tfm.load_from_checkpoint()

    grouped = sales_df.groupby("sku_ck", sort=False)
    all_results: list[dict] = []

    for sku_ck, group in grouped:
        values = group.sort_values("startdate")["qty"].values.astype(np.float32)
        if len(values) < 6:
            continue

        try:
            forecast, _ = tfm.forecast([values], freq=[0])
            preds = forecast[0][:prediction_length]

            for j, month in enumerate(predict_months):
                if j < len(preds):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(preds[j]), 0.0),
                        "algorithm_id": "timesfm",
                    })
        except (RuntimeError, ValueError) as exc:
            logger.warning("TimesFM failed for %s: %s", sku_ck, exc)

    result = pd.DataFrame(all_results)
    logger.info("TimesFM: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# TimeGPT
# ---------------------------------------------------------------------------

def _run_timegpt(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Nixtla TimeGPT (API-based, zero-shot)."""
    if not _check_nixtla():
        logger.info("nixtla not installed; skipping TimeGPT")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    api_key = os.environ.get("NIXTLA_API_KEY")
    if not api_key:
        logger.info("NIXTLA_API_KEY not set; skipping TimeGPT")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    from nixtla import NixtlaClient

    prediction_length = len(predict_months)
    logger.info("TimeGPT: calling API (h=%d)...", prediction_length)

    client = NixtlaClient(api_key=api_key)

    df = sales_df[["sku_ck", "startdate", "qty"]].copy()
    df = df.rename(columns={"sku_ck": "unique_id", "startdate": "ds", "qty": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    try:
        forecast_df = client.forecast(
            df=df,
            h=prediction_length,
            freq="MS",
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )

        result = forecast_df.rename(columns={
            "unique_id": "sku_ck",
            "ds": "startdate",
            "TimeGPT": "basefcst_pref",
        })
        result["basefcst_pref"] = np.maximum(result["basefcst_pref"].values, 0.0)
        result["algorithm_id"] = "timegpt"
        result = result[["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]]

        # Filter to requested predict months
        result["startdate"] = pd.to_datetime(result["startdate"])
        predict_ts = pd.to_datetime(predict_months)
        result = result[result["startdate"].isin(predict_ts)]

        logger.info("TimeGPT: %d predictions for %d DFUs", len(result),
                     result["sku_ck"].nunique() if not result.empty else 0)
        return result

    except (RuntimeError, ValueError) as exc:
        logger.warning("TimeGPT API call failed: %s", exc)
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )


# ---------------------------------------------------------------------------
# Moirai
# ---------------------------------------------------------------------------

def _run_moirai(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Salesforce Moirai foundation model (zero-shot).

    Requires: pip install "uni2ts[torch]"
    Weights downloaded from HuggingFace on first run and cached locally.
    No API key required.
    """
    if not _check_moirai():
        logger.info("uni2ts not installed; skipping Moirai. Install: pip install 'uni2ts[torch]'")
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    model_size = params.get("model_size", "small")
    device = _resolve_device(params.get("device", "auto"))
    prediction_length = len(predict_months)
    context_length = params.get("context_length", 200)
    num_samples = params.get("num_samples", 20)
    batch_size = params.get("batch_size", 32)
    hf_model_id = f"Salesforce/moirai-1.1-R-{model_size}"

    logger.info("Moirai: loading %s on %s...", hf_model_id, device)
    try:
        module = MoiraiModule.from_pretrained(hf_model_id)
        model = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = model.to(device)
        model.eval()
    except (RuntimeError, OSError) as exc:
        logger.warning("Moirai: failed to load model: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    sku_list = list(sales_df["sku_ck"].unique())
    grouped = sales_df.groupby("sku_ck", sort=False)
    all_results: list[dict] = []

    for batch_start in range(0, len(sku_list), batch_size):
        batch_skus = sku_list[batch_start: batch_start + batch_size]
        contexts: list[torch.Tensor] = []
        valid_skus: list[str] = []

        for sku_ck in batch_skus:
            group = grouped.get_group(sku_ck).sort_values("startdate")
            values = group["qty"].values.astype(np.float32)
            if len(values) < 3:
                continue
            ctx = values[-context_length:]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            valid_skus.append(sku_ck)

        if not contexts:
            continue

        # Pad to uniform length within the batch
        max_len = max(t.shape[0] for t in contexts)
        padded = torch.zeros(len(contexts), max_len, 1)
        observed = torch.zeros(len(contexts), max_len, 1, dtype=torch.bool)
        is_pad = torch.ones(len(contexts), max_len, dtype=torch.bool)

        for i, ctx_tensor in enumerate(contexts):
            L = ctx_tensor.shape[0]
            padded[i, -L:, 0] = ctx_tensor
            observed[i, -L:, 0] = True
            is_pad[i, -L:] = False

        padded = padded.to(device)
        observed = observed.to(device)
        is_pad = is_pad.to(device)

        try:
            with torch.no_grad():
                samples = model(
                    past_target=padded,
                    past_observed_target=observed,
                    past_is_pad=is_pad,
                )
            # samples: (batch, num_samples, prediction_length)
            medians = samples.median(dim=1).values.cpu().numpy()

            for i, sku_ck in enumerate(valid_skus):
                for j, month in enumerate(predict_months):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(medians[i, j]), 0.0),
                        "algorithm_id": "moirai",
                    })
        except (RuntimeError, ValueError) as exc:
            logger.warning("Moirai batch %d failed: %s", batch_start, exc)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                "Moirai: %d/%d DFUs processed",
                min(batch_start + batch_size, len(sku_list)), len(sku_list),
            )

    result = pd.DataFrame(all_results)
    logger.info(
        "Moirai: %d predictions for %d DFUs",
        len(result), result["sku_ck"].nunique() if not result.empty else 0,
    )
    return result


# ---------------------------------------------------------------------------
# Lag-Llama
# ---------------------------------------------------------------------------

def _run_lag_llama(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Lag-Llama foundation model (zero-shot).

    Requires: pip install "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama"
              pip install huggingface_hub gluonts
    Checkpoint downloaded from HuggingFace on first run and cached locally.
    No API key required.
    """
    if not _check_lag_llama():
        logger.info(
            "lag_llama not installed; skipping Lag-Llama. "
            "Install: pip install 'lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama'"
        )
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    import torch
    from gluonts.dataset.common import ListDataset
    from huggingface_hub import hf_hub_download
    from lag_llama.gluon.estimator import LagLlamaEstimator

    device = _resolve_device(params.get("device", "auto"))
    prediction_length = len(predict_months)
    context_length = params.get("context_length", 32)
    num_samples = params.get("num_samples", 20)
    batch_size = params.get("batch_size", 64)

    logger.info("Lag-Llama: downloading checkpoint from HuggingFace...")
    try:
        ckpt_path = hf_hub_download(
            "time-series-foundation-models/Lag-Llama", filename="lag-llama.ckpt"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_kwargs = ckpt["hyper_parameters"]["model_kwargs"]
    except (OSError, KeyError) as exc:
        logger.warning("Lag-Llama: failed to load checkpoint: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    logger.info("Lag-Llama: building estimator (device=%s)...", device)
    try:
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,
            **model_kwargs,
            num_parallel_samples=num_samples,
            batch_size=batch_size,
            nonnegative_pred_samples=True,
            device=torch.device(device),
        )
        lightning_module = estimator.load_from_checkpoint(checkpoint_path=ckpt_path)
        predictor = estimator.create_lightning_predictor(
            lightning_module, batch_size=batch_size
        )
    except (RuntimeError, TypeError) as exc:
        logger.warning("Lag-Llama: estimator build failed: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    # Build GluonTS ListDataset — one entry per DFU, in sku_ck order
    sku_order: list[str] = []
    entries: list[dict] = []
    for sku_ck, group in sales_df.groupby("sku_ck", sort=False):
        group = group.sort_values("startdate")
        if len(group) < 3:
            continue
        sku_order.append(str(sku_ck))
        entries.append({
            "start": pd.Period(group["startdate"].iloc[0], freq="M"),
            "target": group["qty"].values.astype(np.float32),
        })

    if not entries:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    test_data = ListDataset(entries, freq="M")

    all_results: list[dict] = []
    try:
        for sku_ck, forecast in zip(sku_order, predictor.predict(test_data)):
            # forecast.samples: (num_samples, prediction_length)
            median_preds = np.median(forecast.samples, axis=0)
            for j, month in enumerate(predict_months):
                if j < len(median_preds):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(median_preds[j]), 0.0),
                        "algorithm_id": "lag_llama",
                    })
    except (RuntimeError, StopIteration) as exc:
        logger.warning("Lag-Llama: prediction failed: %s", exc)

    result = pd.DataFrame(all_results)
    logger.info(
        "Lag-Llama: %d predictions for %d DFUs",
        len(result), result["sku_ck"].nunique() if not result.empty else 0,
    )
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FOUNDATION_DISPATCH: dict[str, Any] = {
    "chronos": _run_chronos,
    "timesfm": _run_timesfm,
    "timegpt": _run_timegpt,
    "moirai": _run_moirai,
    "lag_llama": _run_lag_llama,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_foundation_models(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
) -> pd.DataFrame:
    """Run all enabled foundation models.

    Foundation models are zero-shot — they produce forecasts without any
    training on the target dataset. They are pretrained on large-scale
    time series corpora.

    Args:
        sales_df: Historical sales with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        enabled_models: {model_id: params_dict} for enabled foundation models.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    empty = pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )

    if not enabled_models:
        return empty

    if sales_df.empty:
        return empty

    all_results: list[pd.DataFrame] = []

    for model_id, params in enabled_models.items():
        fn = _FOUNDATION_DISPATCH.get(model_id)
        if fn is None:
            logger.info("Foundation model '%s' not yet implemented; skipping", model_id)
            continue

        t0 = time.time()
        logger.info("Running foundation model: %s...", model_id)

        try:
            result = fn(sales_df, predict_months, params)
            if not result.empty:
                all_results.append(result)
                logger.info(
                    "%s complete: %d predictions in %.1fs",
                    model_id, len(result), time.time() - t0,
                )
            else:
                logger.info("%s: no predictions produced", model_id)
        except (RuntimeError, ValueError, ImportError) as exc:
            logger.warning("Foundation model %s failed: %s", model_id, exc)

    if not all_results:
        return empty

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "Foundation models complete: %d predictions from %d models",
        len(combined), combined["algorithm_id"].nunique(),
    )
    return combined
