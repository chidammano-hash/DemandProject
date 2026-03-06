"""SHAP feature importance API endpoints (Feature 42).

Reads SHAP CSV outputs written by run_tree_backtest() + save_shap_outputs()
from data/backtest/<model_id>/shap/. No database queries — all data is
served directly from the filesystem CSVs.

Endpoints:
    GET /forecast/shap/models
    GET /forecast/shap/{model_id}/summary
    GET /forecast/shap/{model_id}/timeframes
    GET /forecast/shap/{model_id}/timeframe/{idx}
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["shap"])

# Root of the model-scoped backtest output directories
_BACKTEST_DATA_DIR = Path(os.environ.get("BACKTEST_DATA_DIR", "data/backtest"))


def _shap_dir(model_id: str) -> Path:
    return _BACKTEST_DATA_DIR / model_id / "shap"


def _summary_csv(model_id: str) -> Path:
    return _shap_dir(model_id) / "shap_summary.csv"


def _timeframe_csv(model_id: str, idx: int) -> Path:
    return _shap_dir(model_id) / f"shap_timeframe_{idx:02d}.csv"


# ---------------------------------------------------------------------------
# GET /forecast/shap/models
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/models")
async def shap_models() -> dict:
    """List model IDs that have SHAP outputs available."""
    if not _BACKTEST_DATA_DIR.exists():
        return {"models": []}
    models = [
        d.name
        for d in sorted(_BACKTEST_DATA_DIR.iterdir())
        if d.is_dir() and _summary_csv(d.name).exists()
    ]
    return {"models": models}


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/summary
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/summary")
async def shap_summary(
    model_id: str,
    top_n: int = Query(default=15, ge=1, le=200),
) -> dict:
    """Cross-timeframe SHAP importance summary for a model.

    Returns features sorted by mean_abs_shap_across_timeframes descending.
    """
    csv_path = _summary_csv(model_id)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP summary found for model '{model_id}'. "
                   f"Run backtest with --shap-select first.",
        )
    df = pd.read_csv(csv_path)
    features = df.head(top_n).to_dict(orient="records")
    return {
        "model_id": model_id,
        "total_features": len(df),
        "features": features,
    }


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/timeframes
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/timeframes")
async def shap_timeframes(model_id: str) -> dict:
    """List all timeframes for which SHAP outputs exist for a model."""
    shap_d = _shap_dir(model_id)
    if not shap_d.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP outputs found for model '{model_id}'.",
        )
    timeframe_files = sorted(shap_d.glob("shap_timeframe_*.csv"))
    timeframes = []
    for f in timeframe_files:
        try:
            df = pd.read_csv(f, usecols=["timeframe", "cutoff_date"])
            idx = int(df["timeframe"].iloc[0])
            cutoff = df["cutoff_date"].iloc[0]
            label = chr(ord("A") + idx)
            timeframes.append({
                "index": idx,
                "label": label,
                "cutoff_date": str(cutoff),
            })
        except Exception:
            continue
    return {"model_id": model_id, "timeframes": timeframes}


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/timeframe/{idx}
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/timeframe/{idx}")
async def shap_timeframe_detail(
    model_id: str,
    idx: int,
    top_n: int = Query(default=15, ge=1, le=200),
) -> dict:
    """Per-timeframe SHAP feature importance detail.

    Returns features sorted by mean_abs_shap descending (rank 1 = most important).
    """
    csv_path = _timeframe_csv(model_id, idx)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP data for model '{model_id}' timeframe {idx}.",
        )
    df = pd.read_csv(csv_path)
    df = df.sort_values("rank").reset_index(drop=True)
    label = chr(ord("A") + idx) if 0 <= idx <= 25 else str(idx)
    cutoff_date = str(df["cutoff_date"].iloc[0]) if not df.empty else ""
    features = df.head(top_n).to_dict(orient="records")
    return {
        "model_id": model_id,
        "timeframe_idx": idx,
        "label": label,
        "cutoff_date": cutoff_date,
        "total_features": len(df),
        "features": features,
    }
