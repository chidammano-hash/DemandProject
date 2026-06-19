"""Fine-tune Chronos-Bolt on this company's demand history (spec 32).

Chronos-Bolt's underlying model (`ChronosBoltModelForForecasting`) is a HuggingFace
`PreTrainedModel` whose `forward(context, mask, target, target_mask)` returns a
quantile (pinball) loss — so it fine-tunes with a standard HF `Trainer`. We adapt the
zero-shot base (`amazon/chronos-bolt-base`) to our portfolio, save the checkpoint under
`data/models/chronos_bolt_ft/v{N}/`, and let the existing inference path load it via
`ChronosBoltPipeline.from_pretrained(<dir>)` (see `_run_chronos_bolt_ft` in
`common/ml/expert_panel/foundation_models.py`).

CAUSALITY: a model fine-tuned on ALL history would leak into early backtest timeframes.
So we train ONLY on months strictly before ``--train-cutoff`` and record the cutoff in
`training_metadata.json`. The fine-tuned variant must therefore be SCORED only on months
>= the cutoff (the post-cutoff holdout) — e.g. via /forecast/accuracy/decomposition with
month_from=<cutoff>. Earlier timeframes are in-sample and must be excluded.

Usage:
    python scripts/ml/finetune_chronos_bolt.py --train-cutoff 2026-03-01
    python scripts/ml/finetune_chronos_bolt.py --train-cutoff 2026-03-01 \
        --max-steps 2 --limit-series 200 --device cpu      # smoke test

Config: forecast_pipeline_config.yaml -> algorithms.chronos_bolt_ft.params(.fine_tune).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":  # bootstrap only (CLAUDE.md: parents[N] allowed here)
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.paths import DATA_DIR
from common.core.utils import _ts, get_algorithm_params
from common.ml.backtest_framework import load_backtest_data
from common.ml.expert_panel.foundation_models import _resolve_device

logger = logging.getLogger(__name__)

MODEL_ID = "chronos_bolt_ft"


# ---------------------------------------------------------------------------
# Training-window dataset
# ---------------------------------------------------------------------------
class _BoltWindowDataset:
    """Sliding (context, target) windows over per-series monthly demand.

    Each item is a dict of torch tensors the model.forward consumes directly:
    ``context`` (Lc,), ``mask`` (Lc, bool), ``target`` (H,), ``target_mask`` (H, bool).
    Contexts are left-padded to ``context_length``; targets right-padded to ``horizon``.
    """

    def __init__(self, windows: list[dict]):
        self._w = windows

    def __len__(self) -> int:
        return len(self._w)

    def __getitem__(self, idx: int) -> dict:
        return self._w[idx]


def _build_windows(
    sales: pd.DataFrame,
    *,
    context_length: int,
    horizon: int,
    min_series_months: int,
    stride: int,
    max_windows_per_series: int,
) -> list[dict]:
    """Build training windows from [sku_ck, startdate, qty] sales (causal cutoff already applied)."""
    import torch

    windows: list[dict] = []
    sales = sales.sort_values(["sku_ck", "startdate"])
    for _sku, grp in sales.groupby("sku_ck", sort=False):
        qty = grp["qty"].to_numpy(dtype=np.float32)
        n = len(qty)
        if n < min_series_months:
            continue
        # Window end-points: each yields context = qty[:e], target = qty[e:e+horizon].
        # Need >= 1 future month and a non-trivial context (>= 3 obs).
        ends = list(range(max(3, n - max_windows_per_series * stride), n, stride))
        ends = [e for e in ends if e < n][-max_windows_per_series:]
        for e in ends:
            ctx = qty[max(0, e - context_length):e]
            tgt = qty[e:e + horizon]
            if len(ctx) < 3 or len(tgt) < 1:
                continue
            # Left-pad context to context_length; right-pad target to horizon.
            cbuf = np.zeros(context_length, dtype=np.float32)
            cmask = np.zeros(context_length, dtype=bool)
            cbuf[-len(ctx):] = ctx
            cmask[-len(ctx):] = True
            tbuf = np.zeros(horizon, dtype=np.float32)
            tmask = np.zeros(horizon, dtype=bool)
            tbuf[:len(tgt)] = tgt
            tmask[:len(tgt)] = True
            windows.append({
                "context": torch.from_numpy(cbuf),
                "mask": torch.from_numpy(cmask),
                "target": torch.from_numpy(tbuf),
                "target_mask": torch.from_numpy(tmask),
            })
    return windows


def _collate(batch: list[dict]) -> dict:
    import torch
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-Bolt on company demand (spec 32)")
    parser.add_argument("--train-cutoff", type=str, default=None,
                        help="Train only on startdate < this YYYY-MM-DD (leakage-free holdout above it). "
                             "Default: all history up to the planning date.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override fine_tune.num_steps (smoke runs)")
    parser.add_argument("--limit-series", type=int, default=None, help="Cap number of series (smoke runs)")
    parser.add_argument("--loc", type=str, default=None, help="Filter to a single location")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/mps/cuda)")
    parser.add_argument("--version", type=str, default=None, help="Checkpoint version dir name (default: next v{N})")
    args = parser.parse_args()

    params = get_algorithm_params(MODEL_ID)
    ft = dict(params.get("fine_tune", {}))
    base_model = params.get("base_model") or f"amazon/chronos-bolt-{params.get('model_size', 'base')}"
    checkpoint_root = Path(params.get("checkpoint_dir") or (DATA_DIR / "models" / MODEL_ID))

    context_length = int(ft.get("context_length", 36))
    horizon = int(ft.get("prediction_length", params.get("prediction_length", 6)))
    lr = float(ft.get("learning_rate", 1e-4))
    num_steps = int(args.max_steps if args.max_steps is not None else ft.get("num_steps", 2000))
    batch_size = int(ft.get("batch_size", 32))
    warmup_ratio = float(ft.get("warmup_ratio", 0.1))
    min_series_months = int(ft.get("min_series_months", 12))
    stride = int(ft.get("window_stride", 1))
    max_windows = int(ft.get("max_windows_per_series", 24))
    seed = int(ft.get("seed", 42))
    device = _resolve_device(args.device or params.get("device", "auto"))

    logger.info("[%s] fine-tune start: base=%s device=%s cutoff=%s steps=%d",
                _ts(), base_model, device, args.train_cutoff, num_steps)

    # --- Data (causal cutoff) ---
    sales, _dfu, _item = load_backtest_data(get_db_params(), include_item_attrs=False)
    sales = sales[["sku_ck", "startdate", "qty"]].copy()
    sales["startdate"] = pd.to_datetime(sales["startdate"])
    if args.train_cutoff:
        cutoff = pd.Timestamp(args.train_cutoff)
        sales = sales[sales["startdate"] < cutoff]
    if args.loc:
        # sku_ck encodes loc; filter via the dfu attrs map is overkill for a CLI filter.
        keep = _dfu.loc[_dfu["loc"] == args.loc, "sku_ck"] if "loc" in _dfu.columns else None
        if keep is not None:
            sales = sales[sales["sku_ck"].isin(set(keep))]
    if args.limit_series:
        keep_skus = sales["sku_ck"].drop_duplicates().head(args.limit_series)
        sales = sales[sales["sku_ck"].isin(set(keep_skus))]
    logger.info("[%s] training rows=%d series=%d", _ts(), len(sales), sales["sku_ck"].nunique())

    windows = _build_windows(
        sales, context_length=context_length, horizon=horizon,
        min_series_months=min_series_months, stride=stride, max_windows_per_series=max_windows,
    )
    if not windows:
        logger.error("No training windows built (check min_series_months / cutoff). Aborting.")
        raise SystemExit(1)
    logger.info("[%s] built %d training windows", _ts(), len(windows))
    dataset = _BoltWindowDataset(windows)

    # --- Model + Trainer ---
    import torch
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(seed)
    model = ChronosBoltModelForForecasting.from_pretrained(base_model)
    model.train()

    class _BoltTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    version = args.version or f"v{1 + _existing_versions(checkpoint_root)}"
    out_dir = checkpoint_root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir / "_hf_trainer"),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_steps=num_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=max(1, num_steps // 10),
        save_strategy="no",
        report_to=[],
        seed=seed,
        use_cpu=(device == "cpu"),
        dataloader_num_workers=0,
    )
    trainer = _BoltTrainer(model=model, args=targs, train_dataset=dataset, data_collator=_collate)
    train_out = trainer.train()
    final_loss = float(train_out.training_loss) if train_out.training_loss is not None else None
    logger.info("[%s] training done: final_loss=%s", _ts(), final_loss)

    # --- Save checkpoint (config.json + weights) for ChronosBoltPipeline.from_pretrained ---
    model.save_pretrained(str(out_dir))
    metadata = {
        "model_id": MODEL_ID,
        "base_model": base_model,
        "version": version,
        "train_cutoff": args.train_cutoff,
        "holdout_note": "Score only months >= train_cutoff (post-cutoff) to avoid leakage.",
        "context_length": context_length,
        "horizon": horizon,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "n_windows": len(windows),
        "n_series": int(sales["sku_ck"].nunique()),
        "final_train_loss": final_loss,
        "device": device,
        "saved_at": _ts(),
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("[%s] checkpoint saved: %s", _ts(), out_dir)
    logger.info("Load it with ChronosBoltPipeline.from_pretrained('%s')", out_dir)


def _existing_versions(root: Path) -> int:
    """Highest existing v{N} index under root (0 if none)."""
    if not root.is_dir():
        return 0
    nums = [int(d.name[1:]) for d in root.glob("v*") if d.name[1:].isdigit()]
    return max(nums) if nums else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
