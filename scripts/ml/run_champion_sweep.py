"""
Champion strategy sweep (tournament) runner.

Orchestrates the EXISTING champion-experiment machinery: expands a grid of
candidate champion configs (each a real ``champion_experiment`` row), runs each
child (in-process, sequential by default), ranks them globally AND within demand
segments, assembles a per-segment composite, gates everything against current
production, and writes the recommendation back to ``champion_sweep``.

Designed to be called as a subprocess by the job engine::

    python scripts/ml/run_champion_sweep.py --sweep-id <int>

The sweep config (mode, segment_axis, objective, grid_spec, parallel, baseline)
is read from the ``champion_sweep`` DB row. See spec
``docs/specs/02-forecasting/30-champion-strategy-sweep.md``.

No new champion *selection* math lives here — children reuse
``run_champion_experiment.run_experiment`` and the strategy registry; the sweep
only adds grid expansion, ranking, per-segment slicing, and composite assembly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402 — after sys.path bootstrap
from common.core.utils import (  # noqa: E402
    get_competing_model_ids,
    load_config,
    load_forecast_pipeline_config,
)
from common.ml.champion import compute_strategy_accuracy  # noqa: E402
from common.ml.champion.segment import _classify_demand_segments  # noqa: E402
from scripts.ml.run_champion_experiment import run_experiment  # noqa: E402
from scripts.ml.run_champion_selection import load_monthly_errors_df  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DFU_COLS = ["item_id", "customer_group", "loc"]
_TEMPLATES_PATH = ROOT / "config" / "forecasting" / "champion_experiment_templates.yaml"
_WINNERS_DIR = ROOT / "data" / "champion"

# Syntetos-Boylan defaults — must match common/ml/champion/segment.py strategy_per_segment.
_ADI_THRESHOLD = 1.32
_CV2_THRESHOLD = 0.49


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _sweep_cfg() -> dict[str, Any]:
    """Read the ``sweep:`` block from the forecast pipeline config (with defaults)."""
    try:
        cfg = load_forecast_pipeline_config()
    except (FileNotFoundError, KeyError, ValueError):
        logger.warning("Could not load forecast pipeline config; using sweep defaults", exc_info=True)
        cfg = {}
    sweep = (cfg or {}).get("sweep", {}) or {}
    return {
        "max_candidates": int(sweep.get("max_candidates", 24)),
        "robust_lambda": float(sweep.get("robust_lambda", 0.5)),
        "robust_mu": float(sweep.get("robust_mu", 0.25)),
        "min_segment_dfus": int(sweep.get("min_segment_dfus", 30)),
    }


def _gate_cfg() -> dict[str, Any]:
    """Read ``champion.promote_gate`` — reused, never duplicated."""
    try:
        cfg = load_forecast_pipeline_config()
    except (FileNotFoundError, KeyError, ValueError):
        return {}
    champ = (cfg or {}).get("champion", {}) or {}
    return champ.get("promote_gate", {}) or {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(db: dict[str, Any]) -> psycopg.Connection:
    return psycopg.connect(**db, autocommit=True)


def _load_sweep(conn: psycopg.Connection, sweep_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT sweep_id, label, mode, segment_axis, objective, grid_spec,
               parallel, baseline_experiment_id, status
        FROM champion_sweep
        WHERE sweep_id = %s
        """,
        (sweep_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Champion sweep {sweep_id} not found")
    cols = [
        "sweep_id", "label", "mode", "segment_axis", "objective", "grid_spec",
        "parallel", "baseline_experiment_id", "status",
    ]
    d = dict(zip(cols, row, strict=False))
    if isinstance(d["grid_spec"], str):
        d["grid_spec"] = json.loads(d["grid_spec"])
    return d


def _set_running(conn: psycopg.Connection, sweep_id: int, candidate_count: int) -> None:
    conn.execute(
        "UPDATE champion_sweep SET status = 'running', started_at = NOW(), "
        "candidate_count = %s WHERE sweep_id = %s",
        (candidate_count, sweep_id),
    )


def _bump_completed(conn: psycopg.Connection, sweep_id: int) -> None:
    conn.execute(
        "UPDATE champion_sweep SET completed_count = completed_count + 1 WHERE sweep_id = %s",
        (sweep_id,),
    )


def _set_failed(conn: psycopg.Connection, sweep_id: int, error: str, runtime: float) -> None:
    conn.execute(
        """
        UPDATE champion_sweep
        SET status = 'failed', completed_at = NOW(),
            notes = COALESCE(notes, '') || %s, runtime_seconds = %s
        WHERE sweep_id = %s
        """,
        (f"\n\nERROR:\n{error}", round(runtime, 1), sweep_id),
    )


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def _config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the parts that define an experiment's behaviour."""
    key = {
        "strategy": cfg.get("strategy"),
        "strategy_params": cfg.get("strategy_params") or {},
        "models": sorted(cfg.get("models") or []),
        "metric": cfg.get("metric"),
        "lag_mode": cfg.get("lag_mode"),
        "min_sku_rows": cfg.get("min_sku_rows"),
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def _load_templates() -> dict[str, dict[str, Any]]:
    raw = load_config("forecasting/champion_experiment_templates") or {}
    out: dict[str, dict[str, Any]] = {}
    for t in raw.get("templates", []) or []:
        if isinstance(t, dict) and t.get("id"):
            out[t["id"]] = t
    return out


def _resolve_template(tpl: dict[str, Any], base_champion: dict[str, Any]) -> dict[str, Any]:
    """Turn a template into a concrete (strategy, params) config.

    ``source: pipeline_config`` templates inherit the live production
    champion settings so the baseline always reflects what's running.
    """
    if tpl.get("source") == "pipeline_config":
        return {
            "strategy": base_champion.get("strategy", "rolling"),
            "strategy_params": dict(base_champion.get("strategy_params") or {}),
        }
    return {
        "strategy": tpl.get("strategy", "expanding"),
        "strategy_params": dict(tpl.get("strategy_params") or {}),
    }


def expand_grid(
    grid_spec: dict[str, Any],
    *,
    base_champion: dict[str, Any],
    default_models: list[str],
    default_metric: str,
    default_lag: str,
    default_min_sku: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Expand a grid_spec into a deduped list of concrete experiment configs.

    Two forms: explicit ``configs`` (full bodies) or ``templates`` x ``models_variants``
    x ``metric``. Segmentation is NOT an axis — it is post-hoc slicing — so the
    candidate count is independent of segment_axis. Raises ValueError above the cap.
    """
    candidates: list[dict[str, Any]] = []

    if grid_spec.get("configs"):
        for raw in grid_spec["configs"]:
            candidates.append({
                "strategy": raw.get("strategy", "expanding"),
                "strategy_params": dict(raw.get("strategy_params") or {}),
                "models": list(raw.get("models") or default_models),
                "metric": raw.get("metric", default_metric),
                "lag_mode": str(raw.get("lag_mode", default_lag)),
                "min_sku_rows": int(raw.get("min_sku_rows", default_min_sku)),
                "label": raw.get("label"),
            })
    else:
        templates = _load_templates()
        tpl_ids = grid_spec.get("templates") or []
        models_variants = grid_spec.get("models_variants") or [default_models]
        metrics = grid_spec.get("metric") or [default_metric]
        for tpl_id in tpl_ids:
            tpl = templates.get(tpl_id)
            if tpl is None:
                logger.warning("Unknown template '%s' — skipping", tpl_id)
                continue
            resolved = _resolve_template(tpl, base_champion)
            for models in models_variants:
                for metric in metrics:
                    candidates.append({
                        "strategy": resolved["strategy"],
                        "strategy_params": dict(resolved["strategy_params"]),
                        "models": list(models),
                        "metric": metric,
                        "lag_mode": str(default_lag),
                        "min_sku_rows": int(default_min_sku),
                        "label": tpl.get("label", tpl_id),
                    })

    # Dedup by config_hash (keep first).
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in candidates:
        h = _config_hash(c)
        if h in seen:
            continue
        seen.add(h)
        c["config_hash"] = h
        deduped.append(c)

    if not deduped:
        raise ValueError("grid_spec expanded to zero candidates")
    if len(deduped) > max_candidates:
        raise ValueError(
            f"grid_spec expands to {len(deduped)} candidates, exceeding the "
            f"max_candidates cap of {max_candidates}. Reduce templates/variants."
        )
    return deduped


def _validate_candidate_models(
    candidates: list[dict[str, Any]],
    allowed_models: list[str],
) -> None:
    """Fail before creating children when a stored sweep contains retired models."""
    allowed = set(allowed_models)
    for index, candidate in enumerate(candidates):
        models = candidate.get("models")
        if not isinstance(models, list) or any(not isinstance(model_id, str) for model_id in models):
            raise ValueError(f"Sweep candidate {index + 1} models must be a list of model ID strings")
        unsupported = sorted(set(models) - allowed)
        if unsupported:
            raise ValueError(
                f"Sweep candidate {index + 1} contains unsupported champion model(s) "
                f"{unsupported}; valid competing models are {allowed_models}"
            )
        if len(models) < 2 or len(models) != len(set(models)):
            raise ValueError(
                f"Sweep candidate {index + 1} must contain at least 2 distinct competing models"
            )


# ---------------------------------------------------------------------------
# Child experiment lifecycle
# ---------------------------------------------------------------------------

def _find_completed_duplicate(conn: psycopg.Connection, cfg: dict[str, Any]) -> int | None:
    """Return the id of an existing completed experiment with the same behaviour, if any.

    Matches on strategy + params + models + metric + lag_mode + min_sku_rows so a
    foundation-heavy config isn't re-run when an identical one already succeeded.
    """
    rows = conn.execute(
        """
        SELECT experiment_id, strategy, strategy_params, models, metric, lag_mode, min_sku_rows
        FROM champion_experiment
        WHERE status = 'completed' AND strategy = %s AND metric = %s
              AND lag_mode = %s AND min_sku_rows = %s
        ORDER BY completed_at DESC
        """,
        (cfg["strategy"], cfg["metric"], cfg["lag_mode"], cfg["min_sku_rows"]),
    ).fetchall()
    target_hash = cfg["config_hash"]
    for r in rows:
        params = r[2] if isinstance(r[2], dict) else json.loads(r[2] or "{}")
        models = r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]")
        existing = {
            "strategy": r[1], "strategy_params": params, "models": models,
            "metric": r[4], "lag_mode": r[5], "min_sku_rows": r[6],
        }
        if _config_hash(existing) == target_hash:
            return int(r[0])
    return None


def _insert_child_experiment(
    conn: psycopg.Connection, sweep_label: str, cfg: dict[str, Any],
) -> int:
    """Create a queued champion_experiment row for one candidate."""
    label = cfg.get("label") or cfg["strategy"]
    row = conn.execute(
        """
        INSERT INTO champion_experiment
            (label, strategy, strategy_params, models, metric, lag_mode, min_sku_rows,
             cluster_experiment_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s,
                (SELECT experiment_id FROM cluster_experiment
                 WHERE is_promoted ORDER BY promoted_at DESC LIMIT 1))
        RETURNING experiment_id
        """,
        (
            f"[sweep] {sweep_label}: {label}",
            cfg["strategy"], json.dumps(cfg["strategy_params"]),
            json.dumps(cfg["models"]), cfg["metric"], cfg["lag_mode"], cfg["min_sku_rows"],
        ),
    ).fetchone()
    return int(row[0])


def _link_member(
    conn: psycopg.Connection, sweep_id: int, experiment_id: int,
    config_hash: str, *, is_composite: bool = False, skipped_duplicate: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO champion_sweep_member
            (sweep_id, experiment_id, config_hash, is_composite, skipped_duplicate)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (sweep_id, experiment_id) DO NOTHING
        """,
        (sweep_id, experiment_id, config_hash, is_composite, skipped_duplicate),
    )


def _experiment_results(conn: psycopg.Connection, experiment_id: int) -> dict[str, Any]:
    """Read a completed child's headline + per-lag/per-month accuracy for scoring."""
    row = conn.execute(
        """
        SELECT status, champion_accuracy, ceiling_accuracy, gap_bps, strategy, strategy_params
        FROM champion_experiment WHERE experiment_id = %s
        """,
        (experiment_id,),
    ).fetchone()
    lag_acc = [
        float(r[0]) for r in conn.execute(
            "SELECT champion_accuracy FROM champion_experiment_lag "
            "WHERE experiment_id = %s AND champion_accuracy IS NOT NULL",
            (experiment_id,),
        ).fetchall()
    ]
    month_acc = [
        float(r[0]) for r in conn.execute(
            "SELECT champion_accuracy FROM champion_experiment_month "
            "WHERE experiment_id = %s AND champion_accuracy IS NOT NULL",
            (experiment_id,),
        ).fetchall()
    ]
    params = row[5] if isinstance(row[5], (dict, type(None))) else json.loads(row[5] or "{}")
    return {
        "status": row[0],
        "champion_accuracy": float(row[1]) if row[1] is not None else None,
        "ceiling_accuracy": float(row[2]) if row[2] is not None else None,
        "gap_bps": float(row[3]) if row[3] is not None else None,
        "strategy": row[4],
        "strategy_params": params or {},
        "lag_acc": lag_acc,
        "month_acc": month_acc,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def objective_score(
    res: dict[str, Any], objective: str, *, lam: float, mu: float,
) -> float | None:
    """Score a result on the chosen objective. Higher is always better."""
    acc = res.get("champion_accuracy")
    if acc is None:
        return None
    if objective == "accuracy":
        return round(acc, 4)
    if objective == "gap_to_ceiling":
        gap = res.get("gap_bps")
        return round(-gap, 4) if gap is not None else round(acc, 4)
    # robust (default): mean lag accuracy penalised by lag + month dispersion
    lag_acc = res.get("lag_acc") or [acc]
    mean_lag = sum(lag_acc) / len(lag_acc)
    score = mean_lag - lam * _stdev(lag_acc) - mu * _stdev(res.get("month_acc") or [])
    return round(score, 4)


def segment_score(
    accuracy: float | None, objective: str,
) -> float | None:
    """Segment-restricted score. With only a point accuracy available per slice,
    accuracy and robust collapse to accuracy; gap_to_ceiling is not meaningful
    per-slice so it also falls back to accuracy."""
    return round(accuracy, 4) if accuracy is not None else None


# ---------------------------------------------------------------------------
# Promote gate (reuses champion.promote_gate policy, evaluated on accuracy)
# ---------------------------------------------------------------------------

def gate_eligible(
    candidate_acc: float | None, baseline_acc: float | None, gate: dict[str, Any],
) -> bool:
    """Lightweight gate check for sweep ranking — mirrors backtest_management's
    WAPE-improvement gate using accuracy as the proxy (accuracy = 100 - WAPE, so a
    WAPE improvement is an accuracy gain). Coverage is enforced at promote time by
    the real endpoint. Disabled/absent gate or no baseline → eligible."""
    if not gate or not gate.get("enabled", False):
        return True
    if candidate_acc is None:
        return False
    if baseline_acc is None:
        return True
    # WAPE = 100 - accuracy. Relative WAPE improvement vs baseline.
    base_wape = 100.0 - baseline_acc
    cand_wape = 100.0 - candidate_acc
    if base_wape <= 0:
        return cand_wape <= 0
    rel_improvement_pct = (base_wape - cand_wape) / base_wape * 100.0
    return rel_improvement_pct >= float(gate.get("min_wape_improvement_pct", 0.0))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _segment_map(
    db: dict[str, Any], monthly_errors_df: pd.DataFrame, segment_axis: str,
) -> dict[tuple[str, str, str], str]:
    """Map each DFU -> segment label on the chosen axis."""
    if segment_axis == "demand_class":
        return _classify_demand_segments(monthly_errors_df, _ADI_THRESHOLD, _CV2_THRESHOLD)
    # ml_cluster / abc_xyz: diagnostic-only axes.
    out: dict[tuple[str, str, str], str] = {}
    with psycopg.connect(**db) as conn:
        if segment_axis == "ml_cluster":
            rows = conn.execute("""
                SELECT d.item_id, d.customer_group, d.loc, ca.ml_cluster
                FROM dim_sku d
                LEFT JOIN current_sku_cluster_assignment ca
                       ON ca.sku_ck = d.sku_ck
            """).fetchall()
        else:
            rows = conn.execute(
                "SELECT item_id, customer_group, loc, abc_xyz_segment FROM dim_sku"
            ).fetchall()
    for item_id, cust, loc, seg in rows:
        out[(str(item_id), str(cust), str(loc))] = str(seg) if seg is not None else "unassigned"
    return out


def _winners_csv(experiment_id: int) -> Path:
    return _WINNERS_DIR / f"experiment_{experiment_id}_winners.csv"


def _load_winners(experiment_id: int) -> pd.DataFrame | None:
    path = _winners_csv(experiment_id)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"item_id": str, "customer_group": str, "loc": str})
    return df


def _segment_accuracy(winners_df: pd.DataFrame, dfus: set[tuple[str, str, str]]) -> tuple[float | None, int]:
    """Restrict a winners frame to a segment's DFUs and compute accuracy + DFU count."""
    if winners_df is None or winners_df.empty:
        return None, 0
    keys = list(zip(winners_df["item_id"], winners_df["customer_group"], winners_df["loc"], strict=False))
    mask = pd.Series([k in dfus for k in keys], index=winners_df.index)
    sliced = winners_df[mask]
    if sliced.empty:
        return None, 0
    n_dfus = sliced[_DFU_COLS].drop_duplicates().shape[0]
    return compute_strategy_accuracy(sliced)["accuracy_pct"], n_dfus


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(sweep_id: int) -> None:
    """Run one champion strategy sweep end-to-end."""
    load_dotenv(ROOT / ".env")
    db = get_db_params()
    cfg = _sweep_cfg()
    gate = _gate_cfg()
    conn: psycopg.Connection | None = None
    t0 = time.time()

    try:
        conn = _connect(db)
        sweep = _load_sweep(conn, sweep_id)
        mode = sweep["mode"]
        objective = sweep["objective"]
        segment_axis = sweep["segment_axis"]
        logger.info(
            "Champion sweep #%d: mode=%s, axis=%s, objective=%s",
            sweep_id, mode, segment_axis, objective,
        )

        pipe = load_forecast_pipeline_config() or {}
        base_champion = (pipe.get("champion") or {})
        competing_models = get_competing_model_ids()
        default_models = base_champion.get("models") or competing_models
        default_metric = base_champion.get("metric", "wape")
        default_lag = base_champion.get("lag", "execution")
        default_min_sku = int(base_champion.get("min_sku_rows", 3))

        # 1. Expand grid
        candidates = expand_grid(
            sweep["grid_spec"],
            base_champion=base_champion,
            default_models=default_models,
            default_metric=default_metric,
            default_lag=default_lag,
            default_min_sku=default_min_sku,
            max_candidates=cfg["max_candidates"],
        )
        _validate_candidate_models(candidates, competing_models)
        _set_running(conn, sweep_id, len(candidates))
        logger.info("Expanded to %d candidate configs", len(candidates))

        # 2. Run children (sequential; reuse completed duplicates)
        member_ids: list[int] = []
        for c in candidates:
            dup_id = _find_completed_duplicate(conn, c)
            if dup_id is not None:
                logger.info("Reusing completed experiment #%d for %s", dup_id, c["config_hash"])
                _link_member(conn, sweep_id, dup_id, c["config_hash"], skipped_duplicate=True)
                member_ids.append(dup_id)
                _bump_completed(conn, sweep_id)
                continue
            exp_id = _insert_child_experiment(conn, sweep["label"], c)
            _link_member(conn, sweep_id, exp_id, c["config_hash"])
            logger.info("Running child experiment #%d (%s)...", exp_id, c["strategy"])
            try:
                run_experiment(exp_id)  # in-process; writes its own results + winners CSV
            except SystemExit:
                logger.warning("Child experiment #%d exited non-zero (marked failed)", exp_id)
            member_ids.append(exp_id)
            _bump_completed(conn, sweep_id)

        # 3. Resolve baseline (explicit, else current promoted experiment)
        baseline_id = sweep["baseline_experiment_id"]
        if baseline_id is None:
            r = conn.execute(
                "SELECT experiment_id FROM champion_experiment "
                "WHERE is_promoted = TRUE ORDER BY promoted_at DESC LIMIT 1"
            ).fetchone()
            baseline_id = int(r[0]) if r else None
        baseline_acc = None
        if baseline_id is not None:
            br = _experiment_results(conn, baseline_id)
            baseline_acc = br.get("champion_accuracy")

        # 4. Global ranking
        scored: list[dict[str, Any]] = []
        for exp_id in member_ids:
            res = _experiment_results(conn, exp_id)
            if res["status"] != "completed":
                continue
            score = objective_score(res, objective, lam=cfg["robust_lambda"], mu=cfg["robust_mu"])
            scored.append({
                "experiment_id": exp_id,
                "score": score,
                "accuracy": res["champion_accuracy"],
                "gate_eligible": gate_eligible(res["champion_accuracy"], baseline_acc, gate),
                "res": res,
            })
        scored.sort(key=lambda s: (s["score"] is not None, s["score"] or float("-inf")), reverse=True)
        for rank, s in enumerate(scored, start=1):
            conn.execute(
                "UPDATE champion_sweep_member SET global_rank = %s, global_score = %s, "
                "gate_eligible = %s WHERE sweep_id = %s AND experiment_id = %s",
                (rank, s["score"], s["gate_eligible"], sweep_id, s["experiment_id"]),
            )
        best_global = scored[0] if scored else None

        # 5. Per-segment slicing + composite (skipped in pure global mode)
        composite_id: int | None = None
        if mode in ("per_segment", "both") and scored:
            composite_id = _build_segments_and_composite(
                conn, db, sweep_id, sweep, scored, segment_axis, objective, cfg, gate, baseline_acc,
            )

        # 6. Recommend: higher-scoring gate-passing of {best global, composite}
        candidates_for_rec: list[dict[str, Any]] = []
        if best_global is not None:
            candidates_for_rec.append(best_global)
        if composite_id is not None:
            comp_res = _experiment_results(conn, composite_id)
            comp_score = objective_score(comp_res, objective, lam=cfg["robust_lambda"], mu=cfg["robust_mu"])
            comp_gate = gate_eligible(comp_res["champion_accuracy"], baseline_acc, gate)
            candidates_for_rec.append({
                "experiment_id": composite_id, "score": comp_score,
                "accuracy": comp_res["champion_accuracy"], "gate_eligible": comp_gate,
            })
            # Persist the composite's member scoring + a rank relative to the
            # global candidates (the ranking loop in step 5 ran before it existed).
            comp_rank = 1 + sum(
                1 for s in scored
                if s["score"] is not None and (comp_score is None or s["score"] > comp_score)
            )
            conn.execute(
                "UPDATE champion_sweep_member SET global_rank = %s, global_score = %s, "
                "gate_eligible = %s WHERE sweep_id = %s AND experiment_id = %s",
                (comp_rank, comp_score, comp_gate, sweep_id, composite_id),
            )
        recommended = _pick_recommendation(candidates_for_rec)

        runtime = time.time() - t0
        conn.execute(
            """
            UPDATE champion_sweep
            SET status = 'completed', completed_at = NOW(), runtime_seconds = %s,
                best_global_experiment_id = %s, composite_experiment_id = %s,
                recommended_experiment_id = %s, recommended_score = %s,
                recommended_gate_eligible = %s
            WHERE sweep_id = %s
            """,
            (
                round(runtime, 1),
                best_global["experiment_id"] if best_global else None,
                composite_id,
                recommended["experiment_id"] if recommended else None,
                recommended["score"] if recommended else None,
                recommended["gate_eligible"] if recommended else None,
                sweep_id,
            ),
        )
        logger.info(
            "Champion sweep #%d completed in %.1fs — %d ranked, recommended=%s",
            sweep_id, runtime, len(scored),
            recommended["experiment_id"] if recommended else None,
        )

    except Exception as exc:
        runtime = time.time() - t0
        logger.exception("Champion sweep #%d failed: %s", sweep_id, exc)
        if conn is not None:
            try:
                _set_failed(conn, sweep_id, traceback.format_exc(), runtime)
            except psycopg.Error as db_exc:
                logger.error("Failed to mark sweep failed: %s", db_exc)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


def _pick_recommendation(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer the highest-scoring gate-eligible option; else the highest-scoring."""
    if not options:
        return None
    eligible = [o for o in options if o.get("gate_eligible")]
    pool = eligible or options
    return max(pool, key=lambda o: (o["score"] is not None, o["score"] or float("-inf")))


def _build_segments_and_composite(
    conn: psycopg.Connection,
    db: dict[str, Any],
    sweep_id: int,
    sweep: dict[str, Any],
    scored: list[dict[str, Any]],
    segment_axis: str,
    objective: str,
    cfg: dict[str, Any],
    gate: dict[str, Any],
    baseline_acc: float | None,
) -> int | None:
    """Score every candidate within each segment, pick per-segment winners (with the
    min_segment_dfus fallback to the global winner), and synthesise a composite
    champion as a ``per_segment`` config — natively reproducible in production."""
    # Need monthly errors to classify demand segments (axis=demand_class).
    default_models = sweep_keys_models(conn, scored)
    try:
        monthly = load_monthly_errors_df(db, default_models, "execution")
    except (psycopg.Error, ValueError, KeyError):
        logger.warning("Could not load monthly errors for segmentation; skipping composite", exc_info=True)
        return None
    seg_map = _segment_map(db, monthly, segment_axis)
    if not seg_map:
        logger.warning("Empty segment map; skipping composite")
        return None

    # DFU sets per segment.
    seg_to_dfus: dict[str, set[tuple[str, str, str]]] = {}
    for dfu, seg in seg_map.items():
        seg_to_dfus.setdefault(seg, set()).add(dfu)

    best_global = scored[0]
    min_dfus = cfg["min_segment_dfus"]

    # Score each candidate per segment; collect rows + per-segment winner.
    seg_winner: dict[str, dict[str, Any]] = {}
    seg_rows: dict[str, list[dict[str, Any]]] = {}
    for seg, dfus in seg_to_dfus.items():
        # Anti-overfit guard up front: a segment too small to trust never gets to
        # pick a specialist — it falls back to the global winner. We skip scoring
        # (and emit no per-segment rows) so the UI doesn't show a phantom
        # acc=None / dfus=0 "winner" for near-empty segments (e.g. lumpy on
        # dense-demand data where only a handful of DFUs are intermittent/lumpy).
        if len(dfus) < min_dfus:
            seg_winner[seg] = {
                "experiment_id": best_global["experiment_id"],
                "fallback": True, "reason": "below_min_segment_dfus", "n_dfus": len(dfus),
            }
            logger.info(
                "Segment '%s' has %d DFUs (< min_segment_dfus=%d) — using global winner",
                seg, len(dfus), min_dfus,
            )
            continue
        for s in scored:
            winners_df = _load_winners(s["experiment_id"])
            acc, n = _segment_accuracy(winners_df, dfus)
            seg_rows.setdefault(seg, []).append({
                "experiment_id": s["experiment_id"], "accuracy": acc, "n_dfus": n,
                "score": segment_score(acc, objective),
            })
        ranked = sorted(
            seg_rows[seg],
            key=lambda r: (r["score"] is not None, r["score"] or float("-inf")),
            reverse=True,
        )
        for rank, r in enumerate(ranked, start=1):
            conn.execute(
                """
                INSERT INTO champion_sweep_segment_score
                    (sweep_id, experiment_id, segment, n_dfus, accuracy, score, segment_rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sweep_id, experiment_id, segment) DO UPDATE SET
                    n_dfus = EXCLUDED.n_dfus, accuracy = EXCLUDED.accuracy,
                    score = EXCLUDED.score, segment_rank = EXCLUDED.segment_rank
                """,
                (sweep_id, r["experiment_id"], seg, r["n_dfus"], r["accuracy"], r["score"], rank),
            )
        top = ranked[0] if ranked else None
        # A scored segment can still fall back if its best candidate has no usable score.
        if top is None or top["score"] is None:
            seg_winner[seg] = {"experiment_id": best_global["experiment_id"], "fallback": True}
        else:
            seg_winner[seg] = {"experiment_id": top["experiment_id"], "fallback": False}

    # Composite is only PROMOTABLE on the demand_class axis (maps to per_segment).
    if segment_axis != "demand_class":
        logger.info("segment_axis=%s is diagnostic-only; no promotable composite assembled", segment_axis)
        return None

    return _assemble_composite(conn, sweep, seg_winner, seg_to_dfus)


def sweep_keys_models(conn: psycopg.Connection, scored: list[dict[str, Any]]) -> list[str]:
    """Union of model sets across the scored children (for segmentation data load)."""
    models: set[str] = set()
    for s in scored:
        row = conn.execute(
            "SELECT models FROM champion_experiment WHERE experiment_id = %s",
            (s["experiment_id"],),
        ).fetchone()
        if row:
            ms = row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
            models.update(ms)
    return sorted(models) or get_competing_model_ids()


def _assemble_composite(
    conn: psycopg.Connection,
    sweep: dict[str, Any],
    seg_winner: dict[str, dict[str, Any]],
    seg_to_dfus: dict[str, set[tuple[str, str, str]]],
) -> int | None:
    """Concatenate each segment's winning child's winners (restricted to that segment)
    into one composite winners CSV, and record it as a per_segment experiment whose
    params encode the discovered segment->strategy map (production-reproducible)."""
    frames: list[pd.DataFrame] = []
    segment_strategy_map: dict[str, dict[str, Any]] = {}
    for seg, win in seg_winner.items():
        winners_df = _load_winners(win["experiment_id"])
        if winners_df is None or winners_df.empty:
            continue
        dfus = seg_to_dfus.get(seg, set())
        keys = list(zip(winners_df["item_id"], winners_df["customer_group"], winners_df["loc"], strict=False))
        mask = pd.Series([k in dfus for k in keys], index=winners_df.index)
        frames.append(winners_df[mask])
        # Record the winner's strategy+params for this segment (for reproducibility).
        res = _experiment_results(conn, win["experiment_id"])
        entry = {"strategy": res["strategy"], **(res["strategy_params"] or {})}
        segment_strategy_map[seg] = entry

    if not frames:
        logger.warning("Composite assembly produced no rows; skipping")
        return None
    composite_df = pd.concat(frames, ignore_index=True)

    # Create a per_segment experiment row to carry the composite (status=completed).
    # strategy_params encode the discovered segment->strategy map so the next
    # production champion run reproduces the composite natively.
    stats = compute_strategy_accuracy(composite_df)
    params = {"segment_strategy_map": segment_strategy_map}
    composite_models = sweep_keys_models(conn, [
        {"experiment_id": w["experiment_id"]} for w in seg_winner.values()
    ])
    row = conn.execute(
        """
        INSERT INTO champion_experiment
            (label, strategy, strategy_params, models, metric, lag_mode, min_sku_rows,
             status, completed_at, champion_accuracy, n_dfu_months,
             n_champions, cluster_experiment_id)
        VALUES (%s, 'per_segment', %s, %s, %s, %s, %s, 'completed', NOW(), %s, %s, %s,
                (SELECT experiment_id FROM cluster_experiment
                 WHERE is_promoted ORDER BY promoted_at DESC LIMIT 1))
        RETURNING experiment_id
        """,
        (
            f"[sweep] {sweep['label']}: per-segment composite",
            json.dumps(params),
            json.dumps(composite_models),
            "wape", "execution", 3,
            stats["accuracy_pct"],
            int(composite_df[_DFU_COLS].drop_duplicates().shape[0]),
            len(composite_df),
        ),
    ).fetchone()
    composite_id = int(row[0])
    cached_composite = composite_df.copy()
    if "source_mix" in cached_composite.columns:
        cached_composite["source_mix"] = cached_composite["source_mix"].apply(
            lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value
        )
    cached_composite.to_csv(_winners_csv(composite_id), index=False)
    _link_member(conn, sweep["sweep_id"], composite_id, f"composite_{sweep['sweep_id']}", is_composite=True)
    logger.info("Assembled composite experiment #%d (acc=%s%%)", composite_id, stats["accuracy_pct"])
    return composite_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a champion strategy sweep")
    parser.add_argument("--sweep-id", type=int, required=True, help="champion_sweep row id")
    args = parser.parse_args()
    run_sweep(args.sweep_id)


if __name__ == "__main__":
    main()
