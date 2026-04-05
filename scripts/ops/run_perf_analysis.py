"""Performance analysis CLI — profile scripts, API queries, and pipelines.

Usage:
    python scripts/ops/run_perf_analysis.py --mode script --script compute_safety_stock
    python scripts/ops/run_perf_analysis.py --mode api
    python scripts/ops/run_perf_analysis.py --mode pipeline
    python scripts/ops/run_perf_analysis.py --mode report --output data/perf_reports/full.json
"""

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.utils import load_config  # noqa: E402
from common.db import get_db_params  # noqa: E402
from common.services.perf_profiler import (  # noqa: E402
    PerfReport,
    QuerySummary,
    SectionMetrics,
    auto_wrap_connections,
    ensure_rollback,
    generate_report,
    persist_report,
    profile_script,
    profiled_section,
    wrap_connection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_perf_config(override_path: str | None = None) -> dict:
    """Load perf_config.yaml or return empty dict on missing file."""
    if override_path:
        import yaml  # type: ignore[import-untyped]

        with open(override_path) as fh:
            return yaml.safe_load(fh) or {}
    try:
        return load_config("perf_config.yaml")
    except FileNotFoundError:
        logger.warning("perf_config.yaml not found — using defaults")
        return {}


def _auto_load_script_config(script_name: str) -> dict | None:
    """Try to auto-load a YAML config for a script by naming convention.

    Maps script names to config files:
      compute_eoq -> config/eoq_config.yaml
      compute_demand_variability -> config/forecast_domain_config.yaml
    """
    import yaml  # type: ignore[import-untyped]

    # Explicit overrides for scripts whose config doesn't match naming convention
    _CONFIG_MAP = {
        "compute_demand_variability": "forecast_domain_config.yaml",
        "compute_lead_time_variability": "inventory_planning_config.yaml",
        "compute_service_level_actuals": "service_level_config.yaml",
        "run_ss_simulation": "inventory_planning_config.yaml",
        "compute_financial_plan": "financial_plan_config.yaml",
    }
    if script_name in _CONFIG_MAP:
        path = ROOT / "config" / _CONFIG_MAP[script_name]
        if path.exists():
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            logger.info("Auto-loaded config: %s", path)
            return cfg

    # Strip common prefixes to derive config name
    name = script_name
    for prefix in ("compute_", "generate_", "run_", "apply_", "detect_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    candidates = [
        ROOT / "config" / f"{name}_config.yaml",
        ROOT / "config" / f"{script_name}_config.yaml",
        ROOT / "config" / f"{name.replace('_', '-')}_config.yaml",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            logger.info("Auto-loaded config: %s", path)
            return cfg
    logger.info("No config found for '%s' — tried %s", script_name,
                [str(p.name) for p in candidates])
    return None


def _default_output_path(mode: str) -> Path:
    """Generate timestamped output path under data/perf_reports/."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / "perf_reports" / f"{mode}_{ts}.json"


def _get_connection(readonly: bool = True):
    """Open a psycopg3 connection, optionally wrapped for profiling.

    Returns (conn, wrapped) or (None, False) if DB is unreachable.
    """
    try:
        import psycopg

        params = get_db_params()
        conn = psycopg.connect(**params)
        if readonly:
            conn = wrap_connection(conn, readonly=True)
        return conn, True
    except ImportError:
        logger.error("psycopg not installed — DB profiling unavailable")
        return None, False
    except Exception as exc:
        logger.warning("Cannot connect to DB (%s) — DB profiling unavailable", exc)
        return None, False


def _output_report(report: PerfReport, output_path: Path) -> None:
    """Write report to JSON, log console summary, and persist to DB."""
    logger.info(report.to_console())
    report.to_json(output_path)
    logger.info("JSON report → %s", output_path)

    # Persist to perf_run tables (separate writable connection)
    try:
        import psycopg
        params = get_db_params()
        with psycopg.connect(**params) as write_conn:
            run_id = persist_report(report, write_conn)
            logger.info("Persisted to DB → run_id=%d", run_id)
    except ImportError:
        logger.debug("psycopg not available — skipping DB persistence")
    except Exception as exc:
        logger.warning("DB persistence failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Mode: script
# ---------------------------------------------------------------------------


def _run_script_mode(
    script_name: str,
    config: dict,
    *,
    readonly: bool = True,
    include_suggestions: bool = True,
) -> PerfReport:
    """Profile a target script by name."""
    presets = config.get("script_presets", {}).get(script_name, {})
    preset_args = presets.get("args", [])
    logger.info(
        "Profiling script '%s' (preset args: %s, readonly: %s)",
        script_name,
        preset_args,
        readonly,
    )

    with profile_script(script_name) as collector:
        # --- Auto-wrap all psycopg.connect() calls for query tracking ---
        with auto_wrap_connections(readonly=readonly):
            # --- DB connection setup ---
            with profiled_section("db_connection_setup"):
                conn, db_available = _get_connection(readonly=readonly)

            # --- Sanity query ---
            if conn is not None:
                with profiled_section("sanity_query"):
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1")
                            cur.fetchone()
                    except Exception as exc:
                        logger.warning("Sanity query failed: %s", exc)
            else:
                logger.info("DB not available — skipping query profiling")

            # --- Dynamic import ---
            with profiled_section("script_import"):
                module = None
                try:
                    module = importlib.import_module(f"scripts.{script_name}")
                except ImportError:
                    # Scripts may live under a subdirectory
                    _found = False
                    for subpkg in ("etl", "ml", "forecasting", "inventory", "ops", "ai"):
                        try:
                            module = importlib.import_module(
                                f"scripts.{subpkg}.{script_name}"
                            )
                            _found = True
                            break
                        except ImportError:
                            continue
                    if not _found:
                        logger.error(
                            "Cannot import script '%s' — verify it exists under scripts/",
                            script_name,
                        )

            # --- Determine entry point ---
            custom_callable = presets.get("callable")
            callable_kwargs = presets.get("callable_kwargs", {})

            if module is not None and custom_callable:
                # Preset specifies a custom callable (e.g. compute_plan)
                fn = getattr(module, custom_callable, None)
                if fn is not None:
                    with profiled_section("script_run"):
                        try:
                            fn(**callable_kwargs)
                        except Exception as exc:
                            logger.warning(
                                "script.%s() raised: %s", custom_callable, exc
                            )
                else:
                    logger.error(
                        "Script '%s' has no '%s' function",
                        script_name,
                        custom_callable,
                    )
            elif module is not None and hasattr(module, "run"):
                with profiled_section("script_run"):
                    try:
                        if preset_args:
                            module.run(*preset_args)
                        else:
                            module.run()
                    except TypeError:
                        # run() may require a config dict — auto-load it
                        try:
                            import inspect
                            sig = inspect.signature(module.run)
                            params = list(sig.parameters.keys())
                            if params and params[0] == "config":
                                # Load config by convention: script name → config/<name>.yaml
                                cfg = _auto_load_script_config(script_name)
                                if cfg is not None:
                                    module.run(cfg)
                                else:
                                    logger.warning(
                                        "script.run(config) needed but no config found"
                                    )
                            else:
                                module.run()
                        except Exception as exc:
                            logger.warning("script.run() raised: %s", exc)
                    except Exception as exc:
                        logger.warning("script.run() raised: %s", exc)
            elif module is not None and hasattr(module, "main"):
                with profiled_section("script_run"):
                    # Patch sys.argv so argparse inside main() sees preset args
                    saved_argv = sys.argv
                    sys.argv = [script_name] + list(preset_args)
                    try:
                        module.main()
                    except SystemExit:
                        pass  # argparse may call sys.exit
                    except Exception as exc:
                        logger.warning("script.main() raised: %s", exc)
                    finally:
                        sys.argv = saved_argv
            elif module is not None:
                logger.info(
                    "Script '%s' has no run()/main() function — "
                    "profiling import only",
                    script_name,
                )

            # --- Cleanup ---
            if conn is not None:
                ensure_rollback(conn)
                try:
                    conn.close()
                except Exception:
                    pass

    return generate_report(
        collector, include_suggestions=include_suggestions, config=config
    )


# ---------------------------------------------------------------------------
# Mode: api
# ---------------------------------------------------------------------------


def _run_api_mode(
    config: dict,
    *,
    include_suggestions: bool = True,
) -> PerfReport:
    """Analyze API query performance from fact_query_performance table."""
    logger.info("Analyzing API query performance")

    with profile_script("api_analysis") as collector:
        conn, db_available = _get_connection(readonly=True)

        if conn is None:
            logger.warning(
                "DB not available — cannot analyze API query performance"
            )
            return generate_report(
                collector, include_suggestions=include_suggestions, config=config
            )

        with profiled_section("query_performance_analysis"):
            try:
                with conn.cursor() as cur:
                    # Check if tracking table exists
                    cur.execute(
                        "SELECT EXISTS ("
                        "  SELECT 1 FROM information_schema.tables "
                        "  WHERE table_name = 'fact_query_performance'"
                        ")"
                    )
                    row = cur.fetchone()
                    table_exists = row[0] if row else False

                    if not table_exists:
                        logger.info(
                            "fact_query_performance table does not exist — "
                            "query tracking is not configured"
                        )
                        section = collector.current_section
                        if section:
                            section.metadata["status"] = "table_not_found"
                    else:
                        # Aggregate recent query performance
                        cur.execute(
                            "SELECT "
                            "  COUNT(*) AS total_queries, "
                            "  AVG(duration_ms) AS avg_ms, "
                            "  PERCENTILE_CONT(0.95) WITHIN GROUP "
                            "    (ORDER BY duration_ms) AS p95_ms, "
                            "  PERCENTILE_CONT(0.99) WITHIN GROUP "
                            "    (ORDER BY duration_ms) AS p99_ms "
                            "FROM fact_query_performance "
                            "WHERE created_at > NOW() - INTERVAL '24 hours'"
                        )
                        stats = cur.fetchone()

                        if stats and stats[0] > 0:
                            section = collector.current_section
                            if section:
                                section.metadata.update({
                                    "total_queries": stats[0],
                                    "avg_response_ms": float(stats[1] or 0),
                                    "p95_ms": float(stats[2] or 0),
                                    "p99_ms": float(stats[3] or 0),
                                })
                            logger.info(
                                "API stats (24h): %d queries, avg %.0fms, "
                                "P95 %.0fms, P99 %.0fms",
                                stats[0],
                                stats[1] or 0,
                                stats[2] or 0,
                                stats[3] or 0,
                            )

                        # Slowest endpoints
                        cur.execute(
                            "SELECT endpoint, AVG(duration_ms) AS avg_ms, "
                            "  COUNT(*) AS calls "
                            "FROM fact_query_performance "
                            "WHERE created_at > NOW() - INTERVAL '24 hours' "
                            "GROUP BY endpoint "
                            "ORDER BY avg_ms DESC "
                            "LIMIT 10"
                        )
                        slow_endpoints = cur.fetchall()
                        if slow_endpoints:
                            section = collector.current_section
                            if section:
                                section.metadata["slowest_endpoints"] = [
                                    {
                                        "endpoint": r[0],
                                        "avg_ms": float(r[1]),
                                        "calls": r[2],
                                    }
                                    for r in slow_endpoints
                                ]
            except Exception as exc:
                logger.warning("API performance analysis failed: %s", exc)

        ensure_rollback(conn)
        try:
            conn.close()
        except Exception:
            pass

    return generate_report(
        collector, include_suggestions=include_suggestions, config=config
    )


# ---------------------------------------------------------------------------
# Mode: pipeline
# ---------------------------------------------------------------------------


def _run_pipeline_mode(
    config: dict,
    *,
    include_suggestions: bool = True,
) -> PerfReport:
    """Profile the ETL pipeline orchestrator."""
    logger.info("Profiling pipeline execution")

    with profile_script("pipeline") as collector:
        with profiled_section("pipeline_import"):
            try:
                pipeline_mod = importlib.import_module("scripts.etl.run_pipeline")
            except ImportError:
                logger.error(
                    "Cannot import scripts.etl.run_pipeline — "
                    "verify it exists"
                )
                return generate_report(
                    collector,
                    include_suggestions=include_suggestions,
                    config=config,
                )

        # Attempt to run pipeline in dry-run / report-only fashion
        if hasattr(pipeline_mod, "run"):
            with profiled_section("pipeline_run"):
                try:
                    result = pipeline_mod.run("--mode", "full", "--dry-run")
                    if isinstance(result, dict):
                        section = collector.current_section
                        if section:
                            section.metadata["domain_timings"] = result
                except TypeError:
                    try:
                        result = pipeline_mod.run()
                        if isinstance(result, dict):
                            section = collector.current_section
                            if section:
                                section.metadata["domain_timings"] = result
                    except Exception as exc:
                        logger.warning("pipeline.run() raised: %s", exc)
                except Exception as exc:
                    logger.warning("pipeline.run() raised: %s", exc)
        else:
            logger.info(
                "scripts.etl.run_pipeline has no run() function — "
                "profiling import and DB connectivity only"
            )

            conn, db_available = _get_connection(readonly=True)
            if conn is not None:
                with profiled_section("pipeline_db_check"):
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1")
                            cur.fetchone()
                    except Exception as exc:
                        logger.warning("Pipeline DB check failed: %s", exc)
                ensure_rollback(conn)
                try:
                    conn.close()
                except Exception:
                    pass

    return generate_report(
        collector, include_suggestions=include_suggestions, config=config
    )


# ---------------------------------------------------------------------------
# Mode: report (combined)
# ---------------------------------------------------------------------------


def _run_report_mode(
    config: dict,
    *,
    include_suggestions: bool = True,
) -> PerfReport:
    """Run all modes and combine into a single comprehensive report."""
    logger.info("Running comprehensive performance report (all modes)")

    all_sections: list[SectionMetrics] = []
    all_suggestions = []
    combined_qs = QuerySummary()
    total_wall = 0.0
    total_cpu = 0.0
    peak_mem = 0.0

    for mode_name, mode_fn in [
        ("api", lambda: _run_api_mode(config, include_suggestions=include_suggestions)),
        ("pipeline", lambda: _run_pipeline_mode(config, include_suggestions=include_suggestions)),
    ]:
        logger.info("--- Running sub-mode: %s ---", mode_name)
        try:
            sub_report = mode_fn()
            # Wrap each sub-report's sections under a parent section
            wrapper = SectionMetrics(
                name=f"mode_{mode_name}",
                wall_time_s=sub_report.total_wall_time_s,
                cpu_time_s=sub_report.total_cpu_time_s,
                memory_peak_mb=sub_report.peak_memory_mb,
                children=sub_report.sections,
            )
            all_sections.append(wrapper)
            all_suggestions.extend(sub_report.suggestions)

            # Accumulate totals
            total_wall += sub_report.total_wall_time_s
            total_cpu += sub_report.total_cpu_time_s
            peak_mem = max(peak_mem, sub_report.peak_memory_mb)
            combined_qs.total_queries += sub_report.query_summary.total_queries
            combined_qs.total_query_time_ms += sub_report.query_summary.total_query_time_ms
            if sub_report.query_summary.slowest_query_ms > combined_qs.slowest_query_ms:
                combined_qs.slowest_query_ms = sub_report.query_summary.slowest_query_ms
                combined_qs.slowest_query_sql = sub_report.query_summary.slowest_query_sql
            if sub_report.query_summary.n_plus_1_detected:
                combined_qs.n_plus_1_detected = True
            combined_qs.unbatched_inserts += sub_report.query_summary.unbatched_inserts
        except Exception as exc:
            logger.warning("Sub-mode '%s' failed: %s", mode_name, exc)

    return PerfReport(
        script_name="combined_report",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_wall_time_s=total_wall,
        total_cpu_time_s=total_cpu,
        peak_memory_mb=peak_mem,
        sections=all_sections,
        query_summary=combined_qs,
        suggestions=all_suggestions,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Performance analysis CLI for the Supply Chain Command Center"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("script", "api", "pipeline", "report"),
        help="Profiling mode",
    )
    parser.add_argument(
        "--script",
        default=None,
        help="Script name to profile (required for --mode script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path (default: auto-generated)",
    )
    parser.add_argument(
        "--no-suggest",
        action="store_true",
        help="Suppress performance suggestions",
    )
    parser.add_argument(
        "--no-readonly",
        action="store_true",
        help="Disable read-only DB mode (development only)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to perf_config.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.mode == "script" and not args.script:
        logger.error("--script is required when --mode is 'script'")
        sys.exit(1)

    config = _load_perf_config(args.config)
    include_suggestions = not args.no_suggest
    output_path = Path(args.output) if args.output else _default_output_path(args.mode)

    if args.mode == "script":
        report = _run_script_mode(
            args.script,
            config,
            readonly=not args.no_readonly,
            include_suggestions=include_suggestions,
        )
    elif args.mode == "api":
        report = _run_api_mode(config, include_suggestions=include_suggestions)
    elif args.mode == "pipeline":
        report = _run_pipeline_mode(config, include_suggestions=include_suggestions)
    elif args.mode == "report":
        report = _run_report_mode(config, include_suggestions=include_suggestions)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    _output_report(report, output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
