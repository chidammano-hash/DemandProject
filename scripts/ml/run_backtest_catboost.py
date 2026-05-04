"""CatBoost backtest — delegates to run_backtest.py with --model catboost."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section


def main() -> None:
    """Run CatBoost backtest via the unified run_backtest entry point."""
    from scripts.ml.run_backtest import main as _main

    # Inject --model catboost before other CLI args (idempotent if already present)
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "catboost"])
    with profiled_section("catboost_backtest"):
        _main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
