"""CLI entry point for the same governed refresh used by named pipelines."""

from __future__ import annotations

import json
import logging

from common.services.champion_refresh import run_governed_champion_refresh

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the fail-closed five-model experiment and atomic promotion."""
    result = run_governed_champion_refresh({})
    logger.info("Governed champion refresh completed: %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
