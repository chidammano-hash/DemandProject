"""Shared boilerplate for pipeline scripts.

Provides three helpers that eliminate repeated setup code:

    setup_logging()         — configures the standard log format
    add_common_args()       — registers --config, --dry-run, --output-dir
                              (and optionally --workers, --resume)
    load_project_env()      — loads .env from the project root

Usage::

    from common.scripts_base import setup_logging, add_common_args, load_project_env

    setup_logging()
    load_project_env()

    parser = argparse.ArgumentParser(...)
    add_common_args(parser, include_workers=True)
    args = parser.parse_args()
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path


# Project root is two levels up from this file (common/scripts_base.py → root)
_ROOT = Path(__file__).resolve().parents[1]

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the standard log format for pipeline scripts.

    Idempotent: if the root logger already has handlers this is a no-op,
    which prevents duplicate log lines when imported in tests.
    """
    if logging.root.handlers:
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_workers: bool = False,
    include_resume: bool = False,
) -> None:
    """Add standard CLI arguments to *parser*.

    Always added:
        --config      path to the YAML config file for this script
        --dry-run     preview without writing to the database

    Optional (controlled by keyword flags):
        --output-dir  directory for generated artifacts  (include_workers has no
                      effect on this; it is included via ``include_workers=True``
                      only as a convenience because those scripts usually also
                      write output files)
        --workers     max parallel worker processes      (include_workers=True)
        --resume      resume from a previous checkpoint  (include_resume=True)

    Args:
        parser:          The :class:`argparse.ArgumentParser` to modify in-place.
        include_workers: When ``True``, add ``--workers`` and ``--output-dir``.
        include_resume:  When ``True``, add ``--resume``.
    """
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file (default: script-specific default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing to the database",
    )
    if include_workers:
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Directory for output artifacts (default: script-specific default)",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Max parallel worker processes (default: 4)",
        )
    if include_resume:
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from checkpoints if a previous run crashed",
        )


def load_project_env() -> None:
    """Load the project-root .env file via python-dotenv.

    Safe to call when python-dotenv is not installed (silently skipped)
    or when the .env file does not exist.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return
    env_path = _ROOT / ".env"
    load_dotenv(env_path)
