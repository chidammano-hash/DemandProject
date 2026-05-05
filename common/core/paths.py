"""Single source of truth for filesystem paths anchored at the project root.

Use this module instead of recomputing :class:`pathlib.Path` chains like
``Path(__file__).resolve().parents[N]`` in every script and router. The depth
``N`` differs per file location and is a frequent source of bugs when files
are moved between subdirectories.

Example
-------
.. code-block:: python

    from common.core.paths import PROJECT_ROOT, CONFIG_DIR, DATA_DIR, SQL_DIR

    cfg = CONFIG_DIR / "forecasting" / "forecast_pipeline_config.yaml"

The only legitimate reason to keep a local ``Path(__file__).resolve().parents[N]``
is the bootstrap dance in standalone scripts that run via ``python scripts/X.py``
(rather than ``python -m scripts.X``) and need to ``sys.path.insert`` the project
root before any ``common.*`` import resolves. Everything else should import from
this module.
"""

from __future__ import annotations

from pathlib import Path

# Resolved once at import time. This file lives at common/core/paths.py, so the
# project root is two directories up from this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Common subdirectories.
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
SQL_DIR: Path = PROJECT_ROOT / "sql"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "SQL_DIR",
    "SCRIPTS_DIR",
    "DOCS_DIR",
]
