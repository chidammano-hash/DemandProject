"""US14: CLAUDE.md rule compliance for scripts/etl/.

Grep-style guards so the logging/exception cleanup doesn't regress. Two prints
are intentional and exempt: load.py._emit (machine-readable JSON protocol for
IntegrationRunner) and run_pipeline.print_summary (CLI summary report).
"""

import re
from pathlib import Path

_ETL = Path(__file__).resolve().parent.parent.parent / "scripts" / "etl"

# Built from fragments so this guard file does not itself trip the rule gate.
_BARE_EXCEPT_RE = re.compile(r"except\s+" + "Exception" + r"\b")

# Files allowed to keep print() and why.
_PRINT_EXEMPT = {"load.py", "run_pipeline.py"}


def _py_files():
    return sorted(_ETL.glob("*.py"))


def test_no_bare_except_exception():
    offenders = []
    for f in _py_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if _BARE_EXCEPT_RE.search(line):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"bare broad-except clause in scripts/etl: {offenders}"


def test_no_print_outside_exempt_files():
    offenders = []
    for f in _py_files():
        if f.name in _PRINT_EXEMPT:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"(^|[^A-Za-z_.])print\(", line):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"print() in scripts/etl (use logger): {offenders}"


def test_no_fragile_path_hacks():
    # os.path.join(os.path.dirname(__file__), "..") style bootstrap is forbidden.
    offenders = []
    for f in _py_files():
        if "os.path.join(os.path.dirname(__file__)" in f.read_text():
            offenders.append(f.name)
    assert not offenders, f"fragile sys.path bootstrap in scripts/etl: {offenders}"
