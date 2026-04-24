"""Unit tests for scripts/tools/check_fstring_sql.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "check_fstring_sql.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_fstring_sql", _TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_fstring_sql"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_detects_variable_interpolation(tmp_path):
    tool = _load_tool()
    bad = tmp_path / "bad.py"
    bad.write_text(
        'def run():\n'
        '    cur.execute(f"SELECT * FROM {table} WHERE id = {user_id}")\n'
    )
    offenders = tool.check_file(bad)
    assert len(offenders) == 1
    # Both interpolations must be captured
    line_no, snippet, variables = offenders[0]
    assert line_no == 2
    assert "cur.execute(f" in snippet
    assert "table" in variables
    assert "user_id" in variables


def test_ignores_parameterised_queries(tmp_path):
    tool = _load_tool()
    good = tmp_path / "good.py"
    good.write_text(
        'def run():\n'
        '    cur.execute("SELECT * FROM fact_sales WHERE item_id = %s", (item,))\n'
    )
    assert tool.check_file(good) == []


def test_ignores_format_spec_only_placeholders(tmp_path):
    """An f-string with only whitespace/format-spec placeholders is not flagged."""
    tool = _load_tool()
    p = tmp_path / "fmt.py"
    p.write_text(
        'def run():\n'
        '    cur.execute(f"SELECT 1")\n'
    )
    assert tool.check_file(p) == []


def test_detects_triple_quoted_fstring(tmp_path):
    tool = _load_tool()
    p = tmp_path / "triple.py"
    p.write_text(
        'def run():\n'
        '    cur.execute(f"""\n'
        '        SELECT *\n'
        '        FROM {dynamic_table}\n'
        '    """)\n'
    )
    offenders = tool.check_file(p)
    assert len(offenders) == 1
    _, _, variables = offenders[0]
    assert "dynamic_table" in variables


def test_main_returns_zero_on_clean_tree(tmp_path, monkeypatch, capsys):
    tool = _load_tool()
    # Create a "project" with one clean file.
    (tmp_path / "clean.py").write_text(
        'def run():\n'
        '    cur.execute("SELECT 1")\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_fstring_sql.py", str(tmp_path)])
    rc = tool.main()
    assert rc == 0


def test_main_returns_nonzero_on_violation(tmp_path, monkeypatch, capsys):
    tool = _load_tool()
    (tmp_path / "bad.py").write_text(
        'def run():\n'
        '    cur.execute(f"SELECT * FROM {t}")\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_fstring_sql.py", str(tmp_path)])
    rc = tool.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "bad.py" in captured.out
