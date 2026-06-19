"""Unit tests for the fine-tuned Chronos-Bolt dispatcher wiring (spec 32).

Validates checkpoint resolution (versioned / flat / missing -> base fallback) and
that the fine-tuned variant is registered in the foundation dispatch table.
"""

from common.ml.expert_panel.foundation_models import (
    _FOUNDATION_DISPATCH,
    _resolve_ft_checkpoint,
)


def test_ft_variant_registered():
    assert "chronos_bolt_ft" in _FOUNDATION_DISPATCH


def test_resolve_missing_falls_back_to_base():
    path, is_ft = _resolve_ft_checkpoint("/no/such/dir", "amazon/chronos-bolt-base")
    assert is_ft is False
    assert path == "amazon/chronos-bolt-base"


def test_resolve_none_checkpoint_dir():
    path, is_ft = _resolve_ft_checkpoint(None, "amazon/chronos-bolt-base")
    assert is_ft is False
    assert path == "amazon/chronos-bolt-base"


def test_resolve_picks_highest_version(tmp_path):
    for v in ("v1", "v2", "v10"):
        d = tmp_path / v
        d.mkdir()
        (d / "config.json").write_text("{}")
    (tmp_path / "v3_no_config").mkdir()  # missing config.json -> not a valid checkpoint
    path, is_ft = _resolve_ft_checkpoint(str(tmp_path), "base")
    assert is_ft is True
    # numeric version sort -> v10 is newest (lexicographic would wrongly pick v2)
    assert path.rstrip("/").split("/")[-1] == "v10"


def test_resolve_flat_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    path, is_ft = _resolve_ft_checkpoint(str(tmp_path), "base")
    assert is_ft is True
    assert path == str(tmp_path)
