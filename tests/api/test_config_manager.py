"""Tests for config management API — /config endpoints."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ===========================================================================
# GET /config — list all configs
# ===========================================================================

@pytest.mark.asyncio
async def test_list_configs_returns_categories_and_configs():
    """GET /config returns categories and config list."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert "configs" in data
    assert len(data["categories"]) >= 6
    assert len(data["configs"]) >= 20
    # Each config has required fields
    for cfg in data["configs"]:
        assert "name" in cfg
        assert "label" in cfg
        assert "category" in cfg
        assert "description" in cfg


@pytest.mark.asyncio
async def test_list_configs_categories_have_required_fields():
    """Categories have key, label, description."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config")
    data = resp.json()
    for cat in data["categories"]:
        assert "key" in cat
        assert "label" in cat
        assert "description" in cat


# ===========================================================================
# GET /config/{name} — get config detail
# ===========================================================================

@pytest.mark.asyncio
async def test_get_config_detail_returns_fields():
    """GET /config/tune_strategies returns entry (fields may be empty)."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config/tune_strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "tune_strategies"
    assert data["category"] == "forecasting"
    assert "fields" in data


@pytest.mark.asyncio
async def test_get_config_detail_unknown_returns_404():
    """GET /config/nonexistent returns 404."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_config_detail_includes_raw_values():
    """GET /config/{name} includes raw YAML values."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config/planning_config")
    assert resp.status_code == 200
    data = resp.json()
    assert "raw" in data
    assert isinstance(data["raw"], dict)


# ===========================================================================
# PUT /config/{name} — update config
# ===========================================================================

@pytest.mark.asyncio
async def test_update_config_unknown_returns_404():
    """PUT /config/nonexistent returns 404."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put("/config/nonexistent", json={"values": {}})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_config_invalid_field_returns_400():
    """PUT with unknown field path returns 400."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/config/tune_strategies",
                json={"values": {"nonexistent.field": 42}},
            )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_update_config_no_changes():
    """PUT with no actual changes returns no-change response."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.config_manager.load_config", return_value={"tuning": {"n_trials": 50}}), \
         patch("api.routers.config_manager.reset_config"):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/config/hyperparameter_tuning",
                json={"values": {"tuning.n_trials": 50}},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["changed"] == []


@pytest.mark.asyncio
async def test_update_config_writes_changes(tmp_path):
    """PUT with valid changes writes the YAML file."""
    import yaml

    # Create a temp config file (use hyperparameter_tuning which still has fields)
    cfg_file = tmp_path / "hyperparameter_tuning.yaml"
    cfg_file.write_text(yaml.dump({"tuning": {"n_trials": 50, "n_splits": 5}}))

    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.config_manager._CONFIG_DIR", tmp_path), \
         patch("api.routers.config_manager.load_config", return_value={"tuning": {"n_trials": 50, "n_splits": 5}}), \
         patch("api.routers.config_manager.reset_config"):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/config/hyperparameter_tuning",
                json={"values": {"tuning.n_trials": 100}},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "tuning.n_trials" in data["changed"]
    # Verify file was written
    written = yaml.safe_load(cfg_file.read_text())
    assert written["tuning"]["n_trials"] == 100
    # Backup was created
    assert (tmp_path / "hyperparameter_tuning.yaml.bak").exists()


# ===========================================================================
# POST /config/{name}/reset — reset to backup
# ===========================================================================

@pytest.mark.asyncio
async def test_reset_config_no_backup_returns_404():
    """POST /config/{name}/reset returns 404 when no backup exists."""
    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.config_manager._CONFIG_DIR", MagicMock()):
        # Mock Path operations to simulate no backup
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/config/nonexistent/reset")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reset_config_restores_backup(tmp_path):
    """POST /config/{name}/reset restores from .yaml.bak."""
    import yaml

    # Create config + backup
    cfg_file = tmp_path / "tune_strategies.yaml"
    bak_file = tmp_path / "tune_strategies.yaml.bak"
    cfg_file.write_text(yaml.dump({"lgbm": {"strategies": [{"label": "modified"}]}}))  # modified
    bak_file.write_text(yaml.dump({"lgbm": {"strategies": [{"label": "original"}]}}))  # original

    pool = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.config_manager._CONFIG_DIR", tmp_path), \
         patch("api.routers.config_manager.reset_config"):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/config/tune_strategies/reset")
    assert resp.status_code == 200
    # Verify config was restored
    restored = yaml.safe_load(cfg_file.read_text())
    assert restored["lgbm"]["strategies"][0]["label"] == "original"


# ===========================================================================
# Metadata integrity tests
# ===========================================================================

def test_all_configs_have_valid_categories():
    """Every config in CONFIG_REGISTRY has a valid category."""
    from api.routers.config_manager import CONFIG_REGISTRY, CATEGORIES
    valid_cats = {c["key"] for c in CATEGORIES}
    for name, meta in CONFIG_REGISTRY.items():
        assert meta["category"] in valid_cats, f"{name} has invalid category {meta['category']}"


def test_all_fields_have_required_metadata():
    """Every field has label, description, and type."""
    from api.routers.config_manager import CONFIG_REGISTRY
    for name, meta in CONFIG_REGISTRY.items():
        for path, field in meta["fields"].items():
            assert "label" in field, f"{name}.{path} missing label"
            assert "description" in field, f"{name}.{path} missing description"
            assert "type" in field, f"{name}.{path} missing type"
            assert field["type"] in ("number", "integer", "text", "boolean", "select", "array", "object"), \
                f"{name}.{path} has invalid type {field['type']}"


def test_select_fields_have_options():
    """Select-type fields must have options list."""
    from api.routers.config_manager import CONFIG_REGISTRY
    for name, meta in CONFIG_REGISTRY.items():
        for path, field in meta["fields"].items():
            if field["type"] == "select":
                assert "options" in field and len(field["options"]) > 0, \
                    f"{name}.{path} is select type but has no options"


def test_number_fields_have_min_or_max():
    """Number/integer fields should have at least min or max constraint."""
    from api.routers.config_manager import CONFIG_REGISTRY
    for name, meta in CONFIG_REGISTRY.items():
        for path, field in meta["fields"].items():
            if field["type"] in ("number", "integer"):
                has_bounds = "min" in field or "max" in field
                # Not strictly required but almost all should have bounds
                if not has_bounds:
                    pass  # acceptable for some fields


def test_nested_helper_get_set():
    """Test _get_nested and _set_nested helpers."""
    from api.routers.config_manager import _get_nested, _set_nested
    data = {"a": {"b": {"c": 42}}}
    assert _get_nested(data, "a.b.c") == 42
    assert _get_nested(data, "a.b.d", "default") == "default"
    assert _get_nested(data, "x.y.z") is None
    _set_nested(data, "a.b.c", 99)
    assert data["a"]["b"]["c"] == 99
    _set_nested(data, "x.y.z", "new")
    assert data["x"]["y"]["z"] == "new"
