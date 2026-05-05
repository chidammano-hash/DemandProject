"""Tests for common/db.py — database connection helpers."""

import os
from unittest.mock import patch

from common.core.db import get_db_params


class TestGetDbParams:
    def test_returns_dict(self):
        result = get_db_params()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_db_params()
        assert "host" in result
        assert "port" in result
        assert "dbname" in result
        assert "user" in result
        assert "password" in result

    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_db_params()
            assert result["host"] == "localhost"
            assert result["port"] == 5440
            assert result["dbname"] == "demand_mvp"
            assert result["user"] == "demand"
            assert result["password"] == "demand"

    def test_respects_env_vars(self):
        env = {
            "POSTGRES_HOST": "db.example.com",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "testdb",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
        }
        with patch.dict(os.environ, env, clear=True):
            result = get_db_params()
            assert result["host"] == "db.example.com"
            assert result["port"] == 5432
            assert result["dbname"] == "testdb"
            assert result["user"] == "testuser"
            assert result["password"] == "testpass"

    def test_port_is_int(self):
        result = get_db_params()
        assert isinstance(result["port"], int)
