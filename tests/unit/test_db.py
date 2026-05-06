"""Tests for common/db.py — database connection helpers."""

import os
from unittest.mock import patch

from common.core.db import get_db_params, get_read_replica_params


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


class TestGetReadReplicaParams:
    """Read-replica routing — Item 24 (SCAFFOLD scope).

    The fall-back-to-primary path is the critical contract: when
    ``READ_REPLICA_URL`` is unset, every caller must behave identically
    to the primary-only path. These tests verify that contract.
    """

    def test_returns_none_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_read_replica_params() is None

    def test_returns_none_when_env_empty(self):
        # Empty string is treated as unset — operators sometimes export the
        # var without a value to "clear" it without unset-ing.
        with patch.dict(os.environ, {"READ_REPLICA_URL": ""}, clear=True):
            assert get_read_replica_params() is None

    def test_parses_full_url(self):
        url = "postgres://reader:secret@replica.example.com:5433/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True):
            result = get_read_replica_params()
        assert result == {
            "host": "replica.example.com",
            "port": 5433,
            "dbname": "demand_mvp",
            "user": "reader",
            "password": "secret",
        }

    def test_postgresql_scheme_accepted(self):
        # Both ``postgres://`` and ``postgresql://`` are valid in the wild.
        url = "postgresql://reader:secret@replica.example.com:5433/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True):
            assert get_read_replica_params() is not None

    def test_default_port_when_omitted(self):
        url = "postgres://reader:secret@replica.example.com/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True):
            result = get_read_replica_params()
        assert result is not None
        assert result["port"] == 5432

    def test_unparseable_url_falls_back_to_none(self):
        # Bad scheme: returns None and logger.warning fires (not asserted —
        # we only care that we degrade safely to the primary path).
        with patch.dict(os.environ, {"READ_REPLICA_URL": "not-a-url"}, clear=True):
            assert get_read_replica_params() is None

    def test_percent_encoded_password_decoded(self):
        # Passwords with special chars must be percent-encoded in the URL
        # and decoded back into the params dict; otherwise psycopg sees a
        # different string than what the operator set.
        url = "postgres://reader:p%40ss@replica.example.com:5433/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True):
            result = get_read_replica_params()
        assert result is not None
        assert result["password"] == "p@ss"


class TestReadReplicaPoolFallback:
    """End-to-end fallback contract for the read pool helpers in api.pool.

    The acceptance criterion for Item 24 SCAFFOLD scope is: when the env
    var is unset, ``get_read_only_conn()`` must use the primary pool and
    behave identically to ``get_conn()``. This test pins that contract.
    """

    def test_read_replica_configured_false_when_unset(self):
        from api.pool import _read_replica_configured

        with patch.dict(os.environ, {}, clear=True):
            assert _read_replica_configured() is False

    def test_read_replica_configured_true_when_set(self):
        from api.pool import _read_replica_configured

        url = "postgres://reader:secret@replica.example.com:5433/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True):
            assert _read_replica_configured() is True

    def test_get_read_only_conn_uses_primary_pool_when_unset(self):
        """The fallback path: with READ_REPLICA_URL unset, the helper must
        pull a connection from the primary pool — bit-for-bit equivalent
        to ``get_conn()``."""
        from unittest.mock import MagicMock

        from api.core import get_read_only_conn

        primary_pool = MagicMock()
        # Mimic ConnectionPool.connection() context manager.
        conn = MagicMock()
        primary_pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        primary_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(os.environ, {}, clear=True), \
             patch("api.core._get_pool", return_value=primary_pool) as primary_mock, \
             patch("api.core._get_read_pool") as read_mock:
            with get_read_only_conn() as got:
                assert got is conn
            # Primary was used; read pool was never touched.
            primary_mock.assert_called_once()
            read_mock.assert_not_called()

    def test_get_read_only_conn_uses_read_pool_when_configured(self):
        from unittest.mock import MagicMock

        from api.core import get_read_only_conn

        read_pool = MagicMock()
        conn = MagicMock()
        read_pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        read_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        url = "postgres://reader:secret@replica.example.com:5433/demand_mvp"
        with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True), \
             patch("api.core._get_read_pool", return_value=read_pool) as read_mock, \
             patch("api.core._get_pool") as primary_mock:
            with get_read_only_conn() as got:
                assert got is conn
            # Read pool was used; primary was never touched.
            read_mock.assert_called_once()
            primary_mock.assert_not_called()


class TestAsyncReadReplicaPoolFallback:
    """Async sibling of TestReadReplicaPoolFallback. Async handlers (Item 19
    pilot — customer_analytics) opt in via ``get_async_read_only_conn``."""

    def test_get_async_read_only_conn_uses_primary_when_unset(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from api.core import get_async_read_only_conn

        primary_pool = MagicMock()
        conn = MagicMock()
        primary_pool.connection.return_value.__aenter__ = AsyncMock(return_value=conn)
        primary_pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch.dict(os.environ, {}, clear=True), \
                 patch("api.core._get_async_pool", return_value=primary_pool) as primary_mock, \
                 patch("api.core._get_async_read_pool") as read_mock:
                async with get_async_read_only_conn() as got:
                    assert got is conn
                primary_mock.assert_called_once()
                read_mock.assert_not_called()

        asyncio.run(run())

    def test_get_async_read_only_conn_uses_read_pool_when_configured(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from api.core import get_async_read_only_conn

        read_pool = MagicMock()
        conn = MagicMock()
        read_pool.connection.return_value.__aenter__ = AsyncMock(return_value=conn)
        read_pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)

        async def run():
            url = "postgres://reader:secret@replica.example.com:5433/demand_mvp"
            with patch.dict(os.environ, {"READ_REPLICA_URL": url}, clear=True), \
                 patch("api.core._get_async_read_pool", return_value=read_pool) as read_mock, \
                 patch("api.core._get_async_pool") as primary_mock:
                async with get_async_read_only_conn() as got:
                    assert got is conn
                read_mock.assert_called_once()
                primary_mock.assert_not_called()

        asyncio.run(run())
