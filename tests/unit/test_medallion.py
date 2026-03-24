"""Tests for common/engines/medallion.py — file hashing and batch tracking."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.medallion import file_hash, create_batch, complete_batch, fail_batch


class TestFileHash:
    def test_returns_sha256_hex(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_bytes(b"hello world")
        result = file_hash(p)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_io_error_returns_empty(self):
        assert file_hash(Path("/nonexistent/file.csv")) == ""


class TestCreateBatch:
    def test_inserts_and_returns_batch_id(self):
        cur = MagicMock()
        cur.fetchone.return_value = (42,)
        result = create_batch(cur, "sales", source_file="f.csv", source_hash="abc")
        cur.execute.assert_called_once()
        sql, params = cur.execute.call_args.args
        assert "INSERT INTO audit_load_batch" in sql
        assert "'direct'" in sql
        assert "%s" in sql
        assert params == ["sales", "f.csv", "abc"]
        assert result == 42


class TestCompleteBatch:
    def test_updates_with_correct_params(self):
        cur = MagicMock()
        complete_batch(cur, 7, row_count_in=100, row_count_out=95)
        sql, params = cur.execute.call_args.args
        assert "status = 'completed'" in sql
        assert params == [100, 95, 7]


class TestFailBatch:
    def test_updates_with_correct_params(self):
        cur = MagicMock()
        fail_batch(cur, 7, "kaboom")
        sql, params = cur.execute.call_args.args
        assert "status = 'failed'" in sql
        assert params == ["kaboom", 7]
