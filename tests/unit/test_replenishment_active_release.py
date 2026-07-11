"""Active-release selection for replenishment planning."""

from scripts.inventory.compute_replenishment_plan import get_active_plan_version


class _Cursor:
    def __init__(self, row: tuple[str] | None) -> None:
        self.row = row
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.sql = sql

    def fetchone(self) -> tuple[str] | None:
        return self.row


class _Connection:
    def __init__(self, row: tuple[str] | None) -> None:
        self.cur = _Cursor(row)

    def cursor(self) -> _Cursor:
        return self.cur


def test_active_plan_version_uses_verified_active_release() -> None:
    conn = _Connection(("2026-07",))

    assert get_active_plan_version(conn) == "2026-07"
    compact = " ".join(conn.cur.sql.split())
    assert "model_promotion_log" in compact
    assert "promotion.is_active = TRUE" in compact
    assert "forecast.promotion_log_id = promotion.id" in compact
    assert "forecast.run_id = promotion.production_run_id" in compact
    assert "forecast.lineage_status = 'verified'" in compact


def test_active_plan_version_is_none_without_a_verified_release() -> None:
    assert get_active_plan_version(_Connection(None)) is None
