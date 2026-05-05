"""Partition metadata for ETL domains, enabling per-slice (per-month) reload."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DomainPartition:
    field: str            # SQL column to filter on, e.g. "startdate"
    format: str           # "YYYY-MM" | "YYYY_MM" | "YYYYMM" | "YYYY-MM-DD"
    file_glob: str | None = None  # e.g. "Inventory_Snapshot_*.csv"


PARTITION_SPECS: dict[str, DomainPartition] = {
    "customer_demand": DomainPartition(field="startdate", format="YYYY-MM"),
    "inventory":       DomainPartition(field="snapshot_date", format="YYYY_MM",
                                       file_glob="Inventory_Snapshot_*.csv"),
    "forecast":        DomainPartition(field="fcstdate", format="YYYY-MM"),
    "sales":           DomainPartition(field="startdate", format="YYYY-MM"),
}


def get_partition(domain: str) -> DomainPartition | None:
    """Return partition spec for the domain, or None if not partitioned."""
    return PARTITION_SPECS.get(domain)


def is_partitioned(domain: str) -> bool:
    """True if the domain has a partition spec registered."""
    return domain in PARTITION_SPECS


def _next_month(year: int, month: int) -> tuple[int, int]:
    # Roll over Dec -> Jan of next year; avoids needing calendar.monthrange.
    if month == 12:
        return year + 1, 1
    return year, month + 1


def slice_to_date_range(slice_str: str, fmt: str) -> tuple[date, date]:
    """Convert a slice token to [start, end_exclusive) date range.

    Examples:
      ("2026-03",   "YYYY-MM")    -> (date(2026,3,1), date(2026,4,1))
      ("2026_03",   "YYYY_MM")    -> (date(2026,3,1), date(2026,4,1))
      ("202603",    "YYYYMM")     -> (date(2026,3,1), date(2026,4,1))
      ("2026-03-15","YYYY-MM-DD") -> (date(2026,3,15), date(2026,3,16))
    Raises ValueError on malformed input.
    """
    s = slice_str.strip()
    # Be lenient on the YYYY-MM/YYYY_MM separator — users routinely type one when
    # the other is expected. Normalize either to whatever the format requires.
    if fmt in ("YYYY-MM", "YYYY_MM") and len(s) == 7 and s[4] in ("-", "_"):
        s = s[:4] + ("-" if fmt == "YYYY-MM" else "_") + s[5:]
    try:
        if fmt == "YYYY-MM":
            if len(s) != 7 or s[4] != "-":
                raise ValueError(f"expected YYYY-MM, got {slice_str!r}")
            year, month = int(s[:4]), int(s[5:7])
        elif fmt == "YYYY_MM":
            if len(s) != 7 or s[4] != "_":
                raise ValueError(f"expected YYYY_MM, got {slice_str!r}")
            year, month = int(s[:4]), int(s[5:7])
        elif fmt == "YYYYMM":
            if len(s) != 6:
                raise ValueError(f"expected YYYYMM, got {slice_str!r}")
            year, month = int(s[:4]), int(s[4:6])
        elif fmt == "YYYY-MM-DD":
            if len(s) != 10 or s[4] != "-" or s[7] != "-":
                raise ValueError(f"expected YYYY-MM-DD, got {slice_str!r}")
            year, month, day = int(s[:4]), int(s[5:7]), int(s[8:10])
            start = date(year, month, day)
            # Day-grain slice: end is the next day (exclusive).
            end_year, end_month, end_day = year, month, day + 1
            try:
                end = date(end_year, end_month, end_day)
            except ValueError:
                # Day rolled past end-of-month; advance to first of next month.
                ny, nm = _next_month(year, month)
                end = date(ny, nm, 1)
            return start, end
        else:
            raise ValueError(f"unsupported format: {fmt!r}")
    except ValueError:
        raise
    except Exception as exc:  # numeric parsing fallthrough
        raise ValueError(f"malformed slice {slice_str!r} for format {fmt!r}") from exc

    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in slice {slice_str!r}")
    start = date(year, month, 1)
    ny, nm = _next_month(year, month)
    end = date(ny, nm, 1)
    return start, end
