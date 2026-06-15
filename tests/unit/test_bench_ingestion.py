"""Tests for scripts/tools/bench_ingestion.py (US2 — ingestion baseline harness).

Verifies the harness emits structured timing records and flags slow stages
using the threshold from config/platform/perf_config.yaml.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.tools.bench_ingestion import (
    StageTiming,
    flag_slow,
    time_stage,
    to_markdown,
)


class TestTimeStage:
    def test_returns_structured_record(self):
        result = time_stage("sales", "load", lambda: sum(range(1000)))
        assert isinstance(result, StageTiming)
        assert result.domain == "sales"
        assert result.stage == "load"
        assert result.seconds >= 0.0
        assert result.slow is False

    def test_passes_through_args(self):
        captured = {}

        def fn(a, b):
            captured["sum"] = a + b

        time_stage("forecast", "normalize", fn, 2, 3)
        assert captured["sum"] == 5


class TestFlagSlow:
    def test_explicit_threshold_marks_slow(self):
        timings = [
            StageTiming("sales", "load", 5.0),
            StageTiming("forecast", "load", 15.0),
        ]
        flagged = flag_slow(timings, threshold_s=10)
        assert flagged[0].slow is False
        assert flagged[1].slow is True

    def test_threshold_read_from_config_when_absent(self):
        timings = [StageTiming("sales", "load", 11.0)]
        fake_cfg = {"thresholds": {"function_slow_s": 10}}
        with patch(
            "scripts.tools.bench_ingestion._load_perf_config", return_value=fake_cfg
        ):
            flagged = flag_slow(timings)
        assert flagged[0].slow is True


class TestToMarkdown:
    def test_renders_table_with_header_and_rows(self):
        md = to_markdown([StageTiming("sales", "load", 1.5, slow=False)])
        assert "| Domain | Stage | Seconds | Slow |" in md
        assert "| sales | load | 1.50 |" in md

    def test_slow_marker_rendered(self):
        md = to_markdown([StageTiming("forecast", "load", 99.0, slow=True)])
        assert "⚠" in md
