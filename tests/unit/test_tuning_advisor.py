"""Tests for common.ai.tuning_advisor — TuningAdvisorAgent and tool functions."""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------


class TestRecommendParams:
    """Test the recommend_params validation tool."""

    def test_valid_recommendation(self):
        from common.ai.tuning_advisor import _recommend_params
        result = _recommend_params(
            strategy_label="test_v1",
            description="Test strategy",
            overrides={"learning_rate": 0.03, "reg_lambda": 2.0},
            expected_impact="+0.3% accuracy",
            risk_assessment="Low risk",
            base_on_run_id=5,
        )
        assert result["strategy_label"] == "test_v1"
        assert result["overrides"]["learning_rate"] == 0.03
        assert result["base_on_run_id"] == 5

    def test_empty_overrides_returns_error(self):
        from common.ai.tuning_advisor import _recommend_params
        result = _recommend_params(
            strategy_label="test",
            description="Test",
            overrides={},
            expected_impact="None",
            risk_assessment="None",
        )
        assert "error" in result

    def test_empty_label_returns_error(self):
        from common.ai.tuning_advisor import _recommend_params
        result = _recommend_params(
            strategy_label="",
            description="Test",
            overrides={"lr": 0.01},
            expected_impact="None",
            risk_assessment="None",
        )
        assert "error" in result


class TestListTuningRuns:
    """Test the list_tuning_runs tool."""

    @patch("common.ai.tuning_advisor.get_db_params", return_value={})
    @patch("common.ai.tuning_advisor.psycopg")
    def test_returns_run_list(self, mock_psycopg, mock_db_params):
        from common.ai.tuning_advisor import _list_tuning_runs

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "baseline", "lgbm_cluster", "2026-03-22", "2026-03-22",
             "completed", 69.34, 30.66, -0.013, 2725140, 50602,
             '{"learning_rate": 0.02}', 45, None),
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_psycopg.connect.return_value = mock_conn

        result = _list_tuning_runs(limit=5)
        assert len(result) == 1
        assert result[0]["run_id"] == 1
        assert result[0]["accuracy_pct"] == 69.34


class TestCheckRunStatus:
    """Test the check_run_status tool."""

    @patch("common.ai.tuning_advisor.get_db_params", return_value={})
    @patch("common.ai.tuning_advisor.psycopg")
    def test_returns_status(self, mock_psycopg, mock_db_params):
        from datetime import datetime, timezone
        from common.ai.tuning_advisor import _check_run_status

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        started = datetime(2026, 3, 22, 10, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 3, 22, 11, 0, 0, tzinfo=timezone.utc)
        mock_cursor.fetchone.return_value = (
            1, "test_run", "completed", started, completed,
            71.5, 28.5, -0.01, 2700000, 50000,
        )
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_psycopg.connect.return_value = mock_conn

        result = _check_run_status(run_id=1)
        assert result["status"] == "completed"
        assert result["accuracy_pct"] == 71.5
        assert result["elapsed_seconds"] == 3600

    @patch("common.ai.tuning_advisor.get_db_params", return_value={})
    @patch("common.ai.tuning_advisor.psycopg")
    def test_not_found(self, mock_psycopg, mock_db_params):
        from common.ai.tuning_advisor import _check_run_status

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_psycopg.connect.return_value = mock_conn

        result = _check_run_status(run_id=999)
        assert "error" in result


# ---------------------------------------------------------------------------
# Agent class tests
# ---------------------------------------------------------------------------

class TestTuningAdvisorAgent:
    """Test the TuningAdvisorAgent class."""

    @patch("openai.OpenAI")
    def test_dispatch_unknown_tool(self, mock_openai):
        from common.ai.tuning_advisor import TuningAdvisorAgent
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 100,
            "temperature": 0.3,
        }
        agent = TuningAdvisorAgent(config)
        result = agent._dispatch_tool("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @patch("openai.OpenAI")
    def test_dispatch_recommend_params(self, mock_openai):
        from common.ai.tuning_advisor import TuningAdvisorAgent
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 100,
            "temperature": 0.3,
        }
        agent = TuningAdvisorAgent(config)
        result = agent._dispatch_tool("recommend_params", {
            "strategy_label": "test_v1",
            "description": "A test",
            "overrides": {"learning_rate": 0.05},
            "expected_impact": "+0.5%",
            "risk_assessment": "Low",
        })
        assert result["strategy_label"] == "test_v1"

    @patch("openai.OpenAI")
    def test_max_turns_respected(self, mock_openai_cls):
        from common.ai.tuning_advisor import TuningAdvisorAgent
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 100,
            "temperature": 0.3,
            "max_turns": 1,
            "token_budget": 100000,
        }

        # Mock OpenAI to return a stop response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Done after 1 turn"
        mock_choice.message.tool_calls = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 500
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        agent = TuningAdvisorAgent(config)

        text, calls = agent.run_turn("session-1", [{"role": "user", "content": "test"}])
        assert text == "Done after 1 turn"
        assert mock_client.chat.completions.create.call_count == 1


class TestGetCurrentConfig:
    """Test the get_current_config tool."""

    @patch("common.ai.tuning_advisor.load_config")
    @patch("common.utils.get_algorithm_params")
    def test_returns_config(self, mock_get_algo, mock_load_config):
        from common.ai.tuning_advisor import _get_current_config

        mock_get_algo.return_value = {"learning_rate": 0.02, "n_estimators": 1500}
        mock_load_config.return_value = {
            "lgbm": {"strategies": [{"label": "test", "overrides": {}}]},
        }
        result = _get_current_config()
        assert "current_lgbm_params" in result
        assert result["current_lgbm_params"]["learning_rate"] == 0.02
        assert len(result["available_strategies"]) == 1


class TestToolDefinitions:
    """Test that tool definitions are well-formed."""

    def test_all_tools_have_required_fields(self):
        from common.ai.tuning_advisor import _TOOL_DEFINITIONS
        for tool in _TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
            assert tool["input_schema"]["type"] == "object"

    def test_tools_to_openai_conversion(self):
        from common.ai.tuning_advisor import _tools_to_openai, _TOOL_DEFINITIONS
        oai_tools = _tools_to_openai(_TOOL_DEFINITIONS)
        assert len(oai_tools) == len(_TOOL_DEFINITIONS)
        for t in oai_tools:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]
            assert "parameters" in t["function"]

    def test_expected_tool_count(self):
        from common.ai.tuning_advisor import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 7
