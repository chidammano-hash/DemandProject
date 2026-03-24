"""AI Tuning Advisor — interactive LGBM hyperparameter tuning agent.

Provides a conversational chat interface for reviewing backtest runs,
analysing cluster/timeframe patterns, recommending parameter changes,
and (with user confirmation) triggering new backtest experiments.

Follows the ``AIPlannerAgent`` pattern from ``common/ai/ai_planner.py``:
agentic tool-use loop with circuit breakers and provider-agnostic support.

Usage (via API router):
    advisor = TuningAdvisorAgent(config)
    response_text, tool_calls = advisor.run_turn(session_id, messages)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import psycopg

from common.db import get_db_params
from common.utils import load_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit-breaker constants (overridable via config)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TURNS = 20
_DEFAULT_TOKEN_BUDGET = 50_000


# ---------------------------------------------------------------------------
# Tool implementations — all direct psycopg queries
# ---------------------------------------------------------------------------

def _list_tuning_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent tuning runs with metrics."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, run_label, model_id, started_at, completed_at,
                       status, accuracy_pct, wape, bias, n_predictions, n_dfus,
                       params, feature_count, notes
                FROM lgbm_tuning_run
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [
                "run_id", "run_label", "model_id", "started_at", "completed_at",
                "status", "accuracy_pct", "wape", "bias", "n_predictions", "n_dfus",
                "params", "feature_count", "notes",
            ]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _get_run_detail(run_id: int) -> dict[str, Any]:
    """Return full run detail including timeframes, clusters, and months."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            # Run row
            cur.execute(
                """
                SELECT run_id, run_label, model_id, started_at, completed_at,
                       status, accuracy_pct, wape, bias, n_predictions, n_dfus,
                       params, feature_count, features, notes
                FROM lgbm_tuning_run WHERE run_id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Run {run_id} not found"}
            run_cols = [
                "run_id", "run_label", "model_id", "started_at", "completed_at",
                "status", "accuracy_pct", "wape", "bias", "n_predictions", "n_dfus",
                "params", "feature_count", "features", "notes",
            ]
            run = dict(zip(run_cols, row))

            # Timeframes
            cur.execute(
                """
                SELECT timeframe, train_end, predict_start, predict_end,
                       n_predictions, accuracy_pct, wape, bias
                FROM lgbm_tuning_timeframe WHERE run_id = %s ORDER BY timeframe
                """,
                (run_id,),
            )
            tf_cols = ["timeframe", "train_end", "predict_start", "predict_end",
                       "n_predictions", "accuracy_pct", "wape", "bias"]
            run["timeframes"] = [dict(zip(tf_cols, r)) for r in cur.fetchall()]

            # Cluster breakdowns
            cur.execute(
                """
                SELECT cluster_type, cluster_value, n_predictions, n_dfus,
                       accuracy_pct, wape, bias
                FROM lgbm_tuning_cluster WHERE run_id = %s
                ORDER BY cluster_type, accuracy_pct DESC
                """,
                (run_id,),
            )
            cl_cols = ["cluster_type", "cluster_value", "n_predictions", "n_dfus",
                       "accuracy_pct", "wape", "bias"]
            run["clusters"] = [dict(zip(cl_cols, r)) for r in cur.fetchall()]

            # Month breakdowns
            cur.execute(
                """
                SELECT month_start, n_predictions, n_dfus, accuracy_pct, wape, bias
                FROM lgbm_tuning_month WHERE run_id = %s ORDER BY month_start
                """,
                (run_id,),
            )
            mo_cols = ["month_start", "n_predictions", "n_dfus", "accuracy_pct", "wape", "bias"]
            run["months"] = [dict(zip(mo_cols, r)) for r in cur.fetchall()]

            return run


def _compare_runs(baseline_id: int, candidate_id: int) -> dict[str, Any]:
    """Compare two runs using tuning_tracker.compare_runs."""
    from common.ml.tuning_tracker import compare_runs
    return compare_runs(baseline_id, candidate_id)


def _analyze_cluster_patterns(limit_runs: int = 10) -> dict[str, Any]:
    """Analyse cluster performance trends across recent runs."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            # Get recent completed run IDs
            cur.execute(
                """
                SELECT run_id, run_label, accuracy_pct, params
                FROM lgbm_tuning_run
                WHERE status = 'completed'
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit_runs,),
            )
            runs_cols = ["run_id", "run_label", "accuracy_pct", "params"]
            runs = [dict(zip(runs_cols, r)) for r in cur.fetchall()]

            if not runs:
                return {"runs": [], "cluster_trends": []}

            run_ids = [r["run_id"] for r in runs]

            # Get per-cluster accuracy for all these runs
            cur.execute(
                """
                SELECT c.run_id, r.run_label, c.cluster_type, c.cluster_value,
                       c.accuracy_pct, c.wape, c.n_dfus
                FROM lgbm_tuning_cluster c
                JOIN lgbm_tuning_run r ON r.run_id = c.run_id
                WHERE c.run_id = ANY(%s)
                ORDER BY c.cluster_type, c.cluster_value, r.started_at DESC
                """,
                (run_ids,),
            )
            ct_cols = ["run_id", "run_label", "cluster_type", "cluster_value",
                       "accuracy_pct", "wape", "n_dfus"]
            cluster_data = [dict(zip(ct_cols, r)) for r in cur.fetchall()]

            return {"runs": runs, "cluster_trends": cluster_data}


def _get_current_config() -> dict[str, Any]:
    """Return current LGBM algorithm config and tried strategies."""
    algo_cfg = load_config("algorithm_config.yaml")
    lgbm_params = algo_cfg.get("algorithms", {}).get("lgbm", {})

    strategies: list[dict[str, Any]] = []
    try:
        strat_cfg = load_config("auto_tune_strategies.yaml")
        strategies = strat_cfg.get("strategies", [])
    except FileNotFoundError:
        pass

    return {
        "current_lgbm_params": lgbm_params,
        "available_strategies": strategies,
    }


def _recommend_params(
    strategy_label: str,
    description: str,
    overrides: dict[str, Any],
    expected_impact: str,
    risk_assessment: str,
    base_on_run_id: int | None = None,
) -> dict[str, Any]:
    """Validate and structure a parameter recommendation for the frontend."""
    if not overrides:
        return {"error": "overrides must be non-empty"}
    if not strategy_label:
        return {"error": "strategy_label is required"}

    return {
        "strategy_label": strategy_label,
        "description": description,
        "overrides": overrides,
        "expected_impact": expected_impact,
        "risk_assessment": risk_assessment,
        "base_on_run_id": base_on_run_id,
    }


def _check_run_status(run_id: int) -> dict[str, Any]:
    """Check current status of a tuning run."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, run_label, status, started_at, completed_at,
                       accuracy_pct, wape, bias, n_predictions, n_dfus
                FROM lgbm_tuning_run WHERE run_id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Run {run_id} not found"}
            cols = ["run_id", "run_label", "status", "started_at", "completed_at",
                    "accuracy_pct", "wape", "bias", "n_predictions", "n_dfus"]
            result = dict(zip(cols, row))

            # Compute elapsed time
            if result["started_at"] and not result["completed_at"]:
                import datetime
                elapsed = datetime.datetime.now(datetime.timezone.utc) - result["started_at"]
                result["elapsed_seconds"] = int(elapsed.total_seconds())
            elif result["started_at"] and result["completed_at"]:
                elapsed = result["completed_at"] - result["started_at"]
                result["elapsed_seconds"] = int(elapsed.total_seconds())

            return result


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic format, converted to OpenAI on demand)
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_tuning_runs",
        "description": (
            "List recent LGBM tuning runs with accuracy, WAPE, bias, parameters, "
            "and status. Call this first to understand what has been tried."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max runs to return (default 20)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_run_detail",
        "description": (
            "Get full detail for a specific run including per-timeframe accuracy "
            "(A-J), per-cluster breakdowns (ml_cluster and business_cluster), "
            "and per-month accuracy. Use this to drill into a specific run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "integer", "description": "The run ID to inspect"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "compare_runs",
        "description": (
            "Compare two runs side-by-side: accuracy/WAPE/bias deltas, "
            "per-timeframe deltas, and verdict (improved/degraded/neutral). "
            "Use this to understand the impact of parameter changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "baseline_id": {"type": "integer", "description": "Baseline run ID"},
                "candidate_id": {"type": "integer", "description": "Candidate run ID"},
            },
            "required": ["baseline_id", "candidate_id"],
        },
    },
    {
        "name": "analyze_cluster_patterns",
        "description": (
            "Analyse cluster performance trends across multiple recent runs. "
            "Returns per-cluster accuracy for each run, enabling identification "
            "of clusters that consistently improve or degrade with certain parameters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit_runs": {"type": "integer", "description": "Number of recent runs to analyse (default 10)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_current_config",
        "description": (
            "Return the current production LGBM hyperparameters from "
            "algorithm_config.yaml and the list of available tuning strategies "
            "from auto_tune_strategies.yaml."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "recommend_params",
        "description": (
            "Structure a hyperparameter recommendation for the user. The frontend "
            "renders this as a confirmation card with Confirm/Reject buttons. "
            "Always call this BEFORE suggesting the user start a run. Include "
            "the specific parameter overrides, expected impact, and risk assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_label": {
                    "type": "string",
                    "description": "Short label for this experiment (e.g. 'adaptive_reg_v2')",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable explanation of why these params are recommended",
                },
                "overrides": {
                    "type": "object",
                    "description": "Parameter name -> value overrides to apply to LGBM config",
                },
                "expected_impact": {
                    "type": "string",
                    "description": "Expected accuracy impact (e.g. '+0.3-0.5% accuracy based on cluster 3 pattern')",
                },
                "risk_assessment": {
                    "type": "string",
                    "description": "Risk level and reasoning (e.g. 'Low — moderate regularization change')",
                },
                "base_on_run_id": {
                    "type": "integer",
                    "description": "Run ID whose params to build upon (optional)",
                },
            },
            "required": ["strategy_label", "description", "overrides", "expected_impact", "risk_assessment"],
        },
    },
    {
        "name": "check_run_status",
        "description": (
            "Check the current status of a running or completed tuning run. "
            "Returns status, elapsed time, and results if completed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "integer", "description": "The run ID to check"},
            },
            "required": ["run_id"],
        },
    },
]


def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an AI Tuning Advisor for an LGBM demand forecasting model.

## Context

You help supply chain planners optimize LGBM hyperparameters through iterative experimentation.
The platform uses a multi-timeframe backtest (timeframes A-J) with expanding windows.
Each run produces: overall accuracy%, WAPE, bias, plus per-cluster and per-month breakdowns.

## Key metrics

- **Accuracy**: 100 - (100 * SUM(|F-A|) / |SUM(A)|)  — higher is better
- **WAPE**: SUM(|F-A|) / |SUM(A)|  — lower is better
- **Bias**: (SUM(F) / SUM(A)) - 1  — closer to 0 is better (positive = over-forecast)
- **Verdict**: improved (delta >= +0.05%), degraded (delta <= -0.05%), neutral

## LGBM hyperparameters you can tune

- `learning_rate` (0.005 to 0.1) — smaller = more precise but needs more trees
- `n_estimators` (500 to 3000) — number of boosting rounds
- `num_leaves` (15 to 255) — tree complexity
- `max_depth` (4 to 16) — tree depth limit
- `min_child_samples` (5 to 100) — minimum samples per leaf
- `subsample` (0.5 to 1.0) — row subsampling ratio
- `colsample_bytree` (0.5 to 1.0) — column subsampling ratio
- `reg_lambda` (0.0 to 10.0) — L2 regularization

## Workflow

1. Always call `list_tuning_runs` first to understand the experiment history.
2. Use `analyze_cluster_patterns` to identify which clusters are struggling and which params help them.
3. Use `get_run_detail` to drill into specific runs that are interesting.
4. Use `compare_runs` to understand the impact of specific parameter changes.
5. When you have a recommendation, call `recommend_params` to present it as a structured card.
6. NEVER suggest starting a run without first calling `recommend_params`.
7. After a run completes, use `check_run_status` and `compare_runs` to analyse the results.

## Recommendation guidelines

- Base recommendations on observed patterns, not random exploration.
- Reference specific cluster/timeframe data when explaining your reasoning.
- Consider the trajectory: if regularization helped last time, try more of it.
- Be explicit about expected impact range and risk level.
- Avoid recommending too many parameter changes at once — change 1-3 params per run.
- If accuracy is already high (>75%), recommend conservative changes.

## Communication style

- Be concise and data-driven.
- Lead with the key insight, then supporting evidence.
- Use specific numbers from the tools (e.g., "cluster 3 at 65.2%, 4.7% below average").
- When recommending, explain the causal logic connecting parameters to cluster behaviour.
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class TuningAdvisorAgent:
    """LLM tool-use agent for interactive LGBM hyperparameter tuning.

    Supports OpenAI and Anthropic providers (configured via tuning_advisor_config.yaml).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            raw = load_config("tuning_advisor_config.yaml")
            config = raw

        self.config = config
        self.provider = config.get("provider", "openai").lower()
        self.model = config.get("model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.3)
        self.max_turns = config.get("max_turns", _DEFAULT_MAX_TURNS)
        self.token_budget = config.get("token_budget", _DEFAULT_TOKEN_BUDGET)

        if self.provider == "anthropic":
            import anthropic as _anthropic
            self.client = _anthropic.Anthropic()
        else:
            from openai import OpenAI
            import os
            api_key = os.getenv("OPENAI_API_KEY", "")
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # ------------------------------------------------------------------
    def _dispatch_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Route a tool call to the correct Python handler."""
        fns: dict[str, Any] = {
            "list_tuning_runs":         lambda: _list_tuning_runs(**tool_input),
            "get_run_detail":           lambda: _get_run_detail(**tool_input),
            "compare_runs":             lambda: _compare_runs(**tool_input),
            "analyze_cluster_patterns": lambda: _analyze_cluster_patterns(**tool_input),
            "get_current_config":       lambda: _get_current_config(),
            "recommend_params":         lambda: _recommend_params(**tool_input),
            "check_run_status":         lambda: _check_run_status(**tool_input),
        }
        fn = fns.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn()
        except Exception as exc:
            log.warning("Tool %s failed: %s", tool_name, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    def run_turn(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run one agentic turn with the given conversation history.

        Returns (ai_text_response, list_of_tool_calls_made).
        Each tool call dict has: {tool_name, tool_input, tool_result}.
        """
        if self.provider == "anthropic":
            return self._run_anthropic_turn(session_id, messages)
        return self._run_openai_turn(session_id, messages)

    # ------------------------------------------------------------------
    def _run_openai_turn(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """OpenAI function-calling loop for a single conversational turn."""
        oai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

        # Convert chat history to OpenAI format
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                oai_messages.append({"role": role, "content": content})

        oai_tools = _tools_to_openai(_TOOL_DEFINITIONS)
        tool_calls_made: list[dict[str, Any]] = []
        turn = 0
        total_tokens = 0
        final_text = ""

        while turn < self.max_turns and total_tokens < self.token_budget:
            turn += 1
            t0 = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=oai_tools,
                    messages=oai_messages,
                )
            except Exception as exc:
                log.error("OpenAI API error: %s", exc)
                return f"Error calling AI model: {exc}", tool_calls_made

            latency_ms = int((time.monotonic() - t0) * 1000)
            log.debug("Turn %d latency=%dms session=%s", turn, latency_ms, session_id)

            usage = response.usage
            if usage:
                total_tokens += usage.total_tokens or 0

            choice = response.choices[0]
            msg = choice.message

            # Append assistant message to conversation
            oai_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ] or None,
            })

            if choice.finish_reason == "stop":
                final_text = msg.content or ""
                break

            if choice.finish_reason == "tool_calls":
                for tc in msg.tool_calls or []:
                    raw_input = json.loads(tc.function.arguments)
                    t1 = time.monotonic()
                    result = self._dispatch_tool(tc.function.name, raw_input)
                    tool_ms = int((time.monotonic() - t1) * 1000)

                    tool_calls_made.append({
                        "tool_name": tc.function.name,
                        "tool_input": raw_input,
                        "tool_result": result,
                        "latency_ms": tool_ms,
                    })

                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })
                continue

            log.warning("Unexpected finish_reason: %s", choice.finish_reason)
            break

        if turn >= self.max_turns:
            log.warning("OpenAI loop hit MAX_TURNS=%d session=%s", self.max_turns, session_id)
        if total_tokens >= self.token_budget:
            log.warning("OpenAI loop hit TOKEN_BUDGET=%d session=%s", self.token_budget, session_id)

        return final_text, tool_calls_made

    # ------------------------------------------------------------------
    def _run_anthropic_turn(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic tool_use loop for a single conversational turn."""
        ant_messages: list[dict[str, Any]] = []

        # Convert chat history to Anthropic format
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                ant_messages.append({"role": role, "content": content})

        tool_calls_made: list[dict[str, Any]] = []
        turn = 0
        total_tokens = 0
        final_text = ""

        while turn < self.max_turns and total_tokens < self.token_budget:
            turn += 1
            t0 = time.monotonic()
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=_SYSTEM_PROMPT,
                    tools=_TOOL_DEFINITIONS,
                    messages=ant_messages,
                )
            except Exception as exc:
                log.error("Anthropic API error: %s", exc)
                return f"Error calling AI model: {exc}", tool_calls_made

            latency_ms = int((time.monotonic() - t0) * 1000)
            log.debug("Turn %d latency=%dms session=%s", turn, latency_ms, session_id)

            if hasattr(response, "usage") and response.usage:
                used = getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)
                total_tokens += used

            ant_messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = " ".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    raw_input = json.loads(json.dumps(block.input))
                    t1 = time.monotonic()
                    result = self._dispatch_tool(block.name, raw_input)
                    tool_ms = int((time.monotonic() - t1) * 1000)

                    tool_calls_made.append({
                        "tool_name": block.name,
                        "tool_input": raw_input,
                        "tool_result": result,
                        "latency_ms": tool_ms,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

                ant_messages.append({"role": "user", "content": tool_results})
                continue

            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        if turn >= self.max_turns:
            log.warning("Anthropic loop hit MAX_TURNS=%d session=%s", self.max_turns, session_id)
        if total_tokens >= self.token_budget:
            log.warning("Anthropic loop hit TOKEN_BUDGET=%d session=%s", self.token_budget, session_id)

        return final_text, tool_calls_made
