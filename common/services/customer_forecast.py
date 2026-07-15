"""Generation-only customer-level forecasting services (Spec 35)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from common.core.utils import load_forecast_pipeline_config
from common.ml.croston import parse_croston_parameters
from common.ml.customer_forecast_rules import (
    ADIDA_ROUTE_ID,
    CROSTON_ROUTE_ID,
    CUSTOMER_FORECAST_ROUTE_IDS,
    CUSTOMER_RULE_ROUTER_MODEL_ID,
    HOLT_DAMPED_ROUTE_ID,
    MOVING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    SES_ROUTE_ID,
    TRAILING_AVERAGE_ROUTE_ID,
    TSB_ROUTE_ID,
    CustomerForecastRuleParameters,
    forecast_customer_demand,
    parse_customer_forecast_rule_parameters,
    parse_customer_statistical_parameters,
    select_customer_forecast_route,
)

_IDENTITY_COLUMNS = ["item_id", "location_id", "customer_no"]
_CUSTOMER_PROFILE_HISTORY_MONTHS = 18


@dataclass(frozen=True)
class CustomerForecastWindow:
    planning_month: date
    history_start: date
    history_end: date
    forecast_start: date
    forecast_end: date
    history_months: int
    horizon_months: int
    forecast_months: tuple[date, ...]


@dataclass
class PreparedCustomerHistory:
    model_input: pd.DataFrame
    identity_by_sku: dict[str, tuple[str, str, str]]
    route_by_sku: dict[str, str]
    effective_history_months_by_sku: dict[str, int]
    skipped_series: list[dict[str, str]]

    @property
    def eligible_series_count(self) -> int:
        return len(self.identity_by_sku)


def _shift_month(month: date, offset: int) -> date:
    absolute = month.year * 12 + month.month - 1 + offset
    return date(absolute // 12, absolute % 12 + 1, 1)


def build_customer_forecast_window(
    planning_date: date,
    history_months: int,
    horizon_months: int,
) -> CustomerForecastWindow:
    """Resolve closed-history and future-output windows from one planning date."""
    if history_months <= 0 or horizon_months <= 0:
        raise ValueError("Customer forecast history and horizon must be positive")
    planning_month = planning_date.replace(day=1)
    history_start = _shift_month(planning_month, -history_months)
    history_end = planning_month - timedelta(days=1)
    forecast_end = _shift_month(planning_month, horizon_months) - timedelta(days=1)
    months = tuple(_shift_month(planning_month, offset) for offset in range(horizon_months))
    return CustomerForecastWindow(
        planning_month=planning_month,
        history_start=history_start,
        history_end=history_end,
        forecast_start=planning_month,
        forecast_end=forecast_end,
        history_months=history_months,
        horizon_months=horizon_months,
        forecast_months=months,
    )


def get_customer_forecast_settings() -> dict[str, Any]:
    config = load_forecast_pipeline_config()
    try:
        settings = dict(config["customer_forecast"])
        history_months = int(settings["history_months"])
        horizon_months = int(settings["horizon_months"])
        model_id = str(settings["model_id"])
        rule_params = parse_customer_forecast_rule_parameters(dict(settings["rule_params"]))
        croston_params = parse_croston_parameters(dict(settings["croston_params"]))
        statistical_params = parse_customer_statistical_parameters(
            dict(settings["statistical_params"])
        )
        recent_sales_lookback_months = int(settings["recent_sales_lookback_months"])
        batch_size = int(settings["batch_size"])
        cpu_workers = int(settings["cpu_workers"])
        max_batch_attempts = int(settings["max_batch_attempts"])
        progress_interval_seconds = float(settings["progress_interval_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast settings are incomplete") from exc
    if history_months != _CUSTOMER_PROFILE_HISTORY_MONTHS or horizon_months <= 0:
        raise ValueError("Customer forecasting requires 18 history months and a positive horizon")
    if model_id != CUSTOMER_RULE_ROUTER_MODEL_ID:
        raise ValueError("Customer forecasting requires the customer rule router")
    if rule_params.recent_demand_lookback_months > history_months:
        raise ValueError("Customer forecast rules exceed the available history")
    if recent_sales_lookback_months <= 0 or recent_sales_lookback_months > history_months:
        raise ValueError("Customer forecast recent-sales lookback is invalid")
    if batch_size <= 0 or cpu_workers <= 0:
        raise ValueError("Customer forecast batch worker settings are invalid")
    if max_batch_attempts <= 0 or progress_interval_seconds <= 0:
        raise ValueError("Customer forecast batch execution settings are invalid")
    if not isinstance(settings.get("enabled"), bool):
        raise ValueError("Customer forecast enabled setting must be boolean")
    return {
        **settings,
        "history_months": history_months,
        "horizon_months": horizon_months,
        "model_id": model_id,
        "recent_sales_lookback_months": recent_sales_lookback_months,
        "batch_size": batch_size,
        "cpu_workers": cpu_workers,
        "max_batch_attempts": max_batch_attempts,
        "progress_interval_seconds": progress_interval_seconds,
        "rule_params": rule_params.as_dict(),
        "croston_params": {
            "alpha": croston_params.alpha,
            "variant": croston_params.variant,
            "recursive": croston_params.recursive,
            "recursive_damping": croston_params.recursive_damping,
        },
        "statistical_params": statistical_params.as_dict(),
        "route_model_ids": list(CUSTOMER_FORECAST_ROUTE_IDS),
    }


def customer_forecast_config_checksum(settings: dict[str, Any]) -> str:
    generation_keys = (
        "enabled",
        "model_id",
        "rule_params",
        "croston_params",
        "statistical_params",
        "history_months",
        "horizon_months",
        "recent_sales_lookback_months",
        "batch_size",
        "cpu_workers",
        "max_batch_attempts",
        "progress_interval_seconds",
    )
    payload = {"customer_forecast": {key: settings[key] for key in generation_keys}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


CUSTOMER_EFFECTIVE_HISTORY_SQL = """LEAST(
    config.history_months,
    COALESCE(
        (
            EXTRACT(YEAR FROM AGE(
                config.forecast_start,
                profile.first_demand_month
            )) * 12
            + EXTRACT(MONTH FROM AGE(
                config.forecast_start,
                profile.first_demand_month
            ))
        )::integer,
        config.history_months
    )
)"""

CUSTOMER_ROUTE_CASE_SQL = """CASE
           WHEN profile.last_sales_month IS NULL
             OR profile.last_sales_month < config.recent_sales_start
           THEN NULL
           WHEN profile.first_demand_month >= config.recent_demand_start
            AND profile.first_demand_month < config.forecast_start
           THEN 'moving_average_3'
           WHEN profile.demand_months_last_18 < config.minimum_positive_demand_months
           THEN 'trailing_average_6'
           WHEN __EFFECTIVE_HISTORY__ >= config.seasonal_min_history_months
            AND profile.seasonal_repeat_validated
           THEN 'seasonal_repeat_12'
           WHEN __EFFECTIVE_HISTORY__::numeric
                    / NULLIF(profile.demand_months_last_18, 0)
                    >= config.intermittent_adi_threshold
            AND (
                (
                    EXTRACT(YEAR FROM AGE(
                        config.forecast_start,
                        profile.last_demand_month
                    )) * 12
                    + EXTRACT(MONTH FROM AGE(
                        config.forecast_start,
                        profile.last_demand_month
                    ))
                    - 1
                ) > config.decay_gap_adi_multiplier
                    * __EFFECTIVE_HISTORY__::numeric
                    / NULLIF(profile.demand_months_last_18, 0)
                OR (
                    __EFFECTIVE_HISTORY__ >= 12
                    AND
                    profile.demand_months_previous_6 > 0
                    AND profile.demand_months_recent_6
                        <= config.declining_occurrence_ratio
                           * profile.demand_months_previous_6
                )
            )
           THEN 'tsb'
           WHEN __EFFECTIVE_HISTORY__::numeric
                    / NULLIF(profile.demand_months_last_18, 0)
                    >= config.intermittent_adi_threshold
            AND CASE
                    WHEN profile.demand_months_last_18 > 1
                     AND profile.demand_sum_last_18 > 0
                    THEN (
                        (
                            profile.demand_sumsq_last_18
                            - profile.demand_sum_last_18
                              * profile.demand_sum_last_18
                              / profile.demand_months_last_18
                        ) / (profile.demand_months_last_18 - 1)
                    ) / POWER(
                        profile.demand_sum_last_18
                            / profile.demand_months_last_18,
                        2
                    )
                    ELSE 0
                END >= config.lumpy_cv2_threshold
           THEN 'adida'
           WHEN __EFFECTIVE_HISTORY__::numeric
                    / NULLIF(profile.demand_months_last_18, 0)
                    >= config.intermittent_adi_threshold
           THEN 'croston'
           WHEN __EFFECTIVE_HISTORY__ >= 12
            AND (
                (
                    profile.demand_sum_previous_6 = 0
                    AND profile.demand_sum_recent_6 > 0
                ) OR (
                    profile.demand_sum_previous_6 > 0
                    AND ABS(
                        profile.demand_sum_recent_6
                        - profile.demand_sum_previous_6
                    ) / profile.demand_sum_previous_6
                        >= config.trend_relative_change_threshold
                )
            )
           THEN 'holt_damped'
           ELSE 'ses'
       END""".replace("__EFFECTIVE_HISTORY__", CUSTOMER_EFFECTIVE_HISTORY_SQL)


def customer_route_config_params(
    window: CustomerForecastWindow,
    *,
    recent_sales_lookback_months: int,
    rule_params: CustomerForecastRuleParameters,
) -> tuple[Any, ...]:
    """Return the ordered values used by SQL route classification."""
    return (
        _shift_month(window.forecast_start, -recent_sales_lookback_months),
        _shift_month(
            window.forecast_start,
            -rule_params.recent_demand_lookback_months,
        ),
        window.forecast_start,
        window.history_months,
        rule_params.minimum_positive_demand_months,
        rule_params.seasonal_min_history_months,
        rule_params.intermittent_adi_threshold,
        rule_params.decay_gap_adi_multiplier,
        rule_params.declining_occurrence_ratio,
        rule_params.lumpy_cv2_threshold,
        rule_params.trend_relative_change_threshold,
    )


def load_customer_forecast_readiness(
    conn: Any,
    window: CustomerForecastWindow,
    *,
    recent_sales_lookback_months: int,
    rule_params: CustomerForecastRuleParameters,
) -> dict[str, Any]:
    """Return source freshness and eligibility for the resolved run window."""
    query = """WITH config AS (
               SELECT %s::date AS recent_sales_start,
                      %s::date AS recent_demand_start,
                      %s::date AS forecast_start,
                      %s::integer AS history_months,
                      %s::integer AS minimum_positive_demand_months,
                      %s::integer AS seasonal_min_history_months,
                      %s::numeric AS intermittent_adi_threshold,
                      %s::numeric AS decay_gap_adi_multiplier,
                      %s::numeric AS declining_occurrence_ratio,
                      %s::numeric AS lumpy_cv2_threshold,
                      %s::numeric AS trend_relative_change_threshold
           ), classified AS (
               SELECT profile.*, __ROUTE_CASE__ AS route_model_id
               FROM mv_customer_demand_series_profile profile
               CROSS JOIN config
               WHERE profile.first_month < config.forecast_start
           )
           SELECT MAX(source_latest_month),
                  COUNT(*),
                  COUNT(route_model_id),
                  COUNT(*) FILTER (WHERE route_model_id IS NULL),
                  COUNT(*) FILTER (WHERE route_model_id = 'moving_average_3'),
                  COUNT(*) FILTER (WHERE route_model_id = 'trailing_average_6'),
                  COUNT(*) FILTER (WHERE route_model_id = 'seasonal_repeat_12'),
                  COUNT(*) FILTER (WHERE route_model_id = 'tsb'),
                  COUNT(*) FILTER (WHERE route_model_id = 'adida'),
                  COUNT(*) FILTER (WHERE route_model_id = 'croston'),
                  COUNT(*) FILTER (WHERE route_model_id = 'ses'),
                  COUNT(*) FILTER (WHERE route_model_id = 'holt_damped'),
                  0::bigint,
                  0::bigint,
                  0::bigint,
                  (
                      SELECT batch_id
                      FROM audit_load_batch
                      WHERE domain = 'customer_demand'
                        AND status = 'completed'
                      ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                      LIMIT 1
                  ),
                  (
                      SELECT source_batch_id
                      FROM customer_demand_profile_refresh_state
                      WHERE singleton_id = 1
                  ),
                  (
                      SELECT COUNT(*)
                      FROM audit_load_batch
                      WHERE domain = 'customer_demand'
                        AND status = 'running'
                  )
           FROM classified""".replace("__ROUTE_CASE__", CUSTOMER_ROUTE_CASE_SQL)
    history_end_month = window.history_end.replace(day=1)
    with conn.cursor() as cur:
        cur.execute(
            query,
            customer_route_config_params(
                window,
                recent_sales_lookback_months=recent_sales_lookback_months,
                rule_params=rule_params,
            ),
        )
        row = cur.fetchone()
    (
        latest_month,
        total_series,
        eligible_series,
        zero_series,
        moving_average_series,
        trailing_average_series,
        seasonal_repeat_series,
        tsb_series,
        adida_series,
        croston_series,
        ses_series,
        holt_damped_series,
        invalid_keys,
        duplicate_grains,
        negative_rows,
        source_customer_demand_batch_id,
        profile_customer_demand_batch_id,
        active_customer_demand_loads,
    ) = row or (
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        0,
    )
    blockers: list[str] = []
    if source_customer_demand_batch_id is None:
        blockers.append("A completed customer-demand load is required before generating forecasts")
    if int(active_customer_demand_loads or 0) > 0:
        blockers.append("An active customer-demand load is in progress; wait for it to complete")
    elif source_customer_demand_batch_id is not None and (
        profile_customer_demand_batch_id is None
        or int(profile_customer_demand_batch_id) != int(source_customer_demand_batch_id)
    ):
        blockers.append("The customer-demand profile does not represent the latest completed load")
    if latest_month != history_end_month:
        blockers.append(f"Load customer demand through {window.history_end.strftime('%B %Y')}")
    total_series_count = int(total_series or 0)
    eligible_series_count = int(eligible_series or 0)
    zero_series_count = int(zero_series or 0)
    route_counts = {
        MOVING_AVERAGE_ROUTE_ID: int(moving_average_series or 0),
        TRAILING_AVERAGE_ROUTE_ID: int(trailing_average_series or 0),
        SEASONAL_REPEAT_ROUTE_ID: int(seasonal_repeat_series or 0),
        TSB_ROUTE_ID: int(tsb_series or 0),
        ADIDA_ROUTE_ID: int(adida_series or 0),
        CROSTON_ROUTE_ID: int(croston_series or 0),
        SES_ROUTE_ID: int(ses_series or 0),
        HOLT_DAMPED_ROUTE_ID: int(holt_damped_series or 0),
    }
    if sum(route_counts.values()) != eligible_series_count:
        blockers.append("Customer forecast route profile is inconsistent")
    if total_series_count == 0:
        blockers.append("Load customer demand history before generating forecasts")
    if int(invalid_keys or 0) > 0:
        blockers.append("Resolve invalid item, location, or customer keys")
    if int(duplicate_grains or 0) > 0:
        blockers.append("Aggregate duplicate customer-demand months")
    if int(negative_rows or 0) > 0:
        blockers.append("Resolve negative customer demand quantities")
    return {
        "ready": not blockers,
        "planning_month": window.planning_month.isoformat(),
        "history_start": window.history_start.isoformat(),
        "history_end": window.history_end.isoformat(),
        "forecast_start": window.forecast_start.isoformat(),
        "forecast_end": window.forecast_end.isoformat(),
        "history_months": window.history_months,
        "horizon_months": window.horizon_months,
        "source_latest_month": latest_month.isoformat() if latest_month else None,
        "source_customer_demand_batch_id": (
            int(source_customer_demand_batch_id)
            if source_customer_demand_batch_id is not None
            else None
        ),
        "profile_customer_demand_batch_id": (
            int(profile_customer_demand_batch_id)
            if profile_customer_demand_batch_id is not None
            else None
        ),
        "active_customer_demand_loads": int(active_customer_demand_loads or 0),
        "total_series": total_series_count,
        "eligible_series": eligible_series_count,
        "moving_average_series": route_counts[MOVING_AVERAGE_ROUTE_ID],
        "trailing_average_series": route_counts[TRAILING_AVERAGE_ROUTE_ID],
        "seasonal_repeat_series": route_counts[SEASONAL_REPEAT_ROUTE_ID],
        "tsb_series": route_counts[TSB_ROUTE_ID],
        "adida_series": route_counts[ADIDA_ROUTE_ID],
        "croston_series": route_counts[CROSTON_ROUTE_ID],
        "ses_series": route_counts[SES_ROUTE_ID],
        "holt_damped_series": route_counts[HOLT_DAMPED_ROUTE_ID],
        "model_route_counts": route_counts,
        "dormant_series": zero_series_count,
        "forecastable_series": eligible_series_count,
        "skipped_series": zero_series_count,
        "invalid_key_rows": int(invalid_keys or 0),
        "duplicate_grains": int(duplicate_grains or 0),
        "negative_rows": int(negative_rows or 0),
        "blockers": blockers,
    }


def prepare_customer_history(
    frame: pd.DataFrame,
    window: CustomerForecastWindow,
    *,
    recent_sales_lookback_months: int,
    rule_params: CustomerForecastRuleParameters,
) -> PreparedCustomerHistory:
    """Validate, filter, densify, and route active customer histories."""
    required = {
        *_IDENTITY_COLUMNS,
        "startdate",
        "demand_qty",
        "sales_qty",
        "series_first_month",
        "series_first_demand_month",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Customer history is missing columns: {sorted(missing)}")
    if frame.empty:
        return PreparedCustomerHistory(
            model_input=pd.DataFrame(columns=["sku_ck", "startdate", "qty"]),
            identity_by_sku={},
            route_by_sku={},
            effective_history_months_by_sku={},
            skipped_series=[],
        )
    if frame[_IDENTITY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("Customer history contains an invalid identity")

    history = frame.copy()
    history["startdate"] = pd.to_datetime(history["startdate"], errors="coerce")
    history["series_first_month"] = pd.to_datetime(history["series_first_month"], errors="coerce")
    raw_first_demand = history["series_first_demand_month"]
    history["series_first_demand_month"] = pd.to_datetime(raw_first_demand, errors="coerce")
    history["demand_qty"] = pd.to_numeric(history["demand_qty"], errors="coerce")
    history["sales_qty"] = pd.to_numeric(history["sales_qty"], errors="coerce")
    if history["series_first_month"].isna().any():
        raise ValueError("Customer history contains an invalid first month")
    if (raw_first_demand.notna() & history["series_first_demand_month"].isna()).any():
        raise ValueError("Customer history contains an invalid first demand month")
    first_demand_metadata = history["series_first_demand_month"].dropna()
    if not first_demand_metadata.eq(
        first_demand_metadata.dt.to_period("M").dt.to_timestamp()
    ).all():
        raise ValueError("Customer history first demand month must start a month")
    numeric = history["demand_qty"].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Customer history contains non-finite demand")
    if (numeric < 0).any():
        raise ValueError("Customer history contains negative demand")
    sales_numeric = history["sales_qty"].to_numpy(dtype=float)
    if not np.isfinite(sales_numeric).all() or (sales_numeric < 0).any():
        raise ValueError("Customer history contains invalid sales")

    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    model_frames: list[pd.DataFrame] = []
    identities: dict[str, tuple[str, str, str]] = {}
    routes: dict[str, str] = {}
    effective_history_months_by_sku: dict[str, int] = {}
    skipped: list[dict[str, str]] = []
    forecast_start = pd.Timestamp(window.forecast_start)
    recent_sales_start = pd.Timestamp(
        _shift_month(window.forecast_start, -recent_sales_lookback_months)
    )
    recent_demand_start = pd.Timestamp(
        _shift_month(
            window.forecast_start,
            -rule_params.recent_demand_lookback_months,
        )
    )
    grouped = history.groupby(_IDENTITY_COLUMNS, sort=True, dropna=False)
    for identity, group in grouped:
        item_id, location_id, customer_no = (str(value) for value in identity)
        bounded = group[
            group["startdate"].notna()
            & (group["startdate"] >= pd.Timestamp(window.history_start))
            & (group["startdate"] < pd.Timestamp(window.forecast_start))
        ]
        monthly = bounded.groupby("startdate", as_index=True)["demand_qty"].sum()
        monthly_sales = bounded.groupby("startdate", as_index=True)["sales_qty"].sum()
        dense = monthly.reindex(calendar, fill_value=0.0).astype(float)
        dense_sales = monthly_sales.reindex(calendar, fill_value=0.0).astype(float)
        has_recent_sales = bool(
            dense_sales.loc[dense_sales.index >= recent_sales_start].gt(0).any()
        )
        if not has_recent_sales:
            skipped.append(
                {
                    "item_id": item_id,
                    "location_id": location_id,
                    "customer_no": customer_no,
                    "reason": "no_sales_last_6_months",
                }
            )
            continue

        first_demand_values = group["series_first_demand_month"].drop_duplicates()
        if len(first_demand_values) != 1:
            raise ValueError("Customer history contains inconsistent first demand months")
        first_demand_value = first_demand_values.iloc[0]
        observed_first_demand = group.loc[
            group["startdate"].notna() & group["demand_qty"].gt(0),
            "startdate",
        ].min()
        if pd.isna(first_demand_value):
            if pd.notna(observed_first_demand):
                raise ValueError("Customer history first demand month contradicts demand history")
            first_demand_month = None
        else:
            first_demand_month = pd.Timestamp(first_demand_value)
        if (
            first_demand_month is not None
            and pd.notna(observed_first_demand)
            and first_demand_month > observed_first_demand
        ):
            raise ValueError("Customer history first demand month contradicts demand history")
        demand_started_within_recent_window = bool(
            first_demand_month is not None
            and recent_demand_start <= first_demand_month < forecast_start
        )
        if first_demand_month is None or first_demand_month < pd.Timestamp(window.history_start):
            effective_history_months = window.history_months
        else:
            effective_history_months = min(
                window.history_months,
                max(
                    1,
                    (forecast_start.year - first_demand_month.year) * 12
                    + forecast_start.month
                    - first_demand_month.month,
                ),
            )
        route_model_id = select_customer_forecast_route(
            dense.to_numpy(dtype=float),
            demand_started_within_recent_window=demand_started_within_recent_window,
            params=rule_params,
            effective_history_months=effective_history_months,
        )
        sku_ck = f"customer_series_{len(identities) + 1}"
        identities[sku_ck] = (item_id, location_id, customer_no)
        routes[sku_ck] = route_model_id
        effective_history_months_by_sku[sku_ck] = effective_history_months
        model_frames.append(
            pd.DataFrame({"sku_ck": sku_ck, "startdate": calendar, "qty": dense.to_numpy()})
        )

    model_input = (
        pd.concat(model_frames, ignore_index=True)
        if model_frames
        else pd.DataFrame(columns=["sku_ck", "startdate", "qty"])
    )
    model_input.attrs["history_end"] = (
        pd.Timestamp(window.history_end).to_period("M").to_timestamp()
    )
    return PreparedCustomerHistory(
        model_input=model_input,
        identity_by_sku=identities,
        route_by_sku=routes,
        effective_history_months_by_sku=effective_history_months_by_sku,
        skipped_series=skipped,
    )


def build_customer_forecast_rows(
    prepared: PreparedCustomerHistory,
    window: CustomerForecastWindow,
    *,
    rule_params: CustomerForecastRuleParameters,
    croston_params: dict[str, Any],
    statistical_params: dict[str, Any],
) -> pd.DataFrame:
    """Forecast every active customer series with its selected rule route."""
    parsed_croston_params = parse_croston_parameters(croston_params)
    parsed_statistical_params = parse_customer_statistical_parameters(statistical_params)
    frames: list[pd.DataFrame] = []
    for sku_ck, group in prepared.model_input.groupby("sku_ck", sort=False):
        route_model_id = prepared.route_by_sku[str(sku_ck)]
        forecast = forecast_customer_demand(
            group.sort_values("startdate")["qty"].to_numpy(dtype=float),
            horizon=window.horizon_months,
            route_model_id=route_model_id,
            rule_params=rule_params,
            croston_params=parsed_croston_params,
            statistical_params=parsed_statistical_params,
            effective_history_months=prepared.effective_history_months_by_sku[str(sku_ck)],
        )
        identity = prepared.identity_by_sku[str(sku_ck)]
        frames.append(
            pd.DataFrame(
                {
                    **dict(zip(_IDENTITY_COLUMNS, identity, strict=True)),
                    "forecast_month": pd.DatetimeIndex(window.forecast_months),
                    "forecast_qty": forecast,
                    "lower_bound": None,
                    "upper_bound": None,
                    "model_id": route_model_id,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                *_IDENTITY_COLUMNS,
                "forecast_month",
                "forecast_qty",
                "lower_bound",
                "upper_bound",
                "model_id",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        [*_IDENTITY_COLUMNS, "forecast_month"], ignore_index=True
    )


def _frame_checksum(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["sku_ck", "startdate"], ignore_index=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _resolve_run_window(
    conn: Any,
    run_id: str,
    settings: dict[str, Any],
) -> tuple[CustomerForecastWindow, str]:
    """Restore the immutable request window and reject configuration drift."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_status, planning_month, history_start, history_end, "
            "forecast_start, forecast_end, history_months, horizon_months, "
            "model_id, config_checksum, source_customer_demand_batch_id, "
            "(SELECT batch_id FROM audit_load_batch "
            " WHERE domain = 'customer_demand' AND status = 'completed' "
            " ORDER BY completed_at DESC NULLS LAST, batch_id DESC LIMIT 1), "
            "(SELECT source_batch_id FROM customer_demand_profile_refresh_state "
            " WHERE singleton_id = 1), "
            "(SELECT COUNT(*) FROM audit_load_batch "
            " WHERE domain = 'customer_demand' AND status = 'running') "
            "FROM customer_forecast_run "
            "WHERE run_id = %s::uuid",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] not in {"queued", "generating", "failed", "cancelled"}:
        raise RuntimeError("Customer forecast run is missing or not runnable")
    checksum = customer_forecast_config_checksum(settings)
    if row[8] != settings["model_id"] or row[9] != checksum:
        raise RuntimeError("Customer forecast configuration changed; submit a new run")
    if int(row[13] or 0) > 0:
        raise RuntimeError("Customer demand load is active; retry after it completes")
    if (
        row[10] is None
        or row[11] is None
        or row[12] is None
        or int(row[10]) != int(row[11])
        or int(row[10]) != int(row[12])
    ):
        raise RuntimeError("Customer demand changed after run submission; submit a new run")
    window = build_customer_forecast_window(row[1], int(row[6]), int(row[7]))
    if (row[2], row[3], row[4], row[5]) != (
        window.history_start,
        window.history_end,
        window.forecast_start,
        window.forecast_end,
    ):
        raise RuntimeError("Customer forecast run window is inconsistent")
    return window, checksum


def mark_customer_forecast_run_terminal(
    conn: Any,
    run_id: str,
    status: str,
    error_summary: str | None = None,
) -> None:
    if status not in {"failed", "cancelled"}:
        raise ValueError("Customer forecast terminal status must be failed or cancelled")
    summary = (error_summary or status)[:500]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = %s, error_summary = %s, "
            "completed_at = NOW() WHERE run_id = %s::uuid "
            "AND run_status IN ('queued', 'generating', 'failed')",
            (status, summary, run_id),
        )
    conn.commit()
