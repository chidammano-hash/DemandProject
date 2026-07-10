import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ForecastReleaseGateCard } from "../ForecastReleaseGateCard";
import type { ForecastReleaseReadiness } from "@/api/queries";
import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const fetchForecastReleaseReadiness = vi.fn();

vi.mock("@/api/queries", () => ({
  fetchForecastReleaseReadiness: (...args: unknown[]) =>
    fetchForecastReleaseReadiness(...args),
  forecastReleaseKeys: { readiness: () => ["forecast-release", "readiness"] },
}));

const blockedPayload = {
  ready: false,
  policy_enabled: true,
  release_version: "2026-07",
  planning_month: "2026-07-01",
  champion_experiment_id: 53,
  quality: {
    lookback_months: 6,
    first_month: "2026-01-01",
    last_month: "2026-05-01",
    dfu_months: 45_812,
    dfus: 12_023,
    closed_months: 5,
    actual_volume: 900_000,
    champion_observations: 50_000,
    champion_dfus: 12_500,
    common_observation_coverage_frac: 0.9162,
    common_dfu_coverage_frac: 0.9618,
    champion_wape_pct: 29.38,
    champion_accuracy_pct: 70.62,
    champion_bias_pct: -0.75,
    naive_wape_pct: 33.56,
    external_wape_pct: 26.5,
    relative_wape_lift_vs_naive_pct: 12.46,
    accuracy_delta_vs_external_pct_points: -2.88,
  },
  lineage: {
    active_promotion_id: 22,
    active_promotion_count: 1,
    champion_results_promoted: true,
    results_promoted_at: "2026-06-20T00:00:00+00:00",
    champion_rows_modified_at: "2026-06-19T00:00:00+00:00",
    results_promoted_experiment_count: 1,
    champion_cluster_experiment_id: null,
    current_cluster_experiment_id: 33,
    promoted_cluster_experiment_count: 1,
    matches: false,
    cluster_assignment_count: 0,
    stale_tuning_profiles: 9,
  },
  freshness: {
    release_promoted_at: "2026-07-10T00:00:00+00:00",
    release_generated_at: "2026-06-22T00:00:00+00:00",
    latest_sales_load: "2026-07-09T00:00:00+00:00",
    fresh: false,
  },
  coverage: {
    eligible_dfus: 13_968,
    complete_plan_dfus: 0,
    covered_eligible_dfus: 0,
    current_plan_rows: 0,
    coverage_frac: 0,
    forecast_start: null,
    forecast_end: null,
    required_end: "2026-12-01",
    minimum_history_months: 3,
  },
  release_integrity: {
    run_ids: 0,
    invalid_quantity_rows: 0,
    missing_source_rows: 0,
    invalid_interval_rows: 0,
    confidence_interval_rows: 0,
    confidence_interval_coverage_frac: null,
    minimum_confidence_interval_coverage_frac: 0.95,
    valid: false,
  },
  archive: {
    active_plan_version: "2026-06",
    outgoing_promotion_id: 21,
    outgoing_plan_version: "2026-06",
    outgoing_promoted_at: "2026-06-10T00:00:00+00:00",
    replacement_at: "2026-07-10T00:00:00+00:00",
    staging_record_month: "2026-07-01",
    models: 0,
    roster_rows: 0,
    champion_roster_rows: 0,
    contender_ranks: 0,
    model_lag_pairs: 0,
    minimum_rows: 0,
    champion_run_ids: 0,
    lineage_mismatches: 0,
    complete: false,
  },
  checks: [
    {
      id: "lift_vs_naive",
      status: "pass",
      value: 12.46,
      threshold: 10,
      message: "Relative champion WAPE improvement versus seasonal naive.",
    },
    {
      id: "delta_vs_external",
      status: "block",
      value: -2.88,
      threshold: 0,
      message: "Champion accuracy-point delta versus the external forecast.",
    },
    {
      id: "current_plan_coverage",
      status: "block",
      value: 0,
      threshold: 0.95,
      message: "Forecastable active DFUs have a complete six-month champion plan.",
    },
  ],
  next_action: {
    tab: "jobs",
    pipeline: "model-refresh",
    label: "Open Jobs for model refresh",
    reason: "Forecast evidence or model lineage is stale or below policy.",
  },
} satisfies ForecastReleaseReadiness;

describe("ForecastReleaseGateCard", () => {
  beforeEach(() => fetchForecastReleaseReadiness.mockReset());

  it("shows a blocked release with common-cohort metrics and next action", async () => {
    fetchForecastReleaseReadiness.mockResolvedValue(blockedPayload);
    const onNavigate = vi.fn();

    render(<ForecastReleaseGateCard onNavigate={onNavigate} />, {
      wrapper: TestQueryWrapper,
    });

    await screen.findByText("Release 2026-07 blocked");
    expect(screen.getByText("70.6%")).toBeInTheDocument();
    expect(screen.getByText("12.5%")).toBeInTheDocument();
    expect(screen.getByText("-2.9 pts")).toBeInTheDocument();
    expect(screen.getByText(/45,812 common DFU-months/)).toBeInTheDocument();
    const action = screen.getByRole("button", { name: "Open Jobs for model refresh" });
    fireEvent.click(action);
    expect(onNavigate).toHaveBeenCalledWith("jobs");
  });

  it("shows a planner-safe state when every gate passes", async () => {
    fetchForecastReleaseReadiness.mockResolvedValue({
      ...blockedPayload,
      ready: true,
      checks: blockedPayload.checks.map((check) => ({ ...check, status: "pass" })),
      next_action: null,
    });

    render(<ForecastReleaseGateCard onNavigate={vi.fn()} />, {
      wrapper: TestQueryWrapper,
    });

    await waitFor(() =>
      expect(screen.getByText("Release 2026-07 planner-ready")).toBeInTheDocument(),
    );
  });

  it("discloses every blocker without overwhelming the initial view", async () => {
    const blockerIds = [
      "common_cohort_coverage",
      "common_cohort_months",
      "delta_vs_external",
      "actual_alignment",
      "cluster_lineage",
      "sales_freshness",
      "current_plan_version",
      "outgoing_archive",
    ] as const;
    fetchForecastReleaseReadiness.mockResolvedValue({
      ...blockedPayload,
      checks: blockerIds.map((id, index) => ({
        id,
        status: "block",
        value: index,
        threshold: 0,
        message: `Blocker ${index + 1}`,
      })),
    });

    render(<ForecastReleaseGateCard onNavigate={vi.fn()} />, {
      wrapper: TestQueryWrapper,
    });

    await screen.findByText("Release 2026-07 blocked");
    expect(screen.getByText("2 more blockers included in release evidence.")).toBeInTheDocument();
    expect(screen.queryByText("Blocker 7")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show all 8 blockers" }));
    expect(screen.getByText("Blocker 7")).toBeInTheDocument();
    expect(screen.getByText("Blocker 8")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show fewer blockers" }));
    expect(screen.queryByText("Blocker 7")).not.toBeInTheDocument();
  });
});
