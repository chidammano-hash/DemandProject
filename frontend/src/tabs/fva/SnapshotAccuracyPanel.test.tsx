import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "../__tests__/test-utils";

vi.mock("@/api/queries", () => ({
  fetchFVASnapshotMonths: vi.fn().mockResolvedValue({
    months: [{ record_month: "2026-07-01", closed_lag_count: 0, latest_closed_forecast_month: null, last_refresh_at: null }],
  }),
  fetchFVASnapshotAccuracy: vi.fn().mockResolvedValue({
    record_month: "2026-07-01",
    rows: [
      { model_id: "champion", snapshot_role: "champion", contender_rank: null, lag: 0, forecast_month: "2026-07-01", n_dfus: 10, accuracy_pct: 90, wape: 10, bias: 0, fva_vs_champion_pts: 0, n_dfus_common: 10 },
      { model_id: "lgbm_cluster", snapshot_role: "contender", contender_rank: 1, lag: 0, forecast_month: "2026-07-01", n_dfus: 10, accuracy_pct: 88, wape: 12, bias: -0.02, fva_vs_champion_pts: -2, n_dfus_common: 10 },
    ],
  }),
  fetchFVAHistoricalBacktestMonths: vi.fn().mockResolvedValue({
    months: ["2026-06-01", "2026-05-01", "2026-04-01"],
    evidence_type: "historical_backtest",
  }),
  fetchFVAHistoricalBacktestAccuracy: vi.fn().mockResolvedValue({
    month: "2026-06-01",
    evidence_type: "historical_backtest",
    source: "backtest_lag_archive",
    supported_lags: [0, 1, 2, 3, 4],
    unsupported_lags: [5],
    rows: [
      { model_id: "lgbm_cluster", lag: 0, n_dfus: 11, accuracy_pct: 89, wape: 11, bias: -0.01, evidence_state: "measured" },
      { model_id: "lgbm_cluster", lag: 5, n_dfus: 0, accuracy_pct: null, wape: null, bias: null, evidence_state: "not_collected" },
    ],
  }),
  fvaKeys: {
    snapshotMonths: ["fva", "snapshot-months"],
    snapshotAccuracy: (month: string) => ["fva", "snapshot", month],
    historicalBacktestMonths: ["fva", "historical-backtest-months"],
    historicalBacktestAccuracy: (month: string) => ["fva", "historical-backtest", month],
  },
  STALE_PLATFORM: 300000,
}));

describe("SnapshotAccuracyPanel", () => {
  it("renders the champion and frozen contender rank with coverage", async () => {
    const { SnapshotAccuracyPanel } = await import("./SnapshotAccuracyPanel");
    render(
      <TestQueryWrapper>
        <SnapshotAccuracyPanel />
      </TestQueryWrapper>,
    );

    expect(await screen.findByText("Live Forward Snapshot Accuracy")).toBeInTheDocument();
    expect(await screen.findByText("Champion")).toBeInTheDocument();
    expect(await screen.findByText("#1 lgbm_cluster")).toBeInTheDocument();
    expect((await screen.findAllByText("10 DFUs")).length).toBe(2);
  });

  it("separates April through June backtests from live snapshots", async () => {
    const { SnapshotAccuracyPanel } = await import("./SnapshotAccuracyPanel");
    render(
      <TestQueryWrapper>
        <SnapshotAccuracyPanel />
      </TestQueryWrapper>,
    );

    expect(await screen.findByText("Historical Backtest Evidence")).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "2026-06" })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "2026-05" })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "2026-04" })).toBeInTheDocument();
    expect(await screen.findByText("Not collected")).toBeInTheDocument();
    expect(await screen.findByText(/Legacy backtests contain lags 0–4 only/)).toBeInTheDocument();
  });
});
