import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { BacktestStagePanel } from "../BacktestStagePanel";
import type { ModelInfo } from "../_types";

// Detail panel fetches its own data — stub it so the grid renders standalone.
vi.mock("../BacktestDetailPanel", () => ({
  BacktestDetailPanel: () => <div data-testid="detail-panel" />,
}));

vi.mock("@/components/Toaster", () => ({
  toast: { success: vi.fn(), info: vi.fn(), error: vi.fn() },
}));

const submitBacktestRun = vi.fn().mockResolvedValue({
  status: "queued",
  job_id: "j1",
  model_id: "lgbm_cluster",
  run_id: 1,
});

vi.mock("@/api/queries/backtest-management", () => ({
  backtestMgmtKeys: { summary: ["backtest-management", "summary"] as const },
  fetchBacktestSummary: vi.fn().mockResolvedValue({
    lgbm_cluster: {
      latest_run: { id: 1, status: "completed", is_loaded_to_db: true },
      has_predictions: true,
      current_accuracy: 66.5,
    },
  }),
  submitBacktestRun: (...args: unknown[]) => submitBacktestRun(...args),
}));

vi.mock("@/api/queries/accuracy", () => ({
  lagLeaderboardKeys: { list: () => ["lag-leaderboard"] as const },
  fetchLagLeaderboard: vi.fn().mockResolvedValue({
    source: "agg_accuracy_lag_archive",
    limit: 50,
    lags: [
      { lag: 0, rankings: [{ model_id: "lgbm_cluster", accuracy_pct: 71.2 }] },
      { lag: 1, rankings: [{ model_id: "lgbm_cluster", accuracy_pct: 68.4 }] },
      { lag: 2, rankings: [{ model_id: "lgbm_cluster", accuracy_pct: 64.1 }] },
      { lag: 3, rankings: [{ model_id: "lgbm_cluster", accuracy_pct: 60.3 }] },
      { lag: 4, rankings: [{ model_id: "lgbm_cluster", accuracy_pct: 57.8 }] },
    ],
  }),
}));

const MODELS: ModelInfo[] = [
  { id: "lgbm_cluster", label: "LightGBM", type: "tree", tunable: true },
  { id: "chronos2_enriched", label: "Chronos 2E", type: "foundation", tunable: false },
  { id: "mstl", label: "MSTL", type: "statistical", tunable: false },
  { id: "nhits", label: "N-HiTS", type: "deep_learning", tunable: false },
  { id: "nbeats", label: "N-BEATS", type: "deep_learning", tunable: false },
];

function renderPanel() {
  return render(
    <TestQueryWrapper>
      <BacktestStagePanel
        models={MODELS}
        selectedModelId="lgbm_cluster"
        selectedModelInfo={MODELS[0]}
        onSelectModel={vi.fn()}
      />
    </TestQueryWrapper>
  );
}

describe("BacktestStagePanel — grouped table", () => {
  beforeEach(() => submitBacktestRun.mockClear());

  it("renders a section per model family", () => {
    renderPanel();
    expect(screen.getAllByText("LightGBM").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Foundation")).toBeDefined();
    expect(screen.getByText("Statistical")).toBeDefined();
    expect(screen.getByText("Deep Learning")).toBeDefined();
  });

  it("offers 'Run all' only for groups with multiple models and no Load button", () => {
    renderPanel();
    // Only the deep-learning group has multiple models in the canonical roster.
    expect(screen.getAllByRole("button", { name: /Run all/ }).length).toBe(1);
    // The redundant Load button is gone from the grid.
    expect(screen.queryByRole("button", { name: "Load" })).toBeNull();
    expect(screen.queryByText("Load to DB")).toBeNull();
  });

  it("shows 'Loaded' status once a run has auto-loaded", async () => {
    renderPanel();
    expect(await screen.findByText("Loaded")).toBeDefined();
  });

  it("shows failed instead of completed when the latest job failed", async () => {
    const { fetchBacktestSummary } = await import("@/api/queries/backtest-management");
    vi.mocked(fetchBacktestSummary).mockResolvedValueOnce({
      nbeats: {
        latest_run: {
          id: 2,
          model_id: "nbeats",
          job_id: "failed-job",
          status: "failed",
          accuracy_pct: null,
          wape: null,
          bias: null,
          n_predictions: null,
          n_dfus: null,
          is_loaded_to_db: false,
          loaded_at: null,
          load_job_id: null,
          created_at: "2026-07-12T00:00:00Z",
          completed_at: "2026-07-12T00:01:00Z",
        },
        has_predictions: true,
        current_accuracy: null,
        current_wape: null,
      },
    });
    renderPanel();

    const labelCell = screen.getAllByText("N-BEATS").find((el) => el.closest("tr"));
    const row = labelCell!.closest("tr") as HTMLElement;
    expect(await within(row).findByText("Failed")).toBeDefined();
    expect(within(row).queryByText("Completed")).toBeNull();
  });

  it("shows execution-lag and fixed-lag accuracy for each model", async () => {
    renderPanel();

    const labelCell = screen.getAllByText("LightGBM").find((el) => el.closest("tr"));
    const row = labelCell!.closest("tr") as HTMLElement;
    expect(await within(row).findByText("Exec 66.5%")).toBeDefined();
    expect(within(row).getByText("L0 71.2%")).toBeDefined();
    expect(within(row).getByText("L1 68.4%")).toBeDefined();
    expect(within(row).getByText("L2 64.1%")).toBeDefined();
    expect(within(row).getByText("L3 60.3%")).toBeDefined();
    expect(within(row).getByText("L4 57.8%")).toBeDefined();
  });

  it("runs a single model's backtest via its row Run button", () => {
    renderPanel();
    // "LightGBM" also appears as the detail-card title — pick the table-row cell.
    const labelCell = screen.getAllByText("LightGBM").find((el) => el.closest("tr"));
    const row = labelCell!.closest("tr") as HTMLElement;
    const runBtn = within(row).getByRole("button", { name: /Run/ });
    fireEvent.click(runBtn);
    expect(submitBacktestRun).toHaveBeenCalledWith("lgbm_cluster", false);
  });

  it("runs only the checked models from the five-model roster", async () => {
    renderPanel();

    fireEvent.click(screen.getByRole("checkbox", { name: "Include MSTL" }));
    fireEvent.click(screen.getByRole("button", { name: "Run selected (4)" }));

    await waitFor(() => expect(submitBacktestRun).toHaveBeenCalledTimes(4));
    expect(submitBacktestRun).toHaveBeenCalledWith("lgbm_cluster", false);
    expect(submitBacktestRun).toHaveBeenCalledWith("chronos2_enriched", false);
    expect(submitBacktestRun).toHaveBeenCalledWith("nhits", false);
    expect(submitBacktestRun).toHaveBeenCalledWith("nbeats", false);
    expect(submitBacktestRun).not.toHaveBeenCalledWith("mstl", false);
  });
});
