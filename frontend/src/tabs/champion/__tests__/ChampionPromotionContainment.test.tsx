import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const fetchChampionExperiments = vi.fn().mockResolvedValue({
  experiments: [
    {
      experiment_id: 41,
      label: "Legacy experiment",
      notes: null,
      template_id: null,
      status: "completed",
      created_at: "2026-07-01T00:00:00Z",
      started_at: "2026-07-01T00:00:01Z",
      completed_at: "2026-07-01T00:05:00Z",
      runtime_seconds: 299,
      job_id: "job-41",
      strategy: "rolling",
      strategy_params: {},
      meta_learner_params: null,
      models: ["lgbm_cluster", "nhits", "nbeats", "mstl", "chronos2_enriched"],
      metric: "accuracy_pct",
      lag_mode: "execution",
      min_sku_rows: 3,
      champion_accuracy: 84.5,
      ceiling_accuracy: 88.2,
      gap_bps: 370,
      n_champions: 100,
      n_dfu_months: 1200,
      model_distribution: {},
      is_promoted: false,
      promoted_at: null,
      is_results_promoted: false,
      results_promoted_at: null,
      results_promote_job_id: null,
    },
  ],
  total: 1,
  offset: 0,
  limit: 50,
});

vi.mock("@/api/queries", () => ({
  championExperimentKeys: {
    all: ["champion-experiments"],
    experiments: () => ["champion-experiments", "list"],
    logs: (id: number, offset: number) => ["champion-experiments", "logs", id, offset],
  },
  CHAMPION_EXP_STALE: { EXPERIMENTS: 60_000 },
  fetchChampionExperiments: (...args: unknown[]) => fetchChampionExperiments(...args),
  fetchChampionExperimentLogs: vi.fn().mockResolvedValue({ log: "", has_more: false }),
  cancelChampionExperiment: vi.fn(),
  deleteChampionExperiment: vi.fn(),
  championSweepKeys: { list: () => ["champion-sweeps", "list"] },
  fetchChampionSweeps: vi.fn().mockResolvedValue({ sweeps: [] }),
}));

vi.mock("@/tabs/model-tuning/LagFilterBar", () => ({
  LagFilterBar: () => <div data-testid="lag-filter" />,
}));
vi.mock("../ChampionExperimentBuilder", () => ({
  ChampionExperimentBuilder: () => null,
}));
vi.mock("../ChampionComparisonPanel", () => ({
  ChampionComparisonPanel: () => null,
}));
vi.mock("../SweepBuilder", () => ({ SweepBuilder: () => null }));
vi.mock("../SweepResultsPanel", () => ({ SweepResultsPanel: () => null }));

describe("legacy champion promotion containment", () => {
  it("does not offer manual promotion from the experiment table", async () => {
    const { ChampionExperimentsPanel } = await import("../ChampionExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ChampionExperimentsPanel />
      </TestQueryWrapper>,
    );

    expect(await screen.findByText("Legacy experiment")).toBeDefined();
    expect(screen.queryByTitle("Promote")).toBeNull();
    expect(screen.queryByRole("button", { name: /promote/i })).toBeNull();
  });

  it("renders the retired modal as guidance without mutation actions", async () => {
    const { ChampionPromoteModal } = await import("../ChampionPromoteModal");
    const experiment = (await fetchChampionExperiments()).experiments[0];

    render(
      <TestQueryWrapper>
        <ChampionPromoteModal experiment={experiment} open onClose={vi.fn()} />
      </TestQueryWrapper>,
    );

    expect(screen.getByText(/named model-refresh pipeline/i)).toBeDefined();
    expect(screen.queryByRole("button", { name: /promote config/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /load results/i })).toBeNull();
  });
});
