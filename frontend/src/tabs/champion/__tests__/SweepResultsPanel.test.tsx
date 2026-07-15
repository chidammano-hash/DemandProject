import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { fetchSweepLeaderboard } from "@/api/queries";
import { SweepResultsPanel } from "../SweepResultsPanel";

const promoteSweepWinner = vi.fn().mockResolvedValue({ sweep_id: 1, promoted_experiment_id: 7, promoted: true });

vi.mock("@/api/queries", () => ({
  championSweepKeys: {
    detail: (id: number) => ["champion-sweeps", "detail", id],
    leaderboard: (id: number) => ["champion-sweeps", "leaderboard", id],
    segments: (id: number) => ["champion-sweeps", "segments", id],
  },
  fetchChampionSweep: vi.fn().mockResolvedValue({
    sweep_id: 1, label: "June tournament", status: "completed", mode: "both",
    segment_axis: "demand_class", candidate_count: 3, completed_count: 3,
    recommended_experiment_id: 7, recommended_score: 86.2, recommended_gate_eligible: true,
    best_global_experiment_id: 3, composite_experiment_id: 7,
  }),
  fetchSweepLeaderboard: vi.fn().mockResolvedValue({
    sweep_id: 1,
    members: [
      { experiment_id: 7, global_rank: 1, global_score: 86.2, gate_eligible: true, is_composite: true,
        skipped_duplicate: false, label: "composite", strategy: "per_segment", strategy_params: {},
        models: ["lgbm_cluster"], metric: "wape", champion_accuracy: 88.0, ceiling_accuracy: 92.0, gap_bps: 400, status: "completed" },
      { experiment_id: 3, global_rank: 2, global_score: 84.0, gate_eligible: false, is_composite: false,
        skipped_duplicate: false, label: "rolling", strategy: "rolling", strategy_params: { window_months: 6 },
        models: ["lgbm_cluster"], metric: "wape", champion_accuracy: 85.0, ceiling_accuracy: 92.0, gap_bps: 700, status: "completed" },
    ],
  }),
  fetchSweepSegments: vi.fn().mockResolvedValue({
    sweep_id: 1,
    segments: [
      { segment: "smooth", winner: { experiment_id: 7, strategy: "ensemble", accuracy: 90.0, n_dfus: 120, score: 90, segment_rank: 1, label: "e" }, candidates: [] },
      { segment: "intermittent", winner: { experiment_id: 3, strategy: "rolling", accuracy: 70.0, n_dfus: 40, score: 70, segment_rank: 1, label: "r" }, candidates: [] },
    ],
  }),
  promoteSweepWinner: (...a: unknown[]) => promoteSweepWinner(...a),
}));

function renderPanel(execLag?: number) {
  return render(
    <TestQueryWrapper>
      <SweepResultsPanel sweepId={1} execLag={execLag} />
    </TestQueryWrapper>,
  );
}

describe("SweepResultsPanel", () => {
  beforeEach(() => promoteSweepWinner.mockClear());

  it("renders the global leaderboard with both members", async () => {
    renderPanel();
    expect(await screen.findByText("Global leaderboard")).toBeDefined();
    expect((await screen.findAllByText("per_segment")).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("rolling")).length).toBeGreaterThan(0);
  });

  it("shows composite-vs-global headline and per-segment winners", async () => {
    renderPanel();
    expect(await screen.findByText(/Per-segment composite vs\. best global/)).toBeDefined();
    expect(screen.getByText("smooth")).toBeDefined();
    expect(screen.getByText("intermittent")).toBeDefined();
  });

  it("keeps sweep recommendations read-only", async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByText(/Recommended #7/)).toBeDefined());
    expect(screen.queryByRole("button", { name: /Promote winner/ })).toBeNull();
    expect(promoteSweepWinner).not.toHaveBeenCalled();
  });

  it("collapses duplicate composite members and shows one dense rank per row", async () => {
    const member = {
      gate_eligible: true, skipped_duplicate: false, strategy_params: {},
      models: ["lgbm_cluster"], metric: "wape", ceiling_accuracy: null,
      gap_bps: null, status: "completed" as const,
    };
    vi.mocked(fetchSweepLeaderboard).mockResolvedValueOnce({
      sweep_id: 1,
      members: [
        // A re-run sweep persists two identical composites, and the backend
        // ranks composites and non-composites separately (three rank-1 rows).
        { ...member, experiment_id: 7, global_rank: 1, global_score: 86.2,
          is_composite: true, label: "composite", strategy: "per_segment", champion_accuracy: 88.0 },
        { ...member, experiment_id: 8, global_rank: 1, global_score: 86.2,
          is_composite: true, label: "composite", strategy: "per_segment", champion_accuracy: 88.0 },
        { ...member, experiment_id: 3, global_rank: 1, global_score: 84.0,
          is_composite: false, label: "ensemble", strategy: "ensemble", champion_accuracy: 85.0 },
      ],
    });

    renderPanel();
    await screen.findByText("Global leaderboard");

    // The two identical composites collapse to one displayed row.
    expect(await screen.findAllByText("per_segment")).toHaveLength(1);

    // Ranks are dense by score: composite 1, ensemble 2 — no duplicate #1.
    const ensembleRow = screen.getAllByRole("row").find((row) =>
      within(row).queryByText("ensemble"),
    );
    expect(ensembleRow).toBeDefined();
    expect(within(ensembleRow as HTMLElement).getByText("2")).toBeInTheDocument();
  });

  it("makes the portfolio-only tournament scope explicit when a fixed lag is selected", async () => {
    renderPanel(2);

    expect(await screen.findByText("June tournament")).toBeDefined();
    expect(
      screen.getByText(
        "Tournament snapshot stays portfolio-wide; Lag 2 applies to the KPI cards, ranking, comparison, and experiment table.",
      ),
    ).toBeInTheDocument();
  });
});
