import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
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

function renderPanel() {
  return render(
    <TestQueryWrapper>
      <SweepResultsPanel sweepId={1} />
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

  it("promote winner is enabled and fires when gate-eligible", async () => {
    renderPanel();
    const btn = await screen.findByRole("button", { name: /Promote winner/ });
    await waitFor(() => expect((btn as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(btn);
    await waitFor(() => expect(promoteSweepWinner).toHaveBeenCalledTimes(1));
  });
});
