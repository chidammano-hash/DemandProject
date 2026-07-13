import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ChampionPanel } from "../ChampionPanel";

describe("ChampionPanel", () => {
  it("identifies the promoted experiment and its exact winner artifact", () => {
    render(
      <ChampionPanel
        championSummary={{
          experiment_id: 84,
          experiment_label: "Assigned from #83: champ1",
          strategy: "per_cluster",
          artifact_name: "experiment_84_winners.csv",
          total_dfus: 2,
          total_dfu_months: 3,
          total_champion_rows: 3,
          model_wins: { lgbm_cluster: 2, mstl: 1 },
          overall_champion_wape: 24.621,
          overall_champion_accuracy_pct: 75.379,
          overall_ceiling_wape: 15.4321,
          overall_ceiling_accuracy_pct: 84.5679,
          gap_bps: 918.89,
          run_ts: "2026-07-13T18:31:00Z",
        }}
      />
    );

    expect(screen.getByText("Promoted Champion Results")).toBeInTheDocument();
    expect(screen.getByText("Experiment #84 · Assigned from #83: champ1")).toBeInTheDocument();
    expect(screen.getByText("Per-cluster routing")).toBeInTheDocument();
    expect(screen.getByText("experiment_84_winners.csv")).toBeInTheDocument();
    expect(screen.getByText("LightGBM")).toBeInTheDocument();
    expect(screen.queryByText("ensemble")).not.toBeInTheDocument();
  });
});
