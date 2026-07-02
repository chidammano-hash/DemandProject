import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { SweepBuilder } from "../SweepBuilder";

const createChampionSweep = vi.fn().mockResolvedValue({
  sweep_id: 1, job_id: "j1", status: "queued", candidate_count: 2, label: "t",
});

vi.mock("@/api/queries", () => ({
  championSweepKeys: { all: ["champion-sweeps"] },
  createChampionSweep: (...a: unknown[]) => createChampionSweep(...a),
  fetchChampionTemplates: vi.fn().mockResolvedValue({
    templates: [
      { id: "rolling_6m", label: "Rolling 6M", description: "rolling window" },
      { id: "ensemble_top3_inverse", label: "Ensemble Top-3", description: "blend" },
    ],
  }),
}));

// Model roster + current champion come from the pipeline config.
vi.mock("@/api/queries/unified-model-tuning", () => ({
  pipelineConfigKeys: { config: ["pipeline-config"] },
  fetchPipelineConfig: vi.fn().mockResolvedValue({
    algorithms: {
      lgbm_cluster: { type: "tree", enabled: true, compete: true, forecast: true },
      catboost_cluster: { type: "tree", enabled: true, compete: true, forecast: true },
      nbeats: { type: "deep_learning", enabled: true, compete: false, forecast: true },
      chronos2_enriched: { type: "foundation", enabled: true, compete: true, forecast: true },
    },
    champion: { models: ["lgbm_cluster", "chronos2_enriched"] },
  }),
}));

function renderBuilder() {
  return render(
    <TestQueryWrapper>
      <SweepBuilder onClose={vi.fn()} />
    </TestQueryWrapper>,
  );
}

describe("SweepBuilder", () => {
  beforeEach(() => createChampionSweep.mockClear());

  it("renders template chips, model roster, and mode/axis/objective selects", async () => {
    renderBuilder();
    expect(await screen.findByText("Rolling 6M")).toBeDefined();
    expect(screen.getByText("Ensemble Top-3")).toBeDefined();
    // Config-driven model roster (labels via model-labels).
    expect(await screen.findByText("LightGBM")).toBeDefined();
    expect(screen.getByText("Chronos 2E")).toBeDefined();
    expect(screen.getByText("Mode")).toBeDefined();
    expect(screen.getByText("Segment axis")).toBeDefined();
    expect(screen.getByText("Objective")).toBeDefined();
  });

  it("offers model-subset presets", async () => {
    renderBuilder();
    // Presets show with counts; "All tree (2)" from two tree models.
    expect(await screen.findByText(/All tree \(2\)/)).toBeDefined();
    expect(screen.getByText(/All foundation \(1\)/)).toBeDefined();
  });

  it("candidate count = number of templates (model subset doesn't multiply)", async () => {
    renderBuilder();
    // Champion models (2) auto-selected → 1 template = 1 candidate, 2 = 2.
    fireEvent.click(await screen.findByText("Rolling 6M"));
    fireEvent.click(screen.getByText("Ensemble Top-3"));
    expect(await screen.findByText("2")).toBeDefined();
    const launch = screen.getByRole("button", { name: /Launch sweep/ });
    expect((launch as HTMLButtonElement).disabled).toBe(true); // no label
  });

  it("launches with one model_variant = the selected subset", async () => {
    renderBuilder();
    fireEvent.click(await screen.findByText("Rolling 6M"));
    fireEvent.change(screen.getByPlaceholderText(/June champion/), { target: { value: "My sweep" } });
    const launch = screen.getByRole("button", { name: /Launch sweep/ });
    await waitFor(() => expect((launch as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(launch);
    await waitFor(() => expect(createChampionSweep).toHaveBeenCalledTimes(1));
    const body = createChampionSweep.mock.calls[0][0];
    expect(body.label).toBe("My sweep");
    expect(body.grid_spec.templates).toEqual(["rolling_6m"]);
    expect(body.grid_spec.models_variants).toEqual([["lgbm_cluster", "chronos2_enriched"]]);
  });
});
