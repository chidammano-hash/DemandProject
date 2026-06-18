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

function renderBuilder() {
  return render(
    <TestQueryWrapper>
      <SweepBuilder onClose={vi.fn()} />
    </TestQueryWrapper>,
  );
}

describe("SweepBuilder", () => {
  beforeEach(() => createChampionSweep.mockClear());

  it("renders template chips and mode/axis/objective selects", async () => {
    renderBuilder();
    expect(await screen.findByText("Rolling 6M")).toBeDefined();
    expect(screen.getByText("Ensemble Top-3")).toBeDefined();
    expect(screen.getByText("Mode")).toBeDefined();
    expect(screen.getByText("Segment axis")).toBeDefined();
    expect(screen.getByText("Objective")).toBeDefined();
  });

  it("previews candidate count = templates × variants and disables launch with no label", async () => {
    renderBuilder();
    // One model variant is pre-selected; pick 2 templates → 2 candidates.
    fireEvent.click(await screen.findByText("Rolling 6M"));
    fireEvent.click(screen.getByText("Ensemble Top-3"));
    expect(screen.getByText("2")).toBeDefined();
    // No label yet → launch disabled.
    const launch = screen.getByRole("button", { name: /Launch sweep/ });
    expect((launch as HTMLButtonElement).disabled).toBe(true);
  });

  it("launches with the assembled grid_spec", async () => {
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
    expect(body.grid_spec.models_variants.length).toBe(1);
  });
});
