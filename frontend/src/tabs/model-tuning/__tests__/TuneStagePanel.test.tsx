import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

import { TuneStagePanel } from "../TuneStagePanel";
import type { ModelInfo } from "../_types";

const { fetchModelExperiments, fetchModelSummary } = vi.hoisted(() => ({
  fetchModelExperiments: vi.fn(),
  fetchModelSummary: vi.fn(),
}));

vi.mock("@/api/queries/model-tuning", () => ({
  fetchModelExperiments,
  fetchModelSummary,
}));

vi.mock("@/api/queries/unified-model-tuning", () => ({
  modelTuningKeys: {
    summary: (model: string) => ["model-tuning", model, "summary"],
    experiments: (model: string, params: unknown) => [
      "model-tuning",
      model,
      "experiments",
      params,
    ],
  },
}));

vi.mock("@/api/queries", () => ({
  STALE: { TWO_MIN: 120_000 },
}));

vi.mock("../../lgbm-tuning/ClusterEDAPanel", () => ({ ClusterEDAPanel: () => null }));
vi.mock("../../lgbm-tuning/FeatureLabPanel", () => ({ FeatureLabPanel: () => null }));
vi.mock("../LagFilterBar", () => ({ LagFilterBar: () => null }));
vi.mock("../EnhancedComparisonPanel", () => ({ EnhancedComparisonPanel: () => null }));
vi.mock("../RunHistoryTable", () => ({
  RunHistoryTable: ({ activeRunsCount }: { activeRunsCount: number }) => (
    <div data-testid="history-active-count">{activeRunsCount}</div>
  ),
}));

const MODELS: ModelInfo[] = [
  { id: "lgbm_cluster", label: "LightGBM", type: "tree", tunable: true, modelType: "lgbm" },
];

function renderPanel() {
  return render(
    <TestQueryWrapper>
      <TuneStagePanel
        models={MODELS}
        selectedModelId="lgbm_cluster"
        selectedModelInfo={MODELS[0]}
        selectedModel="lgbm"
        isTunable
        modelDetailTab="experiments"
        baselineId={null}
        candidateId={null}
        page={0}
        sortCol="run_id"
        sortDir="desc"
        statusFilter="all"
        execLag={undefined}
        setExecLag={vi.fn()}
        onSelectModel={vi.fn()}
        onSetDetailTab={vi.fn()}
        onSetStatusFilter={vi.fn()}
        onSelectRow={vi.fn()}
        onClearSelection={vi.fn()}
        onToggleSort={vi.fn()}
        onSetPage={vi.fn()}
        onShowLogs={vi.fn()}
        onPromote={vi.fn()}
        onOpenBuilder={vi.fn()}
      />
    </TestQueryWrapper>,
  );
}

describe("TuneStagePanel", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    fetchModelSummary
      .mockResolvedValueOnce({ best: null, runs: 1, active: 1, promoted: null })
      .mockResolvedValue({ best: null, runs: 1, active: 0, promoted: null });
    fetchModelExperiments
      .mockResolvedValueOnce({
        experiments: [{ run_id: 91, run_label: "Balanced", status: "queued" }],
        total: 1,
      })
      .mockResolvedValue({
        experiments: [{ run_id: 91, run_label: "Balanced", status: "cancelled" }],
        total: 1,
      });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("polls an active tuning run until its terminal status unlocks training", async () => {
    renderPanel();

    expect(await screen.findByRole("button", { name: "Training…" })).toBeDisabled();
    expect(screen.getByTestId("history-active-count")).toHaveTextContent("1");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "New Experiment" })).toBeEnabled();
    });
    expect(screen.getByTestId("history-active-count")).toHaveTextContent("0");
    expect(fetchModelExperiments).toHaveBeenCalledTimes(2);
  });
});
