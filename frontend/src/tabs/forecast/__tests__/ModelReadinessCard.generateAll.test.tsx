import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModelReadinessCard } from "../ModelReadinessCard";
import type { ForecastAlgorithm } from "../forecastPanelShared";
import type {
  SnapshotRosterReadiness,
  StagingSummary,
} from "@/api/queries/backtest-management";

const treeAlgo: ForecastAlgorithm = {
  id: "lgbm_cluster",
  type: "tree",
  enabled: true,
  forecast: true,
  compete: true,
  hasPredictions: true,
  accuracy: 92,
};

const neuralAlgo: ForecastAlgorithm = {
  id: "nhits",
  type: "deep_learning",
  enabled: true,
  forecast: true,
  compete: true,
  hasPredictions: true,
  accuracy: 88,
};

const readyCandidate: StagingSummary = {
  model_id: "lgbm_cluster",
  source_run_id: "new-run",
  run_status: "ready",
  promotion_eligible: true,
  generation_purpose: "release_candidate",
  row_count: 6,
  dfu_count: 1,
  source_model_count: 1,
  forecast_month_generated: "2026-07-01",
  last_generated_at: "2026-07-10T12:00:00Z",
  min_forecast_month: "2026-07-01",
  max_forecast_month: "2026-12-01",
};

const readySnapshotRoster: SnapshotRosterReadiness = {
  planning_month: "2026-07-01",
  ready: true,
  champion_ready: true,
  roster_model_count: 4,
  ready_contender_count: 3,
  required_contender_count: 3,
  contenders: [
    { model_id: "lgbm_cluster", rank: 1, ready: true, stale_reason: null },
    { model_id: "nhits", rank: 2, ready: true, stale_reason: null },
    { model_id: "mstl", rank: 3, ready: true, stale_reason: null },
  ],
  stale_reason: null,
  action_pipeline: "forecast-publish",
};

function renderCard(props: Partial<React.ComponentProps<typeof ModelReadinessCard>> = {}) {
  const onGenerateAll = vi.fn();
  const onTrainAll = vi.fn();
  const onPreparePublish = vi.fn();
  const base: React.ComponentProps<typeof ModelReadinessCard> = {
    forecastAlgos: [treeAlgo],
    trainingStatus: {
      lgbm_cluster: {
        model_id: "lgbm_cluster",
        type: "tree",
        trained: true,
        ready: true,
        trained_at: null,
        training_mode: "production",
        n_dfus: 1,
        planning_date: null,
      },
    },
    staging: {},
    trainableAlgos: [treeAlgo],
    trainedArtifactCount: 1,
    allRequiredArtifactsReady: true,
    isTraining: false,
    trainingModelId: null,
    generatingModelId: null,
    isGenerating: false,
    promotingModelId: null,
    promotedExperiment: null,
    championConstituents: [],
    championMissingModels: [],
    championReady: false,
    championDfuCount: 0,
    isChampionPromoted: false,
    snapshotReadiness: readySnapshotRoster,
    isPreparingPublish: false,
    onTrain: () => {},
    onTrainAll,
    onGenerate: () => {},
    onGenerateAll,
    generatableCount: 1,
    onPromote: () => {},
    onPreparePublish,
  };
  render(<ModelReadinessCard {...base} {...props} />);
  return { onGenerateAll, onTrainAll, onPreparePublish };
}

describe("ModelReadinessCard — Generate All", () => {
  it("renders the Generate All button with the ready count and fires onGenerateAll", () => {
    const { onGenerateAll } = renderCard();
    const btn = screen.getByText(/Generate All \(1\)/);
    fireEvent.click(btn);
    expect(onGenerateAll).toHaveBeenCalledTimes(1);
  });

  it("disables Generate All when no models are ready", () => {
    const { onGenerateAll } = renderCard({ generatableCount: 0 });
    const btn = screen.getByText(/Generate All \(0\)/).closest("button");
    expect(btn).not.toBeNull();
    expect(btn).toBeDisabled();
    fireEvent.click(btn!);
    expect(onGenerateAll).not.toHaveBeenCalled();
  });

  it("shows a spinner label while generating all", () => {
    renderCard({ generatingModelId: "__all__", isGenerating: true });
    expect(screen.getByText(/Generating All/)).toBeDefined();
  });

  it("offers one bulk training action when required artifacts are missing", () => {
    const { onTrainAll } = renderCard({
      trainedArtifactCount: 0,
      allRequiredArtifactsReady: false,
    });
    fireEvent.click(screen.getByRole("button", { name: /Train Production Models/ }));
    expect(onTrainAll).toHaveBeenCalledTimes(1);
  });

  it("blocks neural generation until its final-refit artifact is ready", () => {
    renderCard({
      forecastAlgos: [neuralAlgo],
      trainableAlgos: [neuralAlgo],
      trainedArtifactCount: 0,
      allRequiredArtifactsReady: false,
      generatableCount: 0,
      trainingStatus: {
        nhits: {
          model_id: "nhits",
          type: "deep_learning",
          trained: false,
          ready: false,
          trained_at: null,
          training_mode: null,
          n_dfus: null,
          planning_date: null,
        },
      },
    });

    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });

  it("keeps individual model candidates diagnostic-only", () => {
    renderCard({
      staging: { lgbm_cluster: readyCandidate },
    });

    expect(screen.getByText("Diagnostic only")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Promote" })).not.toBeInTheDocument();
  });

  it("disables promotion and offers one canonical preparation action when the roster is stale", () => {
    const { onPreparePublish } = renderCard({
      staging: { lgbm_cluster: readyCandidate },
      snapshotReadiness: {
        ...readySnapshotRoster,
        ready: false,
        ready_contender_count: 2,
        stale_reason: "One contender is stale.",
      },
    });

    expect(screen.queryByRole("button", { name: "Promote" })).not.toBeInTheDocument();
    expect(screen.getByText("Champion + 2/3 contenders ready")).toBeInTheDocument();
    const action = screen.getByRole("button", { name: "Prepare Release" });
    fireEvent.click(action);
    expect(onPreparePublish).toHaveBeenCalledWith("forecast-publish");
    expect(screen.getAllByRole("button", { name: "Prepare Release" })).toHaveLength(1);
  });

  it("offers Model Refresh when governed champion evidence is missing", () => {
    const { onPreparePublish } = renderCard({
      snapshotReadiness: {
        ...readySnapshotRoster,
        ready: false,
        champion_ready: false,
        stale_reason: "Run the named Model Refresh pipeline.",
        action_pipeline: "model-refresh",
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Refresh Models" }));
    expect(onPreparePublish).toHaveBeenCalledWith("model-refresh");
  });

  it("shows pipeline progress without allowing duplicate preparation", () => {
    renderCard({
      snapshotReadiness: { ...readySnapshotRoster, ready: false },
      isPreparingPublish: true,
    });

    expect(screen.getByRole("button", { name: "Preparing Release..." })).toBeDisabled();
  });

  it("does not report cleaned contender staging as a failure after publication", () => {
    renderCard({
      isChampionPromoted: true,
      championDfuCount: 12_476,
      generatableCount: 5,
      snapshotReadiness: {
        ...readySnapshotRoster,
        ready: false,
        ready_contender_count: 0,
        stale_reason: "Snapshot contender evidence failed an integrity check.",
        action_pipeline: null,
      },
    });

    expect(screen.getByText("Production release published")).toBeInTheDocument();
    expect(
      screen.getByText(/Generate All creates 5 diagnostic comparison forecasts/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Champion + 0/3 contenders ready")).not.toBeInTheDocument();
    expect(screen.queryByText(/failed an integrity check/)).not.toBeInTheDocument();
  });

});
