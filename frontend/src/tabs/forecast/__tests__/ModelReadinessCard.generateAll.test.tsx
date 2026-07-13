import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModelReadinessCard } from "../ModelReadinessCard";
import type { ForecastAlgorithm } from "../forecastPanelShared";
import type { StagingSummary } from "@/api/queries/backtest-management";
import type { ChampionExperiment } from "@/api/queries/champion-experiments";

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
  promotion_eligible: false,
  generation_purpose: "release_candidate",
  row_count: 6,
  dfu_count: 1,
  source_model_count: 1,
  forecast_month_generated: "2026-07-01",
  last_generated_at: "2026-07-10T12:00:00Z",
  min_forecast_month: "2026-07-01",
  max_forecast_month: "2026-12-01",
};

const perClusterChampion: ChampionExperiment = {
  experiment_id: 84,
  label: "Current champion",
  notes: null,
  template_id: null,
  status: "completed",
  created_at: "2026-07-10T12:00:00Z",
  started_at: "2026-07-10T12:00:00Z",
  completed_at: "2026-07-10T12:02:00Z",
  runtime_seconds: 120,
  job_id: "champion-job",
  strategy: "per_cluster",
  strategy_params: { min_prior_months: 3 },
  meta_learner_params: null,
  models: ["lgbm_cluster", "nhits", "nbeats", "mstl", "chronos2_enriched"],
  metric: "accuracy_pct",
  lag_mode: "execution",
  min_sku_rows: 3,
  champion_accuracy: 75.4,
  ceiling_accuracy: 84.6,
  gap_bps: 919,
  n_champions: 9_353,
  n_dfu_months: 42_329,
  model_distribution: null,
  is_promoted: true,
  promoted_at: "2026-07-10T12:03:00Z",
  is_results_promoted: true,
  results_promoted_at: "2026-07-10T12:03:00Z",
  results_promote_job_id: "champion-job",
};

function renderCard(props: Partial<React.ComponentProps<typeof ModelReadinessCard>> = {}) {
  const onGenerateAll = vi.fn();
  const onTrainAll = vi.fn();
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
    promotedExperiment: null,
    championConstituents: [],
    championReady: false,
    championDfuCount: 0,
    isChampionPromoted: false,
    activeProductionModelId: null,
    onTrain: () => {},
    onTrainAll,
    onGenerate: () => {},
    onGenerateAll,
    generatableCount: 1,
  };
  render(<ModelReadinessCard {...base} {...props} />);
  return { onGenerateAll, onTrainAll };
}

describe("ModelReadinessCard — Generate All", () => {
  it("renders the Generate All Drafts button with the ready count and fires onGenerateAll", () => {
    const { onGenerateAll } = renderCard();
    const btn = screen.getByText(/Generate All Drafts \(1\)/);
    fireEvent.click(btn);
    expect(onGenerateAll).toHaveBeenCalledTimes(1);
  });

  it("disables Generate All when no models are ready", () => {
    const { onGenerateAll } = renderCard({ generatableCount: 0 });
    const btn = screen.getByText(/Generate All Drafts \(0\)/).closest("button");
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

  it("marks new individual model output as a generated draft", () => {
    renderCard({
      staging: { lgbm_cluster: readyCandidate },
    });

    expect(screen.getByText("Generated draft")).toBeInTheDocument();
    expect(screen.queryByText("Diagnostic only")).not.toBeInTheDocument();
  });

  it("states that Period Roll does not block the first production release", () => {
    renderCard({ staging: { lgbm_cluster: readyCandidate } });

    expect(screen.getByText("Ready for first production release")).toBeInTheDocument();
    expect(
      screen.getByText(/Period Roll is independent and never blocks this action/i)
    ).toBeInTheDocument();
  });

  it("does not report cleaned contender staging as a failure after publication", () => {
    renderCard({
      isChampionPromoted: true,
      activeProductionModelId: "champion",
      championDfuCount: 12_476,
      generatableCount: 5,
    });

    expect(screen.getByText("Production release published")).toBeInTheDocument();
    expect(
      screen.getByText(/Generate All creates 5 staged comparison forecasts/)
    ).toBeInTheDocument();
  });

  it("labels a per-cluster champion as routed instead of an ensemble", () => {
    renderCard({
      promotedExperiment: perClusterChampion,
      championConstituents: perClusterChampion.models,
    });

    expect(screen.getByText("Per-cluster routing")).toBeInTheDocument();
    expect(screen.queryByText("ensemble")).not.toBeInTheDocument();
  });
});
