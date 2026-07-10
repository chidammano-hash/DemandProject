import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModelReadinessCard } from "../ModelReadinessCard";
import type { ForecastAlgorithm } from "../forecastPanelShared";
import type { PromotionStatus, StagingSummary } from "@/api/queries/backtest-management";

const treeAlgo: ForecastAlgorithm = {
  id: "lgbm_cluster",
  type: "tree",
  enabled: true,
  forecast: true,
  compete: true,
  hasPredictions: true,
  accuracy: 92,
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

function promotion(sourceRunId: string): PromotionStatus {
  return {
    id: 1,
    model_id: "lgbm_cluster",
    promotion_type: "single",
    champion_experiment_id: null,
    plan_version: "2026-07",
    promoted_at: "2026-07-10T12:00:00Z",
    dfu_count: 1,
    total_rows: 6,
    promoted_by: "test",
    notes: null,
    source_run_id: sourceRunId,
    production_run_id: "production-run",
    candidate_checksum: null,
    production_checksum: null,
    archive_checksum: null,
    archived_at: null,
  };
}

function renderCard(props: Partial<React.ComponentProps<typeof ModelReadinessCard>> = {}) {
  const onGenerateAll = vi.fn();
  const base: React.ComponentProps<typeof ModelReadinessCard> = {
    forecastAlgos: [treeAlgo],
    trainingStatus: {
      lgbm_cluster: {
        model_id: "lgbm_cluster",
        type: "tree",
        trained: true,
        trained_at: null,
        training_mode: "production",
        n_dfus: 1,
        planning_date: null,
      },
    },
    staging: {},
    treeAlgos: [treeAlgo],
    trainedTreeCount: 1,
    allTreesTrained: true,
    isTraining: false,
    trainingModelId: null,
    generatingModelId: null,
    isGenerating: false,
    promotingModelId: null,
    isSubmitting: false,
    promotedModel: null,
    promotedExperiment: null,
    championConstituents: [],
    championMissingModels: [],
    championCanGenerate: false,
    championReady: false,
    championDfuCount: 0,
    isChampionPromoted: false,
    onTrain: () => {},
    onTrainAll: () => {},
    onGenerate: () => {},
    onGenerateAll,
    generatableCount: 1,
    onPromote: () => {},
    onGenerateChampion: () => {},
  };
  render(<ModelReadinessCard {...base} {...props} />);
  return { onGenerateAll };
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

  it("allows a new run for the currently promoted model to be promoted", () => {
    renderCard({
      staging: { lgbm_cluster: readyCandidate },
      promotedModel: promotion("old-run"),
    });

    expect(screen.getByRole("button", { name: "Promote" })).toBeEnabled();
  });

  it("marks only the exact published source run as promoted", () => {
    renderCard({
      staging: { lgbm_cluster: readyCandidate },
      promotedModel: promotion("new-run"),
    });

    expect(screen.getByRole("button", { name: "Promoted" })).toBeDisabled();
  });
});
