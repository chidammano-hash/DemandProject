import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModelReadinessCard } from "../ModelReadinessCard";
import type { ForecastAlgorithm } from "../forecastPanelShared";

const treeAlgo: ForecastAlgorithm = {
  id: "lgbm_cluster",
  type: "tree",
  enabled: true,
  forecast: true,
  compete: true,
  hasPredictions: true,
  accuracy: 92,
};

function renderCard(props: Partial<React.ComponentProps<typeof ModelReadinessCard>> = {}) {
  const onGenerateAll = vi.fn();
  const base: React.ComponentProps<typeof ModelReadinessCard> = {
    forecastAlgos: [treeAlgo],
    trainingStatus: { lgbm_cluster: { model_id: "lgbm_cluster", type: "tree", trained: true, trained_at: null, training_mode: "production", n_dfus: 1, planning_date: null } },
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
});
