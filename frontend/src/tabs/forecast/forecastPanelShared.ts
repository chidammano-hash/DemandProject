/**
 * Shared helpers, constants, and types for the ForecastPanel and its sub-cards.
 */
import type { PipelineAlgorithm } from "@/api/queries/unified-model-tuning";
import type { BacktestSummary } from "@/api/queries/backtest-management";
import { isForecastModelId } from "@/lib/model-labels";

// ---------------------------------------------------------------------------
// Query keys & stale times
// ---------------------------------------------------------------------------

export const forecastPanelKeys = {
  versions: ["forecast-panel", "versions"] as const,
  jobs: (offset: number) => ["forecast-panel", "jobs", offset] as const,
  publishPipeline: (pipelineId: string) =>
    ["forecast-panel", "publish-pipeline", pipelineId] as const,
};

export const STALE = {
  VERSIONS: 30_000,
  CONFIG: 60_000,
  JOBS: 10_000,
} as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export interface ForecastAlgorithm {
  id: string;
  type: string;
  enabled: boolean;
  forecast: boolean;
  compete: boolean;
  hasPredictions: boolean;
  accuracy: number | null;
}

export function deriveForecastAlgos(
  algorithms: Record<string, PipelineAlgorithm> | undefined,
  backtestSummary: BacktestSummary | undefined
): ForecastAlgorithm[] {
  if (!algorithms) return [];
  return Object.entries(algorithms)
    .filter(([id, a]) => isForecastModelId(id) && a.forecast)
    .map(([id, a]) => {
      const summary = backtestSummary?.[id];
      return {
        id,
        type: a.type,
        enabled: a.enabled,
        forecast: a.forecast,
        compete: a.compete,
        hasPredictions: summary?.has_predictions ?? false,
        accuracy: summary?.current_accuracy ?? null,
      };
    });
}

/** True for model types that require a persisted final-refit artifact. */
export function requiresTraining(type: string): boolean {
  return type === "tree" || type === "deep_learning";
}

// ---------------------------------------------------------------------------
// Extended types for fields present in YAML but not in PipelineConfig interface
// ---------------------------------------------------------------------------

export interface ProdConfigExtended {
  horizon_months: number;
  min_history_months: number;
  cold_start_model_id: string;
  cold_start_min_months: number;
  recursive?: boolean;
  confidence_interval?: {
    enabled: boolean;
    source_model_ids?: string[];
  };
}
