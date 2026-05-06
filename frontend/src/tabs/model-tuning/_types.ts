/**
 * Shared types for the Model Experimentation Studio tab.
 */
import type { TuningRun, ModelType } from "@/api/queries";

export interface ModelInfo {
  id: string;
  label: string;
  type: string;
  tunable: boolean;
  modelType?: ModelType;
}

export type PipelineStage = "clustering" | "backtest" | "tune" | "champion" | "forecast";

export type ModelDetailTab = "experiments" | "feature-lab" | "cluster-eda";

export type StatusFilter = "all" | "running" | "completed" | "failed";

export interface ModelSummaryCardData {
  best: number | null;
  runs: number;
  active: number;
  promoted: number | null;
}

export interface TabState {
  selectedModelId: string;
  modelDetailTab: ModelDetailTab;
  baselineId: number | null;
  candidateId: number | null;
  selectedRunForLogs: number | null;
  selectedRunForPromote: TuningRun | null;
  showBuilder: boolean;
  statusFilter: StatusFilter;
  page: number;
  sortCol: string;
  sortDir: "asc" | "desc";
}

export type TabAction =
  | { type: "SELECT_MODEL"; modelId: string }
  | { type: "SET_DETAIL_TAB"; tab: ModelDetailTab }
  | { type: "SELECT_ROW"; runId: number }
  | { type: "CLEAR_SELECTION" }
  | { type: "SET_LOGS"; runId: number | null }
  | { type: "SET_PROMOTE"; run: TuningRun | null }
  | { type: "SET_BUILDER"; open: boolean }
  | { type: "SET_STATUS_FILTER"; filter: StatusFilter }
  | { type: "SET_PAGE"; page: number }
  | { type: "TOGGLE_SORT"; col: string };
