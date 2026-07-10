/**
 * Constants, the model-grid derivation helper, and the tab reducer for
 * the Model Experimentation Studio.
 */
import {
  FlaskConical,
  BarChart3,
  Microscope,
  Target,
  Crown,
  SlidersHorizontal,
  TrendingUp,
} from "lucide-react";

import type { ModelType } from "@/api/queries";
import type { PipelineAlgorithm } from "@/api/queries/unified-model-tuning";
import { MODEL_LABELS as MODEL_LABEL_FALLBACK } from "@/lib/model-labels";

import type { ModelDetailTab, ModelInfo, PipelineStage, TabAction, TabState } from "./_types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Fallback model list for initial render before pipeline config loads */
export const DEFAULT_MODELS: ModelInfo[] = [
  { id: "lgbm_cluster", label: "LightGBM", type: "tree", tunable: true, modelType: "lgbm" },
  { id: "chronos2_enriched", label: "Chronos 2E", type: "foundation", tunable: false },
  { id: "mstl", label: "MSTL", type: "statistical", tunable: false },
  { id: "nbeats", label: "N-BEATS", type: "deep_learning", tunable: false },
  { id: "nhits", label: "N-HiTS", type: "deep_learning", tunable: false },
];

/** Map algorithm id to ModelType for tunable models (used for experiment API routing) */
export const ID_TO_MODEL_TYPE: Record<string, ModelType> = {
  lgbm_cluster: "lgbm",
};

/** Derive the model grid from pipeline config algorithms */
export function deriveModelsFromConfig(algorithms: Record<string, PipelineAlgorithm>): ModelInfo[] {
  return Object.entries(algorithms)
    .filter(([, algo]) => algo.enabled)
    .map(([id, algo]) => ({
      id,
      label:
        ((algo as unknown as Record<string, unknown>).display_name as string) ||
        MODEL_LABEL_FALLBACK[id] ||
        id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      type: algo.type,
      tunable: algo.tune,
      modelType: ID_TO_MODEL_TYPE[id] as ModelType | undefined,
    }));
}

export const TYPE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  foundation: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  statistical: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  deep_learning: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
};

// Pipeline stage tabs
export const STAGE_TABS: { key: PipelineStage; label: string; icon: typeof FlaskConical }[] = [
  { key: "clustering", label: "Clustering", icon: Target },
  { key: "backtest", label: "Backtest", icon: FlaskConical },
  { key: "tune", label: "Tune", icon: SlidersHorizontal },
  { key: "champion", label: "Champion", icon: Crown },
  { key: "forecast", label: "Forecast", icon: TrendingUp },
];

// Sub-tabs shown when a tunable model is selected on the Tune stage
export const MODEL_DETAIL_TABS: {
  key: ModelDetailTab;
  label: string;
  icon: typeof FlaskConical;
}[] = [
  { key: "experiments", label: "Experiments", icon: FlaskConical },
  { key: "feature-lab", label: "Feature Lab", icon: Microscope },
  { key: "cluster-eda", label: "Cluster EDA", icon: BarChart3 },
];

export const PAGE_SIZE = 25;

export const INITIAL_STATE: TabState = {
  selectedModelId: "lgbm_cluster",
  modelDetailTab: "experiments",
  baselineId: null,
  candidateId: null,
  selectedRunForLogs: null,
  selectedRunForPromote: null,
  showBuilder: false,
  statusFilter: "all",
  page: 0,
  sortCol: "run_id",
  sortDir: "desc",
};

export function tabReducer(state: TabState, action: TabAction): TabState {
  switch (action.type) {
    case "SELECT_MODEL":
      if (action.modelId === state.selectedModelId) return state;
      return {
        ...state,
        selectedModelId: action.modelId,
        modelDetailTab: "experiments",
        baselineId: null,
        candidateId: null,
        selectedRunForPromote: null,
        selectedRunForLogs: null,
        page: 0,
      };
    case "SET_DETAIL_TAB":
      return { ...state, modelDetailTab: action.tab };
    case "SELECT_ROW": {
      if (state.baselineId === null) {
        return { ...state, baselineId: action.runId };
      }
      if (state.candidateId === null) {
        if (action.runId === state.baselineId) return state;
        return { ...state, candidateId: action.runId };
      }
      return { ...state, baselineId: action.runId, candidateId: null };
    }
    case "CLEAR_SELECTION":
      return { ...state, baselineId: null, candidateId: null };
    case "SET_LOGS":
      return { ...state, selectedRunForLogs: action.runId };
    case "SET_PROMOTE":
      return { ...state, selectedRunForPromote: action.run };
    case "SET_BUILDER":
      return { ...state, showBuilder: action.open };
    case "SET_STATUS_FILTER":
      return { ...state, statusFilter: action.filter, page: 0 };
    case "SET_PAGE":
      return { ...state, page: action.page };
    case "TOGGLE_SORT":
      if (state.sortCol === action.col) {
        return { ...state, sortDir: state.sortDir === "asc" ? "desc" : "asc" };
      }
      return { ...state, sortCol: action.col, sortDir: "desc" };
    default:
      return state;
  }
}
