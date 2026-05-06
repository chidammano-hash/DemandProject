/**
 * ModelTuningTab -- Model Experimentation Studio.
 *
 * Pipeline-aligned layout:
 *   Clustering -> Backtest -> Tune -> Champion -> Forecast
 *
 * The "Backtest" stage shows ALL models with run/load actions.
 * The "Tune" stage shows only tunable models with experiment UI.
 * The "Forecast" stage triggers production forecast generation.
 */
import { lazy, Suspense, useMemo, useReducer, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers } from "lucide-react";

import { type ModelType } from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { cn } from "@/lib/utils";

import {
  fetchPipelineConfig,
  pipelineConfigKeys,
} from "@/api/queries/unified-model-tuning";

// Lazy: heavy stage panels — only the active stage's chunk is fetched.
const ClusterExperimentsPanel = lazy(() =>
  import("./clusters/ClusterExperimentsPanel").then((m) => ({ default: m.ClusterExperimentsPanel })),
);
const ChampionExperimentsPanel = lazy(() =>
  import("./champion/ChampionExperimentsPanel").then((m) => ({ default: m.ChampionExperimentsPanel })),
);
const ForecastPanel = lazy(() =>
  import("./forecast/ForecastPanel").then((m) => ({ default: m.ForecastPanel })),
);
const BacktestStagePanel = lazy(() =>
  import("./model-tuning/BacktestStagePanel").then((m) => ({ default: m.BacktestStagePanel })),
);
const TuneStagePanel = lazy(() =>
  import("./model-tuning/TuneStagePanel").then((m) => ({ default: m.TuneStagePanel })),
);
// Lazy: modal/slide-over overlays — chunk only fetched once user opens them.
const ExperimentBuilder = lazy(() =>
  import("./model-tuning/ExperimentBuilder").then((m) => ({ default: m.ExperimentBuilder })),
);
const LogViewer = lazy(() =>
  import("./model-tuning/LogViewer").then((m) => ({ default: m.LogViewer })),
);
const EnhancedPromoteModal = lazy(() =>
  import("./model-tuning/EnhancedPromoteModal").then((m) => ({ default: m.EnhancedPromoteModal })),
);
// Eager: small floating button always visible.
import { AIAdvisorFAB } from "./model-tuning/AIAdvisorFAB";
import {
  DEFAULT_MODELS,
  INITIAL_STATE,
  STAGE_TABS,
  deriveModelsFromConfig,
  tabReducer,
} from "./model-tuning/_helpers";
import type { PipelineStage } from "./model-tuning/_types";

function StageLoader() {
  return (
    <div className="flex items-center justify-center text-sm text-muted-foreground rounded-md border border-dashed h-64">
      Loading...
    </div>
  );
}

export default function ModelTuningTab() {
  useChartColors(); // ensure theme context
  const queryClient = useQueryClient();

  // ---- Consolidated state via reducer ------------------------------------
  const [state, dispatch] = useReducer(tabReducer, INITIAL_STATE);
  const {
    selectedModelId,
    modelDetailTab,
    baselineId,
    candidateId,
    selectedRunForLogs,
    selectedRunForPromote,
    showBuilder,
    statusFilter,
    page,
    sortCol,
    sortDir,
  } = state;

  // Non-reducer state (independent concerns)
  const [stage, setStage] = useState<PipelineStage>("backtest");
  const [execLag, setExecLag] = useState<number | undefined>(undefined);

  // ---- Fetch pipeline config for dynamic model roster ----------------------
  const { data: pipelineConfig } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: 60_000, // 1 min -- config rarely changes
  });

  // Derive model grid from pipeline config, falling back to hardcoded defaults
  const ALL_MODELS = useMemo(() => {
    if (!pipelineConfig?.algorithms) return DEFAULT_MODELS;
    const derived = deriveModelsFromConfig(pipelineConfig.algorithms);
    return derived.length > 0 ? derived : DEFAULT_MODELS;
  }, [pipelineConfig]);

  // Resolve selected model info
  const selectedModelInfo = ALL_MODELS.find((m) => m.id === selectedModelId) ?? ALL_MODELS[0];
  const selectedModel: ModelType = selectedModelInfo.modelType ?? "lgbm";
  const isTunable = selectedModelInfo.tunable;

  return (
    <div className="flex flex-col gap-5 p-4 md:p-6">
      {/* ---- Header ---- */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Layers className="h-5 w-5 text-muted-foreground" />
            Model Experimentation Studio
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Run experiments, compare results, and promote to production.
          </p>
        </div>
      </div>

      {/* ---- Pipeline Stage Tabs ---- */}
      <div className="flex gap-1 border-b border-border pb-1">
        {STAGE_TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setStage(key)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors",
              stage === key
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Stage panels — Suspense boundary fetches the active stage's chunk on demand. */}
      <Suspense fallback={<StageLoader />}>
        {/* Clustering stage */}
        {stage === "clustering" && <ClusterExperimentsPanel />}

        {/* Champion stage */}
        {stage === "champion" && <ChampionExperimentsPanel />}

        {/* Forecast stage */}
        {stage === "forecast" && <ForecastPanel />}

        {/* Backtest stage -- ALL models with run/load actions */}
        {stage === "backtest" && (
          <BacktestStagePanel
            models={ALL_MODELS}
            selectedModelId={selectedModelId}
            selectedModelInfo={selectedModelInfo}
            onSelectModel={(modelId) => dispatch({ type: "SELECT_MODEL", modelId })}
          />
        )}

        {/* Tune stage -- tunable models only with experiment UI */}
        {stage === "tune" && (
          <TuneStagePanel
            models={ALL_MODELS}
            selectedModelId={selectedModelId}
            selectedModelInfo={selectedModelInfo}
            selectedModel={selectedModel}
            isTunable={isTunable}
            modelDetailTab={modelDetailTab}
            baselineId={baselineId}
            candidateId={candidateId}
            page={page}
            sortCol={sortCol}
            sortDir={sortDir}
            statusFilter={statusFilter}
            execLag={execLag}
            setExecLag={setExecLag}
            onSelectModel={(modelId) => dispatch({ type: "SELECT_MODEL", modelId })}
            onSetDetailTab={(tab) => dispatch({ type: "SET_DETAIL_TAB", tab })}
            onSetStatusFilter={(filter) => dispatch({ type: "SET_STATUS_FILTER", filter })}
            onSelectRow={(runId) => dispatch({ type: "SELECT_ROW", runId })}
            onClearSelection={() => dispatch({ type: "CLEAR_SELECTION" })}
            onToggleSort={(col) => dispatch({ type: "TOGGLE_SORT", col })}
            onSetPage={(p) => dispatch({ type: "SET_PAGE", page: p })}
            onShowLogs={(runId) => dispatch({ type: "SET_LOGS", runId })}
            onPromote={(run) => dispatch({ type: "SET_PROMOTE", run })}
            onOpenBuilder={() => dispatch({ type: "SET_BUILDER", open: true })}
          />
        )}
      </Suspense>

      {/* ---- Experiment Builder Modal — only mount when actually open so its chunk loads on first use. ---- */}
      {isTunable && showBuilder && (
        <Suspense fallback={null}>
          <ExperimentBuilder
            model={selectedModel}
            open={showBuilder}
            onClose={() => dispatch({ type: "SET_BUILDER", open: false })}
            onSubmitted={() => {
              dispatch({ type: "SET_BUILDER", open: false });
              queryClient.invalidateQueries({ queryKey: ["model-tuning-runs", selectedModel] });
              queryClient.invalidateQueries({ queryKey: ["model-summary", selectedModel] });
            }}
          />
        </Suspense>
      )}

      {/* ---- Log Viewer Slide-Over ---- */}
      {isTunable && selectedRunForLogs !== null && (
        <Suspense fallback={null}>
          <LogViewer
            model={selectedModel}
            runId={selectedRunForLogs ?? 0}
            open={selectedRunForLogs !== null}
            onClose={() => dispatch({ type: "SET_LOGS", runId: null })}
          />
        </Suspense>
      )}

      {/* ---- Enhanced Promote Modal ---- */}
      {selectedRunForPromote && isTunable && (
        <Suspense fallback={null}>
          <EnhancedPromoteModal
            model={selectedModel}
            run={selectedRunForPromote}
            open={true}
            onClose={() => dispatch({ type: "SET_PROMOTE", run: null })}
            onPromoted={() => {
              dispatch({ type: "SET_PROMOTE", run: null });
              queryClient.invalidateQueries({ queryKey: ["model-tuning-runs", selectedModel] });
              queryClient.invalidateQueries({ queryKey: ["model-summary", selectedModel] });
            }}
          />
        </Suspense>
      )}

      {/* ---- Floating AI Tuning Advisor ---- */}
      <AIAdvisorFAB />
    </div>
  );
}
