/**
 * ModelTuningTab -- Model Experimentation Studio.
 *
 * Pipeline-aligned layout:
 *   Clustering -> Backtest -> Tune -> Champion -> Forecast -> Period Roll
 *   Customer Forecast is an independent generation-only stage.
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
import { PipelineReadinessBanner } from "@/components/PipelineReadinessBanner";
import { PageHeader } from "@/components/PageHeader";
import { Skeleton } from "@/components/Skeleton";
import { TabStrip } from "@/components/ui/tabs";

import {
  fetchPipelineConfig,
  modelTuningKeys,
  pipelineConfigKeys,
} from "@/api/queries/unified-model-tuning";

// Lazy: heavy stage panels — only the active stage's chunk is fetched.
const ClusterExperimentsPanel = lazy(() =>
  import("./clusters/ClusterExperimentsPanel").then((m) => ({ default: m.ClusterExperimentsPanel }))
);
const ChampionExperimentsPanel = lazy(() =>
  import("./champion/ChampionExperimentsPanel").then((m) => ({
    default: m.ChampionExperimentsPanel,
  }))
);
const ForecastPanel = lazy(() =>
  import("./forecast/ForecastPanel").then((m) => ({ default: m.ForecastPanel }))
);
const PeriodRollPanel = lazy(() =>
  import("./forecast/PeriodRollPanel").then((m) => ({ default: m.PeriodRollPanel }))
);
const CustomerForecastPanel = lazy(() =>
  import("./forecast/CustomerForecastPanel").then((m) => ({ default: m.CustomerForecastPanel }))
);
const BacktestStagePanel = lazy(() =>
  import("./model-tuning/BacktestStagePanel").then((m) => ({ default: m.BacktestStagePanel }))
);
const TuneStagePanel = lazy(() =>
  import("./model-tuning/TuneStagePanel").then((m) => ({ default: m.TuneStagePanel }))
);
// Lazy: modal/slide-over overlays — chunk only fetched once user opens them.
const ExperimentBuilder = lazy(() =>
  import("./model-tuning/ExperimentBuilder").then((m) => ({ default: m.ExperimentBuilder }))
);
const LogViewer = lazy(() =>
  import("./model-tuning/LogViewer").then((m) => ({ default: m.LogViewer }))
);
const EnhancedPromoteModal = lazy(() =>
  import("./model-tuning/EnhancedPromoteModal").then((m) => ({ default: m.EnhancedPromoteModal }))
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
    <div className="space-y-3 rounded-xl border border-border/60 bg-card p-4 shadow-card" aria-busy="true">
      <Skeleton className="h-5 w-48" />
      <Skeleton className="h-3 w-72" />
      <Skeleton className="h-48 w-full" />
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

  const handleStageChange = (nextStage: PipelineStage) => {
    if (nextStage === "tune" && !isTunable) {
      const tunableModel = ALL_MODELS.find((model) => model.tunable);
      if (tunableModel) {
        dispatch({ type: "SELECT_MODEL", modelId: tunableModel.id });
      }
    }
    setStage(nextStage);
  };

  return (
    <div className="flex flex-col gap-5 p-4 md:p-6">
      {/* ---- Header ---- */}
      <PageHeader
        icon={Layers}
        title="Forecasting"
        description="Run the item-location lifecycle and monthly roll, or independently generate customer-level forecasts."
      />

      {/* ---- Dependency readiness (e.g. clustering stale after a dim_sku reload) ---- */}
      <PipelineReadinessBanner />

      {/* ---- Pipeline Stage Tabs ---- */}
      <TabStrip
        aria-label="Forecasting pipeline stages"
        variant="pills"
        value={stage}
        onValueChange={(key) => handleStageChange(key as PipelineStage)}
        items={STAGE_TABS.map(({ key, label, icon }) => ({ key, label, icon }))}
      />

      {/* Stage panels — Suspense boundary fetches the active stage's chunk on demand. */}
      <Suspense fallback={<StageLoader />}>
        {/* Clustering stage */}
        {stage === "clustering" && <ClusterExperimentsPanel />}

        {/* Champion stage */}
        {stage === "champion" && <ChampionExperimentsPanel />}

        {/* Forecast stage */}
        {stage === "forecast" && <ForecastPanel />}

        {/* Generation-only customer forecast stage */}
        {stage === "customer-forecast" && <CustomerForecastPanel />}

        {/* Period Roll stage */}
        {stage === "period-roll" && <PeriodRollPanel />}

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
              queryClient.invalidateQueries({ queryKey: modelTuningKeys.all });
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
              queryClient.invalidateQueries({ queryKey: modelTuningKeys.all });
            }}
          />
        </Suspense>
      )}

      {/* ---- Floating AI Tuning Advisor ---- */}
      <AIAdvisorFAB />
    </div>
  );
}
