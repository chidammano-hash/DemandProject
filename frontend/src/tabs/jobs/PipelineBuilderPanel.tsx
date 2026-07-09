/**
 * PipelineBuilderPanel — submit multi-step job pipelines from the UI.
 *
 * Two sections:
 *   1. Pre-built templates (one-click S&OP / Inventory / Weekly refresh)
 *   2. Custom builder — pick any job types, reorder, name, and submit
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Plus, Trash2, ArrowRight, ChevronDown, ChevronUp, Loader2, CheckCircle2, Save } from "lucide-react";
import type { Job, JobType } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { GROUP_ICONS } from "./jobsShared";

// ---------------------------------------------------------------------------
// Pipeline bundles
// ---------------------------------------------------------------------------

interface PipelineStep {
  type: string;
  params: Record<string, unknown>;
}

interface PipelineBundle {
  id: string;
  label: string;
  description: string;
  steps: PipelineStep[];
  estimatedMinutes: number;
  source: "default" | "custom";
}

const CUSTOM_BUNDLES_STORAGE_KEY = "demandProject.pipelineBuilder.customBundles.v1";

function getBundleStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  const storage = window.localStorage;
  if (
    !storage ||
    typeof storage.getItem !== "function" ||
    typeof storage.setItem !== "function"
  ) {
    return null;
  }
  return storage;
}

const DEFAULT_PIPELINE_BUNDLES: PipelineBundle[] = [
  {
    id: "delta_data_load",
    label: "Delta Data Load",
    description: "Incremental ETL refresh using source-change detection.",
    steps: [
      { type: "etl_pipeline", params: { mode: "refresh", domains: null, parallel: false } },
    ],
    estimatedMinutes: 20,
    source: "default",
  },
  {
    id: "full_data_load",
    label: "Full Data Load",
    description: "Full ETL reload with parallel normalization, loading, and MV refresh.",
    steps: [
      { type: "etl_pipeline", params: { mode: "full", domains: null, parallel: true } },
    ],
    estimatedMinutes: 45,
    source: "default",
  },
  {
    id: "forecast_feature_prep",
    label: "Forecast Feature Prep",
    description: "Refresh SKU features, clusters, and customer analytics before modeling.",
    steps: [
      { type: "compute_sku_features", params: { time_window_months: 36 } },
      { type: "cluster_pipeline", params: {} },
      { type: "refresh_customer_analytics", params: {} },
    ],
    estimatedMinutes: 40,
    source: "default",
  },
  {
    id: "core_tree_backtests",
    label: "Core Tree Backtests",
    description: "Run LGBM, CatBoost, and XGBoost backtests, then load their results.",
    steps: [
      { type: "backtest_lgbm", params: {} },
      { type: "backtest_load_model", params: { model_id: "lgbm_cluster" } },
      { type: "backtest_catboost", params: {} },
      { type: "backtest_load_model", params: { model_id: "catboost_cluster" } },
      { type: "backtest_xgboost", params: {} },
      { type: "backtest_load_model", params: { model_id: "xgboost_cluster" } },
      { type: "refresh_forecast_views", params: {} },
    ],
    estimatedMinutes: 180,
    source: "default",
  },
  {
    id: "champion_selection",
    label: "Champion Selection",
    description: "Select the current champion model from loaded backtest results.",
    steps: [
      { type: "champion_select", params: {} },
    ],
    estimatedMinutes: 15,
    source: "default",
  },
  {
    id: "production_forecast",
    label: "Production Forecast",
    description: "Train production models, generate multi-step forecasts, and refresh views.",
    steps: [
      { type: "train_production_model", params: { all_models: true } },
      { type: "generate_production_forecast", params: { horizon: 24, confidence_intervals: true } },
      { type: "refresh_forecast_views", params: {} },
    ],
    estimatedMinutes: 75,
    source: "default",
  },
  {
    id: "inventory_refresh",
    label: "Inventory Refresh",
    description: "Recompute all inventory planning metrics: SS, EOQ, policies, health, exceptions",
    steps: [
      { type: "compute_safety_stock", params: {} },
      { type: "compute_eoq", params: {} },
      { type: "assign_policies", params: {} },
      { type: "refresh_health_scores", params: {} },
      { type: "generate_exceptions", params: {} },
    ],
    estimatedMinutes: 15,
    source: "default",
  },
  {
    id: "weekly_data_refresh",
    label: "Weekly Data Refresh",
    description: "Refresh classification, variability, signals, and storyboard exceptions",
    steps: [
      { type: "seasonality_pipeline", params: {} },
      { type: "classify_abc_xyz", params: {} },
      { type: "compute_variability", params: {} },
      { type: "compute_demand_signals", params: {} },
      { type: "generate_storyboard", params: {} },
    ],
    estimatedMinutes: 20,
    source: "default",
  },
];

function loadCustomBundles(): PipelineBundle[] {
  const storage = getBundleStorage();
  if (!storage) return [];
  try {
    const raw = storage.getItem(CUSTOM_BUNDLES_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((bundle) =>
        typeof bundle?.id === "string" &&
        typeof bundle?.label === "string" &&
        Array.isArray(bundle?.steps)
      )
      .map((bundle) => ({
        id: bundle.id,
        label: bundle.label,
        description: typeof bundle.description === "string" ? bundle.description : "User-created pipeline bundle.",
        steps: bundle.steps,
        estimatedMinutes: typeof bundle.estimatedMinutes === "number"
          ? bundle.estimatedMinutes
          : Math.max(5, bundle.steps.length * 10),
        source: "custom",
      }));
  } catch {
    return [];
  }
}

function saveCustomBundles(bundles: PipelineBundle[]): void {
  const storage = getBundleStorage();
  if (!storage) return;
  try {
    storage.setItem(CUSTOM_BUNDLES_STORAGE_KEY, JSON.stringify(bundles));
  } catch {
    // Storage can be disabled or quota-limited; custom bundles still work in memory.
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Grace period after a step completes before declaring the pipeline done (ms). */
const PIPELINE_DONE_GRACE_MS = 10_000;

type TemplateStatus = "idle" | "submitting" | "queued" | "running" | "done";

interface Props {
  jobTypes: JobType[];
  activeJobs?: Job[];
  onSubmit: (args: { steps: PipelineStep[]; label: string }) => void;
  isSubmitting?: boolean;
}

export function PipelineBuilderPanel({ jobTypes, activeJobs, onSubmit, isSubmitting }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [customSteps, setCustomSteps] = useState<PipelineStep[]>([]);
  const [selectedType, setSelectedType] = useState("");
  const [pipelineName, setPipelineName] = useState("");
  const [customBundles, setCustomBundles] = useState<PipelineBundle[]>(loadCustomBundles);
  // Per-template lifecycle: idle → queued → running → done
  const [templateStatus, setTemplateStatus] = useState<Record<string, TemplateStatus>>({});
  // Ref so the effect can read current status without re-triggering on status changes
  const templateStatusRef = useRef<Record<string, TemplateStatus>>({});
  // Pending "done" timers — cancelled if the next pipeline step appears before they fire
  const pendingDoneRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  useEffect(() => {
    templateStatusRef.current = templateStatus;
  });

  useEffect(() => {
    saveCustomBundles(customBundles);
  }, [customBundles]);

  // Sync template status with active jobs (runs only when activeJobs reference changes)
  useEffect(() => {
    const statuses = templateStatusRef.current;
    for (const t of [...DEFAULT_PIPELINE_BUNDLES, ...customBundles]) {
      const current = statuses[t.id];
      if (!current || current === "idle" || current === "done") continue;

      const isActive = (activeJobs ?? []).some(
        (j) => j.job_label?.startsWith(`[${t.label}`) || j.job_label === t.label,
      );

      if (isActive) {
        // Cancel any pending "done" — the next step started
        if (pendingDoneRef.current[t.id]) {
          clearTimeout(pendingDoneRef.current[t.id]);
          delete pendingDoneRef.current[t.id];
        }
        if (current !== "running") {
          setTemplateStatus((p) => ({ ...p, [t.id]: "running" }));
        }
      } else if (current === "running" && !pendingDoneRef.current[t.id]) {
        // Was running but no matching active job — could be the gap between pipeline steps.
        // Wait 10s before declaring done so the next step has time to appear.
        const tid = t.id;
        pendingDoneRef.current[tid] = setTimeout(() => {
          setTemplateStatus((p) => ({ ...p, [tid]: "done" }));
          delete pendingDoneRef.current[tid];
        }, PIPELINE_DONE_GRACE_MS);
      }
    }
  }, [activeJobs, customBundles]);

  // Cleanup timers on unmount
  useEffect(() => () => {
    for (const id of Object.values(pendingDoneRef.current)) clearTimeout(id);
  }, []);

  // O(1) lookup map for job types
  const jobTypeMap = useMemo(() => new Map(jobTypes.map((jt) => [jt.type_id, jt])), [jobTypes]);
  const allBundles = useMemo(
    () => [...DEFAULT_PIPELINE_BUNDLES, ...customBundles],
    [customBundles],
  );

  // Group job types for <optgroup> rendering.
  const byGroup = useMemo(() => {
    const map: Record<string, JobType[]> = {};
    for (const jt of jobTypes) {
      const g = jt.group || "other";
      if (!map[g]) map[g] = [];
      map[g].push(jt);
    }
    return map;
  }, [jobTypes]);

  function addStep() {
    if (!selectedType) return;
    setCustomSteps((prev) => [...prev, { type: selectedType, params: {} }]);
    setSelectedType("");
  }

  function removeStep(idx: number) {
    setCustomSteps((prev) => prev.filter((_, i) => i !== idx));
  }

  function visualGroupFor(type: string): string {
    const group = jobTypeMap.get(type)?.group || "clustering";
    if (GROUP_CONFIG[group]) return group;
    if (group === "features") return "seasonality";
    if (group === "backtest_load") return "backtest";
    return group === "etl" ? "platform" : "clustering";
  }

  function submitBundle(bundle: PipelineBundle) {
    setTemplateStatus((prev) => ({ ...prev, [bundle.id]: "queued" }));
    onSubmit({ steps: bundle.steps, label: bundle.label });
  }

  function submitCustom() {
    if (customSteps.length < 1 || !pipelineName.trim()) return;
    onSubmit({ steps: customSteps, label: pipelineName.trim() });
    setCustomSteps([]);
    setPipelineName("");
  }

  function saveCustomBundle() {
    const label = pipelineName.trim();
    if (customSteps.length < 1 || !label) return;
    const bundle: PipelineBundle = {
      id: `custom_${Date.now().toString(36)}`,
      label,
      description: "User-created pipeline bundle.",
      steps: customSteps,
      estimatedMinutes: Math.max(5, customSteps.length * 10),
      source: "custom",
    };
    setCustomBundles((prev) => [...prev, bundle]);
    setCustomSteps([]);
    setPipelineName("");
  }

  function deleteCustomBundle(id: string) {
    setCustomBundles((prev) => prev.filter((bundle) => bundle.id !== id));
  }

  function renderBundleCard(bundle: PipelineBundle) {
    return (
      <div key={bundle.id} className="rounded-md border border-border bg-muted/20 p-3 flex flex-col gap-2">
        <div className="flex items-start justify-between gap-1">
          <span className="text-sm font-medium leading-tight">{bundle.label}</span>
          <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
            {bundle.steps.length} step{bundle.steps.length === 1 ? "" : "s"}
          </span>
        </div>
        <p className="text-xs text-muted-foreground leading-snug">{bundle.description}</p>
        <div className="flex flex-wrap gap-1">
          {bundle.steps.map((s, i) => {
            const jt = jobTypeMap.get(s.type);
            const group = visualGroupFor(s.type);
            const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
            return (
              <span
                key={`${s.type}-${i}`}
                className={`rounded px-1.5 py-0.5 text-xs ${cfg.bgColor} ${cfg.color}`}
              >
                {jt?.label || s.type}
              </span>
            );
          })}
        </div>
        <div className="flex items-center justify-between gap-2 mt-auto pt-1">
          <span className="text-xs text-muted-foreground">~{bundle.estimatedMinutes}m</span>
          <div className="flex items-center gap-1">
            {bundle.source === "custom" && (
              <button
                type="button"
                onClick={() => deleteCustomBundle(bundle.id)}
                className="rounded border border-border p-1 text-muted-foreground hover:bg-muted"
                aria-label={`Delete ${bundle.label}`}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            )}
            {(() => {
              const status = templateStatus[bundle.id] ?? "idle";
              const busy = status === "queued" || status === "running";
              return (
                <button
                  type="button"
                  onClick={() => submitBundle(bundle)}
                  disabled={isSubmitting || busy}
                  className={`flex items-center gap-1 rounded px-2 py-1 text-xs disabled:cursor-not-allowed transition-colors ${
                    status === "done"
                      ? "bg-muted text-muted-foreground border border-border"
                      : busy
                      ? "bg-blue-600/20 text-blue-700 dark:text-blue-400 border border-blue-600/30"
                      : "bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  }`}
                >
                  {status === "queued" && (
                    <><Loader2 className="h-3 w-3 animate-spin" />Queued</>
                  )}
                  {status === "running" && (
                    <><Loader2 className="h-3 w-3 animate-spin" />Running</>
                  )}
                  {status === "done" && (
                    <><CheckCircle2 className="h-3 w-3" />Done</>
                  )}
                  {(status === "idle" || status === "submitting") && (
                    <><Play className="h-3 w-3" />Run</>
                  )}
                </button>
              );
            })()}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card">
      {/* Header */}
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <Play className="h-4 w-4 text-muted-foreground" />
          <span className="font-semibold text-sm">Pipeline Builder</span>
          <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {allBundles.length} bundles
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-border px-4 pb-4 pt-3 space-y-5">
          {/* ── Default bundles ─────────────────────────────────────────── */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Default Bundles
            </p>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {DEFAULT_PIPELINE_BUNDLES.map(renderBundleCard)}
            </div>
          </div>

          {customBundles.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                Saved Bundles
              </p>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {customBundles.map(renderBundleCard)}
              </div>
            </div>
          )}

          {/* ── Custom builder ──────────────────────────────────────────── */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Custom Pipeline
            </p>
            <div className="space-y-3">
              {/* Step picker */}
              <div className="flex gap-2">
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm"
                >
                  <option value="">Select a job type…</option>
                  {Object.entries(byGroup).map(([group, types]) => {
                    const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
                    return (
                      <optgroup key={group} label={cfg.label}>
                        {types.map((jt) => (
                          <option key={jt.type_id} value={jt.type_id}>
                            {jt.label}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
                <button
                  onClick={addStep}
                  disabled={!selectedType}
                  className="flex items-center gap-1 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-40"
                >
                  <Plus className="h-3.5 w-3.5" />
                  Add
                </button>
              </div>

              {/* Step list */}
              {customSteps.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-border bg-muted/20 px-3 py-2">
                  {customSteps.map((s, i) => {
                    const jt = jobTypeMap.get(s.type);
                    const group = jt?.group || "clustering";
                    const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
                    const Icon = GROUP_ICONS[group] || GROUP_ICONS.clustering;
                    return (
                      <span key={i} className="flex items-center gap-1">
                        {i > 0 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
                        <span
                          className={`flex items-center gap-1 rounded px-2 py-0.5 text-xs ${cfg.bgColor} ${cfg.color}`}
                        >
                          <Icon className="h-3 w-3" />
                          {jt?.label || s.type}
                          <button
                            onClick={() => removeStep(i)}
                            className="ml-0.5 hover:opacity-60"
                            aria-label={`Remove ${s.type}`}
                          >
                            <Trash2 className="h-2.5 w-2.5" />
                          </button>
                        </span>
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Name + submit */}
              <div className="flex flex-wrap gap-2">
                <input
                  type="text"
                  placeholder="Pipeline name…"
                  value={pipelineName}
                  onChange={(e) => setPipelineName(e.target.value)}
                  className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm placeholder:text-muted-foreground"
                />
                <button
                  type="button"
                  onClick={saveCustomBundle}
                  disabled={customSteps.length < 1 || !pipelineName.trim()}
                  className="flex items-center gap-1 rounded-md border border-input px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-40"
                >
                  <Save className="h-3.5 w-3.5" />
                  Save Bundle
                </button>
                <button
                  type="button"
                  onClick={submitCustom}
                  disabled={customSteps.length < 1 || !pipelineName.trim() || isSubmitting}
                  className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
                >
                  <Play className="h-3.5 w-3.5" />
                  Run Custom Bundle
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
