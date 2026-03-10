/**
 * PipelineBuilderPanel — submit multi-step job pipelines from the UI.
 *
 * Two sections:
 *   1. Pre-built templates (one-click S&OP / Inventory / Weekly refresh)
 *   2. Custom builder — pick any job types, reorder, name, and submit
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Plus, Trash2, ArrowRight, ChevronDown, ChevronUp, Loader2, CheckCircle2 } from "lucide-react";
import type { Job, JobType } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { GROUP_ICONS } from "./jobsShared";

// ---------------------------------------------------------------------------
// Pre-built templates
// ---------------------------------------------------------------------------

interface PipelineTemplate {
  id: string;
  label: string;
  description: string;
  steps: { type: string; params: Record<string, unknown> }[];
  estimatedMinutes: number;
}

const PIPELINE_TEMPLATES: PipelineTemplate[] = [
  {
    id: "sop_refresh",
    label: "Full S&OP Refresh",
    description: "End-to-end model refresh: cluster → backtest → champion → forecast → replenishment",
    steps: [
      { type: "cluster_pipeline", params: {} },
      { type: "backtest_lgbm", params: {} },
      { type: "champion_select", params: {} },
      { type: "generate_production_forecast", params: {} },
      { type: "compute_replenishment_plan", params: {} },
    ],
    estimatedMinutes: 90,
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
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Grace period after a step completes before declaring the pipeline done (ms). */
const PIPELINE_DONE_GRACE_MS = 10_000;

type TemplateStatus = "idle" | "submitting" | "queued" | "running" | "done";

interface Props {
  jobTypes: JobType[];
  activeJobs?: Job[];
  onSubmit: (args: { steps: { type: string; params: Record<string, unknown> }[]; label: string }) => void;
  isSubmitting?: boolean;
}

export function PipelineBuilderPanel({ jobTypes, activeJobs, onSubmit, isSubmitting }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [customSteps, setCustomSteps] = useState<{ type: string; params: Record<string, unknown> }[]>([]);
  const [selectedType, setSelectedType] = useState("");
  const [pipelineName, setPipelineName] = useState("");
  // Per-template lifecycle: idle → queued → running → done
  const [templateStatus, setTemplateStatus] = useState<Record<string, TemplateStatus>>({});
  // Ref so the effect can read current status without re-triggering on status changes
  const templateStatusRef = useRef<Record<string, TemplateStatus>>({});
  // Pending "done" timers — cancelled if the next pipeline step appears before they fire
  const pendingDoneRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  useEffect(() => {
    templateStatusRef.current = templateStatus;
  });

  // Sync template status with active jobs (runs only when activeJobs reference changes)
  useEffect(() => {
    const statuses = templateStatusRef.current;
    for (const t of PIPELINE_TEMPLATES) {
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
  }, [activeJobs]);

  // Cleanup timers on unmount
  useEffect(() => () => {
    for (const id of Object.values(pendingDoneRef.current)) clearTimeout(id);
  }, []);

  // O(1) lookup map for job types
  const jobTypeMap = useMemo(() => new Map(jobTypes.map((jt) => [jt.type_id, jt])), [jobTypes]);

  // Group job types for <optgroup> rendering
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

  function submitTemplate(template: PipelineTemplate) {
    setTemplateStatus((prev) => ({ ...prev, [template.id]: "queued" }));
    onSubmit({ steps: template.steps, label: template.label });
  }

  function submitCustom() {
    if (customSteps.length < 2 || !pipelineName.trim()) return;
    onSubmit({ steps: customSteps, label: pipelineName.trim() });
    setCustomSteps([]);
    setPipelineName("");
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
            {PIPELINE_TEMPLATES.length} templates
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
          {/* ── Pre-built templates ─────────────────────────────────────── */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Pre-built Templates
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              {PIPELINE_TEMPLATES.map((t) => (
                <div key={t.id} className="rounded-md border border-border bg-muted/20 p-3 flex flex-col gap-2">
                  <div className="flex items-start justify-between gap-1">
                    <span className="text-sm font-medium leading-tight">{t.label}</span>
                    <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                      {t.steps.length} steps
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-snug">{t.description}</p>
                  {/* Step badges */}
                  <div className="flex flex-wrap gap-1">
                    {t.steps.map((s, i) => {
                      const jt = jobTypeMap.get(s.type);
                      const group = jt?.group || "clustering";
                      const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
                      return (
                        <span
                          key={i}
                          className={`rounded px-1.5 py-0.5 text-xs ${cfg.bgColor} ${cfg.color}`}
                        >
                          {jt?.label || s.type}
                        </span>
                      );
                    })}
                  </div>
                  <div className="flex items-center justify-between mt-auto pt-1">
                    <span className="text-xs text-muted-foreground">~{t.estimatedMinutes}m</span>
                    {(() => {
                      const status = templateStatus[t.id] ?? "idle";
                      const busy = status === "queued" || status === "running";
                      return (
                        <button
                          onClick={() => submitTemplate(t)}
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
              ))}
            </div>
          </div>

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
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Pipeline name…"
                  value={pipelineName}
                  onChange={(e) => setPipelineName(e.target.value)}
                  className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm placeholder:text-muted-foreground"
                />
                <button
                  onClick={submitCustom}
                  disabled={customSteps.length < 2 || !pipelineName.trim() || isSubmitting}
                  className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
                >
                  <Play className="h-3.5 w-3.5" />
                  Run Custom Pipeline
                </button>
              </div>
              {customSteps.length > 0 && customSteps.length < 2 && (
                <p className="text-xs text-muted-foreground">Add at least 2 steps to run a pipeline.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
