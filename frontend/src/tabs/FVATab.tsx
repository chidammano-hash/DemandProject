/**
 * Forecast Value Added (FVA) & ROI Tracking Tab (Spec 08-07).
 *
 * Shows: staged FVA ladder, ceiling benchmark, intervention timeline, ROI KPI cards.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchFVAWaterfall,
  fetchFVAROI,
  fetchFVAInterventions,
  fvaKeys,
  STALE_PLATFORM,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { SnapshotAccuracyPanel } from "./fva/SnapshotAccuracyPanel";

type FVAStage = {
  stage_id: string;
  label: string;
  description?: string;
  accuracy_pct: number | null;
  delta_vs_prev: number | null;
  state: "actual" | "missing" | "planned";
  n_rows?: number;
};

const DEFAULT_STAGES: FVAStage[] = [
  {
    stage_id: "external",
    label: "External",
    description: "Current ERP or external forecast.",
    accuracy_pct: null,
    delta_vs_prev: null,
    state: "missing",
  },
  {
    stage_id: "champion",
    label: "Champion",
    description: "Best measured statistical or ML model.",
    accuracy_pct: null,
    delta_vs_prev: null,
    state: "missing",
  },
  {
    stage_id: "ai_adjusted",
    label: "AI Adjusted",
    description: "Reserved for AI-assisted interventions.",
    accuracy_pct: null,
    delta_vs_prev: null,
    state: "planned",
  },
  {
    stage_id: "planner_adjusted",
    label: "Planner Adjusted",
    description: "Reserved for measured planner overrides.",
    accuracy_pct: null,
    delta_vs_prev: null,
    state: "planned",
  },
];

const STAGE_STYLES: Record<string, string> = {
  external: "border-sky-300/70 bg-sky-50/70 dark:bg-sky-950/20",
  champion: "border-blue-300/70 bg-blue-50/70 dark:bg-blue-950/20",
  ai_adjusted: "border-amber-300/70 bg-amber-50/70 dark:bg-amber-950/20",
  planner_adjusted: "border-violet-300/70 bg-violet-50/70 dark:bg-violet-950/20",
};

function formatCurrencyCompact(value: number): string {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

function formatAccuracy(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function stageValueLabel(stage: FVAStage): string {
  if (stage.state === "planned") return "Coming Soon";
  if (stage.state === "missing") return "No data";
  return formatAccuracy(stage.accuracy_pct);
}

function buildHeadline(stages: FVAStage[], benchmark?: FVAStage | null): string {
  const messages: string[] = [];
  const external = stages.find((stage) => stage.stage_id === "external");
  const champion = stages.find((stage) => stage.stage_id === "champion");
  const ai = stages.find((stage) => stage.stage_id === "ai_adjusted");
  const planner = stages.find((stage) => stage.stage_id === "planner_adjusted");

  if (external?.delta_vs_prev != null) {
    messages.push(
      `External accuracy changed ${external.delta_vs_prev > 0 ? "+" : ""}${external.delta_vs_prev.toFixed(1)} pts.`
    );
  }
  if (champion?.delta_vs_prev != null) {
    messages.push(
      `Champion adds ${champion.delta_vs_prev > 0 ? "+" : ""}${champion.delta_vs_prev.toFixed(1)} pts vs External.`
    );
  }
  if (ai?.state === "planned" || planner?.state === "planned") {
    messages.push(
      "AI and Planner stages are reserved for measured interventions as those workflows come online."
    );
  }
  if (benchmark?.accuracy_pct != null) {
    messages.push(`Ceiling benchmark is ${benchmark.accuracy_pct.toFixed(1)}%.`);
  }

  return messages.join(" ");
}

export default function FVATab() {
  const [months, setMonths] = useState(12);

  const { data: waterfall } = useQuery({
    queryKey: fvaKeys.waterfall(months),
    queryFn: () => fetchFVAWaterfall(months),
    staleTime: STALE_PLATFORM,
  });

  const { data: roi } = useQuery({
    queryKey: fvaKeys.roi(months),
    queryFn: () => fetchFVAROI(months),
    staleTime: STALE_PLATFORM,
  });

  const { data: interventions } = useQuery({
    queryKey: fvaKeys.interventions,
    queryFn: () => fetchFVAInterventions(20),
    staleTime: STALE_PLATFORM,
  });

  const stages: FVAStage[] = waterfall?.waterfall?.stages ?? DEFAULT_STAGES;
  const benchmark: FVAStage | null = waterfall?.waterfall?.benchmark ?? null;
  const headline = buildHeadline(stages, benchmark);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Forecast Value Added</h2>
          <p className="text-sm text-muted-foreground">
            Track the business impact of planning interventions
          </p>
        </div>
        <select
          value={months}
          onChange={(e) => setMonths(Number(e.target.value))}
          className="rounded border border-border bg-card px-3 py-1.5 text-sm"
          aria-label="Lookback window"
        >
          {[3, 6, 12, 24].map((m) => (
            <option key={m} value={m}>
              {m} months
            </option>
          ))}
        </select>
      </div>

      {/* ROI KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <KpiCard
          label="Total Interventions"
          value={(roi?.total_interventions ?? 0).toLocaleString()}
        />
        <KpiCard label="Measured" value={(roi?.measured ?? 0).toLocaleString()} />
        <KpiCard
          label="Estimated Impact"
          value={formatCurrencyCompact(roi?.total_estimated_impact ?? 0)}
        />
        <KpiCard
          label="Actual Impact"
          value={formatCurrencyCompact(roi?.total_actual_impact ?? 0)}
        />
      </div>

      {/* FVA Ladder */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <h3 className="text-sm font-medium text-foreground">Forecast Value Ladder</h3>
            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
              {headline ||
                "Track how accuracy improves from a simple seasonal baseline through production forecasting and, later, measured AI and planner interventions."}
            </p>
          </div>
          <div className="min-w-[180px] rounded-lg border border-emerald-300/70 bg-emerald-50/70 p-3 dark:bg-emerald-950/20">
            <p className="text-xs font-medium text-foreground">
              {benchmark?.label ?? "Ceiling Benchmark"}
            </p>
            <p className="mt-1 font-mono text-2xl font-semibold text-emerald-700 dark:text-emerald-300">
              {benchmark?.accuracy_pct != null ? formatAccuracy(benchmark.accuracy_pct) : "—"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {benchmark?.accuracy_pct != null
                ? "Reference best-case benchmark, kept outside the stage ladder."
                : "No ceiling benchmark available in the selected window."}
            </p>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3 xl:grid-cols-5">
          {stages.map((stage, index) => (
            <div
              key={stage.stage_id}
              className={`rounded-lg border p-3 ${STAGE_STYLES[stage.stage_id] ?? "border-border bg-muted/20"} ${stage.state === "planned" ? "border-dashed" : ""}`}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-medium text-foreground">{stage.label}</p>
                <span className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                  Step {index + 1}
                </span>
              </div>
              <p className="mt-2 font-mono text-2xl font-semibold text-foreground">
                {stageValueLabel(stage)}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">{stage.description}</p>
              <div className="mt-3 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  {stage.delta_vs_prev != null
                    ? `${stage.delta_vs_prev > 0 ? "+" : ""}${stage.delta_vs_prev.toFixed(1)} pts vs prior`
                    : stage.state === "planned"
                      ? "Reserved stage"
                      : "Baseline / no prior delta"}
                </span>
                {stage.n_rows ? (
                  <span className="text-muted-foreground">
                    {stage.n_rows.toLocaleString()} rows
                  </span>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </div>

      <SnapshotAccuracyPanel />

      {/* Recent Interventions */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">Recent Interventions</h3>
        <div className="space-y-2">
          {(interventions?.interventions ?? [])
            .slice(0, 10)
            .map(
              (iv: {
                intervention_id: number;
                intervention_type: string;
                resource_type: string;
                resource_id: string;
                status: string;
                created_at: string | null;
                financial_impact_estimate: number | null;
              }) => (
                <div
                  key={iv.intervention_id}
                  className="flex items-center justify-between rounded-md border border-border/50 px-3 py-2 text-sm"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${iv.status === "measured" ? "bg-green-500" : "bg-amber-500"}`}
                    />
                    <span className="font-medium">{iv.intervention_type}</span>
                    <span className="text-muted-foreground">
                      {iv.resource_type} {iv.resource_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    {iv.financial_impact_estimate != null && (
                      <span className="text-xs text-muted-foreground">
                        ${(iv.financial_impact_estimate / 1000).toFixed(0)}K est.
                      </span>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {iv.created_at?.slice(0, 10)}
                    </span>
                  </div>
                </div>
              )
            )}
          {(!interventions?.interventions || interventions.interventions.length === 0) && (
            <p className="py-4 text-center text-sm text-muted-foreground">
              No interventions recorded yet
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
