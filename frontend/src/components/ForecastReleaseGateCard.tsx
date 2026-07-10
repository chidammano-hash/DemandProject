import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  CheckCircle2,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";

import {
  fetchForecastReleaseReadiness,
  forecastReleaseKeys,
} from "@/api/queries";
import { Skeleton } from "@/components/Skeleton";
import { Badge } from "@/components/ui/badge";
import { formatFixed, formatInt, formatPct } from "@/lib/formatters";
import { cn } from "@/lib/utils";

interface ForecastReleaseGateCardProps {
  onNavigate: (tab: string) => void;
}

interface MetricProps {
  label: string;
  value: string;
  detail: string;
  blocked?: boolean;
}

function Metric({ label, value, detail, blocked = false }: MetricProps) {
  return (
    <div className="rounded-md border bg-background/70 p-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "mt-1 text-lg font-semibold tabular-nums",
          blocked ? "text-destructive" : "text-foreground",
        )}
      >
        {value}
      </p>
      <p className="mt-0.5 text-[11px] text-muted-foreground">{detail}</p>
    </div>
  );
}

export function ForecastReleaseGateCard({
  onNavigate,
}: ForecastReleaseGateCardProps) {
  const [showAllBlockers, setShowAllBlockers] = useState(false);
  const query = useQuery({
    queryKey: forecastReleaseKeys.readiness(),
    queryFn: () => fetchForecastReleaseReadiness(),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  if (query.isLoading) {
    return (
      <div className="rounded-lg border bg-card p-4" aria-label="Forecast release loading">
        <Skeleton className="h-5 w-56" />
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-20" />
          ))}
        </div>
      </div>
    );
  }

  if (query.error || !query.data) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4" role="alert">
        <div className="flex items-center gap-2 text-sm font-medium text-destructive">
          <ShieldAlert className="h-4 w-4" />
          Forecast release evidence is unavailable
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          The planner cannot verify forecast quality, lineage, coverage, and archive safety.
        </p>
      </div>
    );
  }

  const data = query.data;
  const blockers = data.checks.filter((check) => check.status === "block");
  const displayedBlockers = showAllBlockers ? blockers : blockers.slice(0, 6);
  const q = data.quality;
  const coveragePct =
    data.coverage.coverage_frac == null
      ? null
      : 100 * data.coverage.coverage_frac;
  const externalBlocked = blockers.some(
    (check) => check.id === "delta_vs_external",
  );
  const coverageBlocked = blockers.some(
    (check) => check.id === "current_plan_coverage",
  );
  const StatusIcon = data.ready ? ShieldCheck : ShieldAlert;

  return (
    <section
      aria-label="Forecast release readiness"
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4",
        data.ready ? "border-l-emerald-500" : "border-l-amber-500",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <StatusIcon
            className={cn(
              "mt-0.5 h-5 w-5 shrink-0",
              data.ready ? "text-emerald-600" : "text-amber-600",
            )}
          />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold">
                Release {data.release_version} {data.ready ? "planner-ready" : "blocked"}
              </h3>
              <Badge variant={data.ready ? "secondary" : "outline"}>
                {data.ready ? "All gates passed" : `${blockers.length} blockers`}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Post-release verification for planner use: quality, bias, lineage,
              current-plan coverage, and bounded archive safety. Publishing remains
              a separate controlled workflow.
            </p>
          </div>
        </div>
        {data.next_action && (
          <button
            type="button"
            onClick={() => onNavigate(data.next_action?.tab ?? "commandCenter")}
            title={data.next_action.reason}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90"
          >
            {data.next_action.label}
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          label="Champion accuracy"
          value={formatPct(q.champion_accuracy_pct)}
          detail={`WAPE ${formatPct(q.champion_wape_pct)}`}
        />
        <Metric
          label="Lift vs naive"
          value={formatPct(q.relative_wape_lift_vs_naive_pct)}
          detail="Relative WAPE improvement"
        />
        <Metric
          label="Delta vs external"
          value={
            q.accuracy_delta_vs_external_pct_points == null
              ? "—"
              : `${formatFixed(q.accuracy_delta_vs_external_pct_points)} pts`
          }
          detail="Accuracy points on same cohort"
          blocked={externalBlocked}
        />
        <Metric
          label="Current plan coverage"
          value={formatPct(coveragePct)}
          detail={`${formatInt(data.coverage.covered_eligible_dfus)} of ${formatInt(data.coverage.eligible_dfus)} eligible DFUs`}
          blocked={coverageBlocked}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          Evidence: {formatInt(q.dfu_months)} common DFU-months across {formatInt(q.dfus)} DFUs
          {q.first_month && q.last_month
            ? ` (${q.first_month.slice(0, 7)} to ${q.last_month.slice(0, 7)})`
            : ""}
          {q.common_observation_coverage_frac == null
            ? ""
            : ` · ${formatPct(100 * q.common_observation_coverage_frac)} cohort coverage`}
        </span>
        <span>Champion bias {formatPct(q.champion_bias_pct)}</span>
      </div>

      {blockers.length > 0 && (
        <div className="mt-4 border-t pt-3">
          <p className="text-xs font-medium text-foreground">Release blockers</p>
          <ul id="forecast-release-blockers" className="mt-2 grid gap-2 md:grid-cols-2">
            {displayedBlockers.map((check) => (
              <li key={check.id} className="flex items-start gap-2 text-xs text-muted-foreground">
                <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
                <span>{check.message}</span>
              </li>
            ))}
          </ul>
          {blockers.length > 6 && (
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
              {!showAllBlockers && (
                <p className="text-xs text-muted-foreground">
                  {blockers.length - 6} more blockers included in release evidence.
                </p>
              )}
              <button
                type="button"
                onClick={() => setShowAllBlockers((current) => !current)}
                className="text-xs font-medium text-primary hover:underline"
                aria-expanded={showAllBlockers}
                aria-controls="forecast-release-blockers"
              >
                {showAllBlockers
                  ? "Show fewer blockers"
                  : `Show all ${blockers.length} blockers`}
              </button>
            </div>
          )}
        </div>
      )}

      {data.ready && (
        <div className="mt-4 flex items-center gap-2 border-t pt-3 text-xs text-emerald-700 dark:text-emerald-400">
          <CheckCircle2 className="h-4 w-4" />
          This release meets the current evidence policy and can proceed to supervised inventory planning.
        </div>
      )}
    </section>
  );
}
