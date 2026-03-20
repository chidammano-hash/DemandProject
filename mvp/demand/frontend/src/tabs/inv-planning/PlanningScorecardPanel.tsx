import { useQuery } from "@tanstack/react-query";
import {
  insightKeys,
  fetchPlanningScorecard,
  STALE_INSIGHTS,
  type PlanningMetric,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatPct } from "@/lib/formatters";
import { ClipboardCheck, TrendingUp, TrendingDown, Minus } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

function trendIcon(trend: string) {
  if (trend === "up") return TrendingUp;
  if (trend === "down") return TrendingDown;
  return Minus;
}

function trendColor(trend: string): string {
  if (trend === "up") return "text-green-600";
  if (trend === "down") return "text-red-600";
  return "text-muted-foreground";
}

function formatMetricValue(value: number | null, unit: string): string {
  if (value == null) return "—";
  if (unit === "%") return formatPct(value);
  if (unit === "$") return `$${formatFixed(value, 0)}`;
  if (unit === "days") return `${formatFixed(value)} d`;
  return formatFixed(value);
}

function MiniSparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 64;
  const h = 16;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="text-primary/40">
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

export function PlanningScorecardPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.planningScorecard(),
    queryFn: fetchPlanningScorecard,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load planning scorecard: {(error as Error).message}
      </div>
    );
  }

  const healthScore = data?.health_score;
  const metrics = data?.metrics ?? [];

  const healthColor =
    healthScore != null
      ? healthScore >= 80
        ? "text-green-600"
        : healthScore >= 60
          ? "text-yellow-600"
          : "text-red-600"
      : undefined;

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Monthly planning effectiveness scorecard tracking trailing performance across key planning metrics.
        Each metric shows current vs. prior period with trend direction and sparkline history.
      </div>

      {/* Hero KPI */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Overall Planning Health"
          value={isLoading ? "..." : healthScore != null ? formatFixed(healthScore, 0) : "—"}
          sublabel="/100"
          colorClass={healthColor}
          tooltip={{
            title: "Planning Health Score",
            description: "Composite score derived from forecast accuracy, fill rate, exception resolution, and SS optimization. 80+ is healthy.",
          }}
        />
        {data?.period && (
          <KpiCard
            className={PANEL_KPI}
            label="Evaluation Period"
            value={data.period}
          />
        )}
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading scorecard...</p>
      ) : metrics.length === 0 ? (
        <EmptyState
          icon={ClipboardCheck}
          title="No scorecard data available"
          description="The planning scorecard aggregates trailing metrics from forecast accuracy, fill rate, exceptions, and safety stock modules."
          steps={[
            { label: "Run backtest pipeline", command: "make backtest-all" },
            { label: "Compute safety stock", command: "make ss-compute" },
            { label: "Refresh materialized views", command: "make db-apply-sql" },
          ]}
        />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-2 pr-3">Metric</th>
                <th className="text-right py-2 pr-3">Current</th>
                <th className="text-right py-2 pr-3">Prior</th>
                <th className="text-center py-2 pr-3">Trend</th>
                <th className="text-center py-2">Sparkline</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m: PlanningMetric) => {
                const TIcon = trendIcon(m.trend);
                const tc = trendColor(m.trend);
                return (
                  <tr key={m.name} className="border-b last:border-0 hover:bg-muted/40">
                    <td className="py-2 pr-3 font-medium">{m.name}</td>
                    <td className="py-2 pr-3 text-right font-mono tabular-nums">
                      {formatMetricValue(m.current, m.unit)}
                    </td>
                    <td className="py-2 pr-3 text-right font-mono tabular-nums text-muted-foreground">
                      {formatMetricValue(m.prior, m.unit)}
                    </td>
                    <td className="py-2 pr-3 text-center">
                      <span className={`inline-flex items-center gap-1 ${tc}`}>
                        <TIcon className="h-3.5 w-3.5" />
                      </span>
                    </td>
                    <td className="py-2 text-center">
                      <div className="inline-block">
                        <MiniSparkline data={m.sparkline ?? []} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
