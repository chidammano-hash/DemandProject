import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  Gauge,
  PackageX,
  PieChart,
  Scale,
  TrendingUp,
  Users,
  type LucideIcon,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsKpis,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters, KpiMetric } from "@/api/queries/customer-analytics";
import { formatCompactKMB as fmtNum } from "@/lib/formatters";

interface Props {
  filters: CustomerAnalyticsFilters;
}

interface KpiCardDef {
  key: string;
  label: string;
  format: (v: number) => string;
  suffix?: string;
  // Direction in which an increase is *good*. For OOS / lost sales /
  // concentration an increase is bad, so a green up-arrow would mislead.
  goodDirection?: "up" | "down";
  icon: LucideIcon;
}

// Deltas with magnitude below this are treated as flat (no direction).
const FLAT_THRESHOLD = 0.05;

interface DeltaPresentation {
  flat: boolean;
  color: string;
  arrow: string;
}

/**
 * Resolve a delta's color + arrow given which direction is "good" for the
 * metric (U2.3/U2.4). Near-zero deltas render neutral (flat) with no arrow.
 */
export function deltaPresentation(
  delta: number,
  goodDirection: "up" | "down" = "up",
): DeltaPresentation {
  if (Math.abs(delta) < FLAT_THRESHOLD) {
    return { flat: true, color: "text-muted-foreground", arrow: "→" };
  }
  const isUp = delta > 0;
  const isGood = goodDirection === "up" ? isUp : !isUp;
  return {
    flat: false,
    color: isGood ? "text-green-600" : "text-red-600",
    arrow: isUp ? "↑" : "↓",
  };
}

function fmtPct(n: number): string {
  return `${n.toFixed(1)}%`;
}

function fmtRatio(n: number): string {
  return n.toFixed(2);
}

const KPI_DEFS: KpiCardDef[] = [
  { key: "total_demand", label: "Total Demand", format: fmtNum, suffix: " cases", goodDirection: "up", icon: TrendingUp },
  { key: "fill_rate", label: "Fill Rate", format: fmtPct, goodDirection: "up", icon: Gauge },
  { key: "lost_sales_oos", label: "Lost Sales (OOS)", format: fmtNum, suffix: " cases", goodDirection: "down", icon: PackageX },
  { key: "active_customers", label: "Active Customers", format: fmtNum, goodDirection: "up", icon: Users },
  { key: "demand_concentration", label: "Demand Concentration", format: fmtPct, goodDirection: "down", icon: PieChart },
  { key: "order_to_demand_ratio", label: "Order-to-Demand Ratio", format: fmtRatio, goodDirection: "up", icon: Scale },
];

// Backend returns {kpis: [{key, value, delta}, ...]} with keys like
// `oos_volume`, `concentration_top10`, `order_demand_ratio`. Map to the
// planner-facing labels the cards expect, and tolerate the older keyed
// {total_demand: {...}, ...} shape as a fallback.
const BACKEND_KEY_MAP: Record<string, string> = {
  oos_volume: "lost_sales_oos",
  concentration_top10: "demand_concentration",
  order_demand_ratio: "order_to_demand_ratio",
};

function useKpiData(filters: CustomerAnalyticsFilters) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.kpis(filters),
    queryFn: () => fetchCustomerAnalyticsKpis(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });
  const kpis = useMemo(() => {
    if (!data) return null;
    const maybeList = (data as { kpis?: Array<{ key: string; value: number; delta: number }> }).kpis;
    if (Array.isArray(maybeList)) {
      const out: Record<string, KpiMetric> = {};
      for (const row of maybeList) {
        const mapped = BACKEND_KEY_MAP[row.key] ?? row.key;
        out[mapped] = { value: row.value, delta: row.delta };
      }
      return out;
    }
    return data as unknown as Record<string, KpiMetric>;
  }, [data]);
  return { kpis, isLoading };
}

/**
 * Accessible, period-anchored description of a MoM delta (U9.3). Screen readers
 * and the hover title get "up/down N% month-over-month vs prior month" instead
 * of a bare "↑ N% MoM" with no comparison anchor. Mirrors the Demand-History
 * MoM aria pattern (U6.5).
 */
export function deltaAriaLabel(delta: number, flat: boolean): string {
  const magnitude = `${Math.abs(delta).toFixed(1)}% month-over-month vs prior month`;
  if (flat) return `No material change ${magnitude}`;
  return `${delta >= 0 ? "Up" : "Down"} ${magnitude}`;
}

function DeltaBadge({ delta, goodDirection }: { delta: number | null; goodDirection?: "up" | "down" }) {
  // U3.4 — a null delta means the backend has no prior-period anchor (no MoM
  // computed). Render an explicit "no prior period" affordance instead of a
  // fabricated "→ 0.0% MoM" that a planner would read as "genuinely unchanged".
  if (delta == null) {
    const label = "No prior period to compare";
    return (
      <span className="text-xs font-medium text-muted-foreground" aria-label={label} title={label}>
        — no prior period
      </span>
    );
  }
  const { color, arrow, flat } = deltaPresentation(delta, goodDirection);
  const label = deltaAriaLabel(delta, flat);
  return (
    <span className={`text-xs font-medium ${color}`} aria-label={label} title={label}>
      {arrow} {Math.abs(delta).toFixed(1)}% MoM
    </span>
  );
}

function KpiCard({ metric, def }: { metric: KpiMetric | undefined; def: KpiCardDef }) {
  const Icon = def.icon;
  if (!metric) {
    return (
      <Card>
        <CardContent className="py-3 px-4">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-muted-foreground">{def.label}</div>
            <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          </div>
          <div className="text-lg font-semibold mt-1">--</div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className="group hover:border-primary/20 hover:shadow-md">
      <CardContent className="py-3 px-4">
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs text-muted-foreground">{def.label}</div>
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-muted text-muted-foreground transition-colors group-hover:bg-primary/10 group-hover:text-primary">
            <Icon className="h-3.5 w-3.5" aria-hidden="true" />
          </span>
        </div>
        <div className="mt-1 text-lg font-semibold tracking-tight">
          {def.format(metric.value)}{def.suffix ?? ""}
        </div>
        <DeltaBadge delta={metric.delta} goodDirection={def.goodDirection} />
      </CardContent>
    </Card>
  );
}

export function KpiSummaryCards({ filters }: Props) {
  const { kpis, isLoading } = useKpiData(filters);

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3" aria-label="KPI summary cards">
        {KPI_DEFS.map((def) => (
          <Card key={def.key}>
            <CardContent className="py-3 px-4">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-muted-foreground">{def.label}</div>
                <def.icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              </div>
              <div className="text-lg font-semibold mt-1 animate-pulse bg-muted rounded w-16 h-6" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3" aria-label="KPI summary cards">
      {KPI_DEFS.map((def) => {
        const metric = kpis?.[def.key] as KpiMetric | undefined;
        return <KpiCard key={def.key} metric={metric} def={def} />;
      })}
    </div>
  );
}
