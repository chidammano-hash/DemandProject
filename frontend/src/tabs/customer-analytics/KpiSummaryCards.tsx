import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsKpis,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters, KpiMetric } from "@/api/queries/customer-analytics";

interface Props {
  filters: CustomerAnalyticsFilters;
}

interface KpiCardDef {
  key: string;
  label: string;
  format: (v: number) => string;
  suffix?: string;
}

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

function fmtPct(n: number): string {
  return `${n.toFixed(1)}%`;
}

function fmtRatio(n: number): string {
  return n.toFixed(2);
}

const KPI_DEFS: KpiCardDef[] = [
  { key: "total_demand", label: "Total Demand", format: fmtNum, suffix: " cases" },
  { key: "fill_rate", label: "Fill Rate", format: fmtPct },
  { key: "lost_sales_oos", label: "Lost Sales (OOS)", format: fmtNum, suffix: " cases" },
  { key: "active_customers", label: "Active Customers", format: fmtNum },
  { key: "demand_concentration", label: "Demand Concentration", format: fmtPct },
  { key: "order_to_demand_ratio", label: "Order-to-Demand Ratio", format: fmtRatio },
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

function DeltaBadge({ delta }: { delta: number }) {
  const isPositive = delta >= 0;
  const color = isPositive ? "text-green-600" : "text-red-600";
  const arrow = isPositive ? "\u2191" : "\u2193";
  return (
    <span className={`text-xs font-medium ${color}`}>
      {arrow} {Math.abs(delta).toFixed(1)}% MoM
    </span>
  );
}

function KpiCard({ metric, def }: { metric: KpiMetric | undefined; def: KpiCardDef }) {
  if (!metric) {
    return (
      <Card>
        <CardContent className="py-3 px-4">
          <div className="text-xs text-muted-foreground">{def.label}</div>
          <div className="text-lg font-semibold mt-1">--</div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardContent className="py-3 px-4">
        <div className="text-xs text-muted-foreground">{def.label}</div>
        <div className="text-lg font-semibold mt-1">
          {def.format(metric.value)}{def.suffix ?? ""}
        </div>
        <DeltaBadge delta={metric.delta} />
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
              <div className="text-xs text-muted-foreground">{def.label}</div>
              <div className="text-lg font-semibold mt-1 animate-pulse bg-gray-200 rounded w-16 h-6" />
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
