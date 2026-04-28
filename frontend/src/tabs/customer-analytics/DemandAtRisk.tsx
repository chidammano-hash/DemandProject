import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsDemandAtRisk,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

const BAR_COLORS: Record<string, string> = {
  total: "#22c55e",
  risk: "#ef4444",
  secure: "#22c55e",
};

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

// Backend returns `waterfall` as [{category, value}, ...] with categories:
// total_demand, concentration_risk, oos_loss, churn_risk, secure_demand.
// Map category → planner-friendly label + bar type (drives chart color).
const CATEGORY_META: Record<string, { label: string; type: string }> = {
  total_demand: { label: "Total Demand", type: "total" },
  concentration_risk: { label: "Concentration Risk", type: "risk" },
  oos_loss: { label: "OOS Loss", type: "risk" },
  churn_risk: { label: "Churn Risk", type: "risk" },
  secure_demand: { label: "Secure Demand", type: "secure" },
};

export function DemandAtRisk({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.demandAtRisk(filters),
    queryFn: () => fetchCustomerAnalyticsDemandAtRisk(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const chartData = useMemo(() => {
    // Tolerate either response shape — older clients expected `bars`, the
    // current backend returns `waterfall`. Fall back through both.
    const raw = (data as { waterfall?: Array<{ category?: string; label?: string; type?: string; value: number }> } | null | undefined)?.waterfall
      ?? (data as { bars?: Array<{ category?: string; label?: string; type?: string; value: number }> } | null | undefined)?.bars
      ?? [];
    return raw.map((b) => {
      const cat = b.category ?? b.label ?? "";
      const meta = CATEGORY_META[cat];
      return {
        label: meta?.label ?? b.label ?? cat,
        type: b.type ?? meta?.type ?? "risk",
        value: b.value,
        displayValue: Math.abs(b.value),
      };
    });
  }, [data]);

  return (
    <Card aria-label="Demand at risk analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Demand at Risk</CardTitle>
          <ExportButtons panelId="demand-at-risk" getData={() => chartData as unknown as Record<string, unknown>[]} />
        </div>
        <p className="text-xs text-muted-foreground">Waterfall: total demand minus risk categories = secure demand</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[320px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : chartData.length === 0 ? (
          <div className="h-[320px] flex items-center justify-center text-sm text-muted-foreground">No data</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <XAxis dataKey="label" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v: number) => fmtNum(v)} tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(v: number, _name: string, props: { payload: { label: string; value: number } }) => [
                  `${fmtNum(props.payload.value)} cases`,
                  props.payload.label,
                ]}
              />
              <Bar dataKey="displayValue" radius={[4, 4, 0, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={BAR_COLORS[d.type] ?? "#94a3b8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
