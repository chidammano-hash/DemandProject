import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HeatmapGrid } from "@/components/HeatmapGrid";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsLifecycle,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";
import { useChartColors } from "@/hooks/useChartColors";

interface Props {
  filters: CustomerAnalyticsFilters;
}

// Backend response shape:
//   { cohorts: [{cohort_month, months_since: number[], retention_pct: number[]}],
//     waterfall: [{month, new_customers, churned_customers, net_change}] }
interface LifecycleResponse {
  cohorts: Array<{ cohort_month: string; months_since: number[]; retention_pct: number[] }>;
  waterfall: Array<{ month: string; new_customers: number; churned_customers: number; net_change: number }>;
}

/** 4-stop cohort retention scale (red -> yellow -> light green -> green). */
const RETENTION_STOPS = ["#fee2e2", "#fef08a", "#bbf7d0", "#22c55e"];

function lerp(a: string, b: string, t: number): string {
  const ar = parseInt(a.slice(1, 3), 16);
  const ag = parseInt(a.slice(3, 5), 16);
  const ab = parseInt(a.slice(5, 7), 16);
  const br = parseInt(b.slice(1, 3), 16);
  const bg = parseInt(b.slice(3, 5), 16);
  const bb = parseInt(b.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

function retentionColor(pct: number): string {
  const t = Math.max(0, Math.min(1, pct / 100));
  const segments = RETENTION_STOPS.length - 1;
  const segIdx = Math.min(Math.floor(t * segments), segments - 1);
  const segT = t * segments - segIdx;
  return lerp(RETENTION_STOPS[segIdx], RETENTION_STOPS[segIdx + 1], segT);
}

export function CustomerLifecycle({ filters }: Props) {
  const { okabeIto } = useChartColors();
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.lifecycle(filters),
    queryFn: () => fetchCustomerAnalyticsLifecycle(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  const { cohortMonths, columnLabels, heatmapRows, waterfallBars } = useMemo(() => {
    const resp = data as LifecycleResponse | null | undefined;
    const cohorts = resp?.cohorts ?? [];
    const waterfall = resp?.waterfall ?? [];

    let mMax = 0;
    for (const c of cohorts) {
      for (const ms of c.months_since) if (ms > mMax) mMax = ms;
    }
    const months = cohorts.map((c) => c.cohort_month);
    const cols = Array.from({ length: mMax + 1 }, (_, i) => `M${i}`);

    // Build {label, values[]} per cohort. Missing months -> 0 (rendered as
    // the lowest scale stop, which is acceptable for retention).
    const rows = cohorts.map((c) => {
      const valByMs = new Map<number, number>();
      c.months_since.forEach((ms, i) => valByMs.set(ms, c.retention_pct[i] ?? 0));
      return {
        label: c.cohort_month,
        values: cols.map((_, i) => valByMs.get(i) ?? 0),
      };
    });

    const bars = waterfall.flatMap((w) => [
      { label: w.month, value: w.new_customers, type: "new" as const },
      { label: w.month, value: -w.churned_customers, type: "churned" as const },
    ]);

    return {
      cohortMonths: months,
      columnLabels: cols,
      heatmapRows: rows,
      waterfallBars: bars,
    };
  }, [data]);

  // Theme-aware new/churned colors via the okabe-ito palette.
  const newColor = okabeIto[2];      // green-ish
  const churnedColor = okabeIto[5];  // red-ish
  const fmtPct = (v: number) => `${v.toFixed(1)}%`;

  return (
    <Card aria-label="Customer lifecycle analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Lifecycle</CardTitle>
          <ExportButtons panelId="lifecycle" getData={() => waterfallBars as unknown as Record<string, unknown>[]} />
        </div>
        <p className="text-xs text-muted-foreground">Cohort retention heatmap and new/churned waterfall</p>
      </CardHeader>
      <CardContent>
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={cohortMonths.length === 0 && waterfallBars.length === 0}
          height={400}
        >
          <div className="space-y-4">
            <div role="img" aria-roledescription="Cohort retention heatmap">
              <HeatmapGrid
                rows={heatmapRows}
                columnLabels={columnLabels}
                colorScale={retentionColor}
                valueFormat={fmtPct}
                rowLabelWidth={100}
                cellMinWidth={36}
                disablePruning
              />
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">New vs Churned Customers</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={waterfallBars} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => v.toLocaleString()} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {waterfallBars.map((d, i) => (
                      <Cell key={i} fill={d.type === "new" ? newColor : churnedColor} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
