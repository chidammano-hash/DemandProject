import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HeatmapGrid } from "@/components/HeatmapGrid";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsHeatmap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";

type HeatmapMetric = "demand_qty" | "customer_count" | "fill_rate";
type ValueMode = "absolute" | "percentile";

interface Props {
  filters: CustomerAnalyticsFilters;
  metric: HeatmapMetric;
  topN: number;
}

/** Linear interp between two #rrggbb colors. */
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

/** Build a colorScale fn (value -> color) for a given metric and max. */
function buildScale(metric: HeatmapMetric, valueMode: ValueMode, maxVal: number) {
  // Fill-rate uses red-yellow-green (the original ECharts visualMap palette);
  // demand/customer count uses a single-hue blue ramp.
  const stops = metric === "fill_rate"
    ? ["#ef4444", "#eab308", "#22c55e"]
    : ["#eff6ff", "#3b82f6", "#1e3a5f"];
  const max = valueMode === "percentile" ? 100 : Math.max(maxVal, 1);
  return (v: number) => {
    const t = Math.max(0, Math.min(1, v / max));
    const segments = stops.length - 1;
    const segIdx = Math.min(Math.floor(t * segments), segments - 1);
    const segT = t * segments - segIdx;
    return lerp(stops[segIdx], stops[segIdx + 1], segT);
  };
}

export function CustomerHeatmap({ filters, metric: initialMetric, topN }: Props) {
  const [metric, setMetric] = useState<HeatmapMetric>(initialMetric);
  const [valueMode, setValueMode] = useState<ValueMode>("absolute");
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortRow, setSortRow] = useState<string | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.heatmap(metric, topN, filters),
    queryFn: () => fetchCustomerAnalyticsHeatmap(metric, topN, filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  const { rows, columnLabels, scale, valueFormatter } = useMemo(() => {
    if (!data) {
      return {
        rows: [],
        columnLabels: [] as string[],
        scale: (v: number) => `rgb(0,0,0,${v})`,
        valueFormatter: (v: number) => v.toLocaleString(),
      };
    }
    const items = data.items;
    let states = [...data.states];

    const itemDescById = new Map(items.map((i) => [i.item_id, i.item_desc]));
    const valueOf = (c: { fill_rate: number; customer_count: number; demand_qty: number }) =>
      metric === "fill_rate" ? c.fill_rate : metric === "customer_count" ? c.customer_count : c.demand_qty;

    // Sort columns (states) by total if a column header was clicked.
    if (sortCol) {
      const stateValues = new Map<string, number>();
      for (const c of data.cells) {
        stateValues.set(c.state, (stateValues.get(c.state) ?? 0) + valueOf(c));
      }
      states.sort((a, b) => (stateValues.get(b) ?? 0) - (stateValues.get(a) ?? 0));
    }

    // Sort rows (items) by total if a row header was clicked.
    let sortedItemIds = items.map((i) => i.item_id);
    if (sortRow) {
      const itemValues = new Map<string, number>();
      for (const c of data.cells) {
        itemValues.set(c.item_id, (itemValues.get(c.item_id) ?? 0) + valueOf(c));
      }
      sortedItemIds = [...sortedItemIds].sort(
        (a, b) => (itemValues.get(b) ?? 0) - (itemValues.get(a) ?? 0),
      );
    }

    // Build a (item_id -> {state -> value}) map so we can render row-by-row.
    const grid = new Map<string, Map<string, number>>();
    for (const c of data.cells) {
      const v = valueOf(c);
      const inner = grid.get(c.item_id) ?? new Map<string, number>();
      inner.set(c.state, v);
      grid.set(c.item_id, inner);
    }

    // Percentile mode: rank all populated values.
    let lookup = (id: string, st: string) => grid.get(id)?.get(st) ?? 0;
    if (valueMode === "percentile") {
      const allVals: number[] = [];
      for (const inner of grid.values()) {
        for (const v of inner.values()) allVals.push(v);
      }
      allVals.sort((a, b) => a - b);
      const n = allVals.length;
      const pctOf = (v: number) => {
        if (n <= 1) return 50;
        let lo = 0;
        let hi = n;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (allVals[mid] <= v) lo = mid + 1;
          else hi = mid;
        }
        return Math.round((lo / n) * 100);
      };
      lookup = (id: string, st: string) => {
        const inner = grid.get(id);
        if (!inner || !inner.has(st)) return 0;
        return pctOf(inner.get(st) ?? 0);
      };
    }

    let maxVal = 0;
    for (const inner of grid.values()) {
      for (const v of inner.values()) if (v > maxVal) maxVal = v;
    }

    const builtRows = sortedItemIds.map((id) => ({
      label: itemDescById.get(id) ?? id,
      values: states.map((s) => lookup(id, s)),
    }));

    const fmt = (v: number) => {
      if (valueMode === "percentile") return `${v}th percentile`;
      if (metric === "fill_rate") return `${v}%`;
      return v.toLocaleString();
    };

    return {
      rows: builtRows,
      columnLabels: states,
      scale: buildScale(metric, valueMode, maxVal),
      valueFormatter: fmt,
    };
  }, [data, metric, valueMode, sortCol, sortRow]);

  return (
    <Card aria-label="Item by state heatmap">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Item x State Heatmap</CardTitle>
          <ExportButtons panelId="heatmap" getData={() => data?.cells ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">Top {topN} items by demand across states. Click headers to sort.</p>
        <div className="flex gap-2 mt-1 flex-wrap">
          <div className="flex gap-1">
            {(["demand_qty", "customer_count", "fill_rate"] as HeatmapMetric[]).map((m) => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                className={`px-2 py-0.5 text-xs rounded ${metric === m ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
              >
                {m === "demand_qty" ? "Demand" : m === "customer_count" ? "Customers" : "Fill Rate"}
              </button>
            ))}
          </div>
          <div className="flex gap-1">
            {(["absolute", "percentile"] as ValueMode[]).map((vm) => (
              <button
                key={vm}
                onClick={() => setValueMode(vm)}
                className={`px-2 py-0.5 text-xs rounded ${valueMode === vm ? "bg-teal-600 text-white" : "bg-gray-100 text-gray-600"}`}
              >
                {vm === "absolute" ? "Absolute" : "Percentile"}
              </button>
            ))}
          </div>
          {(sortCol || sortRow) && (
            <button
              onClick={() => { setSortCol(null); setSortRow(null); }}
              className="px-2 py-0.5 text-xs rounded bg-gray-100 text-gray-600 hover:bg-gray-200"
            >
              Reset Sort
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={!data?.cells || data.cells.length === 0}
          height={400}
        >
          <div role="img" aria-roledescription="Item by state heatmap chart" data-testid="customer-heatmap">
            <HeatmapGrid
              rows={rows}
              columnLabels={columnLabels}
              colorScale={scale}
              valueFormat={valueFormatter}
              compact
              compactCellHeight={16}
              rowLabelWidth={180}
              cellMinWidth={18}
              disablePruning
              onColumnHeaderClick={(c) => setSortCol((prev) => (prev === c ? null : c))}
              onRowHeaderClick={(r) => setSortRow((prev) => (prev === r ? null : r))}
            />
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
