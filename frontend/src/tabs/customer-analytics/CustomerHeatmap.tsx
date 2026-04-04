import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsHeatmap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

type HeatmapMetric = "demand_qty" | "customer_count" | "fill_rate";
type ValueMode = "absolute" | "percentile";

interface Props {
  filters: CustomerAnalyticsFilters;
  metric: HeatmapMetric;
  topN: number;
}

export function CustomerHeatmap({ filters, metric: initialMetric, topN }: Props) {
  const [metric, setMetric] = useState<HeatmapMetric>(initialMetric);
  const [valueMode, setValueMode] = useState<ValueMode>("absolute");
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortRow, setSortRow] = useState<string | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.heatmap(metric, topN, filters),
    queryFn: () => fetchCustomerAnalyticsHeatmap(metric, topN, filters),
    staleTime: 5 * 60_000,
  });

  const option = useMemo(() => {
    if (!data) return {};
    const items = data.items.map((i) => i.item_desc);
    let states = [...data.states];

    // Sort states if a column is selected for sorting
    if (sortCol) {
      const stateValues = new Map<string, number>();
      for (const c of data.cells) {
        const val = metric === "fill_rate" ? c.fill_rate : metric === "customer_count" ? c.customer_count : c.demand_qty;
        stateValues.set(c.state, (stateValues.get(c.state) ?? 0) + val);
      }
      states.sort((a, b) => (stateValues.get(b) ?? 0) - (stateValues.get(a) ?? 0));
    }

    // Sort items if a row is selected for sorting
    let sortedItems = [...items];
    if (sortRow) {
      const itemValues = new Map<string, number>();
      for (const c of data.cells) {
        const itemDesc = data.items.find((i) => i.item_id === c.item_id)?.item_desc ?? "";
        const val = metric === "fill_rate" ? c.fill_rate : metric === "customer_count" ? c.customer_count : c.demand_qty;
        itemValues.set(itemDesc, (itemValues.get(itemDesc) ?? 0) + val);
      }
      sortedItems.sort((a, b) => (itemValues.get(b) ?? 0) - (itemValues.get(a) ?? 0));
    }

    const rawCellData = data.cells.map((c) => {
      const x = states.indexOf(c.state);
      const y = sortedItems.indexOf(data.items.find((i) => i.item_id === c.item_id)?.item_desc ?? "");
      const val = metric === "fill_rate" ? c.fill_rate : metric === "customer_count" ? c.customer_count : c.demand_qty;
      return [x, y, val];
    });

    // Percentile mode: rank values 0-100
    let cellData = rawCellData;
    if (valueMode === "percentile" && rawCellData.length > 0) {
      const allVals = rawCellData.map((c) => c[2] as number).sort((a, b) => a - b);
      cellData = rawCellData.map(([x, y, v]) => {
        const rank = allVals.filter((av) => av <= (v as number)).length;
        const pct = allVals.length > 1 ? Math.round((rank / allVals.length) * 100) : 50;
        return [x, y, pct];
      });
    }

    const maxVal = Math.max(...cellData.map((c) => c[2] as number), 1);

    return {
      tooltip: {
        position: "top",
        formatter: (p: { value: [number, number, number] }) => {
          const [x, y, v] = p.value;
          const suffix = valueMode === "percentile" ? "th percentile" : (metric === "fill_rate" ? "%" : "");
          return `<b>${sortedItems[y]}</b> — ${states[x]}<br/>${v.toLocaleString()}${suffix}`;
        },
      },
      grid: { left: 180, right: 20, top: 10, bottom: 60 },
      xAxis: {
        type: "category" as const,
        data: states,
        axisLabel: { fontSize: 10, rotate: 45 },
        triggerEvent: true,
      },
      yAxis: {
        type: "category" as const,
        data: sortedItems,
        axisLabel: { fontSize: 10, width: 160, overflow: "truncate" as const },
        triggerEvent: true,
      },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: true,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: {
          color: metric === "fill_rate"
            ? ["#ef4444", "#eab308", "#22c55e"]
            : ["#eff6ff", "#3b82f6", "#1e3a5f"],
        },
      },
      series: [
        {
          type: "heatmap",
          data: cellData,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
        },
      ],
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
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <div role="img" aria-roledescription="Item by state heatmap chart">
            <ReactECharts
              option={option}
              style={{ height: 400 }}
              onEvents={{
                click: (params: { componentType: string; targetType: string; value: string }) => {
                  if (params.componentType === "xAxis") {
                    setSortCol((prev) => prev === params.value ? null : params.value);
                  } else if (params.componentType === "yAxis") {
                    setSortRow((prev) => prev === params.value ? null : params.value);
                  }
                },
              }}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
