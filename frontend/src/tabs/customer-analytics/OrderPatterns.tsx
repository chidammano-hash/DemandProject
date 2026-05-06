import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  CartesianGrid,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsOrderPatterns,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { useMemo } from "react";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";
import { useChartColors } from "@/hooks/useChartColors";

interface Props {
  filters: CustomerAnalyticsFilters;
}

interface ScatterPoint {
  customer: string;
  avg_interval: number;
  cv: number;
  total_orders: number;
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ScatterPoint }>;
}

function ScatterTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border bg-background p-2 text-xs shadow-sm">
      <div className="font-semibold">{p.customer}</div>
      <div>Avg interval: {p.avg_interval.toFixed(1)} months</div>
      <div>CV: {p.cv.toFixed(2)}</div>
      <div>Demand: {p.total_orders.toLocaleString()}</div>
    </div>
  );
}

export function OrderPatterns({ filters }: Props) {
  const { okabeIto, chartColors } = useChartColors();
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.orderPatterns(filters),
    queryFn: () => fetchCustomerAnalyticsOrderPatterns(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  // Backend returns frequency_histogram: [{bucket, count, pct}] and
  // regularity_scatter: [{customer_no, customer_name, avg_interval_months,
  // interval_cv, total_demand}]. Normalize to the shape the charts use,
  // tolerating the legacy {frequency, regularity} naming as a fallback.
  type RawFreq = { bucket?: string; bin?: string; count: number };
  type RawReg = {
    customer_no?: string;
    customer_name?: string;
    customer?: string;
    avg_interval_months?: number;
    avg_interval?: number;
    interval_cv?: number;
    cv?: number;
    total_demand?: number;
    total_orders?: number;
  };
  const raw = data as
    | { frequency_histogram?: RawFreq[]; regularity_scatter?: RawReg[]; frequency?: RawFreq[]; regularity?: RawReg[] }
    | null
    | undefined;
  const freqRaw = raw?.frequency_histogram ?? raw?.frequency ?? [];
  const regRaw = raw?.regularity_scatter ?? raw?.regularity ?? [];

  const freqData = useMemo(
    () => freqRaw.map((f) => ({ bin: f.bin ?? f.bucket ?? "", count: f.count })),
    [freqRaw],
  );

  const { scatterData, maxOrders } = useMemo(() => {
    const points: ScatterPoint[] = regRaw.map((p) => ({
      customer: p.customer ?? p.customer_name ?? p.customer_no ?? "",
      avg_interval: p.avg_interval ?? p.avg_interval_months ?? 0,
      cv: p.cv ?? p.interval_cv ?? 0,
      total_orders: p.total_orders ?? p.total_demand ?? 0,
    }));
    const max = points.length > 0 ? Math.max(...points.map((p) => p.total_orders), 1) : 1;
    return { scatterData: points, maxOrders: max };
  }, [regRaw]);

  const barColor = okabeIto[3];
  const scatterColor = okabeIto[3];

  return (
    <Card aria-label="Order patterns analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Order Patterns</CardTitle>
          <ExportButtons panelId="order-patterns" getData={() => freqRaw as unknown as Record<string, unknown>[]} />
        </div>
        <p className="text-xs text-muted-foreground">Order frequency histogram and regularity scatter</p>
      </CardHeader>
      <CardContent>
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={freqRaw.length === 0 && regRaw.length === 0}
          height={400}
        >
          <div className="space-y-4">
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Order Frequency</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={freqData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill={barColor} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Order Regularity</p>
              <div role="img" aria-roledescription="Order regularity scatter chart">
                <ResponsiveContainer width="100%" height={200}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 40 }}>
                    <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="avg_interval"
                      name="Avg Interval"
                      label={{ value: "Avg Interval (months)", position: "insideBottom", offset: -10, fontSize: 10 }}
                      tick={{ fontSize: 10, fill: chartColors.axis }}
                      stroke={chartColors.axis}
                    />
                    <YAxis
                      type="number"
                      dataKey="cv"
                      name="CV"
                      label={{ value: "CV (regularity)", angle: -90, position: "insideLeft", fontSize: 10 }}
                      tick={{ fontSize: 10, fill: chartColors.axis }}
                      stroke={chartColors.axis}
                    />
                    <ZAxis
                      type="number"
                      dataKey="total_orders"
                      range={[20, 400]}
                      domain={[0, maxOrders]}
                      name="Demand"
                    />
                    <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={scatterData} fill={scatterColor} fillOpacity={0.65} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
