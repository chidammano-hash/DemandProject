import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsOosImpact,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";
import { useChartColors } from "@/hooks/useChartColors";

type Grain = "customer" | "state";

interface Props {
  filters: CustomerAnalyticsFilters;
  grain: Grain;
  onGrainChange: (g: Grain) => void;
}

interface ScatterPoint {
  demand_qty: number;
  fill_rate: number;
  oos_qty: number;
  label: string;
}

function getColor(channel: string, palette: string[]): string {
  // Distribute channels deterministically across the palette so the same
  // channel always gets the same color across re-renders.
  let h = 0;
  for (let i = 0; i < channel.length; i++) {
    h = (Math.imul(31, h) + channel.charCodeAt(i)) | 0;
  }
  return palette[Math.abs(h) % palette.length];
}

const formatK = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v));

interface TooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ScatterPoint; name?: string }>;
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  const channel = payload[0].name ?? "";
  return (
    <div className="rounded-md border bg-background p-2 text-xs shadow-sm">
      <div className="font-semibold">{p.label}</div>
      <div>Channel: {channel}</div>
      <div>Demand: {p.demand_qty.toLocaleString()}</div>
      <div>Fill Rate: {p.fill_rate}%</div>
      <div>OOS: {p.oos_qty.toLocaleString()}</div>
    </div>
  );
}

export function OosImpactBubble({ filters, grain, onGrainChange }: Props) {
  const { okabeIto, chartColors } = useChartColors();
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.oosImpact(grain, filters),
    queryFn: () => fetchCustomerAnalyticsOosImpact(grain, filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  const totalOos = useMemo(() => {
    const bubbles = data?.bubbles ?? [];
    return bubbles.reduce((sum, b) => sum + b.oos_qty, 0);
  }, [data]);

  const { seriesByChannel, maxOos } = useMemo(() => {
    const bubbles = data?.bubbles ?? [];
    const byChannel: Record<string, ScatterPoint[]> = {};
    let max = 1;
    for (const b of bubbles) {
      const ch = b.channel || "Unknown";
      if (!byChannel[ch]) byChannel[ch] = [];
      byChannel[ch].push({
        demand_qty: b.demand_qty,
        fill_rate: b.fill_rate,
        oos_qty: b.oos_qty,
        label: b.label,
      });
      if (b.oos_qty > max) max = b.oos_qty;
    }
    return { seriesByChannel: byChannel, maxOos: max };
  }, [data]);

  const channels = Object.keys(seriesByChannel);

  return (
    <Card aria-label="OOS impact bubble chart">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">OOS Impact Analysis</CardTitle>
          <ExportButtons panelId="oos-impact" getData={() => data?.bubbles ?? []} />
        </div>
        <div className="flex gap-1 mt-1">
          {(["customer", "state"] as const).map((g) => (
            <button
              key={g}
              onClick={() => onGrainChange(g)}
              className={`px-2 py-0.5 text-xs rounded ${grain === g ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
            >
              By {g}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-1">Bubble size = OOS volume. Below the line = action needed.</p>
        {totalOos > 0 && (
          <div className="mt-1 px-2 py-1 bg-red-50 rounded text-xs font-medium text-red-700">
            Total OOS Volume: {totalOos.toLocaleString()} cases
          </div>
        )}
      </CardHeader>
      <CardContent>
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={!data?.bubbles || data.bubbles.length === 0}
          height={400}
        >
          <div role="img" aria-roledescription="OOS impact bubble scatter chart">
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 50 }}>
                <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="demand_qty"
                  name="Demand"
                  label={{ value: "Demand (cases)", position: "insideBottom", offset: -10, fontSize: 11 }}
                  tickFormatter={formatK}
                  tick={{ fontSize: 10, fill: chartColors.axis }}
                  stroke={chartColors.axis}
                />
                <YAxis
                  type="number"
                  dataKey="fill_rate"
                  name="Fill Rate"
                  domain={[0, 105]}
                  label={{ value: "Fill Rate %", angle: -90, position: "insideLeft", fontSize: 11 }}
                  tick={{ fontSize: 10, fill: chartColors.axis }}
                  stroke={chartColors.axis}
                />
                <ZAxis
                  type="number"
                  dataKey="oos_qty"
                  range={[40, 900]}
                  domain={[0, maxOos]}
                  name="OOS"
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {/* Critical fill-rate line at 90% (replaces ECharts markLine) */}
                <ReferenceLine
                  y={90}
                  stroke={chartColors.axis}
                  strokeDasharray="4 4"
                  label={{ value: "90% target", position: "right", fontSize: 10, fill: chartColors.axis }}
                />
                {channels.map((ch) => (
                  <Scatter
                    key={ch}
                    name={ch}
                    data={seriesByChannel[ch]}
                    fill={getColor(ch, okabeIto)}
                    fillOpacity={0.7}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
