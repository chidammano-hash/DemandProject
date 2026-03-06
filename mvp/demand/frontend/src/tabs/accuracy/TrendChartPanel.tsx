import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AccuracyKpis, LagPoint } from "@/types";
import { useChartColors } from "@/hooks/useChartColors";
import { ACCURACY_KPI_OPTIONS } from "./SliceTablePanel";

// ---------------------------------------------------------------------------
// Constants (hoisted to avoid re-renders from inline objects)
// ---------------------------------------------------------------------------

const CHART_MARGIN = { top: 4, right: 16, left: 0, bottom: 4 };
const CHART_DOT_SM = { r: 4 };

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TrendChartPanelProps {
  lagCurveData: LagPoint[];
  lagModels: string[];
  sliceKpis: string[];
  activeLagMetric: string;
  onLagCurveMetricChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TrendChartPanel({
  lagCurveData,
  lagModels,
  sliceKpis,
  activeLagMetric,
  onLagCurveMetricChange,
}: TrendChartPanelProps) {
  const { chartColors, trendColors } = useChartColors();

  const lagMetricOpt = useMemo(
    () => ACCURACY_KPI_OPTIONS.find((k) => k.key === activeLagMetric),
    [activeLagMetric],
  );

  const chartData = useMemo(() => {
    return lagCurveData.map((p) => {
      const row: Record<string, number | string> = { lag: `Lag ${p.lag}` };
      for (const m of lagModels) {
        const val = p.by_model[m]?.[activeLagMetric as keyof AccuracyKpis];
        if (val !== null && val !== undefined) row[m] = val as number;
      }
      return row;
    });
  }, [lagCurveData, lagModels, activeLagMetric]);

  const yFormatter = useMemo(() => {
    const fmtIsPct = lagMetricOpt?.format === "pct";
    const fmtIsBias = lagMetricOpt?.format === "bias";
    return (v: number) =>
      fmtIsPct
        ? `${Number(v).toFixed(0)}%`
        : fmtIsBias
          ? `${(Number(v) * 100).toFixed(0)}%`
          : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }, [lagMetricOpt]);

  const tooltipFormatter = useMemo(() => {
    const fmtIsPct = lagMetricOpt?.format === "pct";
    const fmtIsBias = lagMetricOpt?.format === "bias";
    return (v: number) =>
      fmtIsPct
        ? `${Number(v).toFixed(1)}%`
        : fmtIsBias
          ? `${(Number(v) * 100).toFixed(1)}%`
          : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }, [lagMetricOpt]);

  if (lagCurveData.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          {lagMetricOpt?.label ?? "KPI"} by Lag Horizon
        </p>
        <select
          className="h-7 rounded-md border border-input bg-background px-2 text-xs"
          value={activeLagMetric}
          onChange={onLagCurveMetricChange}
        >
          {ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => (
            <option key={k.key} value={k.key}>
              {k.label}
            </option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis
            dataKey="lag"
            tick={{ fontSize: 11, fill: chartColors.axis }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={yFormatter}
          />
          <Tooltip
            formatter={tooltipFormatter}
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {lagModels.map((m, i) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={trendColors[i % trendColors.length]}
              strokeWidth={2}
              dot={CHART_DOT_SM}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
