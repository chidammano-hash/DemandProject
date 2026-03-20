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

  // Detect flat lines (all lags have same value per model)
  const isFlatLine = useMemo(() => {
    if (chartData.length < 2) return false;
    return lagModels.every((m) => {
      const vals = chartData.map((d) => d[m]).filter((v) => v !== undefined) as number[];
      if (vals.length < 2) return true;
      const range = Math.max(...vals) - Math.min(...vals);
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
      return avg === 0 ? range === 0 : range / Math.abs(avg) < 0.005; // <0.5% variation
    });
  }, [chartData, lagModels]);

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
      <p className="text-xs text-muted-foreground leading-relaxed max-w-3xl">
        <strong>What this chart shows:</strong> Forecast accuracy at each prediction horizon.{" "}
        <strong>Lag 0</strong> = forecast issued the same month as actuals (most recent information).{" "}
        <strong>Lag 1</strong> = forecast issued 1 month before actuals.{" "}
        <strong>Lag 4</strong> = forecast issued 4 months ahead (longest horizon).{" "}
        Accuracy typically degrades as lag increases because the model has less recent data.{" "}
        This chart uses the <strong>backtest lag archive</strong> which stores all 5 lag horizons
        for each backtest model, enabling true forecast-value-added (FVA) analysis.
      </p>
      {isFlatLine && (
        <div className="rounded-md border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30 px-3 py-2">
          <p className="text-xs text-amber-800 dark:text-amber-300">
            <strong>Why are the lines flat?</strong> When the same forecast value is stored
            at all lag horizons (common for the <em>external</em> model which doesn&apos;t re-forecast
            each month), accuracy will be identical across lags. To see meaningful lag degradation,
            filter to backtest models (e.g. lgbm_cluster, catboost_cluster, xgboost_cluster) which
            generate distinct predictions at each horizon. Use the <strong>Models</strong> filter above
            to select specific models.
          </p>
        </div>
      )}
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
