/**
 * SKU Features — Continuous-feature distribution histograms (2x3 grid).
 */
import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { FeatureDistribution, FeatureDistributions } from "@/api/queries/sku-features";
import { useChartColors } from "@/hooks/useChartColors";
import { Skeleton } from "@/components/Skeleton";
import { HISTOGRAM_FEATURES, HISTOGRAM_LABELS } from "./constants";
import { formatNumber } from "./utils";

interface FeatureHistogramProps {
  title: string;
  data: FeatureDistribution[];
  color: string;
  isLoading: boolean;
}

function FeatureHistogram({ title, data, color, isLoading }: FeatureHistogramProps) {
  const chartData = useMemo(
    () =>
      data.map((bin) => ({
        label: formatNumber(bin.bin_start, 1),
        count: bin.count,
        range: `${formatNumber(bin.bin_start, 2)} – ${formatNumber(bin.bin_end, 2)}`,
      })),
    [data],
  );

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-3 space-y-2">
        <Skeleton className="h-3.5 w-32" />
        <Skeleton className="h-[120px] w-full" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-3">
        <h5 className="text-xs font-medium text-foreground mb-1">{title}</h5>
        <p className="text-xs text-muted-foreground py-6 text-center">No data</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h5 className="text-xs font-medium text-foreground mb-1">{title}</h5>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={chartData} margin={{ top: 2, right: 4, left: -20, bottom: 2 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9 }}
            stroke="var(--muted-foreground)"
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 9 }} stroke="var(--muted-foreground)" />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              fontSize: 11,
            }}
            formatter={(value: number) => [value.toLocaleString(), "SKUs"]}
            labelFormatter={(_label: string, payload: Array<{ payload?: { range?: string } }>) =>
              payload?.[0]?.payload?.range ?? _label
            }
          />
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface FeatureHistogramsProps {
  distributions: FeatureDistributions | undefined;
  isLoading: boolean;
}

export function FeatureHistograms({ distributions, isLoading }: FeatureHistogramsProps) {
  const { trendColors, okabeIto } = useChartColors();

  // Six histogram colors. Prefer theme trendColors (6 entries); fall back to
  // the color-blind-safe Okabe-Ito palette so we never render raw hex literals.
  const histogramColorList = useMemo(
    () =>
      trendColors.length >= 6
        ? trendColors.slice(0, 6)
        : [okabeIto[1], okabeIto[6], okabeIto[3], okabeIto[2], okabeIto[0], okabeIto[5]],
    [trendColors, okabeIto],
  );

  return (
    <div>
      <h3 className="mb-2 text-sm font-medium text-foreground">Feature Distributions</h3>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        {HISTOGRAM_FEATURES.map((key, idx) => (
          <FeatureHistogram
            key={key}
            title={HISTOGRAM_LABELS[key]}
            data={distributions?.features?.[key] ?? []}
            color={histogramColorList[idx % histogramColorList.length]}
            isLoading={isLoading}
          />
        ))}
      </div>
    </div>
  );
}
