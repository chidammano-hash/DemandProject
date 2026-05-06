/**
 * SKU Features — Categorical distribution charts (3 horizontal bar charts).
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
  Cell,
} from "recharts";
import type { SkuFeaturesSummary } from "@/api/queries/sku-features";
import { useChartColors } from "@/hooks/useChartColors";
import { Skeleton } from "@/components/Skeleton";
import { CHART_MARGIN, DISTRIBUTION_TITLES } from "./constants";
import { recordToChartData } from "./utils";

interface HorizontalDistributionProps {
  title: string;
  data: { label: string; count: number }[];
  colors: string[];
  isLoading: boolean;
}

function HorizontalDistribution({
  title,
  data,
  colors,
  isLoading,
}: HorizontalDistributionProps) {
  const total = useMemo(() => data.reduce((s, d) => s + d.count, 0), [data]);

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 space-y-3">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-[180px] w-full" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h4 className="text-sm font-medium text-foreground mb-2">{title}</h4>
        <p className="text-xs text-muted-foreground py-8 text-center">No data available</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h4 className="text-sm font-medium text-foreground mb-2">{title}</h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} layout="vertical" margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" />
          <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
          <YAxis
            dataKey="label"
            type="category"
            width={90}
            tick={{ fontSize: 11 }}
            stroke="var(--muted-foreground)"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(value: number) => [
              `${value.toLocaleString()} (${total > 0 ? ((value / total) * 100).toFixed(1) : 0}%)`,
              "Count",
            ]}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, idx) => (
              <Cell key={idx} fill={colors[idx % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface CategoricalDistributionsProps {
  summary: SkuFeaturesSummary | undefined;
  isLoading: boolean;
}

export function CategoricalDistributions({ summary, isLoading }: CategoricalDistributionsProps) {
  const { okabeIto } = useChartColors();

  // Categorical palettes pulled from the color-blind-safe Okabe-Ito palette.
  // Indices chosen to keep semantic ordering (cool -> warm) within each chart.
  const distributionColors: Record<string, string[]> = useMemo(
    () => ({
      // seasonality: none -> low -> moderate -> strong (neutral, blue, yellow, vermillion)
      seasonality_profile: [okabeIto[7], okabeIto[1], okabeIto[3], okabeIto[5]],
      // variability: smooth -> erratic -> intermittent -> lumpy (green, orange, purple, vermillion)
      variability_class: [okabeIto[2], okabeIto[0], okabeIto[6], okabeIto[5]],
      // trend: declining -> flat -> growing (vermillion, neutral, green)
      trend_direction: [okabeIto[5], okabeIto[7], okabeIto[2]],
    }),
    [okabeIto],
  );

  const seasonalityData = useMemo(
    () => recordToChartData(summary?.distributions?.seasonality_profile),
    [summary?.distributions?.seasonality_profile],
  );
  const variabilityData = useMemo(
    () => recordToChartData(summary?.distributions?.variability_class),
    [summary?.distributions?.variability_class],
  );
  const trendData = useMemo(
    () => recordToChartData(summary?.distributions?.trend_direction),
    [summary?.distributions?.trend_direction],
  );

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      <HorizontalDistribution
        title={DISTRIBUTION_TITLES.seasonality_profile}
        data={seasonalityData}
        colors={distributionColors.seasonality_profile}
        isLoading={isLoading}
      />
      <HorizontalDistribution
        title={DISTRIBUTION_TITLES.variability_class}
        data={variabilityData}
        colors={distributionColors.variability_class}
        isLoading={isLoading}
      />
      <HorizontalDistribution
        title={DISTRIBUTION_TITLES.trend_direction}
        data={trendData}
        colors={distributionColors.trend_direction}
        isLoading={isLoading}
      />
    </div>
  );
}
