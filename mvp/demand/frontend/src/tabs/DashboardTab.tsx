import { useQuery } from "@tanstack/react-query";
import { Target, BarChart3, TrendingUp, Package, Activity, Scale } from "lucide-react";

import { KpiCard } from "@/components/KpiCard";
import { AlertPanel } from "@/components/AlertPanel";
import { HeatmapGrid, makeHeatmapScale } from "@/components/HeatmapGrid";
import { TopMovers } from "@/components/TopMovers";
import { ForecastTrendChart } from "@/components/ForecastTrendChart";
import { WidgetGrid, WidgetCard } from "@/components/WidgetGrid";
import { Skeleton } from "@/components/Skeleton";
import { useTheme } from "@/hooks/useTheme";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import {
  queryKeys,
  STALE,
  fetchDashboardKpis,
  fetchDashboardAlerts,
  fetchDashboardTopMovers,
  fetchDashboardHeatmap,
} from "@/api/queries";
import type { DashboardFilterParams } from "@/api/queries";

function formatNumber(n: number | null): string {
  if (n == null) return "N/A";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

function formatPct(n: number | null): string {
  if (n == null) return "N/A";
  return `${n.toFixed(1)}%`;
}

function trendDirection(delta: number | null): "up" | "down" | "flat" {
  if (delta == null || delta === 0) return "flat";
  return delta > 0 ? "up" : "down";
}

interface DashboardTabProps {
  theme: "light" | "dark";
}

export default function DashboardTab({ theme }: DashboardTabProps) {
  const { trendColors, chartColors, productTheme, colorMode } = useTheme();
  const { filters } = useGlobalFilterContext();

  const filterParams: DashboardFilterParams = {
    brand: filters.brand,
    category: filters.category,
    market: filters.market,
    channel: filters.channel,
    item: filters.item,
    location: filters.location,
  };

  // Cast for query keys (they only need serializable values)
  const fk = filterParams as unknown as Record<string, unknown>;

  const kpisQuery = useQuery({
    queryKey: queryKeys.dashboardKpis({ ...fk, window: 3 }),
    queryFn: () => fetchDashboardKpis(3, filterParams),
    staleTime: STALE.TWO_MIN,
  });

  const alertsQuery = useQuery({
    queryKey: queryKeys.dashboardAlerts(fk),
    queryFn: () => fetchDashboardAlerts(10, filterParams),
    staleTime: STALE.TWO_MIN,
  });

  const moversQuery = useQuery({
    queryKey: queryKeys.dashboardTopMovers({ ...fk, limit: 5 }),
    queryFn: () => fetchDashboardTopMovers(5, "both", filterParams),
    staleTime: STALE.TWO_MIN,
  });

  const heatmapQuery = useQuery({
    queryKey: queryKeys.dashboardHeatmap({ ...fk, grain: "category", periods: 4 }),
    queryFn: () => fetchDashboardHeatmap("category", 4, filterParams),
    staleTime: STALE.TWO_MIN,
  });

  const kpis = kpisQuery.data;
  const chartConfig = productTheme.charts[colorMode === "light" && productTheme.charts.light ? "light" : "dark"]!;
  const heatmapScale = makeHeatmapScale(chartConfig.heatmapScale);

  return (
    <div className="animate-fade-in space-y-4">
      {/* KPI Cards Row */}
      <WidgetGrid cols={12} gap="md">
        {kpisQuery.isLoading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="col-span-full sm:col-span-2">
              <Skeleton className="h-20 w-full rounded-md" />
            </div>
          ))
        ) : (
          <>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="Accuracy"
                value={formatPct(kpis?.accuracy_pct ?? null)}
                icon={Target}
                severity={kpis?.accuracy_pct != null && kpis.accuracy_pct >= 85 ? "best" : kpis?.accuracy_pct != null && kpis.accuracy_pct < 70 ? "warning" : "neutral"}
                trend={kpis?.deltas?.accuracy_pct != null ? { delta: kpis.deltas.accuracy_pct, direction: trendDirection(kpis.deltas.accuracy_pct) } : undefined}
              />
            </div>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="WAPE"
                value={formatPct(kpis?.wape_pct ?? null)}
                icon={BarChart3}
                severity={kpis?.wape_pct != null && kpis.wape_pct <= 15 ? "best" : kpis?.wape_pct != null && kpis.wape_pct > 30 ? "warning" : "neutral"}
                trend={kpis?.deltas?.wape_pct != null ? { delta: kpis.deltas.wape_pct, direction: trendDirection(-kpis.deltas.wape_pct) } : undefined}
              />
            </div>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="Bias"
                value={formatPct(kpis?.bias_pct ?? null)}
                icon={Scale}
                severity={kpis?.bias_pct != null && Math.abs(kpis.bias_pct) <= 5 ? "best" : kpis?.bias_pct != null && Math.abs(kpis.bias_pct) > 20 ? "warning" : "neutral"}
                trend={kpis?.deltas?.bias_pct != null ? { delta: kpis.deltas.bias_pct, direction: trendDirection(-Math.abs(kpis.deltas.bias_pct)) } : undefined}
              />
            </div>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="Total Forecast"
                value={formatNumber(kpis?.total_forecast ?? null)}
                icon={TrendingUp}
              />
            </div>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="Total Actual"
                value={formatNumber(kpis?.total_actual ?? null)}
                icon={Activity}
              />
            </div>
            <div className="col-span-full sm:col-span-2">
              <KpiCard
                label="Weeks of Supply"
                value={kpis?.weeks_of_supply != null ? kpis.weeks_of_supply.toFixed(1) : "N/A"}
                icon={Package}
              />
            </div>
          </>
        )}
      </WidgetGrid>

      {/* Middle row: Alert Panel, Heatmap, Top Movers */}
      <WidgetGrid cols={12} gap="md">
        <WidgetCard span={3} title="Alerts">
          {alertsQuery.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <AlertPanel alerts={alertsQuery.data?.alerts ?? []} />
          )}
        </WidgetCard>

        <WidgetCard span={6} title="Performance Heatmap" subtitle="Accuracy % by category">
          {heatmapQuery.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <HeatmapGrid
              rows={heatmapQuery.data?.rows ?? []}
              columnLabels={heatmapQuery.data?.period_labels ?? []}
              colorScale={heatmapScale}
            />
          )}
        </WidgetCard>

        <WidgetCard span={3} title="Top Movers" subtitle="Period-over-period">
          {moversQuery.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <TopMovers movers={moversQuery.data?.movers ?? []} />
          )}
        </WidgetCard>
      </WidgetGrid>

      {/* Forecast trend chart */}
      <WidgetGrid cols={12} gap="md">
        <WidgetCard span={12} title="Forecast vs Actual Trend">
          <ForecastTrendChart
            data={[]}
            theme={theme}
            chartColors={chartColors}
            seriesColors={trendColors}
          />
        </WidgetCard>
      </WidgetGrid>
    </div>
  );
}
