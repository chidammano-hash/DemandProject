import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  customerForecastKeys,
  fetchCustomerBlendTrend,
  fetchLatestCustomerBlend,
  STALE,
  type DashboardFilterParams,
  type TrendPoint,
} from "@/api/queries";
import { CustomerForecastTrendChart } from "@/components/CustomerForecastTrendChart";
import { ForecastTrendChart } from "@/components/ForecastTrendChart";
import { Skeleton } from "@/components/Skeleton";
import { Button } from "@/components/ui/button";
import { formatApiError } from "@/lib/formatApiError";
import { cn } from "@/lib/utils";
import { CollapsibleSection } from "@/components/CollapsibleSection";

const TREND_OPTIONS = [6, 12, 18, 24];
const CURRENT_LINEAGE_RECHECK_MS = 30_000;

interface PortfolioForecastComparisonProps {
  kpiModel: string;
  trendWindow: number;
  onTrendWindowChange: (window: number) => void;
  dashboardFilters: DashboardFilterParams;
  standardMonths: TrendPoint[];
  standardLoading: boolean;
}

export function PortfolioForecastComparison({
  kpiModel,
  trendWindow,
  onTrendWindowChange,
  dashboardFilters,
  standardMonths,
  standardLoading,
}: PortfolioForecastComparisonProps) {
  const [mode, setMode] = useState<"standard" | "customer_blend">("standard");
  const latestBlendQuery = useQuery({
    queryKey: customerForecastKeys.latestBlend,
    queryFn: fetchLatestCustomerBlend,
    staleTime: 5_000,
    enabled: mode === "customer_blend",
    refetchInterval: mode === "customer_blend" ? CURRENT_LINEAGE_RECHECK_MS : false,
    refetchOnWindowFocus: true,
  });
  const latestBlend = latestBlendQuery.data;
  const trendFilters = useMemo(
    () => ({
      run_id: latestBlend?.run_id ?? "",
      window: trendWindow,
      brand: dashboardFilters.brand,
      category: dashboardFilters.category,
      market: dashboardFilters.market,
      channel: dashboardFilters.channel,
      item_id: dashboardFilters.item,
      location_id: dashboardFilters.location,
      cluster: dashboardFilters.cluster,
    }),
    [dashboardFilters, latestBlend?.run_id, trendWindow]
  );
  const trendQuery = useQuery({
    queryKey: customerForecastKeys.blendTrend(trendFilters),
    queryFn: () => fetchCustomerBlendTrend(trendFilters),
    staleTime: STALE.THIRTY_SEC,
    enabled:
      mode === "customer_blend" &&
      Boolean(latestBlend?.run_id) &&
      (latestBlend?.status === "ready" || latestBlend?.status === "promoted"),
    refetchOnWindowFocus: true,
  });

  const error = latestBlendQuery.error ?? trendQuery.error;
  return (
    <CollapsibleSection
      title="Forecast vs Actual"
      headerRight={
        <div className="flex items-center gap-3">
          <div
            className="flex items-center rounded-md border border-border bg-muted/20 p-0.5"
            role="group"
            aria-label="Forecast comparison mode"
          >
            <button
              type="button"
              aria-pressed={mode === "standard"}
              onClick={() => setMode("standard")}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] transition-colors",
                mode === "standard"
                  ? "bg-background font-medium text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Standard
            </button>
            <button
              type="button"
              aria-pressed={mode === "customer_blend"}
              onClick={() => setMode("customer_blend")}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] transition-colors",
                mode === "customer_blend"
                  ? "bg-background font-medium text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Customer Blend
            </button>
          </div>
          <span className="rounded bg-muted/50 px-2 py-0.5 text-[10px] text-muted-foreground">
            {mode === "standard" ? kpiModel : "bottom-up + blend"}
          </span>
          <div className="flex items-center gap-1" role="group" aria-label="Trend window">
            {TREND_OPTIONS.map((window) => (
              <button
                key={window}
                type="button"
                aria-pressed={trendWindow === window}
                onClick={() => onTrendWindowChange(window)}
                className={cn(
                  "rounded px-2 py-0.5 text-[10px] transition-colors",
                  trendWindow === window
                    ? "bg-primary/10 font-medium text-primary"
                    : "text-muted-foreground hover:bg-muted/50"
                )}
              >
                {window}mo
              </button>
            ))}
          </div>
        </div>
      }
    >
      {mode === "standard" ? (
        standardLoading ? (
          <Skeleton className="h-[260px]" />
        ) : (
          <ForecastTrendChart data={standardMonths} />
        )
      ) : latestBlendQuery.isPending || (Boolean(latestBlend) && trendQuery.isPending) ? (
        <Skeleton className="h-[300px]" />
      ) : error ? (
        <div
          className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center text-sm text-destructive"
          role="alert"
        >
          <span>Customer blend comparison could not be loaded: {formatApiError(error)}</span>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() =>
              void (latestBlendQuery.isError ? latestBlendQuery.refetch() : trendQuery.refetch())
            }
          >
            Retry
          </Button>
        </div>
      ) : trendQuery.data ? (
        <CustomerForecastTrendChart trend={trendQuery.data} />
      ) : (
        <div className="flex min-h-[260px] items-center justify-center px-6 text-center text-sm text-muted-foreground">
          {latestBlend
            ? `Customer blend ${latestBlend.status} is not ready for comparison.`
            : "No current customer blend is available."}
        </div>
      )}
    </CollapsibleSection>
  );
}
