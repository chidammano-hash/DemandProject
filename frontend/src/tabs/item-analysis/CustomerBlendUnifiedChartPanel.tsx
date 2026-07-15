import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  fetchStagingForecasts,
  productionForecastKeys,
  type CandidateForecastPoint,
  type CandidateForecastsPayload,
} from "@/api/queries/production-forecast";
import { CustomerBlendLegend } from "@/components/CustomerBlendOverlay";
import { useCustomerBlendOverlay } from "@/hooks/useCustomerBlendOverlay";
import {
  mergeCustomerBlendOverlay,
  toCustomerBlendChartPoints,
} from "@/lib/customer-blend-overlay";
import { applyPairedCustomerStaging, pairCustomerStaging } from "./customerStagingPair";
import { UnifiedChartPanel, type UnifiedChartPanelProps } from "./UnifiedChartPanel";

function candidatePoint(
  month: string,
  forecastQty: number | null,
  actualQty: number | null
): CandidateForecastPoint {
  const error =
    forecastQty != null && actualQty != null && actualQty !== 0
      ? (100 * Math.abs(forecastQty - actualQty)) / Math.abs(actualQty)
      : null;
  return {
    forecast_month: month,
    forecast_qty: forecastQty,
    forecast_qty_lower: null,
    forecast_qty_upper: null,
    actual_qty: actualQty,
    accuracy_pct: error == null ? null : Math.max(0, 100 - error),
    wape: error,
    bias:
      forecastQty != null && actualQty != null && actualQty !== 0
        ? forecastQty / actualQty - 1
        : null,
    horizon_months: 1,
    cluster_id: null,
  };
}

/** Adds the latest exact item/location customer blend to the shared Item Analysis chart. */
export function CustomerBlendUnifiedChartPanel(props: UnifiedChartPanelProps) {
  const exactDfu = props.skuData.mode === "item_location";
  const overlay = useCustomerBlendOverlay(props.skuData.item, props.skuData.location, exactDfu);
  const exactStagingQuery = useQuery({
    queryKey: productionForecastKeys.customerStagingPair(
      props.skuData.item,
      props.skuData.location,
      overlay.runId,
      overlay.bottomUpStagingRunId
    ),
    queryFn: () =>
      fetchStagingForecasts({ item_id: props.skuData.item, loc: props.skuData.location }),
    enabled: exactDfu && overlay.runId !== null && overlay.bottomUpStagingRunId !== null,
    staleTime: 120_000,
  });
  const visibleMonths = useMemo(
    () =>
      overlay.months.filter((point) => {
        const month = point.forecast_month;
        return (
          (!props.skuTimeStart || month >= props.skuTimeStart) &&
          (!props.skuTimeEnd || month <= props.skuTimeEnd)
        );
      }),
    [overlay.months, props.skuTimeEnd, props.skuTimeStart]
  );
  const pairedStaging = useMemo(
    () =>
      pairCustomerStaging(
        exactStagingQuery.data ?? props.stagingForecastData,
        overlay.runId,
        overlay.bottomUpStagingRunId
      ),
    [exactStagingQuery.data, overlay.bottomUpStagingRunId, overlay.runId, props.stagingForecastData]
  );
  const exactBaseSeries = useMemo(
    () => applyPairedCustomerStaging(props.skuFilteredSeries, pairedStaging),
    [pairedStaging, props.skuFilteredSeries]
  );
  const hasStagedBottomUp = pairedStaging.bottomUpMonths.size > 0;
  const hasStagedBlend = pairedStaging.blendMonths.size > 0;
  const visiblePoints = useMemo(
    () =>
      toCustomerBlendChartPoints(visibleMonths).map((point) => ({
        ...point,
        customer_bottom_up_qty: pairedStaging.bottomUpMonths.has(point.month.slice(0, 7))
          ? null
          : point.customer_bottom_up_qty,
        customer_blend_qty: pairedStaging.blendMonths.has(point.month.slice(0, 7))
          ? null
          : point.customer_blend_qty,
      })),
    [pairedStaging.blendMonths, pairedStaging.bottomUpMonths, visibleMonths]
  );
  const historicalPoints = useMemo(
    () =>
      (overlay.trend?.months ?? [])
        .filter((month) => month.phase === "backtest")
        .map((month) => ({
          month: month.month,
          backtest_customer_bottom_up: month.customer_bottom_up_qty,
          backtest_customer_source_champion: month.source_champion_qty,
          backtest_customer_bottom_up_blend: month.customer_blend_qty,
        })),
    [overlay.trend?.months]
  );
  const mergedSeries = useMemo(() => {
    const futureMerged = mergeCustomerBlendOverlay(exactBaseSeries, visiblePoints);
    const historicalByMonth = new Map(
      historicalPoints.map((point) => [point.month.slice(0, 7), point])
    );
    return futureMerged.map((point) => {
      const historical = historicalByMonth.get(String(point.month).slice(0, 7));
      return historical ? { ...point, ...historical, month: point.month } : point;
    });
  }, [exactBaseSeries, historicalPoints, visiblePoints]);
  const candidateForecastData = useMemo<CandidateForecastsPayload | null>(() => {
    const historical = (overlay.trend?.months ?? []).filter((month) => month.phase === "backtest");
    if (historical.length === 0) return props.candidateForecastData ?? null;
    return {
      item_id: props.skuData.item,
      loc: props.skuData.location,
      models: {
        ...(props.candidateForecastData?.models ?? {}),
        customer_bottom_up: historical.map((month) =>
          candidatePoint(month.month, month.customer_bottom_up_qty, month.actual_qty)
        ),
        customer_source_champion: historical.map((month) =>
          candidatePoint(month.month, month.source_champion_qty, month.actual_qty)
        ),
        customer_bottom_up_blend: historical.map((month) =>
          candidatePoint(month.month, month.customer_blend_qty, month.actual_qty)
        ),
      },
    };
  }, [overlay.trend?.months, props.candidateForecastData, props.skuData]);
  const futureMonths = useMemo(() => {
    const lastHistoryMonth = props.skuMonths[props.skuMonths.length - 1] ?? "";
    const months = new Set(props.skuFutureMonths ?? []);
    for (const month of overlay.months) {
      if (month.forecast_month > lastHistoryMonth) months.add(month.forecast_month);
    }
    return Array.from(months).sort();
  }, [overlay.months, props.skuFutureMonths, props.skuMonths]);
  const selectedRangeEmpty = overlay.status === "ready" && visibleMonths.length === 0;
  const legendStatus = selectedRangeEmpty ? "empty" : overlay.status;

  return (
    <div className="space-y-3">
      <CustomerBlendLegend
        months={visibleMonths}
        status={legendStatus}
        runId={overlay.runId}
        planningMonth={overlay.planningMonth}
        stagedSeries={hasStagedBottomUp || hasStagedBlend}
        emptyMessage={
          selectedRangeEmpty ? "No customer blend falls within the selected range." : undefined
        }
      />
      <UnifiedChartPanel
        {...props}
        skuFilteredSeries={mergedSeries}
        skuFutureMonths={futureMonths}
        candidateForecastData={candidateForecastData}
        stagingForecastData={pairedStaging.payload}
      />
    </div>
  );
}
