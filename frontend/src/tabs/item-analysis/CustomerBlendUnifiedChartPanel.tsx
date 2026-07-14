import { useMemo } from "react";

import { CustomerBlendLegend } from "@/components/CustomerBlendOverlay";
import { useCustomerBlendOverlay } from "@/hooks/useCustomerBlendOverlay";
import {
  mergeCustomerBlendOverlay,
  toCustomerBlendChartPoints,
} from "@/lib/customer-blend-overlay";
import { UnifiedChartPanel, type UnifiedChartPanelProps } from "./UnifiedChartPanel";

/** Adds the latest exact item/location customer blend to the shared Item Analysis chart. */
export function CustomerBlendUnifiedChartPanel(props: UnifiedChartPanelProps) {
  const exactDfu = props.skuData.mode === "item_location";
  const overlay = useCustomerBlendOverlay(props.skuData.item, props.skuData.location, exactDfu);
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
  const visiblePoints = useMemo(() => toCustomerBlendChartPoints(visibleMonths), [visibleMonths]);
  const mergedSeries = useMemo(
    () => mergeCustomerBlendOverlay(props.skuFilteredSeries, visiblePoints),
    [props.skuFilteredSeries, visiblePoints]
  );
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
        emptyMessage={
          selectedRangeEmpty ? "No customer blend falls within the selected range." : undefined
        }
      />
      <UnifiedChartPanel
        {...props}
        skuFilteredSeries={mergedSeries}
        skuFutureMonths={futureMonths}
      />
    </div>
  );
}
