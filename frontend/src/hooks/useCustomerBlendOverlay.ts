import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  customerForecastKeys,
  fetchCustomerBlendSeries,
  fetchLatestCustomerBlend,
  type CustomerBlendSeries,
  type CustomerBlendSeriesMonth,
} from "@/api/queries/customerForecast";
import {
  type CustomerBlendOverlayStatus,
  toCustomerBlendChartPoints,
} from "@/lib/customer-blend-overlay";

const EMPTY_MONTHS: CustomerBlendSeriesMonth[] = [];
const VIEWABLE_BLEND_STATUSES = new Set(["ready", "promoted"]);
const CURRENT_LINEAGE_RECHECK_MS = 30_000;

function isNotFound(error: unknown): boolean {
  return error !== null && typeof error === "object" && "status" in error && error.status === 404;
}

export function useCustomerBlendOverlay(itemId: string, locationId: string, enabled = true) {
  const normalizedItem = itemId.trim();
  const normalizedLocation = locationId.trim();
  const queryEnabled = enabled && normalizedItem.length > 0 && normalizedLocation.length > 0;
  const latestQuery = useQuery({
    queryKey: customerForecastKeys.latestBlend,
    queryFn: fetchLatestCustomerBlend,
    enabled: queryEnabled,
    staleTime: 5_000,
    refetchInterval: queryEnabled ? CURRENT_LINEAGE_RECHECK_MS : false,
    refetchOnWindowFocus: true,
  });
  const latest = latestQuery.data;
  const seriesEnabled = Boolean(
    queryEnabled && latest && VIEWABLE_BLEND_STATUSES.has(latest.status)
  );
  const filters = {
    item_id: normalizedItem,
    location_id: normalizedLocation,
    run_id: latest?.run_id,
  };
  const query = useQuery<CustomerBlendSeries | null>({
    queryKey: customerForecastKeys.blendSeries(filters),
    queryFn: async () => {
      try {
        return await fetchCustomerBlendSeries(filters);
      } catch (error) {
        if (isNotFound(error)) return null;
        throw error;
      }
    },
    enabled: seriesEnabled,
    staleTime: 120_000,
  });
  const months = seriesEnabled ? (query.data?.months ?? EMPTY_MONTHS) : EMPTY_MONTHS;
  const points = useMemo(() => toCustomerBlendChartPoints(months), [months]);

  let status: CustomerBlendOverlayStatus = "idle";
  if (queryEnabled && latestQuery.isPending) status = "loading";
  else if (queryEnabled && latestQuery.isError) status = "error";
  else if (queryEnabled && latest?.status === "generating") status = "loading";
  else if (queryEnabled && !seriesEnabled) status = "empty";
  else if (seriesEnabled && query.isPending) status = "loading";
  else if (seriesEnabled && query.isError) status = "error";
  else if (seriesEnabled && months.length === 0) status = "empty";
  else if (seriesEnabled) status = "ready";

  return {
    months,
    points,
    status,
    runId: latest?.run_id ?? null,
    planningMonth: latest?.planning_month ?? null,
    invalidReason: latest?.invalid_reason ?? null,
  };
}
