import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { GlobalFilters } from "@/types/theme";
import { fetchPlanningDate, queryKeys, STALE } from "@/api/queries";

const DEFAULT_FILTERS: GlobalFilters = {
  brand: [],
  category: [],
  market: [],
  channel: [],
  item: [],
  location: [],
  cluster: [],
  timeGrain: "month",
};

function readFiltersFromUrl(): Partial<GlobalFilters> {
  const params = new URLSearchParams(window.location.search);
  const result: Partial<GlobalFilters> = {};
  const brand = params.get("brand");
  if (brand) result.brand = brand.split(",").filter(Boolean);
  const category = params.get("category");
  if (category) result.category = category.split(",").filter(Boolean);
  const market = params.get("market");
  if (market) result.market = market.split(",").filter(Boolean);
  const channel = params.get("channel");
  if (channel) result.channel = channel.split(",").filter(Boolean);
  const item = params.get("item");
  if (item) result.item = item.split(",").filter(Boolean);
  const location = params.get("location");
  if (location) result.location = location.split(",").filter(Boolean);
  const cluster = params.get("cluster");
  if (cluster) result.cluster = cluster.split(",").filter(Boolean);
  const grain = params.get("grain");
  if (grain === "quarter") result.timeGrain = "quarter";
  return result;
}

function syncFiltersToUrl(filters: GlobalFilters) {
  const url = new URL(window.location.href);
  const keysToSync: (keyof Omit<GlobalFilters, "timeGrain">)[] = ["brand", "category", "market", "channel", "item", "location", "cluster"];
  for (const key of keysToSync) {
    if (filters[key].length > 0) {
      url.searchParams.set(key, filters[key].join(","));
    } else {
      url.searchParams.delete(key);
    }
  }
  if (filters.timeGrain !== "month") {
    url.searchParams.set("grain", filters.timeGrain);
  } else {
    url.searchParams.delete("grain");
  }
  window.history.replaceState(null, "", url);
}

export function useGlobalFilters() {
  const [filters, setFiltersState] = useState<GlobalFilters>(() => ({
    ...DEFAULT_FILTERS,
    ...readFiltersFromUrl(),
  }));

  const { data: planningDateInfo } = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: STALE.TEN_MIN,
  });

  // Debounce URL sync
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    debounceRef.current = setTimeout(() => syncFiltersToUrl(filters), 300);
    return () => clearTimeout(debounceRef.current);
  }, [filters]);

  const setFilters = useCallback((partial: Partial<GlobalFilters>) => {
    setFiltersState((prev) => ({ ...prev, ...partial }));
  }, []);

  const resetFilters = useCallback(() => {
    setFiltersState(DEFAULT_FILTERS);
  }, []);

  const hasActiveFilters = filters.brand.length > 0 || filters.category.length > 0 ||
    filters.market.length > 0 || filters.channel.length > 0 ||
    filters.item.length > 0 || filters.location.length > 0 ||
    filters.cluster.length > 0;

  return {
    filters,
    setFilters,
    resetFilters,
    hasActiveFilters,
    planningDate: planningDateInfo?.planning_date ?? null,
  };
}
