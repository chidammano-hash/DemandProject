import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

type FilterMapper<TParams> = {
  [K in keyof GlobalFilters]?: (value: GlobalFilters[K]) => Partial<TParams> | null;
};

interface UseFilteredQueryConfig<TData, TParams extends Record<string, unknown>> {
  filterMapping?: FilterMapper<TParams>;
  baseParams: TParams;
  queryKey: (params: TParams) => readonly unknown[];
  queryFn: (params: TParams) => Promise<TData>;
  staleTime?: number;
  enabled?: boolean;
}

export function useFilteredQuery<TData, TParams extends Record<string, unknown>>(
  config: UseFilteredQueryConfig<TData, TParams>,
) {
  const { filters } = useGlobalFilterContext();

  const effectiveParams = useMemo<TParams>(() => {
    const merged = { ...config.baseParams } as TParams;
    if (config.filterMapping) {
      for (const [key, mapper] of Object.entries(config.filterMapping) as [
        keyof GlobalFilters,
        FilterMapper<TParams>[keyof GlobalFilters],
      ][]) {
        if (!mapper) continue;
        const filterValue = filters[key];
        const result = mapper(filterValue as never);
        if (result) Object.assign(merged, result);
      }
    }
    return merged;
  }, [filters, config.baseParams, config.filterMapping]);

  return useQuery({
    queryKey: config.queryKey(effectiveParams),
    queryFn: () => config.queryFn(effectiveParams),
    staleTime: config.staleTime ?? 5 * 60_000,
    enabled: config.enabled,
  });
}

export function useItemLocationQuery<TData, TParams extends Record<string, unknown>>(config: {
  baseParams: TParams;
  queryKey: (params: TParams & { item?: string; location?: string }) => readonly unknown[];
  queryFn: (params: TParams & { item?: string; location?: string }) => Promise<TData>;
  staleTime?: number;
  enabled?: boolean;
}) {
  type Combined = TParams & { item?: string; location?: string };
  return useFilteredQuery<TData, Combined>({
    filterMapping: {
      item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null) as Partial<Combined> | null,
      location: (v) => (Array.isArray(v) && v.length === 1 ? { location: v[0] } : null) as Partial<Combined> | null,
    },
    baseParams: config.baseParams as TParams & { item?: string; location?: string },
    queryKey: config.queryKey,
    queryFn: config.queryFn,
    staleTime: config.staleTime,
    enabled: config.enabled,
  });
}
