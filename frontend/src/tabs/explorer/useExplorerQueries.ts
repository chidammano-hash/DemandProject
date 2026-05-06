/**
 * React Query hooks for the Data Explorer tab.
 */
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  queryKeys,
  STALE,
  fetchDomainMeta,
  fetchDomainPage,
  fetchForecastModels,
  fetchSkuClusters,
  fetchSamplePair,
} from "@/api/queries";
import type {
  DomainMeta,
  DomainPage,
  ClusterInfo,
  SamplePairPayload,
} from "@/types";

import type { ClusterSource } from "./types";

export interface UseDomainMetaResult {
  meta: DomainMeta | undefined;
  isLoadingMeta: boolean;
  metaError: unknown;
}

export function useDomainMeta(domain: string): UseDomainMetaResult {
  const { data, isLoading, error } = useQuery<DomainMeta>({
    queryKey: queryKeys.domainMeta(domain),
    queryFn: () => fetchDomainMeta(domain),
    staleTime: STALE.TEN_MIN,
  });
  return { meta: data, isLoadingMeta: isLoading, metaError: error };
}

export interface DomainPageParams {
  limit: number;
  offset: number;
  q: string;
  sort_by: string;
  sort_dir: "asc" | "desc";
  filters: Record<string, string> | undefined;
}

export interface UseDomainPageResult {
  pageData: DomainPage | undefined;
  isLoadingPage: boolean;
  isFetchingPage: boolean;
  pageError: unknown;
}

export function useDomainPage(
  domain: string,
  params: DomainPageParams,
  enabled: boolean,
): UseDomainPageResult {
  const queryKeyParams: Record<string, unknown> = { ...params };
  const { data, isLoading, isFetching, error } = useQuery({
    queryKey: queryKeys.domainPage(domain, queryKeyParams),
    queryFn: () =>
      fetchDomainPage(domain, {
        limit: params.limit,
        offset: params.offset,
        q: params.q,
        sort_by: params.sort_by,
        sort_dir: params.sort_dir,
        filters: params.filters,
      }),
    enabled,
    staleTime: STALE.THIRTY_SEC,
    placeholderData: (prev) => prev,
  });
  return {
    pageData: data,
    isLoadingPage: isLoading,
    isFetchingPage: isFetching,
    pageError: error,
  };
}

export function useForecastModels(enabled: boolean): string[] {
  const { data } = useQuery<string[]>({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    enabled,
    staleTime: STALE.FIVE_MIN,
  });
  return data ?? [];
}

export function useSkuClusters(
  source: ClusterSource,
  enabled: boolean,
): ClusterInfo[] {
  const { data } = useQuery({
    queryKey: queryKeys.skuClusters(source),
    queryFn: () => fetchSkuClusters(source),
    enabled,
    staleTime: STALE.FIVE_MIN,
  });
  return data?.clusters ?? [];
}

export function useSamplePair(
  domain: string,
  enabled: boolean,
): SamplePairPayload | undefined {
  const { data } = useQuery({
    queryKey: queryKeys.samplePair(domain),
    queryFn: () => fetchSamplePair(domain),
    enabled,
    staleTime: STALE.TEN_MIN,
  });
  return data;
}

export interface UseAutoSampleArgs {
  domain: string;
  meta: DomainMeta | undefined;
  showFactFilters: boolean;
  itemFilter: string;
  locationFilter: string;
  autoSampledDomain: string;
  setItemFilter: (v: string) => void;
  setLocationFilter: (v: string) => void;
  setOffset: React.Dispatch<React.SetStateAction<number>>;
  setAutoSampledDomain: React.Dispatch<React.SetStateAction<string>>;
}

/**
 * Auto-sample a representative item+location pair on first visit to a fact
 * domain when the user hasn't entered filters. Skips when filters exist; in
 * that case it just records that the domain has already been "sampled".
 */
export function useAutoSamplePair(args: UseAutoSampleArgs): void {
  const {
    domain,
    meta,
    showFactFilters,
    itemFilter,
    locationFilter,
    autoSampledDomain,
    setItemFilter,
    setLocationFilter,
    setOffset,
    setAutoSampledDomain,
  } = args;

  const shouldAutoSample =
    !!meta &&
    showFactFilters &&
    autoSampledDomain !== domain &&
    !itemFilter.trim() &&
    !locationFilter.trim();

  const samplePair = useSamplePair(domain, shouldAutoSample);

  useEffect(() => {
    if (!shouldAutoSample) {
      if (meta && showFactFilters && autoSampledDomain !== domain) {
        if (itemFilter.trim() || locationFilter.trim()) {
          setAutoSampledDomain(domain);
        }
      }
      return;
    }
    if (samplePair) {
      if (samplePair.item) setItemFilter(String(samplePair.item));
      if (samplePair.location) setLocationFilter(String(samplePair.location));
      setOffset(0);
      setAutoSampledDomain(domain);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    samplePair,
    shouldAutoSample,
    domain,
    meta,
    showFactFilters,
    autoSampledDomain,
    itemFilter,
    locationFilter,
  ]);
}
