import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { aiChampionKeys, fetchAiChampionSaved } from "@/api/queries/ai-champion";
import {
  fetchCandidateForecasts,
  fetchProductionForecast,
  fetchStagingForecasts,
  type CandidateForecastsPayload,
  type ProductionForecastPayload,
  type StagingForecastsPayload,
} from "@/api/queries/production-forecast";
import {
  collectFutureForecastMonths,
  mergeItemForecastSeries,
} from "@/tabs/item-analysis/forecastSeries";

interface UseItemForecastOverlaysArgs {
  itemId: string;
  locationId: string;
  historyMonths: string[];
  filteredSeries: Record<string, unknown>[];
  timeEnd: string;
  aiEnabled: boolean;
}

interface KeyedPayload<T> {
  dfuKey: string;
  payload: T | null;
}

export function useItemForecastOverlays({
  itemId,
  locationId,
  historyMonths,
  filteredSeries,
  timeEnd,
  aiEnabled,
}: UseItemForecastOverlaysArgs) {
  const item = itemId.trim();
  const location = locationId.trim();
  const hasDfu = item.length > 0 && location.length > 0;
  const dfuKey = hasDfu ? `${item}\u0000${location}` : "";
  const [productionResult, setProductionResult] =
    useState<KeyedPayload<ProductionForecastPayload> | null>(null);
  const [stagingResult, setStagingResult] = useState<KeyedPayload<StagingForecastsPayload> | null>(
    null
  );
  const [candidateResult, setCandidateResult] =
    useState<KeyedPayload<CandidateForecastsPayload> | null>(null);
  const production = productionResult?.dfuKey === dfuKey ? productionResult.payload : null;
  const staging = stagingResult?.dfuKey === dfuKey ? stagingResult.payload : null;
  const candidate = candidateResult?.dfuKey === dfuKey ? candidateResult.payload : null;

  useEffect(() => {
    if (!hasDfu) {
      setProductionResult(null);
      return;
    }
    let cancelled = false;
    fetchProductionForecast({ item_id: item, loc: location })
      .then((payload) => {
        if (!cancelled) setProductionResult({ dfuKey, payload });
      })
      .catch(() => {
        if (!cancelled) setProductionResult({ dfuKey, payload: null });
      });
    return () => {
      cancelled = true;
    };
  }, [dfuKey, hasDfu, item, location]);

  useEffect(() => {
    if (!hasDfu) {
      setStagingResult(null);
      return;
    }
    let cancelled = false;
    fetchStagingForecasts({ item_id: item, loc: location })
      .then((payload) => {
        if (!cancelled) setStagingResult({ dfuKey, payload });
      })
      .catch(() => {
        if (!cancelled) setStagingResult({ dfuKey, payload: null });
      });
    return () => {
      cancelled = true;
    };
  }, [dfuKey, hasDfu, item, location]);

  useEffect(() => {
    if (!hasDfu) {
      setCandidateResult(null);
      return;
    }
    let cancelled = false;
    fetchCandidateForecasts({ item_id: item, loc: location })
      .then((payload) => {
        if (!cancelled) setCandidateResult({ dfuKey, payload });
      })
      .catch(() => {
        if (!cancelled) setCandidateResult({ dfuKey, payload: null });
      });
    return () => {
      cancelled = true;
    };
  }, [dfuKey, hasDfu, item, location]);

  const aiQuery = useQuery({
    queryKey: aiChampionKeys.saved(item, location),
    queryFn: () => fetchAiChampionSaved(item, location),
    enabled: aiEnabled && hasDfu,
    staleTime: 60_000,
  });
  const aiRows = useMemo(() => aiQuery.data?.rows ?? [], [aiQuery.data?.rows]);
  const futureMonths = useMemo(
    () =>
      collectFutureForecastMonths({
        historyMonths,
        productionForecasts: production?.forecasts ?? [],
        stagingModels: staging?.models ?? {},
        aiRows,
      }),
    [aiRows, historyMonths, production?.forecasts, staging?.models]
  );
  const mergedSeries = useMemo(
    () =>
      mergeItemForecastSeries({
        baseSeries: filteredSeries,
        futureMonths,
        timeEnd,
        productionForecasts: production?.forecasts ?? [],
        stagingModels: staging?.models ?? {},
        candidateModels: candidate?.models ?? {},
        aiRows,
      }),
    [
      aiRows,
      candidate?.models,
      filteredSeries,
      futureMonths,
      production?.forecasts,
      staging?.models,
      timeEnd,
    ]
  );
  const aiChampionLead = aiRows[0] ?? null;

  return {
    prodForecastData: production,
    stagingForecastData: staging,
    candidateForecastData: candidate,
    skuFutureMonths: futureMonths,
    mergedFilteredSeries: mergedSeries,
    aiChampionFetched: aiQuery.isFetched,
    hasAiChampion: aiRows.some((row) => row.ai_qty != null),
    aiChampionLead,
  };
}
