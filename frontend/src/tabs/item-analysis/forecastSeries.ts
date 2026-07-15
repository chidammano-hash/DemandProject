import { clampFutureMonths } from "./monthRange";

interface ForecastPointLike {
  forecast_month: string;
  forecast_qty?: number | null;
}

interface AiForecastPointLike {
  forecast_month: string | null;
  ai_qty?: number | null;
}

interface CollectFutureForecastMonthsArgs {
  historyMonths: string[];
  productionForecasts: ForecastPointLike[];
  stagingModels: Record<string, ForecastPointLike[]>;
  aiRows: AiForecastPointLike[];
}

export function collectFutureForecastMonths({
  historyMonths,
  productionForecasts,
  stagingModels,
  aiRows,
}: CollectFutureForecastMonthsArgs): string[] {
  const lastHistoryMonth = historyMonths[historyMonths.length - 1] ?? "";
  const months = new Set<string>();
  for (const point of productionForecasts) {
    if (point.forecast_month > lastHistoryMonth) months.add(point.forecast_month);
  }
  for (const row of aiRows) {
    if (row.forecast_month && row.forecast_month > lastHistoryMonth) {
      months.add(row.forecast_month);
    }
  }
  for (const points of Object.values(stagingModels)) {
    for (const point of points) {
      if (point.forecast_month > lastHistoryMonth) months.add(point.forecast_month);
    }
  }
  return Array.from(months).sort();
}

interface MergeItemForecastSeriesArgs {
  baseSeries: Record<string, unknown>[];
  futureMonths: string[];
  timeEnd: string;
  productionForecasts: ForecastPointLike[];
  stagingModels: Record<string, ForecastPointLike[]>;
  candidateModels: Record<string, ForecastPointLike[]>;
  aiRows: AiForecastPointLike[];
}

function forecastMaps(
  prefix: string,
  models: Record<string, ForecastPointLike[]>
): Map<string, Map<string, number | null>> {
  const maps = new Map<string, Map<string, number | null>>();
  for (const [modelId, points] of Object.entries(models)) {
    const monthMap = new Map<string, number | null>();
    for (const point of points) {
      monthMap.set(point.forecast_month, point.forecast_qty ?? null);
    }
    maps.set(`${prefix}_${modelId}`, monthMap);
  }
  return maps;
}

export function mergeItemForecastSeries({
  baseSeries,
  futureMonths,
  timeEnd,
  productionForecasts,
  stagingModels,
  candidateModels,
  aiRows,
}: MergeItemForecastSeriesArgs): Record<string, unknown>[] {
  const production = new Map(
    productionForecasts.map((point) => [point.forecast_month, point.forecast_qty ?? null])
  );
  const ai = new Map(
    aiRows
      .filter((row): row is AiForecastPointLike & { forecast_month: string } =>
        Boolean(row.forecast_month)
      )
      .map((row) => [row.forecast_month, row.ai_qty ?? null])
  );
  const staging = forecastMaps("staging", stagingModels);
  const backtest = forecastMaps("backtest", candidateModels);

  const historical = baseSeries.map((point) => {
    const month = String(point.month);
    const extras: Record<string, unknown> = {};
    if (production.has(month)) extras.production_forecast = production.get(month);
    if (ai.has(month)) extras.ai_champion = ai.get(month);
    for (const [key, monthMap] of staging) {
      if (monthMap.has(month)) extras[key] = monthMap.get(month);
    }
    for (const [key, monthMap] of backtest) {
      if (monthMap.has(month)) extras[key] = monthMap.get(month);
    }
    return Object.keys(extras).length > 0 ? { ...point, ...extras } : point;
  });

  const future = clampFutureMonths(futureMonths, timeEnd).map((month) => {
    const point: Record<string, unknown> = { month };
    if (production.has(month)) point.production_forecast = production.get(month);
    if (ai.has(month)) point.ai_champion = ai.get(month);
    for (const [key, monthMap] of staging) {
      if (monthMap.has(month)) point[key] = monthMap.get(month);
    }
    return point;
  });

  return [...historical, ...future];
}
