/** Shared display constants for models and strategies. */

export const FORECAST_MODEL_IDS = [
  "lgbm_cluster",
  "nhits",
  "nbeats",
  "mstl",
  "chronos2_enriched",
] as const;

export type ForecastModelId = (typeof FORECAST_MODEL_IDS)[number];

const FORECAST_MODEL_ID_SET = new Set<string>(FORECAST_MODEL_IDS);

export function isForecastModelId(id: string): id is ForecastModelId {
  return FORECAST_MODEL_ID_SET.has(id);
}

export const MODEL_LABELS: Record<string, string> = {
  lgbm_cluster: "LightGBM",
  chronos2_enriched: "Chronos 2E",
  mstl: "MSTL",
  nbeats: "N-BEATS",
  nhits: "N-HiTS",
};

export const MODEL_TYPE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  foundation: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  statistical: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  deep_learning: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
};

export function modelLabel(id: string): string {
  return MODEL_LABELS[id] || id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Format the champion line/tooltip label from its blend mix or single source.
 *
 * - With a blend mix → "champion (40% NBEATS, 35% LGBM, 25% Chronos)" (sorted desc).
 * - Else with a single source model → "champion (N-BEATS)".
 * - Else → "champion".
 */
export function formatChampionLabel(
  mix?: { model: string; weight: number }[] | null,
  source?: string | null
): string {
  if (Array.isArray(mix) && mix.length > 0) {
    const parts = mix
      .slice()
      .sort((a, b) => b.weight - a.weight)
      .map((m) => `${Math.round(m.weight * 100)}% ${modelLabel(m.model)}`);
    return `champion (${parts.join(", ")})`;
  }
  if (source) return `champion (${modelLabel(source)})`;
  return "champion";
}
