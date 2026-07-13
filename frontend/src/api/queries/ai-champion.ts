/** AI Champion forward forecast — read-only (chart overlay).
 *
 * The interactive adjuster panel was removed in favor of the SKU Chat agentic
 * champion adjustment (approval-gated). What remains here is the read used to
 * draw the "AI Champion" forecast line on the Item Analysis chart — it shows the
 * latest saved adjustment for a DFU regardless of source (the SKU Chat agent's
 * approved adjustments write to the same fact_ai_champion_forecast table).
 * Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md
 */
import { fetchJson } from "./request";

/** A previously-saved adjustment row (GET /ai-champion/forecast). */
export interface AiChampionSavedRow {
  item_id: string;
  loc: string;
  forecast_month: string | null;
  horizon_months: number | null;
  champion_qty: number | null;
  ai_qty: number | null;
  recommendation_code: string;
  pct_change: number | null;
  confidence: number | null;
  rationale: string | null;
}

export interface AiChampionSavedResponse {
  total: number;
  rows: AiChampionSavedRow[];
}

export const aiChampionKeys = {
  saved: (item_id: string, loc: string) => ["ai-champion", "saved", item_id, loc] as const,
};

/** Latest saved adjustment for a DFU (empty rows if none saved yet). */
export async function fetchAiChampionSaved(
  item_id: string,
  loc: string,
): Promise<AiChampionSavedResponse> {
  const sp = new URLSearchParams({ item_id });
  if (loc) sp.set("loc", loc);
  return fetchJson(`/ai-champion/forecast?${sp.toString()}`);
}
