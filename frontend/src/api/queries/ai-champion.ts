/** AI Champion forward adjuster — interactive, single-DFU.
 * Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md
 */
import { fetchJson } from "./core";

export const AI_CHAMPION_PROVIDERS = [
  { value: "ollama", label: "Ollama (local, $0)" },
  { value: "google", label: "Google Gemini" },
  { value: "anthropic", label: "Anthropic (Opus)" },
  { value: "openai", label: "OpenAI (GPT-4o)" },
] as const;

export type AiChampionProvider = (typeof AI_CHAMPION_PROVIDERS)[number]["value"];

/** One forecast month: champion baseline vs the AI-adjusted quantity. */
export interface AiChampionMonth {
  forecast_month: string;
  horizon_months: number | null;
  champion_qty: number | null;
  ai_qty: number | null;
  pct_change: number | null;
}

/** Preview returned by POST /ai-champion/adjust (not yet persisted). */
export interface AiChampionPreview {
  item_id: string;
  loc: string;
  plan_version: string;
  provider: string;
  model: string;
  prompt_version: string;
  recommendation_code: string;
  rec_pct_change: number | null;
  proposed_qty: number[] | null;
  apply_horizon_months: number;
  confidence: number | null;
  rationale: string;
  evidence_keys: string[];
  months: AiChampionMonth[];
}

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

export interface AiChampionSaveResponse {
  item_id: string;
  loc: string;
  plan_version: string;
  run_id: string;
  recommendation_code: string;
  saved_months: number;
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

/** Call the LLM once for a DFU and return a preview (no DB write). */
export async function adjustAiChampion(params: {
  item_id: string;
  loc: string;
  provider?: AiChampionProvider;
  /** Optional free-text steer the planner typed for this DFU. */
  user_comment?: string;
}): Promise<AiChampionPreview> {
  return fetchJson("/ai-champion/adjust", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

/** Persist a previewed adjustment. Quantities are re-derived server-side. */
export async function saveAiChampion(params: {
  item_id: string;
  loc: string;
  provider: string;
  preview: AiChampionPreview;
}): Promise<AiChampionSaveResponse> {
  const { preview } = params;
  return fetchJson("/ai-champion/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_id: params.item_id,
      loc: params.loc,
      provider: params.provider,
      recommendation: {
        recommendation_code: preview.recommendation_code,
        pct_change: preview.rec_pct_change,
        proposed_qty: preview.proposed_qty,
        apply_horizon_months: preview.apply_horizon_months,
        confidence: preview.confidence ?? 0,
        rationale: preview.rationale,
        evidence_keys: preview.evidence_keys,
      },
    }),
  });
}
