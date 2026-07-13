import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Competition queries
// ---------------------------------------------------------------------------
export interface CompetitionConfig {
  metric: string;
  lag: string;
  min_dfu_rows: number;
  champion_model_id: string;
  models: string[];
  strategy?: string;
  strategy_params?: Record<string, unknown>;
}

export interface ChampionSummary {
  total_dfus: number;
  total_dfu_months?: number;
  total_champion_rows: number;
  model_wins: Record<string, number>;
  overall_champion_wape: number | null;
  overall_champion_accuracy_pct: number | null;
  run_ts: string;
  total_ceiling_rows?: number;
  ceiling_model_wins?: Record<string, number>;
  overall_ceiling_wape?: number | null;
  overall_ceiling_accuracy_pct?: number | null;
}

export async function fetchCompetitionConfig(): Promise<{ config: CompetitionConfig; available_models: string[] } | null> {
  try {
    return await fetchJson("/competition/config");
  } catch { return null; }
}

export async function fetchCompetitionSummary(): Promise<{ summary: ChampionSummary } | null> {
  try {
    return await fetchJson("/competition/summary");
  } catch { return null; }
}

export async function saveCompetitionConfig(config: CompetitionConfig): Promise<void> {
  await fetchJson("/competition/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}
