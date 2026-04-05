import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Seasonality profiles
// ---------------------------------------------------------------------------
export interface SeasonalityProfilesPayload {
  profiles: { profile: string; count: number }[];
}

export async function fetchSeasonalityProfiles(): Promise<SeasonalityProfilesPayload> {
  return fetchJson("/domains/sku/seasonality-profiles");
}

// ---------------------------------------------------------------------------
// Seasonality Profiles -- Feature 32 filter support
// ---------------------------------------------------------------------------
export const seasonalityProfileKeys = {
  list: () => ["seasonality-profiles"] as const,
};

/** Returns a plain string[] of distinct seasonality profile names for use in filter dropdowns. */
export async function fetchSeasonalityProfileNames(): Promise<string[]> {
  const data = await fetchSeasonalityProfiles();
  return (data.profiles ?? []).map((p) => p.profile);
}
