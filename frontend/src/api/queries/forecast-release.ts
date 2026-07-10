import { fetchJson } from "./core";
import type { components } from "../generated/schema";

export type ForecastReleaseCheck = components["schemas"]["ForecastReleaseCheck"];
export type ForecastReleaseQuality = components["schemas"]["ForecastReleaseQuality"];
export type ForecastReleaseCoverage = components["schemas"]["ForecastReleaseCoverage"];
export type ForecastReleaseLineage = components["schemas"]["ForecastReleaseLineage"];
export type ForecastReleaseFreshness = components["schemas"]["ForecastReleaseFreshness"];
export type ForecastReleaseArchive = components["schemas"]["ForecastReleaseArchive"];
export type ForecastReleaseIntegrity = components["schemas"]["ForecastReleaseIntegrity"];
export type ForecastReleaseReadiness =
  components["schemas"]["ForecastReleaseReadinessResponse"];

export const forecastReleaseKeys = {
  all: ["forecast-release"] as const,
  readiness: () => [...forecastReleaseKeys.all, "readiness"] as const,
};

export async function fetchForecastReleaseReadiness(): Promise<ForecastReleaseReadiness> {
  return fetchJson("/forecast-release/readiness");
}
