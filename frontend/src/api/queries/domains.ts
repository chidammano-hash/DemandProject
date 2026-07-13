import { fetchJson } from "./request";
import { isForecastModelId } from "@/lib/model-labels";
import type { DomainMeta, DomainPage, SuggestPayload, SamplePairPayload } from "@/types";

// ---------------------------------------------------------------------------
// Domain queries
// ---------------------------------------------------------------------------
export async function fetchDomains(): Promise<{ domains: string[] }> {
  return fetchJson("/domains");
}

export async function fetchDomainMeta(domain: string): Promise<DomainMeta> {
  return fetchJson(`/domains/${encodeURIComponent(domain)}/meta`);
}

export interface PageParams {
  limit: number;
  offset: number;
  q: string;
  sort_by: string;
  sort_dir: string;
  filters?: Record<string, string>;
}

export async function fetchDomainPage(domain: string, params: PageParams): Promise<DomainPage> {
  const qs = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
    q: params.q,
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
  });
  if (params.filters && Object.keys(params.filters).length > 0) {
    qs.set("filters", JSON.stringify(params.filters));
  }
  return fetchJson(`/domains/${encodeURIComponent(domain)}/page?${qs}`);
}

export async function fetchDomainSuggest(
  domain: string,
  field: string,
  q: string,
  filters?: Record<string, string>,
  limit = 12
): Promise<string[]> {
  const qs = new URLSearchParams({ field, q, limit: String(limit) });
  if (filters && Object.keys(filters).length > 0) {
    qs.set("filters", JSON.stringify(filters));
  }
  const payload = await fetchJson<SuggestPayload>(
    `/domains/${encodeURIComponent(domain)}/suggest?${qs}`
  );
  return Array.from(new Set((payload.values || []).filter(Boolean))).slice(0, limit);
}

export async function fetchSamplePair(domain: string): Promise<SamplePairPayload> {
  return fetchJson(`/domains/${encodeURIComponent(domain)}/sample-pair`);
}

// ---------------------------------------------------------------------------
// Forecast model list
// ---------------------------------------------------------------------------
export async function fetchForecastModels(): Promise<string[]> {
  const payload = await fetchJson<{ models?: string[] }>("/domains/forecast/models");
  return (payload.models || []).filter(isForecastModelId);
}
