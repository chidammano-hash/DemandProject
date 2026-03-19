import type {
  StoryboardException,
  StoryboardSummary,
  PlannerDecision,
} from "@/types/storyboard";
import { fetchJson } from "./core";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// Storyboard query key factory
// ---------------------------------------------------------------------------
export const storyboardKeys = {
  summary: () => ["sb-summary"] as const,
  list: (params: Record<string, unknown>) => ["sb-list", params] as const,
  detail: (id: string) => ["sb-detail", id] as const,
};

// ---------------------------------------------------------------------------
// Storyboard fetch functions (Feature 40 — Planner Storyboard)
// ---------------------------------------------------------------------------

export async function fetchSbSummary(): Promise<StoryboardSummary> {
  return fetchJson("/storyboard/exceptions/summary");
}

export async function fetchSbExceptions(params: {
  status?: string;
  exception_type?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: StoryboardException[] }> {
  const qs = buildSearchParams({
    status: params.status && params.status !== "all" ? params.status : undefined,
    exception_type: params.exception_type && params.exception_type !== "all" ? params.exception_type : undefined,
    item: params.item,
    loc: params.loc,
    limit: params.limit ?? 20,
    offset: params.offset ?? 0,
  });
  return fetchJson(`/storyboard/exceptions?${qs}`);
}

export async function fetchSbException(
  id: string
): Promise<{ exception: StoryboardException; decisions: PlannerDecision[] }> {
  return fetchJson(`/storyboard/exceptions/${id}`);
}

export async function updateSbStatus(id: string, status: string): Promise<void> {
  await fetchJson(`/storyboard/exceptions/${id}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export async function submitSbDecision(
  id: string,
  decisionType: string,
  rationale: string
): Promise<void> {
  await fetchJson(`/storyboard/exceptions/${id}/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision_type: decisionType, rationale }),
  });
}
