import type {
  StoryboardException,
  StoryboardSummary,
  PlannerDecision,
} from "@/types/storyboard";

// ---------------------------------------------------------------------------
// Storyboard query key factory
// ---------------------------------------------------------------------------
export const storyboardKeys = {
  summary: () => ["sb-summary"] as const,
  list: (params: Record<string, unknown>) => ["sb-list", params] as const,
  detail: (id: string) => ["sb-detail", id] as const,
};

// ---------------------------------------------------------------------------
// Storyboard fetch functions (Feature 40 — Demand Planner Storyboard)
// ---------------------------------------------------------------------------

export async function fetchSbSummary(): Promise<StoryboardSummary> {
  const res = await fetch("/storyboard/exceptions/summary");
  if (!res.ok) throw new Error("Failed to fetch storyboard summary");
  return res.json();
}

export async function fetchSbExceptions(params: {
  status?: string;
  exception_type?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: StoryboardException[] }> {
  const p = new URLSearchParams();
  if (params.status && params.status !== "all") p.set("status", params.status);
  if (params.exception_type && params.exception_type !== "all")
    p.set("exception_type", params.exception_type);
  if (params.item) p.set("item", params.item);
  if (params.loc) p.set("loc", params.loc);
  p.set("limit", String(params.limit ?? 20));
  p.set("offset", String(params.offset ?? 0));
  const res = await fetch(`/storyboard/exceptions?${p}`);
  if (!res.ok) throw new Error("Failed to fetch exceptions");
  return res.json();
}

export async function fetchSbException(
  id: string
): Promise<{ exception: StoryboardException; decisions: PlannerDecision[] }> {
  const res = await fetch(`/storyboard/exceptions/${id}`);
  if (!res.ok) throw new Error("Failed to fetch exception detail");
  return res.json();
}

export async function updateSbStatus(id: string, status: string): Promise<void> {
  await fetch(`/storyboard/exceptions/${id}/status`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export async function submitSbDecision(
  id: string,
  decisionType: string,
  rationale: string
): Promise<void> {
  await fetch(`/storyboard/exceptions/${id}/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision_type: decisionType, rationale }),
  });
}
