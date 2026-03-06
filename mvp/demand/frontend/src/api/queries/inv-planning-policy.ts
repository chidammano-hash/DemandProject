import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature5: Replenishment Policy Management
// ---------------------------------------------------------------------------

export interface ReplenishmentPolicy {
  policy_id: string;
  policy_name: string;
  policy_type: "continuous_rop" | "periodic_review" | "min_max" | "manual";
  segment: string | null;
  review_cycle_days: number | null;
  service_level: number | null;
  use_eoq: boolean;
  use_safety_stock: boolean;
  active: boolean;
  dfu_count: number;
}

export interface PolicyListPayload {
  policies: ReplenishmentPolicy[];
}

export interface PolicyAssignmentRow {
  item_no: string;
  loc: string;
  policy_id: string;
  policy_name: string;
  policy_type: string;
  override_reason: string | null;
  assigned_by: string;
  effective_date: string | null;
}

export interface PolicyAssignmentsPayload {
  total: number;
  rows: PolicyAssignmentRow[];
}

export interface PolicyComplianceByPolicy {
  policy_name: string;
  policy_type: string;
  dfu_count: number;
  below_ss_pct: number | null;
  avg_ss_coverage: number | null;
  avg_dos: number | null;
}

export interface PolicyCompliancePayload {
  total_dfus: number;
  assigned_count: number;
  unassigned_count: number;
  assignment_pct: number;
  by_policy: Record<string, PolicyComplianceByPolicy>;
}

export interface PolicyAssignResult {
  assigned_count: number;
  failed_count: number;
  already_assigned_count: number;
}

export const policyKeys = {
  list: () => ["policy-list"] as const,
  assignments: (params?: Record<string, unknown>) => ["policy-assignments", params ?? {}] as const,
  compliance: () => ["policy-compliance"] as const,
};

export async function fetchPolicies(): Promise<PolicyListPayload> {
  return fetchJson("/inv-planning/policies");
}

export async function createPolicy(body: {
  policy_id: string;
  policy_name: string;
  policy_type: string;
  segment?: string;
  review_cycle_days?: number;
  service_level?: number;
  use_eoq?: boolean;
  use_safety_stock?: boolean;
  notes?: string;
}): Promise<ReplenishmentPolicy> {
  return fetchJson("/inv-planning/policies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function updatePolicy(
  policyId: string,
  body: Partial<{
    policy_name: string;
    policy_type: string;
    segment: string;
    review_cycle_days: number;
    service_level: number;
    use_eoq: boolean;
    use_safety_stock: boolean;
    active: boolean;
    notes: string;
  }>,
): Promise<ReplenishmentPolicy> {
  return fetchJson(`/inv-planning/policies/${encodeURIComponent(policyId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyAssignments(params: {
  item?: string;
  location?: string;
  policy_id?: string;
  assigned_by?: string;
  limit?: number;
  offset?: number;
}): Promise<PolicyAssignmentsPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.policy_id?.trim()) qs.set("policy_id", params.policy_id.trim());
  if (params.assigned_by?.trim()) qs.set("assigned_by", params.assigned_by.trim());
  return fetchJson(`/inv-planning/policy-assignments?${qs}`);
}

export async function assignPolicy(body: {
  item_no?: string;
  loc?: string;
  policy_id?: string;
  override_reason?: string;
  segment?: string;
}): Promise<PolicyAssignResult> {
  return fetchJson("/inv-planning/policy-assignments/assign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyCompliance(): Promise<PolicyCompliancePayload> {
  return fetchJson("/inv-planning/policy-assignments/compliance");
}
