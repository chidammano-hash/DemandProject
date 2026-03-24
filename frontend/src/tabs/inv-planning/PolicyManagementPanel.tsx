import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  queryKeys,
  fetchPolicies,
  fetchPolicyCompliance,
  assignPolicy,
  updatePolicy,
  STALE,
  type ReplenishmentPolicy,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { formatFixed, formatPct } from "@/lib/formatters";
import { EmptyState } from "@/components/EmptyState";
import { Shield } from "lucide-react";

const POLICY_TYPE_DESCRIPTIONS: Record<string, string> = {
  continuous_rop: "Monitor inventory daily; place an order when on-hand reaches the Reorder Point (ROP)",
  periodic_review: "Review inventory every N days; if below target, order up to max level",
  min_max: "Order up to max quantity when on-hand falls below the min threshold",
  manual: "No automatic trigger; planner reviews and orders manually",
};

const POLICY_TYPE_COLORS: Record<string, string> = {
  continuous_rop:  "bg-blue-100 text-blue-800",
  periodic_review: "bg-sky-100 text-sky-800",
  min_max:         "bg-emerald-100 text-emerald-800",
  manual:          "bg-amber-100 text-amber-800",
};

type EditPolicyState = {
  policy: ReplenishmentPolicy;
  service_level: string;
  review_cycle_days: string;
};

export function PolicyManagementPanel() {
  const queryClient = useQueryClient();
  const [editPolicy, setEditPolicy] = useState<EditPolicyState | null>(null);
  const [autoAssignStatus, setAutoAssignStatus] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewData, setPreviewData] = useState<{
    changes: number;
    ssImpact: string;
    cslImpact: string;
  } | null>(null);

  const { filters } = useGlobalFilterContext();
  const gf = {
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: policyList, isLoading: policyLoading } = useQuery({
    queryKey: queryKeys.policyList(gf),
    queryFn: () => fetchPolicies(gf),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: compliance, isLoading: complianceLoading, refetch: refetchCompliance } = useQuery({
    queryKey: queryKeys.policyCompliance(gf),
    queryFn: () => fetchPolicyCompliance(gf),
    staleTime: STALE.FIVE_MIN,
  });

  const updatePolicyMutation = useMutation({
    mutationFn: ({ policyId, body }: { policyId: string; body: Parameters<typeof updatePolicy>[1] }) =>
      updatePolicy(policyId, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["policy-list"] });
      queryClient.invalidateQueries({ queryKey: ["policy-compliance"] });
      setEditPolicy(null);
    },
  });

  const autoAssignMutation = useMutation({
    mutationFn: async (policies: ReplenishmentPolicy[]) => {
      const results = await Promise.all(
        policies.map((p) => assignPolicy({ segment: p.segment ?? undefined, policy_id: p.policy_id }))
      );
      return results.reduce(
        (acc, r) => ({ assigned: acc.assigned + r.assigned_count, failed: acc.failed + r.failed_count }),
        { assigned: 0, failed: 0 },
      );
    },
    onSuccess: (result) => {
      setAutoAssignStatus(`Assigned ${result.assigned} SKUs${result.failed ? `, ${result.failed} failed` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["policy-compliance"] });
      refetchCompliance();
    },
    onError: () => {
      setAutoAssignStatus("Auto-assign failed. Check auth settings.");
    },
  });

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Policy Management</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Replenishment policies by ABC class and demand variability.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {autoAssignStatus && (
            <span className="text-xs text-muted-foreground">{autoAssignStatus}</span>
          )}
          <button
            className="h-7 rounded border border-input bg-background px-3 text-xs font-medium hover:bg-muted disabled:opacity-50"
            disabled={autoAssignMutation.isPending || policyLoading}
            onClick={() => {
              const policies = policyList?.policies ?? [];
              const unassigned = compliance?.unassigned_count ?? 0;
              setPreviewData({
                changes: unassigned,
                ssImpact: `~$${(unassigned * 12).toLocaleString()}`,
                cslImpact: "+1-3%",
              });
              setShowPreview(true);
            }}
          >
            {autoAssignMutation.isPending ? "Assigning…" : "Auto-assign All"}
          </button>
        </div>
      </div>

      {/* Auto-Assign Impact Preview */}
      {showPreview && previewData && (
        <div className="mb-4 border border-blue-200 rounded-lg p-4 bg-blue-50 dark:bg-blue-950/20">
          <p className="text-sm font-semibold text-foreground mb-2">Impact Preview</p>
          <div className="grid grid-cols-3 gap-4 mb-3">
            <div>
              <p className="text-[10px] text-muted-foreground">Items Changing Policy</p>
              <p className="text-sm font-bold">{previewData.changes.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Est. SS Reduction</p>
              <p className="text-sm font-bold text-green-600">{previewData.ssImpact}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Est. Service Level Change</p>
              <p className="text-sm font-bold text-blue-600">{previewData.cslImpact}</p>
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground mb-3">
            Auto-assign maps: Lumpy items to Manual Review, A-class to Continuous ROP, B-class to Periodic Review, C-class to Min-Max. Manual overrides will not be changed.
          </p>
          <div className="flex gap-2">
            <button
              className="px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground font-medium hover:bg-primary/90 disabled:opacity-50"
              disabled={autoAssignMutation.isPending}
              onClick={() => {
                setShowPreview(false);
                setAutoAssignStatus(null);
                autoAssignMutation.mutate(policyList?.policies ?? []);
              }}
            >
              {autoAssignMutation.isPending ? "Assigning..." : "Confirm Auto-assign"}
            </button>
            <button
              className="px-3 py-1.5 text-xs rounded border hover:bg-muted"
              onClick={() => { setShowPreview(false); setPreviewData(null); }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Compliance Gauge */}
      <div className="mb-5">
        {complianceLoading ? (
          <div className="text-xs text-muted-foreground">Loading compliance…</div>
        ) : compliance ? (
          <div className="flex items-center gap-6">
            {/* Ring gauge (SVG) */}
            <div className="relative flex-shrink-0">
              <svg width="80" height="80" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="34" fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
                <circle
                  cx="40" cy="40" r="34"
                  fill="none"
                  stroke={compliance.assignment_pct >= 80 ? "#22c55e" : compliance.assignment_pct >= 50 ? "#f59e0b" : "#ef4444"}
                  strokeWidth="8"
                  strokeDasharray={`${(compliance.assignment_pct / 100) * 213.6} 213.6`}
                  strokeLinecap="round"
                  transform="rotate(-90 40 40)"
                />
                <text x="40" y="44" textAnchor="middle" fontSize="13" fontWeight="600" fill="currentColor">
                  {compliance.assignment_pct.toFixed(0)}%
                </text>
              </svg>
            </div>
            <div className="text-sm text-foreground">
              <p className="font-medium">DFU Coverage</p>
              <p className="text-muted-foreground text-xs mt-0.5">
                {compliance.assigned_count.toLocaleString()} of {compliance.total_skus.toLocaleString()} SKUs assigned
              </p>
              {compliance.unassigned_count > 0 && (
                <p className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">
                  {compliance.unassigned_count.toLocaleString()} SKUs unassigned
                </p>
              )}
            </div>
          </div>
        ) : null}
      </div>

      {/* Policy Cards */}
      {policyLoading ? (
        <div className="text-xs text-muted-foreground">Loading policies…</div>
      ) : (policyList?.policies ?? []).length === 0 ? (
        <EmptyState
          icon={Shield}
          title="No replenishment policies configured"
          description="Policies define review cycle, service level, and order method (ROP, Min/Max, Periodic, Manual) per demand segment. Auto-assignment maps each DFU to its policy based on ABC class and variability."
          steps={[
            { label: "Apply schema (first time only)", command: "make policy-schema" },
            { label: "Seed default policies and assign SKUs", command: "make policy-assign" },
          ]}
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          {(policyList?.policies ?? []).map((policy) => (
            <div
              key={policy.policy_id}
              className="rounded-md border bg-background p-3 flex flex-col gap-1.5"
            >
              <div className="flex items-start justify-between gap-1">
                <span className="text-xs font-semibold text-foreground leading-tight">{policy.policy_name}</span>
                <button
                  className="text-xs text-muted-foreground hover:text-foreground underline flex-shrink-0"
                  onClick={() =>
                    setEditPolicy({
                      policy,
                      service_level: policy.service_level != null ? String(policy.service_level) : "",
                      review_cycle_days: policy.review_cycle_days != null ? String(policy.review_cycle_days) : "",
                    })
                  }
                >
                  Edit
                </button>
              </div>
              <span
                className={`self-start rounded px-1.5 py-0.5 text-xs font-medium ${POLICY_TYPE_COLORS[policy.policy_type] ?? "bg-gray-100 text-gray-700"}`}
                title={POLICY_TYPE_DESCRIPTIONS[policy.policy_type] ?? ""}
              >
                {policy.policy_type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
              </span>
              <div className="text-xs text-muted-foreground space-y-0.5">
                {policy.segment && <p>Segment: <span className="font-medium text-foreground">{policy.segment}</span></p>}
                {policy.service_level != null && (
                  <p>Service level: <span className="font-medium text-foreground">{(policy.service_level * 100).toFixed(0)}%</span></p>
                )}
                {policy.review_cycle_days != null && (
                  <p>Review cycle: <span className="font-medium text-foreground">{policy.review_cycle_days}d</span></p>
                )}
                <p>SKUs: <span className="font-medium text-foreground">{policy.sku_count.toLocaleString()}</span></p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Compliance Table (by policy) */}
      {compliance && Object.keys(compliance.by_policy).length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-foreground mb-2">Policy Compliance</h4>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-4">Policy</th>
                  <th className="text-left py-1 pr-4">Type</th>
                  <th className="text-right py-1 pr-4">SKUs</th>
                  <th className="text-right py-1 pr-4" title="Percentage of SKUs in this policy currently below their safety stock target">Below SS%</th>
                  <th className="text-right py-1 pr-4" title="Average ratio of on-hand inventory to safety stock target (1.0 = exactly at target, >1.0 = healthy)">SS Coverage</th>
                  <th className="text-right py-1" title="Average Days of Supply — how many days the current inventory will last at current consumption rate">Avg DOS</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(compliance.by_policy)
                  .sort(([, a], [, b]) => (b.below_ss_pct ?? -1) - (a.below_ss_pct ?? -1))
                  .map(([pid, bp]) => (
                    <tr key={pid} className="border-b last:border-0">
                      <td className="py-1 pr-4 font-medium">{bp.policy_name}</td>
                      <td className="py-1 pr-4">
                        <span
                          className={`rounded px-1.5 py-0.5 text-xs ${POLICY_TYPE_COLORS[bp.policy_type] ?? ""}`}
                          title={POLICY_TYPE_DESCRIPTIONS[bp.policy_type] ?? ""}
                        >
                          {bp.policy_type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                        </span>
                      </td>
                      <td className="py-1 pr-4 text-right">{bp.sku_count.toLocaleString()}</td>
                      <td className="py-1 pr-4 text-right">{formatPct(bp.below_ss_pct)}</td>
                      <td className="py-1 pr-4 text-right">{formatPct(bp.avg_ss_coverage)}</td>
                      <td className="py-1 text-right">{formatFixed(bp.avg_dos, 1)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Edit Policy Modal */}
      {editPolicy && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="rounded-lg border bg-card p-6 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-foreground mb-4">Edit Policy</h3>
            <p className="text-xs text-muted-foreground mb-4">{editPolicy.policy.policy_name}</p>
            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">Service Level (0–1)</span>
                <input
                  className="h-8 rounded border border-input bg-background px-2 text-xs"
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={editPolicy.service_level}
                  onChange={(e) => setEditPolicy((s) => s ? { ...s, service_level: e.target.value } : null)}
                />
                <span className="text-xs text-muted-foreground">(e.g. 0.95 = 95% fulfillment probability)</span>
              </label>
              {editPolicy.policy.policy_type === "periodic_review" && (
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Review Cycle (days)</span>
                  <input
                    className="h-8 rounded border border-input bg-background px-2 text-xs"
                    type="number"
                    min="1"
                    value={editPolicy.review_cycle_days}
                    onChange={(e) => setEditPolicy((s) => s ? { ...s, review_cycle_days: e.target.value } : null)}
                  />
                </label>
              )}
            </div>
            <div className="flex gap-2 mt-5">
              <button
                className="flex-1 h-8 rounded bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
                disabled={updatePolicyMutation.isPending}
                onClick={() => {
                  const body: Parameters<typeof updatePolicy>[1] = {};
                  const sl = parseFloat(editPolicy.service_level);
                  if (!isNaN(sl)) body.service_level = sl;
                  const rcd = parseInt(editPolicy.review_cycle_days, 10);
                  if (!isNaN(rcd)) body.review_cycle_days = rcd;
                  updatePolicyMutation.mutate({ policyId: editPolicy.policy.policy_id, body });
                }}
              >
                {updatePolicyMutation.isPending ? "Saving…" : "Save"}
              </button>
              <button
                className="flex-1 h-8 rounded border border-input bg-background text-xs hover:bg-muted"
                onClick={() => setEditPolicy(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
