/**
 * Command Center — exception normalization helpers + severity/label constants.
 *
 * Merges AI Planner insights and Storyboard exceptions into a single
 * `UnifiedException` shape and provides the severity ordering, border/dot color
 * maps, source badges, and type-label dictionaries the feed cards consume.
 */
import type { AiInsight } from "@/types/ai-planner";
import type { StoryboardException } from "@/types/storyboard";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface UnifiedException {
  id: string;
  source: "ai" | "rule";
  severity: "critical" | "high" | "medium" | "low";
  type: string;
  typeLabel: string;
  itemNo: string;
  itemDesc?: string;
  location: string;
  summary: string;
  recommendation?: string;
  financialImpact?: number;
  createdAt: string;
  status: string;
  originalData: AiInsight | StoryboardException;
}

// ---------------------------------------------------------------------------
// Severity helpers
// ---------------------------------------------------------------------------
export const SEVERITY_ORDER: Record<string, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

export const SEVERITY_BORDER: Record<string, string> = {
  critical: "border-l-red-500",
  high: "border-l-orange-500",
  medium: "border-l-yellow-500",
  low: "border-l-gray-400",
};

export const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-yellow-500",
  low: "bg-gray-400",
};

export const SOURCE_BADGE: Record<string, { label: string; className: string }> = {
  ai: {
    label: "AI",
    className:
      "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300",
  },
  rule: {
    label: "Rule",
    className:
      "bg-gray-100 text-gray-700 dark:bg-gray-700/40 dark:text-gray-300",
  },
};

// Map storyboard severity (0-1 float) to categorical
export function mapRuleSeverity(
  score: number
): "critical" | "high" | "medium" | "low" {
  if (score >= 0.75) return "critical";
  if (score >= 0.5) return "high";
  if (score >= 0.25) return "medium";
  return "low";
}

// U7.10 — numeric severity band [min, max) for each categorical chip, matching
// the mapRuleSeverity cutoffs. Pushed down to /storyboard/exceptions so the
// server returns rows in the selected band — the feed is sorted critical-first
// and capped at 50, so client-side filtering of an all-critical page left the
// High/Medium/Low chips structurally empty (dead controls).
export const SEVERITY_BANDS: Record<string, { min: number; max: number }> = {
  critical: { min: 0.75, max: 1.0 },
  high: { min: 0.5, max: 0.75 },
  medium: { min: 0.25, max: 0.5 },
  low: { min: 0.0, max: 0.25 },
};

// Exception type labels for storyboard. Includes the AI-insight rule enums AND
// the replenishment enums the cycle-4 _replenishment_fallback now emits
// (below_ss/stockout/below_rop/excess/zero_velocity) so the card chip never
// leaks a raw enum and names an exception with the SAME words the Inv Planning
// action feed uses (U4.1).
export const EXCEPTION_TYPE_LABELS: Record<string, string> = {
  forecast_bias: "Forecast Bias",
  stockout_risk: "Stockout Risk",
  accuracy_drop: "Accuracy Drop",
  excess_risk: "Excess Risk",
  model_drift: "Model Drift",
  new_item: "New Item",
  // Replenishment exception enums (storyboard fallback)
  below_ss: "Below Safety Stock",
  below_rop: "Below Reorder Point",
  below_rop_critical: "Critically Below Reorder Point",
  stockout: "Stockout",
  excess: "Excess Inventory",
  zero_velocity: "Zero Velocity",
};

// AI insight type labels
export const INSIGHT_TYPE_LABELS: Record<string, string> = {
  stockout_risk: "Stockout Risk",
  excess_inventory: "Excess Inventory",
  forecast_bias: "Forecast Bias",
  policy_gap: "Policy Gap",
  champion_degradation: "Model Degradation",
};

// ---------------------------------------------------------------------------
// Normalization: merge AI insights + storyboard exceptions
// ---------------------------------------------------------------------------
export function normalizeAiInsight(insight: AiInsight): UnifiedException {
  return {
    id: `ai-${insight.insight_id}`,
    source: "ai",
    severity: insight.severity,
    type: insight.insight_type,
    typeLabel: INSIGHT_TYPE_LABELS[insight.insight_type] ?? insight.insight_type,
    itemNo: insight.item_id,
    location: insight.loc,
    summary: insight.summary,
    recommendation: insight.recommendation,
    financialImpact: insight.financial_impact_estimate ?? undefined,
    createdAt: insight.created_at,
    status: insight.status,
    originalData: insight,
  };
}

// U2.2 — the replenishment fallback headline is "<Type> — <item> @ <loc>", but
// the row already prints "<item> @ <loc>" on its identity line. Strip a trailing
// "— <item_id> @ <loc>" so the code pair is not shown twice in a dense triage
// list. Headlines that do not restate the identity (AI-style summaries) are
// returned unchanged.
export function dropRedundantIdentitySuffix(
  headline: string,
  itemId: string,
  loc: string
): string {
  const suffix = `— ${itemId} @ ${loc}`;
  if (headline.endsWith(suffix)) {
    return headline.slice(0, headline.length - suffix.length).trim();
  }
  return headline;
}

export function normalizeStoryboardException(
  exc: StoryboardException
): UnifiedException {
  const rawHeadline = exc.headline ?? `${exc.exception_type} detected`;
  return {
    id: `rule-${exc.exception_id}`,
    source: "rule",
    severity: mapRuleSeverity(exc.severity),
    type: exc.exception_type,
    typeLabel:
      EXCEPTION_TYPE_LABELS[exc.exception_type] ?? exc.exception_type,
    itemNo: exc.item_id,
    itemDesc: exc.item_desc ?? undefined,
    location: exc.loc,
    summary: dropRedundantIdentitySuffix(rawHeadline, exc.item_id, exc.loc),
    recommendation: undefined,
    financialImpact: exc.financial_impact ?? undefined,
    createdAt: exc.generated_at,
    status: exc.status,
    originalData: exc,
  };
}
