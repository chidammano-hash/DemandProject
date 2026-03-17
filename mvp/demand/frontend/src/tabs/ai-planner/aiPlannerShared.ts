/**
 * Shared utilities, constants, and types used across AI Planner panel components.
 */
import type { AiInsight, InsightSeverity, InsightType } from "@/types/ai-planner";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle2,
  Package,
  TrendingDown,
  BarChart3,
  Target,
  Zap,
  DollarSign,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------
export const SEVERITY_ORDER: Record<InsightSeverity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

export const SEVERITY_STYLES: Record<InsightSeverity, { badge: string; border: string; dot: string }> = {
  critical: {
    badge: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    border: "border-l-red-500",
    dot: "bg-red-500",
  },
  high: {
    badge: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
    border: "border-l-orange-500",
    dot: "bg-orange-500",
  },
  medium: {
    badge: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
    border: "border-l-yellow-500",
    dot: "bg-yellow-500",
  },
  low: {
    badge: "bg-gray-100 text-gray-700 dark:bg-gray-700/40 dark:text-gray-300",
    border: "border-l-gray-400",
    dot: "bg-gray-400",
  },
};

// ---------------------------------------------------------------------------
// Insight type config
// ---------------------------------------------------------------------------
export const INSIGHT_TYPE_ICONS: Record<InsightType, React.FC<{ className?: string }>> = {
  stockout_risk: Package,
  excess_inventory: TrendingDown,
  forecast_bias: BarChart3,
  policy_gap: Target,
  champion_degradation: Zap,
};

export const INSIGHT_TYPE_LABELS: Record<InsightType, string> = {
  stockout_risk: "Stockout Risk",
  excess_inventory: "Excess Inventory",
  forecast_bias: "Forecast Bias",
  policy_gap: "Policy Gap",
  champion_degradation: "Model Degradation",
};

export const INSIGHT_TYPE_EXPLAINERS: Record<InsightType, string> = {
  stockout_risk: "Inventory is projected to run out before the next replenishment arrives. Days of Supply (DOS) is below the safety threshold relative to lead time. This can result in lost sales, backorders, and damaged customer relationships. Immediate reorder or expediting may be needed.",
  excess_inventory: "On-hand inventory significantly exceeds projected demand, tying up working capital and warehouse space. This often results from demand over-forecasting, order policy misalignment, or stale cluster assignments. Consider reducing order quantities, delaying planned orders, or redistributing stock.",
  forecast_bias: "The forecasting model has been systematically over- or under-predicting demand for this item-location over multiple months. Persistent bias compounds over time \u2014 over-forecasting leads to excess stock, under-forecasting leads to stockouts. The root cause may be a trend shift, seasonality change, or model staleness.",
  policy_gap: "The assigned replenishment policy does not match the demand characteristics of this item. For example, a high-variability item may be using a continuous ROP policy that doesn\u2019t account for demand spikes, or a low-volume item may have an unnecessarily aggressive reorder policy. Realigning the policy to the actual demand profile can reduce both stockouts and excess.",
  champion_degradation: "The champion forecasting model\u2019s accuracy has declined significantly compared to its historical performance or alternative models. This may indicate a structural shift in demand patterns that the model hasn\u2019t adapted to. Consider retraining, switching to an alternative model, or investigating the underlying demand change.",
};

// ---------------------------------------------------------------------------
// Severity icon helper
// ---------------------------------------------------------------------------
export function SeverityIconComponent({ severity }: { severity: InsightSeverity }) {
  if (severity === "critical") return AlertCircle;
  if (severity === "high") return AlertTriangle;
  if (severity === "medium") return Info;
  return CheckCircle2;
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------
export type ConfidenceTier = "high" | "medium" | "low";

export function deriveConfidence(insight: AiInsight): ConfidenceTier {
  const wape = insight.champion_wape ? insight.champion_wape * 100 : null;
  const bias = insight.forecast_bias_pct ? Math.abs(insight.forecast_bias_pct) : null;
  const hasFinancial = insight.financial_impact_estimate != null && insight.financial_impact_estimate > 0;

  const strongSignal = (wape != null && wape > 50) || (bias != null && bias > 50);
  const moderateSignal = (wape != null && wape > 35) || (bias != null && bias > 20);

  if (strongSignal && hasFinancial) return "high";
  if (moderateSignal || strongSignal) return "medium";
  return "low";
}

export const CONFIDENCE_STYLES: Record<ConfidenceTier, { label: string; className: string; tooltip: string }> = {
  high: {
    label: "HIGH",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-400",
    tooltip: "High confidence \u2014 forecast WAPE < 35% with financial impact confirmed. Act on this insight with high certainty.",
  },
  medium: {
    label: "MED",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-400",
    tooltip: "Medium confidence \u2014 moderate forecast bias or variability detected. Review the AI reasoning before acting.",
  },
  low: {
    label: "LOW",
    className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400",
    tooltip: "Low confidence \u2014 limited signal strength or no financial impact data. Treat as an early warning, not a directive.",
  },
};

// ---------------------------------------------------------------------------
// Causal chain types
// ---------------------------------------------------------------------------
export interface CausalLink {
  layer: "forecast" | "inventory" | "policy" | "financial";
  signal: string;
  impact: string;
  isAlert: boolean;
}

export const CAUSAL_LAYER_LABELS: Record<CausalLink["layer"], string> = {
  forecast: "Forecast",
  inventory: "Inventory",
  policy: "Policy",
  financial: "Impact",
};

export const CAUSAL_LAYER_ICONS: Record<CausalLink["layer"], React.FC<{ className?: string }>> = {
  forecast: BarChart3,
  inventory: Package,
  policy: Target,
  financial: DollarSign,
};

export function buildCausalChain(insight: AiInsight): CausalLink[] {
  const chain: CausalLink[] = [];
  const fmtCurrency = (n: number) => `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  // Forecast layer
  if (insight.champion_wape != null || insight.forecast_bias_pct != null) {
    const wapePct = insight.champion_wape != null
      ? `WAPE ${(insight.champion_wape * 100).toFixed(1)}%` : null;
    const biasPct = insight.forecast_bias_pct != null
      ? `Bias ${insight.forecast_bias_pct > 0 ? "+" : ""}${(insight.forecast_bias_pct * 100).toFixed(1)}%`
      : null;
    const signal = [wapePct, biasPct].filter(Boolean).join(" \u00b7 ") || "\u2013";
    const isAlert =
      (insight.champion_wape ?? 0) > 0.35 || Math.abs(insight.forecast_bias_pct ?? 0) > 0.20;
    chain.push({
      layer: "forecast",
      signal,
      impact: insight.insight_type === "champion_degradation" ? "Model degrading"
            : insight.insight_type === "forecast_bias" ? "Systematic deviation"
            : "Inaccurate signal",
      isAlert,
    });
  }

  // Inventory layer
  if (insight.dos != null) {
    const lt = insight.total_lt_days;
    const signal = `DOS ${insight.dos.toFixed(0)}d${lt ? ` (LT ${lt}d)` : ""}`;
    const isBelow = lt != null && insight.dos < lt;
    const isExcess = insight.dos > 180;
    chain.push({
      layer: "inventory",
      signal,
      impact: isBelow ? "Below lead time" : isExcess ? "Excess stock" : "Tight coverage",
      isAlert: isBelow || isExcess,
    });
  }

  // Policy layer
  if (insight.current_policy_id || insight.insight_type === "policy_gap") {
    chain.push({
      layer: "policy",
      signal: insight.current_policy_id ?? "No policy",
      impact: insight.insight_type === "policy_gap" ? "Mismatch detected"
            : !insight.current_policy_id ? "Unassigned"
            : "Active policy",
      isAlert: insight.insight_type === "policy_gap" || !insight.current_policy_id,
    });
  }

  // Financial layer \u2014 always shown
  chain.push({
    layer: "financial",
    signal: insight.financial_impact_estimate != null
      ? fmtCurrency(insight.financial_impact_estimate)
      : "Est. pending",
    impact: insight.insight_type === "stockout_risk" ? "Lost sales"
          : insight.insight_type === "excess_inventory" ? "Capital locked"
          : "At risk",
    isAlert: (insight.financial_impact_estimate ?? 0) > 1000,
  });

  return chain;
}

// ---------------------------------------------------------------------------
// Confirm modal state
// ---------------------------------------------------------------------------
export interface ConfirmActionState {
  insight: AiInsight;
  status: "acknowledged" | "resolved";
  label: string;
  verb: string;
}

export const SEVERITY_THRESHOLD_LABELS: Record<InsightSeverity, string> = {
  critical: "Critical only",
  high: "High and above",
  medium: "Medium and above",
  low: "All severities",
};
