/**
 * Shared constants, types, and helper functions for Storyboard panel components.
 */

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------
export const EXCEPTION_TYPE_COLORS: Record<string, string> = {
  forecast_bias: "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/40 dark:text-blue-300 dark:border-blue-700",
  stockout_risk: "bg-red-100 text-red-800 border-red-200 dark:bg-red-900/40 dark:text-red-300 dark:border-red-700",
  accuracy_drop: "bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/40 dark:text-orange-300 dark:border-orange-700",
  excess_risk: "bg-cyan-100 text-cyan-800 border-cyan-200 dark:bg-cyan-900/40 dark:text-cyan-300 dark:border-cyan-700",
  model_drift: "bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/40 dark:text-yellow-300 dark:border-yellow-700",
  new_item: "bg-green-100 text-green-800 border-green-200 dark:bg-green-900/40 dark:text-green-300 dark:border-green-700",
};

export const EXCEPTION_TYPE_ICONS: Record<string, string> = {
  forecast_bias: "~",
  stockout_risk: "!",
  accuracy_drop: "v",
  excess_risk: "+",
  model_drift: "*",
  new_item: "N",
};

export const DECISION_TYPE_COLORS: Record<string, string> = {
  override_forecast: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
  accept_exception: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  escalate: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  dismiss: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  request_info: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
};

export const STATUS_COLORS: Record<string, string> = {
  open: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  investigating: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
  resolved: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  dismissed: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
};

export const STATUS_DOT: Record<string, string> = {
  open: "bg-red-500",
  investigating: "bg-yellow-500",
  resolved: "bg-green-500",
  dismissed: "bg-gray-400",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
export function severityLabel(score: number): string {
  if (score >= 0.75) return "Critical";
  if (score >= 0.50) return "High";
  if (score >= 0.25) return "Medium";
  return "Low";
}

export function severityColorClass(severity: number): string {
  if (severity >= 0.75) return "text-red-600 dark:text-red-400";
  if (severity >= 0.50) return "text-orange-600 dark:text-orange-400";
  if (severity >= 0.25) return "text-yellow-600 dark:text-yellow-400";
  return "text-green-600 dark:text-green-400";
}

export function severityBg(severity: number): string {
  if (severity >= 0.75) return "bg-red-500";
  if (severity >= 0.50) return "bg-orange-500";
  if (severity >= 0.25) return "bg-yellow-500";
  return "bg-green-500";
}

// Canonical formatters \u2014 re-exported under the panel-local names the storyboard
// components already use (callers always pass an explicit decimal count to `fmt`).
export { formatFixed as fmt, formatCurrencyFull as fmtCurrency } from "@/lib/formatters";

export function daysAgo(dateStr: string): string {
  const ms = Date.now() - new Date(dateStr).getTime();
  const days = Math.floor(ms / (1000 * 60 * 60 * 24));
  if (days === 0) return "today";
  if (days === 1) return "1d ago";
  return `${days}d ago`;
}

// ---------------------------------------------------------------------------
// Filter / type constants
// ---------------------------------------------------------------------------
export const EXCEPTION_TYPES = [
  "all",
  "forecast_bias",
  "stockout_risk",
  "accuracy_drop",
  "excess_risk",
  "model_drift",
  "new_item",
];

export const STATUS_FILTERS = ["all", "open", "investigating", "resolved", "dismissed"];

export const EXCEPTION_TYPE_LABELS: Record<string, string> = {
  all: "All",
  forecast_bias: "Forecast Bias",
  stockout_risk: "Stockout Risk",
  accuracy_drop: "Accuracy Drop",
  excess_risk: "Excess Risk",
  model_drift: "Model Drift",
  new_item: "New Item",
};

export const DECISION_TYPES = [
  { value: "override_forecast", label: "Override Forecast", desc: "Replace the forecast with your own estimate" },
  { value: "accept_exception", label: "Accept & Monitor", desc: "Acknowledge the exception and continue tracking" },
  { value: "escalate", label: "Escalate", desc: "Flag for senior review or cross-functional discussion" },
  { value: "dismiss", label: "Dismiss", desc: "No action needed \u2014 false positive or resolved naturally" },
  { value: "request_info", label: "Request Information", desc: "Need more data before making a decision" },
];

export const PAGE_SIZE = 20;
