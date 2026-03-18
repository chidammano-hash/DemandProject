/**
 * Unified severity color system — single source of truth.
 * Import this everywhere instead of defining severity styles inline.
 */

export type Severity = "critical" | "high" | "medium" | "low";

export const SEVERITY_ORDER: Record<Severity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

export const SEVERITY_CONFIG: Record<Severity, {
  label: string;
  badge: string;
  border: string;
  dot: string;
  bg: string;
  text: string;
  ring: string;
  icon: string;
  rowBg: string;
}> = {
  critical: {
    label: "Critical",
    badge: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    border: "border-l-red-500",
    dot: "bg-red-500",
    bg: "bg-red-50 dark:bg-red-950/20",
    text: "text-red-700 dark:text-red-300",
    ring: "ring-red-500/20",
    icon: "text-red-500",
    rowBg: "bg-red-50 dark:bg-red-950/20",
  },
  high: {
    label: "High",
    badge: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
    border: "border-l-orange-500",
    dot: "bg-orange-500",
    bg: "bg-orange-50 dark:bg-orange-950/20",
    text: "text-orange-700 dark:text-orange-300",
    ring: "ring-orange-500/20",
    icon: "text-orange-500",
    rowBg: "bg-orange-50 dark:bg-orange-950/20",
  },
  medium: {
    label: "Medium",
    badge: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
    border: "border-l-yellow-500",
    dot: "bg-yellow-500",
    bg: "bg-yellow-50 dark:bg-yellow-950/20",
    text: "text-yellow-700 dark:text-yellow-300",
    ring: "ring-yellow-500/20",
    icon: "text-yellow-500",
    rowBg: "bg-yellow-50 dark:bg-yellow-950/20",
  },
  low: {
    label: "Low",
    badge: "bg-gray-100 text-gray-700 dark:bg-gray-700/40 dark:text-gray-300",
    border: "border-l-gray-400",
    dot: "bg-gray-400",
    bg: "bg-gray-50 dark:bg-gray-800/20",
    text: "text-gray-600 dark:text-gray-400",
    ring: "ring-gray-400/20",
    icon: "text-gray-400",
    rowBg: "bg-gray-50 dark:bg-gray-800/20",
  },
};

/** Get severity config with fallback to "low" for unknown values */
export function getSeverityConfig(severity: string): typeof SEVERITY_CONFIG.critical {
  return SEVERITY_CONFIG[severity as Severity] ?? SEVERITY_CONFIG.low;
}

/** Sort comparator for severity (critical first) */
export function compareSeverity(a: string, b: string): number {
  return (SEVERITY_ORDER[a as Severity] ?? 99) - (SEVERITY_ORDER[b as Severity] ?? 99);
}
