/**
 * Unified severity color system — single source of truth.
 * Import this everywhere instead of defining severity styles inline.
 *
 * Every class below is a semantic token utility (`bg-destructive/10`, not
 * `bg-red-100`), so a single string is correct in light, soft, AND dark mode
 * — the HSL value swaps under the hood via the CSS var, no `dark:` sibling
 * needed. `badge.tsx` and `lib/severityBadge.ts` both derive their tone
 * classes from `SEVERITY_CONFIG` / `STATUS_TONE_BADGE` below instead of
 * keeping their own copies.
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
    badge: "border-destructive/25 bg-destructive/10 text-destructive",
    border: "border-l-destructive",
    dot: "bg-destructive",
    bg: "bg-destructive/10",
    text: "text-destructive",
    ring: "ring-destructive/20",
    icon: "text-destructive",
    rowBg: "bg-destructive/10",
  },
  high: {
    label: "High",
    badge: "border-severity-high/25 bg-severity-high/10 text-severity-high",
    border: "border-l-severity-high",
    dot: "bg-severity-high",
    bg: "bg-severity-high/10",
    text: "text-severity-high",
    ring: "ring-severity-high/20",
    icon: "text-severity-high",
    rowBg: "bg-severity-high/10",
  },
  medium: {
    label: "Medium",
    badge: "border-warning/25 bg-warning/10 text-warning",
    border: "border-l-warning",
    dot: "bg-warning",
    bg: "bg-warning/10",
    text: "text-warning",
    ring: "ring-warning/20",
    icon: "text-warning",
    rowBg: "bg-warning/10",
  },
  low: {
    label: "Low",
    badge: "border-border bg-muted text-muted-foreground",
    border: "border-l-muted-foreground/40",
    dot: "bg-muted-foreground/60",
    bg: "bg-muted",
    text: "text-muted-foreground",
    ring: "ring-border",
    icon: "text-muted-foreground",
    rowBg: "bg-muted",
  },
};

/**
 * Badge/pill tone classes for statuses that aren't severities. `info` and
 * `success` sit alongside `SEVERITY_CONFIG` so every consumer (badge.tsx,
 * severityBadge.ts) derives every tone from this one module.
 */
export const STATUS_TONE_BADGE: Record<"info" | "success", string> = {
  info: "border-info/25 bg-info/10 text-info",
  success: "border-success/25 bg-success/10 text-success",
};

/** Get severity config with fallback to "low" for unknown values */
export function getSeverityConfig(severity: string): typeof SEVERITY_CONFIG.critical {
  return SEVERITY_CONFIG[severity as Severity] ?? SEVERITY_CONFIG.low;
}

/** Sort comparator for severity (critical first) */
export function compareSeverity(a: string, b: string): number {
  return (SEVERITY_ORDER[a as Severity] ?? 99) - (SEVERITY_ORDER[b as Severity] ?? 99);
}
