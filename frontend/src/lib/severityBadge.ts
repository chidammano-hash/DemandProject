/**
 * Shared themed severity / status badge classes (U5.1).
 *
 * Before this helper, 30+ tabs hand-rolled `bg-{color}-100 text-{color}-700`
 * status chips with NO `dark:` companion, so in Dark theme they rendered as a
 * pale pastel tint with dark text on a near-black surface — barely separable
 * from the page. Each entry below pairs a Light tint with a `dark:` tint so the
 * pill stays WCAG-legible in Light / Soft / Dark. Mirrors the `togglePillClass()`
 * pattern used by the Customer-Analytics pills.
 *
 * Use for severity (`critical`/`high`/`medium`/`low`) and status
 * (`info`/`warning`/`success`/`neutral`) chips. Unknown keys fall back to a
 * neutral themed tint (never a bare bg-*-100 with no dark sibling).
 */
const SEVERITY_BADGE_CLASSES: Record<string, string> = {
  critical: "bg-red-100 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900/50",
  high: "bg-orange-100 text-orange-700 border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-900/50",
  medium: "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900/50",
  low: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-900/50",
  info: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-900/50",
  warning: "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900/50",
  success: "bg-green-100 text-green-700 border-green-200 dark:bg-green-950/40 dark:text-green-300 dark:border-green-900/50",
  neutral: "bg-muted text-muted-foreground border-border",
};

const NEUTRAL_BADGE = SEVERITY_BADGE_CLASSES.neutral;

export function severityBadgeClass(severity: string | null | undefined): string {
  if (!severity) return NEUTRAL_BADGE;
  return SEVERITY_BADGE_CLASSES[severity.toLowerCase()] ?? NEUTRAL_BADGE;
}
