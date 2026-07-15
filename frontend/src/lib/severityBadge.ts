import { SEVERITY_CONFIG, STATUS_TONE_BADGE } from "@/constants/severity";

/**
 * @deprecated Thin wrapper over `SEVERITY_CONFIG` / `STATUS_TONE_BADGE` in
 * `constants/severity.ts`, kept only until the remaining tab callers migrate
 * to those exports directly. Every tone below is a token utility
 * (`bg-destructive/10`), so it stays legible in light, soft, AND dark mode —
 * no `dark:` sibling needed. Accepts severity (`critical`/`high`/`medium`/
 * `low`) and status (`info`/`warning`/`success`/`neutral`) keys; unknown keys
 * fall back to the neutral/muted tone.
 */
const NEUTRAL_BADGE = SEVERITY_CONFIG.low.badge;

const TONE_BADGE_CLASSES: Record<string, string> = {
  critical: SEVERITY_CONFIG.critical.badge,
  high: SEVERITY_CONFIG.high.badge,
  medium: SEVERITY_CONFIG.medium.badge,
  low: SEVERITY_CONFIG.low.badge,
  warning: SEVERITY_CONFIG.medium.badge,
  info: STATUS_TONE_BADGE.info,
  success: STATUS_TONE_BADGE.success,
  neutral: NEUTRAL_BADGE,
};

/** @deprecated import `SEVERITY_CONFIG` / `STATUS_TONE_BADGE` from `@/constants/severity` instead. */
export function severityBadgeClass(severity: string | null | undefined): string {
  if (!severity) return NEUTRAL_BADGE;
  return TONE_BADGE_CLASSES[severity.toLowerCase()] ?? NEUTRAL_BADGE;
}
