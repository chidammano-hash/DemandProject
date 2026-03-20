// Design tokens for Supply Chain Command Center
// Semantic color palette — use these constants instead of raw hex codes.

// AI-generated content — teal distinguishes AI annotations from human data
export const AI_COLOR = '#0D9488';

// Severity palette — strict semantic meaning (hex values for non-Tailwind contexts like charts).
// For Tailwind class-based severity styling, use @/constants/severity.ts instead.
export const CRITICAL_COLOR = '#ef4444';  // Red   — action required NOW
export const HIGH_COLOR = '#f59e0b';      // Amber — high risk / watch
export const MEDIUM_COLOR = '#eab308';    // Yellow — medium risk / monitor
export const HEALTHY_COLOR = '#22c55e';  // Green — healthy / positive
export const NEUTRAL_COLOR = '#6b7280';  // Gray  — inactive / historical / resolved

// Interactive / navigation
export const INTERACTIVE_COLOR = '#2563EB';  // Sapphire-600 — buttons, links, selected state

// UX limits — keep primary views focused
export const MAX_PRIMARY_KPIS = 4;          // Never show more than 4 on primary view
export const MAX_LIST_ITEMS_DEFAULT = 7;    // Work queue default visible items

// AI behaviour
export const AI_CACHE_TTL_MS = 3_600_000;  // 1 hour AI result cache
export const INSIGHT_STALE_DAYS = 7;       // Auto-flag stale open insights

// AI confidence thresholds (signal duration → confidence tier)
export const AI_CONFIDENCE_HIGH_MONTHS = 6;    // 6+ months of aligned signal
export const AI_CONFIDENCE_MEDIUM_MONTHS = 3;  // 3+ months
