/**
 * THE single source of truth for every color in the product.
 *
 * "Control-room indigo": three modes (light = daylight ops, soft = planner's
 * paper, dark = night shift) sharing one semantic contract — the same concept
 * always wears the same color on every screen:
 *
 *   actual / history  -> neutral ink          forecast / champion -> indigo
 *   external / ref    -> sky                  good / on-target    -> emerald
 *   risk / error      -> red                  warning / excess    -> amber
 *   ceiling / capacity-> teal                 AI-generated        -> violet
 *
 * Everything derives from here:
 *   - `constants/themes/general.ts` builds the runtime ProductTheme from it.
 *   - `constants/colors.ts` derives the legacy chart exports from it.
 *   - `hooks/useTheme.ts` emits it as CSS vars (HSL triplets only).
 *   - `src/index.css` fallback blocks mirror `core` verbatim
 *     (enforced by `constants/__tests__/paletteSync.test.ts`, together with
 *     the WCAG contrast gates).
 *
 * Do NOT hardcode colors anywhere else. Charts read
 * `useChartColors().series` / `.roles`; UI reads Tailwind token utilities.
 */

import type { ThemePalette } from "@/types/theme";
import { hexToHslTriplet } from "@/lib/color";

export type ColorMode = "light" | "soft" | "dark";

/** Semantic chart roles. Every value MUST be a member of `series`. */
export interface ChartRoles {
  /** Historical truth (sales, shipments): neutral ink. */
  actual: string;
  /** Any forward forecast line: primary indigo. */
  forecast: string;
  /** The promoted champion forecast: same indigo as forecast. */
  champion: string;
  /** Ceiling / capacity / upper bounds: teal. */
  ceiling: string;
  /** Error, risk, stockout: red. */
  error: string;
  /** Warning, excess, drift: amber. */
  warning: string;
  /** Good, on-target, best-in-class: emerald. */
  good: string;
  /** External / reference feeds: sky. */
  reference: string;
  /** AI-generated series or adjustments: violet (distinct from indigo). */
  ai: string;
}

export interface ChartPaletteSpec {
  /** 8-color categorical series, tuned per mode; >= 3:1 vs the mode background. */
  series: readonly string[];
  roles: ChartRoles;
  /** Muted 6-color ramp for unnamed/overflow series (visually secondary). */
  fallback: readonly string[];
  grid: string;
  axis: string;
  tooltipBg: string;
  tooltipBorder: string;
  /** Good -> bad, 5 stops. */
  heatmapScale: readonly string[];
}

export interface ModePalette {
  /** shadcn-style tokens as HSL triplets (feeds CSS vars + Tailwind). */
  core: ThemePalette;
  /** Chart colors as hex (consumed as JS strings via useChartColors). */
  charts: ChartPaletteSpec;
}

/* ------------------------------------------------------------------ */
/* Chart series per mode                                               */
/* ------------------------------------------------------------------ */

const LIGHT_SERIES = [
  "#4F46E5", // 0 indigo  — forecast / champion
  "#334155", // 1 ink     — actual / history
  "#0C8A5F", // 2 emerald — good / on-target
  "#C26A02", // 3 amber   — warning / excess
  "#0284C7", // 4 sky     — external / reference
  "#DC2626", // 5 red     — risk / error
  "#0F766E", // 6 teal    — ceiling / capacity
  "#A21CAF", // 7 violet-magenta — AI-generated (hue-separated from indigo)
] as const;

const SOFT_SERIES = [
  "#5048B8",
  "#44403C",
  "#0B7E58",
  "#B3660A",
  "#0779B3",
  "#C42F2F",
  "#085E6E",
  "#8A3FC7",
] as const;

const DARK_SERIES = [
  "#818CF8",
  "#C7CFDA",
  "#10B981",
  "#FBBF24",
  "#38BDF8",
  "#F87171",
  "#5EEAD4",
  "#C084FC",
] as const;

function rolesFromSeries(series: readonly string[]): ChartRoles {
  return {
    forecast: series[0],
    champion: series[0],
    actual: series[1],
    good: series[2],
    warning: series[3],
    reference: series[4],
    error: series[5],
    ceiling: series[6],
    ai: series[7],
  };
}

/** chart1..8 core tokens are derived from the hex series — one source per color. */
function chartTriplets(series: readonly string[]) {
  return {
    chart1: hexToHslTriplet(series[0]),
    chart2: hexToHslTriplet(series[1]),
    chart3: hexToHslTriplet(series[2]),
    chart4: hexToHslTriplet(series[3]),
    chart5: hexToHslTriplet(series[4]),
    chart6: hexToHslTriplet(series[5]),
    chart7: hexToHslTriplet(series[6]),
    chart8: hexToHslTriplet(series[7]),
  };
}

/* ------------------------------------------------------------------ */
/* Core tokens per mode                                                */
/* ------------------------------------------------------------------ */

const lightCore: ThemePalette = {
  background: "230 24% 97%",
  foreground: "233 34% 14%",
  card: "0 0% 100%",
  cardForeground: "233 34% 14%",
  primary: "245 55% 48%",
  primaryForeground: "0 0% 100%",
  secondary: "232 20% 95%",
  secondaryForeground: "233 24% 28%",
  muted: "232 20% 94%",
  mutedForeground: "233 12% 41%",
  accent: "243 45% 94%",
  accentForeground: "245 55% 44%",
  border: "231 20% 88%",
  input: "231 20% 88%",
  ring: "245 55% 48%",
  destructive: "4 74% 46%",
  destructiveForeground: "0 0% 100%",
  sidebarBg: "232 22% 95%",
  sidebarForeground: "233 14% 37%",
  sidebarActive: "245 55% 48%",
  sidebarHover: "232 18% 91%",
  sidebarBorder: "231 20% 87%",
  ...chartTriplets(LIGHT_SERIES),
  kpiBest: "160 84% 27%",
  kpiWarning: "4 74% 46%",
  kpiCeiling: "192 85% 30%",
  bgGradientPrimary: "hsla(245, 60%, 50%, 0.05)",
  bgGradientSecondary: "hsla(190, 80%, 45%, 0.04)",
  bgGradientBaseStart: "#FBFCFE",
  bgGradientBaseMid: "#F4F6FB",
  bgGradientBaseEnd: "#EDF0F8",
  success: "160 84% 27%",
  successForeground: "0 0% 100%",
  warning: "32 95% 42%",
  warningForeground: "28 70% 10%",
  info: "212 88% 43%",
  infoForeground: "0 0% 100%",
  severityHigh: "22 88% 38%",
  severityHighForeground: "0 0% 100%",
};

const softCore: ThemePalette = {
  background: "40 24% 96%",
  foreground: "30 12% 16%",
  card: "42 30% 99%",
  cardForeground: "30 12% 16%",
  primary: "245 45% 49%",
  primaryForeground: "0 0% 100%",
  secondary: "38 16% 92%",
  secondaryForeground: "30 10% 30%",
  muted: "38 16% 91%",
  mutedForeground: "30 8% 39%",
  accent: "245 25% 92%",
  accentForeground: "245 45% 45%",
  border: "36 14% 85%",
  input: "36 14% 85%",
  ring: "245 45% 49%",
  destructive: "4 66% 46%",
  destructiveForeground: "0 0% 100%",
  sidebarBg: "38 20% 94%",
  sidebarForeground: "30 10% 36%",
  sidebarActive: "245 45% 49%",
  sidebarHover: "38 14% 89%",
  sidebarBorder: "36 14% 83%",
  ...chartTriplets(SOFT_SERIES),
  kpiBest: "160 70% 26%",
  kpiWarning: "4 66% 44%",
  kpiCeiling: "192 75% 28%",
  bgGradientPrimary: "hsla(35, 60%, 50%, 0.05)",
  bgGradientSecondary: "hsla(245, 40%, 55%, 0.035)",
  bgGradientBaseStart: "#FCFAF6",
  bgGradientBaseMid: "#F7F3EC",
  bgGradientBaseEnd: "#F0EAE0",
  success: "160 70% 26%",
  successForeground: "0 0% 100%",
  warning: "32 92% 42%",
  warningForeground: "28 70% 10%",
  info: "212 72% 41%",
  infoForeground: "0 0% 100%",
  severityHigh: "22 80% 36%",
  severityHighForeground: "0 0% 100%",
};

const darkCore: ThemePalette = {
  background: "235 30% 8%",
  foreground: "228 28% 94%",
  card: "235 26% 12%",
  cardForeground: "228 28% 94%",
  primary: "240 88% 77%",
  primaryForeground: "237 40% 12%",
  secondary: "235 22% 16%",
  secondaryForeground: "230 18% 78%",
  muted: "235 22% 15%",
  mutedForeground: "231 14% 64%",
  accent: "240 25% 18%",
  accentForeground: "240 88% 77%",
  border: "235 20% 19%",
  input: "235 20% 19%",
  ring: "240 88% 77%",
  destructive: "2 68% 46%",
  destructiveForeground: "0 0% 100%",
  sidebarBg: "237 32% 6%",
  sidebarForeground: "230 14% 62%",
  sidebarActive: "240 88% 77%",
  sidebarHover: "235 22% 13%",
  sidebarBorder: "235 20% 15%",
  ...chartTriplets(DARK_SERIES),
  kpiBest: "158 62% 55%",
  kpiWarning: "2 90% 72%",
  kpiCeiling: "187 75% 55%",
  bgGradientPrimary: "hsla(243, 85%, 65%, 0.10)",
  bgGradientSecondary: "hsla(190, 85%, 55%, 0.05)",
  bgGradientBaseStart: "#101226",
  bgGradientBaseMid: "#0C0E1D",
  bgGradientBaseEnd: "#080A14",
  success: "158 62% 50%",
  successForeground: "160 65% 8%",
  warning: "38 95% 58%",
  warningForeground: "30 70% 10%",
  info: "212 95% 68%",
  infoForeground: "220 45% 10%",
  severityHigh: "22 90% 62%",
  severityHighForeground: "24 65% 10%",
};

/* ------------------------------------------------------------------ */
/* Assembled palette                                                   */
/* ------------------------------------------------------------------ */

export const PALETTE: Record<ColorMode, ModePalette> = {
  light: {
    core: lightCore,
    charts: {
      series: LIGHT_SERIES,
      roles: rolesFromSeries(LIGHT_SERIES),
      fallback: ["#64748B", "#78716C", "#0F766E", "#B45309", "#6D28D9", "#0E7490"],
      grid: "#E7EAF2",
      axis: "#68718A",
      tooltipBg: "#FFFFFF",
      tooltipBorder: "#E7EAF2",
      heatmapScale: ["#0C8A5F", "#7CCFAC", "#E8C468", "#EE9080", "#DC2626"],
    },
  },
  soft: {
    core: softCore,
    charts: {
      series: SOFT_SERIES,
      roles: rolesFromSeries(SOFT_SERIES),
      fallback: ["#6B7280", "#847C74", "#0F6E63", "#A25E0B", "#6D5AB8", "#0E7490"],
      grid: "#E6E0D4",
      axis: "#877E72",
      tooltipBg: "#FDFCF9",
      tooltipBorder: "#E6E0D4",
      heatmapScale: ["#0B7E58", "#8AD0AE", "#DDAE4A", "#E08A7A", "#C42F2F"],
    },
  },
  dark: {
    core: darkCore,
    charts: {
      series: DARK_SERIES,
      roles: rolesFromSeries(DARK_SERIES),
      fallback: ["#94A3B8", "#A8A29E", "#99F6E4", "#FCD34D", "#C4B5FD", "#67E8F9"],
      grid: "#252941",
      axis: "#8A93AB",
      tooltipBg: "#181B2E",
      tooltipBorder: "#252941",
      heatmapScale: ["#34D399", "#A7F3D0", "#FBBF24", "#F87171", "#EF4444"],
    },
  },
};

/**
 * CSS var name -> ThemePalette key. Shared by `useTheme.applyPalette` and
 * `paletteSync.test.ts` so the runtime emission and the index.css fallbacks
 * can never drift apart.
 */
export const CSS_VAR_MAP: ReadonlyArray<readonly [string, keyof ThemePalette]> = [
  ["--background", "background"],
  ["--foreground", "foreground"],
  ["--card", "card"],
  ["--card-foreground", "cardForeground"],
  ["--primary", "primary"],
  ["--primary-foreground", "primaryForeground"],
  ["--secondary", "secondary"],
  ["--secondary-foreground", "secondaryForeground"],
  ["--muted", "muted"],
  ["--muted-foreground", "mutedForeground"],
  ["--accent", "accent"],
  ["--accent-foreground", "accentForeground"],
  ["--border", "border"],
  ["--input", "input"],
  ["--ring", "ring"],
  ["--destructive", "destructive"],
  ["--destructive-foreground", "destructiveForeground"],
  ["--sidebar-bg", "sidebarBg"],
  ["--sidebar-foreground", "sidebarForeground"],
  ["--sidebar-active", "sidebarActive"],
  ["--sidebar-hover", "sidebarHover"],
  ["--sidebar-border", "sidebarBorder"],
  ["--chart-1", "chart1"],
  ["--chart-2", "chart2"],
  ["--chart-3", "chart3"],
  ["--chart-4", "chart4"],
  ["--chart-5", "chart5"],
  ["--chart-6", "chart6"],
  ["--chart-7", "chart7"],
  ["--chart-8", "chart8"],
  ["--kpi-best", "kpiBest"],
  ["--kpi-warning", "kpiWarning"],
  ["--kpi-ceiling", "kpiCeiling"],
  ["--bg-gradient-primary", "bgGradientPrimary"],
  ["--bg-gradient-secondary", "bgGradientSecondary"],
  ["--bg-gradient-base-start", "bgGradientBaseStart"],
  ["--bg-gradient-base-mid", "bgGradientBaseMid"],
  ["--bg-gradient-base-end", "bgGradientBaseEnd"],
  ["--success", "success"],
  ["--success-foreground", "successForeground"],
  ["--warning", "warning"],
  ["--warning-foreground", "warningForeground"],
  ["--info", "info"],
  ["--info-foreground", "infoForeground"],
  ["--severity-high", "severityHigh"],
  ["--severity-high-foreground", "severityHighForeground"],
] as const;
