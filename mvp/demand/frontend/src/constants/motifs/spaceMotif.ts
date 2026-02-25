import { registerMotif } from "@/constants/motifRegistry";
import type { MotifThemeConfig, TileConfig, MotifPalette, ColorMode } from "@/types/motif";

// ---------------------------------------------------------------------------
// Deep Space motif -- "Stellarium"
//
// NASA Mission Control theme.  Each tab maps to a celestial body or space
// artifact, rendered as emblem-style tiles on a slate-900 background with
// distinct accent-color glows evoking deep-space instrumentation displays.
// ---------------------------------------------------------------------------

const tile = (
  primary: string,
  superscript: string | number,
  label: string,
  restClasses: string,
  activeClasses: string,
  glowClass: string,
  tagline: string,
): TileConfig => ({
  variant: "emblem",
  primary,
  superscript,
  label,
  restClasses,
  activeClasses,
  glowClass,
  tagline,
});

const tiles: Record<string, TileConfig> = {
  explorer: tile(
    "\u2642", "STA-1", "Mars",
    "bg-slate-900/90 text-sky-300 border-sky-800/40",
    "bg-slate-800 text-sky-100 border-sky-500",
    "shadow-[0_0_16px_rgba(56,189,248,0.35)]",
    "Every dataset is unexplored terrain",
  ),
  clusters: tile(
    "\u2726\u2726", "NGC-2", "Pleiades",
    "bg-slate-900/90 text-emerald-300 border-emerald-800/40",
    "bg-slate-800 text-emerald-100 border-emerald-500",
    "shadow-[0_0_16px_rgba(52,211,153,0.35)]",
    "Gravity finds what logic misses",
  ),
  dfuAnalysis: tile(
    "\uD83D\uDCE1", "TRK-7", "Europa",
    "bg-slate-900/90 text-teal-300 border-teal-800/40",
    "bg-slate-800 text-teal-100 border-teal-500",
    "shadow-[0_0_16px_rgba(20,184,166,0.35)]",
    "The signal lives beneath the surface",
  ),
  accuracy: tile(
    "\u2644", "NAV-5", "Saturn",
    "bg-slate-900/90 text-purple-300 border-purple-800/40",
    "bg-slate-800 text-purple-100 border-purple-500",
    "shadow-[0_0_16px_rgba(168,85,247,0.35)]",
    "Perfect rings require perfect math",
  ),
  intel: tile(
    "\uD83D\uDEF0\uFE0F", "SAT-6", "Voyager",
    "bg-slate-900/90 text-cyan-300 border-cyan-800/40",
    "bg-slate-800 text-cyan-100 border-cyan-500",
    "shadow-[0_0_16px_rgba(6,182,212,0.35)]",
    "Signal from 23.9 billion km out",
  ),
  item: tile(
    "\u2B50", "STR-26", "Star",
    "bg-slate-900/90 text-yellow-300 border-yellow-800/40",
    "bg-slate-800 text-yellow-100 border-yellow-500",
    "shadow-[0_0_16px_rgba(234,179,8,0.35)]",
    "Individual brilliance",
  ),
  location: tile(
    "\uD83E\uDE90", "PLT-71", "Planet",
    "bg-slate-900/90 text-orange-300 border-orange-800/40",
    "bg-slate-800 text-orange-100 border-orange-500",
    "shadow-[0_0_16px_rgba(249,115,22,0.35)]",
    "Orbital position",
  ),
  customer: tile(
    "\uD83D\uDC68\u200D\uD83D\uDE80", "CRW-29", "Astronaut",
    "bg-slate-900/90 text-amber-300 border-amber-800/40",
    "bg-slate-800 text-amber-100 border-amber-500",
    "shadow-[0_0_16px_rgba(245,158,11,0.35)]",
    "Mission crew",
  ),
  time: tile(
    "\u23F1\uFE0F", "EPH-22", "Chronometer",
    "bg-slate-900/90 text-rose-300 border-rose-800/40",
    "bg-slate-800 text-rose-100 border-rose-500",
    "shadow-[0_0_16px_rgba(244,63,94,0.35)]",
    "Ephemeris time",
  ),
  dfu: tile(
    "\uD83C\uDF20", "DFU-110", "Shooting Star",
    "bg-slate-900/90 text-lime-300 border-lime-800/40",
    "bg-slate-800 text-lime-100 border-lime-500",
    "shadow-[0_0_16px_rgba(132,204,22,0.35)]",
    "Demand trajectory",
  ),
  sales: tile(
    "\uD83D\uDE80", "MSN-3", "Rocket",
    "bg-slate-900/90 text-sky-300 border-sky-800/40",
    "bg-slate-800 text-sky-100 border-sky-500",
    "shadow-[0_0_16px_rgba(56,189,248,0.35)]",
    "Launch velocity",
  ),
  forecast: tile(
    "\uD83C\uDF19", "PRD-4", "Moon",
    "bg-slate-900/90 text-indigo-300 border-indigo-800/40",
    "bg-slate-800 text-indigo-100 border-indigo-500",
    "shadow-[0_0_16px_rgba(99,102,241,0.35)]",
    "Predictive orbit",
  ),
};

// ---------------------------------------------------------------------------
// Full UI palette — cool navy/black with electric cyan and nebula accents
// ---------------------------------------------------------------------------
const palette: Partial<Record<ColorMode, Partial<MotifPalette>>> = {
  light: {
    background: "215 25% 95%",
    foreground: "220 40% 15%",
    card: "215 20% 98%",
    cardForeground: "220 40% 15%",
    primary: "195 80% 38%",
    primaryForeground: "0 0% 100%",
    secondary: "270 40% 92%",
    secondaryForeground: "270 35% 30%",
    muted: "215 18% 92%",
    mutedForeground: "220 15% 40%",
    accent: "280 55% 52%",
    accentForeground: "0 0% 100%",
    border: "215 16% 82%",
    input: "215 16% 82%",
    ring: "195 80% 38%",
    destructive: "0 72% 51%",
    destructiveForeground: "0 0% 100%",
    chart1: "#0284c7",
    chart2: "#7c3aed",
    chart3: "#e11d48",
    chart4: "#ca8a04",
    chart5: "#059669",
    chart6: "#ea580c",
    kpiBest: "#0284c7",
    kpiWarning: "#dc2626",
    kpiCeiling: "#059669",
    bgGradientPrimary: "rgba(56,189,248,0.10)",
    bgGradientSecondary: "rgba(139,92,246,0.08)",
    bgGradientBaseStart: "#edf2f8",
    bgGradientBaseMid: "#eef0f6",
    bgGradientBaseEnd: "#ededf5",
  },
  dark: {
    background: "225 30% 7%",
    foreground: "210 25% 88%",
    card: "225 25% 11%",
    cardForeground: "210 25% 88%",
    primary: "195 85% 55%",
    primaryForeground: "225 30% 7%",
    secondary: "270 40% 22%",
    secondaryForeground: "270 30% 82%",
    muted: "225 20% 15%",
    mutedForeground: "220 15% 55%",
    accent: "280 60% 55%",
    accentForeground: "0 0% 100%",
    border: "225 20% 18%",
    input: "225 20% 18%",
    ring: "195 85% 55%",
    destructive: "0 62% 55%",
    destructiveForeground: "0 0% 100%",
    chart1: "#38bdf8",
    chart2: "#a78bfa",
    chart3: "#fb7185",
    chart4: "#facc15",
    chart5: "#4ade80",
    chart6: "#f97316",
    kpiBest: "#67e8f9",
    kpiWarning: "#fda4af",
    kpiCeiling: "#86efac",
    bgGradientPrimary: "rgba(56,189,248,0.06)",
    bgGradientSecondary: "rgba(139,92,246,0.04)",
    bgGradientBaseStart: "#0a0e1a",
    bgGradientBaseMid: "#0c1220",
    bgGradientBaseEnd: "#0e1425",
  },
  midnight: {
    background: "230 35% 6%",
    foreground: "210 20% 86%",
    card: "230 30% 9%",
    cardForeground: "210 20% 86%",
    primary: "195 80% 52%",
    primaryForeground: "230 35% 6%",
    secondary: "275 45% 20%",
    secondaryForeground: "275 25% 80%",
    muted: "230 22% 12%",
    mutedForeground: "225 12% 50%",
    accent: "285 55% 52%",
    accentForeground: "0 0% 100%",
    border: "230 22% 15%",
    input: "230 22% 15%",
    ring: "195 80% 52%",
    destructive: "0 55% 60%",
    destructiveForeground: "0 0% 100%",
    chart1: "#7dd3fc",
    chart2: "#c4b5fd",
    chart3: "#fda4af",
    chart4: "#fde68a",
    chart5: "#86efac",
    chart6: "#fdba74",
    kpiBest: "#a5f3fc",
    kpiWarning: "#fda4af",
    kpiCeiling: "#a7f3d0",
    bgGradientPrimary: "rgba(56,189,248,0.05)",
    bgGradientSecondary: "rgba(139,92,246,0.03)",
    bgGradientBaseStart: "#080c18",
    bgGradientBaseMid: "#0a0f1e",
    bgGradientBaseEnd: "#0c1222",
  },
};

const spaceMotif: MotifThemeConfig = {
  id: "space",
  displayName: "Deep Space",
  description:
    "NASA Mission Control-inspired emblem tiles with celestial bodies, deep slate backgrounds, and instrument-glow accents.",
  previewTile: tiles.explorer,
  tiles,
  loading: {
    animationName: "orbit-spin",
    wrapperClasses: "rounded-full border-2 border-dashed border-sky-400/60",
    statusLabel: "Scanning",
  },
  chrome: {
    appName: "Stellarium",
    appTagline: "Deep Space Demand Navigation",
    logoSvgPath: null,
    bgOverlay:
      "radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.06) 0%, transparent 60%)",
    tileRadius: "rounded-xl",
  },
  palette,
};

registerMotif(spaceMotif);

export default spaceMotif;
