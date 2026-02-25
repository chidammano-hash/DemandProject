import { registerMotif } from "@/constants/motifRegistry";
import type { MotifThemeConfig, TileConfig, MotifPalette, ColorMode } from "@/types/motif";

// ---------------------------------------------------------------------------
// Spirits & Wine motif -- "The Cellar"
//
// Each tab maps to a spirit or fine wine, with warm/dark Tailwind palettes
// evoking oak barrels, candlelit cellars, and aged bottles.  All tiles use
// the "card" variant for a refined tasting-card layout.
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
  variant: "card",
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
    "BRN", 86, "Bourbon",
    "bg-amber-950/90 text-amber-200 border-amber-800/60",
    "bg-amber-900 text-amber-100 border-amber-600",
    "shadow-[0_0_12px_rgba(217,119,6,0.4)]",
    "Discovering new territory, one barrel at a time",
  ),
  accuracy: tile(
    "CMG", 2019, "Champagne",
    "bg-yellow-950/90 text-yellow-200 border-yellow-800/60",
    "bg-yellow-900 text-yellow-100 border-yellow-600",
    "shadow-[0_0_12px_rgba(234,179,8,0.4)]",
    "Every bubble, precisely placed",
  ),
  dfuAnalysis: tile(
    "SCT", "18yr", "Islay Scotch",
    "bg-stone-900/90 text-stone-300 border-stone-700/60",
    "bg-stone-800 text-stone-100 border-stone-500",
    "shadow-[0_0_12px_rgba(168,162,158,0.35)]",
    "Complexity rewarded by patience",
  ),
  clusters: tile(
    "FLT", "5prs", "Wine Flight",
    "bg-rose-950/90 text-rose-200 border-rose-800/60",
    "bg-rose-900 text-rose-100 border-rose-600",
    "shadow-[0_0_12px_rgba(225,29,72,0.35)]",
    "Grouped by character, distinguished by nuance",
  ),
  intel: tile(
    "GIN", 44, "Hendrick's Gin",
    "bg-emerald-950/90 text-emerald-200 border-emerald-800/60",
    "bg-emerald-900 text-emerald-100 border-emerald-600",
    "shadow-[0_0_12px_rgba(16,185,129,0.4)]",
    "Botanical clarity in a complex world",
  ),
  item: tile(
    "MLT", "12yr", "Single Malt",
    "bg-orange-950/90 text-orange-200 border-orange-800/60",
    "bg-orange-900 text-orange-100 border-orange-600",
    "shadow-[0_0_12px_rgba(234,88,12,0.35)]",
    "Each expression unique",
  ),
  location: tile(
    "PRT", "40yr", "Port Wine",
    "bg-red-950/90 text-red-200 border-red-800/60",
    "bg-red-900 text-red-100 border-red-600",
    "shadow-[0_0_12px_rgba(185,28,28,0.4)]",
    "Terroir defines character",
  ),
  customer: tile(
    "CGN", "VSOP", "Cognac",
    "bg-amber-950/90 text-amber-300 border-amber-700/60",
    "bg-amber-800 text-amber-100 border-amber-500",
    "shadow-[0_0_12px_rgba(180,83,9,0.4)]",
    "The connoisseur's measure",
  ),
  time: tile(
    "RUM", "15yr", "Aged Rum",
    "bg-yellow-950/90 text-yellow-300 border-yellow-700/60",
    "bg-yellow-900 text-yellow-100 border-yellow-500",
    "shadow-[0_0_12px_rgba(161,98,7,0.4)]",
    "Time is the secret ingredient",
  ),
  dfu: tile(
    "AMR", 32, "Amaro",
    "bg-zinc-900/90 text-zinc-300 border-zinc-700/60",
    "bg-zinc-800 text-zinc-100 border-zinc-500",
    "shadow-[0_0_12px_rgba(161,161,170,0.3)]",
    "Bitter complexity, sweet insight",
  ),
  sales: tile(
    "TQA", "Añejo", "Tequila",
    "bg-lime-950/90 text-lime-200 border-lime-800/60",
    "bg-lime-900 text-lime-100 border-lime-600",
    "shadow-[0_0_12px_rgba(132,204,22,0.35)]",
    "Clarity from agave fields",
  ),
  forecast: tile(
    "VRM", "Dry", "Vermouth",
    "bg-violet-950/90 text-violet-200 border-violet-800/60",
    "bg-violet-900 text-violet-100 border-violet-600",
    "shadow-[0_0_12px_rgba(139,92,246,0.35)]",
    "The blend that completes the picture",
  ),
};

// ---------------------------------------------------------------------------
// Full UI palette — warm amber/oak/candlelit cellar
// ---------------------------------------------------------------------------
const palette: Partial<Record<ColorMode, Partial<MotifPalette>>> = {
  light: {
    background: "35 30% 95%",
    foreground: "25 40% 15%",
    card: "35 25% 98%",
    cardForeground: "25 40% 15%",
    primary: "30 75% 35%",
    primaryForeground: "0 0% 100%",
    secondary: "350 50% 92%",
    secondaryForeground: "350 40% 25%",
    muted: "30 20% 92%",
    mutedForeground: "25 20% 40%",
    accent: "38 80% 52%",
    accentForeground: "25 40% 15%",
    border: "30 18% 82%",
    input: "30 18% 82%",
    ring: "30 75% 35%",
    destructive: "0 72% 51%",
    destructiveForeground: "0 0% 100%",
    chart1: "#b45309",
    chart2: "#9f1239",
    chart3: "#d97706",
    chart4: "#854d0e",
    chart5: "#be185d",
    chart6: "#92400e",
    kpiBest: "#b45309",
    kpiWarning: "#dc2626",
    kpiCeiling: "#059669",
    bgGradientPrimary: "rgba(217,119,6,0.12)",
    bgGradientSecondary: "rgba(159,18,57,0.08)",
    bgGradientBaseStart: "#f5f0e8",
    bgGradientBaseMid: "#f2ece0",
    bgGradientBaseEnd: "#efe8da",
  },
  dark: {
    background: "25 20% 9%",
    foreground: "35 20% 88%",
    card: "25 18% 13%",
    cardForeground: "35 20% 88%",
    primary: "38 85% 55%",
    primaryForeground: "25 20% 9%",
    secondary: "350 40% 20%",
    secondaryForeground: "350 30% 85%",
    muted: "25 15% 17%",
    mutedForeground: "30 15% 55%",
    accent: "25 70% 50%",
    accentForeground: "0 0% 100%",
    border: "25 15% 22%",
    input: "25 15% 22%",
    ring: "38 85% 55%",
    destructive: "0 62% 55%",
    destructiveForeground: "0 0% 100%",
    chart1: "#f59e0b",
    chart2: "#e11d48",
    chart3: "#eab308",
    chart4: "#d97706",
    chart5: "#f472b6",
    chart6: "#a16207",
    kpiBest: "#f59e0b",
    kpiWarning: "#f87171",
    kpiCeiling: "#34d399",
    bgGradientPrimary: "rgba(217,119,6,0.08)",
    bgGradientSecondary: "rgba(159,18,57,0.06)",
    bgGradientBaseStart: "#1a1410",
    bgGradientBaseMid: "#1c1612",
    bgGradientBaseEnd: "#1e1814",
  },
  midnight: {
    background: "25 30% 7%",
    foreground: "35 20% 85%",
    card: "25 25% 10%",
    cardForeground: "35 20% 85%",
    primary: "38 80% 50%",
    primaryForeground: "25 30% 7%",
    secondary: "350 45% 18%",
    secondaryForeground: "350 25% 82%",
    muted: "25 20% 13%",
    mutedForeground: "30 12% 50%",
    accent: "30 65% 48%",
    accentForeground: "0 0% 100%",
    border: "25 20% 18%",
    input: "25 20% 18%",
    ring: "38 80% 50%",
    destructive: "0 55% 60%",
    destructiveForeground: "0 0% 100%",
    chart1: "#fbbf24",
    chart2: "#fb7185",
    chart3: "#fde68a",
    chart4: "#f59e0b",
    chart5: "#f9a8d4",
    chart6: "#d97706",
    kpiBest: "#fbbf24",
    kpiWarning: "#fda4af",
    kpiCeiling: "#6ee7b7",
    bgGradientPrimary: "rgba(217,119,6,0.06)",
    bgGradientSecondary: "rgba(159,18,57,0.04)",
    bgGradientBaseStart: "#141008",
    bgGradientBaseMid: "#16120a",
    bgGradientBaseEnd: "#18140c",
  },
};

const spiritsMotif: MotifThemeConfig = {
  id: "spirits",
  displayName: "The Cellar",
  description:
    "Spirits & wine-inspired tasting cards with warm oak tones, aged labels, and candlelit cellar ambiance.",
  previewTile: tiles.explorer,
  tiles,
  loading: {
    animationName: "pour-shimmer",
    wrapperClasses: "rounded-lg",
    statusLabel: "Pouring",
  },
  chrome: {
    appName: "The Cellar",
    appTagline: "Curated Analytics, Aged to Perfection",
    logoSvgPath: null,
    bgOverlay:
      "linear-gradient(180deg, rgba(44,24,16,0.08) 0%, transparent 50%)",
    tileRadius: "rounded-lg",
  },
  palette,
};

registerMotif(spiritsMotif);

export default spiritsMotif;
