import { registerMotif } from "@/constants/motifRegistry";
import type { MotifThemeConfig, TileConfig } from "@/types/motif";

// ---------------------------------------------------------------------------
// Periodic Table motif -- translates the legacy ELEMENT_CONFIG into the
// MotifThemeConfig shape.  Every tile uses the "periodic" variant so the
// renderer draws the classic element-card layout (superscript atomic number,
// large center symbol, bottom label).
// ---------------------------------------------------------------------------

const tile = (
  primary: string,
  superscript: number,
  label: string,
  restClasses: string,
  activeClasses: string,
  glowClass: string,
): TileConfig => ({
  variant: "periodic",
  primary,
  superscript,
  label,
  restClasses,
  activeClasses,
  glowClass,
});

const tiles: Record<string, TileConfig> = {
  explorer: tile(
    "Dx", 1, "Explorer",
    "bg-pink-50/90 text-pink-800 border-pink-200/60",
    "bg-pink-100 text-pink-950 border-pink-300",
    "shadow-[0_0_12px_rgba(236,72,153,0.3)]",
  ),
  item: tile(
    "It", 26, "Item",
    "bg-pink-50/90 text-pink-800 border-pink-200/60",
    "bg-pink-100 text-pink-950 border-pink-300",
    "shadow-[0_0_12px_rgba(236,72,153,0.3)]",
  ),
  location: tile(
    "Lo", 71, "Location",
    "bg-pink-50/90 text-pink-800 border-pink-200/60",
    "bg-pink-100 text-pink-950 border-pink-300",
    "shadow-[0_0_12px_rgba(236,72,153,0.3)]",
  ),
  customer: tile(
    "Cu", 29, "Customer",
    "bg-amber-50/90 text-amber-800 border-amber-200/60",
    "bg-amber-100 text-amber-950 border-amber-300",
    "shadow-[0_0_12px_rgba(245,158,11,0.3)]",
  ),
  time: tile(
    "Ti", 22, "Time",
    "bg-amber-50/90 text-amber-800 border-amber-200/60",
    "bg-amber-100 text-amber-950 border-amber-300",
    "shadow-[0_0_12px_rgba(245,158,11,0.3)]",
  ),
  dfu: tile(
    "Df", 110, "DFU",
    "bg-lime-50/90 text-lime-800 border-lime-200/60",
    "bg-lime-100 text-lime-950 border-lime-300",
    "shadow-[0_0_12px_rgba(132,204,22,0.3)]",
  ),
  clusters: tile(
    "Cl", 2, "Clusters",
    "bg-emerald-50/90 text-emerald-800 border-emerald-200/60",
    "bg-emerald-100 text-emerald-950 border-emerald-300",
    "shadow-[0_0_12px_rgba(16,185,129,0.3)]",
  ),
  sales: tile(
    "Sa", 3, "Sales",
    "bg-sky-50/90 text-sky-800 border-sky-200/60",
    "bg-sky-100 text-sky-950 border-sky-300",
    "shadow-[0_0_12px_rgba(14,165,233,0.3)]",
  ),
  forecast: tile(
    "Fc", 4, "Forecast",
    "bg-indigo-50/90 text-indigo-800 border-indigo-200/60",
    "bg-indigo-100 text-indigo-950 border-indigo-300",
    "shadow-[0_0_12px_rgba(99,102,241,0.3)]",
  ),
  dfuAnalysis: tile(
    "Da", 7, "DFU Analysis",
    "bg-teal-50/90 text-teal-800 border-teal-200/60",
    "bg-teal-100 text-teal-950 border-teal-300",
    "shadow-[0_0_12px_rgba(20,184,166,0.3)]",
  ),
  accuracy: tile(
    "Ac", 5, "Accuracy",
    "bg-purple-50/90 text-purple-800 border-purple-200/60",
    "bg-purple-100 text-purple-950 border-purple-300",
    "shadow-[0_0_12px_rgba(168,85,247,0.3)]",
  ),
  intel: tile(
    "Mi", 6, "Intel",
    "bg-cyan-50/90 text-cyan-800 border-cyan-200/60",
    "bg-cyan-100 text-cyan-950 border-cyan-300",
    "shadow-[0_0_12px_rgba(6,182,212,0.3)]",
  ),
};

const periodicMotif: MotifThemeConfig = {
  id: "periodic",
  displayName: "Periodic Table",
  description:
    "Chemistry-inspired element tiles with atomic numbers, color-coded by domain group.",
  previewTile: tiles.explorer,
  tiles,
  loading: {
    animationName: "pulse-glow",
    wrapperClasses: "rounded-lg",
    statusLabel: "Loading",
  },
  chrome: {
    appName: "Planthium",
    appTagline: "Periodic Analytics for Demand Forecasting",
    logoSvgPath: null,
    bgOverlay: "none",
    tileRadius: "rounded-xl",
  },
};

registerMotif(periodicMotif);

export default periodicMotif;
