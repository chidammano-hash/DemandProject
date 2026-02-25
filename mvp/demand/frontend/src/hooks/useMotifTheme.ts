import { useCallback, useEffect, useState } from "react";
import type { ColorMode, MotifId, MotifPalette, MotifThemeConfig, TileConfig } from "@/types/motif";
import { getMotif, getAllMotifs, DEFAULT_MOTIF_ID } from "@/constants/motifs";

const STORAGE_KEY = "ds-motif";

/** Map MotifPalette keys → CSS custom property names */
const PALETTE_CSS_MAP: Record<keyof MotifPalette, string> = {
  background: "--background",
  foreground: "--foreground",
  card: "--card",
  cardForeground: "--card-foreground",
  primary: "--primary",
  primaryForeground: "--primary-foreground",
  secondary: "--secondary",
  secondaryForeground: "--secondary-foreground",
  muted: "--muted",
  mutedForeground: "--muted-foreground",
  accent: "--accent",
  accentForeground: "--accent-foreground",
  border: "--border",
  input: "--input",
  ring: "--ring",
  destructive: "--destructive",
  destructiveForeground: "--destructive-foreground",
  chart1: "--chart-1",
  chart2: "--chart-2",
  chart3: "--chart-3",
  chart4: "--chart-4",
  chart5: "--chart-5",
  chart6: "--chart-6",
  kpiBest: "--kpi-best",
  kpiWarning: "--kpi-warning",
  kpiCeiling: "--kpi-ceiling",
  bgGradientPrimary: "--bg-gradient-primary",
  bgGradientSecondary: "--bg-gradient-secondary",
  bgGradientBaseStart: "--bg-gradient-base-start",
  bgGradientBaseMid: "--bg-gradient-base-mid",
  bgGradientBaseEnd: "--bg-gradient-base-end",
};

const ALL_CSS_VARS = Object.values(PALETTE_CSS_MAP);

function readStoredMotif(): MotifId {
  try {
    const stored = localStorage.getItem(STORAGE_KEY) as MotifId | null;
    if (stored) {
      getMotif(stored); // throws if unregistered
      return stored;
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
  return DEFAULT_MOTIF_ID;
}

/** Remove all motif palette overrides from <html> inline styles */
function clearPalette() {
  const style = document.documentElement.style;
  for (const cssVar of ALL_CSS_VARS) {
    style.removeProperty(cssVar);
  }
}

/** Apply a partial palette as inline style overrides on <html> */
function applyPalette(palette: Partial<MotifPalette> | undefined) {
  clearPalette();
  if (!palette) return;
  const style = document.documentElement.style;
  for (const [key, cssVar] of Object.entries(PALETTE_CSS_MAP)) {
    const value = palette[key as keyof MotifPalette];
    if (value != null) {
      style.setProperty(cssVar, value);
    }
  }
}

export interface UseMotifThemeReturn {
  motifId: MotifId;
  motifConfig: MotifThemeConfig;
  setMotif: (id: MotifId) => void;
  cycleMotif: () => void;
  getTile: (tabKey: string) => TileConfig;
}

export function useMotifTheme(colorMode?: ColorMode): UseMotifThemeReturn {
  const [motifId, setMotifId] = useState<MotifId>(readStoredMotif);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, motifId);
    document.documentElement.setAttribute("data-motif", motifId);
  }, [motifId]);

  // Apply CSS variable palette whenever motif or color mode changes
  useEffect(() => {
    const config = getMotif(motifId);
    const mode = colorMode ?? "dark";
    const palette = config.palette?.[mode];
    applyPalette(palette);
    return () => clearPalette();
  }, [motifId, colorMode]);

  const setMotif = useCallback((id: MotifId) => {
    try {
      getMotif(id);
      setMotifId(id);
    } catch {
      console.warn(`[useMotifTheme] Ignored unknown motif id: "${id}"`);
    }
  }, []);

  const cycleMotif = useCallback(() => {
    const all = getAllMotifs();
    const idx = all.findIndex((m) => m.id === motifId);
    const next = all[(idx + 1) % all.length];
    setMotifId(next.id);
  }, [motifId]);

  const motifConfig = getMotif(motifId);

  const getTile = useCallback(
    (tabKey: string): TileConfig => motifConfig.tiles[tabKey] ?? motifConfig.previewTile,
    [motifConfig],
  );

  return { motifId, motifConfig, setMotif, cycleMotif, getTile };
}
