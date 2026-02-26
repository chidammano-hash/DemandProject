import { useCallback, useEffect, useState } from "react";
import type { ColorMode, MotifId, MotifThemeConfig, TileConfig } from "@/types/motif";
import { getMotif, getAllMotifs, DEFAULT_MOTIF_ID } from "@/constants/motifs";
import { getInitialMotif, updateMotifUrl } from "@/hooks/useUrlState";

const STORAGE_KEY = "ds-motif";

function readStoredMotif(): MotifId {
  // URL param takes priority over localStorage (enables deep-linking via ?motif=space)
  const urlMotif = getInitialMotif() as MotifId | null;
  if (urlMotif) {
    try {
      getMotif(urlMotif);
      return urlMotif;
    } catch {
      // ignore unknown motif in URL param
    }
  }
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

export interface UseMotifThemeReturn {
  motifId: MotifId;
  motifConfig: MotifThemeConfig;
  setMotif: (id: MotifId) => void;
  cycleMotif: () => void;
  getTile: (tabKey: string) => TileConfig;
}

export function useMotifTheme(_colorMode?: ColorMode): UseMotifThemeReturn {
  const [motifId, setMotifId] = useState<MotifId>(readStoredMotif);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, motifId);
    document.documentElement.setAttribute("data-motif", motifId);
    updateMotifUrl(motifId);
  }, [motifId]);

  // CSS palette variables are now exclusively managed by useTheme (Feature 36).
  // This hook only manages motif identity, tiles, and data-motif attribute.

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
