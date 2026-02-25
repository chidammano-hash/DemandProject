import type { MotifId, MotifThemeConfig } from "@/types/motif";

const registry = new Map<MotifId, MotifThemeConfig>();

export function registerMotif(config: MotifThemeConfig): void {
  if (registry.has(config.id)) {
    console.warn(`[MotifRegistry] Overwriting motif: ${config.id}`);
  }
  registry.set(config.id, config);
}

export function getMotif(id: MotifId): MotifThemeConfig {
  const motif = registry.get(id);
  if (!motif) throw new Error(`[MotifRegistry] Unknown motif id: "${id}"`);
  return motif;
}

export function getAllMotifs(): MotifThemeConfig[] {
  return Array.from(registry.values());
}

export const DEFAULT_MOTIF_ID: MotifId = "periodic";
