// ---------------------------------------------------------------------------
// Supply series definitions + persistent default-measure preferences
// (localStorage) for the Item Analysis unified chart.
// Relocated verbatim from UnifiedChartPanel.tsx.
// ---------------------------------------------------------------------------
import type { ColorMode } from "@/constants/palette";
import { getItemAnalysisColors } from "./colors";

export interface SupplySeriesDef {
  key: string;
  label: string;
  color: string;
  axis: "left" | "right";
  defaultVisible: boolean;
  dashArray?: string;
  strokeWidth?: number;
}

/** Mode-independent series shape; colors are resolved per mode below. */
const SUPPLY_SERIES_BASE: Array<Omit<SupplySeriesDef, "color">> = [
  { key: "total_on_hand", label: "On Hand", axis: "left", defaultVisible: true },
  { key: "total_on_order", label: "On Order", axis: "left", defaultVisible: false },
  { key: "total_position", label: "Position", axis: "left", defaultVisible: false, dashArray: "8 3" },
  { key: "inv_monthly_sales", label: "Inv Sales", axis: "left", defaultVisible: false },
  { key: "dos", label: "DOS", axis: "right", defaultVisible: true, strokeWidth: 2.5 },
  { key: "avg_lead_time", label: "Lead Time", axis: "right", defaultVisible: false, dashArray: "5 3" },
  { key: "safety_stock", label: "Safety Stock", axis: "left", defaultVisible: false, dashArray: "6 3" },
  { key: "cycle_stock", label: "Cycle Stock", axis: "left", defaultVisible: false },
];

/** Supply series defs with mode-correct palette colors. */
export function getSupplySeriesDefs(mode: ColorMode): SupplySeriesDef[] {
  const { supply } = getItemAnalysisColors(mode);
  return SUPPLY_SERIES_BASE.map((def) => ({ ...def, color: supply[def.key] }));
}

export const DEFAULT_HIDDEN_SUPPLY = new Set(
  SUPPLY_SERIES_BASE.filter((s) => !s.defaultVisible).map((s) => s.key),
);

// ---------------------------------------------------------------------------
// Persistent default-measure preferences (localStorage)
// ---------------------------------------------------------------------------
const LS_KEY_DEMAND = "ds:itemAnalysis:defaultMeasures";
const LS_KEY_SUPPLY = "ds:itemAnalysis:defaultSupply";

/** All static demand measure keys (sales-side). */
const SALES_MEASURE_KEYS = ["tothist_dmd", "sales_qty", "qty_shipped", "qty_ordered"] as const;

/** Load saved demand defaults from localStorage. */
export function loadDefaultMeasures(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_KEY_DEMAND);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch { /* ignore */ }
  return new Set<string>(SALES_MEASURE_KEYS);
}

/** Load saved supply hidden-set from localStorage. */
export function loadDefaultHiddenSupply(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_KEY_SUPPLY);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch { /* ignore */ }
  return new Set(DEFAULT_HIDDEN_SUPPLY);
}

export function saveDemandDefaults(keys: Set<string>) {
  localStorage.setItem(LS_KEY_DEMAND, JSON.stringify([...keys]));
}

export function saveSupplyDefaults(hiddenKeys: Set<string>) {
  localStorage.setItem(LS_KEY_SUPPLY, JSON.stringify([...hiddenKeys]));
}

/** Return a new Set with `key` toggled (added if absent, removed if present). */
export function toggleInSet<T>(prev: Set<T>, key: T): Set<T> {
  const next = new Set(prev);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

/**
 * Build the visible-series set for a new SKU load, using saved defaults.
 * Forecast model keys are always included; sales measures follow user prefs.
 */
export function buildInitialVisibleSeries(
  models: string[],
): Set<string> {
  const defaults = loadDefaultMeasures();
  const keys = new Set<string>();
  for (const k of SALES_MEASURE_KEYS) {
    if (defaults.has(k)) keys.add(k);
  }
  for (const m of models) keys.add(`forecast_${m}`);
  if (defaults.has("production_forecast")) keys.add("production_forecast");
  // The saved AI Champion line is shown by default when present (gated at render
  // by hasAiChampion); it has no separate user-default preference.
  keys.add("ai_champion");
  return keys;
}
