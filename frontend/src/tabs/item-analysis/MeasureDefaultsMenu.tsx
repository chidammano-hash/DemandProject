import { memo, useState, useCallback, useRef, useEffect } from "react";

import { SKU_SALES_COLORS, skuModelColor } from "@/constants/colors";
import { modelLabel } from "@/lib/model-labels";
import { PROD_FORECAST_COLOR, STAGING_COLORS, STAGING_FALLBACK_COLOR } from "./colors";
import {
  type SupplySeriesDef,
  loadDefaultMeasures,
  loadDefaultHiddenSupply,
  saveDemandDefaults,
  saveSupplyDefaults,
  toggleInSet,
} from "./measures";

// ---------------------------------------------------------------------------
// MeasureDefaultsMenu — gear icon dropdown to set default visible measures
// ---------------------------------------------------------------------------
export const MeasureDefaultsMenu = memo(function MeasureDefaultsMenu({
  models,
  hasProdForecast,
  prodForecastLabel,
  hasSupplyData,
  availableSupply,
  stagingModelIds,
  hiddenStagingPills,
  setSkuVisibleSeries,
  setHiddenSupply,
  setHiddenStagingPills,
}: {
  models: string[];
  hasProdForecast: boolean;
  prodForecastLabel: string;
  hasSupplyData: boolean;
  availableSupply: SupplySeriesDef[];
  stagingModelIds: string[];
  hiddenStagingPills: Set<string>;
  setSkuVisibleSeries: (updater: (prev: Set<string>) => Set<string>) => void;
  setHiddenSupply: React.Dispatch<React.SetStateAction<Set<string>>>;
  setHiddenStagingPills: React.Dispatch<React.SetStateAction<Set<string>>>;
}) {
  const [open, setOpen] = useState(false);
  const [demandDefaults, setDemandDefaults] = useState<Set<string>>(loadDefaultMeasures);
  const [supplyHidden, setSupplyHidden] = useState<Set<string>>(loadDefaultHiddenSupply);
  const menuRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number }>({ top: 0, right: 0 });

  // Compute fixed position from button rect when opening
  useEffect(() => {
    if (open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        btnRef.current && !btnRef.current.contains(e.target as Node)
      ) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const toggleDemand = useCallback((key: string) => {
    setDemandDefaults((prev) => {
      const next = toggleInSet(prev, key);
      saveDemandDefaults(next);
      return next;
    });
    // Update live chart state immediately
    setSkuVisibleSeries((prev) => toggleInSet(prev, key));
  }, [setSkuVisibleSeries]);

  const toggleSupply = useCallback((key: string) => {
    setSupplyHidden((prev) => {
      const next = toggleInSet(prev, key);
      saveSupplyDefaults(next);
      return next;
    });
    // Update live chart state immediately
    setHiddenSupply((prev) => toggleInSet(prev, key));
  }, [setHiddenSupply]);

  const toggleStaging = useCallback((mid: string) => {
    setHiddenStagingPills((prev) => toggleInSet(prev, mid));
  }, [setHiddenStagingPills]);

  const demandItems: { key: string; label: string; color: string }[] = [
    { key: "tothist_dmd", label: "Sale Qty (ext)", color: SKU_SALES_COLORS.tothist_dmd },
    { key: "sales_qty", label: "Sale Qty", color: SKU_SALES_COLORS.sales_qty },
    { key: "qty_shipped", label: "Shipped", color: SKU_SALES_COLORS.qty_shipped },
    { key: "qty_ordered", label: "Ordered", color: SKU_SALES_COLORS.qty_ordered },
    ...models.map((m, i) => ({ key: `forecast_${m}`, label: m, color: skuModelColor(m, i) })),
    ...(hasProdForecast ? [{ key: "production_forecast", label: prodForecastLabel, color: PROD_FORECAST_COLOR }] : []),
  ];

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen((v) => !v)}
        className={`flex h-6 items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-medium transition-colors ${
          open
            ? "border-primary text-primary"
            : "border-input text-muted-foreground hover:text-foreground hover:border-foreground"
        }`}
        title="Configure which measures are visible by default when loading a new DFU"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        Defaults
      </button>
      {open && (
        <div
          ref={menuRef}
          className="fixed z-[9999] min-w-[200px] overflow-y-auto rounded-md border bg-white p-2.5 shadow-lg dark:bg-zinc-900"
          style={{ top: pos.top, right: pos.right, maxHeight: `calc(100vh - ${pos.top}px - 16px)` }}
        >
          {/* Demand section */}
          <DefaultsSectionHeader label="Demand" withDivider={false} />
          {demandItems.map(({ key, label, color }) => (
            <DefaultCheckboxRow
              key={key}
              label={label}
              color={color}
              checked={demandDefaults.has(key)}
              onChange={() => toggleDemand(key)}
            />
          ))}

          {/* Staging section */}
          {stagingModelIds.length > 0 && (
            <>
              <DefaultsSectionHeader label="Staging" />
              {stagingModelIds.map((mid) => {
                const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                return (
                  <DefaultCheckboxRow
                    key={`staging_${mid}`}
                    label={modelLabel(mid)}
                    color={color}
                    checked={!hiddenStagingPills.has(mid)}
                    onChange={() => toggleStaging(mid)}
                  />
                );
              })}
            </>
          )}

          {/* Supply section */}
          {hasSupplyData && availableSupply.length > 0 && (
            <>
              <DefaultsSectionHeader label="Supply" />
              {availableSupply.map((s) => (
                <DefaultCheckboxRow
                  key={s.key}
                  label={s.label}
                  color={s.color}
                  checked={!supplyHidden.has(s.key)}
                  onChange={() => toggleSupply(s.key)}
                />
              ))}
            </>
          )}
        </div>
      )}
    </>
  );
});

// ---------------------------------------------------------------------------
// Defaults menu row — checkbox + color dot + label (shared by Demand/Staging/Supply)
// ---------------------------------------------------------------------------
function DefaultsSectionHeader({ label, withDivider = true }: { label: string; withDivider?: boolean }) {
  return (
    <>
      {withDivider && <div className="my-1.5 border-t border-border" />}
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
    </>
  );
}

function DefaultCheckboxRow({
  label,
  color,
  checked,
  onChange,
}: {
  label: string;
  color: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] hover:bg-muted">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-3 w-3 rounded border-muted-foreground accent-current"
        style={{ accentColor: color }}
      />
      <span className="inline-block h-0.5 w-3 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </label>
  );
}
