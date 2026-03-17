import { RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

import { titleCase } from "@/lib/formatters";
import type { DfuAnalysisPayload } from "@/types";
import type { InventoryKpis } from "@/types";
import type { InventoryTrendParams } from "@/types";

// ---------------------------------------------------------------------------
// Criticality color helpers
// ---------------------------------------------------------------------------
function dosColor(v: number | null | undefined): string {
  if (v == null) return "";
  if (v >= 14 && v <= 60) return "text-green-600 dark:text-green-400";
  if (v < 7 || v > 90) return "text-red-600 dark:text-red-400";
  return "text-yellow-600 dark:text-yellow-400";
}

function wocColor(v: number | null | undefined): string {
  if (v == null) return "";
  if (v >= 2 && v <= 8) return "text-green-600 dark:text-green-400";
  if (v < 1 || v > 12) return "text-red-600 dark:text-red-400";
  return "text-yellow-600 dark:text-yellow-400";
}

function turnsColor(v: number | null | undefined): string {
  if (v == null) return "";
  if (v > 8) return "text-green-600 dark:text-green-400";
  if (v < 4) return "text-red-600 dark:text-red-400";
  return "text-yellow-600 dark:text-yellow-400";
}

function ltCoverageColor(v: number | null | undefined): string {
  if (v == null) return "";
  if (v > 1.5) return "text-green-600 dark:text-green-400";
  if (v < 1.0) return "text-red-600 dark:text-red-400";
  return "text-yellow-600 dark:text-yellow-400";
}

function onHandColor(dos: number | null | undefined): string {
  return dosColor(dos);
}

function leadTimeColor(v: number | null | undefined): string {
  if (v == null) return "";
  if (v <= 14) return "text-green-600 dark:text-green-400";
  if (v > 45) return "text-red-600 dark:text-red-400";
  return "text-yellow-600 dark:text-yellow-400";
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface SelectorPanelProps {
  dfuItem: string;
  setDfuItem: (v: string) => void;
  dfuLocation: string;
  setDfuLocation: (v: string) => void;
  dfuPoints: number;
  setDfuPoints: (v: number) => void;
  dfuItemSuggestions: string[];
  dfuLocationSuggestions: string[];
  dfuData: DfuAnalysisPayload | null;
  onReset: () => void;
  kpiData?: InventoryKpis | null;
  trendParams?: InventoryTrendParams | null;
  positionRow?: { snapshot_date: string; qty_on_hand: number; qty_on_order: number; lead_time_days: number | null; mtd_sales: number } | null;
  ltProfile?: { lt_mean_days: number | null; lt_std_days: number | null; lt_cv: number | null; lt_variability_class: string | null; observation_count: number | null } | null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function SelectorPanel({
  dfuItem,
  setDfuItem,
  dfuLocation,
  setDfuLocation,
  dfuPoints,
  setDfuPoints,
  dfuItemSuggestions,
  dfuLocationSuggestions,
  dfuData,
  onReset,
  kpiData,
  trendParams,
  positionRow,
  ltProfile,
}: SelectorPanelProps) {
  const fmt = (v: number | null | undefined) =>
    v != null ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "\u2014";
  const fmtCompact = (v: number | null | undefined) => {
    if (v == null) return "\u2014";
    if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
    if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
    return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  };

  return (
    <CardHeader className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <CardTitle className="text-base">Item Analysis</CardTitle>
          <CardDescription>
            Sales, forecasts, inventory, and supply chain attributes for a single DFU (item @ location)
          </CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={onReset}>
          <RefreshCcw className="mr-1 h-4 w-4" /> Reset
        </Button>
      </div>

      {/* Row 1: Item + Location + Points */}
      <div className="grid gap-2 md:grid-cols-3">
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Item (dmdunit)
          <Input
            className="h-9"
            placeholder="Type to search items..."
            list="dfu-analysis-item-suggest"
            value={dfuItem}
            onChange={(e) => setDfuItem(e.target.value)}
          />
          <datalist id="dfu-analysis-item-suggest">
            {dfuItemSuggestions.map((val) => (
              <option key={val} value={val} />
            ))}
          </datalist>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Location (loc)
          <Input
            className="h-9"
            placeholder="Type to search locations..."
            list="dfu-analysis-loc-suggest"
            value={dfuLocation}
            onChange={(e) => setDfuLocation(e.target.value)}
          />
          <datalist id="dfu-analysis-loc-suggest">
            {dfuLocationSuggestions.map((val) => (
              <option key={val} value={val} />
            ))}
          </datalist>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Points
          <select
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            value={dfuPoints}
            onChange={(e) => setDfuPoints(Number(e.target.value))}
          >
            {[12, 24, 36, 48, 60].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>

      {/* DFU Attributes (collapsible) — includes inventory KPIs, lead-time grouped */}
      {dfuData &&
        dfuData.dfu_attributes &&
        dfuData.dfu_attributes.length > 0 && (
          <details className="group rounded-md border border-input bg-background">
            <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground">
              DFU Attributes ({dfuData.dfu_attributes.length}{" "}
              {dfuData.dfu_attributes.length === 1 ? "record" : "records"})
              <span className="ml-1 text-xs text-muted-foreground group-open:hidden">
                + expand
              </span>
            </summary>
            <div className="border-t border-input px-3 py-2 space-y-3">
              {dfuData.dfu_attributes.map((attrs, dfuIdx) => {
                const allAttrs = Object.entries(attrs);

                return (
                  <div key={dfuIdx}>
                    {dfuData.dfu_attributes.length > 1 && (
                      <p className="mb-1 text-xs font-medium text-foreground">
                        {attrs.dmdunit} / {attrs.dmdgroup} @ {attrs.loc}
                      </p>
                    )}
                    <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                      {allAttrs.map(([key, val]) => (
                        <div
                          key={key}
                          className="flex items-baseline gap-1 text-xs truncate"
                        >
                          <span className="font-medium text-muted-foreground shrink-0">
                            {titleCase(key)}:
                          </span>
                          <span
                            className="text-foreground truncate"
                            title={val ?? "\u2014"}
                          >
                            {val ?? "\u2014"}
                          </span>
                        </div>
                      ))}
                      {/* Inventory KPIs inline in same grid */}
                      {kpiData && (
                        <>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">On-Hand:</span>
                            <span className={`font-semibold ${onHandColor(kpiData.dos)}`}>{fmtCompact(kpiData.total_on_hand)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">On-Order:</span>
                            <span className="text-foreground">{fmtCompact(kpiData.total_on_order)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Avg Lead Time:</span>
                            <span className={`font-semibold ${leadTimeColor(kpiData.avg_lead_time_days)}`}>
                              {kpiData.avg_lead_time_days != null ? `${fmt(kpiData.avg_lead_time_days)}d` : "\u2014"}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Days of Supply:</span>
                            <span className={`font-semibold ${dosColor(kpiData.dos)}`}>{fmt(kpiData.dos)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Weeks of Cover:</span>
                            <span className={`font-semibold ${wocColor(kpiData.woc)}`}>{fmt(kpiData.woc)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Inv Turns/yr:</span>
                            <span className={`font-semibold ${turnsColor(kpiData.inventory_turns)}`}>{fmt(kpiData.inventory_turns)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">LT Coverage:</span>
                            <span className={`font-semibold ${ltCoverageColor(kpiData.lt_coverage)}`}>
                              {kpiData.lt_coverage != null ? `${fmt(kpiData.lt_coverage)}x` : "\u2014"}
                            </span>
                          </div>
                        </>
                      )}
                      {/* Inventory Parameters inline */}
                      {trendParams && (trendParams.safety_stock != null || trendParams.eoq != null || trendParams.order_policy != null) && (
                        <>
                          {trendParams.order_policy != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">Policy:</span>
                              <span className="text-foreground">{trendParams.order_policy} ({trendParams.policy_type?.replace(/_/g, " ")})</span>
                            </div>
                          )}
                          {trendParams.service_level_target != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">SL:</span>
                              <span className="text-foreground">{(trendParams.service_level_target * 100).toFixed(0)}% (Z={trendParams.z_score?.toFixed(2)})</span>
                            </div>
                          )}
                          {trendParams.safety_stock != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">Safety Stock:</span>
                              <span className="text-foreground">{trendParams.safety_stock.toLocaleString()}u</span>
                            </div>
                          )}
                          {trendParams.reorder_point_units != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">ROP:</span>
                              <span className="text-foreground">{trendParams.reorder_point_units.toLocaleString()}u</span>
                            </div>
                          )}
                          {trendParams.eoq != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">EOQ:</span>
                              <span className="text-foreground">{trendParams.eoq.toLocaleString()}u</span>
                            </div>
                          )}
                          {trendParams.demand_cv != null && (
                            <div className="flex items-baseline gap-1 text-xs">
                              <span className="font-medium text-muted-foreground shrink-0">CV:</span>
                              <span className="text-foreground">
                                {trendParams.demand_cv.toFixed(3)} ({trendParams.demand_cv < 0.3 ? "stable" : trendParams.demand_cv < 0.8 ? "moderate" : "volatile"})
                              </span>
                            </div>
                          )}
                        </>
                      )}
                      {/* Position snapshot values inline */}
                      {positionRow && (
                        <>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Snapshot:</span>
                            <span className="text-foreground">{positionRow.snapshot_date}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Qty On Hand:</span>
                            <span className={`font-semibold ${onHandColor(kpiData?.dos)}`}>{fmtCompact(positionRow.qty_on_hand)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Qty On Order:</span>
                            <span className="text-foreground">{fmtCompact(positionRow.qty_on_order)}</span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">Lead Time:</span>
                            <span className={`font-semibold ${leadTimeColor(positionRow.lead_time_days)}`}>
                              {positionRow.lead_time_days != null ? `${positionRow.lead_time_days}d` : "\u2014"}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">MTD Sales:</span>
                            <span className="text-foreground">{fmtCompact(positionRow.mtd_sales)}</span>
                          </div>
                        </>
                      )}
                      {/* Lead Time Profile inline */}
                      {ltProfile && (
                        <>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">LT Mean:</span>
                            <span className={`font-semibold ${leadTimeColor(ltProfile.lt_mean_days)}`}>
                              {ltProfile.lt_mean_days != null ? `${Number(ltProfile.lt_mean_days).toFixed(1)}d` : "\u2014"}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">LT Std:</span>
                            <span className="text-foreground">
                              {ltProfile.lt_std_days != null ? `${Number(ltProfile.lt_std_days).toFixed(1)}d` : "\u2014"}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">LT CV:</span>
                            <span className="text-foreground">
                              {ltProfile.lt_cv != null ? Number(ltProfile.lt_cv).toFixed(3) : "\u2014"}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-1 text-xs">
                            <span className="font-medium text-muted-foreground shrink-0">LT Class:</span>
                            <span className={`font-semibold capitalize ${
                              ltProfile.lt_variability_class === "stable" ? "text-green-600 dark:text-green-400" :
                              ltProfile.lt_variability_class === "volatile" ? "text-red-600 dark:text-red-400" :
                              "text-yellow-600 dark:text-yellow-400"
                            }`}>
                              {ltProfile.lt_variability_class ?? "\u2014"}
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </details>
        )}
    </CardHeader>
  );
}
