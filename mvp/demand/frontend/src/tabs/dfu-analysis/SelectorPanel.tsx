import { RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

import { titleCase } from "@/lib/formatters";
import type { DfuAnalysisMode, DfuAnalysisPayload } from "@/types";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface SelectorPanelProps {
  dfuMode: DfuAnalysisMode;
  setDfuMode: (mode: DfuAnalysisMode) => void;
  dfuItem: string;
  setDfuItem: (v: string) => void;
  dfuLocation: string;
  setDfuLocation: (v: string) => void;
  dfuPoints: number;
  setDfuPoints: (v: number) => void;
  dfuKpiMonths: number;
  setDfuKpiMonths: (v: number) => void;
  dfuItemSuggestions: string[];
  dfuLocationSuggestions: string[];
  seasonalityProfile: string;
  setSeasonalityProfile: (v: string) => void;
  seasonalityProfiles: string[];
  dfuData: DfuAnalysisPayload | null;
  onReset: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function SelectorPanel({
  dfuMode,
  setDfuMode,
  dfuItem,
  setDfuItem,
  dfuLocation,
  setDfuLocation,
  dfuPoints,
  setDfuPoints,
  dfuKpiMonths,
  setDfuKpiMonths,
  dfuItemSuggestions,
  dfuLocationSuggestions,
  seasonalityProfile,
  setSeasonalityProfile,
  seasonalityProfiles,
  dfuData,
  onReset,
}: SelectorPanelProps) {
  return (
    <CardHeader className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <CardTitle className="text-base">DFU Analysis</CardTitle>
          <CardDescription>
            {dfuMode === "item_location"
              ? "Sales + multi-model forecasts for a specific DFU (item @ location)"
              : dfuMode === "all_items_at_location"
                ? "Aggregated sales + forecasts across all items at a location"
                : "Aggregated sales + forecasts for an item across all locations"}
          </CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={onReset}>
          <RefreshCcw className="mr-1 h-4 w-4" /> Reset
        </Button>
      </div>

      {/* Row 1: Analysis scope selector */}
      <div className="grid gap-2 md:grid-cols-3">
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Analysis Scope
          <select
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            value={dfuMode}
            onChange={(e) => setDfuMode(e.target.value as DfuAnalysisMode)}
          >
            <option value="item_location">Item @ Location (single DFU)</option>
            <option value="all_items_at_location">All Items @ Location</option>
            <option value="item_at_all_locations">Item @ All Locations</option>
          </select>
        </label>
        <div className="grid grid-cols-2 gap-2">
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Points
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={dfuPoints}
              onChange={(e) => setDfuPoints(Number(e.target.value))}
            >
              {[12, 24, 36, 48, 60].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            KPI Window
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={dfuKpiMonths}
              onChange={(e) => setDfuKpiMonths(Number(e.target.value))}
            >
              {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
                <option key={m} value={m}>
                  {m} mo
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* Row 2: Item + Location filters */}
      <div className="grid gap-2 md:grid-cols-2">
        {dfuMode !== "all_items_at_location" ? (
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
        ) : (
          <div className="flex items-end">
            <p className="pb-2 text-xs text-muted-foreground italic">
              Item: All (aggregated at location level)
            </p>
          </div>
        )}
        {dfuMode !== "item_at_all_locations" ? (
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
        ) : (
          <div className="flex items-end">
            <p className="pb-2 text-xs text-muted-foreground italic">
              Location: All (aggregated at item level)
            </p>
          </div>
        )}
      </div>

      {/* Row 3: Seasonality Profile filter (Feature 32) */}
      {seasonalityProfiles.length > 0 ? (
        <div className="grid gap-2 md:grid-cols-3">
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Seasonality Profile
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={seasonalityProfile}
              onChange={(e) => setSeasonalityProfile(e.target.value)}
            >
              <option value="">All Profiles</option>
              {seasonalityProfiles.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}

      {/* DFU Attributes (collapsible) */}
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
              {dfuData.dfu_attributes.map((attrs, dfuIdx) => (
                <div key={dfuIdx}>
                  {dfuData.dfu_attributes.length > 1 && (
                    <p className="mb-1 text-xs font-medium text-foreground">
                      {attrs.dmdunit} / {attrs.dmdgroup} @ {attrs.loc}
                    </p>
                  )}
                  <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                    {Object.entries(attrs).map(([key, val]) => (
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
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
    </CardHeader>
  );
}
