import type { AccuracyKpis, AccuracySliceRow } from "@/types";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";
import { LoadingElement } from "@/components/LoadingElement";
import { titleCase, formatPercent } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const ACCURACY_KPI_OPTIONS = [
  { key: "accuracy_pct", label: "Accuracy %", format: "pct" },
  { key: "wape",         label: "WAPE %",     format: "pct" },
  { key: "bias",         label: "Bias",        format: "bias" },
  { key: "sum_forecast", label: "\u03A3 Forecast", format: "num" },
  { key: "sum_actual",   label: "\u03A3 Actual",   format: "num" },
  { key: "sku_count",    label: "DFU Count",  format: "num" },
] as const;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface SliceTablePanelProps {
  sliceGroupBy: string;
  sliceLag: number;
  sliceModels: string;
  sliceKpis: string[];
  sliceMonths: number;
  commonDfus: boolean;
  seasonalityProfile: string;
  seasonalityProfiles: string[];
  loadingSlice: boolean;
  sliceData: AccuracySliceRow[];
  allModels: string[];
  commonDfuCount: number | null;
  skuCounts: Record<string, number> | null;
  onSliceGroupByChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onSliceLagChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onSliceModelsChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSliceMonthsChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onCommonDfusToggle: () => void;
  onKpiToggle: (key: string) => void;
  onSeasonalityProfileChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SliceTablePanel({
  sliceGroupBy,
  sliceLag,
  sliceModels,
  sliceKpis,
  sliceMonths,
  commonDfus,
  seasonalityProfile,
  seasonalityProfiles,
  loadingSlice,
  sliceData,
  allModels,
  commonDfuCount,
  skuCounts,
  onSliceGroupByChange,
  onSliceLagChange,
  onSliceModelsChange,
  onSliceMonthsChange,
  onCommonDfusToggle,
  onKpiToggle,
  onSeasonalityProfileChange,
}: SliceTablePanelProps) {
  return (
    <>
      {/* ── Filter controls ────────────────────────────────────── */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Slice by
          <select
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            value={sliceGroupBy}
            onChange={onSliceGroupByChange}
            disabled={loadingSlice}
          >
            <option value="cluster_assignment">Cluster (Business Label)</option>
            <option value="ml_cluster">Cluster (ML)</option>
            <option value="supplier_desc">Supplier</option>
            <option value="abc_vol">ABC Volume</option>
            <option value="region">Region</option>
            <option value="brand_desc">Brand</option>
            <option value="sku_execution_lag">Execution Lag</option>
            <option value="month_start">Month</option>
          </select>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Lag Filter
          <select
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            value={sliceLag}
            onChange={onSliceLagChange}
            disabled={loadingSlice}
          >
            <option value={-1}>Execution Lag (per DFU)</option>
            <option value={0}>Lag 0 (same month)</option>
            <option value={1}>Lag 1</option>
            <option value={2}>Lag 2</option>
            <option value={3}>Lag 3</option>
            <option value={4}>Lag 4</option>
          </select>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Models (comma-separated, blank = all)
          <input
            className="h-9 w-52 rounded-md border border-input bg-background px-3 text-sm"
            placeholder="e.g. lgbm_global,external"
            value={sliceModels}
            onChange={onSliceModelsChange}
            disabled={loadingSlice}
          />
        </label>
        {seasonalityProfiles.length > 0 ? (
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Seasonality Profile
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={seasonalityProfile}
              onChange={onSeasonalityProfileChange}
              disabled={loadingSlice}
            >
              <option value="">All Profiles</option>
              {seasonalityProfiles.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
        ) : null}
        {sliceGroupBy !== "month_start" ? (
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            KPI Window
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={sliceMonths}
              onChange={onSliceMonthsChange}
              disabled={loadingSlice}
            >
              {Array.from({ length: 12 }, (_, idx) => idx + 1).map((m) => (
                <option key={m} value={m}>
                  {m} month{m > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label className="flex items-center gap-1.5 self-end pb-1.5 cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 rounded border-input accent-blue-600"
            checked={commonDfus}
            onChange={onCommonDfusToggle}
            disabled={loadingSlice}
          />
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap">
            Common SKUs Only
          </span>
        </label>
        {commonDfus && commonDfuCount != null && skuCounts ? (
          <div className="flex items-center gap-2 self-end pb-1.5 text-xs text-muted-foreground tabular-nums">
            <Badge variant="secondary" className="font-mono text-xs">
              {commonDfuCount.toLocaleString()} common
            </Badge>
            {Object.entries(skuCounts).map(([m, cnt]) => (
              <span key={m} className="font-mono">
                {m}: {cnt.toLocaleString()}
              </span>
            ))}
          </div>
        ) : null}
        {loadingSlice ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Loading...</span>
          </div>
        ) : null}
      </div>

      {/* ── KPI checkboxes ─────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">KPIs</span>
        {ACCURACY_KPI_OPTIONS.map((opt) => {
          const checked = sliceKpis.includes(opt.key);
          const isLast = sliceKpis.length === 1 && checked;
          return (
            <label key={opt.key} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 rounded border-input accent-blue-600"
                checked={checked}
                disabled={isLast}
                onChange={() => onKpiToggle(opt.key)}
              />
              {opt.label}
            </label>
          );
        })}
      </div>

      {/* ── Model comparison table ─────────────────────────────── */}
      {sliceData.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Model Comparison &mdash; {sliceData.length} {sliceGroupBy.replace(/_/g, " ")} bucket(s)
          </p>
          <div className="max-h-[400px] overflow-auto rounded-md border border-input">
            <Table>
              <TableHeader className="sticky top-0 z-20 bg-background">
                <TableRow className="border-muted bg-muted">
                  <TableHead className="text-xs sticky left-0 z-30 bg-muted">
                    {titleCase(sliceGroupBy)}
                  </TableHead>
                  {allModels.flatMap((m) =>
                    ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => (
                      <TableHead key={`${m}-${k.key}`} className="text-xs text-right bg-muted">
                        {m} {k.label}
                      </TableHead>
                    )),
                  )}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sliceData.map((row) => {
                  const accValues = allModels
                    .filter((m) => m.toLowerCase() !== "ceiling")
                    .map((m) => row.by_model[m]?.accuracy_pct)
                    .filter((v): v is number => v !== null && v !== undefined);
                  const bestAcc = accValues.length > 0 ? Math.max(...accValues) : null;
                  return (
                    <TableRow key={row.bucket} className="hover:bg-muted/30">
                      <TableCell className="sticky left-0 bg-background font-medium text-sm">
                        {row.bucket}
                      </TableCell>
                      {allModels.flatMap((m) => {
                        const kpi = row.by_model[m];
                        return ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => {
                          const val = kpi?.[k.key as keyof AccuracyKpis] as number | null | undefined;
                          const isBestAcc =
                            k.key === "accuracy_pct" &&
                            val !== null &&
                            val !== undefined &&
                            val === bestAcc;
                          const isBadBias =
                            k.key === "bias" &&
                            val !== null &&
                            val !== undefined &&
                            Math.abs(val) > 0.15;
                          let display: string;
                          if (val === null || val === undefined) {
                            display = "-";
                          } else if (k.format === "pct") {
                            display = formatPercent(val);
                          } else if (k.format === "bias") {
                            display = `${(val * 100).toFixed(1)}%`;
                          } else {
                            display = Number(val).toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            });
                          }
                          return (
                            <TableCell
                              key={`${m}-${k.key}`}
                              className={cn(
                                "text-right text-sm tabular-nums",
                                isBestAcc ? "font-bold text-blue-700 dark:text-blue-400" : "",
                                isBadBias ? "text-red-600 dark:text-red-400" : "",
                              )}
                            >
                              {isBestAcc && (
                                <span className="mr-0.5" title="Best accuracy">
                                  &#9733;
                                </span>
                              )}
                              {isBadBias && (
                                <span className="mr-0.5" title="High bias (|bias| > 15%)">
                                  &#9888;
                                </span>
                              )}
                              {display}
                            </TableCell>
                          );
                        });
                      })}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
          <p className="text-xs text-muted-foreground">
            &#9733; = best accuracy for that row. &#9888; = |bias| &gt; 15%.
          </p>
        </div>
      ) : loadingSlice ? (
        <LoadingElement message="Loading accuracy data..." />
      ) : (
        <p className="text-sm text-muted-foreground">
          No data. Run <code className="rounded bg-muted px-1">make backtest-load</code> to populate the accuracy views.
        </p>
      )}
    </>
  );
}
