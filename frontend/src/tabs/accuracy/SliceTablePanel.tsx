import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
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
import { formatHeatmapAccuracy } from "@/tabs/aggregate-analysis/aggregateShared";

// Virtualization geometry. The slice table grows to brand x category x cluster
// buckets at scale; rendering every <tr> locked the main thread. We virtualize
// the body with the spacer-row technique (top/bottom filler <tr>s + only the
// in-view rows) so the sticky <thead>, the sticky-left bucket column, and table
// column alignment all keep working — a plain absolute-positioned grid would
// lose column auto-sizing across the dynamic model x KPI columns.
const ROW_HEIGHT = 41; // matches the default shadcn table row height
const VIEWPORT_HEIGHT = 400; // was the table's max-h-[400px]

// ---------------------------------------------------------------------------
// Cell formatting
// ---------------------------------------------------------------------------

/**
 * F4.4 — render one accuracy-slice cell. The `accuracy_pct` column floors
 * negative low-base values to `<0%*` (consistent with the accuracy heatmap,
 * F3.2) so a planner reads "low base — see WAPE" instead of an apparent bug.
 * WAPE (also `pct` format) is left raw because a high WAPE is meaningful;
 * bias keeps its signed value; numeric columns fall through to a count.
 */
export function formatSliceCell(
  key: string,
  format: string,
  val: number | null | undefined,
): string {
  if (val === null || val === undefined) return "-";
  if (key === "accuracy_pct") return formatHeatmapAccuracy(val);
  if (format === "pct") return formatPercent(val);
  if (format === "bias") return `${(val * 100).toFixed(1)}%`;
  return Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

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
// Virtualized table body
// ---------------------------------------------------------------------------

/** One bucket row — the model x KPI cells plus the sticky-left bucket label. */
function SliceRow({
  row,
  allModels,
  sliceKpis,
}: {
  row: AccuracySliceRow;
  allModels: string[];
  sliceKpis: string[];
}) {
  const accValues = allModels
    .filter((m) => m.toLowerCase() !== "ceiling")
    .map((m) => row.by_model[m]?.accuracy_pct)
    .filter((v): v is number => v !== null && v !== undefined);
  const bestAcc = accValues.length > 0 ? Math.max(...accValues) : null;
  return (
    <TableRow className="hover:bg-muted/30" style={{ height: ROW_HEIGHT }}>
      <TableCell className="sticky left-0 bg-background font-medium text-sm">
        {row.bucket}
      </TableCell>
      {allModels.flatMap((m) => {
        const kpi = row.by_model[m];
        return ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => {
          const val = kpi?.[k.key as keyof AccuracyKpis] as number | null | undefined;
          const isBestAcc =
            k.key === "accuracy_pct" && val !== null && val !== undefined && val === bestAcc;
          const isBadBias =
            k.key === "bias" && val !== null && val !== undefined && Math.abs(val) > 0.15;
          const display = formatSliceCell(k.key, k.format, val);
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
}

/**
 * Virtualized comparison table. Only the in-viewport bucket rows mount; top and
 * bottom spacer <tr>s reserve the scroll height so the native <table> keeps its
 * column auto-sizing, the sticky <thead>, and the sticky-left bucket column. The
 * column count (one per KPI), not the row count, sets the spacer colSpan.
 */
function VirtualizedSliceTable({
  sliceData,
  allModels,
  sliceKpis,
  sliceGroupBy,
}: {
  sliceData: AccuracySliceRow[];
  allModels: string[];
  sliceKpis: string[];
  sliceGroupBy: string;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: sliceData.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });
  const virtualRows = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();
  const paddingTop = virtualRows.length > 0 ? virtualRows[0].start : 0;
  const paddingBottom =
    virtualRows.length > 0 ? totalSize - virtualRows[virtualRows.length - 1].end : 0;
  // +1 for the sticky bucket column.
  const colSpan = allModels.length * sliceKpis.length + 1;

  return (
    <div
      ref={parentRef}
      className="overflow-auto rounded-md border border-input"
      style={{ maxHeight: VIEWPORT_HEIGHT }}
    >
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
          {paddingTop > 0 && (
            <tr aria-hidden="true">
              <td colSpan={colSpan} style={{ height: paddingTop, padding: 0, border: 0 }} />
            </tr>
          )}
          {virtualRows.map((vRow) => {
            const row = sliceData[vRow.index];
            return (
              <SliceRow
                key={row.bucket}
                row={row}
                allModels={allModels}
                sliceKpis={sliceKpis}
              />
            );
          })}
          {paddingBottom > 0 && (
            <tr aria-hidden="true">
              <td colSpan={colSpan} style={{ height: paddingBottom, padding: 0, border: 0 }} />
            </tr>
          )}
        </TableBody>
      </Table>
    </div>
  );
}

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
  truncated: boolean;
  sliceLimit: number;
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
  truncated,
  sliceLimit,
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
            placeholder="e.g. lgbm_cluster,nhits"
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
          {truncated ? (
            <p className="text-xs font-medium text-amber-600 dark:text-amber-400">
              Showing top {sliceLimit.toLocaleString()} {sliceGroupBy.replace(/_/g, " ")} bucket(s) by
              volume (truncated). Narrow the filters to see lower-volume segments.
            </p>
          ) : null}
          <VirtualizedSliceTable
            sliceData={sliceData}
            allModels={allModels}
            sliceKpis={sliceKpis}
            sliceGroupBy={sliceGroupBy}
          />
          <p className="text-xs text-muted-foreground">
            &#9733; = best accuracy for that row. &#9888; = |bias| &gt; 15%.
            Accuracy = 100 &minus; WAPE; <code className="rounded bg-muted px-1">&lt;0%*</code> = tiny actual base (forecast &gt;&gt; actual) &mdash; read WAPE instead.
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
