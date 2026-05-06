/**
 * SKU Features — paginated, sortable, filterable feature table.
 */
import { useCallback } from "react";
import { ChevronUp, ChevronDown, Search, Database } from "lucide-react";
import type { SkuFeatureRow } from "@/api/queries/sku-features";
import { Skeleton } from "@/components/Skeleton";
import {
  PAGE_SIZE,
  SEASONALITY_OPTIONS,
  VARIABILITY_OPTIONS,
  TREND_OPTIONS,
  SORTABLE_COLUMNS,
} from "./constants";
import { formatNumber, formatPct, trendLabel, badgeClass } from "./utils";

export interface FeatureTableState {
  search: string;
  seasonalityFilter: string;
  variabilityFilter: string;
  trendFilter: string;
  sortBy: string;
  sortDir: "asc" | "desc";
  page: number;
}

export interface FeatureTableHandlers {
  setSearch: (v: string) => void;
  setSeasonalityFilter: (v: string) => void;
  setVariabilityFilter: (v: string) => void;
  setTrendFilter: (v: string) => void;
  setSortBy: (v: string) => void;
  setSortDir: (v: "asc" | "desc") => void;
  setPage: (updater: (p: number) => number) => void;
}

interface FeatureTableProps {
  rows: SkuFeatureRow[];
  totalRows: number;
  isLoading: boolean;
  state: FeatureTableState;
  handlers: FeatureTableHandlers;
}

export function FeatureTable({ rows, totalRows, isLoading, state, handlers }: FeatureTableProps) {
  const {
    search,
    seasonalityFilter,
    variabilityFilter,
    trendFilter,
    sortBy,
    sortDir,
    page,
  } = state;
  const {
    setSearch,
    setSeasonalityFilter,
    setVariabilityFilter,
    setTrendFilter,
    setSortBy,
    setSortDir,
    setPage,
  } = handlers;

  const totalPages = Math.ceil(totalRows / PAGE_SIZE);
  const currentPage = page + 1;

  const handleSort = useCallback(
    (col: string) => {
      if (sortBy === col) {
        setSortDir(sortDir === "asc" ? "desc" : "asc");
      } else {
        setSortBy(col);
        setSortDir("asc");
      }
      setPage(() => 0);
    },
    [sortBy, sortDir, setSortBy, setSortDir, setPage],
  );

  const handleFilterChange = useCallback(
    (setter: (v: string) => void) => (e: React.ChangeEvent<HTMLSelectElement>) => {
      setter(e.target.value);
      setPage(() => 0);
    },
    [setPage],
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearch(e.target.value);
      setPage(() => 0);
    },
    [setSearch, setPage],
  );

  return (
    <div className="rounded-lg border border-border bg-card">
      {/* Table header: filters */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-3">
        <h3 className="text-sm font-medium text-foreground mr-auto">
          Feature Table
          {totalRows > 0 && (
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              ({totalRows.toLocaleString()} SKUs)
            </span>
          )}
        </h3>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search item_id..."
            value={search}
            onChange={handleSearchChange}
            className="h-8 w-44 rounded-md border border-input bg-background pl-8 pr-2.5 text-xs"
          />
        </div>
        <select
          value={seasonalityFilter}
          onChange={handleFilterChange(setSeasonalityFilter)}
          className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
        >
          <option value="">All Seasonality</option>
          {SEASONALITY_OPTIONS.filter(Boolean).map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <select
          value={variabilityFilter}
          onChange={handleFilterChange(setVariabilityFilter)}
          className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
        >
          <option value="">All Variability</option>
          {VARIABILITY_OPTIONS.filter(Boolean).map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <select
          value={trendFilter}
          onChange={handleFilterChange(setTrendFilter)}
          className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
        >
          <option value="">All Trends</option>
          {TREND_OPTIONS.filter(Boolean).map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {/* Table body */}
      <div className="overflow-x-auto">
        {isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-full" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
            <Database className="h-10 w-10 text-muted-foreground/30 mb-3" />
            <p className="text-sm font-medium">No SKU features found</p>
            <p className="text-xs mt-1 max-w-xs text-center">
              {search || seasonalityFilter || variabilityFilter || trendFilter
                ? "Try adjusting your filters or search query."
                : "Run the feature computation pipeline to populate SKU features."}
            </p>
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {SORTABLE_COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    className={`cursor-pointer select-none px-3 py-2.5 font-medium text-muted-foreground hover:text-foreground transition-colors ${
                      col.align === "right" ? "text-right" : "text-left"
                    }`}
                    onClick={() => handleSort(col.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.label}
                      {sortBy === col.key && (
                        sortDir === "asc"
                          ? <ChevronUp className="h-3 w-3" />
                          : <ChevronDown className="h-3 w-3" />
                      )}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.sku_ck}
                  className="border-b border-border/30 transition-colors hover:bg-muted/20"
                >
                  <td className="px-3 py-2 font-mono text-xs font-medium">{row.sku_ck}</td>
                  <td className="px-3 py-2 font-mono">{row.item_id}</td>
                  <td className="px-3 py-2">{row.loc}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.ml_cluster ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    {row.seasonality_profile ? (
                      <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(row.seasonality_profile, "seasonality")}`}>
                        {row.seasonality_profile}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {row.variability_class ? (
                      <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(row.variability_class, "variability")}`}>
                        {row.variability_class}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {row.trend_direction != null ? (
                      <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(trendLabel(row.trend_direction), "trend")}`}>
                        {trendLabel(row.trend_direction)}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.cv_demand)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.seasonal_amplitude)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{formatPct(row.zero_demand_pct)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{formatPct(row.cagr)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.recency_ratio)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-border px-4 py-2.5">
          <span className="text-xs text-muted-foreground">
            Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, totalRows)} of{" "}
            {totalRows.toLocaleString()}
          </span>
          <div className="flex items-center gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              className="rounded border border-input px-3 py-1 text-xs disabled:opacity-40 hover:bg-muted transition-colors"
            >
              Previous
            </button>
            <span className="text-xs text-muted-foreground tabular-nums">
              Page {currentPage} of {totalPages}
            </span>
            <button
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => p + 1)}
              className="rounded border border-input px-3 py-1 text-xs disabled:opacity-40 hover:bg-muted transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
