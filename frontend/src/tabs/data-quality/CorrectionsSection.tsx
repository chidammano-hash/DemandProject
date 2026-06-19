/**
 * Section 7: DQ Corrections Browser.
 *
 * Summary table of corrected SKUs with domain/fix-type filters + pagination,
 * plus an expandable detail panel of correction records for a selected SKU.
 * Extracted verbatim from DataQualityTab.tsx to keep the tab under the 600-line cap.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchCorrectionsByItem,
  fetchCorrectionsSummary,
  correctionKeys,
  STALE_LINEAGE,
} from "@/api/queries";
import type { DQCorrection, DQCorrectionSummary } from "@/api/queries/platform";
import { formatDate } from "@/lib/formatters";
import { interactiveRowProps } from "@/lib/interactiveRow";
import { ClipboardList } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";

export function CorrectionsSection() {
  const [domainFilter, setDomainFilter] = useState("");
  const [fixTypeFilter, setFixTypeFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [selectedSku, setSelectedSku] = useState<{ item_id: string; loc: string } | null>(null);
  const PAGE = 50;

  // Summary query — all corrected SKUs
  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: correctionKeys.summary(domainFilter, fixTypeFilter),
    queryFn: () => fetchCorrectionsSummary(domainFilter || undefined, fixTypeFilter || undefined, PAGE, offset),
    staleTime: STALE_LINEAGE,
  });
  const skus: DQCorrectionSummary[] = summaryData?.skus ?? [];
  const totalSkus = summaryData?.total ?? 0;
  const totalPages = Math.ceil(totalSkus / PAGE);
  const currentPage = Math.floor(offset / PAGE) + 1;

  // Detail query — corrections for selected SKU
  const { data: detailData, isLoading: detailLoading } = useQuery({
    queryKey: selectedSku
      ? correctionKeys.byItem(selectedSku.item_id, selectedSku.loc)
      : ["dq", "corrections", "none"],
    queryFn: () =>
      selectedSku
        ? fetchCorrectionsByItem(selectedSku.item_id, selectedSku.loc, 1000)
        : Promise.resolve({ corrections: [], total: 0 }),
    staleTime: STALE_LINEAGE,
    enabled: !!selectedSku,
  });
  const detailRows: DQCorrection[] = detailData?.corrections ?? [];

  function handleSkuSelect(item_id: string, loc: string) {
    if (selectedSku?.item_id === item_id && selectedSku?.loc === loc) {
      setSelectedSku(null);
    } else {
      setSelectedSku({ item_id, loc });
    }
  }

  const formatNum = (v: number | null) =>
    v != null ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "NULL";

  return (
    <div className="space-y-4">
      {/* ---- Summary table ---- */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <h3 className="text-sm font-medium text-foreground">
            DQ Corrections — Corrected SKUs
            {totalSkus > 0 && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                ({totalSkus.toLocaleString()} SKUs)
              </span>
            )}
          </h3>
          <div className="flex items-center gap-2">
            <select
              className="h-7 rounded border border-input bg-background px-2 text-xs"
              value={domainFilter}
              onChange={(e) => { setDomainFilter(e.target.value); setOffset(0); setSelectedSku(null); }}
            >
              <option value="">All Domains</option>
              <option value="sales">Sales</option>
              <option value="inventory">Inventory</option>
              <option value="forecast">Forecast</option>
              <option value="purchase_order">Purchase Order</option>
            </select>
            <select
              className="h-7 rounded border border-input bg-background px-2 text-xs"
              value={fixTypeFilter}
              onChange={(e) => { setFixTypeFilter(e.target.value); setOffset(0); setSelectedSku(null); }}
            >
              <option value="">All Fix Types</option>
              <option value="outliers">Outliers (IQR/Z-score)</option>
              <option value="range">Range Clamp</option>
              <option value="completeness">Completeness</option>
              <option value="lead_time">Lead Time</option>
            </select>
          </div>
        </div>

        {summaryLoading ? (
          <p className="text-xs text-muted-foreground">Loading\u2026</p>
        ) : skus.length === 0 ? (
          (domainFilter || fixTypeFilter) ? (
            <EmptyState
              variant="filtered"
              title="No corrections match your filters"
              description="Try switching to All Domains or All Fix Types to see more history."
              onAction={() => { setDomainFilter(""); setFixTypeFilter(""); setOffset(0); }}
              actionLabel="Clear filters"
            />
          ) : (
            <EmptyState
              variant="no-data"
              icon={ClipboardList}
              title="No DQ corrections recorded"
              description='Corrections appear after the DQ auto-fixer runs against loaded data.'
              steps={[
                { label: "Apply auto-fixes", command: "uv run python scripts/fix_dq_issues.py --apply" },
              ]}
            />
          )
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 pr-3">Item</th>
                    <th className="pb-2 pr-3">Location</th>
                    <th className="pb-2 pr-3 text-center">Corrections</th>
                    <th className="pb-2 pr-3">Domain</th>
                    <th className="pb-2 pr-3">Columns</th>
                    <th className="pb-2 pr-3">Fix Types</th>
                    <th className="pb-2 pr-3">Period Range</th>
                    <th className="pb-2 pr-3">Last Applied</th>
                  </tr>
                </thead>
                <tbody>
                  {skus.map((s) => {
                    return (
                      <tr
                        key={`${s.item_id}-${s.loc}`}
                        {...interactiveRowProps(() => {
                          window.location.href = `?tab=itemAnalysis&item=${encodeURIComponent(s.item_id)}&loc=${encodeURIComponent(s.loc)}&dqCorrections=1`;
                        })}
                        aria-label={`View item analysis for ${s.item_id} at ${s.loc}`}
                        className="border-b border-border/30 cursor-pointer transition-colors hover:bg-primary/10"
                      >
                        <td className="py-1.5 pr-3 font-mono font-medium">{s.item_id}</td>
                        <td className="py-1.5 pr-3">{s.loc}</td>
                        <td className="py-1.5 pr-3 text-center">
                          <span className="inline-block min-w-[2rem] rounded-full bg-amber-100 px-2 py-0.5 text-center font-semibold text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">
                            {s.correction_count}
                          </span>
                        </td>
                        <td className="py-1.5 pr-3">{s.domains.join(", ")}</td>
                        <td className="py-1.5 pr-3">
                          {s.columns.map((col) => (
                            <span key={col} className="mr-1 rounded bg-muted px-1 py-0.5 text-foreground">{col}</span>
                          ))}
                        </td>
                        <td className="py-1.5 pr-3">
                          {s.fix_types.map((ft) => (
                            <span key={ft} className="mr-1 rounded bg-blue-50 px-1 py-0.5 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">{ft}</span>
                          ))}
                        </td>
                        <td className="py-1.5 pr-3 text-muted-foreground">
                          {s.earliest_period && s.latest_period
                            ? `${s.earliest_period.slice(0, 7)} \u2192 ${s.latest_period.slice(0, 7)}`
                            : "—"}
                        </td>
                        <td className="py-1.5 pr-3 text-muted-foreground">
                          {formatDate(s.latest_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center gap-2 mt-3 text-xs">
                <button
                  disabled={offset === 0}
                  onClick={() => { setOffset(Math.max(0, offset - PAGE)); setSelectedSku(null); }}
                  className="px-2 py-1 rounded border disabled:opacity-40"
                >
                  Prev
                </button>
                <span className="text-muted-foreground">
                  Page {currentPage} of {totalPages} · {totalSkus.toLocaleString()} SKUs
                </span>
                <button
                  disabled={currentPage >= totalPages}
                  onClick={() => { setOffset(offset + PAGE); setSelectedSku(null); }}
                  className="px-2 py-1 rounded border disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* ---- Detail panel — corrections for selected SKU ---- */}
      {selectedSku && (
        <div className="rounded-lg border-2 border-primary/30 bg-card p-4 animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-foreground">
              Corrections for{" "}
              <span className="font-mono font-semibold text-primary">{selectedSku.item_id}</span>
              {" @ "}
              <span className="font-mono">{selectedSku.loc}</span>
              {detailRows.length > 0 && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  ({detailRows.length} changes)
                </span>
              )}
            </h3>
            <div className="flex items-center gap-2">
              <a
                href={`?tab=itemAnalysis&item=${encodeURIComponent(selectedSku.item_id)}&loc=${encodeURIComponent(selectedSku.loc)}&dqCorrections=1`}
                className="rounded border px-2 py-1 text-xs text-muted-foreground hover:text-foreground hover:bg-muted"
                title="Open in Item Analysis"
              >
                View in Item Analysis
              </a>
              <button
                onClick={() => setSelectedSku(null)}
                className="rounded border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
              >
                Close
              </button>
            </div>
          </div>

          {detailLoading ? (
            <p className="text-xs text-muted-foreground">Loading corrections\u2026</p>
          ) : detailRows.length === 0 ? (
            <p className="text-xs text-muted-foreground">No correction records found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 pr-3">Table</th>
                    <th className="pb-2 pr-3">Column</th>
                    <th className="pb-2 pr-3">Period</th>
                    <th className="pb-2 pr-3 text-right">Old Value</th>
                    <th className="pb-2 pr-3 text-center">→</th>
                    <th className="pb-2 pr-3 text-right">New Value</th>
                    <th className="pb-2 pr-3 text-right">Change</th>
                    <th className="pb-2 pr-3">Fix Type</th>
                    <th className="pb-2 pr-3">Strategy</th>
                    <th className="pb-2 pr-3 text-right">Bounds</th>
                  </tr>
                </thead>
                <tbody>
                  {detailRows.map((c) => {
                    const pctChange =
                      c.old_value != null && c.old_value !== 0 && c.new_value != null
                        ? ((c.new_value - c.old_value) / Math.abs(c.old_value)) * 100
                        : null;
                    return (
                      <tr key={c.correction_id} className="border-b border-border/30 hover:bg-muted/30">
                        <td className="py-1.5 pr-3 font-mono text-muted-foreground">{c.table_name.replace("fact_", "").replace("dim_", "")}</td>
                        <td className="py-1.5 pr-3 font-medium">{c.column_name}</td>
                        <td className="py-1.5 pr-3 font-mono">{c.period?.slice(0, 7) ?? "—"}</td>
                        <td className="py-1.5 pr-3 text-right font-mono text-red-600 dark:text-red-400">
                          {formatNum(c.old_value)}
                        </td>
                        <td className="py-1.5 pr-3 text-center text-muted-foreground">→</td>
                        <td className="py-1.5 pr-3 text-right font-mono text-emerald-600 dark:text-emerald-400">
                          {formatNum(c.new_value)}
                        </td>
                        <td className={`py-1.5 pr-3 text-right font-mono ${
                          pctChange != null && pctChange < 0
                            ? "text-red-500"
                            : "text-emerald-500"
                        }`}>
                          {pctChange != null ? `${pctChange > 0 ? "+" : ""}${pctChange.toFixed(1)}%` : "—"}
                        </td>
                        <td className="py-1.5 pr-3">
                          <span className="rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">{c.fix_type}</span>
                        </td>
                        <td className="py-1.5 pr-3 text-muted-foreground">{c.fix_strategy ?? "—"}</td>
                        <td className="py-1.5 pr-3 text-right text-muted-foreground font-mono">
                          {c.lower_bound != null && c.upper_bound != null
                            ? `[${formatNum(c.lower_bound)}, ${formatNum(c.upper_bound)}]`
                            : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
