/**
 * Data Quality & Pipeline Observability Tab (Spec 08-01).
 *
 * Sections:
 *  0. Summary KPI bar — overall health, total checks, pass/fail/warn/error counts, last run
 *  1. Domain Health Grid — clickable domain cards that filter check catalog + issues
 *  2. Check Catalog Table — extracted to CheckCatalogPanel
 *  3. Recent Issues — extracted to RecentIssuesPanel
 *  4. Self-Heal — extracted to SelfHealPanel
 *  6. Pipeline Lineage
 *  7. Corrections Audit Log
 */
import { useState, useEffect, useMemo } from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  fetchDQDashboard,
  fetchDQChecks,
  fetchDQHistory,
  runDQChecks,
  dqKeys,
  STALE_PLATFORM,
  fetchBatches,
  fetchCorrectionsByItem,
  fetchCorrectionsSummary,
  correctionKeys,
  lineageKeys,
  STALE_LINEAGE,
} from "@/api/queries";
import type { DQDomainScore, DQCheck, LoadBatch, DQCorrection, DQCorrectionSummary } from "@/api/queries/platform";
import { formatDate } from "@/lib/formatters";
import { interactiveRowProps } from "@/lib/interactiveRow";
import {
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  MinusCircle,
  Shield,
  Clock,
  GitBranch,
  ClipboardList,
} from "lucide-react";
import { EmptyState } from "@/components/EmptyState";

import {
  CheckCatalogPanel,
  RecentIssuesPanel,
  SelfHealPanel,
  scoreBadgeClass,
  scoreRingColor,
  formatScore,
  relativeTime,
} from "./data-quality";

/* -------------------------------------------------------------------------- */
/*  Component                                                                 */
/* -------------------------------------------------------------------------- */

export default function DataQualityTab() {
  /* ---- data ---- */
  const { data: dashboard } = useQuery({
    queryKey: dqKeys.dashboard,
    queryFn: fetchDQDashboard,
    staleTime: STALE_PLATFORM,
  });

  const { data: checks } = useQuery({
    queryKey: dqKeys.checks,
    queryFn: fetchDQChecks,
    staleTime: STALE_PLATFORM,
  });

  const { data: history } = useQuery({
    queryKey: dqKeys.history(),
    queryFn: () => fetchDQHistory(undefined, 7, 200),
    staleTime: STALE_PLATFORM,
  });

  /* ---- mutation ---- */
  const queryClient = useQueryClient();
  const [showSuccess, setShowSuccess] = useState(false);

  const runChecksMutation = useMutation({
    mutationFn: runDQChecks,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dqKeys.dashboard });
      queryClient.invalidateQueries({ queryKey: dqKeys.checks });
      queryClient.invalidateQueries({ queryKey: ["dq", "history"] });
      setShowSuccess(true);
    },
  });

  useEffect(() => {
    if (!showSuccess) return;
    const timer = setTimeout(() => setShowSuccess(false), 5000);
    return () => clearTimeout(timer);
  }, [showSuccess]);

  /* ---- derived data ---- */
  const domains = dashboard?.domains ?? [];
  const checkList: DQCheck[] = checks?.checks ?? [];
  const historyEntries = history?.entries ?? [];

  /* ---- domain filter ---- */
  const [domainFilter, setDomainFilter] = useState<string | null>(null);

  /* summary KPIs */
  // U3.12 — Overall Health is a CHECK-WEIGHTED scored pass rate, not a
  // mean-of-domain-scores. A simple average let tiny 2-check 0% domains distort
  // the headline (mean-of-means dominated by the smallest domains) and disagree
  // with the Passed/Failed tiles beside it. We re-aggregate the same scored
  // numerator/denominator the per-domain cards use: scored fails exclude info-
  // and warning-severity fails (F3.1/U8.3) and skips, so this equals
  // Σpassed / Σ(passed + critical_fails + warnings).
  const scoredTotals = domains.reduce(
    (acc, d) => {
      const criticalFails = Math.max(0, d.failed - (d.info_fails ?? 0) - (d.warning_fails ?? 0));
      acc.passed += d.passed;
      acc.scored += d.passed + criticalFails + d.warnings;
      return acc;
    },
    { passed: 0, scored: 0 },
  );
  // Overall Health mirrors the per-domain rule: a number when there is a
  // pass-rate to grade, else null (nothing globally scoreable — all skip/info/
  // warn). Null renders a neutral "—", never a misleading green 100% (U4.2).
  const overallScore: number | null = !domains.length
    ? 0
    : scoredTotals.scored
      ? Math.round((100 * scoredTotals.passed) / scoredTotals.scored)
      : null;
  const totalChecks = domains.reduce((s, d) => s + d.total, 0);
  const totalPass = domains.reduce((s, d) => s + d.passed, 0);
  // F7.2 — the summary "Failed" tile must use the SAME severity-aware rule as
  // the per-domain red chip (cycles U8.3 / F3.1): only scored/critical fails are
  // red. Previously this summed the RAW `failed` field, so warning- and info-
  // severity fails (which every card correctly demotes) inflated the headline to
  // an alarm-red "Failed 26" over a grid of "0 fail" cards. Warning-severity
  // fails roll into the amber Warnings tile (they belong there conceptually);
  // info-severity fails surface in their own muted tile.
  const totalFail = domains.reduce(
    (s, d) => s + Math.max(0, d.failed - (d.info_fails ?? 0) - (d.warning_fails ?? 0)),
    0,
  );
  const totalWarn = domains.reduce((s, d) => s + d.warnings + (d.warning_fails ?? 0), 0);
  const totalInfo = domains.reduce((s, d) => s + (d.info_fails ?? 0), 0);
  const totalSkip = domains.reduce((s, d) => s + (d.skipped ?? 0), 0);
  const lastRun = useMemo(() => checkList.reduce<string | null>((latest, c) => {
    if (!c.last_run) return latest;
    if (!latest) return c.last_run;
    return c.last_run > latest ? c.last_run : latest;
  }, null), [checkList]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Data Quality & Observability</h2>
          <p className="text-sm text-muted-foreground">
            Monitor pipeline health and data quality across all domains
          </p>
        </div>
        <button
          onClick={() => runChecksMutation.mutate()}
          disabled={runChecksMutation.isPending}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${runChecksMutation.isPending ? "animate-spin" : ""}`} />
          {runChecksMutation.isPending ? "Running\u2026" : "Run Checks Now"}
        </button>
      </div>

      {/* Success / Error banners */}
      {showSuccess && (
        <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300">
          Data quality checks completed successfully.
        </div>
      )}
      {runChecksMutation.isError && (
        <div className="rounded-md bg-red-50 px-4 py-2 text-sm text-red-800 dark:bg-red-900/20 dark:text-red-300">
          {(runChecksMutation.error as Error)?.message ?? "Failed to run data quality checks"}
        </div>
      )}

      {/* ================================================================== */}
      {/* SECTION 0: Summary KPI Bar                                         */}
      {/* ================================================================== */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-8">
        {/* Overall Health Score */}
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <div className={`mx-auto flex h-14 w-14 items-center justify-center rounded-full border-4 ${scoreRingColor(overallScore)}`}>
            <span className="text-lg font-bold">{formatScore(overallScore)}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Overall Health</p>
        </div>
        {/* Check Runs — F5.3: the dashboard rolls up per domain-pair, so a
            cross-domain referential check is counted once per domain it touches.
            That makes this number (e.g. 166) legitimately larger than the
            "Check Catalog (83)" header, which lists DISTINCT definitions. Relabel
            to "Check Runs" and disclose the definition denominator so the two
            surfaces self-explain instead of looking contradictory. */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-2xl font-bold">{totalChecks}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Check Runs</p>
          {checkList.length > 0 && (
            <p className="text-[10px] text-muted-foreground/80">
              across {checkList.length} definition{checkList.length !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        {/* Passed */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <span className="text-2xl font-bold text-green-600">{totalPass}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Passed</p>
        </div>
        {/* Failed */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-red-600" />
            <span className="text-2xl font-bold text-red-600">{totalFail}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Failed</p>
        </div>
        {/* Warnings */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <span className="text-2xl font-bold text-amber-500">{totalWarn}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Warnings</p>
        </div>
        {/* Info-severity fails — non-passing 'info' checks. Surfaced in their
            own muted tile (F7.2) so a planner can see why some checks are
            non-passing without them masquerading as hard (red) failures. */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <MinusCircle className="h-4 w-4 text-blue-500" />
            <span className="text-2xl font-bold text-blue-500">{totalInfo}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Info</p>
        </div>
        {/* Skipped — checks whose source table was absent at run time. Surfaced
            so the tile counts reconcile with Total (passed+failed+warn+skip),
            and excluded from the health score denominator (F7.1). */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <MinusCircle className="h-4 w-4 text-muted-foreground" />
            <span className="text-2xl font-bold text-muted-foreground">{totalSkip}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Skipped</p>
        </div>
        {/* Last Run */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">{lastRun ? relativeTime(lastRun) : "Never"}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Last Run</p>
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 1: Domain Health Grid                                      */}
      {/* ================================================================== */}
      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">Domain Health</h3>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {domains.map((d: DQDomainScore) => {
            const isSelected = domainFilter === d.domain;
            return (
              <button
                key={d.domain}
                type="button"
                onClick={() => setDomainFilter(isSelected ? null : d.domain)}
                className={`cursor-pointer rounded-lg border p-4 text-left transition-all ${
                  isSelected
                    ? "border-primary bg-primary/5 ring-2 ring-primary/30"
                    : "border-border bg-card hover:border-primary/40"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium capitalize">{d.domain}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-bold ${scoreBadgeClass(d.score)}`}
                    title={
                      d.score === null
                        ? "No pass-rate to grade — this domain's only checks are failing warning/info checks (warn-only). Review the warn/info chips."
                        : undefined
                    }
                  >
                    {formatScore(d.score)}
                  </span>
                </div>
                <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                  <span className="text-green-600">{d.passed} pass</span>
                  {/* F2.1/U2.18 + F3.1 — the red chip shows only CRITICAL fails.
                      info-severity and warning-severity fails are excluded from
                      the score (Check Catalog labels warning fails "WARNING"),
                      so subtract both here to avoid a red "2 fail" / red 0% badge
                      contradicting the catalog. info fails surface on the "{n}
                      info" chip; warning fails roll into the amber warn chip. */}
                  <span className="text-red-600">
                    {Math.max(0, d.failed - (d.info_fails ?? 0) - (d.warning_fails ?? 0))} fail
                  </span>
                  <span className="text-amber-600">
                    {d.warnings + (d.warning_fails ?? 0)} warn
                  </span>
                  {(d.skipped ?? 0) > 0 && (
                    <span
                      className="text-muted-foreground"
                      title="Checks skipped (source table absent); excluded from the score"
                    >
                      {d.skipped} skip
                    </span>
                  )}
                  {(d.info_fails ?? 0) > 0 && (
                    <span
                      className="text-blue-600"
                      title="Informational fails (severity 'info'); excluded from the score so they do not alarm-red the domain"
                    >
                      {d.info_fails} info
                    </span>
                  )}
                </div>
              </button>
            );
          })}
          {domains.length === 0 && (
            <div className="col-span-full">
              <EmptyState
                variant="no-data"
                icon={Shield}
                title="No data quality checks have been run yet"
                description="Run the full DQ battery to populate domain health scores and issue catalogs — no CLI needed."
                onAction={() => { if (!runChecksMutation.isPending) runChecksMutation.mutate(undefined); }}
                actionLabel="Run DQ checks now"
                steps={[
                  { label: "Or trigger from a shell", command: "uv run python scripts/populate_dq_checks.py" },
                ]}
              />
            </div>
          )}
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 2: Check Catalog (extracted)                               */}
      {/* ================================================================== */}
      <CheckCatalogPanel checkList={checkList} domainFilter={domainFilter} />

      {/* ================================================================== */}
      {/* SECTION 3: Recent Issues (extracted)                               */}
      {/* ================================================================== */}
      <RecentIssuesPanel historyEntries={historyEntries} domainFilter={domainFilter} />

      {/* ================================================================== */}
      {/* SECTION 4: Self-Heal (extracted)                                   */}
      {/* ================================================================== */}
      <SelfHealPanel />

      {/* ================================================================== */}
      {/* SECTION 6: Pipeline Lineage                                        */}
      {/* ================================================================== */}
      <PipelineLineageSection />

      {/* ================================================================== */}
      {/* SECTION 7: Corrections Audit Log                                   */}
      {/* ================================================================== */}
      <CorrectionsSection />
    </div>
  );
}

/* ========================================================================== */
/*  Section 6: Pipeline Lineage                                               */
/* ========================================================================== */
function PipelineLineageSection() {
  const { data } = useQuery({
    queryKey: lineageKeys.batches,
    queryFn: () => fetchBatches(undefined, undefined, 20),
    staleTime: STALE_LINEAGE,
  });
  const batches: LoadBatch[] = data?.batches ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-medium text-foreground">Pipeline Lineage</h3>
      {batches.length === 0 ? (
        <EmptyState
          variant="no-data"
          icon={GitBranch}
          title="No pipeline batches yet"
          description='Batches appear after the ETL pipeline runs. Trigger an ingest to start the lineage log.'
          steps={[
            { label: "Normalize source CSVs", command: "make normalize-all" },
            { label: "Load into Postgres", command: "make load-all" },
          ]}
        />
      ) : (
        <div className="space-y-2">
          {batches.map((b) => (
            <div key={b.batch_id} className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2 text-sm">
              <div className="flex items-center gap-2">
                <span className={`inline-block h-2 w-2 rounded-full ${b.status === "completed" ? "bg-emerald-500" : b.status === "failed" ? "bg-red-500" : "bg-amber-500"}`} />
                <span className="font-mono text-xs">{b.domain}</span>
                <span className="text-xs text-muted-foreground">Batch #{b.batch_id}</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>{b.row_count_in?.toLocaleString() ?? "\u2014"} in</span>
                <span>{b.row_count_out?.toLocaleString() ?? "\u2014"} out</span>
                <span>{b.started_at ? new Date(b.started_at).toLocaleString() : "\u2014"}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ========================================================================== */
/*  Section 7: DQ Corrections Browser                                         */
/* ========================================================================== */
function CorrectionsSection() {
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
                            : "\u2014"}
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
                    <th className="pb-2 pr-3 text-center">\u2192</th>
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
                        <td className="py-1.5 pr-3 font-mono">{c.period?.slice(0, 7) ?? "\u2014"}</td>
                        <td className="py-1.5 pr-3 text-right font-mono text-red-600 dark:text-red-400">
                          {formatNum(c.old_value)}
                        </td>
                        <td className="py-1.5 pr-3 text-center text-muted-foreground">\u2192</td>
                        <td className="py-1.5 pr-3 text-right font-mono text-emerald-600 dark:text-emerald-400">
                          {formatNum(c.new_value)}
                        </td>
                        <td className={`py-1.5 pr-3 text-right font-mono ${
                          pctChange != null && pctChange < 0
                            ? "text-red-500"
                            : "text-emerald-500"
                        }`}>
                          {pctChange != null ? `${pctChange > 0 ? "+" : ""}${pctChange.toFixed(1)}%` : "\u2014"}
                        </td>
                        <td className="py-1.5 pr-3">
                          <span className="rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">{c.fix_type}</span>
                        </td>
                        <td className="py-1.5 pr-3 text-muted-foreground">{c.fix_strategy ?? "\u2014"}</td>
                        <td className="py-1.5 pr-3 text-right text-muted-foreground font-mono">
                          {c.lower_bound != null && c.upper_bound != null
                            ? `[${formatNum(c.lower_bound)}, ${formatNum(c.upper_bound)}]`
                            : "\u2014"}
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
