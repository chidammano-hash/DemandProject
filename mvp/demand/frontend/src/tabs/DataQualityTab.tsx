/**
 * Data Quality & Pipeline Observability Tab (Spec 08-01).
 *
 * Four sections:
 *  1. Summary KPI bar — overall health, total checks, pass/fail/warn/error counts, last run
 *  2. Domain Health Grid — clickable domain cards that filter check catalog + issues
 *  3. Check Catalog Table — all checks, filterable by domain/status/severity, sortable
 *  4. Recent Issues — history entries for failed/error/warn checks with details
 *  5. Pipeline Freshness — per-table last-load timestamps
 */
import { useState, useEffect, useMemo } from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  fetchDQDashboard,
  fetchDQChecks,
  fetchDQFreshness,
  fetchDQHistory,
  fetchDQFixPreview,
  applyDQFixes,
  runDQChecks,
  dqKeys,
  STALE_PLATFORM,
  fetchBatches,
  fetchCorrections,
  fetchQuarantine,
  resolveQuarantine,
  lineageKeys,
  STALE_LINEAGE,
} from "@/api/queries";
import type { DQDomainScore, DQCheck, DQHistoryEntry, DQFixItem, LoadBatch, DQCorrection, QuarantineEntry } from "@/api/queries/platform";
import {
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertOctagon,
  MinusCircle,
  Activity,
  Shield,
  ArrowUpDown,
  Filter,
  Clock,
  Lightbulb,
  ChevronDown,
  ChevronRight,
  Wrench,
  Zap,
  Check,
  X,
  Loader2,
} from "lucide-react";

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

type SortField = "check_name" | "domain" | "severity" | "last_status" | "last_run";
type SortDir = "asc" | "desc";

const STATUS_ICON: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
  pass: { icon: CheckCircle2, color: "text-green-600", label: "Pass" },
  fail: { icon: XCircle, color: "text-red-600", label: "Fail" },
  error: { icon: AlertOctagon, color: "text-orange-500", label: "Error" },
  warn: { icon: AlertTriangle, color: "text-amber-500", label: "Warn" },
};

const SEVERITY_STYLE: Record<string, string> = {
  critical: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  high: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400",
  warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  low: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
  info: "bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400",
};

function scoreBadgeClass(score: number): string {
  if (score > 80) return "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400";
  if (score >= 50) return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400";
  return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400";
}

function scoreRingColor(score: number): string {
  if (score > 80) return "border-green-500";
  if (score >= 50) return "border-amber-500";
  return "border-red-500";
}

/** AI-style analysis: root cause + fix steps derived from check type + details */
function analyzeIssue(entry: DQHistoryEntry): { summary: string; rootCause: string; fixSteps: string[] } {
  const d = (entry.details ?? {}) as Record<string, unknown>;
  const val = entry.metric_value;
  const name = entry.check_name;
  const table = entry.table_name;

  if (entry.status === "error") {
    const err = String(d.error ?? "Unknown error");
    const colMatch = err.match(/column (?:reference )?"?([^"]+)"? (?:does not exist|is ambiguous)/i);
    return {
      summary: `SQL error executing check "${name}" on ${table}.`,
      rootCause: colMatch
        ? `Column "${colMatch[1]}" referenced in the check definition does not match the actual database schema for ${table}.`
        : `The SQL template for this check produced an error: ${err.split("\n")[0]}`,
      fixSteps: [
        `Verify the column names in config/data_quality_config.yaml against the actual table schema`,
        `Run: SELECT column_name FROM information_schema.columns WHERE table_name = '${table}'`,
        `Update the config and re-run: uv run python scripts/populate_dq_checks.py`,
      ],
    };
  }

  switch (entry.check_type ?? name.split("_")[0]) {
    case "freshness": {
      const hrs = Number(d.hours_since_load ?? val ?? 0);
      const days = Math.round(hrs / 24);
      return {
        summary: `Table ${table} has not been refreshed in ${days} days (${Math.round(hrs)} hours).`,
        rootCause: `The ETL pipeline for ${table} has not loaded new data recently. This could indicate a stalled data pipeline, upstream source delays, or a scheduler failure.`,
        fixSteps: [
          `Check the ETL job scheduler for ${table} (make check-db to verify row counts)`,
          `Verify upstream data source availability`,
          `Re-run the data load: make load-all or the specific load target for this table`,
          `If this is expected (e.g., monthly data), adjust the freshness threshold in config/data_quality_config.yaml`,
        ],
      };
    }
    case "referential_integrity":
    case "referential": {
      const orphans = Number(d.orphan_keys ?? val ?? 0);
      const src = String(d.source_table ?? table);
      const tgt = String(d.target_table ?? "unknown");
      return {
        summary: `${orphans.toLocaleString()} rows in ${src} reference keys that don't exist in ${tgt}.`,
        rootCause: `Orphan foreign key references indicate that ${src} contains records pointing to master data entries missing from ${tgt}. This typically occurs when data is loaded out of order (facts before dimensions) or when dimension records are deleted without cascading.`,
        fixSteps: [
          `Identify orphan keys: SELECT DISTINCT s.key_col FROM ${src} s LEFT JOIN ${tgt} t ON s.key = t.key WHERE t.key IS NULL LIMIT 20`,
          `Re-load the dimension table: make normalize-all && make load-all`,
          `If orphans are expected (deprecated items), consider adding a "soft delete" flag to the dimension`,
          `For ${orphans > 10000 ? "large orphan counts" : "small counts"}: ${orphans > 10000 ? "prioritize fixing the data pipeline — this affects downstream analytics accuracy" : "review whether these are edge cases that can be safely excluded"}`,
        ],
      };
    }
    case "range": {
      const outliers = Number(d.outliers ?? val ?? 0);
      const total = Number(d.total ?? 0);
      const pct = Number(d.outlier_pct ?? 0);
      const min = d.min != null ? Number(d.min) : null;
      const max = d.max != null ? Number(d.max) : null;
      const col = name.replace(/^range_\w+_/, "");
      const bounds = [min != null ? `min=${min}` : "", max != null ? `max=${max}` : ""].filter(Boolean).join(", ");
      return {
        summary: `${outliers.toLocaleString()} rows (${pct.toFixed(2)}%) in ${table}.${col} are outside the expected range [${bounds}].`,
        rootCause: col.includes("lead_time")
          ? `Lead time values outside [0, ${max}] days suggest data entry errors, missing default values, or records imported without proper validation. Extremely large lead times (e.g., >730 days) are almost certainly data errors.`
          : `Column ${col} contains values outside the acceptable business range. This could be caused by data entry errors, unit conversion issues, or records imported without validation.`,
        fixSteps: [
          `Investigate outliers: SELECT ${col}, COUNT(*) FROM ${table} WHERE ${col} < ${min ?? 0} OR ${col} > ${max ?? "∞"} GROUP BY ${col} ORDER BY COUNT(*) DESC LIMIT 20`,
          `Determine if outliers are data errors or legitimate edge cases`,
          outliers > 100000
            ? `With ${outliers.toLocaleString()} outliers, consider a bulk data cleanup script`
            : `Fix individual records or apply a default value for invalid entries`,
          `Add validation rules to the ETL pipeline to prevent future occurrences`,
        ],
      };
    }
    case "volume_delta":
    case "volume": {
      const pctChange = Number(d.pct_change ?? val ?? 0);
      const prev = Number(d.prev_count ?? 0);
      const latest = Number(d.latest_count ?? 0);
      const direction = latest < prev ? "decreased" : "increased";
      return {
        summary: `Row count in ${table} ${direction} by ${pctChange.toFixed(1)}% (${prev.toLocaleString()} → ${latest.toLocaleString()}).`,
        rootCause: direction === "decreased"
          ? `A significant drop in row count could indicate: incomplete data load (ETL job failed midway), data purge/archival, or a change in source system filtering criteria.`
          : `A significant increase could indicate: duplicate data being loaded, a change in source scope, or backfill of historical data.`,
        fixSteps: [
          `Compare recent loads: SELECT load_ts::date, COUNT(*) FROM ${table} GROUP BY 1 ORDER BY 1 DESC LIMIT 5`,
          `Verify the source data file row counts match expectations`,
          direction === "decreased"
            ? `Check ETL job logs for errors or partial loads`
            : `Check for duplicate records: run the uniqueness check for this table`,
          `If the volume change is expected, adjust max_pct_change in config/data_quality_config.yaml`,
        ],
      };
    }
    case "completeness": {
      const nullPct = Number(d.null_pct ?? val ?? 0);
      const nulls = Number(d.nulls ?? 0);
      const total2 = Number(d.total ?? 0);
      const col2 = name.replace(/^completeness_\w+_/, "");
      return {
        summary: `${nullPct.toFixed(1)}% of ${col2} values in ${table} are NULL (${nulls.toLocaleString()} of ${total2.toLocaleString()} rows).`,
        rootCause: `Missing values in ${col2} reduce data quality for downstream analytics that depend on this field. Common causes: optional field in source system, ETL mapping errors, or schema changes in the source.`,
        fixSteps: [
          `Investigate: SELECT COUNT(*) FROM ${table} WHERE ${col2} IS NULL`,
          `Check if NULLs correlate with a specific load batch: SELECT load_ts::date, COUNT(*) FILTER (WHERE ${col2} IS NULL) FROM ${table} GROUP BY 1 ORDER BY 1 DESC`,
          `Determine if a default value is appropriate, or if the source data needs correction`,
          `Update the null_pct_threshold in config if the current NULL rate is acceptable`,
        ],
      };
    }
    case "uniqueness": {
      const dups = Number(d.duplicate_groups ?? val ?? 0);
      return {
        summary: `${dups.toLocaleString()} duplicate key groups found in ${table}.`,
        rootCause: `Duplicate records violate the expected uniqueness constraint. This typically occurs when data is loaded multiple times without deduplication, or when the natural key definition doesn't match the source system's grain.`,
        fixSteps: [
          `Identify duplicates: SELECT key_columns, COUNT(*) FROM ${table} GROUP BY key_columns HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC LIMIT 20`,
          `Check if duplicates are from multiple loads (compare load_ts values)`,
          `Consider adding a UNIQUE constraint or deduplication step to the ETL pipeline`,
          `Re-load with deduplication: make normalize-all && make load-all`,
        ],
      };
    }
    case "statistical_outlier": {
      const outliers = Number(d.outliers ?? val ?? 0);
      const method = String(d.method ?? "iqr").toUpperCase();
      const lower = d.lower_bound != null ? Number(d.lower_bound).toFixed(2) : "?";
      const upper = d.upper_bound != null ? Number(d.upper_bound).toFixed(2) : "?";
      const pct = Number(d.outlier_pct ?? 0);
      const median = d.median != null ? Number(d.median).toFixed(2) : "?";
      const iqr = d.iqr != null ? Number(d.iqr).toFixed(2) : "?";
      const col3 = name.replace(/^statistical_outlier_\w+_/, "");
      return {
        summary: `${outliers.toLocaleString()} statistical outliers (${pct.toFixed(2)}%) in ${table}.${col3} detected by ${method} method [${lower}, ${upper}].`,
        rootCause: `The ${method} method computed bounds [${lower}, ${upper}] using median=${median}, IQR=${iqr}. Values outside these bounds are statistically anomalous — likely data entry errors, unit mismatches, or extreme edge cases that skew downstream analytics (e.g., safety stock, demand variability).`,
        fixSteps: [
          `Review distribution: SELECT ${col3}, COUNT(*) FROM ${table} WHERE ${col3} < ${lower} OR ${col3} > ${upper} GROUP BY ${col3} ORDER BY COUNT(*) DESC LIMIT 20`,
          `Auto-fix (Winsorize to bounds): uv run python scripts/fix_dq_issues.py --fix outliers --apply`,
          `Preview first: uv run python scripts/fix_dq_issues.py --fix outliers`,
          `Adjust sensitivity: change threshold in config/data_quality_config.yaml under statistical_outlier`,
        ],
      };
    }
    case "distribution_drift": {
      const drift = Number(d.drift_score ?? val ?? 0);
      const maxDrift = Number(d.max_drift ?? 0.1);
      const meanShift = Number(d.mean_shift ?? 0);
      const stddevShift = Number(d.stddev_shift ?? 0);
      const medianShift = Number(d.median_shift ?? 0);
      const latest = (d.latest as Record<string, number>) ?? {};
      const previous = (d.previous as Record<string, number>) ?? {};
      const col4 = name.replace(/^distribution_drift_\w+_/, "");
      return {
        summary: `Distribution drift of ${(drift * 100).toFixed(1)}% detected in ${table}.${col4} (threshold: ${(maxDrift * 100).toFixed(0)}%).`,
        rootCause: `The statistical distribution of ${col4} shifted significantly between load batches. Mean shift: ${(meanShift * 100).toFixed(1)}%, StdDev shift: ${(stddevShift * 100).toFixed(1)}%, Median shift: ${(medianShift * 100).toFixed(1)}%. Previous mean=${previous.mean?.toFixed(2) ?? "?"}, Latest mean=${latest.mean?.toFixed(2) ?? "?"}. This can indicate a change in data source, data processing logic, or a genuine business shift.`,
        fixSteps: [
          `Compare distributions: SELECT load_ts::date, avg(${col4}), stddev(${col4}), count(*) FROM ${table} GROUP BY 1 ORDER BY 1 DESC LIMIT 5`,
          `Verify if the shift is expected (business seasonality, new products, etc.)`,
          `If unexpected, investigate the source data pipeline for the ${d.latest_date ?? "latest"} batch`,
          `Adjust max_drift in config/data_quality_config.yaml if the new distribution is the correct baseline`,
        ],
      };
    }
    case "temporal_gaps": {
      const gapCount = Number(d.gap_count ?? val ?? 0);
      const grain = String(d.grain ?? "month");
      const dateCol = String(d.date_column ?? "startdate");
      const missing = (d.missing_periods as string[]) ?? [];
      const missingStr = missing.slice(0, 5).join(", ");
      return {
        summary: `${gapCount} missing ${grain}(s) in ${table}.${dateCol} between ${d.first_period ?? "?"} and ${d.last_period ?? "?"}.`,
        rootCause: `Time series data in ${table} has gaps — ${gapCount} expected ${grain}(s) have no data. Missing: ${missingStr}${gapCount > 5 ? ` (and ${gapCount - 5} more)` : ""}. This causes forecast models to produce inaccurate predictions and inventory calculations to have blind spots.`,
        fixSteps: [
          `Identify gap periods: SELECT date_trunc('${grain}', ${dateCol})::date AS period, count(*) FROM ${table} GROUP BY 1 ORDER BY 1`,
          `Check if source data was available for missing periods`,
          `Backfill from source: re-run the ETL pipeline for the missing time range`,
          `If gaps are expected (e.g., new product), document in the data catalog`,
        ],
      };
    }
    case "cross_column": {
      const violations = Number(d.violations ?? val ?? 0);
      const violPct = Number(d.violation_pct ?? 0);
      const rule = String(d.rule ?? "");
      const desc = String(d.description ?? "");
      return {
        summary: `${violations.toLocaleString()} rows (${violPct.toFixed(2)}%) in ${table} violate: "${desc || rule}".`,
        rootCause: `Cross-column consistency rule \`${rule}\` found ${violations.toLocaleString()} violations. This indicates logical inconsistencies in the data — likely from ETL transformation errors, manual data entry, or upstream system bugs.`,
        fixSteps: [
          `Investigate violations: SELECT * FROM ${table} WHERE NOT (${rule}) LIMIT 20`,
          `Determine if violations are data errors or legitimate edge cases`,
          `Fix at source: update the ETL pipeline to enforce the rule during load`,
          `For existing violations, consider a targeted UPDATE to correct the data`,
        ],
      };
    }
    case "cardinality_anomaly": {
      const changePct = Number(d.change_pct ?? val ?? 0);
      const newVals = Number(d.new_values ?? 0);
      const droppedVals = Number(d.dropped_values ?? 0);
      const latestDistinct = Number(d.latest_distinct ?? 0);
      const prevDistinct = Number(d.previous_distinct ?? 0);
      const col5 = name.replace(/^cardinality_anomaly_\w+_/, "");
      return {
        summary: `Cardinality of ${table}.${col5} changed ${changePct.toFixed(1)}%: ${newVals} new + ${droppedVals} dropped values (${prevDistinct}→${latestDistinct} distinct).`,
        rootCause: `The set of distinct values in ${col5} changed significantly between load batches. ${newVals > 0 ? `${newVals} new values appeared (could be new products/locations or data errors). ` : ""}${droppedVals > 0 ? `${droppedVals} values disappeared (could indicate data loss or intentional cleanup).` : ""}`,
        fixSteps: [
          newVals > 0 ? `Review new values: SELECT DISTINCT ${col5} FROM ${table} WHERE load_ts::date = '${d.latest_date}' EXCEPT SELECT DISTINCT ${col5} FROM ${table} WHERE load_ts::date = '${d.previous_date}'` : "No new values to review",
          droppedVals > 0 ? `Review dropped values: SELECT DISTINCT ${col5} FROM ${table} WHERE load_ts::date = '${d.previous_date}' EXCEPT SELECT DISTINCT ${col5} FROM ${table} WHERE load_ts::date = '${d.latest_date}'` : "No dropped values to review",
          `Verify with the data source team whether the cardinality change is expected`,
          `Adjust max_change_pct in config/data_quality_config.yaml if the new cardinality is correct`,
        ],
      };
    }
    default: {
      return {
        summary: `Check "${name}" failed on ${table} with value ${val}.`,
        rootCause: "Unable to determine specific root cause for this check type.",
        fixSteps: ["Review the check configuration and database state manually."],
      };
    }
  }
}

function formatDetailsToString(details: DQHistoryEntry["details"]): string {
  if (!details) return "No details available";
  if (typeof details === "string") return details;
  try {
    const entries = Object.entries(details);
    if (entries.length === 0) return "No details available";
    return entries
      .map(([key, val]) => `${key.replace(/_/g, " ")}: ${typeof val === "object" ? JSON.stringify(val) : String(val)}`)
      .join(" | ");
  } catch {
    return JSON.stringify(details);
  }
}

function relativeTime(isoStr: string | null): string {
  if (!isoStr) return "Never";
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

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

  const { data: freshness } = useQuery({
    queryKey: dqKeys.freshness,
    queryFn: fetchDQFreshness,
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
      queryClient.invalidateQueries({ queryKey: dqKeys.freshness });
      queryClient.invalidateQueries({ queryKey: ["dq", "history"] });
      setShowSuccess(true);
    },
  });

  useEffect(() => {
    if (!showSuccess) return;
    const timer = setTimeout(() => setShowSuccess(false), 5000);
    return () => clearTimeout(timer);
  }, [showSuccess]);

  /* ---- self-heal state ---- */
  const [healOpen, setHealOpen] = useState(false);
  const [selectedFixes, setSelectedFixes] = useState<Set<number>>(new Set());
  const [appliedFixes, setAppliedFixes] = useState<Set<number>>(new Set());
  const [rejectedFixes, setRejectedFixes] = useState<Set<number>>(new Set());
  const [applyResult, setApplyResult] = useState<{ total_applied: number; total_rows_fixed: number } | null>(null);

  const { data: fixPreview, isFetching: fixLoading, refetch: refetchPreview } = useQuery({
    queryKey: dqKeys.fixPreview,
    queryFn: fetchDQFixPreview,
    enabled: healOpen,
    staleTime: 0,
  });

  const fixItems = fixPreview?.items ?? [];
  const pendingFixItems = fixItems.filter(
    (f: DQFixItem) => !appliedFixes.has(f.id) && !rejectedFixes.has(f.id)
  );

  const applyFixesMutation = useMutation({
    mutationFn: (ids: number[]) => applyDQFixes(ids),
    onSuccess: (data) => {
      setApplyResult({ total_applied: data.total_applied, total_rows_fixed: data.total_rows_fixed });
      const newApplied = new Set(appliedFixes);
      for (const a of data.applied) newApplied.add(a.id);
      setAppliedFixes(newApplied);
      setSelectedFixes(new Set());
    },
  });

  const toggleFix = (id: number) => {
    setSelectedFixes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAllFixes = () => {
    setSelectedFixes(new Set(pendingFixItems.map((f: DQFixItem) => f.id)));
  };

  const deselectAllFixes = () => setSelectedFixes(new Set());

  const rejectSelected = () => {
    setRejectedFixes((prev) => {
      const next = new Set(prev);
      for (const id of selectedFixes) next.add(id);
      return next;
    });
    setSelectedFixes(new Set());
  };

  const resetHeal = () => {
    setSelectedFixes(new Set());
    setAppliedFixes(new Set());
    setRejectedFixes(new Set());
    setApplyResult(null);
    refetchPreview();
  };

  /* ---- derived data ---- */
  const domains = dashboard?.domains ?? [];
  const checkList = checks?.checks ?? [];
  const tables = freshness?.tables ?? [];
  const historyEntries = history?.entries ?? [];

  /* ---- filters & sort ---- */
  const [domainFilter, setDomainFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>("last_status");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const filteredChecks = useMemo(() => {
    let list = checkList;
    if (domainFilter) list = list.filter((c) => c.domain === domainFilter);
    if (statusFilter) list = list.filter((c) => (c.last_status ?? "none") === statusFilter);
    if (severityFilter) list = list.filter((c) => c.severity === severityFilter);

    const sorted = [...list].sort((a, b) => {
      const av = a[sortField] ?? "";
      const bv = b[sortField] ?? "";
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [checkList, domainFilter, statusFilter, severityFilter, sortField, sortDir]);

  /* issue severity filter */
  const [issueSeverityFilter, setIssueSeverityFilter] = useState<string | null>(null);

  /* issues = history entries that are fail/error/warn */
  const recentIssues = useMemo(() => {
    let list = historyEntries.filter((e) => e.status === "fail" || e.status === "error" || e.status === "warn");
    if (domainFilter) list = list.filter((e) => e.domain === domainFilter);
    if (issueSeverityFilter) list = list.filter((e) => e.severity === issueSeverityFilter);
    return list.slice(0, 50);
  }, [historyEntries, domainFilter, issueSeverityFilter]);

  /* summary KPIs */
  const overallScore = domains.length
    ? Math.round(domains.reduce((s, d) => s + d.score, 0) / domains.length)
    : 0;
  const totalChecks = domains.reduce((s, d) => s + d.total, 0);
  const totalPass = domains.reduce((s, d) => s + d.passed, 0);
  const totalFail = domains.reduce((s, d) => s + d.failed, 0);
  const totalWarn = domains.reduce((s, d) => s + d.warnings, 0);
  const totalError = totalChecks - totalPass - totalFail - totalWarn;
  const lastRun = checkList.reduce<string | null>((latest, c) => {
    if (!c.last_run) return latest;
    if (!latest) return c.last_run;
    return c.last_run > latest ? c.last_run : latest;
  }, null);

  /* expanded issues */
  const [expandedIssue, setExpandedIssue] = useState<string | null>(null);

  /* unique filter values */
  const uniqueStatuses = useMemo(() => {
    const set = new Set(checkList.map((c) => c.last_status ?? "none"));
    return Array.from(set).sort();
  }, [checkList]);
  const uniqueSeverities = useMemo(() => {
    const set = new Set(checkList.map((c) => c.severity));
    return Array.from(set).sort();
  }, [checkList]);

  const clearFilters = () => {
    setDomainFilter(null);
    setStatusFilter(null);
    setSeverityFilter(null);
  };
  const hasFilters = domainFilter || statusFilter || severityFilter;

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
      <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
        {/* Overall Health Score */}
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <div className={`mx-auto flex h-14 w-14 items-center justify-center rounded-full border-4 ${scoreRingColor(overallScore)}`}>
            <span className="text-lg font-bold">{overallScore}%</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Overall Health</p>
        </div>
        {/* Total Checks */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-2xl font-bold">{totalChecks}</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Total Checks</p>
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
                  <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${scoreBadgeClass(d.score)}`}>
                    {d.score}%
                  </span>
                </div>
                <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                  <span className="text-green-600">{d.passed} pass</span>
                  <span className="text-red-600">{d.failed} fail</span>
                  <span className="text-amber-600">{d.warnings} warn</span>
                </div>
              </button>
            );
          })}
          {domains.length === 0 && (
            <div className="col-span-full py-8 text-center text-sm text-muted-foreground">
              No data quality checks have been run yet. Click &quot;Run Checks Now&quot; to trigger a check.
            </div>
          )}
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 2: Check Catalog Table                                     */}
      {/* ================================================================== */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-medium text-foreground">
            Check Catalog ({filteredChecks.length}{filteredChecks.length !== checkList.length ? ` of ${checkList.length}` : ""})
          </h3>
          <div className="flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            {/* Status filter */}
            <select
              value={statusFilter ?? ""}
              onChange={(e) => setStatusFilter(e.target.value || null)}
              className="rounded border border-border bg-background px-2 py-1 text-xs"
              aria-label="Filter by status"
            >
              <option value="">All Statuses</option>
              {uniqueStatuses.map((s) => (
                <option key={s} value={s}>{s === "none" ? "Not Run" : s}</option>
              ))}
            </select>
            {/* Severity filter */}
            <select
              value={severityFilter ?? ""}
              onChange={(e) => setSeverityFilter(e.target.value || null)}
              className="rounded border border-border bg-background px-2 py-1 text-xs"
              aria-label="Filter by severity"
            >
              <option value="">All Severities</option>
              {uniqueSeverities.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="rounded px-2 py-1 text-xs text-primary hover:underline"
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-card">
              <tr className="border-b border-border text-left text-muted-foreground">
                <th className="pb-2 pr-3 w-10">Status</th>
                <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("severity")}>
                  <span className="inline-flex items-center gap-1">
                    Severity <ArrowUpDown className="h-3 w-3" />
                  </span>
                </th>
                <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("check_name")}>
                  <span className="inline-flex items-center gap-1">
                    Check Name <ArrowUpDown className="h-3 w-3" />
                  </span>
                </th>
                <th className="pb-2 pr-3">Type</th>
                <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("domain")}>
                  <span className="inline-flex items-center gap-1">
                    Domain <ArrowUpDown className="h-3 w-3" />
                  </span>
                </th>
                <th className="pb-2 pr-3">Table</th>
                <th className="pb-2 pr-3">Last Value</th>
                <th className="pb-2 cursor-pointer select-none" onClick={() => toggleSort("last_run")}>
                  <span className="inline-flex items-center gap-1">
                    Last Run <ArrowUpDown className="h-3 w-3" />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredChecks.map((c: DQCheck) => {
                const st = STATUS_ICON[c.last_status ?? ""] ?? {
                  icon: MinusCircle,
                  color: "text-gray-400",
                  label: c.last_status ?? "Not run",
                };
                const Icon = st.icon;
                return (
                  <tr key={c.check_id} className="border-b border-border/30 hover:bg-muted/30">
                    <td className="py-1.5 pr-3">
                      <span className="inline-flex items-center gap-1" title={st.label}>
                        <Icon className={`h-3.5 w-3.5 ${st.color}`} />
                      </span>
                    </td>
                    <td className="py-1.5 pr-3">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${SEVERITY_STYLE[c.severity] ?? SEVERITY_STYLE.low}`}>
                        {c.severity}
                      </span>
                    </td>
                    <td className="py-1.5 pr-3 font-medium">{c.check_name}</td>
                    <td className="py-1.5 pr-3 text-muted-foreground">{c.check_type}</td>
                    <td className="py-1.5 pr-3 capitalize text-muted-foreground">{c.domain}</td>
                    <td className="py-1.5 pr-3 font-mono text-muted-foreground">{c.table_name}</td>
                    <td className="py-1.5 pr-3 text-muted-foreground">
                      {c.last_value != null ? c.last_value.toFixed(2) : "\u2014"}
                    </td>
                    <td className="py-1.5 text-muted-foreground">{c.last_run ? relativeTime(c.last_run) : "\u2014"}</td>
                  </tr>
                );
              })}
              {filteredChecks.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-muted-foreground">
                    {checkList.length === 0 ? "No checks configured yet." : "No checks match the current filters."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 3: Recent Issues                                           */}
      {/* ================================================================== */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            <h3 className="text-sm font-medium text-foreground">
              Recent Issues ({recentIssues.length})
            </h3>
            {domainFilter && (
              <span className="rounded bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                {domainFilter}
              </span>
            )}
          </div>
          <select
            value={issueSeverityFilter ?? ""}
            onChange={(ev) => setIssueSeverityFilter(ev.target.value || null)}
            className="rounded border border-border bg-background px-2 py-1 text-xs text-foreground"
            aria-label="Filter issues by severity"
          >
            <option value="">All severities</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
          </select>
        </div>
        {recentIssues.length === 0 ? (
          <div className="py-6 text-center text-sm text-muted-foreground">
            {historyEntries.length === 0
              ? "No check history available. Run checks to see results here."
              : "No recent issues found. All checks are passing."}
          </div>
        ) : (
          <div className="max-h-[600px] space-y-2 overflow-y-auto">
            {recentIssues.map((e, i) => {
              const st = STATUS_ICON[e.status] ?? STATUS_ICON.fail;
              const Icon = st.icon;
              const issueKey = `${e.check_id}-${e.run_ts}-${i}`;
              const isExpanded = expandedIssue === issueKey;
              const analysis = analyzeIssue(e);
              return (
                <div
                  key={issueKey}
                  className={`rounded-md border px-3 py-2.5 transition-all ${
                    e.status === "fail"
                      ? "border-red-200 bg-red-50/50 dark:border-red-900/40 dark:bg-red-950/20"
                      : e.status === "error"
                        ? "border-orange-200 bg-orange-50/50 dark:border-orange-900/40 dark:bg-orange-950/20"
                        : "border-amber-200 bg-amber-50/50 dark:border-amber-900/40 dark:bg-amber-950/20"
                  }`}
                >
                  {/* Header row */}
                  <button
                    type="button"
                    onClick={() => setExpandedIssue(isExpanded ? null : issueKey)}
                    className="flex w-full items-start justify-between gap-2 text-left"
                  >
                    <div className="flex items-center gap-2">
                      {isExpanded
                        ? <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
                        : <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
                      }
                      <Icon className={`h-4 w-4 flex-shrink-0 ${st.color}`} />
                      <div>
                        <span className="text-sm font-medium">{e.check_name}</span>
                        <span className="ml-2 text-xs text-muted-foreground capitalize">
                          {e.domain} / {e.table_name}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${SEVERITY_STYLE[e.severity] ?? SEVERITY_STYLE.low}`}>
                        {e.severity}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {e.run_ts ? relativeTime(e.run_ts) : ""}
                      </span>
                    </div>
                  </button>

                  {/* Summary (always visible) */}
                  <div className="mt-1.5 ml-9 text-xs font-medium text-foreground/80">
                    <Lightbulb className="mr-1 inline h-3 w-3 text-amber-500" />
                    {analysis.summary}
                  </div>

                  {/* Expanded analysis */}
                  {isExpanded && (
                    <div className="mt-3 ml-9 space-y-3 border-t border-border/40 pt-3">
                      {/* Root Cause */}
                      <div>
                        <div className="flex items-center gap-1 text-xs font-semibold text-foreground/70 uppercase tracking-wide">
                          <AlertTriangle className="h-3 w-3" />
                          Root Cause
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                          {analysis.rootCause}
                        </p>
                      </div>

                      {/* Steps to Fix */}
                      <div>
                        <div className="flex items-center gap-1 text-xs font-semibold text-foreground/70 uppercase tracking-wide">
                          <Wrench className="h-3 w-3" />
                          Steps to Fix
                        </div>
                        <ol className="mt-1 list-decimal list-inside space-y-1">
                          {analysis.fixSteps.map((step, si) => (
                            <li key={si} className="text-xs text-muted-foreground leading-relaxed">
                              {step.includes("SELECT") || step.includes("make ")
                                ? <>{step.split(/(SELECT .+|make \S+)/g).map((part, pi) =>
                                    /^(SELECT |make )/.test(part)
                                      ? <code key={pi} className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]">{part}</code>
                                      : <span key={pi}>{part}</span>
                                  )}</>
                                : step
                              }
                            </li>
                          ))}
                        </ol>
                      </div>

                      {/* Raw Details (collapsed) */}
                      <details className="group">
                        <summary className="cursor-pointer text-[10px] text-muted-foreground/60 hover:text-muted-foreground">
                          Raw check details
                        </summary>
                        <pre className="mt-1 rounded bg-muted/50 p-2 text-[10px] font-mono text-muted-foreground overflow-x-auto">
                          {JSON.stringify(e.details, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ================================================================== */}
      {/* SECTION 4: Self-Heal                                               */}
      {/* ================================================================== */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-teal-500" />
            <h3 className="text-sm font-medium text-foreground">Self-Heal</h3>
            <span className="text-xs text-muted-foreground">
              Statistical auto-fix for detected issues
            </span>
          </div>
          <button
            onClick={() => { setHealOpen(!healOpen); if (!healOpen) resetHeal(); }}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs font-medium text-foreground hover:bg-muted/50"
          >
            <Zap className="h-3 w-3" />
            {healOpen ? "Close" : "Scan for Fixes"}
          </button>
        </div>

        {healOpen && (
          <div className="space-y-3">
            {/* Success banner */}
            {applyResult && (
              <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300">
                Applied {applyResult.total_applied} fix(es) affecting {applyResult.total_rows_fixed.toLocaleString()} rows.
              </div>
            )}

            {/* Loading */}
            {fixLoading && (
              <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Scanning for fixable issues...
              </div>
            )}

            {/* Empty state */}
            {!fixLoading && fixItems.length === 0 && (
              <div className="py-6 text-center text-sm text-muted-foreground">
                No fixable issues found. All data quality checks are within acceptable bounds.
              </div>
            )}

            {/* Fix items */}
            {!fixLoading && fixItems.length > 0 && (
              <>
                {/* Toolbar */}
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-muted/30 px-3 py-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={selectedFixes.size === pendingFixItems.length && pendingFixItems.length > 0 ? deselectAllFixes : selectAllFixes}
                      className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted/50"
                    >
                      {selectedFixes.size === pendingFixItems.length && pendingFixItems.length > 0 ? "Deselect All" : "Select All"}
                    </button>
                    <span className="text-xs text-muted-foreground">
                      {selectedFixes.size} of {pendingFixItems.length} selected
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => applyFixesMutation.mutate(Array.from(selectedFixes))}
                      disabled={selectedFixes.size === 0 || applyFixesMutation.isPending}
                      className="inline-flex items-center gap-1 rounded-md bg-teal-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-teal-700 disabled:opacity-50"
                    >
                      {applyFixesMutation.isPending
                        ? <><Loader2 className="h-3 w-3 animate-spin" /> Applying...</>
                        : <><Check className="h-3 w-3" /> Accept Selected</>
                      }
                    </button>
                    <button
                      onClick={rejectSelected}
                      disabled={selectedFixes.size === 0}
                      className="inline-flex items-center gap-1 rounded-md border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950/30"
                    >
                      <X className="h-3 w-3" /> Reject Selected
                    </button>
                    <button
                      onClick={resetHeal}
                      className="rounded px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                    >
                      Reset
                    </button>
                  </div>
                </div>

                {/* Fix list */}
                <div className="max-h-[500px] space-y-1.5 overflow-y-auto">
                  {fixItems.map((f: DQFixItem) => {
                    const isApplied = appliedFixes.has(f.id);
                    const isRejected = rejectedFixes.has(f.id);
                    const isSelected = selectedFixes.has(f.id);
                    const isPending = !isApplied && !isRejected;

                    return (
                      <div
                        key={f.id}
                        className={`flex items-start gap-3 rounded-md border px-3 py-2.5 transition-all ${
                          isApplied
                            ? "border-green-200 bg-green-50/50 dark:border-green-900/40 dark:bg-green-950/20"
                            : isRejected
                              ? "border-border/30 bg-muted/20 opacity-50"
                              : isSelected
                                ? "border-teal-300 bg-teal-50/30 dark:border-teal-800 dark:bg-teal-950/20"
                                : "border-border bg-card hover:border-border/60"
                        }`}
                      >
                        {/* Checkbox */}
                        {isPending && (
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleFix(f.id)}
                            className="mt-0.5 h-4 w-4 rounded border-border accent-teal-600"
                            aria-label={`Select fix: ${f.description}`}
                          />
                        )}
                        {isApplied && <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-600" />}
                        {isRejected && <X className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />}

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                              f.fix_type === "outliers" || f.fix_type === "range"
                                ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                                : f.fix_type === "completeness"
                                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                                  : f.fix_type === "lead_time"
                                    ? "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400"
                                    : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                            }`}>
                              {f.fix_type}
                            </span>
                            <span className="text-xs font-medium text-foreground truncate">
                              {f.description}
                            </span>
                          </div>
                          <div className="mt-1 flex items-center gap-3 text-[11px] text-muted-foreground">
                            <span>{f.affected_rows.toLocaleString()} rows affected</span>
                            {f.recommendation && (
                              <span className="text-amber-600 dark:text-amber-400">
                                {f.recommendation}
                              </span>
                            )}
                            {isApplied && <span className="font-medium text-green-600">Applied</span>}
                            {isRejected && <span className="font-medium text-red-400">Rejected</span>}
                          </div>
                        </div>

                        {/* Single accept / reject buttons */}
                        {isPending && !isSelected && (
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <button
                              onClick={() => applyFixesMutation.mutate([f.id])}
                              disabled={applyFixesMutation.isPending}
                              className="rounded p-1 text-teal-600 hover:bg-teal-50 dark:hover:bg-teal-950/30"
                              title="Accept this fix"
                            >
                              <Check className="h-3.5 w-3.5" />
                            </button>
                            <button
                              onClick={() => setRejectedFixes((prev) => new Set(prev).add(f.id))}
                              className="rounded p-1 text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30"
                              title="Reject this fix"
                            >
                              <X className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* ================================================================== */}
      {/* SECTION 5: Pipeline Freshness                                      */}
      {/* ================================================================== */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">Pipeline Freshness</h3>
        <div className="space-y-2">
          {tables.map((t: { table: string; last_load: string | null }) => (
            <div key={t.table} className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2 text-sm">
              <span className="font-mono text-xs">{t.table}</span>
              <span className={`text-xs ${t.last_load ? "text-muted-foreground" : "text-red-500"}`}>
                {t.last_load ? new Date(t.last_load).toLocaleString() : "Never loaded"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 6: Pipeline Lineage (Medallion)                            */}
      {/* ================================================================== */}
      <PipelineLineageSection />

      {/* ================================================================== */}
      {/* SECTION 7: Corrections Audit Log                                   */}
      {/* ================================================================== */}
      <CorrectionsSection />

      {/* ================================================================== */}
      {/* SECTION 8: Quarantine Queue                                        */}
      {/* ================================================================== */}
      <QuarantineSection />
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
        <p className="text-xs text-muted-foreground">No medallion batches yet. Run <code className="rounded bg-muted px-1">make medallion-load-all</code> to ingest data.</p>
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
                <span>{b.row_count_in?.toLocaleString() ?? "—"} in</span>
                <span>{b.row_count_out?.toLocaleString() ?? "—"} out</span>
                {(b.row_count_quarantined ?? 0) > 0 && (
                  <span className="text-amber-600">{b.row_count_quarantined} quarantined</span>
                )}
                <span>{b.started_at ? new Date(b.started_at).toLocaleString() : "—"}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ========================================================================== */
/*  Section 7: Corrections Audit Log                                          */
/* ========================================================================== */
function CorrectionsSection() {
  const { data } = useQuery({
    queryKey: lineageKeys.corrections,
    queryFn: () => fetchCorrections(undefined, undefined, undefined, 50),
    staleTime: STALE_LINEAGE,
  });
  const corrections: DQCorrection[] = data?.corrections ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-medium text-foreground">Corrections Audit Log</h3>
      {corrections.length === 0 ? (
        <p className="text-xs text-muted-foreground">No DQ corrections recorded. Corrections appear after running <code className="rounded bg-muted px-1">make medallion-load-sales-fix</code>.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-left text-muted-foreground">
                <th className="pb-2 pr-3">Domain</th>
                <th className="pb-2 pr-3">Column</th>
                <th className="pb-2 pr-3">Fix Type</th>
                <th className="pb-2 pr-3">Old</th>
                <th className="pb-2 pr-3">New</th>
                <th className="pb-2 pr-3">Applied</th>
              </tr>
            </thead>
            <tbody>
              {corrections.map((c) => (
                <tr key={c.correction_id} className="border-b border-border/30">
                  <td className="py-1.5 pr-3 font-mono">{c.domain}</td>
                  <td className="py-1.5 pr-3">{c.column_name}</td>
                  <td className="py-1.5 pr-3">
                    <span className="rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">{c.fix_type}</span>
                  </td>
                  <td className="py-1.5 pr-3 text-red-600 dark:text-red-400">{c.old_value ?? "NULL"}</td>
                  <td className="py-1.5 pr-3 text-emerald-600 dark:text-emerald-400">{c.new_value ?? "NULL"}</td>
                  <td className="py-1.5 pr-3 text-muted-foreground">{c.applied_at ? new Date(c.applied_at).toLocaleString() : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ========================================================================== */
/*  Section 8: Quarantine Queue                                               */
/* ========================================================================== */
function QuarantineSection() {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: lineageKeys.quarantine,
    queryFn: () => fetchQuarantine(undefined, false, 50),
    staleTime: STALE_LINEAGE,
  });
  const items: QuarantineEntry[] = data?.quarantine ?? [];

  const resolveMut = useMutation({
    mutationFn: (id: number) => resolveQuarantine(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: lineageKeys.quarantine }),
  });

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-medium text-foreground">Quarantine Queue</h3>
      {items.length === 0 ? (
        <p className="text-xs text-muted-foreground">No quarantined rows. Rows failing DQ gate checks appear here.</p>
      ) : (
        <div className="space-y-2">
          {items.map((q) => (
            <div key={q.quarantine_id} className="rounded-md border border-amber-200 bg-amber-50/50 px-3 py-2 dark:border-amber-800/40 dark:bg-amber-950/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                  <span className="font-mono text-xs">{q.domain}</span>
                  <span className="text-xs text-muted-foreground">#{q.quarantine_id}</span>
                </div>
                <button
                  onClick={() => resolveMut.mutate(q.quarantine_id)}
                  disabled={resolveMut.isPending}
                  className="rounded px-2 py-0.5 text-xs text-muted-foreground hover:bg-background"
                >
                  Dismiss
                </button>
              </div>
              <div className="mt-1 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">{q.rejection_reason}</span>
                {" — Batch #{q.load_batch_id}"}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
