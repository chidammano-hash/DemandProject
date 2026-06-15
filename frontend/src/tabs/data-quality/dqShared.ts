/**
 * Shared types, constants, and helpers for Data Quality sub-panels.
 */
import type { DQHistoryEntry } from "@/api/queries/platform";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertOctagon,
  type LucideIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type SortField = "check_name" | "domain" | "severity" | "last_status" | "last_run";
export type SortDir = "asc" | "desc";

// ---------------------------------------------------------------------------
// Status / Severity constants
// ---------------------------------------------------------------------------
export const STATUS_ICON: Record<string, { icon: LucideIcon; color: string; label: string }> = {
  pass: { icon: CheckCircle2, color: "text-green-600", label: "Pass" },
  fail: { icon: XCircle, color: "text-red-600", label: "Fail" },
  error: { icon: AlertOctagon, color: "text-orange-500", label: "Error" },
  warn: { icon: AlertTriangle, color: "text-amber-500", label: "Warn" },
};

export const SEVERITY_STYLE: Record<string, string> = {
  critical: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  high: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400",
  warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  low: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
  info: "bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400",
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
// A null score means "nothing scoreable" — the domain ran only warning/info
// checks (all flagging) so there is no pass-rate to grade. It must render a
// NEUTRAL/muted badge, never a green 100% (which would hide a real warning-only
// integrity gap) or a red 0% (which would over-alarm a non-critical warn-only
// domain). U4.2.
const NEUTRAL_BADGE =
  "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400";

export function scoreBadgeClass(score: number | null): string {
  if (score === null) return NEUTRAL_BADGE;
  if (score > 80) return "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400";
  if (score >= 50) return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400";
  return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400";
}

export function scoreRingColor(score: number | null): string {
  if (score === null) return "border-slate-400";
  if (score > 80) return "border-green-500";
  if (score >= 50) return "border-amber-500";
  return "border-red-500";
}

// Render a domain/overall score for display: a number renders as "NN%", a null
// (warn-only / nothing scoreable) renders an em-dash so the badge never shows a
// misleading "100%" or "0%".
export function formatScore(score: number | null): string {
  return score === null ? "—" : `${score}%`;
}

export function relativeTime(isoStr: string | null): string {
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

export function formatDetailsToString(details: DQHistoryEntry["details"]): string {
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

// ---------------------------------------------------------------------------
// Issue analysis — AI-style root cause + fix steps
// ---------------------------------------------------------------------------
export interface IssueAnalysis {
  summary: string;
  rootCause: string;
  fixSteps: string[];
}

export function analyzeIssue(entry: DQHistoryEntry): IssueAnalysis {
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
          `Investigate outliers: SELECT ${col}, COUNT(*) FROM ${table} WHERE ${col} < ${min ?? 0} OR ${col} > ${max ?? "\u221e"} GROUP BY ${col} ORDER BY COUNT(*) DESC LIMIT 20`,
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
        summary: `Row count in ${table} ${direction} by ${pctChange.toFixed(1)}% (${prev.toLocaleString()} \u2192 ${latest.toLocaleString()}).`,
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
        rootCause: `The ${method} method computed bounds [${lower}, ${upper}] using median=${median}, IQR=${iqr}. Values outside these bounds are statistically anomalous \u2014 likely data entry errors, unit mismatches, or extreme edge cases that skew downstream analytics (e.g., safety stock, demand variability).`,
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
        rootCause: `Time series data in ${table} has gaps \u2014 ${gapCount} expected ${grain}(s) have no data. Missing: ${missingStr}${gapCount > 5 ? ` (and ${gapCount - 5} more)` : ""}. This causes forecast models to produce inaccurate predictions and inventory calculations to have blind spots.`,
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
        rootCause: `Cross-column consistency rule \`${rule}\` found ${violations.toLocaleString()} violations. This indicates logical inconsistencies in the data \u2014 likely from ETL transformation errors, manual data entry, or upstream system bugs.`,
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
        summary: `Cardinality of ${table}.${col5} changed ${changePct.toFixed(1)}%: ${newVals} new + ${droppedVals} dropped values (${prevDistinct}\u2192${latestDistinct} distinct).`,
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
