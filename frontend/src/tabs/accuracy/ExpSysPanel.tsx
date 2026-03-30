/**
 * ExpSysPanel — Expert System Backtest results panel.
 *
 * Shows per-lag accuracy (lag 0–4 + execution-lag) and a per-segment
 * breakdown for the ExpSys production forecast backtest.
 */

import { useQuery } from "@tanstack/react-query";
import { Activity, Database } from "lucide-react";
import {
  expSysKeys,
  fetchExpSysLagAccuracy,
  fetchExpSysStatus,
  type ExpSysLagStats,
} from "@/api/queries/expsys";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const STALE_5M = 5 * 60 * 1000;
const STALE_30S = 30 * 1000;

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBar({
  completedTimeframes,
  dbRowCount,
  reportUpdatedAt,
}: {
  completedTimeframes: number;
  dbRowCount: number;
  reportUpdatedAt: string | null;
}) {
  const updated = reportUpdatedAt
    ? new Date(reportUpdatedAt).toLocaleString()
    : "—";
  return (
    <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-4">
      <span className="flex items-center gap-1">
        <Activity className="h-3.5 w-3.5" />
        {completedTimeframes} / 10 timeframes complete
      </span>
      <span className="flex items-center gap-1">
        <Database className="h-3.5 w-3.5" />
        {dbRowCount.toLocaleString()} rows in DB
      </span>
      <span>Last run: {updated}</span>
    </div>
  );
}

function AccuracyCell({ value }: { value: number | undefined }) {
  if (value === undefined) return <td className="px-3 py-2 text-center text-muted-foreground">—</td>;
  const cls =
    value >= 75
      ? "text-emerald-600 dark:text-emerald-400 font-semibold"
      : value >= 65
        ? "text-amber-600 dark:text-amber-400"
        : "text-red-600 dark:text-red-400";
  return <td className={`px-3 py-2 text-center ${cls}`}>{value.toFixed(1)}%</td>;
}

function LagTable({
  byLag,
  executionLag,
}: {
  byLag: Record<string, ExpSysLagStats>;
  executionLag: ExpSysLagStats | null;
}) {
  const lags = Object.keys(byLag)
    .map(Number)
    .sort((a, b) => a - b);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b text-muted-foreground">
            <th className="px-3 py-2 text-left">Horizon</th>
            <th className="px-3 py-2 text-center">Accuracy</th>
            <th className="px-3 py-2 text-center">WAPE</th>
            <th className="px-3 py-2 text-right">DFUs</th>
            <th className="px-3 py-2 text-right">DFU-months</th>
          </tr>
        </thead>
        <tbody>
          {lags.map((lag) => {
            const s = byLag[String(lag)];
            return (
              <tr key={lag} className="border-b hover:bg-muted/30">
                <td className="px-3 py-2 font-medium">Lag {lag}</td>
                <AccuracyCell value={s.accuracy_pct} />
                <td className="px-3 py-2 text-center">{s.wape.toFixed(1)}%</td>
                <td className="px-3 py-2 text-right">{s.n_dfus.toLocaleString()}</td>
                <td className="px-3 py-2 text-right">{s.n_dfu_months.toLocaleString()}</td>
              </tr>
            );
          })}
          {executionLag && executionLag.n_dfus > 0 && (
            <tr className="bg-muted/20 font-semibold border-t-2">
              <td className="px-3 py-2">Exec Lag</td>
              <AccuracyCell value={executionLag.accuracy_pct} />
              <td className="px-3 py-2 text-center">{executionLag.wape.toFixed(1)}%</td>
              <td className="px-3 py-2 text-right">{executionLag.n_dfus.toLocaleString()}</td>
              <td className="px-3 py-2 text-right">{executionLag.n_dfu_months.toLocaleString()}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function SegmentTable({
  byLag,
  executionLag,
}: {
  byLag: Record<string, ExpSysLagStats>;
  executionLag: ExpSysLagStats | null;
}) {
  const lags = Object.keys(byLag)
    .map(Number)
    .sort((a, b) => a - b);

  const segments = Array.from(
    new Set(
      Object.values(byLag).flatMap((s) => Object.keys(s.per_segment ?? {})),
    ),
  ).sort();

  if (segments.length === 0) return null;

  return (
    <div className="mt-4 overflow-x-auto">
      <p className="text-xs font-medium text-muted-foreground mb-2">
        Per-segment accuracy (%) by lag
      </p>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b text-muted-foreground">
            <th className="px-3 py-2 text-left">Segment</th>
            {lags.map((lag) => (
              <th key={lag} className="px-3 py-2 text-center">
                Lag {lag}
              </th>
            ))}
            {executionLag && <th className="px-3 py-2 text-center">Exec</th>}
          </tr>
        </thead>
        <tbody>
          {segments.map((seg) => (
            <tr key={seg} className="border-b hover:bg-muted/30">
              <td className="px-3 py-2 text-muted-foreground capitalize">
                {seg.replace(/_/g, " ")}
              </td>
              {lags.map((lag) => (
                <AccuracyCell
                  key={lag}
                  value={byLag[String(lag)]?.per_segment?.[seg]}
                />
              ))}
              {executionLag && (
                <AccuracyCell value={executionLag.per_segment?.[seg]} />
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExpSysPanel() {
  const { data: status } = useQuery({
    queryKey: expSysKeys.status(),
    queryFn: fetchExpSysStatus,
    staleTime: STALE_30S,
  });

  const { data: report, isLoading, isError } = useQuery({
    queryKey: expSysKeys.lagAccuracy(),
    queryFn: () => fetchExpSysLagAccuracy(),
    staleTime: STALE_5M,
    enabled: status?.has_report !== false,
    retry: false,
  });

  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          <CardTitle className="text-base">Expert System Backtest (ExpSys)</CardTitle>
        </div>
        <CardDescription>
          Full-population backtest using segment-assigned algorithms (nbeats / chronos / mstl).
          Accuracy at lag 0–4 and execution lag across all 10 timeframes.
          Run <code className="text-xs bg-muted px-1 rounded">make expsys-backtest</code> to refresh.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {status && (
          <StatusBar
            completedTimeframes={status.completed_timeframes}
            dbRowCount={status.db_row_count}
            reportUpdatedAt={status.report_updated_at}
          />
        )}

        {isLoading && (
          <p className="text-sm text-muted-foreground">Loading accuracy report…</p>
        )}

        {isError && !isLoading && (
          <p className="text-sm text-muted-foreground">
            No ExpSys report found. Run{" "}
            <code className="bg-muted px-1 rounded text-xs">make expsys-backtest</code> to
            generate results.
          </p>
        )}

        {report && (
          <>
            <LagTable byLag={report.by_lag} executionLag={report.execution_lag} />
            <SegmentTable byLag={report.by_lag} executionLag={report.execution_lag} />
          </>
        )}
      </CardContent>
    </Card>
  );
}
