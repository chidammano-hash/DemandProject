import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, ChevronLeft, ChevronRight } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchVariabilitySummary,
  fetchVariabilityDetail,
} from "@/api/queries";
import type { VariabilityDetailRow } from "@/api/queries";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { LoadingElement } from "@/components/LoadingElement";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/formatters";

const VAR_PAGE = 20;

export function DemandVariabilityPanel() {
  const [panelOpen, setPanelOpen] = useState(false);
  const [classFilter, setClassFilter] = useState("");
  const [varOffset, setVarOffset] = useState(0);

  const { data: varSummary, isLoading: loadingVarSummary } = useQuery({
    queryKey: queryKeys.variabilitySummary({ abc_vol: "" }),
    queryFn: () => fetchVariabilitySummary({}),
    staleTime: STALE.FIVE_MIN,
    enabled: panelOpen,
  });

  const varDetailParams = useMemo(
    () => ({
      variability_class: classFilter || undefined,
      limit: VAR_PAGE,
      offset: varOffset,
      sort_by: "demand_cv",
      sort_dir: "desc",
    }),
    [classFilter, varOffset],
  );

  const { data: varDetail, isLoading: loadingVarDetail } = useQuery({
    queryKey: queryKeys.variabilityDetail(varDetailParams),
    queryFn: () => fetchVariabilityDetail(varDetailParams),
    staleTime: STALE.FIVE_MIN,
    enabled: panelOpen,
  });

  const varRows: VariabilityDetailRow[] = varDetail?.rows ?? [];

  const CLASS_COLORS: Record<string, string> = {
    low: "text-green-600 dark:text-green-400",
    medium: "text-yellow-600 dark:text-yellow-400",
    high: "text-orange-600 dark:text-orange-400",
    lumpy: "text-red-600 dark:text-red-400",
  };

  const ROW_COLORS: Record<string, string> = {
    low: "bg-green-50 dark:bg-green-950/20",
    medium: "bg-yellow-50 dark:bg-yellow-950/20",
    high: "bg-orange-50 dark:bg-orange-950/20",
    lumpy: "bg-red-50 dark:bg-red-950/20",
  };

  const BADGE_COLORS: Record<string, string> = {
    low: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    high: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
    lumpy: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  };

  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setPanelOpen((o) => !o)}
      >
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Demand Variability Profile</CardTitle>
          {panelOpen ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </div>
        <CardDescription>
          CV-based variability class per DFU — low / medium / high / lumpy
        </CardDescription>
      </CardHeader>

      {panelOpen && (
        <CardContent className="space-y-4">
          {/* Summary class breakdown */}
          {loadingVarSummary ? (
            <LoadingElement
              tabKey="variability"
              message="Loading variability summary..."
            />
          ) : varSummary ? (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-3">
                {(["low", "medium", "high", "lumpy"] as const).map((cls) => (
                  <button
                    key={cls}
                    onClick={() =>
                      setClassFilter(classFilter === cls ? "" : cls)
                    }
                    className={cn(
                      "flex flex-col items-center rounded-lg border px-4 py-2 text-sm transition-colors",
                      classFilter === cls
                        ? "border-primary bg-primary/10"
                        : "border-border hover:border-primary/50",
                    )}
                  >
                    <span
                      className={cn(
                        "text-xl font-bold tabular-nums",
                        CLASS_COLORS[cls],
                      )}
                    >
                      {varSummary.by_class[cls]}
                    </span>
                    <span className="capitalize text-muted-foreground">
                      {cls}
                    </span>
                  </button>
                ))}
                <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                  <span className="text-xl font-bold tabular-nums">
                    {varSummary.avg_cv != null
                      ? Number(varSummary.avg_cv).toFixed(2)
                      : "—"}
                  </span>
                  <span className="text-muted-foreground">Avg CV</span>
                </div>
                <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                  <span className="text-xl font-bold tabular-nums">
                    {varSummary.avg_intermittency_ratio != null
                      ? `${(Number(varSummary.avg_intermittency_ratio) * 100).toFixed(1)}%`
                      : "—"}
                  </span>
                  <span className="text-muted-foreground">Avg Intermittency</span>
                </div>
              </div>

              {/* CV percentile bar */}
              <div className="text-xs text-muted-foreground">
                CV percentiles — p25:{" "}
                {varSummary.cv_percentiles.p25 != null
                  ? Number(varSummary.cv_percentiles.p25).toFixed(2)
                  : "—"}{" "}
                · p50:{" "}
                {varSummary.cv_percentiles.p50 != null
                  ? Number(varSummary.cv_percentiles.p50).toFixed(2)
                  : "—"}{" "}
                · p75:{" "}
                {varSummary.cv_percentiles.p75 != null
                  ? Number(varSummary.cv_percentiles.p75).toFixed(2)
                  : "—"}{" "}
                · p95:{" "}
                {varSummary.cv_percentiles.p95 != null
                  ? Number(varSummary.cv_percentiles.p95).toFixed(2)
                  : "—"}
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No variability data. Run{" "}
              <code>make variability-compute</code> first.
            </p>
          )}

          {/* Detail table */}
          {loadingVarDetail ? (
            <LoadingElement
              tabKey="variability-detail"
              message="Loading detail..."
            />
          ) : varRows.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">
                  {classFilter
                    ? `${classFilter.charAt(0).toUpperCase() + classFilter.slice(1)} variability SKUs`
                    : "Top volatile SKUs"}{" "}
                  <span className="text-muted-foreground">
                    ({varDetail?.total ?? 0} total)
                  </span>
                </p>
                {classFilter && (
                  <button
                    onClick={() => setClassFilter("")}
                    className="text-xs text-primary underline"
                  >
                    Clear filter
                  </button>
                )}
              </div>
              <div className="overflow-x-auto rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Item</TableHead>
                      <TableHead>Loc</TableHead>
                      <TableHead>ABC</TableHead>
                      <TableHead className="text-right">CV</TableHead>
                      <TableHead className="text-right">Std</TableHead>
                      <TableHead className="text-right">Mean</TableHead>
                      <TableHead className="text-right">Intermittency</TableHead>
                      <TableHead>Class</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {varRows.map((r: VariabilityDetailRow) => (
                      <TableRow
                        key={`${r.item_id}-${r.loc}`}
                        className={ROW_COLORS[r.variability_class ?? ""] ?? ""}
                      >
                        <TableCell className="font-mono text-xs">
                          {r.item_id}
                        </TableCell>
                        <TableCell className="text-xs">{r.loc}</TableCell>
                        <TableCell className="text-xs">
                          {r.abc_vol ?? "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.demand_cv != null
                            ? Number(r.demand_cv).toFixed(3)
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.demand_std != null
                            ? formatNumber(r.demand_std)
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.demand_mean != null
                            ? formatNumber(r.demand_mean)
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.intermittency_ratio != null
                            ? `${(Number(r.intermittency_ratio) * 100).toFixed(1)}%`
                            : "—"}
                        </TableCell>
                        <TableCell>
                          <span
                            className={cn(
                              "rounded px-1.5 py-0.5 text-xs font-medium capitalize",
                              BADGE_COLORS[r.variability_class ?? ""] ?? "",
                            )}
                          >
                            {r.variability_class ?? "—"}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              {(varDetail?.total ?? 0) > VAR_PAGE && (
                <div className="flex items-center justify-between text-sm">
                  <button
                    disabled={varOffset === 0}
                    onClick={() =>
                      setVarOffset(Math.max(0, varOffset - VAR_PAGE))
                    }
                    className="flex items-center gap-1 disabled:opacity-40"
                  >
                    <ChevronLeft className="h-4 w-4" /> Prev
                  </button>
                  <span className="text-muted-foreground">
                    {varOffset + 1}–
                    {Math.min(varOffset + VAR_PAGE, varDetail?.total ?? 0)} of{" "}
                    {varDetail?.total}
                  </span>
                  <button
                    disabled={
                      varOffset + VAR_PAGE >= (varDetail?.total ?? 0)
                    }
                    onClick={() => setVarOffset(varOffset + VAR_PAGE)}
                    className="flex items-center gap-1 disabled:opacity-40"
                  >
                    Next <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>
          ) : panelOpen && !loadingVarDetail ? (
            <p className="text-sm text-muted-foreground">
              No variability data available.
            </p>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
}
