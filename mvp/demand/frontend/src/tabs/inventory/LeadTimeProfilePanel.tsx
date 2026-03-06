import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, ChevronLeft, ChevronRight } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchLtSummary,
  fetchLtProfile,
} from "@/api/queries";
import type { LtProfileRow } from "@/api/queries";
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

const LT_PAGE = 20;

export function LeadTimeProfilePanel() {
  const [panelOpen, setPanelOpen] = useState(false);
  const [classFilter, setClassFilter] = useState("");
  const [ltOffset, setLtOffset] = useState(0);

  const { data: ltSummary, isLoading: loadingLtSummary } = useQuery({
    queryKey: queryKeys.ltSummary({ abc_vol: "" }),
    queryFn: () => fetchLtSummary({}),
    staleTime: STALE.FIVE_MIN,
    enabled: panelOpen,
  });

  const ltProfileParams = useMemo(
    () => ({
      lt_variability_class: classFilter || undefined,
      limit: LT_PAGE,
      offset: ltOffset,
      sort_by: "lt_cv",
      sort_dir: "desc",
    }),
    [classFilter, ltOffset],
  );

  const { data: ltProfileData, isLoading: loadingLtProfile } = useQuery({
    queryKey: queryKeys.ltProfile(ltProfileParams),
    queryFn: () => fetchLtProfile(ltProfileParams),
    staleTime: STALE.FIVE_MIN,
    enabled: panelOpen,
  });

  const ltRows: LtProfileRow[] = ltProfileData?.rows ?? [];

  const CLASS_COLORS: Record<string, string> = {
    stable: "border-green-400 bg-green-50 dark:bg-green-950/20",
    moderate: "border-yellow-400 bg-yellow-50 dark:bg-yellow-950/20",
    volatile: "border-red-400 bg-red-50 dark:bg-red-950/20",
  };

  const ROW_COLORS: Record<string, string> = {
    stable: "bg-green-50 dark:bg-green-950/20",
    moderate: "bg-yellow-50 dark:bg-yellow-950/20",
    volatile: "bg-red-50 dark:bg-red-950/20",
  };

  const BADGE_COLORS: Record<string, string> = {
    stable: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    moderate: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    volatile: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  };

  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setPanelOpen((o) => !o)}
      >
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Lead Time Profile</CardTitle>
            <CardDescription>
              LT variability by item-location — stable / moderate / volatile
            </CardDescription>
          </div>
          {panelOpen ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </CardHeader>

      {panelOpen && (
        <CardContent className="space-y-4">
          {loadingLtSummary ? (
            <LoadingElement
              tabKey="lt-summary"
              message="Loading LT summary..."
            />
          ) : ltSummary && ltSummary.total_profiles > 0 ? (
            <div className="space-y-3">
              {/* Class breakdown + avg stats */}
              <div className="flex flex-wrap gap-2">
                {(["stable", "moderate", "volatile"] as const).map((cls) => (
                  <button
                    key={cls}
                    onClick={() => {
                      setClassFilter(classFilter === cls ? "" : cls);
                      setLtOffset(0);
                    }}
                    className={cn(
                      "flex flex-col items-center rounded-lg border px-4 py-2 text-sm transition-colors",
                      CLASS_COLORS[cls],
                      classFilter === cls && "ring-2 ring-primary",
                    )}
                  >
                    <span className="text-xl font-bold tabular-nums">
                      {ltSummary.by_class[cls]}
                    </span>
                    <span className="capitalize text-muted-foreground">
                      {cls}
                    </span>
                  </button>
                ))}
                <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                  <span className="text-xl font-bold tabular-nums">
                    {ltSummary.avg_lt_cv != null
                      ? Number(ltSummary.avg_lt_cv).toFixed(2)
                      : "—"}
                  </span>
                  <span className="text-muted-foreground">Avg LT CV</span>
                </div>
                <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                  <span className="text-xl font-bold tabular-nums">
                    {ltSummary.avg_lt_mean_days != null
                      ? `${Number(ltSummary.avg_lt_mean_days).toFixed(1)}d`
                      : "—"}
                  </span>
                  <span className="text-muted-foreground">Avg LT Mean</span>
                </div>
              </div>

              {/* LT CV percentile bar */}
              <div className="text-xs text-muted-foreground">
                LT CV percentiles — p50:{" "}
                {ltSummary.lt_cv_p50 != null
                  ? Number(ltSummary.lt_cv_p50).toFixed(2)
                  : "—"}{" "}
                · p95:{" "}
                {ltSummary.lt_cv_p95 != null
                  ? Number(ltSummary.lt_cv_p95).toFixed(2)
                  : "—"}
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No lead time profile data. Run{" "}
              <code>make lt-profile-compute</code> first.
            </p>
          )}

          {/* Detail table */}
          {loadingLtProfile ? (
            <LoadingElement
              tabKey="lt-profile"
              message="Loading profiles..."
            />
          ) : ltRows.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">
                  {classFilter
                    ? `${classFilter.charAt(0).toUpperCase() + classFilter.slice(1)} LT DFUs`
                    : "Most volatile item-locations"}{" "}
                  <span className="text-muted-foreground">
                    ({ltProfileData?.total ?? 0} total)
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
                      <TableHead className="text-right">LT Mean</TableHead>
                      <TableHead className="text-right">LT Std</TableHead>
                      <TableHead className="text-right">LT CV</TableHead>
                      <TableHead className="text-right">LT p95</TableHead>
                      <TableHead className="text-right">Obs</TableHead>
                      <TableHead>Class</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ltRows.map((r: LtProfileRow) => (
                      <TableRow
                        key={`${r.item_no}-${r.loc}`}
                        className={ROW_COLORS[r.lt_variability_class ?? ""] ?? ""}
                      >
                        <TableCell className="font-mono text-xs">
                          {r.item_no}
                        </TableCell>
                        <TableCell className="text-xs">{r.loc}</TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.lt_mean_days != null
                            ? `${Number(r.lt_mean_days).toFixed(1)}d`
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.lt_std_days != null
                            ? `${Number(r.lt_std_days).toFixed(1)}d`
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.lt_cv != null ? Number(r.lt_cv).toFixed(3) : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.lt_p95_days != null
                            ? `${Number(r.lt_p95_days).toFixed(1)}d`
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-xs">
                          {r.observation_count ?? "—"}
                        </TableCell>
                        <TableCell>
                          <span
                            className={cn(
                              "rounded px-1.5 py-0.5 text-xs font-medium capitalize",
                              BADGE_COLORS[r.lt_variability_class ?? ""] ?? "",
                            )}
                          >
                            {r.lt_variability_class ?? "—"}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              {(ltProfileData?.total ?? 0) > LT_PAGE && (
                <div className="flex items-center justify-between text-sm">
                  <button
                    disabled={ltOffset === 0}
                    onClick={() =>
                      setLtOffset(Math.max(0, ltOffset - LT_PAGE))
                    }
                    className="flex items-center gap-1 disabled:opacity-40"
                  >
                    <ChevronLeft className="h-4 w-4" /> Prev
                  </button>
                  <span className="text-muted-foreground">
                    {ltOffset + 1}–
                    {Math.min(ltOffset + LT_PAGE, ltProfileData?.total ?? 0)}{" "}
                    of {ltProfileData?.total}
                  </span>
                  <button
                    disabled={
                      ltOffset + LT_PAGE >= (ltProfileData?.total ?? 0)
                    }
                    onClick={() => setLtOffset(ltOffset + LT_PAGE)}
                    className="flex items-center gap-1 disabled:opacity-40"
                  >
                    Next <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>
          ) : panelOpen && !loadingLtProfile ? (
            <p className="text-sm text-muted-foreground">
              No lead time profiles available.
            </p>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
}
